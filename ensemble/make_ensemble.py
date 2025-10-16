import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from utils.utilities import TargetType, list_submissions
from utils.set_logger import set_logger
from ensemble.sample import sample_unweighted, sample_weighted, sample_selected

logger = set_logger("ensemble_logger", "logs/ensemble.log")


def process_month_id(month, id, samples_dict, loa, expected_samples, weights, metrics_dict):
    """
    Process a month and id to make the ensemble.

    Args:
        month: The month to make an ensemble for.
        id: The id to make an ensemble for.
        samples_dict: A dictionary of samples for each model. The keys are the model names and the values are dictionaries of samples for each month and id.
        loa: The level of analysis.
        expected_samples: The number of samples to draw from the ensemble.
        weights: The weights to use for the ensemble. Can be "unweighted", "weighted", or "selected".
        metrics_dict: Dict of metric -> {model_name: value} 
    """
    try:
        if weights == "unweighted":
            all_samples = sample_unweighted(samples_dict, expected_samples, month, id)
        elif weights == "weighted":
            if metrics_dict is None:
                raise ValueError("metrics_dict must be provided for weighted ensemble")
            all_samples = sample_weighted(samples_dict, expected_samples, month, id, metrics_dict["crps"])
        elif weights == "selected":
            if metrics_dict is None:
                raise ValueError("metrics_dict must be provided for selected ensemble")
            all_samples = sample_selected(samples_dict, expected_samples, month, id, metrics_dict)
        else:
            raise ValueError(f"Invalid weight type: {weights}")

        member_indices = pd.MultiIndex.from_product(
            [[month], [id], range(expected_samples)], names=["month_id", loa, "member"]
        )
        month_df = pd.DataFrame({"outcome": all_samples}, index=member_indices)

        return month_df
    except Exception as e:
        logger.error(f"Error processing month={month}, id={id}, loa={loa}, weights={weights}: {str(e)}")
        raise


def process_single_file(file_path, level, submission):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    model_samples = {}
    for (month, id), group in df.groupby(level=["month_id", level]):
        model_samples[(month, id)] = group["outcome"].values
    return submission.name, model_samples


def read_data(target, window, submissions):
    """
    Read and parse all model files for a given (target, window) ONCE.
    Returns samples_dict and month_level_pairs.
    """
    if target == "cm":
        level = "country_id"
    elif target == "pgm":
        level = "priogrid_gid"
    else:
        raise ValueError(f"Invalid level of analysis: {target}")

    file_paths = [
        (
            submission
            / target
            / f"window={window}"
            / f"{submission.name}_{target}_{window}.parquet",
            level,
            submission,
        )
        for submission in submissions
        if any((submission / target).glob("**/*.parquet"))
    ]

    logger.info(f"Reading {target} {window} with {len(file_paths)} models")
    samples_dict = {}
    month_level_pairs = None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_file, file_path, level, submission)
                   for file_path, level, submission in file_paths]
        for future in futures:
            model_name, model_samples = future.result()
            samples_dict[model_name] = model_samples
            if month_level_pairs is None and model_samples:
                month_level_pairs = list(model_samples.keys())

    return samples_dict, month_level_pairs


def process_data(
    target: TargetType,
    window: str,
    samples_dict: dict,
    month_level_pairs: list,
    save_to: Path,
    expected_samples: int,
    weight: str,
    metrics_dict: dict,
    ):
    """
    Process a window of to make the ensemble for a single weight type, using pre-built samples_dict and month_level_pairs.
    """
    try:
        if target == "cm":
            level = "country_id"
        elif target == "pgm":
            level = "priogrid_gid"
        else:
            raise ValueError(f"Invalid level of analysis: {target}")

        logger.info(f"Processing {target} {window} with {len(samples_dict)} models for weight {weight}")

        save_path = (
            save_to / f"{weight}" / target / f"window={window}" / f"ensemble_{weight}.parquet"
        )

        all_dfs = []
        for month, id in month_level_pairs:
            month_df = process_month_id(
                month, id, samples_dict, level, expected_samples, weight, metrics_dict
            )
            all_dfs.append(month_df)

        if all_dfs:
            combined_df = pd.concat(all_dfs)
            table = pa.Table.from_pandas(combined_df)
            pq.write_table(table, save_path)

    except Exception as e:
        logger.error(f"Error processing {target} {window}: {str(e)}")
        import traceback
        traceback.print_exc()


def create_metrics_dict(submissions: list[Path], target: TargetType):
    """
    Create a dictionary of metrics for each model.
    """
    metrics_dict = {}
    for metric in ["crps", "mis", "ign"]:
        metrics_dict[metric] = {}

        for submission in submissions:
            base = submission / "eval" / f"{target}"
            pattern = f"*/metric={metric}/{metric}.parquet"
            file_paths = list(base.glob(pattern))

            if len(file_paths) != 0:
                values = []
                for file_path in file_paths:
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    values.append(df["value"].mean())
                metrics_dict[metric][submission.name] = np.mean(values)
    return metrics_dict
   

def load_metrics_dict(submissions: list[Path], target: TargetType, weight: str, save_to: Path):
    metrics_dict = None
    if weight == "weighted" or weight == "selected":
        try:
            with open(save_to / f"metrics_dict_{target}.json", "r") as f:
                metrics_dict = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Metrics file not found for {weight} {target} ensemble. Creating it now.")
            metrics_dict = create_metrics_dict(submissions, target)
            with open(save_to / f"metrics_dict_{target}.json", "w") as f:
                json.dump(metrics_dict, f)
    return metrics_dict


def make_ensemble(
    submissions: str | os.PathLike,
    save_to: str | os.PathLike,
    targets: list[TargetType] = ["cm", "pgm"],
    windows: list[str] = [
        "Y2018",
        "Y2019",
        "Y2020",
        "Y2021",
        "Y2022",
        "Y2023",
        "Y2024",
        "Y2025",
    ],
    expected_samples: int = 1000,
    weights: list[str] = ["unweighted", "weighted", "selected"],
    ):
    """
    Make an ensemble of the submissions.

    Args:
        submissions: The path to the submissions folder.
        save_to: The path to save the ensemble.
        targets: The level of analysis.
        windows: The windows to make an ensemble for.
        expected_samples: The number of samples to draw from the ensemble.
        weights: The weights to use for the ensemble. Can be "unweighted", "weighted", or "selected".
    """
    submissions = Path(submissions)
    save_to = Path(save_to)
    submissions = list_submissions(submissions)

    start_time = time.time()

    for target in targets:
        with tqdm(total=len(windows)*len(weights), desc=f"Processing {target}") as pbar:
            for window in windows:
                samples_dict, month_level_pairs = read_data(target, window, submissions)
                for weight in weights:
                    # Load metrics dict if it exists otherwise create it
                    metrics_dict = load_metrics_dict(submissions, target, weight, save_to)
                    
                    window_start = time.time()
                    (save_to / f"{weight}" / target / f"window={window}").mkdir(
                        parents=True, exist_ok=True
                    )
                    process_data(
                        target, window, samples_dict, month_level_pairs, save_to, expected_samples, weight, metrics_dict
                    )
                    window_time = time.time() - window_start
                    logger.info(f"Completed {target} {window} ({weight}) in {window_time:.2f} seconds")
                    pbar.update(1)
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Make an ensemble of the submissions.")
    parser.add_argument(
        "-s",
        metavar="submissions",
        type=str,
        required=True,
        help="Path to the submissions folder",
    )
    parser.add_argument(
        "-st",
        metavar="save_to",
        type=str,
        required=True,
        help="Path to save the ensemble",
    )
    parser.add_argument(
        "-t",
        metavar="targets",
        type=str,
        nargs="+",
        default=["cm", "pgm"],
        help="Targets to make an ensemble for",
    )
    parser.add_argument(
        "-w",
        metavar="windows",
        type=str,
        nargs="+",
        default=[
            "Y2018",
            "Y2019",
            "Y2020",
            "Y2021",
            "Y2022",
            "Y2023",
            "Y2024",
            "Y2025",
        ],
        help="Windows to make an ensemble for",
    )
    parser.add_argument(
        "-es",
        metavar="expected_samples",
        type=int,
        default=1000,
        help="Number of samples to draw from the ensemble",
    )
    parser.add_argument(
        "-we",
        metavar="weights",
        type=str,
        nargs="+",
        default=["unweighted", "weighted", "selected"],
        help="Weights to use for the ensemble",
    )
    args = parser.parse_args()

    make_ensemble(args.s, args.st, args.t, args.w, args.es, args.we)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()