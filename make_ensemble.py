import os
import numpy as np
import pandas as pd
from pathlib import Path
from utilities import TargetType, list_submissions
import time
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def process_month_id(month, id, samples_dict, loa, expected_samples, weights):
    try:
        all_samples = np.zeros(expected_samples, dtype=np.float32)
            
        if weights is None:
            samples = np.concatenate([model_samples[(month, id)] 
                                    for model_samples in samples_dict.values() 
                                    if (month, id) in model_samples])
            all_samples[:] = np.random.choice(samples, size=expected_samples, replace=True)
        else:
            model_weights = weights.loc[(month, id)].values
            samples = []
            sample_counts = []
    
            for model_samples in samples_dict.values():
                if (month, id) in model_samples:
                    samples.append(model_samples[(month, id)])
                    sample_counts.append(len(model_samples[(month, id)]))
            
            if samples:
                sample_weights = np.repeat(model_weights, sample_counts)
                sample_weights = sample_weights / sample_weights.sum()
                
                samples = np.concatenate(samples)
                
                all_samples[:] = np.random.choice(samples, size=expected_samples, replace=True, p=sample_weights)

        member_indices = pd.MultiIndex.from_product(
            [[month], [id], range(expected_samples)], names=["month_id", loa, "member"]
        )
        month_df = pd.DataFrame({"outcome": all_samples}, index=member_indices)

        return month_df
    except Exception as e:
        print(f"Error processing month={month}, id={id}: {str(e)}")
        raise


def process_window(target, window, submissions, save_to, expected_samples, weights):
    try:
        if target == "cm":
            level = "country_id"
        elif target == "pgm":
            level = "priogrid_gid"
        else:
            raise ValueError(f"Invalid level of analysis: {target}")

        file_paths = [
            (submission / target / f"window={window}" / f"{submission.name}_{target}_{window}.parquet", level, submission)
            for submission in submissions
            if any((submission / target).glob("**/*.parquet"))
        ]

        print(f"Reading {target} {window} with {len(file_paths)} models")
        samples_dict = {}
        month_level_pairs = None
        
        for file_path, level, submission in tqdm(file_paths, desc="Reading files"):
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            model_samples = {}
            for (month, id), group in df.groupby(level=["month_id", level]):
                model_samples[(month, id)] = group["outcome"].values
                
            samples_dict[submission.name] = model_samples
            if month_level_pairs is None:
                month_level_pairs = df.index.droplevel("member").unique()

        print(f"Processing {target} {window} with {len(samples_dict)} models")
        save_path = save_to / target / f"window={window}" / "ensemble.parquet"

        all_dfs = []
        for month, id in tqdm(month_level_pairs, desc="Processing pairs"):
            month_df = process_month_id(month, id, samples_dict, level, expected_samples, weights)
            all_dfs.append(month_df)

        if all_dfs:
            combined_df = pd.concat(all_dfs)
            table = pa.Table.from_pandas(combined_df)
            pq.write_table(table, save_path)
            
    except Exception as e:
        print(f"Error processing {target} {window}: {str(e)}")
        import traceback
        traceback.print_exc()


def resampling_ensemble(
    submissions: str | os.PathLike,
    save_to: str | os.PathLike,
    targets: list[TargetType] = ["cm", "pgm"],
    windows: list[str] = [
        "Y2018", "Y2019", "Y2020", "Y2021", 
        "Y2022", "Y2023", "Y2024", "Y2025"
    ],
    expected_samples: int = 1000,
    weights: pd.DataFrame = None,
):
    submissions = Path(submissions)
    save_to = Path(save_to)
    submissions = list_submissions(submissions)

    start_time = time.time()

    for target in targets:
        for window in windows:
            window_start = time.time()
            (save_to / target / f"window={window}").mkdir(parents=True, exist_ok=True)
            process_window(
                target, window, submissions, save_to, expected_samples, weights
            )
            window_time = time.time() - window_start
            print(f"Completed {target} {window} in {window_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")


if __name__ == "__main__":
    resampling_ensemble(
        submissions="./final_submissions_cleaned",
        save_to="./ensembles",
        targets=["pgm"],
    ) 