from pathlib import Path
from CompetitionEvaluation import structure_data, calculate_metrics
from utilities import list_submissions, get_target_data, TargetType, reformat_output
import os
import xarray
import numpy as np
import numpy.typing as npt
from scipy.signal import resample
import argparse
import pandas as pd
import pyarrow
import logging
import time

logger = logging.getLogger('evaluate_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('evaluate_submission.log', encoding="utf-8")
formatter = logging.Formatter('%(asctime)s %(pathname)s [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def evaluate_forecast(
    forecast: pd.DataFrame,
    actuals: pd.DataFrame,
    target: TargetType,
    expected_samples: int,
    draw_column: str,
    data_column: str,
    bins: list[float], 
    reformat: bool,
    save_path: str | os.PathLike,
) -> None:
    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError(f'Target {target} must be either "pgm" or "cm".')

    # Cast to xarray
    observed, predictions = structure_data(
        actuals, forecast, draw_column_name=draw_column, data_column_name=data_column
    )

    if bool(predictions['outcome'].isnull().any()):
        logger.warning(
            "Found NaN values. These are replaced with 0 before further calculations."
        )
        predictions["outcome"] = xarray.where(
            predictions["outcome"].isnull(), 0, predictions["outcome"]
        )

    if bool((predictions["outcome"] > 10e9).any()):
        logger.warning(
            f"Found predictions larger than earth population. These are censored at 10 billion."
        )
        predictions["outcome"] = xarray.where(
            predictions["outcome"] > 10e9, 10e9, predictions["outcome"]
        )

    if predictions.dims["member"] != expected_samples:
        logger.warning(
            f'Number of samples ({predictions.dims["member"]}) is not 1000. Using scipy.signal.resample to get {expected_samples} samples when calculating Ignorance Score.'
        )
        np.random.seed(284975)
        arr = resample(predictions.to_array(), expected_samples, axis=3)
        arr = np.where(
            arr < 0, 0, arr
        )  # For the time when resampling happens to go below zero.

        new_container = predictions.sel(member=1)
        new_container = (
            new_container.expand_dims({"member": range(0, expected_samples)})
            .to_array()
            .transpose("variable", "month_id", unit, "member")
        )
        predictions: xarray.Dataset = xarray.DataArray(
            data=arr, coords=new_container.coords
        ).to_dataset(dim="variable")

    if bool((predictions["outcome"] < 0).any()):
        logger.warning(
            f"Found negative predictions. These are censored at 0 before calculating Ignorance Score."
        )
        predictions["outcome"] = xarray.where(
            predictions["outcome"] < 0, 0, predictions["outcome"]
        )

    crps = calculate_metrics(
        observed, predictions, metric="crps", aggregate_over="nothing"
    )
    mis = calculate_metrics(
        observed,
        predictions,
        metric="mis",
        prediction_interval_level=0.9,
        aggregate_over="nothing",
    )

    ign = calculate_metrics(
        observed, predictions, metric="ign", bins=bins, aggregate_over="nothing"
    )

    # Save data in .parquet long-format (month_id, unit_id, metric, value)
    dfs = {"crps": crps, "ign": ign, "mis": mis}

    if not reformat:
        for metric in ["crps", "ign", "mis"]:
            dfs[metric].rename(columns={metric: "value"}, inplace=True)
            metric_dir = save_path / f"metric={metric}"
            metric_dir.mkdir(exist_ok=True, parents=True)
            dfs[metric].to_parquet(metric_dir / f"{metric}.parquet")

    return dfs
        

def match_forecast_with_actuals(
    submission, actuals_folder, target: TargetType, window: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filter = pyarrow.compute.field("window") == window
    actuals = get_target_data(actuals_folder, target=target, filters=filter)
    predictions = get_target_data(submission, target=target, filters=filter)

    predictions.drop(columns=["window"], inplace=True)
    actuals.drop(columns=["window"], inplace=True)

    return actuals, predictions


def evaluate_submission(
    submission: str | os.PathLike,
    acutals: str | os.PathLike,
    targets: list[TargetType],
    windows: list[str],
    expected: int,
    bins: list[float],
    draw_column: str,
    data_column: str,
    reformat: bool,
    save_to: str | os.PathLike,
    file_format: str
) -> None:
    """Loops over all targets and windows in a submission folder, match them with the correct test dataset, and estimates evaluation metrics.
    Stores evaluation data as .parquet files in {submission}/eval/{target}/window={window}/.

    Parameters
    ----------
    submission : str | os.PathLike
        Path to a folder structured like a submission_template
    acutals : str | os.PathLike
        Path to actuals folder structured like {actuals}/{target}/window={window}/data.parquet
    targets : list[TargetType]
        A list of strings, either ["pgm"] for PRIO-GRID-months, or ["cm"] for country-months, or both.
    windows : list[str]
        A list of strings indicating the window of the test dataset. The string should match windows in data in the actuals folder.
    expected : int
        The expected numbers of samples. Due to how Ignorance Score is defined, all IGN metric comparisons must be across models with equal number of samples.
    bins : list[float]
        The binning scheme used in the Ignorance Score.
    draw_column : str
        The name of the sample column. We assume samples are drawn independently from the model. Default = "draw"
    data_column : str
        The name of the data column. Default = "outcome"
    reformat: bool
        If True, the output is reformatted based on the format discussed. Default = False
    save_to: str | os.PathLike
        Path to save reformatted data. Only used if reformat=True. Default = None
    file_format: str
        File format for the output. Default is json.
    """

    submission = Path(submission)
    logger.info(f"Evaluating {submission.name}")
    for target in targets:
        all_window = []
        for window in windows:
            if any(
                (submission / target).glob("**/*.parquet")
            ):  # test if there are prediction files in the target
                logger.info(f"Target: {target}, Window: {window}")
                observed_df, pred_df = match_forecast_with_actuals(
                    submission, acutals, target, window
                )
                
                save_path = submission / "eval" / f"{target}" / f"window={window}"
                df = evaluate_forecast(
                    forecast=pred_df,
                    actuals=observed_df,
                    target=target,
                    expected_samples=expected,   
                    draw_column=draw_column,
                    data_column=data_column,
                    bins=bins,
                    reformat=reformat,
                    save_path=save_path
                )
                all_window.append(df)

        if reformat and len(all_window) > 0:
            logger.info(f"Reformatting {submission.name}")
            save_path = Path(save_to) / target
            Path(save_path).mkdir(parents=True, exist_ok=True)
            dfs = {key: pd.concat([df[key] for df in all_window]) for key in all_window[0].keys()}
            reformat_output(dfs["crps"], dfs["ign"], dfs["mis"], target, save_path, submission.name, file_format)


def evaluate_all_submissions(
    submissions: str | os.PathLike,
    acutals: str | os.PathLike,
    targets: list[TargetType],
    windows: list[str],
    expected: int,
    bins: list[float],
    draw_column: str,
    data_column: str,
    reformat: bool,
    save_to: str | os.PathLike,
    file_format: str
) -> None:
    """Loops over all submissions in the submissions folder, match them with the correct test dataset, and estimates evaluation metrics.
    Stores evaluation data as .parquet files in {submissions}/{submission_name}/eval/{target}/window={window}/.

    Parameters
    ----------
    submissions : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template
    acutals : str | os.PathLike
        Path to actuals folder structured like {actuals}/{target}/window={window}/data.parquet
    targets : list[TargetType]
        A list of strings, either ["pgm"] for PRIO-GRID-months, or ["cm"] for country-months, or both.
    windows : list[str]
        A list of strings indicating the window of the test dataset. The string should match windows in data in the actuals folder.
    expected : int
        The expected numbers of samples. Due to how Ignorance Score is defined, all IGN metric comparisons must be across models with equal number of samples.
    bins : list[float]
        The binning scheme used in the Ignorance Score.
    draw_column : str
        The name of the sample column. We assume samples are drawn independently from the model. Default = "draw"
    data_column : str
        The name of the data column. Default = "outcome"
    reformat: bool
        If True, the output is reformatted based on the format discussed. Default = False
    save_to: str | os.PathLike
        Path to save reformatted data. Only used if reformat=True. Default = None
    file_format: str
        File format for the output. Default is json.
    """
    submissions = Path(submissions)
    submissions = list_submissions(submissions)
    acutals = Path(acutals)

    for submission in submissions:
        try:
            evaluate_submission(
                submission,
                acutals,
                targets,
                windows,
                expected,
                bins,
                draw_column,
                data_column,
                reformat,
                save_to,
                file_format
            )
        except Exception as e:
            logger.error(f"{str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Method for evaluation of submissions to the ViEWS Prediction Challenge",
        epilog="Example usage: python evaluate_submissions.py -s ./submissions -a ./actuals -e 100",
    )
    parser.add_argument(
        "-s",
        "--submissions",
        type=str,
        help="path to folder with submissions complying with submission_template",
    )
    parser.add_argument(
        "-a", 
        "--actuals", 
        type=str, 
        help="path to folder with actuals"
    )
    parser.add_argument(
        "-t",
        "--targets",
        nargs="+",
        type=str,
        default=["pgm", "cm"],
        help="pgm or cm or both"
    )
    parser.add_argument(
        "-w",
        "--windows",
        nargs="+",
        type=str,
        default=["Y2018", "Y2019", "Y2020", "Y2021", "Y2022", "Y2023", "Y2024", "Y2025"],
        help="windows to evaluate"
    )
    parser.add_argument(
        "-e", 
        "--expected", 
        type=int, 
        default=1000,
        help="expected samples"
    )
    parser.add_argument(
        "-sc",
        "--draw_column",
        type=str,
        default="draw",
        help="(Optional) name of column for the unique samples"
    )
    parser.add_argument(
        "-dc",
        "--data_column",
        type=str,
        default="outcome",
        help="(Optional) name of column with data, must be same in both observed and predictions data"
    )
    parser.add_argument(
        "-ib",
        "--bins",
        nargs="+",
        type=float,
        default=[0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5],
        help='Set a binning scheme for the ignorance score. List or integer (nbins). E.g., "--ib 0 0.5 1 5 10 100 1000". None also allowed.'
    )
    parser.add_argument(
        "-r", 
        "--reformat", 
        action="store_true",
        help="Flag to reformat data to match the requirement of dashboard"
    )
    parser.add_argument(
        "-st", 
        "--save_to", 
        type=str, 
        default=None,
        help="Path to save reformatted data. Only used if reformat=True."
    )
    parser.add_argument(
        "-ff",
        "--file_format",
        type="str",
        default="json",
        choices=["json", "parquet", "all"],
        help="File format for the output. Default is json."
    )

    args = parser.parse_args()

    submissions = Path(args.submissions)
    acutals = Path(args.actuals)
    expected = args.expected
    targets = args.targets
    windows = args.windows
    draw_column = args.draw_column
    data_column = args.data_column
    bins = args.bins
    reformat = args.reformat
    save_to = Path(args.save_to) if args.save_to else None
    file_format = args.file_format

    start_time = time.time()

    evaluate_all_submissions(
        submissions, acutals, targets, windows, expected, bins, draw_column, data_column, reformat, save_to, file_format)
    
    end_time = time.time()
    minutes = (end_time - start_time) / 60
    logger.info(f'Done. Runtime: {minutes:.3f} minutes.\n')


if __name__ == "__main__":
    main()
    # submission = './final_submissions_cleaned'
    # actuals = './actuals'
    # targets = ['pgm']
    # windows = ['Y2018', 'Y2019', 'Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024', 'Y2025']
    # expected = 1000
    # bins = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5]
    # draw_column = 'draw'
    # data_column = 'outcome'
    # reformat = True
    # save_to = './eval_04Dec'
    # file_format = "json"
    # start_time = time.time()
    # # evaluate_submission(submission, actuals, targets, windows, expected, bins, draw_column, data_column, reformat, save_to, file_format)
    # evaluate_all_submissions(submission, actuals, targets, windows, expected, bins, draw_column, data_column, reformat, save_to, file_format)
    # end_time = time.time()
    # minutes = (end_time - start_time) / 60
    # logger.info(f'Done. Runtime: {minutes:.3f} minutes.\n')
