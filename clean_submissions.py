from utilities import list_submissions, get_target_data, remove_duplicated_indexes, TargetType
import os
from pathlib import Path
import pandas as pd
import argparse
import xarray as xr
import pyarrow
import numpy as np
import numpy.typing as npt
from scipy.signal import resample
import logging

logger = logging.getLogger('clean_logger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('clean_submissions.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def check_values(predictions: xr.Dataset,
                 target: str,
                 expected_samples: int,) -> pd.DataFrame:
    # Check if the predictions have the correct values.
    # This function is adapted from evaluate_forecast in evaluate_submissions.py

    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError(f'Target {target} must be either "pgm" or "cm".')

    if bool(predictions['outcome'].isnull().any()):
        logger.warning(
            "Found NaN values. These are replaced with 0 before further calculations."
        )
        predictions["outcome"] = xr.where(
            predictions["outcome"].isnull(), 0, predictions["outcome"]
        )

    if bool((predictions["outcome"] > 10e9).any()):
        logger.warning(
            f"Found predictions larger than earth population. These are censored at 10 billion."
        )
        predictions["outcome"] = xr.where(
            predictions["outcome"] > 10e9, 10e9, predictions["outcome"]
        )

    if predictions.dims["member"] != expected_samples:
        logger.warning(
            f'Number of samples ({predictions.dims["member"]}) is not 1000. Using scipy.signal.resample to get {expected_samples} samples when calculating Ignorance Score.'
        )
        np.random.seed(284975)
        arr: npt.ArrayLike = resample(predictions.to_array(), expected_samples, axis=3)
        arr = np.where(
            arr < 0, 0, arr
        )  # For the time when resampling happens to go below zero.

        new_container = predictions.sel(member=1)
        new_container = (
            new_container.expand_dims({"member": range(0, expected_samples)})
            .to_array()
            .transpose("variable", "month_id", unit, "member")
        )
        predictions: xr.Dataset = xr.DataArray(
            data=arr, coords=new_container.coords
        ).to_dataset(dim="variable")

    if bool((predictions["outcome"] < 0).any()):
        logger.warning(
            f"Found negative predictions. These are censored at 0 before calculating Ignorance Score."
        )
        predictions["outcome"] = xr.where(
            predictions["outcome"] < 0, 0, predictions["outcome"]
        )

    return predictions.to_dataframe()


def check_structure(predictions: pd.DataFrame,
                draw_column_name: str = "draw",
                data_column_name: str = "outcome",) -> xr.DataArray:
    # Check if the predictions have the correct structure.
    # This function is adapted from structure_data in CompetitionEvaluation.py

    if predictions.index.names != [None]:
        predictions = predictions.reset_index()

    # To simplify internal affairs:
    predictions = predictions.rename(
        columns={draw_column_name: "member", data_column_name: "outcome"}
    )
    predictions["month_id"] = predictions["month_id"].astype(int)

    # Set up multi-index to easily convert to xarray
    if "priogrid_gid" in predictions.columns:
        predictions["priogrid_gid"] = predictions["priogrid_gid"].astype(int)
        predictions = predictions.set_index(["month_id", "priogrid_gid", "member"])
    elif "country_id" in predictions.columns:
        predictions["country_id"] = predictions["country_id"].astype(int)
        predictions = predictions.set_index(["month_id", "country_id", "member"])
    else:
        TypeError("priogrid_gid or country_id must be an identifier")

    # Some groups have multiple values for the same index, this function removes duplicates
    predictions = remove_duplicated_indexes(predictions)
    xpred = predictions.to_xarray()

    return xpred


def clean_submission(
    submission: str | os.PathLike,
    save_to: str | os.PathLike,
    targets: list[TargetType],
    windows: list[str],
    expected_samples: int = 1000,
    draw_column: str = "draw",
    data_column: str = "outcome",
) -> None:
    """Loops over all targets and windows in a submission folder and clean them
    Stores cleaned data to save_to with the same structure as the submission folder

    Parameters
    ----------
    submission : str | os.PathLike
        Path to a folder structured like a submission_template
    save_to: str | os.PathLike
        Path to save reformatted data. Only used if reformat=True.
    targets : list[TargetType]
        A list of strings, either ["pgm"] for PRIO-GRID-months, or ["cm"] for country-months, or both.
    windows : list[str]
        A list of strings indicating the window of the test dataset. The string should match windows in data in the actuals folder.
    expected_samples : int
        The expected numbers of samples.
    draw_column : str
        The name of the sample column. We assume samples are drawn independently from the model. Default = "draw"
    data_column : str
        The name of the data column. Default = "outcome"
    """

    submission = Path(submission)
    logger.info(f"Checking {submission.name}")
    for target in targets:
        for window in windows:
            if any(
                (submission / target).glob("**/*.parquet")
            ):  # test if there are prediction files in the target
                filter = pyarrow.compute.field("window") == window
                predictions = get_target_data(submission, target=target, filters=filter)
                predictions.drop(columns=["window"], inplace=True)

                predictions = check_structure(predictions, draw_column, data_column)
                predictions = check_values(predictions, target, expected_samples)

                save_path = Path(save_to) / submission.name / target / f"window={window}"
                save_path.mkdir(exist_ok=True, parents=True)
                predictions.to_parquet(save_path / f"{submission.name}_{target}_{window}.parquet")


def clean_all_submissions(
    submissions: str | os.PathLike,
    save_to: str | os.PathLike,
    targets: list[TargetType],
    windows: list[str],
    expected_samples: int = 1000,
    draw_column: str = "draw",
    data_column: str = "outcome",
) -> None:
    """Loops over all submissions in the submissions folder and clean them
    Stores cleaned data to save_to with the same structure as the submission folder

    Parameters
    ----------
    submissions : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template
    save_to: str | os.PathLike
        Path to save reformatted data. Only used if reformat=True.
    targets : list[TargetType]
        A list of strings, either ["pgm"] for PRIO-GRID-months, or ["cm"] for country-months, or both.
    windows : list[str]
        A list of strings indicating the window of the test dataset.
    expected : int
        The expected numbers of samples.
    draw_column : str
        The name of the sample column. We assume samples are drawn independently from the model. Default = "draw"
    data_column : str
        The name of the data column. Default = "outcome"
    """
    submissions = Path(submissions)
    submissions = list_submissions(submissions)

    for submission in submissions:
        try:
            clean_submission(submission,
                             save_to,
                             targets,
                             windows,
                             expected_samples,
                             draw_column,
                             data_column)
        except Exception as e:
            logger.error(f"{str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Method for evaluation of submissions to the ViEWS Prediction Challenge",
        epilog="Example usage: python evaluate_submissions.py -s ./submissions -st ./submissions_cleaned -e 100",
    )
    parser.add_argument(
        "-s",
        metavar="submissions",
        type=str,
        help="path to folder with submissions complying with submission_template",
    )
    parser.add_argument(
        "-st",
        metavar="save_to",
        type=str,
        help="Path to save cleaned data."
    )
    parser.add_argument(
        "-t",
        metavar="targets",
        nargs="+",
        type=str,
        help="pgm or cm or both",
        default=["pgm", "cm"],
    )
    parser.add_argument(
        "-w",
        metavar="windows",
        nargs="+",
        type=str,
        help="windows to evaluate",
        default=["Y2018", "Y2019", "Y2020", "Y2021", "Y2022", "Y2023", "Y2024"],
    )
    parser.add_argument(
        "-e",
        metavar="expected_samples",
        type=int,
        help="expected samples",
        default=1000
    )
    parser.add_argument(
        "-sc",
        metavar="draw_column",
        type=str,
        help="(Optional) name of column for the unique samples",
        default="draw",
    )
    parser.add_argument(
        "-dc",
        metavar="data_column",
        type=str,
        help="(Optional) name of column with data, must be same in both observed and predictions data",
        default="outcome",
    )

    args = parser.parse_args()
    submissions = Path(args.s)
    save_to = Path(args.st)
    expected_samples = args.e
    targets = args.t
    windows = args.w
    draw_column = args.sc
    data_column = args.dc

    clean_all_submissions(
        submissions, save_to, targets, windows, expected_samples, draw_column, data_column)

if __name__ == "__main__":
    main()
    # submission = './final_submissions/conflictforecast_v2'
    # save_to = './final_submissions_cleaned/'
    # targets = ['pgm', 'cm']
    # windows = ['Y2018', 'Y2019', 'Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024']
    # expected = 1000
    # clean_submission(submission, save_to, targets, windows, expected)
    # clean_all_submissions(submission, save_to, targets, windows, expected)

