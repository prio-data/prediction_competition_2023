from pathlib import Path
import pyarrow.dataset as pad
import pyarrow.parquet as pq
import pandas as pd
import os
import itertools
from typing import Literal
import datetime
import yaml
import json

from pg2nga import pgIds

TargetType = Literal["cm", "pgm"]


def get_submission_details(submission: str | os.PathLike) -> dict:
    """Reads the submission_details.yml file in a submission and returns a dictionary with its items.

    Parameters
    ----------
    submission : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template

    Returns
    -------
    dict
        A dictionary with the elements in the YAML file.
    """
    with open(submission / "submission_details.yml") as f:
        submission_details = yaml.safe_load(f)
    return submission_details


def date_to_views_month_id(date: datetime.date) -> int:
    """Takes a date and converts it to a "month_id", which is a count of months starting at 1 on January 1980. Works on vectors/columns of dates.

    Parameters
    ----------
    date : datetime.date

    Returns
    -------
    int
        The month_id of the given date.
    """
    views_start_date = datetime.date(1980, 1, 1)
    r = relativedelta(date, views_start_date)
    return (r.years * 12) + r.months + 1


def views_month_id_to_year(month_id: int) -> int:
    """Converts a month_id to the calendar year of the month_id. Works on vectors/columns of month_ids.

    Parameters
    ----------
    month_id : int
        A count of months starting (from 1) on January 1980.

    Returns
    -------
    int
        The calendar year of the month_id.
    """
    return 1980 + (month_id - 1) // 12


def views_month_id_to_month(month_id: int) -> int:
    """Converts a month_id to the calendar month of the month_id

    Parameters
    ----------
    month_id : int
        A count of months starting (from 1) on January 1980.

    Returns
    -------
    int
        The calendar month of the month_id.
    """
    return ((month_id - 1) % 12) + 1


def views_month_id_to_date(month_id: int) -> datetime.date:
    """Converts a month_id to a datetime.date format.

    Parameters
    ----------
    month_id : int
        A count of months starting (from 1) on January 1980.

    Returns
    -------
    datetime.date
        The date of the month_id, using first day of the month format.
    """
    df = pd.DataFrame()
    df["year"] = views_month_id_to_year(month_id)
    df["month"] = views_month_id_to_month(month_id)
    df["day"] = 1
    return pd.to_datetime(df, format="%Y%M")


def migrate_submission_from_old(submission: str | os.PathLike) -> None:
    """Helper function to migrate old submission folder structure to new structure based on Apache Hive.

    Parameters
    ----------
    submission : str | os.PathLike
        Folder following the old submission_template setup
    """

    submission = Path(submission)
    eval_data = itertools.chain(
        submission.glob(f"**/cm/*/eval/*.parquet"),
        submission.glob(f"**/pgm/*/eval/*.parquet"),
    )

    def folder_rename(d):
        new_name = d.parent / f'window=Y{d.name.split("_")[-1]}'
        d.rename(new_name)

    # Rename window folders in Apache Hive format
    window_folders = itertools.chain(
        submission.glob(f"cm/test_window_*"), submission.glob(f"pgm/test_window_*")
    )
    [folder_rename(d) for d in window_folders]
    # Delete evaluation files (must be re-estimated)
    [f.unlink() for f in eval_data]
    [f.unlink() for f in submission.glob(f"eval*.parquet")]

    # Cleanup old folders (only deletes if empty)
    [d.rmdir() for d in submission.glob(f"**/cm/*/eval/")]
    [d.rmdir() for d in submission.glob(f"**/pgm/*/eval/")]


def list_submissions(submissions_folder: str | os.PathLike) -> list[os.PathLike]:
    """Creates a list of paths to folders inside the submissions_folder.

    Parameters
    ----------
    submissions_folder : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template

    Returns
    -------
    list[os.PathLike]
        List of paths to submissions.
    """
    submissions_folder = Path(submissions_folder)
    return [
        submission
        for submission in submissions_folder.iterdir()
        if submission.is_dir() and not submission.stem == "__MACOSX"
    ]


def read_parquet(data_path: str | os.PathLike, filters=None) -> pd.DataFrame:
    """This function does not need to be used directly, it is mostly to document how to read folders with parquet files or single parquet files with a filter.

    Notes
    -----

    See https://arrow.apache.org/docs/python/compute.html#filtering-by-expressions for filter examples

    Examples
    --------
    >>> import pyarrow.compute as pac
    >>> import pyarrow.parquet as pq
    >>> filter = pac.field("year") >= 2017
    >>> df = pq.ParquetDataset(path_to_folder_with_apache_hive_structured_parquet_files, filters = filter).read().to_pandas()

    """

    table = pq.ParquetDataset(data_path, filters=filters)
    return table.read().to_pandas()


def is_parquet_in_target(submission: str | os.PathLike, target: TargetType) -> bool:
    """Test if there are any .parquet-files in the {submission}/{target} sub-folders.

    Parameters
    ----------
    submission : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template
    target : TargetType
        A string, either "pgm" for PRIO-GRID-months, or "cm" for country-months.

    Returns
    -------
    bool
        True if there are any .parquet files in target sub-folders.
    """
    return any((submission / target).glob("**/*.parquet"))


def get_target_data(
    submission: str | os.PathLike, target: TargetType, filters=None
) -> pd.DataFrame:
    """Reads folders with a "pgm" or "cm" sub-folder containing Apache Hive structured parquet-files.

    Notes
    -----
    See https://arrow.apache.org/docs/python/compute.html#filtering-by-expressions for filter examples.

    Examples
    --------
    >>> import pyarrow.compute as pac
    >>> import pyarrow.parquet as pq
    >>> from utilities import list_submissions, get_target_data

    >>> filter = pac.field("year") >= 2017
    >>> subs = list_submissions("./submissions/")
    >>> df = get_target_data(subs[0], target = "pgm", filters = filter)

    """
    submission = Path(submission)
    return read_parquet(submission / target, filters=filters)


def get_window_filters(window):
    pass


def reformat_output(df_crps: pd.DataFrame, df_ign: pd.DataFrame, df_mis: pd.DataFrame,
                  unit: str, save_to: str | os.PathLike) -> None:
    """
    Formats the output data and saves it as JSON files based on the specified level.
    Based on discussion with Henrik, the json file of cm level should be structured as follows:
    {
        "country_id": [crps, ign, mis],
        "country_id": [crps, ign, mis],
        ...
    }
    The file is saved as cm/{model_name}/{month_id}.json

    The json file of pgm level should be structured as follows:
    {
        "priogrid_gid": [crps, ign, mis],
        "priogrid_gid": [crps, ign, mis],
        ...
    }
    The file is saved as pgm/{model_name}/{nga}/{month_id}.json. Here, nga is the country code.
    The country code can be obtained by using the pg2nga.py file.

    Args:
        df_crps (DataFrame): DataFrame containing the CRPS values.
        df_ign (DataFrame): DataFrame containing the IGN values.
        df_mis (DataFrame): DataFrame containing the MIS values.
        unit (str): Level at which the data should be grouped ('country_id' or 'priogrid_gid').
        save_to (str or Path): Path to the specific model directory where the JSON files should be saved.
    """

    # df_crps = df_crps.rename(columns={'value': 'crps'})
    # df_ign = df_ign.rename(columns={'value': 'ign'})
    # df_mis = df_mis.rename(columns={'value': 'mis'})
    df_concat = pd.concat([df_crps, df_ign, df_mis], axis=1)

    save_to = Path(save_to)

    if unit == 'country_id':
        for month_id, group_by_month in df_concat.groupby(level='month_id'):
            data_dict = group_by_month[['crps', 'ign', 'mis']].groupby(level=unit).apply(
                lambda x: x.values.flatten().tolist()).to_dict()
            with open(f'{save_to}/{month_id}.json', 'w') as json_file:
                json.dump(data_dict, json_file, indent=4)
                
    elif unit == 'priogrid_gid':
        df_concat['nga'] = df_concat.index.get_level_values(level=unit).map(get_nga_by_pg)

        for nga, group_by_nga in df_concat.groupby(by='nga'):
            eval_nga_path = save_to / nga
            eval_nga_path.mkdir(parents=True, exist_ok=True)

            for month_id, group_by_month in group_by_nga.groupby(level='month_id'):
                data_dict = group_by_month[['crps', 'ign', 'mis']].groupby(level=unit).apply(
                    lambda x: x.values.flatten().tolist()).to_dict()

                with open(f'{eval_nga_path}/{month_id}.json', 'w') as json_file:
                    json.dump(data_dict, json_file, indent=4)


def match_actual_prediction_index(actuals, predictions):
    """
    Matches the month and target range of the actual and prediction dataframes.
    There is one team that has more country_ids that we have in the actuals data.
    So we need to filter out the extra country_ids from the predictions data.

    For year 2024, predictions do not cover the whole year window (this is by design).
    So we need to align the actuals and predictions dataframes to the same month range.

    Args:
        actuals (DataFrame): DataFrame containing the actual values.
        predictions (DataFrame): DataFrame containing the predictions.
    """

    # match target range
    predictions_unit = predictions.index.get_level_values(1)
    actuals_unit = actuals.index.get_level_values(1)
    if predictions_unit.unique().difference(actuals_unit.unique()).any():
        # logging.warning(f"Target range mismatch! Prediction unit values "
        #                 f"{predictions_unit.unique().difference(actuals_unit.unique()).tolist()} "
        #                 f"are not included in the actuals. Changing predictions target range to match actuals unit range.")
        predictions = predictions[predictions_unit.isin(actuals_unit)]

    if actuals_unit.unique().difference(predictions_unit.unique()).any():
        raise ValueError(f"Target range mismatch! Target unit values "
                         f"{actuals_unit.unique().difference(predictions_unit.unique()).tolist()} "
                            f"are not included in the predictions. Please update the predictions data.")


    # match month_id
    predictions_month = predictions.index.get_level_values("month_id")
    actuals_month = actuals.index.get_level_values("month_id")
    predictions_start, predictions_end = predictions_month.min(), predictions_month.max()
    actuals_start, actuals_end = actuals_month.min(), actuals_month.max()

    # The predictions might not start at the same month as the actuals start.
    # Also, we might not have actuals for the whole year, so the end is the last month of the actuals
    start, end = predictions_start, actuals_end

    if actuals_end < predictions_start:
        raise ValueError(f"Actuals end month {actuals_end} is before predictions start month {predictions_start}. "
                         f"Please update the actuals data.")

    if predictions_start != actuals_start or predictions_end != actuals_end:
        # logging.warning(
        #     f"Month range mismatches! Actuals month range: {actuals_start} to {actuals_end}, "
        #     f"Predictions month range: {predictions_start} to {predictions_end}. "
        #     f"Changing index of actuals and predictions to range: {start} to {end}.")
        actuals = actuals[(actuals_month >= start) & (actuals_month <= end)]
        predictions = predictions[(predictions_month >= start) & (predictions_month <= end)]
        return actuals, predictions

    return actuals, predictions


def get_nga_by_pg(value):
    for key, values in pgIds.items():
        if value in values:
            return key
    raise ValueError(f"{value} doesn't have a corresponding country")


def remove_duplicated_indexes(df):
    if df.index.duplicated().any():
        df_unique = df[~df.index.duplicated(keep='first')]
    else:
        df_unique = df
    return df_unique
