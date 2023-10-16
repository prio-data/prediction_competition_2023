from pathlib import Path
import pyarrow.dataset as pad
import pyarrow.parquet as pq
import pandas as pd
import os
import itertools
from typing import Literal
import datetime

type TargetType = Literal["cm", "pgm"]

def date_to_views_month_id(date: datetime.date) -> int:
    views_start_date = datetime.date(1980, 1, 1)
    r = relativedelta(date, views_start_date)
    return (r.years * 12) + r.months + 1

def views_month_id_to_year(month_id):
    return 1980 + (month_id - 1) // 12

def views_month_id_to_month(month_id):
    return ((month_id - 1) % 12)+1

def views_month_id_to_date(month_id: int) -> datetime.date:
    df = pd.DataFrame()
    df["year"] = views_month_id_to_year(month_id)
    df["month"] = views_month_id_to_month(month_id)
    df["day"] = 1
    return pd.to_datetime(df, format = "%Y%M")

def migrate_submission_from_old(submission: str|os.PathLike) -> None:
    submission = Path(submission)
    eval_data = itertools.chain(submission.glob(f'**/cm/*/eval/*.parquet'),
                                submission.glob(f'**/pgm/*/eval/*.parquet'))
    def file_rename(f):
        target, window = f.parent.parts[-3:-1]
        eval_folder_path = submission / "eval" / target / window
        eval_folder_path.mkdir(parents=True, exist_ok=True)
        f.rename(eval_folder_path / f.name)
    
    def folder_rename(d):
        new_name = d.parent / f'year={d.name.split("_")[-1]}'
        d.rename(new_name)

    # Rename window folders in Apache Hive format
    window_folders = itertools.chain(submission.glob(f'cm/test_window_*'), submission.glob(f'pgm/test_window_*'))
    [folder_rename(d) for d in window_folders]
    # Move evaluation files into eval folder
    [file_rename(f) for f in eval_data]        
    # Move summary files into eval folder
    [f.rename(submission / "eval" / f.name) for f in submission.glob(f'eval*.parquet')]
    
    # Cleanup old folders (only deletes if empty)
    [d.rmdir() for d in submission.glob(f'**/cm/*/eval/')]
    [d.rmdir() for d in submission.glob(f'**/pgm/*/eval/')]

def list_submissions(submissions_folder: str|os.PathLike) -> list[os.PathLike]:
    submissions_folder = Path(submissions_folder)
    return [submission for submission in submissions_folder.iterdir() if submission.is_dir() and not submission.stem == "__MACOSX"]

def read_features(feature_folder: str|os.PathLike, target: TargetType, filters) -> pd.DataFrame:
    """
    See https://arrow.apache.org/docs/python/compute.html#filtering-by-expressions for filter examples
    
    import pyarrow.compute as pac
    filter = pac.field("year") >= 2017
    """
    feature_folder = Path(feature_folder)
    assert (feature_folder / target).exists()
    assert (feature_folder / "cm_features.parquet").exists()

    if target == "pgm":
        data_path = feature_folder / target
    elif target == "cm":
        data_path = feature_folder / "cm_features.parquet"
    else:
        ValueError(f'Target must be either "pgm" or "cm".')
        
    table = pq.ParquetDataset(data_path, filters = filters)
    return table.read().to_pandas()


def get_predictions(submission: str|os.PathLike, target: TargetType) -> pad.Dataset:
    pass