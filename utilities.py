from pathlib import Path
import pyarrow.dataset as pad
import os
import itertools
from typing import Literal

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
        new_name = d.parent / f'window={d.name.split("_")[-1]}'
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

TargetType = Literal["cm", "pgm"]
def get_predictions(submission: str|os.PathLike, target: TargetType) -> pad.Dataset:
    pass