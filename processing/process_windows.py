import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils.set_logger import set_logger
import shutil
from utils.utilities import list_submissions

logger = set_logger('process_logger', 'logs/process_windows.log')


def separate_df_by_month(df):
    df_2024 = df[df.index.get_level_values('month_id') < 541]
    df_2025 = df[df.index.get_level_values('month_id') >= 541]
    return df_2024, df_2025


def process_windows(submissions: Path | str, save_to: Path | str=None) -> None:
    """Process windows in the submission folder.
    For window=Y2024, separate the data into Y2024 and Y2025.
    For other windows, copy the data as is.

    Parameters
    ----------
    submissions : Path | str
        Path to the source folder
    save_to : Path | str
        Path to the target folder
    """
    submissions = Path(submissions)
    if not save_to:
        save_to = Path(submissions)
        logger.info(f"No target folder provided, saving to {save_to}")
    else:
        save_to = Path(save_to)

    for submission in list_submissions(submissions):
        for item in submission.iterdir():
            if item.is_file():
                shutil.copy2(item, save_to / item.name)

    for folder in tqdm(submissions.rglob('*'), desc='Processing', total=len(list(submissions.rglob('*')))):
        if folder.is_dir():
            if folder.name == "window=Y2024":
                parquet_files = list(folder.glob('*.parquet'))
                if parquet_files:
                    parquet_file = parquet_files[0]
                    df = pd.read_parquet(parquet_file) 
                    df_2024, df_2025 = separate_df_by_month(df)

                    target_folder_2024 = save_to / folder.relative_to(submissions)
                    target_folder_2024.mkdir(parents=True, exist_ok=True)
                    df_2024.to_parquet(target_folder_2024 / parquet_file.name)

                    target_folder_2025 = target_folder_2024.parent / 'window=Y2025'
                    target_folder_2025.mkdir(parents=True, exist_ok=True)
                    if "2024" in parquet_file.name:
                        df_2025.to_parquet(target_folder_2025 / parquet_file.name.replace('2024', '2025'))
                    else:
                        df_2025.to_parquet(target_folder_2025 / parquet_file.name)
            else:
                target_folder = save_to / folder.relative_to(submissions)
                target_folder.mkdir(parents=True, exist_ok=True)
                
                for parquet_file in folder.glob('*.parquet'):
                    shutil.copy2(parquet_file, target_folder / parquet_file.name)
