import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.utilities import list_submissions
from utils.set_logger import set_logger

logger = set_logger("process_logger", "logs/process_windows.log")


def separate_df_by_month(df):
    df_2024 = df[df.index.get_level_values("month_id") < 541]
    df_2025 = df[df.index.get_level_values("month_id") >= 541]
    return df_2024, df_2025


def process_windows(submissions: Path | str, save_to: Path | str = None) -> None:
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
        save_to.mkdir(parents=True, exist_ok=True)

    for submission in list_submissions(submissions):
        submission_dir = save_to / submission.name
        submission_dir.mkdir(parents=True, exist_ok=True)
        for item in submission.iterdir():
            if item.is_file():
                shutil.copy2(item, submission_dir / item.name)

    for folder in tqdm(
        submissions.rglob("*"),
        desc="Processing",
        total=len(list(submissions.rglob("*"))),
    ):
        if folder.is_dir():
            if folder.name == "window=Y2024":
                parquet_files = list(folder.glob("*.parquet"))
                if parquet_files:
                    parquet_file = parquet_files[0]
                    df = pd.read_parquet(parquet_file)
                    df_2024, df_2025 = separate_df_by_month(df)

                    target_folder_2024 = save_to / folder.relative_to(submissions)
                    target_folder_2024.mkdir(parents=True, exist_ok=True)
                    df_2024.to_parquet(target_folder_2024 / parquet_file.name)

                    target_folder_2025 = target_folder_2024.parent / "window=Y2025"
                    target_folder_2025.mkdir(parents=True, exist_ok=True)
                    if "2024" in parquet_file.name:
                        df_2025.to_parquet(
                            target_folder_2025
                            / parquet_file.name.replace("2024", "2025")
                        )
                    else:
                        df_2025.to_parquet(target_folder_2025 / parquet_file.name)
            else:
                target_folder = save_to / folder.relative_to(submissions)
                target_folder.mkdir(parents=True, exist_ok=True)

                for parquet_file in folder.glob("*.parquet"):
                    shutil.copy2(parquet_file, target_folder / parquet_file.name)


def main():
    parser = argparse.ArgumentParser(
        description="Process windows in the submission folder."
    )
    parser.add_argument(
        "-s",
        metavar="submissions",
        type=str,
        required=True,
        help="Path to the source folder",
    )
    parser.add_argument(
        "-st",
        metavar="save_to",
        type=str,
        default=None,
        help="Path to the target folder. If not provided, the source folder will be used.",
    )
    args = parser.parse_args()

    process_windows(args.s, args.st)


if __name__ == "__main__":
    main()
