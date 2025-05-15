import pandas as pd
from pathlib import Path
from tqdm import tqdm


def separate_df_by_month(df):
    df_2024 = df[df.index.get_level_values('month_id') < 541]
    df_2025 = df[df.index.get_level_values('month_id') >= 541]
    return df_2024, df_2025


def process_windows(folder1: str, folder2: str=None) -> None:
    """
    By design the window for year 2025 is included in the window for year 2024. 
    This function separates the data by year and creates a new folder for the year 2025.
    """
    base_path1 = Path(folder1)
    if folder2 is None:
        base_path2 = Path(folder1)
        logger.info(f"No target folder provided, saving to {base_path2}")
    else:
        base_path2 = Path(folder2)

    for folder in tqdm(base_path1.rglob('*'), desc='Processing', total=len(list(base_path1.rglob('*')))):
        if folder.is_dir() and folder.name == "window=Y2024":
            parquet_files = list(folder.glob('*.parquet'))
            if parquet_files:
                # logger.info(f"Processing {parquet_files[0]}")
                parquet_file = parquet_files[0]
                df = pd.read_parquet(parquet_file) 
                df_2024, df_2025 = separate_df_by_month(df)

                target_folder_2024 = base_path2 / folder.relative_to(base_path1)
                target_folder_2024.mkdir(parents=True, exist_ok=True)
                df_2024.to_parquet(target_folder_2024 / parquet_file.name)

                target_folder_2025 = target_folder_2024.parent / 'window=Y2025'
                target_folder_2025.mkdir(parents=True, exist_ok=True)
                if "2024" in parquet_file.name:
                    df_2025.to_parquet(target_folder_2025 / parquet_file.name.replace('2024', '2025'))
                else:
                    df_2025.to_parquet(target_folder_2025 / parquet_file.name)


if __name__ == "__main__":
    process_windows('./final_submissions_cleaned/')