import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from utilities import TargetType, list_submissions
from concurrent.futures import ProcessPoolExecutor, as_completed


def resampling(df_list, loa, expected_samples=1000, weights=None, batch_size=500, save_path=None):
    """
    Args:
        df_list: List of DataFrames where each has:
                - MultiIndex: (month_id, country_id, member)
                - Column: 'outcome' (the sample values)
        loa: level of analysis (cm or pgm)
        weights: Optional DataFrame with index (month_id, country_id)
                 and columns for each model's weight
        batch_size: Number of samples to process at once to manage memory
        save_path: Path to save results incrementally
    """
    if loa == "cm":
        level = "country_id"
    elif loa == "pgm":
        level = "priogrid_gid"
    else:
        raise ValueError(f"Invalid level of analysis: {loa}")

    month_level_pairs = df_list[0].index.droplevel("member").unique()
    n_pairs = len(month_level_pairs)
    
    all_samples = np.zeros((n_pairs, expected_samples))
    
    for i, (month, id) in tqdm(enumerate(month_level_pairs), total=n_pairs, desc="Resampling"):
        samples = np.vstack([
            df.xs((month, id), level=["month_id", level])["outcome"].values
            for df in df_list
        ])
        
        if weights is None:
            all_samples[i] = np.random.choice(samples.ravel(), size=expected_samples, replace=True)
        else:
            model_weights = weights.loc[(month, id)].values
            sample_weights = np.repeat(model_weights, samples.shape[1])
            all_samples[i] = np.random.choice(
                samples.ravel(),
                size=expected_samples,
                replace=True,
                p=sample_weights / sample_weights.sum(),
            )
        
        if save_path:
            member_indices = pd.MultiIndex.from_product(
                [[month], [id], range(expected_samples)],
                names=["month_id", level, "member"]
            )
            month_df = pd.DataFrame({"outcome": all_samples[i]}, index=member_indices)
            
            if i == 0:
                month_df.to_parquet(save_path, engine='pyarrow')
            else:
                existing_df = pd.read_parquet(save_path, engine='pyarrow')
                combined_df = pd.concat([existing_df, month_df])
                combined_df.to_parquet(save_path, engine='pyarrow')
                del existing_df, combined_df
            
            del month_df
        
        del samples
    
    if not save_path:
        member_indices = pd.MultiIndex.from_product(
            [month_level_pairs.get_level_values(0),
             month_level_pairs.get_level_values(1),
             range(expected_samples)],
            names=["month_id", level, "member"]
        )
        return pd.DataFrame({"outcome": all_samples.ravel()}, index=member_indices)
    return None


def process_window(args):
    try:
        target, window, submissions, save_to, expected_samples, weights = args
        
        df_list = []
        for submission in submissions:
            if any((submission / target).glob("**/*.parquet")):
                file_name = f"{submission.name}_{target}_{window}.parquet"
                df = pd.read_parquet(
                    submission / target / f"window={window}" / file_name,
                    engine='pyarrow',
                    use_threads=True
                )
                df_list.append(df)
        
        if df_list:
            print(f"Resampling {target} {window} with {len(df_list)} models")
            save_path = save_to / target / f"window={window}" / "ensemble.parquet"
            resampling(df_list, target, expected_samples, weights, save_path=save_path)
            del df_list
    except Exception as e:
        print(f"Error processing {target} {window}: {str(e)}")


def resampling_ensemble(
    submissions: str | os.PathLike,
    save_to: str | os.PathLike,
    targets: list[TargetType] = ["cm", "pgm"],
    windows: list[str] = [
        "Y2018", "Y2019", "Y2020", "Y2021",
        "Y2022", "Y2023", "Y2024", "Y2025",
    ],
    expected_samples: int = 1000,
    weights: pd.DataFrame = None,
    n_workers: int = None,
):
    """
    Args:
        submissions: List of submission paths
        save_to: Path to save the resampled ensemble
        targets: List of targets
        windows: List of windows
        expected_samples: Number of samples to resample
        weights: DataFrame of weights
        n_workers: Number of parallel workers (default: number of CPU cores)
    """
    submissions = Path(submissions)
    save_to = Path(save_to)
    submissions = list_submissions(submissions)        
    
    args_list = []
    for target in targets:
        for window in windows:
            (save_to / target / f"window={window}").mkdir(parents=True, exist_ok=True)
            args_list.append((target, window, submissions, save_to, expected_samples, weights))

    n_workers = n_workers or os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_window, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error in parallel processing: {str(e)}")


if __name__ == "__main__":
    resampling_ensemble(
        submissions="./final_submissions_cleaned",
        save_to="./ensembles",
        targets=["cm"],
    )
