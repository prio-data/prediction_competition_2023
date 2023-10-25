import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import pyarrow.compute as pac
import yaml


def main():
    parser = argparse.ArgumentParser(description="Global Bootstrap Benchmark Script")
    parser.add_argument("--feature_folder", type=str, help="Path to the feature folder")
    parser.add_argument("--target", type=str, choices=["pgm", "cm"], help="Target type (pgm or cm)")
    parser.add_argument("--years", type=int, nargs='+', help="List of years")
    parser.add_argument("--benchmark_name", type=str, help="boot - Bootstrap, last - last historical")
    parser.add_argument("--month_lag", type=int, help="Specify the month lag for prediction")
    parser.add_argument("--save_folder_path", type=str, help="Specify a folder path name to save parquet files")
    
    args = parser.parse_args()
    
    feature_folder = Path(args.feature_folder)
    target = args.target
    years = args.years
    benchmark_name = args.benchmark_name
    month_lag = args.month_lag
    save_folder_path = args.save_folder_path
    
    for year in years:
        if benchmark_name == 'boot':
            result = global_bootstrap(feature_folder, target, year, month_lag, num_samples=1000)
        elif benchmark_name == 'last':
            result = last_observed_poisson(feature_folder, target, year, month_lag, num_samples=1000)
        save_results(result, benchmark_name, target, year, save_folder_path)
    
    # Do something with the result, e.g., print or save it.
    print(result)
    
def save_results(result, benchmark_name, target, year, save_folder_path):
    # Create a folder with the benchmark name
    benchmark_folder = Path(save_folder_path) / benchmark_name
    benchmark_folder.mkdir(parents=True, exist_ok=True)
    
    # Create a folder for the year
    year_folder = benchmark_folder / f'test_window_{year}'
    year_folder.mkdir(parents=True, exist_ok=True)
    
    # Define the file name
    file_name = f'bm_{benchmark_name}_{year}.parquet'
    
    # Save the result to a Parquet file
    result.to_parquet(year_folder / file_name)
    
     # Create the YAML file with submission details
    submission_details = {
        "team": "benchmark",
        "short_title": f"{benchmark_name} {target} {year}",
        "even_shorter_identifier": f"{benchmark_name}_{target}_{year}",
        "authors": [
            {
                "name": "Your Name",
                "affil": "Your Affiliation",
                "contact": "Your Contact Info"
            }
        ]
    }
    
    with open(benchmark_folder / "submission_details.yml", 'w') as yaml_file:
        yaml.dump(submission_details, yaml_file)

    
def global_bootstrap(feature_folder, target, year, month_lag=3, num_samples=1000):
    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm.')
    
    filter_year = pac.field("year") == year
    df = pq.ParquetDataset(feature_folder / target, filters=filter_year).read(columns=[unit, "month_id"]).to_pandas()
    
    filter_year = (pac.field("month_id") <= df.month_id.min() - month_lag) & (pac.field("month_id") > df.month_id.min() - (12 + month_lag))
    pool = pq.ParquetDataset(feature_folder / target, filters=filter_year).read(columns=["ged_sb"]).to_pandas()
    df['outcome'] = np.random.choice(pool["ged_sb"], size=(pool.shape[0], num_samples), replace=True).tolist()
    df = df.explode('outcome').astype('int32')
    df['draw'] = df.groupby(['month_id', unit]).cumcount()
    df.set_index(['month_id', unit, 'draw'], inplace=True)
    return df

def last_observed_poisson(feature_folder, target, year, month_lag=3, num_samples=1000):
    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm.')
    
    filter_year = pac.field("year") == year
    df = pq.ParquetDataset(feature_folder / target, filters=filter_year).read(columns=[unit, "month_id"]).to_pandas()

    filter_year = (pac.field("month_id") == df.month_id.min() - month_lag)
    min_month = pq.ParquetDataset(feature_folder / target, filters=filter_year).read(columns=["ged_sb"]).to_pandas()
    copies = [min_month.copy() for _ in range(12)]
    
    # Concatenate the copies row-wise
    min_month = pd.concat(copies, axis=0, ignore_index=True)
    df['outcome'] = [np.random.poisson(value, num_samples) for value in min_month["ged_sb"]]
    df = df.explode('outcome').astype('int32')
    df['draw'] = df.groupby(['month_id', unit]).cumcount()
    df.set_index(['month_id', unit, 'draw'], inplace=True)
    return df

if __name__ == "__main__":
    main()
