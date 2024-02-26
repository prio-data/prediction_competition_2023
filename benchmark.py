# Run this in Terminal (MacOS):
# python3 benchmark.py --feature_folder /Documents/features --target pgm --year 2018 2019 2020 2021 --benchmark_name boot --month_lag 3 --save_folder_path /Documents/features

import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import pyarrow.compute as pac
import yaml
import datetime
from dateutil.relativedelta import relativedelta
import os

def date_to_month_id(date: datetime.date):
    """
    Convert a date to a corresponding month ID starting from January 1980.

    Args:
        date (datetime.date): The date to convert.

    Returns:
        int: The month ID.
    """
    views_start_date = datetime.date(1980, 1, 1)
    r = relativedelta(date, views_start_date)
    return (r.years * 12) + r.months + 1

def main():
    parser = argparse.ArgumentParser(description="Global Bootstrap Benchmark Script")
    parser.add_argument("--feature_folder", type=str, help="Path to the feature folder")
    parser.add_argument("--target", type=str, choices=["pgm", "cm"], help="Target type (pgm or cm)")
    parser.add_argument("--years", type=int, nargs='+', help="List of years")
    parser.add_argument("--benchmark_name", type=str, help="boot - Bootstrap, last - last historical, conflictology - conflictology benchmarking")
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
        start_date = datetime.date(year = year, month = 1, day = 1)
        if benchmark_name == 'boot':
            result = global_bootstrap(feature_folder, target, start_date, month_lag, num_samples=1000)
        elif benchmark_name == 'last':
            result = last_observed_poisson(feature_folder, target, start_date, month_lag, num_samples=1000)
        elif benchmark_name == 'conflictology':
            result = conflictology(feature_folder, target, start_date, month_lag, num_samples=12)
        save_results(result, benchmark_name, target, start_date, save_folder_path)
    
    # Do something with the result, e.g., print or save it.
    print(result)
    
def save_results(result, benchmark_name, target, start_date, save_folder_path):
    """
    Save benchmark results to Parquet and create a YAML submission details file.

    Args:
        result (pandas.DataFrame): The benchmark result to be saved.
        benchmark_name (str): The name of the benchmark.
        target (str): The target type (e.g., 'cm' or 'pgm').
        start_date (datetime.date): The start date for the benchmark.
        save_folder_path (str): The path to the folder where results will be saved.
    """
    # Create a folder with the benchmark name
    benchmark_folder = os.path.join(save_folder_path, benchmark_name)
    os.makedirs(benchmark_folder, exist_ok=True)

    year = start_date.year

    # Create a folder for the year within the unit folder (cm or pgm)
    unit_folder = os.path.join(benchmark_folder, target)
    os.makedirs(unit_folder, exist_ok=True)

    # Create a folder for the year within the unit folder
    year_folder = os.path.join(unit_folder, f'window=Y{year}')
    os.makedirs(year_folder, exist_ok=True)

    # Define the file name
    file_name = f'bm_{benchmark_name}_{target}_{year}.parquet'

    # Save the result to a Parquet file
    result.to_parquet(os.path.join(year_folder, file_name))

    # Create the YAML file with submission details within the benchmark folder
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

    
    yaml_file_path = os.path.join(benchmark_folder, "submission_details.yml")
    
    if not os.path.exists(yaml_file_path):
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(submission_details, yaml_file,default_flow_style=False,sort_keys=False)
        

    
def global_bootstrap(feature_folder, target, start_date, month_lag=3, num_samples=1000):
    """
    Perform global bootstrap for benchmarking.

    Args:
        feature_folder (str): The path to the feature folder containing the data.
        target (str): The target type, either 'cm' or 'pgm'.
        start_date (datetime.date): The start date for the benchmark.
        month_lag (int, optional): The month lag for prediction. Default is 3.
        num_samples (int, optional): The number of bootstrap samples. Default is 1000.

    Returns:
        pandas.DataFrame: The benchmark results after global bootstrap.
    """
    
    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm.')
    
    window_start_month = date_to_month_id(start_date)
    last_observed_month_id = pac.field("month_id") == window_start_month - month_lag
    last_observed_data = pq.ParquetDataset(feature_folder / target, filters = last_observed_month_id).read(columns = [unit, "ged_sb"]).to_pandas()
    
    filter_last_year = (pac.field("month_id") <= window_start_month - month_lag) & (pac.field("month_id") > window_start_month - (12 + month_lag))
    pool = pq.ParquetDataset(feature_folder / target, filters=filter_last_year).read(columns=["ged_sb"]).to_pandas()
    
    dfs = []
    for month_id in range(window_start_month, window_start_month+12):
        df = last_observed_data.copy()
        df["month_id"] = month_id
        df['outcome'] = np.random.choice(pool["ged_sb"], size=(df.shape[0], num_samples), replace=True).tolist()
        df = df.drop(columns = "ged_sb")
        df = df.explode('outcome').astype('int32')
        df['draw'] = df.groupby(['month_id', unit]).cumcount()
        df.set_index(['month_id', unit, 'draw'], inplace=True)
        dfs.append(df)
    return pd.concat(dfs)

def last_observed_poisson(feature_folder, target, start_date, month_lag=3, num_samples=1000):
    """
    Perform last observed Poisson benchmarking.

    Args:
        feature_folder (str): The path to the feature folder containing the data.
        target (str): The target type, either 'cm' or 'pgm'.
        start_date (datetime.date): The start date for the benchmark.
        month_lag (int, optional): The month lag for prediction. Default is 3.
        num_samples (int, optional): The number of Poisson samples to generate. Default is 1000.

    Returns:
        pandas.DataFrame: The benchmark results after last observed Poisson benchmarking.
    """
    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm.')
    
    window_start_month = date_to_month_id(start_date)
    last_observed_month_id = (pac.field("month_id") == window_start_month - month_lag)
    last_observed_data = pq.ParquetDataset(feature_folder / target, filters=last_observed_month_id).read(columns=[unit, "ged_sb"]).to_pandas()
        
    dfs = []
    for month_id in range(window_start_month, window_start_month+12):
        df = last_observed_data.copy()
        df["month_id"] = month_id
        df["outcome"] = df["ged_sb"].apply(lambda x: np.random.poisson(x, num_samples))
        df = df.drop(columns = "ged_sb")
        df = df.explode("outcome").astype("int32")
        df['draw'] = df.groupby(['month_id', unit]).cumcount()
        df.set_index(['month_id', unit, 'draw'], inplace=True)
        dfs.append(df)
    # Concatenate the copies row-wise
    return pd.concat(dfs)


def conflictology(feature_folder, target, start_date, month_lag=3, num_samples=1000):
    """
    Perform conflictology benchmarking.

    Args:
        feature_folder (str): The path to the feature folder containing the data.
        target (str): The target type, either 'cm' or 'pgm'.
        start_date (datetime.date): The start date for the benchmark.
        month_lag (int, optional): The month lag for prediction. Default is 3.
        num_samples (int, optional): The number of conflictology samples to generate. Default is 1000.

    Returns:
        pandas.DataFrame: The benchmark results after conflictology benchmarking.
    """
    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm.')

    window_start_month = date_to_month_id(start_date)
    last_observed_month_id = (pac.field("month_id") ==
                              window_start_month - month_lag)
    last_observed_data = pq.ParquetDataset(
        feature_folder / target, filters=last_observed_month_id).read(columns=[unit, "ged_sb"]).to_pandas()

    dfs = []
    for month_id in range(window_start_month, window_start_month+12):
        df = last_observed_data.copy()
        df["month_id"] = month_id
        print(df)
        print(df.shape[0])
        df['outcome'] = np.random.choice(df["ged_sb"], size=(df.shape[0], num_samples), replace=True).tolist()
        df = df.drop(columns="ged_sb")
        df = df.explode("outcome").astype("int32")
        df['draw'] = df.groupby(['month_id', unit]).cumcount()
        df.set_index(['month_id', unit, 'draw'], inplace=True)
        dfs.append(df)
    # Concatenate the copies row-wise
    return pd.concat(dfs)

if __name__ == "__main__":
    main()

#python3 benchmark.py --feature_folder /Documents/features --target pgm --year 2018 2019 2020 2021 --benchmark_name boot --month_lag 3 --save_folder_path /Documents/features