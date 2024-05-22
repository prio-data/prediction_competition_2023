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
import time

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
        start_date = datetime.date(year = year, month = 1, day = 1)
        if benchmark_name == 'boot':
            result = global_bootstrap(feature_folder, target, start_date, month_lag, num_samples=1000)
        elif benchmark_name == 'last':
            result = last_observed_poisson(feature_folder, target, start_date, month_lag, num_samples=1000)
        elif benchmark_name == 'last_without_poisson':
            result = last_observed_without_poisson(feature_folder, target, start_date, month_lag, num_samples=1000)
        elif benchmark_name == 'boot_240':
            result = global_bootstrap_240_months(feature_folder, target, start_date, month_lag, num_samples=1000)
        elif benchmark_name == 'zero':
            result = exactly_zero(feature_folder, target, start_date, month_lag, num_samples=1000)
        elif benchmark_name == 'conflictology':
            result = conflictology(feature_folder, target, start_date, month_lag)
        elif benchmark_name == 'conflictology_n':
            result = conflictology_neighbors(feature_folder, target, start_date, month_lag)

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


def last_observed_without_poisson(feature_folder, target, start_date, month_lag=3, num_samples=1000):
    """
    Perform last observed benchmarking without Poisson sampling.

    Args:
        feature_folder (str): The path to the feature folder containing the data.
        target (str): The target type, either 'cm' or 'pgm'.
        start_date (datetime.date): The start date for the benchmark.
        month_lag (int, optional): The month lag for prediction. Default is 3.
        num_samples (int, optional): The number of Poisson samples to generate. Default is 1000.

    Returns:
        pandas.DataFrame: The benchmark results after last observed benchmarking.
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
        df["outcome"] = df["ged_sb"]
        df = df.drop(columns="ged_sb")
        df = df.explode("outcome").astype("int32")
        df['draw'] = df.groupby(['month_id', unit]).cumcount()
        df.set_index(['month_id', unit, 'draw'], inplace=True)
        dfs.append(df)
    # Concatenate the copies row-wise
    return pd.concat(dfs)


def global_bootstrap_240_months(feature_folder, target, start_date, month_lag=3, num_samples=1000):
    """
    Perform global bootstrap for benchmarking for last 240 months

    Args:
        feature_folder (str): The path to the feature folder containing the data.
        target (str): The target type, either 'cm' or 'pgm'.
        start_date (datetime.date): The start date for the benchmark.
        month_lag (int, optional): The month lag for prediction. Default is 3.
        num_samples (int, optional): The number of bootstrap samples. Default is 1000.

    Returns:
        pandas.DataFrame: The benchmark results after global bootstrap for last 240 months
    """

    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm.')

    window_start_month = date_to_month_id(start_date)
    last_observed_month_id = pac.field(
        "month_id") == window_start_month - month_lag
    last_observed_data = pq.ParquetDataset(
        feature_folder / target, filters=last_observed_month_id).read(columns=[unit, "ged_sb"]).to_pandas()

    filter_last_year = (pac.field("month_id") <= window_start_month - month_lag) & (
        pac.field("month_id") > window_start_month - (240 + month_lag))
    pool = pq.ParquetDataset(
        feature_folder / target, filters=filter_last_year).read(columns=["ged_sb"]).to_pandas()

    dfs = []
    for month_id in range(window_start_month, window_start_month+12):
        df = last_observed_data.copy()
        df["month_id"] = month_id
        df['outcome'] = np.random.choice(pool["ged_sb"], size=(
            df.shape[0], num_samples), replace=True).tolist()
        df = df.drop(columns="ged_sb")
        df = df.explode('outcome').astype('int32')
        df['draw'] = df.groupby(['month_id', unit]).cumcount()
        df.set_index(['month_id', unit, 'draw'], inplace=True)
        dfs.append(df)
    return pd.concat(dfs)


def exactly_zero(feature_folder, target, start_date, month_lag=3, num_samples=1000):
    """
    A benchmark that predicts exactly zero for all draws and units.

    Args:
        feature_folder (str): The path to the feature folder containing the data.
        target (str): The target type, either 'cm' or 'pgm'.
        start_date (datetime.date): The start date for the benchmark.
        month_lag (int, optional): The month lag for prediction. Default is 3.
        num_samples (int, optional): The number of samples. Default is 1000.

    Returns:
        pandas.DataFrame: The benchmark exactly zero as outcome.
    """

    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm.')

    window_start_month = date_to_month_id(start_date)
    last_observed_month_id = pac.field(
        "month_id") == window_start_month - month_lag
    last_observed_data = pq.ParquetDataset(
        feature_folder / target, filters=last_observed_month_id).read(columns=[unit, "ged_sb"]).to_pandas()
    dfs = []
    for month_id in range(window_start_month, window_start_month+12):
        df = last_observed_data.copy()
        df["month_id"] = month_id
        df['outcome'] = [np.zeros(num_samples).tolist() for _ in range(len(df))]
        df = df.drop(columns="ged_sb")
        df = df.explode('outcome').astype('int32')
        df['draw'] = df.groupby(['month_id', unit]).cumcount()
        df.set_index(['month_id', unit, 'draw'], inplace=True)
        dfs.append(df)
    return pd.concat(dfs)


def conflictology(feature_folder, target, start_date, month_lag=3):
    """
    12 - month Conflictological prediction for year X based on the data from November X-2 through October X-1 

    Args:
        feature_folder (str): The path to the feature folder containing the data.
        target (str): The target type, either 'cm' or 'pgm'.
        start_date (datetime.date): The start date for the benchmark.
        month_lag (int, optional): The month lag for prediction. Default is 3.

    Returns:
        pandas.DataFrame: The predictions for 12 - month conflictology
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

    filter_last_year = (pac.field("month_id") <= window_start_month - month_lag) & (
        pac.field("month_id") > window_start_month - (12 + month_lag))
    pool = pq.ParquetDataset(
        feature_folder / target, filters=filter_last_year).read(columns=['month_id', unit, "ged_sb"]).to_pandas()
    print('Pool: ', pool)

    print('Last observed data: ', last_observed_data)

    df = pool.copy()
    grouped = df.groupby(unit)['ged_sb'].apply(list).reset_index()
    df['outcome'] = None
    outcome_dict = dict(zip(grouped[unit], grouped['ged_sb']))
    df['outcome'] = df[unit].map(outcome_dict)

    for month_id in range(window_start_month, window_start_month+12):
        print('Month ID: ', month_id)
        print(df.loc[df['month_id'] == month_id-month_lag-11, 'month_id'])
        df.loc[df['month_id'] == month_id-month_lag-11, 'month_id'] = month_id
        print(df.loc[df['month_id'] == month_id, 'month_id'])
    ###
    df = df.drop(columns="ged_sb")
    df = df.explode("outcome").astype("int32")
    df['draw'] = df.groupby(['month_id', unit]).cumcount()
    df.set_index(['month_id', unit, 'draw'], inplace=True)
    return df


def conflictology_neighbors(feature_folder, target, start_date, month_lag=3):
    """
    12 - month Conflictological prediction for 9 priogrids only at pgm level only. The 8 nearby neighbors of a priogrid are cocatenated to the priogrid to create a distribution.

    Args:
        feature_folder (str): The path to the feature folder containing the data.
        target (str): The target type, 'pgm'.
        start_date (datetime.date): The start date for the benchmark.
        month_lag (int, optional): The month lag for prediction. Default is 3.

    Returns:
        pandas.DataFrame: The predictions with 9 neighbors with 12 - month conflictology
    """
    if target == "pgm":
        unit = "priogrid_gid"
    else:
        raise ValueError('Target must be "pgm"')

    window_start_month = date_to_month_id(start_date)
    last_observed_month_id = (pac.field("month_id") ==
                              window_start_month - month_lag)
    last_observed_data = pq.ParquetDataset(
        feature_folder / target, filters=last_observed_month_id).read(columns=[unit, "ged_sb"]).to_pandas()

    filter_last_year = (pac.field("month_id") <= window_start_month - month_lag) & (
        pac.field("month_id") > window_start_month - (12 + month_lag))
    pool = pq.ParquetDataset(
        feature_folder / target, filters=filter_last_year).read(columns=['month_id', unit, "ged_sb"]).to_pandas()


    df = pool.copy()
    grouped = df.groupby(unit)['ged_sb'].apply(list).reset_index()
    df['outcome'] = None
    outcome_dict = dict(zip(grouped[unit], grouped['ged_sb']))
    df['outcome'] = df[unit].map(outcome_dict)
    # additional code for neighbors :)
    print('sdjfksdjflsdjfldsjfldsjfldsj',df['outcome'])
    
    for month_id in range(window_start_month, window_start_month+12):
        print('Month ID: ', month_id)
        print(df.loc[df['month_id'] == month_id-month_lag-11, 'month_id'])
        df.loc[df['month_id'] == month_id-month_lag-11, 'month_id'] = month_id

        #print(df.loc[df['month_id'] == month_id, 'month_id'])
    ###
# for u in range(62356,190512):
    # initialize a multiindex dictionary
    midict = {}
    
    start_time = time.time()

    for u in df[unit].unique().tolist():
    #for u in [62356, 80318]:
    #for u in np.random.choice(df[unit].unique().tolist(), size=100, replace=False):
        for m in range(window_start_month, window_start_month+12):
            print(window_start_month)
            # df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'] =  df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+720), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-720), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+1), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-1), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+719),'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-719),'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+721),'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-721), 'outcome']
            
            try:
                
                #df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'] = df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+720), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-720), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+1), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-1), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+719),'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-719),'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+721),'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-721), 'outcome']
                print('Unit: ', df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'])
                print('Unit: ', df.loc[(df['month_id'] == m) & (
                    df['priogrid_gid'] == u-1), 'outcome'])
                print('Unit: ', df.loc[(df['month_id'] == m) & (
                    df['priogrid_gid'] == u+1), 'outcome'] + df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-1), 'outcome'])

                print('xxxxxxxxxxx', 'Unit: ', df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'].tolist() + (df.loc[(df['month_id'] == m) & (
                    df['priogrid_gid'] == u-1), 'outcome'].tolist()) if (u - 1 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                print('yyyyyyyyyyyyyy', 'Unit: ', df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'].tolist() + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-1), 'outcome'].tolist()) if (u - 1 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+1), 'outcome'].tolist()) if (u + 1 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-720), 'outcome'].tolist()) if (u - 720 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+720), 'outcome'].tolist()) if (u + 720 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-719), 'outcome'].tolist()) if (u - 719 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+719), 'outcome'].tolist()) if (u + 719 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-721), 'outcome'].tolist()) if (u - 721 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+721), 'outcome'].tolist()) if (u + 721 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                print('yyyyyyyyyyyyyy', 'Unit: ', len(df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'].tolist() + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-1), 'outcome'].tolist()) if (u - 1 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+1), 'outcome'].tolist()) if (u + 1 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-720), 'outcome'].tolist()) if (u - 720 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+720), 'outcome'].tolist()) if (u + 720 in df[unit].unique(
                )) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-719), 'outcome'].tolist()) if (u - 719 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+719), 'outcome'].tolist()) if (u + 719 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-721), 'outcome'].tolist()) if (u - 721 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+721), 'outcome'].tolist()) if (u + 721 in df[unit].unique()) else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))


                print('yesssssssssssss', 'Unit: ',
                    (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'].tolist()) + ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-1), 'outcome'].tolist()) if (u - 1 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) +
                    ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+1), 'outcome'].tolist()) if (u + 1 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) +
                    ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-720), 'outcome'].tolist()) if (u - 720 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) +
                    ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+720), 'outcome'].tolist()) if (u + 720 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) +
                    ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-719), 'outcome'].tolist()) if (u - 719 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) +
                    ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+719), 'outcome'].tolist()) if (u + 719 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) +
                    ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-721), 'outcome'].tolist()) if (u - 721 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) +
                    ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+721),
                             'outcome'].tolist()) if (u + 721 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                )
                part1 = (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'].tolist()) + ((df.loc[(df['month_id'] == m) & (
                    df['priogrid_gid'] == u-1), 'outcome'].tolist()) if (u - 1 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


                print('part1:', part1)

                part2 = ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+1), 'outcome'].tolist())
                        if (u + 1 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                print('part2:', part2)

                part3 = ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-720), 'outcome'].tolist())
                        if (u - 720 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                print('part3:', part3)

                part4 = ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+720), 'outcome'].tolist())
                        if (u + 720 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                print('part4:', part4)

                part5 = ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-719), 'outcome'].tolist())
                        if (u - 719 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                print('part5:', part5)

                part6 = ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+719), 'outcome'].tolist())
                        if (u + 719 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                print('part6:', part6)

                part7 = ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-721), 'outcome'].tolist())
                        if (u - 721 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                print('part7:', part7)

                part8 = ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+721), 'outcome'].tolist())
                        if (u + 721 in df[unit].unique()) else [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                print('part8:', part8)

                nested_list = (df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u), 'outcome'].tolist() + ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-1), 'outcome'].tolist()) if (u - 1 in df[unit].unique()) else [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) +
                               ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+1), 'outcome'].tolist()) if (u + 1 in df[unit].unique()) else [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) +
                               ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-720), 'outcome'].tolist()) if (u - 720 in df[unit].unique()) else [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) +
                               ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+720), 'outcome'].tolist()) if (u + 720 in df[unit].unique()) else [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) +
                               ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-719), 'outcome'].tolist()) if (u - 719 in df[unit].unique()) else [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) +
                               ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+719), 'outcome'].tolist()) if (u + 719 in df[unit].unique()) else [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) +
                                ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u-721), 'outcome'].tolist()) if (u - 721 in df[unit].unique()) else [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) +
                                ((df.loc[(df['month_id'] == m) & (df['priogrid_gid'] == u+721),
                         'outcome'].tolist()) if (u + 721 in df[unit].unique()) else [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
                
                flattened_list = [
                    item for sublist in nested_list for item in sublist]
                print ('Flattened list: ', flattened_list)
                print(len(df.loc[(df['month_id'] == m) & (
                    df['priogrid_gid'] == u), 'outcome']))

                midict[(m, u)] = {'outcome': flattened_list}

                

            except Exception as e:
                print(f"An error occurred: {e}")
                # which line is the error

                continue

    print(df)
    print(midict)
    df_n = pd.DataFrame.from_dict(midict,orient='index')
    df_n = df_n.rename_axis(index=['month_id', 'priogrid_gid'])

    print(df_n)
    df_n.reset_index(inplace=True)
    #
    #df = df.drop(columns="ged_sb")
    df_n = df_n.explode("outcome").astype("int32")
    df_n['draw'] = df_n.groupby(['month_id', unit]).cumcount()
    df_n.set_index(['month_id', unit, 'draw'], inplace=True)
    
    end_time = time.time()
    print('Time taken: ', end_time - start_time)
    return df_n

if __name__ == "__main__":
    main()

# python3 benchmark.py --feature_folder /Users/noorainkazmi/Documents/features --target pgm --year 2018 2019 2020 2021 --benchmark_name conflictology_n --month_lag 3 --save_folder_path /Users/noorainkazmi/Documents/Newssss
