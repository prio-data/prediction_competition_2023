# python3 your_script.py --feature_folder /path/to/features --target cm --year 2018 --benchmark_name boot

import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import pyarrow.compute as pac

def main():
    parser = argparse.ArgumentParser(description="Global Bootstrap Benchmark Script")
    parser.add_argument("--feature_folder", type=str, help="Path to the feature folder")
    parser.add_argument("--target", type=str, choices=["pgm", "cm"], help="Target type (pgm or cm)")
    parser.add_argument("--year", type=int, help="Year")
    #new argument
    parser.add_argument("--benchmark_name",type=str,help="boot - Bootstrap, hist - last historical")
    parser.add_argument("--month_lag",type=int,help="specify the month lag for prediction" )
    args = parser.parse_args()
    feature_folder = Path(args.feature_folder)
    target = args.target
    year = args.year
    benchmark_name = args.benchmark_name
    month_lag = args.month_lag
    result = global_benchmark(benchmark_name, feature_folder, target, year,month_lag)
    result.to_parquet(f'{feature_folder}/{benchmark_name}_{target}_{year}.parquet')

    # Do something with the result, e.g., print or save it.
    print(result)
    
def select_unit(target):
    """
    Selects and returns the appropriate unit identifier based on the provided target.

    Args:
        target (str): A string representing the target unit. Should be one of the following:
            - "pgm" for the Political Geographic Model.
            - "cm" for the Country Model.

    Returns:
        str: The unit identifier based on the target.
        
    Raises:
        ValueError: If the target is not "pgm" or "cm".

    Example:
        To select the unit for the Political Geographic Model:
        ```
        unit = select_unit("pgm")
        ```

        To select the unit for the Country Model:
        ```
        unit = select_unit("cm")
        ```

    Note:
        This function is designed to handle specific targets, and any other values for the target will result in a ValueError.

    """
    if target == "pgm":
        return "priogrid_gid"
    elif target == "cm":
        return "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm".')
    
def filter_units(feature_folder,target,year,unit,month_lag):
    """
    Filter and retrieve data from Parquet datasets based on specified criteria.

    This function filters Parquet datasets based on the provided `year`, `unit`, and `month_lag` values. It reads data from the specified dataset and returns DataFrames containing the filtered data.

    Args:
        feature_folder (pathlib.Path): The folder containing Parquet datasets.
        target (str): The name of the Parquet dataset file (e.g., 'ged_data.parquet').
        year (int): The target year for filtering data.
        unit (str): The unit to filter on, such as 'country_id' or 'priogrid_gid'.
        month_lag (int): The number of months to lag behind the minimum 'month_id' for filtering.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: A tuple containing two DataFrames:
        - The first DataFrame contains the filtered data based on the specified year and unit.
        - The second DataFrame contains the filtered data based on the specified month_lag.

    Example:
        To filter data for a specific year, unit, and month lag:
        ```
        feature_folder = pathlib.Path('/path/to/parquet/datasets')
        target = 'ged_data.parquet'
        year = 2023
        unit = 'country_id'
        month_lag = 2

        df, pool = filter_units(feature_folder, target, year, unit, month_lag)
        ```

    Note:
        - `feature_folder` should be a pathlib.Path object pointing to the folder where the Parquet datasets are located.
        - `target` is the name of the Parquet dataset file within the specified folder.
        - `year` is used to filter data for a specific year.
        - `unit` specifies the unit to filter on, such as 'country_id' or 'priogrid_gid.'
        - `month_lag` determines the number of months to lag behind the minimum 'month_id' for filtering data.

    """
    filter = pac.field("year") == year
    df = pq.ParquetDataset(feature_folder / target, filters=filter).read(columns=[unit, "month_id"]).to_pandas()
    
    filter = (pac.field("month_id") <= df.month_id.min() - month_lag) & (pac.field("month_id") > df.month_id.min() - (12+month_lag))
    pool = pq.ParquetDataset(feature_folder / target, filters=filter).read(columns=["ged_sb"]).to_pandas()
    return df,pool

def historical_poisson_benchmark(data, value_column_name, num_samples=1000) -> list:
    """
    Generate Poisson-distributed samples based on last observed values.

    This function generates Poisson-distributed samples for each value in the specified data column, using the last observed value as the Poisson lambda parameter. The generated samples are returned as a list.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data.
        value_column_name (str): The name of the column in the DataFrame from which to extract values for generating Poisson samples.
        num_samples (int, optional): The number of Poisson-distributed samples to generate for each value. Default is 1000.

    Returns:
        list: A list of lists, where each inner list contains Poisson-distributed samples for a value in the specified column.

    Example:
        To generate Poisson samples based on the 'count' column in a DataFrame:
        ```
        data = pd.DataFrame({'value_column_name': [5, 10, 15, 20]})
        num_samples = 1000

        samples = last_observed_poisson_benchmark(data, 'value_column_name', num_samples)
        ```

    Note:
        - `data` should be a pandas DataFrame containing the data.
        - `value_column_name` is the name of the column from which values are extracted for generating samples.
        - `num_samples` specifies the number of Poisson-distributed samples to generate for each value. The default is 1000.

    """
    return [np.random.poisson(value, num_samples) for value in data[value_column_name]]

def historical_bootstrap_benchmark(data, value_column_name, num_samples=1000) -> list:
    return np.random.choice(data[value_column_name], size=(data.shape[0], num_samples), replace=True).tolist()

def global_benchmark(benchmark_name, feature_folder,  target,year, month_lag):
    """
    Perform global benchmarking based on specified parameters.

    This function performs global benchmarking by selecting the benchmarking method, filtering data, generating outcomes, and organizing the results.

    Args:
        benchmark_name (str): The benchmarking method to use. Should be one of the following:
            - "hist" for historical benchmarking.
            - "boot" for bootstrap benchmarking.
        feature_folder (pathlib.Path): The folder containing Parquet datasets.
        target (str): The name of the Parquet dataset file (e.g., 'ged_data.parquet').
        year (int): The target year for filtering data.
        month_lag (int): The number of months to lag behind the minimum 'month_id' for filtering.

    Returns:
        pandas.DataFrame: A DataFrame containing benchmarked outcomes and relevant data.

    Example:
        To perform historical benchmarking with specified parameters:
        ```
        benchmark_name = "hist"
        feature_folder = pathlib.Path('/path/to/parquet/datasets')
        target = 'ged_data.parquet'
        year = 2023
        month_lag = 2

        result_df = global_benchmark(benchmark_name, feature_folder, target, year, month_lag)
        ```

    Note:
        - `benchmark_name` should be "hist" for historical benchmarking or "boot" for bootstrap benchmarking.
        - `feature_folder` should be a pathlib.Path object pointing to the folder where the Parquet datasets are located.
        - `target` is the name of the Parquet dataset file within the specified folder.
        - `year` is used to filter data for a specific year.
        - `month_lag` determines the number of months to lag behind the minimum 'month_id' for filtering data.

    """
    unit = select_unit(target)
    df,pool = filter_units(feature_folder,target,year,unit,month_lag)
    if benchmark_name == "hist":
        df['outcome']=historical_poisson_benchmark(pool, value_column_name = 'ged_sb', num_samples=1000)

    elif benchmark_name == "boot":
        df['outcome']=historical_bootstrap_benchmark(pool, value_column_name = 'ged_sb', num_samples=1000)        
    else:
        raise ValueError('Benchmark must be "boot" or "hist"')
    df = df.explode('outcome').astype('int32')
    df['draw'] = df.groupby(['month_id', unit]).cumcount()
    df.set_index(['month_id',unit,'draw'],inplace=True)
    return df 


if __name__ == "__main__":
    main()

# python3 your_script.py --feature_folder /path/to/features --target cm --year 2018 --benchmark_name boot