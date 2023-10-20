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
    if target == "pgm":
        return "priogrid_gid"
    elif target == "cm":
        return "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm')
    
def filter_units(feature_folder,target,year,unit,month_lag):
    filter = pac.field("year") == year
    df = pq.ParquetDataset(feature_folder / target, filters=filter).read(columns=[unit, "month_id"]).to_pandas()
    
    filter = (pac.field("month_id") <= df.month_id.min() - month_lag) & (pac.field("month_id") > df.month_id.min() - (12+month_lag))
    pool = pq.ParquetDataset(feature_folder / target, filters=filter).read(columns=["ged_sb"]).to_pandas()
    return df,pool

def generate_historical_poisson(data, value_column_name, num_samples=1000):
    return [np.random.poisson(value, num_samples) for value in data[value_column_name]]

def generate_bootstrap(data, value_column_name, num_samples=1000):
    return np.random.choice(data[value_column_name], size=(data.shape[0], num_samples), replace=True).tolist()

def global_benchmark(benchmark_name, feature_folder,  target,year, month_lag):
    unit = select_unit(target)
    df,pool = filter_units(feature_folder,target,year,unit,month_lag)
    if benchmark_name == "hist":
        df['outcome']=generate_historical_poisson(pool, value_column_name = 'ged_sb', num_samples=1000)

    elif benchmark_name == "boot":
        df['outcome']=generate_bootstrap(pool, value_column_name = 'ged_sb', num_samples=1000)        
    else:
        raise ValueError('Benchmark must be "boot" or "hist"')
    df = df.explode('outcome').astype('int32')
    df['draw'] = df.groupby(['month_id', unit]).cumcount()
    df.set_index(['month_id',unit,'draw'],inplace=True)
    return df 


if __name__ == "__main__":
    main()

# python3 your_script.py --feature_folder /path/to/features --target cm --year 2018 --benchmark_name boot