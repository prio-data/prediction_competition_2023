import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import pac

def global_bootstrap_benchmark(feature_folder, target, year) -> pd.DataFrame:
    """
    Draws a random sample with replacement from the complete global pool of fatalities data from the year before.
    """
    
    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError('Target must be "pgm" or "cm.')

    filter = pac.field("year") == year - 1
    pool = pq.ParquetDataset(feature_folder / target, filters=filter).read(columns=["ged_sb"]).to_pandas()

    filter = pac.field("year") == year
    df = pq.ParquetDataset(feature_folder / target, filters=filter).read(columns=[unit, "month_id"]).to_pandas()

    df["outcome"] = np.random.choice(pool["ged_sb"], size=(df.shape[0], 1000), replace=True).tolist()
    df = df.explode('outcome').astype('int32')
    df['draw'] = df.groupby(['month_id', unit]).cumcount()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Bootstrap Benchmark Script")
    parser.add_argument("--feature_folder", type=str, help="Path to the feature folder")
    parser.add_argument("--target", type=str, choices=["pgm", "cm"], help="Target type (pgm or cm)")
    parser.add_argument("--year", type=int, help="Year")

    args = parser.parse_args()

    feature_folder = Path(args.feature_folder)
    target = args.target
    year = args.year

    result = global_bootstrap_benchmark(feature_folder, target, year)

    # Do something with the result, e.g., print or save it.
    print(result)
