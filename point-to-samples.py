from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import os
import argparse

def calc_poisson(df: pd.DataFrame, test: bool) -> pd.DataFrame:
    np.random.seed(574309)
    df["outcome"] = df["outcome"].apply(np.random.poisson, size = 1000)

    if test:
        df["mean"] = df["outcome"].apply(np.mean)
        df["var"] = df["outcome"].apply(np.var)
        df["sample_mean_deviance"] =  df["mean"] - df["prediction"]
        df["sample_var_deviance"] =  df["var"] - df["prediction"]
        print(df.describe())
    return df

def samples_from_point_predictions(parquet_file: str | os.PathLike, type: str, test: bool = False) -> None:
    df: pd.DataFrame = pq.read_table(parquet_file).to_pandas()
    df["outcome"] = np.where(df["outcome"] < 0, 0, df["outcome"]) # cannot be negative, as np.random.poisson will complain
    
    match type:
        case "poisson":
            df = calc_poisson(df, test = test)
        case "exact":
            df = df.loc[df.index.repeat(1000)]
        case "zero":
            df["outcome"] = 0
            df = df.loc[df.index.repeat(1000)]
        case _:
            raise ValueError(f'Method must be one of "poisson", "exact" or "zero", not {type}')        

    window: str = parquet_file.parent.parts[-1]
    target: str = parquet_file.parent.parts[-2]
    name: str = parquet_file.parent.parts[-3]
    
    if target == "cm":
        unit = "country_id"
    elif target == "pgm":
        unit = "priogrid_gid"
    else:
        raise ValueError(f'Unable to ascertain observation unit. Should be "cm" or "pgm", getting {target}.')
    

    df = df.explode("outcome")
    df["draw"] = df.groupby(["month_id", unit]).cumcount()
    df = df.reset_index()
    df = df[["month_id", unit, "draw", "outcome"]]
    df.set_index(['month_id', unit, "draw"], inplace = True)
    

    out_file_name: str = parquet_file.stem + f'_{type}-samples.parquet'

    write_path = parquet_file.parent.parent.parent.parent / "results" / f'{name}-{type}-samples' / target / window
    write_path.mkdir(exist_ok=True, parents=True)
    df.to_parquet(write_path / out_file_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Method for making samples from point-predictions in the ViEWS Prediction Challenge",
                                    epilog = "Example usage: python point-to-samples.py -s path/to/submission/with/point/data -m zero")
    parser.add_argument('-s', metavar='submission', type=str, help='path to a submission_template folder')
    parser.add_argument('-m', metavar='method', type=str, help='one of "poisson", "exact", or "zero"', default = "poisson")
    args = parser.parse_args()

    submission = Path(args.s)
    prediction_files = list(submission.glob("**/*.parquet"))
    prediction_files = [f for f in prediction_files if not f.stem.split("_")[0] == "eval"]
    prediction_files = [f for f in prediction_files if not f.parent.parts[-1] == "eval"]
    prediction_files = [f for f in prediction_files if "__MACOSX" not in f.parts]

    [samples_from_point_predictions(f, type = args.m) for f in prediction_files]

if __name__ == "__main__":
    main()