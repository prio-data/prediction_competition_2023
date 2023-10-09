from scipy.signal import resample
from pathlib import Path

import yaml
import pyarrow.parquet as pq
import pandas as pd
import os
import argparse

def resample_predictions(parquet_file: str|os.PathLike, num_samples: int) -> pd.DataFrame:
    parquet_file = Path(parquet_file)
    target: str = parquet_file.parent.parts[-2]

    if target == "cm":
        unit = "country_id"
    elif target == "pgm":
        unit = "priogrid_gid"
    else:
        raise ValueError(f'Unable to ascertain observation unit. Should be "cm" or "pgm", getting {target}.')

    df = pq.read_table(parquet_file).to_pandas()
    df = df.reset_index()
    df = df.drop(columns = ["draw"])
    df = df.groupby(["month_id", unit]).agg(lambda x: x.tolist()).reset_index()
    df["outcome"] = df["outcome"].apply(resample, num = num_samples)

    df = df.explode("outcome")
    df["draw"] = df.groupby(["month_id", unit]).cumcount()
    df = df.reset_index()
    df["outcome"] = np.where(df["outcome"]<0, 0, df["outcome"])
    return df[["month_id", unit, "draw", "outcome"]]

def get_prediction_files(submission: str|os.PathLike) -> list[str]:
    prediction_files = list(submission.glob("**/*.parquet"))
    prediction_files = [f for f in prediction_files if not f.stem.split("_")[0] == "eval"]
    prediction_files = [f for f in prediction_files if not f.parent.parts[-1] == "eval"]
    prediction_files = [f for f in prediction_files if "__MACOSX" not in f.parts]
    return prediction_files

def build_resampled_submissions(submissions_path: str|os.PathLike, resampled_path: str|os.PathLike, num_samples: int) -> None:
    submissions_path = Path(submissions_path)
    resampled_path = Path(resampled_path)

    submissions = [submission for submission in submissions_path.iterdir() if submission.is_dir() and not submission.stem == "__MACOSX"]

    for submission in submissions:
        submission_name = submission.stem
        prediction_files = get_prediction_files(submission)

        with open(submission/"submission_details.yml") as f:
            submission_details = yaml.safe_load(f)
        
        submission_details["even_shorter_identifier"] = submission_details["even_shorter_identifier"] + "_resample"

        (resampled_path / submission_name).mkdir(parents = True, exist_ok = True)
        with open(resampled_path / submission_name / 'data.yml', 'w') as outfile:
            yaml.dump(submission_details, outfile, default_flow_style=False)

        for prediction_file in prediction_files:
            window = prediction_file.parent.parts[-1]
            target = prediction_file.parent.parts[-2]
            fname = prediction_file.name

            (resampled_path / submission_name / target / window).mkdir(parents = True, exist_ok = True)
            df = resample_predictions(prediction_file, num_samples)
            df.to_parquet(resampled_path / submission_name / target / window / fname)

def main():
    parser = argparse.ArgumentParser(description="Method for resampling submissions to the ViEWS Prediction Challenge",
                                     epilog = "Example usage: python resample_predictions.py -s ./submissions -t ./resampled_submissions -n 100")
    parser.add_argument('-s', metavar='submissions', type=str, help='path to folder with submissions complying with submission_template')
    parser.add_argument('-t', metavar='save_to', type=str, help='path to folder you want the resampled submissions')
    parser.add_argument('-n', metavar='num_samples', type=int, help='number of samples to sample', default = 100)
    args = parser.parse_args()

    build_resampled_submissions(args.s, args.t, args.n)

if __name__ == "__main__":
    main()