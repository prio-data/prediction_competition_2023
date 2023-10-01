from pathlib import Path
import pyarrow.parquet as pq
import os
import yaml
import pandas as pd
import argparse

def setup_eval_df(eval_file: str|os.PathLike, team: str, identifier: str, target: str, window: str) -> pd.DataFrame:
    df: pd.DataFrame = pq.read_table(eval_file).to_pandas()
    df["window"] = window
    df["target"] = target
    df["team"] = team
    df["identifier"] = identifier
    unit: str = df.index.name
    df.reset_index(inplace = True)
    df.set_index(["team", "identifier", "target", unit, "window"], inplace = True)
    return df

def concat_eval(submission: str|os.PathLike, target: str, aggregation: str, metric: str) -> pd.DataFrame:
    eval_data: list[str|os.PathLike] = list(submission.glob(f'**/{metric}_{aggregation}.parquet'))
    eval_data: list[str|os.PathLike] = [f for f in eval_data if f.parts[-4] == target]

    with open(submission/"submission_details.yml") as f:
        submission_details = yaml.safe_load(f)
    identifier = submission_details["even_shorter_identifier"]
    team = submission_details["team"]
    
    if len(eval_data) == 0:
        return print(f'No files to collect in submission {submission} for target {target} and aggregation {aggregation}.')
    
    dfs: list = []
    for f in eval_data:
        window = f.parts[-3]
        target2 = f.parts[-4]
        assert target == target2

        sdf: pd.DataFrame = setup_eval_df(f, team, identifier, target, window)
        
        dfs.append(sdf)
    
    return pd.concat(dfs)

def merge_eval(submission: str|os.PathLike, target, aggregation) -> None:
    metrics = ["crps", "ign", "mis"]

    dfs: list = []
    
    for metric in metrics:
        res = concat_eval(submission, target, aggregation, metric)
        if isinstance(res, pd.DataFrame):
            dfs.append(res)
    try:
        df = pd.concat(dfs, axis = 1) # Columnar concatenation
        fpath: str|os.PathLike = submission / f'eval_{target}_{aggregation}.parquet'
        df.to_parquet(fpath)
    except ValueError:
        pass

def collect_evaluations(submissions_path: str|os.PathLike) -> None:
    submission_path = Path(submissions_path)
    submissions = [submission for submission in submission_path.iterdir() if submission.is_dir() and not submission.stem == "__MACOSX"]

    targets = ["cm", "pgm"]
    aggregations = ["per_unit", "per_month"]

    for submission in submissions:        
        for target in targets:
            for aggregation in aggregations:
                merge_eval(submission, target, aggregation)

    for target in targets:
        for aggregation in aggregations:
            eval_data = list(submission_path.glob(f'*/eval_{target}_{aggregation}.parquet'))
            dfs = [pq.read_table(f).to_pandas() for f in eval_data]
            df = pd.concat(dfs)
            fpath = submission_path / f'eval_{target}_{aggregation}.parquet'
            df.to_parquet(fpath)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Method for collating evaluations from all submissions in the ViEWS Prediction Challenge",
                                     epilog = "Example usage: python collect_performance.py -s ./submissions")
    parser.add_argument('-s', metavar='submissions', type=str, help='path to folder with submissions complying with submission_template')
    args = parser.parse_args()

    collect_evaluations(args.s)