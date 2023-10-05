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

def get_global_performance(eval_file: str|os.PathLike, save_to: str|os.PathLike = "", by_window: bool = False):

    target = eval_file.stem.split("_")[1]

    df: pd.DataFrame = pq.read_table(eval_file).to_pandas()
    df = df.reset_index()
    agg_level = df.columns[3]
    df = df.drop(columns = ["target", agg_level])

    if by_window:
        df = df.set_index(["team", "identifier", "window"])
        df = df.groupby(level=[0,1,2]).mean()
        df = df.sort_values("crps")
        df.index = df.index.rename(["Team", "Model", "Window"])
    else:
        df = df.set_index(["team", "identifier"])
        test_windows = df["window"].groupby(level=[0,1]).nunique()
        df = df.drop(index=test_windows[test_windows !=max(test_windows)].index, 
                columns = "window")
        df = df.groupby(level=[0,1]).mean()
        df = df.sort_values("crps")
        df.index = df.index.rename(["Team", "Model"])

    df.columns = df.columns.str.upper()

    if save_to == "":
        return df
    else:
        if by_window:
            file_stem = f"results_by_window_{target}_{agg_level}"
        else:
            file_stem = f"global_results_{target}_{agg_level}"

        df = df.style.format(decimal=',', thousands='.', precision=3)
        df.to_latex(save_to / f'{file_stem}.tex')
        df.to_markdown(save_to / f'{file_stem}.md')

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Method for collating evaluations from all submissions in the ViEWS Prediction Challenge",
                                     epilog = "Example usage: python collect_performance.py -s ./submissions")
    parser.add_argument('-s', metavar='submissions', type=str, help='path to folder with submissions complying with submission_template')
    parser.add_argument('-t', metavar='tables_folder', type=str, help='path to folder to save result tables in latex', default=None)
    args = parser.parse_args()

    collect_evaluations(args.s)

    if args.t is not None and Path(args.t).exists():
        eval_files = list(Path(args.s).glob("eval*.parquet"))
        [get_global_performance(f, save_to = Path(args.t)) for f in eval_files]
        [get_global_performance(f, save_to = Path(args.t), by_window=True) for f in eval_files]