from pathlib import Path
import pyarrow.parquet as pq
import os
import yaml
import pandas as pd
from test_compliance import get_prediction_files, get_unit
from collect_performance import get_global_performance
import seaborn as sns
import seaborn.objects as so
import numpy as np
import matplotlib.pyplot as plt

def team_and_model(submission: str|os.PathLike) -> tuple[str, str]:
    with open(submission/"submission_details.yml") as f:
        submission_details = yaml.safe_load(f)
        identifier = submission_details["even_shorter_identifier"]
        team = submission_details["team"]
    return (team, identifier)

def subset_predictions(pred_file, id, model):
    window = pred_file.parent.parts[-1]
    window = int(window.split("_")[-1])
    unit = get_unit(pred_file)

    df = pq.read_table(pred_file).to_pandas()
    if df.index.names != [None]:
        df = df.reset_index()
    df = df[df[unit] == id]
    if "draw" in df.columns:
        df = df.groupby(["month_id", unit]).agg({"outcome": lambda x: x.tolist()}).reset_index()
    
    df["window"] = window
    df["model"] = model
    return df

def get_model_info(model: str|os.PathLike, target = str) -> tuple[list[pd.DataFrame], list[os.PathLike], str, str]:
        preds = get_prediction_files(model)
        preds = [f for f in preds if f.parent.parts[-2] == target]
        eval = list(model.glob(f"eval_{target}_per_month.parquet"))
        team, model = team_and_model(model)
        
        return preds, eval, team, model

def collect_plotting_data(models: list[str], actual_folder: str|os.PathLike, target: str, unit_id: int, max_fatalities:int = None) -> pd.DataFrame:
    
    actuals = list((actual_folder / f"{target}").glob("**/*.parquet"))
    actuals_df = pd.concat([subset_predictions(f, unit_id, "actuals") for f in actuals])

    data_list = [actuals_df]
    for model in models:
        preds, eval, team, model_name = get_model_info(model, target)
        sdf = pd.concat([subset_predictions(f, id = unit_id, model = f'{team}: {model_name}') for f in preds])
        data_list.append(sdf)

    df = pd.concat(data_list)

    if max_fatalities != None:
        df["outcome"] = df["outcome"].apply(lambda x: np.where(np.array(x) > max_fatalities, max_fatalities, np.array(x)))

    df["p50"] = df["outcome"].apply(np.quantile, q = 0.5)
    df["p95"] = df["outcome"].apply(np.quantile, q = 0.95)
    df["p05"] = df["outcome"].apply(np.quantile, q = 0.05)
    df["p75"] = df["outcome"].apply(np.quantile, q = 0.75)
    df["p25"] = df["outcome"].apply(np.quantile, q = 0.25)

    mask = df["model"] == "actuals"
    df.loc[mask, ["p05", "p25", "p75", "p95"]] = np.nan

    df = df.drop(columns="outcome")
    return df

def ribbon_plot(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(1)

    df = df.sort_values("month_id")
    for l, dat in df.groupby("model"):
        dat.plot(x = "month_id", y = "p50", label = l, ax = ax)
        ax.fill_between(dat.month_id, dat.p25, dat.p75, alpha = 0.2)
    ax.set_title(title)
