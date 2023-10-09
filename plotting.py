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

def collect_plotting_data(model_a: str|os.PathLike, model_b: str|os.PathLike, actual_folder: str|os.PathLike, target: str, unit_id: int) -> pd.DataFrame:
    
    def get_info(model: str|os.PathLike, target = str) -> tuple[list[pd.DataFrame], list[os.PathLike], str, str]:
        preds = get_prediction_files(model)
        preds = [f for f in preds if f.parent.parts[-2] == target]
        eval = list(model.glob(f"eval_{target}_per_month.parquet"))
        team, model = team_and_model(model)
        
        return preds, eval, team, model

    actuals = list((actual_folder / f"{target}").glob("**/*.parquet"))
    actuals_df = pd.concat([subset_predictions(f, unit_id, "actuals") for f in actuals])

    a_preds, a_eval, a_team, a_model = get_info(model_a, target)
    b_preds, b_eval, b_team, b_model = get_info(model_b, target)

    a_df = pd.concat([subset_predictions(f, id = unit_id, model = a_model) for f in a_preds])
    b_df = pd.concat([subset_predictions(f, id = unit_id, model = b_model) for f in b_preds])
    
    df = pd.concat([comparison_df, baseline_df, actuals_df])

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
        ax.fill_between(dat.month_id, dat.p05, dat.p95, alpha = 0.2)
    ax.set_title("Case comparison: Mali 2018-2021, cm-models")
