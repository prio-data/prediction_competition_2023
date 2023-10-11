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
import geopandas as gpd
import matplotlib
import cartopy.crs as ccrs
import requests
import mapclassify
from PIL import Image
import numpy as np

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

def prepare_geo_data(submission, target, shapefile, window = None:
    if target == "cm":
        unit = "country_id"
    elif target == "pgm":
        unit = "priogrid_gid"
    else:
        raise ValueError(f'Target must be "cm" or "pgm".')
    
    with open(submission/"submission_details.yml") as f:
        submission_details = yaml.safe_load(f)
        
    map = gpd.read_file(shapefile)
    if target == "pgm":
         map = map.rename(columns = {"priogrid_i": "priogrid_gid"}) # fix this in shapefile

    eval_file = submission / f"eval_{target}_per_unit.parquet"
    eval_df = pq.read_table(eval_file).to_pandas()
    eval_df = eval_df.reset_index()

    if window != None:
        eval_df = eval_df[eval_df["window"] == window]
    
    df = pd.merge(eval_df, map, left_on = unit, right_on = unit)
    team = submission_details["team"]
    model = submission_details["even_shorter_identifier"]

    df = df.query(f'team == "{team}" and identifier == "{model}"')
    #df = df[df["team"] == submission_details["team"] and df["identifier"] == submission_details["even_shorter_identifier"]]
    df = gpd.GeoDataFrame(df)
    return df, eval_df, team, model

def choropleth_map(submission, metric, target, window = None, cmap = "viridis", metric_ticks = None, crs = ccrs.EqualEarth(), views_logo = True) -> None:
    df, eval_df, team, model = prepare_geo_data(submission, target = target, window = window)

    if metric == "crps" and target == "cm" and metric_ticks == None:
        drop_value = 1000
        metric_ticks = [0, 1, 10, 100, 500, 1000]
        
        vmin, vmax = df[metric].min(), df[metric].max()
        cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.SymLogNorm(linthresh = 10, vmin=vmin, vmax=vmax, base = 10), cmap = cmap)

    if metric == "ign" and target == "cm" and metric_ticks == None:
        drop_value = 4
        metric_ticks = [0, 1, 2, 3, 4]
        
        vmin, vmax = df[metric].min(), df[metric].max()
        cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.Normalize(vmin=0, vmax=4, clip = True), cmap = cmap)
    
    if metric == "crps" and target == "pgm" and metric_ticks == None:
        drop_value = 1000
        metric_ticks = [0, 1, 10, 100, 500, 1000]
        
        vmin, vmax = df[metric].min(), df[metric].max()
        cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.SymLogNorm(linthresh = 10, vmin=vmin, vmax=vmax, base = 10), cmap = cmap)

    if metric == "ign" and target == "pgm" and metric_ticks == None:
        drop_value = 4
        metric_ticks = [0, 1, 2, 3, 4]
        
        vmin, vmax = df[metric].min(), df[metric].max()
        cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.Normalize(vmin=0, vmax=4, clip = True), cmap = cmap)


    drop_percentage_due_to_high_value = (df[metric]>drop_value).mean()
    df[metric] = np.where(df[metric] > drop_value, drop_value, df[metric])

    sns.set_theme(style = "white")

    df = df.to_crs(crs.proj4_init + " +over")
    
    fig, ax = plt.subplots(figsize = (24, 8), subplot_kw = {"projection": crs})
    ax.gridlines(draw_labels=True, zorder = 0)
    ax.add_geometries(df["geometry"], crs = crs, facecolor = "none", edgecolor = "whitesmoke")
    ax.set_global()

    if views_logo:
        # Download and display the VIEWS logo image
        logo_url = "https://cdn.cloud.prio.org/images/c784369fb4ae42acb7ee882e91056d92.png?x=800&"
        response = requests.get(logo_url, stream=True)

        if response.status_code == 200:
            logo_img = Image.open(response.raw)
            logo_ax = fig.add_axes([0.2, 0.06, 0.1, 0.1])  # Define the position and size of the logo [left, bottom, width, height]
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')  # Turn off axis labels and ticks for the logo
        else:
            print("Failed to download the logo image")


    df.plot(ax=ax, 
            column = metric,
            norm = matplotlib.colors.SymLogNorm(linthresh = 1),
            cmap = cmap)
    
    fig.colorbar(cbar, ax=ax, 
                format = matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')),
                label = metric.upper(),
                ticks = metric_ticks)

    # Title
    formatted_year = int(window.split("_")[-1])
    formatted_model = model.replace(team, '').replace("_", " ").strip().title()
    formatted_team = team.replace("_", " ").strip().title()
    ax.set_title(f'{formatted_team}, {formatted_model}, {formatted_year}. Share of values constrained to {drop_value}: {drop_percentage_due_to_high_value:.0%} ', fontsize=16, pad = 20)