from pathlib import Path
import pyarrow.parquet as pq
import yaml
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib
import cartopy.crs as ccrs
import requests
from PIL import Image
import numpy as np


def prepare_geo_forecast_data(submission: Path, target: str, window: str|int, month: str|int):
    if target == "cm":
        unit = "country_id"
        shapefile_path = "shapefiles/countries.shp"
    elif target == "pgm":
        unit = "priogrid_gid"
        shapefile_path = "shapefiles/priogrid.shp"
    else:
        raise ValueError(f'Target must be "cm" or "pgm".')

    with open(submission/"submission_details.yml") as f:
        submission_details = yaml.safe_load(f)
        
    map = gpd.read_file(shapefile_path)
    if target == "pgm":
        map = map.rename(columns = {"priogrid_i": "priogrid_gid"}) # fix this in shapefile
    
    outcome_path = submission / f'{target}' / f'window=Y{window}' 
    outcome_file = outcome_path.glob('**/*.parquet')
    outcome_file = list(outcome_path.glob('**/*.parquet'))[0]
    outcome_df = pq.read_table(outcome_file).to_pandas()
    outcome_df = outcome_df.groupby(['month_id', unit])['outcome'].median()
    outcome_df = outcome_df.reset_index()

    if month not in outcome_df.month_id.unique():
        raise ValueError(f"Month {month} not found.")
    
    df = pd.merge(outcome_df, map, left_on = unit, right_on = unit).query(f'month_id == {month}')
    team = submission_details["team"]
    model = submission_details["even_shorter_identifier"]
    df = gpd.GeoDataFrame(df)
    
    return df, outcome_df, team, model


def prepare_geo_evaluation_data(submission, target, window = None):
    if target == "cm":
        unit = "country_id"
        shapefile_path = "shapefiles/countries.shp"
    elif target == "pgm":
        unit = "priogrid_gid"
        shapefile_path = "shapefiles/priogrid.shp"
    else:
        raise ValueError(f'Target must be "cm" or "pgm".')

    with open(submission/"submission_details.yml") as f:
        submission_details = yaml.safe_load(f)
        
    map = gpd.read_file(shapefile_path)
    if target == "pgm":
        map = map.rename(columns = {"priogrid_i": "priogrid_gid"}) # fix this in shapefile

    eval_file = submission / f"eval_{target}_per_unit.parquet"
    eval_df = pq.read_table(eval_file).to_pandas()
    eval_df = eval_df.reset_index()

    if window != None:
        eval_df = eval_df[eval_df["window"] == f"window=Y{window}"]

    df = pd.merge(eval_df, map, left_on = unit, right_on = unit)
    team = submission_details["team"]
    model = submission_details["even_shorter_identifier"]

    df = df.query(f'team == "{team}" and identifier == "{model}"')
    #df = df[df["team"] == submission_details["team"] and df["identifier"] == submission_details["even_shorter_identifier"]]
    df = gpd.GeoDataFrame(df)
    return df, eval_df, team, model

def choropleth_map_forecast(submission, target, window, month, cmap = "viridis", metric_ticks = None,  views_logo = True, info_box_placement = [0.66, 0.08, 0.1, 0.1]) -> None:
    if not isinstance(submission, Path):
        submission = Path(submission)
    
    pgm_map = gpd.read_file('shapefiles/priogrid.shp')
    cm_map = gpd.read_file('shapefiles/countries.shp')
        
    df, outcome_df, team, model = prepare_geo_forecast_data(submission, target = target, window = window, month = month)

    vmin, vmax = df['outcome'].min(), df['outcome'].max()
    cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.SymLogNorm(linthresh = 10, vmin=vmin, vmax=vmax, base = 10), cmap = cmap)

    sns.set_theme(style = "white")

    if target == "cm":
        crs = ccrs.EqualEarth()
        df = df.to_crs(crs.proj4_init + " +over")
    elif target == "pgm":
        crs = ccrs.PlateCarree()
        df = df.to_crs(pgm_map.crs)

    fig, ax = plt.subplots(figsize = (24, 8), subplot_kw = {"projection": crs})

    if target == "cm":
        ax.add_geometries(df["geometry"], crs = crs, facecolor = "none", edgecolor = "whitesmoke")
        ax.set_global()
    elif target == "pgm":
        cm_map.boundary.plot(ax=ax, linewidth=0.2)
        ax.set_extent([-20, 70, -40, 45])

    if views_logo:
        # Download and display the VIEWS logo image
        logo_url = "https://cdn.cloud.prio.org/images/c784369fb4ae42acb7ee882e91056d92.png?x=800&"
        response = requests.get(logo_url, stream=True)

        if response.status_code == 200:
            logo_img = Image.open(response.raw)
            if target == "cm":
                left = 0.2
            elif target == "pgm":
                left = 0.4
            logo_ax = fig.add_axes([left, 0.06, 0.1, 0.1])  # Define the position and size of the logo [left, bottom, width, height]
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')  # Turn off axis labels and ticks for the logo
        else:
            print("Failed to download the logo image")


    df.plot(ax=ax, 
            column = 'outcome',
            norm = matplotlib.colors.SymLogNorm(linthresh = 1),
            edgecolor = "#FF000000",
            cmap = cmap,
            )

    fig.colorbar(cbar, ax=ax, 
                format = matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')),
                label = 'Fatalities',
                ticks = metric_ticks)
    
    ax.gridlines(draw_labels=True, zorder = 0)
    # Title
    year, month = int(window), int(month)
    formatted_model = model.replace(team, '').strip().lstrip("_")
    formatted_team = team.replace("_", " ").strip().title()
    ax.set_title(f'Team: {formatted_team}, Model: {formatted_model}, Window: {year}, Month: {month}', fontsize=16, pad = 20, loc = "left", fontweight = "bold")


def choropleth_map_evaluation(submission, metric, target, window = None, cmap = "viridis", metric_ticks = None,  views_logo = True, info_box_placement = [0.66, 0.08, 0.1, 0.1]) -> None:
    if not isinstance(submission, Path):
        submission = Path(submission)
    
    pgm_map = gpd.read_file('shapefiles/priogrid.shp')
    cm_map = gpd.read_file('shapefiles/countries.shp')
        
    df, eval_df, team, model = prepare_geo_evaluation_data(submission, target = target, window = window)

    if metric == "crps" and target == "cm" and metric_ticks == None:
        drop_value = 1000
        metric_ticks = [0, 1, 10, 100, 500, 1000]
        
        vmin, vmax = df[metric].min(), drop_value
        cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.SymLogNorm(linthresh = 10, vmin=vmin, vmax=vmax, base = 10), cmap = cmap)

    if metric == "ign" and target == "cm" and metric_ticks == None:
        drop_value = 10
        metric_ticks = [0, 2, 4, 6, 8, 10]
        
        vmin, vmax = df[metric].min(), drop_value
        cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.Normalize(vmin=0, vmax=10, clip = True), cmap = cmap)

    if metric == "mis" and target == "cm" and metric_ticks == None:
        drop_value = 10000
        metric_ticks = [0, 10, 100, 1000, 5000, 10000]
        
        vmin, vmax = df[metric].min(), drop_value
        cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.SymLogNorm(linthresh = 10, vmin=vmin, vmax=vmax, base = 10), cmap = cmap)

    if metric == "crps" and target == "pgm" and metric_ticks == None:
        drop_value = 100
        metric_ticks = [0, 1, 10, 30, 50, 100]
        
        vmin, vmax = df[metric].min(), drop_value
        cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.SymLogNorm(linthresh = 10, vmin=vmin, vmax=vmax, base = 10), cmap = cmap)

    if metric == "ign" and target == "pgm" and metric_ticks == None:
        drop_value = 10
        metric_ticks = [0, 2, 4, 6, 8, 10]
        
        vmin, vmax = df[metric].min(), drop_value
        cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.Normalize(vmin=0, vmax=10, clip = True), cmap = cmap)
    
    if metric == "mis" and target == "pgm" and metric_ticks == None:
        drop_value = 500
        metric_ticks = [0, 10, 100, 200, 500]
        
        vmin, vmax = df[metric].min(), drop_value
        cbar = plt.cm.ScalarMappable(norm = matplotlib.colors.SymLogNorm(linthresh = 10, vmin=vmin, vmax=vmax, base = 10), cmap = cmap)


    num_countries_dropped = (df[metric]>drop_value).sum()
    df[metric] = np.where(df[metric] > drop_value, drop_value, df[metric])

    sns.set_theme(style = "white")

    if target == "cm":
        crs = ccrs.EqualEarth()
        df = df.to_crs(crs.proj4_init + " +over")
    elif target == "pgm":
        crs = ccrs.PlateCarree()
        df = df.to_crs(pgm_map.crs)

    fig, ax = plt.subplots(figsize = (24, 8), subplot_kw = {"projection": crs})
    if target == "cm":
        ax.add_geometries(df["geometry"], crs = crs, facecolor = "none", edgecolor = "whitesmoke")
        ax.set_global()
    elif target == "pgm":
        cm_map.boundary.plot(ax=ax, linewidth=0.2)
        ax.set_extent([-20, 70, -40, 45])

    if views_logo:
        # Download and display the VIEWS logo image
        logo_url = "https://cdn.cloud.prio.org/images/c784369fb4ae42acb7ee882e91056d92.png?x=800&"
        response = requests.get(logo_url, stream=True)

        if response.status_code == 200:
            logo_img = Image.open(response.raw)
            if target == "cm":
                left = 0.2
            elif target == "pgm":
                left = 0.4
            logo_ax = fig.add_axes([left, 0.06, 0.1, 0.1])  # Define the position and size of the logo [left, bottom, width, height]
            logo_ax.imshow(logo_img)
            logo_ax.axis('off')  # Turn off axis labels and ticks for the logo
        else:
            print("Failed to download the logo image")

    df.plot(ax=ax, 
            column = metric,
            norm = matplotlib.colors.SymLogNorm(linthresh = 1),
            edgecolor = "#FF000000",
            cmap = cmap)

    fig.colorbar(cbar, ax=ax, 
                format = matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')),
                label = metric.upper(),
                ticks = metric_ticks)
    
    if num_countries_dropped > 0:
        info_text = f'# countries truncated to {drop_value}: {num_countries_dropped}'
        if target == 'pgm':
            info_box_placement = [0.76, 0.08, 0.1, 0.1]
        ib_ax = fig.add_axes(info_box_placement)
        ib_ax.text(0, 0, info_text, ha = "left", fontstyle = "italic")
        ib_ax.axis('off')

    ax.gridlines(draw_labels=True, zorder = 0)
    # Title
    formatted_year = int(window)
    formatted_model = model.replace(team, '').strip().lstrip("_")
    formatted_team = team.replace("_", " ").strip().title()
    ax.set_title(f'Team: {formatted_team}, Model: {formatted_model}, Window: {formatted_year}', fontsize=16, pad = 20, loc = "left", fontweight = "bold")