from pathlib import Path
from typing import Tuple, Optional, List, Union
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


def get_shapefile_path(target: str) -> Path:
    """Get the absolute path to the shapefile based on the target type.

    Args:
        target: Either "cm" for country-level or "pgm" for priogrid-level data

    Returns:
        Path object pointing to the shapefile
    """
    script_dir = Path(__file__).parent
    if target == "cm":
        return script_dir / "shapefiles" / "countries.shp"
    elif target == "pgm":
        return script_dir / "shapefiles" / "priogrid.shp"
    else:
        raise ValueError(f'Target must be "cm" or "pgm".')


def get_unit_column(target: str) -> str:
    return "country_id" if target == "cm" else "priogrid_gid"


def load_submission_details(submission: Path) -> Tuple[str, str]:
    with open(submission / "submission_details.yml") as f:
        details = yaml.safe_load(f)
    return details["team"], details["even_shorter_identifier"]


def setup_map_plot(target: str) -> Tuple[plt.Figure, plt.Axes, ccrs.Projection]:
    if target == "cm":
        crs = ccrs.EqualEarth()
    else:
        crs = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(figsize=(24, 8), subplot_kw={"projection": crs})
    return fig, ax, crs


def add_views_logo(fig: plt.Figure, target: str) -> None:
    logo_url = "https://cdn.cloud.prio.org/images/c784369fb4ae42acb7ee882e91056d92.png?x=800&"
    response = requests.get(logo_url, stream=True)
    
    if response.status_code == 200:
        logo_img = Image.open(response.raw)
        left = 0.2 if target == "cm" else 0.4
        logo_ax = fig.add_axes([left, 0.06, 0.1, 0.1])
        logo_ax.imshow(logo_img)
        logo_ax.axis("off")


def format_title(team: str, model: str, window: str, month: Optional[str] = None) -> str:
    formatted_model = model.replace(team, "").strip().lstrip("_")
    formatted_team = team.replace("_", " ").strip().title()
    formatted_year = int(window)
    
    if month:
        return f"Team: {formatted_team}, Model: {formatted_model}, Window: {formatted_year}, Month: {month}"
    return f"Team: {formatted_team}, Model: {formatted_model}, Window: {formatted_year}"


def prepare_geo_forecast_data(submission: Path, target: str, window: str | int, month: str | int) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, str, str]:
    unit = get_unit_column(target)
    shapefile_path = get_shapefile_path(target)
    team, model = load_submission_details(submission)
    
    map = gpd.read_file(shapefile_path)
    if target == "pgm":
        map = map.rename(columns={"priogrid_i": "priogrid_gid"})
    
    outcome_path = submission / f"{target}" / f"window=Y{window}"
    outcome_file = list(outcome_path.glob("**/*.parquet"))[0]
    
    outcome_df = pq.read_table(outcome_file).to_pandas()
    outcome_df = outcome_df.groupby(["month_id", unit])["outcome"].median().reset_index()
    
    if month not in outcome_df.month_id.unique():
        raise ValueError(f"Month {month} not found.")
    
    df = pd.merge(outcome_df, map, left_on=unit, right_on=unit).query(f"month_id == {month}")
    return gpd.GeoDataFrame(df), outcome_df, team, model


def prepare_geo_evaluation_data(submission: Path, target: str, window: str, metric: str) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, str, str]:
    unit = get_unit_column(target)
    shapefile_path = get_shapefile_path(target)
    team, model = load_submission_details(submission)
    
    map = gpd.read_file(shapefile_path)
    if target == "pgm":
        map = map.rename(columns={"priogrid_i": "priogrid_gid"})
    
    eval_file = submission / "eval" / f"{target}" / f"window=Y{window}" / f"metric={metric}" / f"{metric}.parquet"
    eval_df = pq.read_table(eval_file).to_pandas().reset_index()
    
    df = pd.merge(eval_df, map, left_on=unit, right_on=unit)
    return gpd.GeoDataFrame(df), eval_df, team, model


def get_metric_ticks(metric: str, target: str) -> Tuple[List[int], int]:
    if metric == "crps":
        if target == "cm":
            return [0, 1, 10, 100, 500, 1000], 1000
        return [0, 1, 10, 30, 50, 100], 100
    elif metric == "ign":
        return [0, 2, 4, 6, 8, 10], 10
    elif metric == "mis":
        if target == "cm":
            return [0, 10, 100, 1000, 5000, 10000], 10000
        return [0, 10, 100, 200, 500], 500
    return None, None


def choropleth_map_forecast(
    submission: Path | str,
    target: str,
    window: str,
    month: str | int,
    cmap: str = "viridis",
    metric_ticks: Optional[List[int]] = None,
    views_logo: bool = True,
    info_box_placement: List[float] = [0.66, 0.08, 0.1, 0.1],
) -> None:
    submission = Path(submission)
    if window.startswith("Y"):
        window = int(window.replace("Y", ""))
    
    df, outcome_df, team, model = prepare_geo_forecast_data(submission, target, window, month)
    fig, ax, crs = setup_map_plot(target)
    
    vmin, vmax = df["outcome"].min(), df["outcome"].max()
    cbar = plt.cm.ScalarMappable(
        norm=matplotlib.colors.SymLogNorm(linthresh=10, vmin=vmin, vmax=vmax, base=10),
        cmap=cmap,
    )
    
    sns.set_theme(style="white")
    
    if target == "cm":
        df = df.to_crs(crs.proj4_init + " +over")
        ax.add_geometries(df["geometry"], crs=crs, facecolor="none", edgecolor="whitesmoke")
        ax.set_global()
    else:
        cm_map = gpd.read_file(get_shapefile_path("cm"))
        df = df.to_crs(crs)
        cm_map.boundary.plot(ax=ax, linewidth=0.2)
        ax.set_extent([-20, 70, -40, 45])
    
    if views_logo:
        add_views_logo(fig, target)
    
    df.plot(
        ax=ax,
        column="outcome",
        norm=matplotlib.colors.SymLogNorm(linthresh=1),
        edgecolor="#FF000000",
        cmap=cmap,
    )
    
    fig.colorbar(
        cbar,
        ax=ax,
        format=matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")),
        label="Fatalities",
        ticks=metric_ticks,
    )
    
    ax.gridlines(draw_labels=True, zorder=0)
    ax.set_title(
        format_title(team, model, window, month),
        fontsize=16,
        pad=20,
        loc="left",
        fontweight="bold",
    )


def choropleth_map_evaluation(
    submission: Path | str,
    metric: str,
    target: str,
    window: str,
    cmap: str = "viridis",
    metric_ticks: Optional[List[int]] = None,
    views_logo: bool = True,
    info_box_placement: List[float] = [0.66, 0.08, 0.1, 0.1],
) -> None:
    submission = Path(submission)
    if window.startswith("Y"):
        window = int(window.replace("Y", ""))
    
    df, eval_df, team, model = prepare_geo_evaluation_data(submission, target, window, metric)
    fig, ax, crs = setup_map_plot(target)
    
    if metric_ticks is None:
        metric_ticks, drop_value = get_metric_ticks(metric, target)
        vmin, vmax = df["value"].min(), drop_value
        cbar = plt.cm.ScalarMappable(
            norm=matplotlib.colors.SymLogNorm(linthresh=10, vmin=vmin, vmax=vmax, base=10),
            cmap=cmap,
        )
    
    num_countries_dropped = (df["value"] > drop_value).sum()
    df["value"] = np.where(df["value"] > drop_value, drop_value, df["value"])
    
    sns.set_theme(style="white")
    
    if target == "cm":
        df = df.to_crs(crs.proj4_init + " +over")
        ax.add_geometries(df["geometry"], crs=crs, facecolor="none", edgecolor="whitesmoke")
        ax.set_global()
    else:
        cm_map = gpd.read_file(get_shapefile_path("cm"))
        df = df.to_crs(crs)
        cm_map.boundary.plot(ax=ax, linewidth=0.2)
        ax.set_extent([-20, 70, -40, 45])
    
    if views_logo:
        add_views_logo(fig, target)
    
    df.plot(
        ax=ax,
        column="value",
        norm=matplotlib.colors.SymLogNorm(linthresh=1),
        edgecolor="#FF000000",
        cmap=cmap,
    )
    
    fig.colorbar(
        cbar,
        ax=ax,
        format=matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ",")),
        label=metric.upper(),
        ticks=metric_ticks,
    )
    
    if num_countries_dropped > 0:
        info_text = f"# countries truncated to {drop_value}: {num_countries_dropped}"
        if target == "pgm":
            info_box_placement = [0.76, 0.08, 0.1, 0.1]
        ib_ax = fig.add_axes(info_box_placement)
        ib_ax.text(0, 0, info_text, ha="left", fontstyle="italic")
        ib_ax.axis("off")
    
    ax.gridlines(draw_labels=True, zorder=0)
    ax.set_title(
        format_title(team, model, window),
        fontsize=16,
        pad=20,
        loc="left",
        fontweight="bold",
    )
