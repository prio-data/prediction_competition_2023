import argparse
import pandas as pd
import pyarrow.parquet as pq
from zipfile import ZipFile
import numpy as np
#from pathlib import Path

# mamba install -c conda-forge xskillscore
import xarray as xr
import xskillscore as xs

from IgnoranceScore import ensemble_ignorance_score_xskillscore
from IntervalScore import mean_interval_score_xskillscore

from typing import List, Union
Dim = Union[List[str], str]


def load_data(observed_path: str, forecasts_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads in parquet files of observed/actual/test and prediction/forecast data and returns pandas dataframes.
    Parameters
    ----------
    observed_path : str
        Path to the parquet-file with observed/actual/test data.
    forecasts_path : str
        Path to the parquet-file with prediction/forecast data.
    
    Returns
    -------
    observed, predictions : tuple (pandas.DataFrame, pandas.DataFrame)
    """
        
    predictions = pq.read_table(forecasts_path)
    predictions = predictions.to_pandas()
    observed = pq.read_table(observed_path)
    observed = observed.to_pandas()
    return observed, predictions

def structure_data(observed: pd.DataFrame, predictions: pd.DataFrame, draw_column_name: str = "draw", data_column_name: str = "outcome") -> tuple[xr.DataArray, xr.DataArray]:
    """Structures data to the ViEWS 2023 Prediction Competition for use in calculate_metrics() function.
    Parameters
    ----------
    observed : pandas.DataFrame
        Observations/actual/test data. Must be a Pandas dataframe with 'month_id', 'country_id' (or 'priogrid_gid'), and a column with the observed outcomes named 'data_column_name'. Gives error if more columns are present.
    predictions : pandas.DataFrame
        Predictions/forecasted data. Must be a Pandas dataframe with 'month_id', 'country_id' (or 'priogrid_gid'), a column indicating which sample/draw it is named 'draw_column_name' and a column with the predicted outcomes named 'data_column_name'. Gives error if more columns are present, or if there are missing data vis-a-vis the observed.
    draw_column_name : str
        The name of the column indicating forecast samples/draws in the predictions data.
    data_column_name : str
        The name of the column with outcomes. Must be the same in both the observed/test and predictions data.
    
    Returns
    -------
    observed, predictions : tuple (xarray.DataArray, xarray.DataArray)
    """

    # The samples must be named "member" and the outcome variable needs to be named the same in xs.crps_ensemble()
    
    predictions = predictions.reset_index()
    observed = observed.reset_index()

    if "index" in observed.columns:
        observed = observed.drop(columns = ["index"])
    if "index" in predictions.columns:
        predictions = predictions.drop(columns = ["index"])

    assert "month_id" in observed.columns,  f"'month_id' column not found in observed data. Columns in data: {observed.columns}."
    assert "month_id" in predictions.columns,  f"'month_id' column not found in predictions data. Columns in data: {predictions.columns}."

    assert observed.columns.isin(['country_id','priogrid_gid']).any(),  f"'country_id'/'priogrid_gid' column not found in observed data."
    assert predictions.columns.isin(['country_id','priogrid_gid']).any(),  f"'country_id'/'priogrid_gid' column not found in predictions data."

    assert draw_column_name in predictions.columns, f"{draw_column_name} not in predictions data. Columns in data: {predictions.columns}."

    assert data_column_name in predictions.columns, f"{data_column_name} not in predictions data. Columns in data: {predictions.columns}."
    assert data_column_name in observed.columns, f"{data_column_name} not in observed data. Columns in data: {observed.columns}."


    assert len(observed.columns) == 3, f"Observed data should only be three variables: 'month_id', 'country_id' (or 'priogrid_gid'), and {data_column_name}. Columns in data: {observed.columns}."
    assert len(predictions.columns) == 4, f"Predictions data should only be four variables: 'month_id', 'country_id' (or 'priogrid_gid'), {draw_column_name}, and {data_column_name}. Columns in data: {predictions.columns}."

    onmonths = len(observed["month_id"].unique())
    pnmonths = len(observed["month_id"].unique())

    if "priogrid_gid" in predictions.columns:
        onunits = len(observed["priogrid_gid"].unique())
        pnunits = len(observed["priogrid_gid"].unique())
    elif "country_id" in predictions.columns:
        onunits = len(observed["country_id"].unique())
        pnunits = len(observed["country_id"].unique())
    else:
        TypeError("priogrid_gid or country_id must be an identifier")
    
    pnmembers = len(predictions[draw_column_name].unique())

    assert len(predictions.index) == pnmonths*pnunits*pnmembers, f"Predictions data is not a balanced dataset with nobs: {len(predictions.index)} != months: {pnmonths} * units: {pnunits} * {draw_column_name}: {pnmembers}."
    assert len(observed.index) == onmonths*onunits, f"Observed data is not a balanced dataset with nobs: {len(observed.index)} != months: {onmonths} * units: {onunits}."
    assert pnmonths == onmonths, f'Months in prediction dataset ({pnmonths}) is not the same as in observed dataset ({onmonths}).'
    assert pnunits == onunits, f'Units in prediction dataset ({pnunits}) is not the same as in observed dataset ({onunits}).'

    # To simplify internal affairs:
    predictions = predictions.rename(columns = {draw_column_name: "member", data_column_name: "outcome"})
    observed = observed.rename(columns = {data_column_name: "outcome"})

    # Set up multi-index to easily convert to xarray
    if "priogrid_gid" in predictions.columns:
        predictions = predictions.set_index(['month_id', 'priogrid_gid', 'member'])
        observed = observed.set_index(['month_id', 'priogrid_gid'])
    elif "country_id" in predictions.columns:
        predictions = predictions.set_index(['month_id', 'country_id', 'member'])
        observed = observed.set_index(['month_id', 'country_id'])
    else:
        TypeError("priogrid_gid or country_id must be an identifier")
    
    # Convert to xarray
    xpred = predictions.to_xarray()
    xobserved = observed.to_xarray()

    odim = dict(xobserved.dims)
    pdim = dict(xpred.dims)
    if "member" in pdim:
        del pdim["member"]
    assert odim == pdim, f"observed and predictions must have matching shapes or matching shapes except the '{draw_column_name}' dimension"

    return xobserved, xpred

def calculate_metrics(observed: xr.DataArray, predictions: xr.DataArray, metric: str, aggregate_over: Dim, **kwargs) -> pd.DataFrame:
    """Calculates evaluation metrics for the ViEWS 2023 Prediction Competition
    Parameters
    ----------
    observations : xarray.DataArray
        Observations/actual data. Use structure_data to cast pandas data to this form.
    forecasts : xarray.DataArray
        Forecasts with required member dimension ``member_dim``. Use structure_data to cast pandas data to this form.
    metric : str
        One of 'crps' (Ranked Probability Score), 'ign' (Ignorance Score), and 'mis' (Interval Score). 
    aggregate_over : str or list of str, optional
        Dimensions over which to compute mean after computing the evaluation metrics. E.g., 'month_id' calculates mean scores per unit, while ['month_id', 'country_id'] returns the global average.
    
    Additional arguments:
    
    If metric is 'ign':
    bins : int or monotonic list of scalars. If int, it defines the number of equal-width bins in the range low_bin - high_bin.
    low_bin : lower range of bins if bins is an int.
    high_bin : upper range of bins if bins is an int.

    If metric is 'mis':
    prediction_interval_level : The size of the prediction interval. Must be a number in the range (0, 1).

    Returns
    -------
    xarray.Dataset or xarray.DataArray
    """

    assert metric in ['crps', 'ign', "mis"], f'Metric: "{metric}" must be "crps", "ign", or "mis".'

    ign_args = ['prob_type', 'ign_max', 'round_values', 'axis', 'bins', 'low_bin', 'high_bin']
    ign_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in ign_args}

    interval_args = ['prediction_interval_level']
    interval_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in interval_args}

    # Calculate average crps for each step (so across the other dimensions)
    if metric == "crps":
        ensemble = xs.crps_ensemble(observed, predictions, dim=aggregate_over)
    elif metric == "ign":
        ensemble = ensemble_ignorance_score_xskillscore(observed, predictions, dim=aggregate_over, **ign_dict)
    elif metric == "mis":
        ensemble = mean_interval_score_xskillscore(observed, predictions, dim=aggregate_over, **interval_dict)
    else:
        TypeError(f'Metric: "{metric}" must be "crps", "ign", or "mis".')

    if not ensemble.dims: # dicts return False if empty, dims is empty if only one value.
        metrics = pd.DataFrame(ensemble.to_array().to_numpy(), columns = ["outcome"])
    else:
        metrics = ensemble.to_dataframe()
    metrics = metrics.rename(columns = {"outcome": metric})
    return metrics


def write_metrics_to_file(metrics: pd.DataFrame, filepath: str) -> None:
    metrics.to_csv(filepath)
    return None


def main():
    parser = argparse.ArgumentParser(description="This calculates metrics for the ViEWS 2023 Forecast Competition",
                                     epilog = "Example usage: python CompetitionEvaluation.py -o ./data/bm_cm_historical_values_2018.parquet -p ./data/bm_cm_ensemble_2018.parquet -m crps -ag month_id country_id")
    parser.add_argument('-o', metavar='observed', type=str, help='path to csv-file where the observed outcomes are recorded')
    parser.add_argument('-p', metavar='predictions', type=str, help='path to parquet file where the predictions are recorded in long format')
    parser.add_argument('-m', metavar='metric', type=str, help='metric to compute: "crps" or "ign"')
    parser.add_argument('-ag', metavar='aggregate_over', nargs = "+", type=str, help='Dimensions to aggregate over. Can be a list of several separated with whitespace. E.g., "-ag month_id country_id"', default = None)
    parser.add_argument('-f', metavar='file', type=str, help='(Optional) path to csv-file where you want metrics to be stored')
    parser.add_argument('-sc', metavar='sample-column-name', type=str, help='(Optional) name of column for the unique samples', default = "draw")
    parser.add_argument('-dc', metavar='data-column-name', type=str, help='(Optional) name of column with data, must be same in both observed and predictions data', default = "outcome")
    #parser.add_argument('-ipt', metavar = 'probability-type', type = int, help='One of 0-5, implements how probabilities are calculated. 3 is exact (elem_count / total).', default = 3)
    #parser.add_argument('-imx', metavar = 'max-ign', type = float, help='Set a max ignorance score. None also allowed.', default = None)
    parser.add_argument('-ib', metavar = 'ign-bins', nargs = "+", type = float, help='Set a binning scheme for the ignorance score. List or integer (nbins). E.g., "--ib 0 0.5 1 5 10 100 1000". None also allowed.', default = None)
    parser.add_argument('-ibl', metavar = 'max-ign', type = int, help='Set a min bin value when binning is an integer.', default = 0)
    parser.add_argument('-ibh', metavar = 'max-ign', type = int, help='Set a max bin value when binning is an integer.', default = 1000)
    parser.add_argument('-pil', metavar = 'prediction-interval-level', type = float, help='Set prediction interval level for the interval score', default = 0.95)
    

    args = parser.parse_args()

    observed, predictions = load_data(args.o, args.p)
    observed, predictions = structure_data(observed, predictions, draw_column_name=args.sc, data_column_name = args.dc)
    metrics = calculate_metrics(observed, predictions, metric = args.m, aggregate_over=args.ag, bins = args.ib, low_bin = args.ibl, high_bin = args.ibh, prediction_interval_level = args.pil)
    if(args.f != None):
        write_metrics_to_file(metrics, args.f)
    else:
        print(metrics)


if __name__ == "__main__":
    main()