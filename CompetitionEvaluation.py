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


def load_data(observed_path: str, forecasts_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pq.read_table(forecasts_path)
    predictions = predictions.to_pandas()
    observed = pq.read_table(observed_path)
    observed = observed.to_pandas()
    return observed, predictions

def structure_data(observed: pd.DataFrame, predictions: pd.DataFrame, draw_column_name: str = "draw", data_column_name: str = "prediction", axis: int = -1) -> tuple[xr.DataArray, xr.DataArray]:
    # The samples must be named "member" and the outcome variable needs to be named the same in xs.crps_ensemble()
    
    predictions = predictions.reset_index()
    observed = observed.reset_index()

    if "index" in observed.columns:
        observed = observed.drop(columns = ["index"])
    if "index" in predictions.columns:
        predictions = predictions.drop(columns = ["index"])

    assert "month_id" in observed.columns,  f"'month_id' column not found in observed data."
    assert "month_id" in predictions.columns,  f"'month_id' column not found in predictions data."

    assert observed.columns.isin(['country_id','priogrid_gid']).any(),  f"'country_id'/'priogrid_gid' column not found in observed data."
    assert predictions.columns.isin(['country_id','priogrid_gid']).any(),  f"'country_id'/'priogrid_gid' column not found in predictions data."

    assert draw_column_name in predictions.columns, f"{draw_column_name} not in predictions data"

    assert data_column_name in predictions.columns, f"{data_column_name} not in predictions data"
    assert data_column_name in observed.columns, f"{data_column_name} not in observed data"


    assert len(observed.columns) == 3, f"Observed data should only be three variables: 'month_id', 'country_id' (or 'priogrid_gid'), and {data_column_name}."
    assert len(predictions.columns) == 4, f"Predictions data should only be four variables: 'month_id', 'country_id' (or 'priogrid_gid'), {draw_column_name}, and {data_column_name}."

    nmonths = len(observed["month_id"].unique())
    nunits =  len(observed["country_id"].unique())
    nmembers = len(predictions[draw_column_name].unique())

    assert len(predictions.index) == nmonths*nunits*nmembers, f"Predictions data is not a balanced dataset with nobs = unique_months * unique_units * unique_{draw_column_name}."
    assert len(observed.index) == nmonths*nunits, f"Observed data is not a balanced dataset with nobs = unique_months * unique_units."

    # To simplify internal affairs:
    predictions = predictions.rename(columns = {draw_column_name: "member", data_column_name: "outcome"})
    observed = observed.rename(columns = {data_column_name: "outcome"})

    # Expand the actuals to cover all steps used in the prediction file
    #unique_steps = predictions["step"].unique()
    #if(len(unique_steps) > 1):
    #    observed["step"] = [unique_steps for i in observed.index]
    #    observed = observed.explode("step")
    #elif(len(unique_steps) == 1):
    #    observed["step"] = unique_steps[0]
    #else: 
    #    TypeError("Predictions does not contain unique steps.")

    # Set up multi-index to easily convert to xarray
    if "priogrid_gid" in predictions.columns:
        #predictions = predictions.set_index(['month_id', 'priogrid_gid', 'step', 'member'])
        predictions = predictions.set_index(['month_id', 'priogrid_gid', 'member'])
        #observed = observed.set_index(['month_id', 'priogrid_gid', 'step'])
        observed = observed.set_index(['month_id', 'priogrid_gid'])
    elif "country_id" in predictions.columns:
        #predictions = predictions.set_index(['month_id', 'country_id', 'step', 'member'])
        predictions = predictions.set_index(['month_id', 'country_id', 'member'])
        #observed = observed.set_index(['month_id', 'country_id', 'step'])
        observed = observed.set_index(['month_id', 'country_id'])
    else:
        TypeError("priogrid_gid or country_id must be an identifier")
    

    # Convert to xarray
    xpred = predictions.to_xarray()
    xobserved = observed.to_xarray()

    assert np.issubdtype(xpred["outcome"].dtype, np.integer),  f"Forecasts not integers when converted to xarray. Probably not balanced set, or missing data in forecasts."    
    assert np.issubdtype(xobserved["outcome"].dtype, np.integer),  f"Observations not integers when converted to xarray. Probably not balanced set, or missing data in forecasts."

    odim = dict(xobserved.dims)
    pdim = dict(xpred.dims)
    if "member" in pdim:
        del pdim["member"]
    assert odim == pdim, f"observed and predictions must have matching shapes or matching shapes except the '{draw_column_name}' dimension"

    return xobserved, xpred

def calculate_metrics(observed: xr.DataArray, predictions: xr.DataArray, metric: str, **kwargs) -> pd.DataFrame:
    # Calculate average crps for each step (so across the other dimensions)
    if "priogrid_gid" in predictions.coords:
        if metric == "crps":
            ensemble = xs.crps_ensemble(observed, predictions, dim=['month_id', 'priogrid_gid'])
        elif metric == "ign":
            ensemble = ensemble_ignorance_score_xskillscore(observed, predictions, dim=['month_id', 'priogrid_gid'], **kwargs)
        else: 
            TypeError("metric must be 'crps' or 'ign'.")

    elif "country_id" in predictions.coords:
        if metric == "crps":
            ensemble = xs.crps_ensemble(observed, predictions, dim=['month_id', 'country_id'])
        elif metric == "ign":
            ensemble = ensemble_ignorance_score_xskillscore(observed, predictions, dim=['month_id', 'country_id'], **kwargs)
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
                                     epilog = "Example usage: python CompetitionEvaluation.py -o ./data/cm_actuals.csv -p ./data/cm_benchmark_1.parquet -f ./results/tst.csv")
    parser.add_argument('-o', metavar='observed', type=str, help='path to csv-file where the observed outcomes are recorded')
    parser.add_argument('-p', metavar='predictions', type=str, help='path to parquet file where the predictions are recorded in long format')
    parser.add_argument('-f', metavar='file', type=str, help='path to csv-file where you want metrics to be stored')

    args = parser.parse_args()

    observed, predictions = load_data(args.o, args.p)
    observed, predictions = structure_data(observed, predictions)
    metrics = calculate_metrics(observed, predictions, metric = "crps")
    if(args.f != None):
        write_metrics_to_file(metrics, args.f)
    else:
        print(metrics)


if __name__ == "__main__":
    main()