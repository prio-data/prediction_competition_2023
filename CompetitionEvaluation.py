import argparse
import pandas as pd
import pyarrow.parquet as pq
from zipfile import ZipFile
from pathlib import Path

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

def structure_data(observed: pd.DataFrame, predictions: pd.DataFrame) -> tuple[xr.DataArray, xr.DataArray]:
    # The samples must be named "member" and the outcome variable needs to be named the same in xs.crps_ensemble()
    
    predictions = predictions.reset_index()
    observed = observed.reset_index()

    predictions = predictions.rename(columns = {"draw": "member", "prediction": "outcome"})
    observed = observed.rename(columns = {"ged_sb": "outcome"})

    

    # Expand the actuals to cover all steps used in the prediction file
    unique_steps = predictions["step"].unique()
    if(len(unique_steps) > 1):
        observed["step"] = [unique_steps for i in observed.index]
        observed = observed.explode("step")
    elif(len(unique_steps) == 1):
        observed["step"] = unique_steps[0]
    else: 
        TypeError("Predictions does not contain unique steps.")

    # Set up multi-index to easily convert to xarray
    if "priogrid_gid" in predictions.columns:
        predictions = predictions.set_index(['month_id', 'priogrid_gid', 'step', 'member'])
        observed = observed.set_index(['month_id', 'priogrid_gid', 'step'])
    elif "country_id" in predictions.columns:
        predictions = predictions.set_index(['month_id', 'country_id', 'step', 'member'])
        observed = observed.set_index(['month_id', 'country_id', 'step'])
    else:
        TypeError("priogrid_gid or country_id must be an identifier")
    

    # Convert to xarray
    xpred = predictions.to_xarray()
    xobserved = observed.to_xarray()
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