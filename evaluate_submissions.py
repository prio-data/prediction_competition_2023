from pathlib import Path
import yaml
from CompetitionEvaluation import load_data, structure_data, calculate_metrics
from utilities import list_submissions, get_target_data, TargetType
from dataclasses import dataclass
import os
import xarray
import numpy as np
import numpy.typing as npt
from scipy.signal import resample
import argparse
import pandas as pd
import pyarrow

import logging
logging.getLogger(__name__)
logging.basicConfig(filename='evaluate_submission.log', encoding='utf-8', level=logging.INFO)

def evaluate_forecast(forecast: pd.DataFrame, 
                      actuals: pd.DataFrame, 
                      target: TargetType,
                      expected_samples: int,
                      save_to: str|os.PathLike,
                      draw_column: str ="draw", 
                      data_column: str = "outcome",
                      bins: list[float] = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5]) -> None:

    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError(f'Target {target} must be either "pgm" or "cm".')

    # Cast to xarray
    observed, predictions = structure_data(actuals, forecast, draw_column_name=draw_column, data_column_name = data_column)

    if bool((predictions["outcome"] > 10e9).any()):
        logging.warning(f'Found predictions larger than earth population. These are censored at 10 billion.')
        predictions["outcome"] = xarray.where(predictions["outcome"]>10e9, 10e9, predictions["outcome"])
        
    crps = calculate_metrics(observed, predictions, metric = "crps", aggregate_over="nothing")
    mis = calculate_metrics(observed, predictions, metric = "mis", prediction_interval_level = 0.9, aggregate_over="nothing")

    if predictions.dims['member'] != expected_samples:
        logging.warning(f'Number of samples ({predictions.dims["member"]}) is not 1000. Using scipy.signal.resample to get {expected_samples} samples when calculating Ignorance Score.')
        np.random.seed(284975)
        arr: npt.ArrayLike = resample(predictions.to_array(), expected_samples, axis = 3)
        arr = np.where(arr<0, 0, arr) # For the time when resampling happens to go below zero.

        new_container = predictions.sel(member = 1)
        new_container = new_container.expand_dims({"member": range(0,expected_samples)}).to_array().transpose("variable", "month_id", unit, "member")
        predictions: xarray.Dataset = xarray.DataArray(data = arr, coords = new_container.coords).to_dataset(dim="variable")

    if bool((predictions["outcome"] < 0).any()):
        logging.warning(f'Found negative predictions. These are censored at 0 before calculating Ignorance Score.')
        predictions["outcome"] = xarray.where(predictions["outcome"]<0, 0, predictions["outcome"])


    ign = calculate_metrics(observed, predictions, metric = "ign", bins = bins, aggregate_over="nothing")
    
    # Long-format out
    crps.rename(columns={"crps": "value"}, inplace=True)
    ign.rename(columns={"ign": "value"}, inplace=True)
    mis.rename(columns={"mis": "value"}, inplace=True)
    
    (save_to / "metric=crps").mkdir(exist_ok=True, parents = True)
    (save_to / "metric=ign").mkdir(exist_ok=True, parents = True)
    (save_to / "metric=mis").mkdir(exist_ok=True, parents = True)
    crps.to_parquet(save_to / "metric=crps" / "crps.parquet")
    ign.to_parquet(save_to / "metric=ign" / "ign.parquet")
    mis.to_parquet(save_to / "metric=mis" / "mis.parquet")

def match_forecast_with_actuals(submission, actuals_folder, target: TargetType, window: str):

    filter = pyarrow.compute.field("window") == window
    actuals = get_target_data(actuals_folder, target = target, filters = filter)
    predictions = get_target_data(submission, target = target, filters = filter)

    predictions.drop(columns=["window"], inplace = True)
    actuals.drop(columns=["window"], inplace = True)

    return actuals, predictions

def evaluate_submission(submission: str|os.PathLike, 
                        acutals: str|os.PathLike, 
                        targets: list[TargetType], 
                        windows: list[str], 
                        expected: int, 
                        bins: list[float],
                        draw_column: str = "draw", 
                        data_column: str = "outcome") -> None:
    """Loops over all targets and windows in a submission folder, match them with the correct test dataset, and estimates evaluation metrics. 
    Stores evaluation data as .parquet files in {submission}/eval/{target}/window={window}/.

    Parameters
    ----------
    submission : str | os.PathLike
        Path to a folder structured like a submission_template
    acutals : str | os.PathLike
        Path to actuals folder structured like {actuals}/{target}/window={window}/data.parquet
    targets : list[TargetType]
        A list of strings, either ["pgm"] for PRIO-GRID-months, or ["cm"] for country-months, or both.
    windows : list[str]
        A list of strings indicating the window of the test dataset. The string should match windows in data in the actuals folder.
    expected : int
        The expected numbers of samples. Due to how Ignorance Score is defined, all IGN metric comparisons must be across models with equal number of samples.
    bins : list[float]
        The binning scheme used in the Ignorance Score.
    draw_column : str
        The name of the sample column. We assume samples are drawn independently from the model. Default = "draw"
    data_column : str
        The name of the data column. Default = "outcome"
    """
    
    for target in targets:
        for window in windows:
            if any((submission / target).glob("**/*.parquet")): # test if there are prediction files in the target
                observed_df, pred_df = match_forecast_with_actuals(submission, acutals, target, window)
                save_to = submission / "eval" / f'{target}' / f'window={window}'
                evaluate_forecast(forecast = pred_df, 
                                actuals = observed_df, 
                                target = target,
                                expected_samples = expected,
                                draw_column=draw_column,
                                data_column=data_column,
                                bins = bins,
                                save_to = save_to)

def evaluate_all_submissions(submissions: str|os.PathLike, 
                             acutals: str|os.PathLike, 
                             targets: list[TargetType], 
                             windows: list[str], 
                             expected: int, 
                             bins: list[float],
                             draw_column: str = "draw", 
                             data_column: str = "outcome") -> None:
    """Loops over all submissions in the submissions folder, match them with the correct test dataset, and estimates evaluation metrics. 
    Stores evaluation data as .parquet files in {submissions}/{submission_name}/eval/{target}/window={window}/.

    Parameters
    ----------
    submissions : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template
    acutals : str | os.PathLike
        Path to actuals folder structured like {actuals}/{target}/window={window}/data.parquet
    targets : list[TargetType]
        A list of strings, either ["pgm"] for PRIO-GRID-months, or ["cm"] for country-months, or both.
    windows : list[str]
        A list of strings indicating the window of the test dataset. The string should match windows in data in the actuals folder.
    expected : int
        The expected numbers of samples. Due to how Ignorance Score is defined, all IGN metric comparisons must be across models with equal number of samples.
    bins : list[float]
        The binning scheme used in the Ignorance Score.
    draw_column : str
        The name of the sample column. We assume samples are drawn independently from the model. Default = "draw"
    data_column : str
        The name of the data column. Default = "outcome"
    """
    
    submissions = Path(submissions)
    submissions = list_submissions(submissions)
    acutals = Path(acutals)

    for submission in submissions:
        try:
            logging.info(f'Evaluating {submission.name}')
            evaluate_submission(submission, acutals, targets, windows, expected, bins, draw_column, data_column)
        except Exception as e:
            logging.error(f'{str(e)}')

def main():
    parser = argparse.ArgumentParser(description="Method for evaluation of submissions to the ViEWS Prediction Challenge",
                                     epilog = "Example usage: python evaluate_submissions.py -s ./submissions -a ./actuals -e 100")
    parser.add_argument('-s', metavar='submissions', type=str, help='path to folder with submissions complying with submission_template')
    parser.add_argument('-a', metavar='actuals', type=str, help='path to folder with actuals')
    parser.add_argument('-t', metavar='targets', nargs = "+", type=str, help='pgm or cm or both', default = ["pgm", "cm"])
    parser.add_argument('-w', metavar='windows', nargs = "+", type=str, help='windows to evaluate', default = ["Y2018", "Y2019", "Y2020", "Y2021"])
    parser.add_argument('-e', metavar='expected', type=int, help='expected samples', default = 1000)
    parser.add_argument('-sc', metavar='draw_column', type=str, help='(Optional) name of column for the unique samples', default = "draw")
    parser.add_argument('-dc', metavar='data_column', type=str, help='(Optional) name of column with data, must be same in both observed and predictions data', default = "outcome")
    parser.add_argument('-ib', metavar = 'bins', nargs = "+", type = float, help='Set a binning scheme for the ignorance score. List or integer (nbins). E.g., "--ib 0 0.5 1 5 10 100 1000". None also allowed.', default = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5])    

    args = parser.parse_args()


    submissions = Path(args.s)
    acutals = Path(args.a)
    expected = args.e
    targets = args.t
    windows = args.w
    draw_column = args.sc
    data_column = args.dc
    bins = args.ib

    evaluate_all_submissions(submissions, acutals, targets, windows, expected, draw_column, data_column, bins)


if __name__ == "__main__":
    main()