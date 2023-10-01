from pathlib import Path
import yaml
from CompetitionEvaluation import load_data, structure_data, calculate_metrics
from dataclasses import dataclass
import os
import xarray
import numpy as np
import numpy.typing as npt
from scipy.signal import resample
import argparse

import logging
logging.getLogger(__name__)
logging.basicConfig(filename='evaluate_submission.log', encoding='utf-8', level=logging.INFO)

def evaluate_forecast(forecast_file: str | os.PathLike, expected_samples: int, actuals_folder: str|os.PathLike, submission_root: str|os.PathLike) -> None:

    @dataclass(frozen=True)
    class Actuals:
        target: str
        test_window: str
        parquet_file: str

    actuals = [Actuals("cm", "test_window_2018", "cm_actuals_2018.parquet"),
    Actuals("cm", "test_window_2019", "cm_actuals_2019.parquet"),
    Actuals("cm", "test_window_2020", "cm_actuals_2020.parquet"),
    Actuals("cm", "test_window_2021", "cm_actuals_2021.parquet"),
    Actuals("pgm", "test_window_2018", "pgm_actuals_2018.parquet"),
    Actuals("pgm", "test_window_2019", "pgm_actuals_2019.parquet"),
    Actuals("pgm", "test_window_2020", "pgm_actuals_2020.parquet"),
    Actuals("pgm", "test_window_2021", "pgm_actuals_2021.parquet")]

    _, name, target, window = forecast_file.relative_to(submission_root).parent.parts
    actual = [actual for actual in actuals if actual.target == target and actual.test_window == window]

    if target == "pgm":
        target_column = "priogrid_gid"
    elif target == "cm":
        target_column = "country_id"
    else:
        raise ValueError(f'Target {target} must be either "pgm" or "cm".')

    if len(actual) != 1:
        raise ValueError("Only one hit allowed.")
    actual = actual[0]
    actual_file = actuals_folder/actual.target/actual.test_window/actual.parquet_file

    observed, predictions = load_data(observed_path = actual_file, forecasts_path=forecast_file)

    if predictions.index.names != ['month_id', target_column, 'draw']:
        if predictions.index.names == [None] and all([var in predictions.columns for var in ['month_id', target_column, 'draw']]):
            # Index is not set, but index variables are in the file.
            predictions[target_column] = predictions[target_column].astype(int)
            predictions["month_id"] = predictions["month_id"].astype(int)
            predictions.set_index(['month_id', target_column, 'draw'], inplace = True)
        else: 
            logging.warning(f'Predictions file {forecast_file} does not have correct index. Currently: {predictions.index.names}')
            if len(predictions.index.names) == 3:
                logging.warning(f'Attempts to rename index.')
                predictions.index.names = ['month_id', target_column, 'draw']
            else:
                raise ValueError(f'Predictions file {forecast_file} does not contain correct index columns.')

    if len(observed.columns) == 1 and "outcome" not in observed.columns:
        logging.warning(f'Actuals file {actual_file} does not have the "outcome" folder.')
        logging.warning(f'Renaming column.')
        observed.columns = ["outcome"]

    if len(predictions.columns) == 1 and "outcome" not in observed.columns:
        logging.warning(f'Predictions file {forecast_file} does not have the "outcome" folder.')
        logging.warning(f'Renaming column.')
        predictions.columns = ["outcome"]

    if len(predictions.columns) != 1:
        raise ValueError("Predictions file can only have 1 column.")

    if len(observed.columns) != 1:
        raise ValueError("Actuals file can only have 1 column.")
    
    units_in_predictions_not_in_observed = [c for c in predictions.index.unique(level=target_column) if c not in observed.index.unique(level=target_column)]
    units_in_observed_not_in_predictions = [c for c in observed.index.unique(level=target_column) if c not in predictions.index.unique(level=target_column)]

    times_in_observed_not_in_predictions = [c for c in observed.index.unique(level="month_id") if c not in predictions.index.unique(level="month_id")]
    times_in_predictions_not_in_observed = [c for c in predictions.index.unique(level="month_id") if c not in observed.index.unique(level="month_id")]

    assert len(units_in_predictions_not_in_observed) == 0, f'Lacking actuals for {target_column}: {units_in_predictions_not_in_observed}'
    assert len(units_in_observed_not_in_predictions) == 0, f'Lacking predictions for {target_column}: {units_in_observed_not_in_predictions}'
    assert len(times_in_observed_not_in_predictions) == 0, f'Lacking predictions for {target_column}: {times_in_observed_not_in_predictions}'
    assert len(times_in_predictions_not_in_observed) == 0, f'Lacking actuals for {target_column}: {times_in_predictions_not_in_observed}'

    observed, predictions = structure_data(observed, predictions, draw_column_name="draw", data_column_name = "outcome")

    predictions[target_column] in observed[target_column]
    predictions["month_id"] in observed["month_id"]

        

    crps_per_unit = calculate_metrics(observed, predictions, metric = "crps", aggregate_over="month_id")
    mis_per_unit = calculate_metrics(observed, predictions, metric = "mis", prediction_interval_level = 0.9, aggregate_over="month_id")

    crps_per_month = calculate_metrics(observed, predictions, metric = "crps", aggregate_over=target_column)
    mis_per_month = calculate_metrics(observed, predictions, metric = "mis", prediction_interval_level = 0.9, aggregate_over=target_column)

    if predictions.dims['member'] != expected_samples:
        logging.warning(f'Number of samples ({predictions.dims["member"]}) is not 1000. Using scipy.signal.resample to get {expected_samples} samples when calculating Ignorance Score.')
        np.random.seed(284975)
        arr: npt.ArrayLike = resample(predictions.to_array(), expected_samples, axis = 3)
        arr = np.where(arr<0, 0, arr) # For the time when resampling happens to go below zero.

        new_container = predictions.sel(member = 1)
        new_container = new_container.expand_dims({"member": range(0,expected_samples)}).to_array().transpose("variable", "month_id", target_column, "member")
        predictions: xarray.Dataset = xarray.DataArray(data = arr, coords = new_container.coords).to_dataset(dim="variable")

    if bool((predictions["outcome"] < 0).any()):
        logging.warning(f'Found negative predictions. These are censored at 0 before calculating Ignorance Score.')
        predictions["outcome"] = xarray.where(predictions["outcome"]<0, 0, predictions["outcome"])
        
    ign_per_unit = calculate_metrics(observed, predictions, metric = "ign", bins = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5], aggregate_over="month_id")
    ign_per_month = calculate_metrics(observed, predictions, metric = "ign", bins = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5], aggregate_over=target_column)
    
    eval_path = forecast_file.parent/"eval"
    eval_path.mkdir(exist_ok=True)
    crps_per_unit.to_parquet(eval_path/"crps_per_unit.parquet")
    crps_per_month.to_parquet(eval_path/"crps_per_month.parquet")
    ign_per_unit.to_parquet(eval_path/"ign_per_unit.parquet")
    ign_per_month.to_parquet(eval_path/"ign_per_month.parquet")
    mis_per_unit.to_parquet(eval_path/"mis_per_unit.parquet")
    mis_per_month.to_parquet(eval_path/"mis_per_month.parquet")

def main():
    parser = argparse.ArgumentParser(description="Method for evaluation of submissions to the ViEWS Prediction Challenge",
                                     epilog = "Example usage: python evaluate_submissions.py -s ./submissions -a ./actuals -e 100")
    parser.add_argument('-s', metavar='submissions', type=str, help='path to folder with submissions complying with submission_template')
    parser.add_argument('-a', metavar='actuals', type=str, help='path to folder with actuals')
    parser.add_argument('-e', metavar='expected', type=int, help='expected samples', default = 1000)
    args = parser.parse_args()


    submission_path = Path(args.s)
    actuals_folder = Path(args.a)
    expected_samples = args.e

    submission_root = submission_path.parent

    submissions = [submission for submission in submission_path.iterdir() if submission.is_dir() and not submission.stem == "__MACOSX"]

    for submission in submissions:

        try:
            with open(submission/"submission_details.yml") as f:
                submission_details = yaml.safe_load(f)
        except:
            logging.error(f'{submission/"submission_details.yml"} could not be loaded.')

        prediction_files = list(submission.glob("**/*.parquet"))
        prediction_files = [f for f in prediction_files if not f.stem.split("_")[0] == "eval"]
        prediction_files = [f for f in prediction_files if not f.parent.parts[-1] == "eval"]
        prediction_files = [f for f in prediction_files if "__MACOSX" not in f.parts]
        
        for f in prediction_files:
            try:
                logging.info(f'Evaluating {str(f)}')
                evaluate_forecast(f, expected_samples, actuals_folder, submission_root)
            except Exception as e:
                logging.error(f'{str(e)}')


if __name__ == "__main__":
    main()