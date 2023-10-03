from pathlib import Path
import xarray
import os
import yaml
import re
from dataclasses import dataclass
import pyarrow.parquet as pq
import argparse

import logging
logging.getLogger(__name__)
logging.basicConfig(filename='test_compliance.log', encoding='utf-8', level=logging.INFO)

def compliant_yaml(submission: str|os.PathLike): 
    try:
        with open(submission/"submission_details.yml") as f:
            submission_details = yaml.safe_load(f)
    except:
        logging.error(f'{submission/"submission_details.yml"} could not be loaded.')

    def test_is_str(entry):
        assert isinstance(entry, str)
    def test_is_not_zero(entry):
        assert len(entry) > 0

    for elem in ["team", "short_title", "even_shorter_identifier"]:
        test_is_str(submission_details[elem])
        test_is_not_zero(submission_details[elem])

    EMAIL_REGEX = re.compile(r'[^@]+@[^@]+\.[^@]+')
    assert EMAIL_REGEX.match(submission_details["contact"]), f'{submission_details["contact"]} is not a proper email address.'
    assert isinstance(submission_details["authors"], list)


def get_unit(forecast_file:str|os.PathLike) -> str:
    target = forecast_file.parent.parts[-2]
    if target == "pgm":
        unit = "priogrid_gid"
    elif target == "cm":
        unit = "country_id"
    else:
        raise ValueError(f'target ({target}) must be either "pgm" or "cm"')
    return unit

def get_prediction_files(submission: str|os.PathLike) -> list[str]:
    prediction_files = list(submission.glob("**/*.parquet"))
    prediction_files = [f for f in prediction_files if not f.stem.split("_")[0] == "eval"]
    prediction_files = [f for f in prediction_files if not f.parent.parts[-1] == "eval"]
    prediction_files = [f for f in prediction_files if "__MACOSX" not in f.parts]
    return prediction_files

def test_number_of_files(submission: str|os.PathLike):
    prediction_files = get_prediction_files(submission)
    pred_targets = [p.parent.parts[-2] for p in prediction_files]
    assert len(pred_targets) / len(set(pred_targets)) == 4, f'{submission} lacks one or more .parquet files.'

def match_prediction_with_actual(forecast_file: str|os.PathLike, actuals_folder: str|os.PathLike) -> os.PathLike:
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

    window = forecast_file.parent.parts[-1]
    target = forecast_file.parent.parts[-2]
    actual = [actual for actual in actuals if actual.target == target and actual.test_window == window]

    if len(actual) != 1:
        raise ValueError("Only one hit allowed.")
    actual = actual[0]
    actual_file = actuals_folder/actual.target/actual.test_window/actual.parquet_file
    return actual_file

def test_contains_expected_columns(predictions, forecast_file):
    unit = get_unit(forecast_file)
    column_diff =  set(predictions.columns) ^ set(["month_id", unit, "draw", "outcome"])
    assert len(column_diff) == 0, f'{forecast_file} does not contain correct column names. Missing: {column_diff}'

def test_month_id(predictions, observed, forecast_file):
    month_id_diff = set(predictions.index.unique(level="month_id")) ^ set(observed.index.unique(level="month_id"))
    assert len(month_id_diff) == 0, f'{forecast_file} does not contain the same set of month_id as actuals. Symmetric_difference: {month_id_diff}'

def test_unit_id(predictions, observed, forecast_file):
    unit = get_unit(forecast_file)
    unit_diff = set(predictions.index.unique(level=unit)) ^ set(observed.index.unique(level=unit))
    assert len(unit_diff) == 0, f'{forecast_file} does not contain the same set of {unit} as actuals. Symmetric_difference: {unit_diff}'

def test_outcome_nan(predictions, forecast_file):
    num_nan = predictions["outcome"].isna().sum()
    assert num_nan == 0, f'{forecast_file} contains n: {num_nan} missing values.'

def test_outcome_negative(predictions, forecast_file):
    num_negative = (predictions["outcome"]<0).sum()
    assert num_negative == 0, f'{forecast_file} contains n: {num_negative} negative values.'


def prediction_file_compliance(forecast_file: str|os.PathLike, actuals_folder: str|os.PathLike, expected_samples: int):
    actual_file = match_prediction_with_actual(forecast_file, actuals_folder)

    predictions = pq.read_table(forecast_file).to_pandas()
    observed = pq.read_table(actual_file).to_pandas()
    
    if predictions.index.names != [None]:
        predictions.reset_index(inplace = True)

    try:
        test_contains_expected_columns(predictions, forecast_file)
        unit = get_unit(forecast_file)
        predictions["month_id"] = predictions["month_id"].astype(int)
        predictions[unit] = predictions[unit].astype(int)
        predictions["draw"] = predictions["draw"].astype(int)
        predictions.set_index(["month_id", unit, "draw"], inplace = True)
    except Exception as e:
        logging.error(e)
    try:
        test_unit_id(predictions, observed, forecast_file)
    except Exception as e:
        logging.error(e)
    try:
        test_month_id(predictions, observed, forecast_file)
    except Exception as e:
        logging.error(e)
    try:
        test_outcome_negative(predictions, forecast_file)
    except Exception as e:
        logging.error(e)
    try:
        test_outcome_nan(predictions, forecast_file)
    except Exception as e:
        logging.error(e)

def main():
    parser = argparse.ArgumentParser(description="Method for evaluation of the compliance of submissions to the ViEWS Prediction Challenge",
                                     epilog = "Example usage: python test_compliance.py -s ./submissions -a ./actuals -e 100")
    parser.add_argument('-s', metavar='submissions', type=str, help='path to folder with submissions complying with submission_template')
    parser.add_argument('-a', metavar='actuals', type=str, help='path to folder with actuals')
    parser.add_argument('-e', metavar='expected', type=int, help='expected samples', default = 1000)
    args = parser.parse_args()

    submission_path = Path(args.s)
    actuals_folder = Path(args.a)
    expected_samples = args.e

    submissions = [submission for submission in submission_path.iterdir() if submission.is_dir() and not submission.stem == "__MACOSX"]

    for submission in submissions:
        logging.info(f'Testing submission: {submission}')
        try: 
            compliant_yaml(submission)
        except Exception as e:
            logging.error(f'{str(e)}')
        try: 
            test_number_of_files(submission)
        except Exception as e:
            logging.error(f'{str(e)}')

        prediction_files = get_prediction_files(submission)    
        for f in prediction_files:
            logging.info(f'Testing file: {f}')
            prediction_file_compliance(f, actuals_folder, expected_samples)


if __name__ == "__main__":
    main()
        
