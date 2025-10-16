import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import glob
from utils.utilities import list_submissions


def get_performers(df: pd.DataFrame, n: int) -> dict:
    """
    Get the n worst and best performing predictions for each unit and for each month separately.

    Args:
        df: pd.DataFrame with a multi-index of month and unit
        n: number of worst and best performing predictions to return

    Returns:
        dict: A dictionary with the n worst and best performing predictions for each unit and for each month separately.
    """
    col = df.columns[0]
    time, unit = df.index.names
    
    def convert_to_dict(group_df):
        return {
            (idx[0], idx[1]): val[col]
            for idx, val in group_df.iterrows()
        }
    
    worst_by_unit = convert_to_dict(
        df.groupby(level=1, group_keys=False)
        .apply(lambda x: x.nlargest(n, col))
    )
    
    best_by_unit = convert_to_dict(
        df.groupby(level=1, group_keys=False)
        .apply(lambda x: x.nsmallest(n, col))
    )
    
    worst_by_month = convert_to_dict(
        df.groupby(level=0, group_keys=False)
        .apply(lambda x: x.nlargest(n, col))
    )
    
    best_by_month = convert_to_dict(
        df.groupby(level=0, group_keys=False)
        .apply(lambda x: x.nsmallest(n, col))
    )
    
    return {
        "worst_by_unit": worst_by_unit,
        "best_by_unit": best_by_unit,
        "worst_by_month": worst_by_month,
        "best_by_month": best_by_month,
    }


def analyze_single_model(submission: Path|str, target: str, window: str, n: int = 10) -> dict:
    submission = Path(submission)
    model_name = submission.name
    results = {model_name: {}}

    pattern = f"{submission}/eval/{target}/window={window}/*/*.parquet"
    parquet_files = glob.glob(pattern)
    for file in parquet_files:
        metric = Path(file).stem
        df = pd.read_parquet(file)

        results[model_name][metric] = get_performers(df, n)

    return results


def analyze_all_models(submissions: Path|str, target: str, window: str, n: int = 10) -> dict:
    """
    Analyze worst and best performing predictions for each metric and model.

    Args:
        submissions: Path to the submissions directory
        target: target variable, either "cm" or "pgm
        window: window size
        n: number of worst and best performing predictions to return

    Returns:
        dict: A dictionary with the n worst and best performing predictions for each metric and model.
    """
    submissions = Path(submissions)
    submissions = list_submissions(submissions)
    results = {}
    for submission in submissions:

        results.update(analyze_single_model(submission, target, window, n))

    return results


def print_table(data, title):
    if not data:
        return
    print(f"\n{title}:")
    print("Month  Unit  Value")
    print("-" * 20)
    for (m, u), v in data:
        print(f"{m:6d}  {u:4d}  {v:10.4f}")


def print_analysis(
    results: dict,
    metric: str | list[str] | None = None,
    month: int | list[int] | None = None,
    unit: int | list[int] | None = None,
    show_best: bool = False,
    show_worst: bool = False,
):
    """
    Print the analysis results in a readable format.
    
    Args:
        results: Dictionary containing the analysis results
        metric: Single metric or list of metrics to filter by. If None, shows all metrics.
        month: Single month or list of months to filter by. If None, shows all months.
        unit: Single unit or list of units to filter by. If None, shows all units.
        show_best: Whether to show best performing predictions.
        show_worst: Whether to show worst performing predictions.
    """
    if isinstance(month, int):
        month = [month]
    if isinstance(unit, int):
        unit = [unit]
    if isinstance(metric, str):
        metric = [metric]
    if not show_best and not show_worst:
        show_best = True
        show_worst = True

    if unit and month:
        raise ValueError("Cannot filter by both unit and month")

    for model, metrics in results.items():
        print(f"\nModel: {model}")
        print("-" * 50)

        for metric_name, metric_results in metrics.items():
            if metric is not None and metric_name not in metric:
                continue

            print(f"\n{metric_name.upper()} Analysis:")
            print("-" * 30)            

            if show_worst:
                if unit:
                    worst_by_unit = []
                    for (m, u), value in metric_results["worst_by_unit"].items():
                        if u in unit:
                            worst_by_unit.append(((m, u), value))
                    print_table(worst_by_unit, "Worst performing predictions by unit")
                else:
                    worst_by_month = []
                    for (m, u), value in metric_results["worst_by_month"].items():
                        if month is None or m in month:
                            worst_by_month.append(((m, u), value))
                    print_table(worst_by_month, "Worst performing predictions by month")
            
            if show_best:
                if unit:
                    best_by_unit = []
                    for (m, u), value in metric_results["best_by_unit"].items():
                        if u in unit:
                            best_by_unit.append(((m, u), value))
                    print_table(best_by_unit, "Best performing predictions by unit")
                else:
                    best_by_month = []
                    for (m, u), value in metric_results["best_by_month"].items():
                        if month is None or m in month:
                            best_by_month.append(((m, u), value))
                    print_table(best_by_month, "Best performing predictions by month")
            
            print()

