import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import glob
from utils.utilities import list_submissions


def get_performers(df, n):
    """Get the n worst and best performing predictions for each unit and for each month separately."""
    col = df.columns[0]
    time, unit = df.index.names
    worst_by_unit = (
        df.groupby(level=1, group_keys=False)
        .apply(lambda x: x.nlargest(n, col))
        .reset_index()
        .set_index([unit, time])
    )
    best_by_unit = (
        df.groupby(level=1, group_keys=False)
        .apply(lambda x: x.nsmallest(n, col))
        .reset_index()
        .set_index([unit, time])
    )
    worst_by_month = (
        df.groupby(level=0, group_keys=False)
        .apply(lambda x: x.nlargest(n, col))
        .reset_index()
        .set_index([time, unit])
    )
    best_by_month = (
        df.groupby(level=0, group_keys=False)
        .apply(lambda x: x.nsmallest(n, col))
        .reset_index()
        .set_index([time, unit])
    )
    return {
        "worst_by_unit": worst_by_unit,
        "best_by_unit": best_by_unit,
        "worst_by_month": worst_by_month,
        "best_by_month": best_by_month,
    }


def analyze_single_model(submission, target_type, window, n=10):
    model_name = submission.name
    results = {model_name: {}}

    pattern = f"{submission}/{target_type}/window={window}/*/*.parquet"
    parquet_files = glob.glob(pattern)

    for file in parquet_files:
        metric = Path(file).stem
        df = pd.read_parquet(file)

        results[model_name][metric] = get_performers(df, n)

    return results


def analyze_all_models(submissions, target_type, window, n=10):
    """Analyze worst and best performing predictions for each metric and model."""
    submissions = Path(submissions)
    submissions = list_submissions(submissions)
    results = {}
    for submission in submissions:

        results.update(analyze_single_model(submission, target_type, window, n))

    return results


def print_worst_performers(results):
    """Print the worst and best performing predictions in a readable format."""
    for model, metrics in results.items():
        print(f"\nModel: {model}")
        print("-" * 50)

        for metric_name, metric_results in metrics.items():
            print(f"\nTop worst {metric_name.upper()} values for each unit and month:")
            print(metric_results["worst"])
            print(f"\nTop best {metric_name.upper()} values for each unit and month:")
            print(metric_results["best"])
            print("\n")


if __name__ == "__main__":
    # Example usage
    print("\nExample output:")
    print("=" * 80)

    # Create sample data
    import numpy as np

    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
    units = ["A", "B", "C"]

    # Create multi-index
    index = pd.MultiIndex.from_product([dates, units], names=["month_id", "unit"])

    # Create sample values
    values = np.random.normal(100, 20, len(index))
    df = pd.DataFrame({"value": values}, index=index)

    # Get worst and best performers
    result = get_performers(df, n=3)
    print("\nSample DataFrame:")
    print(df)
    print("\nWorst Performers by unit (n=3):")
    print(result["worst_by_unit"])
    print("\nBest Performers by unit (n=3):")
    print(result["best_by_unit"])
    print("\nWorst Performers by month (n=3):")
    print(result["worst_by_month"])
    print("\nBest Performers by month (n=3):")
    print(result["best_by_month"])
