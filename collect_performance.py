from pathlib import Path
import os
import pandas as pd
import argparse

from utilities import (
    list_submissions,
    get_target_data,
    views_month_id_to_year,
    views_month_id_to_month,
    TargetType,
    get_submission_details,
    data_in_target,
)


def get_eval(
    submission: str | os.PathLike,
    target: TargetType,
    groupby: str | list[str] = None,
    aggregate_submissions: bool = False,
) -> pd.DataFrame:
    """Convenience function to read and aggregate evaluation metrics from a submission.

    Parameters
    ----------
    submission : str | os.PathLike
        Path to a folder structured like a submission_template
    target : TargetType
        A string, either "pgm" for PRIO-GRID-months, or "cm" for country-months.
    groupby : str | list[str], optional
        A dimension to aggregate results across. Some options (all except None and "pooled" can be combined in a list):
        None: no aggregation
        "pooled": complete aggregation
        "year": aggregate by calendar year
        "month": aggregate by calendar month
        "month_id": aggregate by month_id (1 is January 1980)
        "country_id": aggregate by country (currently only works for target == "cm")
        "priogrid_gid": aggregate by PRIO-GRID id.
    aggregate_submissions : bool
        Aggregate across submissions. Default false (i.e., aggregate by [team, model])

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        Target must be "cm" or "pgm".
    FileNotFoundError
        There must be .parquet-files in the submission/{target} sub-folders.
    """

    if target == "cm":
        unit = "country_id"
    elif target == "pgm":
        unit = "priogrid_gid"
    else:
        raise ValueError(f'Target must be either "cm" or "pgm".')

    submission = Path(submission)
    if not data_in_target(submission, target):
        raise FileNotFoundError

    groupby_inner = groupby.copy()

    match groupby_inner:
        case None:
            sdetails = get_submission_details(submission)
            df["team"] = sdetails["team"]
            df["model"] = sdetails["even_shorter_identifier"]
            return df
        case str():
            if groupby_inner == "pooled":
                groupby_inner = []
            else:
                groupby_inner = [groupby_inner]
        case list():
            pass
        case _:
            raise ValueError

    df = get_target_data(submission / "eval", target=target)
    if df.index.names != [None]:
        df = df.reset_index()

    if "year" in groupby_inner:
        df["year"] = views_month_id_to_year(df["month_id"])
    if "month" in groupby_inner:
        df["month"] = views_month_id_to_month(df["month_id"])

    for col in ["month_id", unit, "window"]:
        if col not in groupby_inner:
            df = df.drop(columns=col)

    if aggregate_submissions:
        pass
    else:
        sdetails = get_submission_details(submission)
        df["team"] = sdetails["team"]
        df["model"] = sdetails["even_shorter_identifier"]
        groupby_inner.extend(["team", "model"])

    # Aggregate metric values
    groupby_inner.append("metric")
    df = df.set_index(groupby_inner)
    df = df.groupby(level=groupby_inner, observed=True).mean()
    groupby_inner.pop()

    # Pivot metrics to wide
    df = df.pivot_table(values=["value"], index=groupby_inner, columns="metric")
    df.columns = df.columns.get_level_values(1).to_list()

    return df


def evaluation_table(
    submissions: str | os.PathLike,
    target: TargetType,
    groupby: str | list[str],
    save_to: str | os.PathLike = None,
    aggregate_submissions: bool = False,
) -> None | pd.DataFrame:
    """Convenience function to make aggregated result tables of the evaulation metrics and store them to LaTeX, HTML, and excel format.

    Parameters
    ----------
    submissions : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template
    target : TargetType
        A string, either "pgm" for PRIO-GRID-months, or "cm" for country-months.
    groupby : str | list[str], optional
        A dimension to aggregate results across. Some options (all except "pooled" can be combined in a list):
        "pooled": complete aggregation
        "window": aggregate by test window
        "year": aggregate by calendar year
        "month": aggregate by calendar month
        "month_id": aggregate by month_id (1 is January 1980)
        "country_id": aggregate by country (currently only works for target == "cm")
        "priogrid_gid": aggregate by PRIO-GRID id.
    save_to : str | os.PathLike, optional
        Folder to store evaulation tables in LaTeX, HTML, and excel format.
    aggregate_submissions : bool
        Aggregate across submissions


    Returns
    -------
    pandas.DataFrame
        If save_to is None, or if groupby is a list or None, the function returns the dataframe.
        It can be useful to collate all evaluation data into one dataframe, but not to write everything out to a table.

    """

    groupby_inner = groupby.copy()
    match groupby_inner:
        case None:
            # Edge case if user specify a list of dimensions to groupby or no aggregation. Probably not something to plot tables for.
            return df
        case str():
            if groupby_inner == "pooled":
                groupby_inner = []
            else:
                groupby_inner = [groupby_inner]
        case list():
            pass
        case _:
            raise ValueError
    if aggregate_submissions:
        pass
    else:
        groupby_inner.extend(["team", "model"])

    submissions = list_submissions(Path(submissions))
    submissions = [
        submission for submission in submissions if data_in_target(submission, target)
    ]

    # Silently accept that there might not be evaluation data for all submissions for all targets for all windows.
    eval_data = []
    for submission in submissions:
        try:
            eval_df = get_eval(submission, target, groupby, aggregate_submissions)
            eval_data.append(eval_df)
        except FileNotFoundError as e:
            pass
    df = pd.concat(eval_data)

    if df.index.names != [None]:
        df = df.reset_index()

    # Aggregate metric values
    if len(groupby_inner) > 0:
        df = df.set_index(groupby_inner)
        df = df.groupby(level=groupby_inner, observed=True).mean().reset_index()
    else:
        # This is the case where groupby is "pooled" and across submissions is True. Should just return the global mean.
        df = df.mean()

    # Pull windows to wide
    if "window" in groupby_inner:
        sorting_column = df["window"].unique()[0]
        df = df.pivot_table(
            values=["crps", "ign", "mis"],
            index=[g for g in groupby_inner if g != "window"],
            aggfunc={"crps": "mean", "ign": "mean", "mis": "mean"},
            columns="window",
        )
        df = df.sort_values(("crps", sorting_column))
    else:
        if isinstance(df, pd.Series):
            # No need to sort single events.
            pass
        else:
            df = df.sort_values("crps")

    if save_to == None:
        return df
    else:
        file_stem = f"metrics_{target}_by={groupby_inner}"

        css_alt_rows = "background-color: #e6e6e6; color: black;"
        highlight_props = "background-color: #00718f; color: #fafafa;"
        df = (
            df.style.format(decimal=".", thousands=" ", precision=3)
            .highlight_min(axis=0, props=highlight_props)
            .set_table_styles(
                [{"selector": "tr:nth-child(even)", "props": css_alt_rows}]
            )
        )
        df.to_latex(save_to / f"{file_stem}.tex")
        df.to_html(save_to / f"{file_stem}.html")
        df.to_excel(save_to / f"{file_stem}.xlsx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Method for collating evaluations from all submissions in the ViEWS Prediction Challenge",
        epilog="Example usage: python collect_performance.py -s ./submissions",
    )
    parser.add_argument(
        "-s",
        metavar="submissions",
        type=str,
        help="path to folder with submissions complying with submission_template",
    )
    parser.add_argument(
        "-o",
        metavar="output_folder",
        type=str,
        help="path to folder to save result tables",
        default=None,
    )
    parser.add_argument(
        "-tt",
        metavar="target_type",
        type=str,
        help='target "pgm" or "cm"',
        default=None,
    )
    parser.add_argument(
        "-g",
        metavar="groupby",
        nargs="+",
        type=str,
        help="string or list of strings of dimensions to aggregate over",
        default=None,
    )

    args = parser.parse_args()

    evaluation_table(submissions=args.s, target=args.tt, groupby=args.g, save_to=args.o)
