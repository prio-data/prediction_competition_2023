# Instructions

Here, you'll find a small collection of functions that work with prediction-data in the format specified by the [ViEWS Prediction Challenge](https://viewsforecasting.org/prediction-competition-2/).

## Installation

1. Install [mamba](https://github.com/conda-forge/miniforge#mambaforge).

2. Add [conda-lock](https://github.com/conda/conda-lock) to the base environment.

``` console
mamba install --channel=conda-forge --name=base conda-lock
```

3. Install [git](https://git-scm.com/downloads).

4. Download our package from github:

```console
git clone https://github.com/prio-data/prediction_competition_2023.git
cd prediction_competition_2023
```
5. Create the virtual environment based on lock-file created from environment.yml

``` console
conda-lock
conda-lock install -n pred_eval_env  --mamba
mamba activate pred_eval_env
```
6. Run poetry (inside environment) to add additional python package requirements.

```console
poetry install
```

## Useage

The functions in this repository works on folders structured in a particular way (see below). This structure is similar to what is used in Apache Hive, and can therefore easily be read and filtered with functions in the Apache Arrow library that supports several programming languages including Python and R. See [Apache Arrow - Writing partitioned datasets](https://arrow.apache.org/docs/python/parquet.html#partitioned-datasets-multiple-files) for more information about the data structure we are using. See [here](https://arrow.apache.org/docs/r/articles/dataset.html) for information about how to read and write such data-structures in R.

Most functions run fairly quickly. We recommend at least 32Gb RAM when evaluating pgm-level data, and a full evaluation of the four defined prediction-windows take approx. 4 minutes on our computer. cm-level data should complete more or less instantly.

### The submission_template
Create one submission_template-folder for each unique model-description, possibly having predictions at both cm and pgm level. There should only be one .parquet-file in each "{target}/window={window}" subfolder. If you have more than one model, you will need several submission-folders with unique submission_details.yml. Please to not rename "submission_details.yml". 

```bash submission_template folder structure
.
├── cm
│   ├── window=Y2018
│   │   └── predictions_2018.parquet
│   ├── window=Y2019
│   │   └── predictions_2019.parquet
│   ├── window=Y2020
│   │   └── predictions_2020.parquet
│   └── window=Y2021
│       └── predictions_2021.parquet
├── pgm
│   ├── window=Y2018
│   │   └── predictions_2018.parquet
│   ├── window=Y2019
│   │   └── predictions_2019.parquet
│   ├── window=Y2020
│   │   └── predictions_2020.parquet
│   └── window=Y2021
│       └── predictions_2021.parquet
└── submission_details.yml
```

```yaml submission_details.yml
team: # Ideally 3-8 characters, to be used in plots and tables
short_title: "My title here" # Wrap this in "" if you are using special characters like ":" or "-".
even_shorter_identifier: # 1-2 words long, to be used in plots and tables. Do not need the team name.
authors: # Include all authors with one name+affil entry for each. Note the "-" for each entry.
  - name:
    affil:
  - name:
    affil:
contact:
```
### Test data / actuals
The code assumes that the test/actuals data is similarly stored. The observations (unit and month_id) inside must have the same window definition and contain the same units. E.g.: "Y2018" contains all month_id's in the calendar year of 2018. While the functions do expose the name of the column indicating each independent sample from a model and the column containing the forecasts, the defaults we use are "draw" and "outcome", respectively.

```bash
actuals
├── cm
│   ├── window=Y2018
│   │   └── cm_actuals_2018.parquet
│   ├── window=Y2019
│   │   └── cm_actuals_2019.parquet
│   ├── window=Y2020
│   │   └── cm_actuals_2020.parquet
│   └── window=Y2021
│       └── cm_actuals_2021.parquet
└── pgm
    ├── window=Y2018
    │   └── pgm_actuals_2018.parquet
    ├── window=Y2019
    │   └── pgm_actuals_2019.parquet
    ├── window=Y2020
    │   └── pgm_actuals_2020.parquet
    └── window=Y2021
        └── pgm_actuals_2021.parquet
```

### Data features
We also provide data with various features including our prediction target ("ged_sb") for both country-month and PRIO-GRID month aggregation levels in the same structure. To ease reading these into a computer with limited RAM, we have partitioned the pgm-level data on calendar years. Note that you can easily subset on any other dimension before reading into memory using [filtering expressions](https://arrow.apache.org/docs/python/compute.html#filtering-by-expressions) in Apache Arrow. When using this data, note that the competition assumes you will be making forecasts for a varying window number of months ahead. For instance, the "Y2018" window assumes you are using training data up to October 2017, then making predictions for each month in 2018 (so a 3-15 months ahead forecast). The final forecast window will not follow the calendar year.

```bash
.
├── cm
│   └── cm_features.parquet
└── pgm
    ├── year=1990
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=1991
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=1992
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=1993
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=1994
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=1995
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=1996
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=1997
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=1998
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=1999
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2000
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2001
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2002
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2003
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2004
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2005
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2006
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2007
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2008
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2009
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2010
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2011
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2012
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2013
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2014
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2015
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2016
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2017
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2018
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    ├── year=2019
    │   └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
    └── year=2020
        └── 9b5cbc69a393493fb2002b4a53188b67-0.parquet
```

### Matching tables
We also provide matching tables to get the country_id of a given PRIO-GRID-month (since borders change, this can change over time). Note that this classification is using the areal-majority as the rule.

The countries.csv is a matching table between the country_id, gwcode, and ISO code.

```bash
matching_tables
├── countries.csv
└── priogrid_gid_to_country_id.csv
```

### Shapefiles
You can download shapefiles that match to country_ids (these are not time-varying, so this is just an utility for easy plotting of recent data) and priogrid_gids.

```bash
shapefiles
├── countries.cpg
├── countries.csv
├── countries.dbf
├── countries.prj
├── countries.shp
├── countries.shx
├── priogrid.cpg
├── priogrid.dbf
├── priogrid.prj
├── priogrid.shp
└── priogrid.shx
```

### How to use
Read data (e.g., predictions or actuals, or any data structured as folder/{target}/{Apache Hive}:

```python
from pathlib import Path
from utilities import views_month_id_to_year, views_month_id_to_month, views_month_id_to_date, get_target_data, list_submissions

submissions = Path("path/to/submissions/folder")

# Read predictions
df = get_target_data(list_submissions(submissions)[0], "cm")
df = df.reset_index()

# Get date info from month_id
df["year"] = views_month_id_to_year(df["month_id"])
df["month"] = views_month_id_to_month(df["month_id"])
df["date"] = views_month_id_to_date(df["month_id"])
df
```

Evaluate data based on CRPS, IGN, MIS
```python
actuals = "path/to/actuals/folder"
bins = [0, 0.5, 2.5, 5.5, 10.5, 25.5, 50.5, 100.5, 250.5, 500.5, 1000.5]

# Get paths to submissions in a folder
submission_paths = list_submissions(submissions)

# Evaluate a single submission
evaluate_submission(submission_paths[0], actuals, targets = "cm", windows = ["Y2018", "Y2019", "Y2020", "Y2021"], expected = 1000, bins = bins)

# Evaluate a folder with many submissions
evaluate_all_submissions(submission_paths, actuals, targets = "cm", windows = ["Y2018", "Y2019", "Y2020", "Y2021"], expected = 1000, bins = bins)
```

Table evaluations
```python
tables = "path/to/where/to/save/tables"

# Collect summary data for all submissions and aggregate across dimensions
evaluation_table(submissions, target = "cm", groupby = ["window"], aggregate_submissions=False)

# Unit id, month_id, month, year, and window are all allowed. Windows will be pivoted into wide (might want to change this behavior)
evaluation_table(submissions, target = "cm", groupby = ["country_id"], aggregate_submissions=False)

# It is also possible to aggregate across submissions
evaluation_table(submissions, target = "cm", groupby = ["month", "window"], aggregate_submissions=True)

# You can write tables to LaTeX, HTML, and Excel format.
evaluation_table(submissions, target = "cm", groupby = ["month", "window"], aggregate_submissions=True, save_to=tables)
```

### Evaluation data
When evaluation metrics have been estimated, they are stored in an "eval" folder inside each submission folder in a long-format (i.e., metric being one of "crps", "ign" or "mis", and the value stored in a "value" column).

```bash
.
├── cm
├── pgm
├── eval
│   └── cm
│       ├── window=Y2018
│       │   ├── metric=crps
│       │   │   └── crps.parquet
│       │   ├── metric=ign
│       │   │   └── ign.parquet
│       │   └── metric=mis
│       │       └── mis.parquet
│       ├── window=Y2019
│       │   ├── metric=crps
│       │   │   └── crps.parquet
│       │   ├── metric=ign
│       │   │   └── ign.parquet
│       │   └── metric=mis
│       │       └── mis.parquet
│       ├── window=Y2020
│       │   ├── metric=crps
│       │   │   └── crps.parquet
│       │   ├── metric=ign
│       │   │   └── ign.parquet
│       │   └── metric=mis
│       │       └── mis.parquet
│       └── window=Y2021
│           ├── metric=crps
│           │   └── crps.parquet
│           ├── metric=ign
│           │   └── ign.parquet
│           └── metric=mis
│               └── mis.parquet
└── submission_details.yml
```

### Command-line tools
In addition to using the functions in a Python environment, some core functionality is available as command-line tools.

To estimate Poisson samples from point-predictions:

```console
python point-to-samples.py -s /path/to/submission/template/folder/with/point/predictions
```

To test compliance of submission with submission standards (will write to a test_compliance.log file in current working folder):

```console
python test_compliance.py -s /path/to/folder/containing/only/folders/like/submission_template -a /path/to/actuals
```

To estimate evaluation metrics (will also write a evaluate_submission.log in current working folder):

```console
python evaluate_submissions.py -s /path/to/folder/containing/only/folders/like/submission_template -a /path/to/actuals
```

To collate evaluation metrics:

```console
python collect_performance.py -s /path/to/folder/containing/only/folders/like/submission_template
```
This will result in four .parquet-files in the "path_to_submissions" folder with aggregated evaluation metrics per month and per unit at both the pgm and cm level. 

You can also get tables of global metrics in LaTeX, HTML, and Excel (this will also do the above step first):

```console
python collect_performance.py -s /path/to/folder/containing/only/folders/like/submission_template -t /path/to/folder/you/want/tables/in
```

## Monthly update
### 1. Update the actuals
Make sure the actuals are up-to-date. If you don't know where to get them, ask Jim.

### 2. Run the evaluation
The dashboard requires a different structure than the one we use for evaluation. To get the data in the right format, run the following command:

```console
python evaluate_submissions.py -s /path/to/folder/containing/only/folders/like/submission_template -a /path/to/actuals -r -st path/to/save
```
The new evaluation folder looks like this:
```bash
evaluation_folder_name
├── cm
│   ├── team_idenfitier_1.json
│   └── team_idenfitier_2.json
└── pgm
    ├── team_idenfitier_3
    │   ├── NGA_1.json
    │   └── NGA_2.json
    └── team_idenfitier_4
        ├── NGA_1.json
        └── NGA_2.json
```

### Optional: Clean the submissions
Before running the evaluation, you can also clean the submissions to make sure they are compliant with the correct format. The cleaned data will be saved to another folder to make sure the original data is not overwritten.
This is done by running the following command:

```console
clean_submissions.py -s /path/to/folder/containing/only/folders/like/submission_template -st /path/to/save
```