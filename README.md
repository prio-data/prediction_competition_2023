# Instructions

Here, you'll find a small collection of functions that work with prediction-data in the format specified by the ViEWS Prediction Challenge.

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

Assuming all submissions comply with the submission_template, the below functions should work. Note that these functions do take some time to finish if you are working on the PRIO-GRID level. A 64Gb RAM workstation with fast hard-drive is recommended, but they finish in reasonable time (10-20 minutes) on a laptop with 32Gb RAM.

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

