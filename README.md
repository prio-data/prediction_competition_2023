# Instructions

Here, you'll find a small collection of functions that work with prediction-data in the format specified by the ViEWS Prediction Challenge.

## Installation

1. Install [mamba](https://github.com/conda-forge/miniforge#mambaforge).

2. Add [conda-lock](https://github.com/conda/conda-lock) to the base environment.

``` console
$ mamba install --channel=conda-forge --name=base conda-lock
```

3. Install [git](https://git-scm.com/downloads).

4. Download our package from github:

```console
$ git clone https://github.com/prio-data/<project_name>
$ cd <project_name>
```
5. Create the virtual environment based on lock-file created from environment.yml

``` console
$ conda-lock install -n my_project_env  --mamba
$ mamba activate my_project_env
```
6. Run poetry to add additional python package requirements.

```console
(my_project_env) $ poetry install
```

## Useage

Assuming all submissions comply with the submission_template, the below functions should work. Note that these functions do take some time to finish if you are working on the PRIO-GRID level. A 64Gb RAM workstation with fast hard-drive is recommended, but they finish in reasonable time (10-20 minutes) on a laptop with 32Gb RAM.

To estimate Poisson samples from point-predictions:

```console
(my_project_env) $ python point-to-samples.py -s path/to/submission/template/folder/with/point/predictions
```

To estimate evaluation metrics:

```console
(my_project_env) $ python evaluate_submissions -s path_to_submissions -a path_to_actuals 
```

To collate evaluation metrics:

```console
(my_project_env) $ python collect_performance.py -s path_to_submissions
```

This will result in four .parquet-files in the "path_to_submissions" folder with aggregated evaluation metrics per month and per unit at both the pgm and cm level. 