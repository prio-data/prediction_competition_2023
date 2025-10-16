# Ensemble Maker

This module provides a script (`make_ensemble.py`) for creating ensemble predictions from multiple model submissions. It supports different ensemble strategies (unweighted, weighted, and selected) and can process predictions for different targets and time windows.

## Purpose

The script combines probabilistic samples from multiple model submissions into ensemble predictions, supporting:
- Unweighted ensemble: equal sampling from all models
- Weighted ensemble: sampling weighted by inverse CRPS (Continuous Ranked Probability Score)
- Selected ensemble: sampling from top-performing models based on multiple metrics (CRPS, MIS, IGN)

## Usage

You can run the script from the command line:

```bash
python -m ensemble.make_ensemble -s <submissions_folder> -st <save_to_folder> [options]
```

### Required Arguments
- `-s <submissions_folder>`: Path to the folder containing model submissions
- `-st <save_to_folder>`: Path to the folder where the ensemble results will be saved

### Optional Arguments
- `-t <targets>`: List of targets to process (default: `cm pgm`)
- `-w <windows>`: List of time windows to process (default: `Y2018 Y2019 Y2020 Y2021 Y2022 Y2023 Y2024 Y2025`)
- `-es <expected_samples>`: Number of samples to draw for each ensemble (default: `1000`)
- `-we <weights>`: List of ensemble strategies to use (default: `unweighted weighted selected`)

### Example

```bash
python -m ensemble.make_ensemble \
    -s /data/processed/final_submissions_cleaned_May21 \
    -st /data/ensembles/ensemble_Jul10 \
    -t cm pgm \
    -we unweighted weighted selected
```

## Output Structure

The script will create ensemble prediction files in the specified `save_to` directory, organized by ensemble type, target, and window. For example:

```
<save_to>/
  unweighted/
    cm/
      window=Y2023/
        ensemble_unweighted.parquet
    pgm/
      window=Y2023/
        ensemble_unweighted.parquet
  weighted/
    ...
  selected/
    ...
```

## Metrics

For weighted and selected ensembles, the script uses evaluation metrics (CRPS, MIS, IGN) found in each submission's `eval` directory. If metrics are not already cached, they will be computed and saved as JSON files in the output directory.

## Logging

Logs are written to `logs/ensemble.log`.


## Notes

- The script uses parallel processing for efficiency.
- If you encounter issues with multiprocessing on MacOS, ensure you are using the `spawn` start method (set by default).
- For more details, see the docstrings in `make_ensemble.py`. 