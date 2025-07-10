import numpy as np


def sample_unweighted(model_samples, expected_samples, month, id):
    """
    Sample by drawing from each model equally.
    """
    model_arrays = [
        model_samples[m][(month, id)]
        for m in model_samples if (month, id) in model_samples[m]
    ]
    num_models = len(model_arrays)
    if num_models == 0:
        raise ValueError(f"No models found for month={month}, id={id}")
    draws = np.full(num_models, expected_samples // num_models, dtype=int)
    draws[:expected_samples % num_models] += 1
    all_samples = np.empty(expected_samples, dtype=np.float32)
    start = 0
    np.random.seed(42)
    for arr, n_draw in zip(model_arrays, draws):
        all_samples[start:start+n_draw] = np.random.choice(arr, size=n_draw, replace=True)
        start += n_draw
    return all_samples


def sample_weighted(model_samples, expected_samples, month, id, crps_dict):
    """
    Sample by weighting the models by the inverse of their CRPS.
    1. Compute the average CRPS_m for all the test years 2019-2024 for each model m
    2. Draw from each model proportional to 1/CRPS_m for each 
    """
    model_names = [m for m in model_samples if (month, id) in model_samples[m]]
    crps_values = np.array([crps_dict[m] for m in model_names])
    inv_crps = 1.0 / crps_values
    weights = inv_crps / inv_crps.sum()

    # Draw from each model proportional to 1/CRPS_m for each 
    raw_draws = weights * expected_samples
    draws = np.floor(raw_draws).astype(int) # Round down to the nearest integer
    remainder = expected_samples - draws.sum()
    fractional = raw_draws - draws 
    for i in np.argsort(fractional)[-remainder:]:
        draws[i] += 1 

    all_samples = np.empty(expected_samples, dtype=np.float32)
    start = 0
    np.random.seed(42)
    for m, n_draw in zip(model_names, draws):
        arr = model_samples[m][(month, id)]
        all_samples[start:start+n_draw] = np.random.choice(arr, size=n_draw, replace=True)
        start += n_draw
    return all_samples


def sample_selected(model_samples, expected_samples, month, id, metrics_dict):
    """
    Sample from the top models for a given month and id.
    1. Compute the average CRPS, MIS, IGN scores for all the test years 2019-24. 
    2. Rank all the models for each of these metrics. Include all models that is in the top 3 of any of these 3 lists. This will result in 3-9 models. 
    3. Draw samples from these models so that we have 1,000 in total - no weighting for these draws within this set of top-performing models.
    """
    top_models = set()
    for metric in ['crps', 'mis', 'ign']:
        sorted_models = sorted(metrics_dict[metric], key=metrics_dict[metric].get)
        top_models.update(sorted_models[:3])
    top_models = [m for m in top_models if (month, id) in model_samples[m]]
    num_models = len(top_models)
    if num_models == 0:
        raise ValueError(f"No top models found for month={month}, id={id}")

    draws = np.full(num_models, expected_samples // num_models, dtype=int)
    draws[:expected_samples % num_models] += 1

    all_samples = np.empty(expected_samples, dtype=np.float32)
    start = 0
    np.random.seed(42)
    for m, n_draw in zip(top_models, draws):
        arr = model_samples[m][(month, id)]
        all_samples[start:start+n_draw] = np.random.choice(arr, size=n_draw, replace=True)
        start += n_draw
    return all_samples