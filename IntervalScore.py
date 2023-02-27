import numpy as np

# https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/types.py
from typing import List, Union
import xarray as xr

XArray = Union[xr.Dataset, xr.DataArray]
# XArray = xr.Dataset | xr.DataArray # raises error during build: TypeError: unsupported operand type(s) for |: 'ABCMeta' and 'type'
Dim = Union[List[str], str]
# Dim = List[str] | str

def interval_score(observed: np.array, predictions: np.array, prediction_interval_level: float = 0.95) -> np.array:
    """
    Interval Score implemented based on the scaled Mean Interval Score in the R tsRNN package https://rdrr.io/github/thfuchs/tsRNN/src/R/metrics_dist.R

    The Interval Score is a probabilistic prediction evaluation metric that weights between the narrowness of the forecast range and the ability to correctly hit the observed value within that interval.
    
    :param observed: observed values
    :type observed: array_like
    :param predictions: probabilistic predictions with the latter axis (-1) being the forecasts for each observed value
    :type predictions: array_like
    :param prediction_interval_level: prediction interval between [0, 1]
    :type prediction_interval_level: float
    :returns array_like with the interval score for each observed value
    :rtype array_like

    observed = np.random.negative_binomial(5, 0.8, size = 600)
    forecasts = np.random.negative_binomial(5, 0.8, size = (600, 1000))

    score = interval_score(observed, forecasts)
    print(f'MIS: {score.mean()}')

    """

    assert 0 < prediction_interval_level < 1, f"'prediction_interval_level' must be a number between 0 and 1." 

    alpha = 1 - prediction_interval_level
    lower = np.quantile(predictions, q = alpha/2, axis = -1)
    upper = np.quantile(predictions, q = 1 - (alpha/2), axis = -1)

    interval_width = upper - lower
    lower_coverage = (2/alpha)*(lower-observed) * (observed<lower)
    upper_coverage = (2/alpha)*(observed-upper) * (observed>upper)

    return(interval_width + lower_coverage + upper_coverage)

def _probabilistic_broadcast(
    observations: XArray, forecasts: XArray, member_dim: str = "member"
) -> XArray:
    """Broadcast dimension except for member_dim in forecasts."""
    observations = observations.broadcast_like(
        forecasts.isel({member_dim: 0}, drop=True)
    )
    forecasts = forecasts.broadcast_like(observations)
    return observations, forecasts

def mean_interval_score_xskillscore(
    observations: XArray,
    forecasts: XArray,
    member_weights: XArray = None,
    member_dim: str = "member",
    dim: Dim = None,
    keep_attrs: bool = False,
    **kwargs
) -> XArray:
    """Continuous Ranked Probability Score with the ensemble distribution.
    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations.
    forecasts : xarray.Dataset or xarray.DataArray
        Forecast with required member dimension ``member_dim``.
    member_dim : str, optional
        Name of ensemble member dimension. By default, 'member'.
    dim : str or list of str, optional
        Dimension over which to compute mean after computing ``ign_ensemble``.
        Defaults to None implying averaging over all dimensions.
    keep_attrs : bool
        If True, the attributes (attrs) will be copied
        from the first input to the new one.
        If False (default), the new object will
        be returned without attributes.
    Returns
    -------
    xarray.Dataset or xarray.DataArray
    """
    observations, forecasts = _probabilistic_broadcast(
        observations, forecasts, member_dim=member_dim
    )
    res = xr.apply_ufunc(
        interval_score,
        observations,
        forecasts,
        input_core_dims=[[], [member_dim]],
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    
    return res.mean(dim, keep_attrs=keep_attrs)



# Mean Interval Score when averaged across observations in the xskillscore implementation (return res.mean(dim, keep_attrs=keep_attrs)).