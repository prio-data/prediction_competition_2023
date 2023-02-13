from __future__ import annotations
import numpy as np

# https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/types.py
from typing import List, Union
import xarray as xr

XArray = Union[xr.Dataset, xr.DataArray]
# XArray = xr.Dataset | xr.DataArray # raises error during build: TypeError: unsupported operand type(s) for |: 'ABCMeta' and 'type'
Dim = Union[List[str], str]
# Dim = List[str] | str

def _ensemble_ignorance_score(observations, forecasts, type=2, nmax=10000, ign_max=np.inf, round_values=False, axis=-1):
    """
    This implements the Ensemble (Ranked) Ignorance Score from the easyVerification R-package in Python. Also inspired by properscoring.crps_ensemble(),
    and has interface that works with the xskillscore package.


    Parameters
    ----------
    observations : float or array_like
        Observations float or array. Missing values (NaN) are given scores of
        NaN.
    forecasts : float or array_like
        Array of forecasts ensemble members, of the same shape as observations
        except for the axis along which RIGN is calculated (which should be the
        axis corresponding to the ensemble). If forecasts has the same shape as
        observations, the forecasts are treated as deterministic. Missing
        values (NaN) are ignored.
    type:
    nmax: max possible observation (setting this higher will result in poorer performance, but more accurate measurement).
    ign_max: if the observations are outside of the range of the forecast distribution, Ignorance Score is not well defined. Use this parameter to set a maximum score. If None, then use probability of the closest forecast member.
    round_values: converts input data to integers by rounding.
    axis : int, optional
    Axis in forecasts and weights which corresponds to different ensemble
    members, along which to calculate CRPS.


    Returns
    -------
    out : np.ndarray
        RIGN for each ensemble forecast against the observations.

    easyVerification::convert2prob
    function (x, prob = NULL, threshold = NULL, ref.ind = NULL, multi.model = FALSE)
    {
        stopifnot(is.vector(x) | is.matrix(x))
        stopifnot(any(!is.na(x)))
        if (is.null(prob) & is.null(threshold))
            return(x)
        if (!is.null(prob) & !is.null(threshold)) {
            stop("Both probability and absolute thresholds provided")
        }
        if (!is.null(prob)) {
            stopifnot(unlist(ref.ind) %in% 1:nrow(as.matrix(x)))
            threshold <- prob2thresh(x = x, prob = prob, ref.ind = ref.ind,
                multi.model = multi.model)
        }
        else {
            if (is.null(prob))
                threshold <- expandthresh(threshold, x)
        }
        nclass <- nrow(threshold) + 1
        xtmp <- array(apply(rep(x, each = nrow(threshold)) > threshold,
            -1, sum), dim(as.matrix(x))) + 1
        xout <- t(apply(xtmp, 1, tabulate, nbins = nclass))
        xout[apply(as.matrix(is.na(x)), 1, any), ] <- NA
        return(xout)
    }

    easyVerification::count2prob
    function (x, type = 3)
    {
        stopifnot(is.matrix(x))
        stopifnot(any(!is.na(x)))
        stopifnot(type %in% 1:6)
        is.wholenumber <- function(x, tol = .Machine$double.eps^0.5) ifelse(is.na(x),
            TRUE, abs(x - round(x)) < tol)
        rs <- rowSums(x)
        stopifnot(is.wholenumber(rs))
        if (isTRUE(all.equal(rs, round(rs/abs(rs))))) {
            xout <- x
        }
        else {
            a <- c(0, 0.3, 1/3, 1, 1/2, 2/5)[type]
            n <- rowSums(x) + 1
            xout <- (x + 1 - a)/(n + 1 - 2 * a)
        }
        return(xout)
    }

    easyVerification::EnsIgn
    function (ens, obs, type = 3, ...)
    {
        stopifnot(is.matrix(ens), is.matrix(obs), length(obs) ==
            length(ens))
        ens.prob <- count2prob(ens, type = type)
        ign <- -log2(ens.prob[as.logical(obs)])
        return(ign)
    }
    """
    assert type in [0, 1, 2, 3, 4, 5], f"Type must be integer between 0-5."


    if round_values:
        observations = np.asarray(observations, dtype=int)
        forecasts = np.asarray(forecasts, dtype=int)
    else:
        observations = np.asarray(observations)
        forecasts = np.asarray(forecasts)

    if axis != -1:
        forecasts = move_axis_to_end(forecasts, axis)

    if observations.shape not in [forecasts.shape, forecasts.shape[:-1]]:
        raise ValueError('observations and forecasts must have matching '
                        'shapes or matching shapes except along `axis=%s`'
                        % axis)

    assert forecasts.dtype == int,  f"Forecasts must be integers."
    assert observations.dtype == int,  f"Observations must be integers."

    assert np.all(forecasts >= 0), f"Forecasts must be positive integers."
    assert np.all(
        observations >= 0), f"Observations must be positive integers."

    assert np.any(
        observations > nmax) == False, f"Larger observed values than nmax. Please increase nmax."

    if observations.shape == forecasts.shape:
        # exact prediction yields 0 ign
        ign_score = np.array(observations != forecasts, dtype=float)
        ign_score[ign_score > 0] = ign_max  # wrong prediction yields ign_max
        return ign_score  # and we are done

    forecasts_categorical = np.eye(N=nmax + 1, dtype=bool)[forecasts]
    x = forecasts_categorical.sum(axis=1).T

    a = [0, 0.3, 1/3, 1, 1/2, 2/5][type]
    n = forecasts.shape[1]  # sum over ensembles

    probs = (x + 1 - a) / (n + 1 - a)

    ign_score = -np.log2(np.diag(probs[observations]))

    if(ign_max != None):
        forecast_range_overlap = (forecasts.max(axis=1) >= observations) & (
            forecasts.min(axis=1) <= observations)
        ign_score[forecast_range_overlap == False] = ign_max

    return ign_score

def _probabilistic_broadcast(
    observations: XArray, forecasts: XArray, member_dim: str = "member"
) -> XArray:
    """Broadcast dimension except for member_dim in forecasts."""
    observations = observations.broadcast_like(
        forecasts.isel({member_dim: 0}, drop=True)
    )
    forecasts = forecasts.broadcast_like(observations)
    return observations, forecasts

def ign_ensemble(
    observations: XArray,
    forecasts: XArray,
    member_weights: XArray = None,
    member_dim: str = "member",
    dim: Dim = None,
    type: int = 2, 
    nmax: int = 10000, 
    ign_max: int = np.inf, 
    round_values: bool = False,
    keep_attrs: bool = False,
) -> XArray:
    """Continuous Ranked Probability Score with the ensemble distribution.
    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The observations or set of observations.
    forecasts : xarray.Dataset or xarray.DataArray
        Forecast with required member dimension ``member_dim``.
    member_weights : xarray.Dataset or xarray.DataArray
        If provided, the CRPS is calculated exactly with the assigned
        probability weights to each forecast. Weights should be positive,
        but do not need to be normalized. By default, each forecast is
        weighted equally.
    issorted : bool, optional
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
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
        _ensemble_ignorance_score,
        observations,
        forecasts,
        input_core_dims=[[], [member_dim]],
        kwargs={"axis": -1, "type": 2, "nmax": 25000, "ign_max": np.inf, "round_values": True},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    
    return res.mean(dim, keep_attrs=keep_attrs)
