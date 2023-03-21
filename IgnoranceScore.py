from __future__ import annotations
import numpy as np
from collections import Counter

# https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/types.py
from typing import List, Union
import xarray as xr

XArray = Union[xr.Dataset, xr.DataArray]
# XArray = xr.Dataset | xr.DataArray # raises error during build: TypeError: unsupported operand type(s) for |: 'ABCMeta' and 'type'
Dim = Union[List[str], str]
# Dim = List[str] | str

def _ensemble_ignorance_score_old(predictions, n, prob_type, observed):
        c = Counter(predictions)
        # n = c.total() : this works from python version 3.10, avoid this for a while.
        a = [0, 0.3, 1/3, 1, 1/2, 2/5][prob_type]
        prob = (c[observed] + 1 - a) / (n + 1 - a) # if counter[observed] is 0, then this returns correctly
        return -np.log2(prob)

def _ensemble_ignorance_score(predictions, n, observed):
        c = Counter(predictions)
        # n = c.total() : this works from python version 3.10, avoid this for a while.
        prob = c[observed] / n # if counter[observed] is 0, then this returns correctly
        return -np.log2(prob)

def _ensemble_ignorance_score_interpolate(predictions, n, observed):
    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    if observed > predictions.max():
        prob = 0
    else:
        c = Counter(predictions)
        probs = np.array([c[i]/n for i in np.arange(predictions.max() + 1)])
        probs[probs == 0] = np.NaN
        if(predictions.min() > 0): # Do not interpolate outside of the prediction sample range
            probs[0:predictions.min()] = 0
        
        nans, x = nan_helper(probs)
        probs[nans]= np.interp(x(nans), x(~nans), probs[~nans]) # Linear interpolation of probabilities within sample range
        
        prob = probs[observed]
    return -np.log2(prob) 

def ensemble_ignorance_score(observations, forecasts, bins, low_bin = 0, high_bin = 10000):
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
    round_values: converts input data to integers by rounding.
    

    Returns
    -------
    out : np.ndarray
        RIGN for each ensemble forecast against the observations.
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)

    assert np.all(forecasts >= 0), f"Forecasts must be non-negative."
    assert np.all(observations >= 0), f"Observations must be non-negative."

    assert isinstance(bins, (int, list)), f"bins must be an integer or a list with floats"
    if isinstance(bins, int):
        assert bins > 0, f"bins must be an integer above 0 or a list with floats."

    def digitize_minus_one(x, bins, right=False):
        return np.digitize(x, bins, right) - 1

    edges = np.histogram_bin_edges(forecasts[..., :], bins = bins, range = (low_bin, high_bin))
    binned_forecasts =  np.apply_along_axis(digitize_minus_one, axis = 1, arr = forecasts, bins = edges)
    binned_observations = digitize_minus_one(observations, edges)

    # Append one observation in each bin-category to the forecasts to prevent 0 probability occuring.
    unique_categories = np.arange(0, len(bins))
    binned_forecasts = np.concatenate((binned_forecasts, np.tile(unique_categories, binned_forecasts.shape[:-1] + (1,))), axis = -1)
    
    n = binned_forecasts.shape[-1]

    #if observations.shape == forecasts.shape:
        # exact prediction yields 0 ign
    ign_score = np.empty_like(binned_observations, dtype = float)
    for index in np.ndindex(ign_score.shape):
        ign_score[index] = _ensemble_ignorance_score(binned_forecasts[index], n, binned_observations[index])
    
    
    return ign_score



def ensemble_ignorance_score_old(observations, forecasts, prob_type = 2, ign_max = None, round_values = False, axis = -1, bins = None, low_bin = 0, high_bin = 10000):
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
    prob_type:
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
    assert prob_type in [0, 1, 2, 3, 4, 5], f"prob_type must be integer between 0-5."


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

    assert np.issubdtype(forecasts.dtype, np.integer),  f"Forecasts must be integers."
    assert np.issubdtype(observations.dtype, np.integer),  f"Observations must be integers."

    assert np.all(forecasts >= 0), f"Forecasts must be positive integers."
    assert np.all(observations >= 0), f"Observations must be positive integers."

    if observations.shape == forecasts.shape:
        # exact prediction yields 0 ign
        ign_score = np.array(observations != forecasts, dtype=float)
        if ign_max == None:
            ign_score[ign_score > 0] = np.inf  # wrong prediction yields the maximum error
        else:    
            ign_score[ign_score > 0] = ign_max  # wrong prediction yields the user defined maximum error
        return ign_score  # and we are done

    
    n = forecasts.shape[-1]

    if bins != None:
        assert isinstance(bins, (int, list)), f"bins must be an integer or a list with floats"
        if isinstance(bins, int):
            assert bins > 0, f"bins must be an integer above 0."

        def digitize_minus_one(x, bins, right=False):
            return np.digitize(x, bins, right) - 1

        edges = np.histogram_bin_edges(forecasts[..., :], bins = bins, range = (low_bin, high_bin))

        binned_forecasts =  np.apply_along_axis(digitize_minus_one, axis = 1, arr = forecasts, bins = edges)
        #prediction_counts = [(Counter(binned_forecasts[..., :])) for i in range(0, binned_forecasts.shape[0], 1)] # count unique predictions

        edges = np.histogram_bin_edges(forecasts[0, :], bins = bins, range = (low_bin, high_bin))
        binned_observations = digitize_minus_one(observations, edges)

        ign_score = np.empty_like(binned_observations, dtype = float)
        for index in np.ndindex(ign_score.shape):
            if (ign_max != None) & (not binned_forecasts[index].min() >= binned_observations[index] >= binned_forecasts[index].max()):
                ign_score[index] = ign_max
            else:
                #ign_score[index] = _ensemble_ignorance_score(binned_forecasts[index], n, prob_type, binned_observations[index])
                ign_score[index] = _ensemble_ignorance_score_interpolate(binned_forecasts[index], n, binned_observations[index])
            
        #ign_score = [_ensemble_ignorance_score(counter, n, prob_type, binned_observations[i]) for i, counter in enumerate(prediction_counts)]
    else:
        ign_score = np.empty_like(observations, dtype = float)
        for index in np.ndindex(ign_score.shape):
            if (ign_max != None) & (not forecasts[index].min() >= observations[index] >= forecasts[index].max()):
                ign_score[index] = ign_max
            else:
                #ign_score[index] = _ensemble_ignorance_score(forecasts[index], n, prob_type, observations[index])
                ign_score[index] = _ensemble_ignorance_score_interpolate(binned_forecasts[index], n, binned_observations[index])
            

    #ign_score = [_ensemble_ignorance_score(counter, n, prob_type, observations[i]) for i, counter in enumerate(prediction_counts)]
    #ign_score = np.array(ign_score, dtype=float)
    
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

def ensemble_ignorance_score_xskillscore(
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
        ensemble_ignorance_score,
        observations,
        forecasts,
        input_core_dims=[[], [member_dim]],
        kwargs=kwargs,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    
    return res.mean(dim, keep_attrs=keep_attrs)
