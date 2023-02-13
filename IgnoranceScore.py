import numpy as np


def ensemble_ignorance_score(observations, forecasts, type=2, nmax=10000, ign_max=np.inf, round_values=False, axis=-1):
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
    assert type in [0, 1, 2, 3, 4, 5]


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
