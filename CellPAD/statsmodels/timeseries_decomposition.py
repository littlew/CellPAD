# Copyright (c) 2017 Wu Jun (littlewj187@gmail.com)

from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np


def remove_trend(x, model="multiplicative", freq=None):
    """
    It remove the trend components from the time series.
    :param x: list<float>, the time series.
    :param model: str | the identification of the model type, which is 'multiplicative' or 'additive'
    :param freq: int | the duration of the period.
    :return: the decomposed series with the trend components.
    """
    x = np.asanyarray(x).squeeze()

    if not np.all(np.isfinite(x)):
        raise ValueError("This function does not handle missing values")
    if model.startswith('m'):
        if np.any(x <= 0):
            for i in range(len(x)):
                if x[i] <= 0:
                    x[i] = 1e-7

    decompose_results = seasonal_decompose(x=x, model=model, freq=freq)
    trend = decompose_results.trend
    
    the_first = 0
    the_last = 0
    for i in range(len(trend)):
        if np.isnan(trend[i]) and (not np.isnan(trend[i+1])):
            the_first = trend[i+1]
        if not np.isnan(trend[i]) and np.isnan(trend[i+1]):
            the_last = trend[i]
            break
    for i in range(len(trend)):
        if np.isnan(trend[i]):
            if i < len(trend) // 2:
                trend[i] = the_first
            else:
                trend[i] = the_last
    
    if model.startswith('m'):
        detrended = x / trend
    else:
        detrended = x - trend

    return detrended
