# Copyright (c) 2017 Wu Jun (littlewj187@gmail.com)

from CellPAD.statsmodels.timeseries_decomposition import remove_trend as remove_center_mean
import numpy as np


class Preprocessor:
    def __index__(self):
        pass

    def remove_trend(self, series, period_len, method, model="multiplicative"):
        """
        :param series: the raw KPI series.
        :param period_len: the duration of (the count of data points in) one period.
        :param method: "center_mean", the trend at time i is the mean of the points in [i-84,i+83].
                       "past_mean", the trend at time i is the mean of the points in [i-167,i].
        :return: series / trends
        """
        if method == "center_mean":
            return remove_center_mean(x=series, freq=period_len, model=model)
        if method == "past_mean":
            trends = []
            for i in range(len(series)):
                if i < period_len:
                    t = np.mean(series[:i+1]) if np.mean(series[:i+1]) > 0.0 else 1e-7
                    trends.append(t)
                else:
                    t = np.mean(series[i-period_len+1:i+1]) if np.mean(series[i-period_len+1:i+1]) > 0.0 else 1e-7
                    trends.append(t)
            if model.startswith('m'):
                detrended = series / trends
            else:
                detrended = series - trends
            return detrended