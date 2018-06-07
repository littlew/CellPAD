# Copyright (c) 2017 Wu Jun (littlewj187@gmail.com)

import numpy as np
from CellPAD.statsmodels.holt_winters import HoltWinters


class HW:
    def __init__(self):
        pass

    def predict_next_period(self, series, period_len=168, mtype='multiplicative'):
        """
        It predict the next period of time series using the holt-winter algorithm.
        :param series: list<float> | the historical time series.
        :param period_len: int | the period length.
        :param mtype: str | 'multiplicative' or 'additive'
        :return: list<float> | the predicted series for the next period.
        """
        holt = HoltWinters()
        predicted_series = \
            holt.predict_next_period(ts=series, p=period_len, sp=2 * period_len,
                                     ahead=period_len, mtype=mtype)
        return np.array(predicted_series)


class WMA:
    def __init__(self):
        pass

    def wma(self, series):
        """
        It implements a Weighted Moving Average.
        :param series: list<float> | the series.
        :return: float | the predicted value of the current instance.
        """
        non_len = len(series)
        sum_v = 0.0
        sum_weight = 0.0
        for i in range(non_len):
            weight = i + 1.0
            sum_v += weight * series[i]
            sum_weight += weight
        return sum_v / sum_weight

    def predict_next_period(self, series, labels, period_len):
        """
        It predict the next period of time series using the holt-winter algorithm.
        It skips the detected anomalies.
        :param series: list<float> | the historical time series.
        :param labels: list<bool> | the labels of the historical data.
        :param period_len: int | the period length.
        :return: list<float> | the predicted series for the next period.
        """
        predicted_series = []
        for i in range(period_len):
            vs = []
            for j in range(i, len(series), period_len):
                if not labels[j]:
                    vs.append(series[j])
            pre = self.wma(vs)
            predicted_series.append(pre)
        return np.array(predicted_series)


class EWMA:
    def __init__(self):
        pass

    def ewma(self, series, beta=0.8):
        """
        It implements a Exponential Weighted Moving Average.
        :param series: list<float> | the series.
        :return: float | the predicted value of the current instance.
        """
        non_len = len(series)
        sum_v = 0.0
        sum_weight = 0.0
        for i in range(non_len):
            weight = beta ** (non_len - i - 1)
            sum_v += weight * series[i]
            sum_weight += weight
        return sum_v / sum_weight

    def predict_next_period(self, series, labels, period_len):
        """
       It predict the next period of time series using the holt-winter algorithm.
       It skips the detected anomalies.
       :param series: list<float> | the historical time series.
       :param labels: list<bool> | the labels of the historical data.
       :param period_len: int | the period length.
       :return: list<float> | the predicted series for the next period.
       """
        predict_series = []
        for i in range(period_len):
            vs = []
            for j in range(i, len(series), period_len):
                if not labels[j]:
                    vs.append(series[j])
            pre = self.ewma(vs)
            predict_series.append(pre)
        return np.array(predict_series)
