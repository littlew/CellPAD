# Copyright (c) 2017 Wu Jun (littlewj187@gmail.com)

from CellPAD.statsmodels.smoothing_algorithms import EWMA, WMA
import numpy as np
import pandas as pd


class FeatureTools:
    def __init__(self):
        """
        It initiates the windows size parameters when it calculates the numerical features.
        """
        self.hourly_wins = [2, 5, 10, 17, 24]
        self.daily_wins = [2, 5, 7, 10, 15]
        self.weekly_wins = [3, 5, 7, 10, 13]

    def set_window_size(self, hourly_wins, daily_wins, weekly_wins):
        """
        It provides a interface to set the windows size parameters.
        :param hourly_wins: list<int> | a list of the count of the hours.
        :param daily_wins: list<int> |  a list of the count of the days.
        :param weekly_wins: list<int> |  a list of the count of the weeks.
        :return:
        """
        self.hourly_wins = hourly_wins
        self.daily_wins = daily_wins
        self.weekly_wins = weekly_wins

    def set_feature_names(self, feature_types, feature_time_grain, feature_operations):
        """
        It reads the Indexical and Numerical features.
        :param feature_types: list<str> | a set of {'Indexical', 'Numerical'}
        :param feature_time_grain: str | the time grain of the numerical features.
        :param feature_operations: str | the identification of the operations of the numerical features.
        :return: list<str> | a list of feature names e.g. '3_Daily_Mean',
                             means the feature is the average of the previous 3 days.
        """
        feature_list = []
        if "Indexical" in feature_types:
            if "Weekly" in feature_time_grain:
                feature_list.append("Hour")
                feature_list.append("Day")
            if "Daily" in feature_time_grain:
                feature_list.append("Hour")

        if "Numerical" in feature_types:
            for operation in feature_operations:
                if operation == "Raw":
                    feature_list.append("Raw")
                    continue
                if "Hourly" in feature_time_grain:
                    for win in self.hourly_wins:
                        feature_name = "%d_%s_%s" % (win, "Hourly", operation)
                        feature_list.append(feature_name)
                if "Daily" in feature_time_grain:
                    for win in self.daily_wins:
                        feature_name = "%d_%s_%s" % (win, "Daily", operation)
                        feature_list.append(feature_name)
                if "Weekly" in feature_time_grain:
                    for win in self.weekly_wins:
                        feature_name = "%d_%s_%s" % (win, "Weekly", operation)
                        feature_list.append(feature_name)
        return feature_list

    def compute_feature_matrix(self,
                               timestamps, series, labels,
                               ts_period_len, feature_list,
                               start_pos, end_pos):
        """
        It computes and returns the feature matrix.
        :param timestamps: list<str> | the time stamps of the time series.
        :param series: list<float> | the values of the time series.
        :param labels: list<bool> | the labels of the time series.
        :param ts_period_len: int | the number of the instances in one period.
        :param feature_list: list<str> | the list of the feature names.
        :param start_pos: int | the first index of the current iteration.
        :param end_pos: int | the last index of the current iteration.
        :return: array<float> | a two-dimension array storing the feature matrix.
        """
        extractor = FeatureExtractor(timestamps, series, labels,
                                     ts_period_len, feature_list)
        features = extractor.compute_features(start_pos, end_pos)
        return features


class FeatureExtractor:
    def __init__(self, timestamps, series, labels, ts_period_len, feature_list):
        """
        It initiates the object by the variables below.
        :param timestamps: list<str> | the time stamps of the time series.
        :param series: list<float> | the values of the time series.
        :param labels: list<bool> | the labels of the time series.
        :param ts_period_len: int | the number of the instances in one period.
        :param feature_list: list<str> | the list of the feature names.
        """
        self.timestamps = pd.to_datetime(timestamps)
        self.series = series
        self.labels = labels
        self.ts_period_len = ts_period_len
        self.feature_list = feature_list

    def compute_feature_period_len(self, period_grain):
        """
        It computes the period length of different numerical features.
        :param period_grain: str | the identification, which is 'Hourly', 'Daily' or 'Weekly'.
        :return: int | the count of instances corresponding to the period_grain.
        """
        time_delta = self.timestamps[1] - self.timestamps[0]
        hourly_time_delta = pd.to_datetime("2018/1/9 1:00:00") - pd.to_datetime("2018/1/9 0:00:00")
        daily_time_delta = pd.to_datetime("2018/1/9 0:00:00") - pd.to_datetime("2018/1/8 0:00:00")
        weekly_time_delta = pd.to_datetime("2018/1/15 0:00:00") - pd.to_datetime("2018/1/8 0:00:00")
        if period_grain == "Hourly":
            return int(hourly_time_delta / time_delta)
        if period_grain == "Daily":
            return int(daily_time_delta / time_delta)
        if period_grain == "Weekly":
            return int(weekly_time_delta / time_delta)

    def get_sametime_instances(self, current_index, feature_period_len,
                                     ts_period_len, instance_count):
        """
        It results the list of instances occurred in the same time of the current point.
        It will skip the anomalies which are have been detected.
        :param current_index: int | the index of the current point(instance).
        :param feature_period_len: int | the interval size. In hourly basis time series, it is 24 for the 'hourly' feature.
        :param ts_period_len: int | the weekly period length of the time series.
        :param instance_count: int | the count of the same-time instances.
        :return:
        """
        series = self.series
        labels = list(self.labels) + list([False] * ts_period_len)
        ret_series = []
        pos = current_index - ts_period_len
        cnt = 0

        while pos >= 0 and cnt < instance_count:
            if not labels[pos]:
                ret_series.append(series[pos])
            pos = pos - feature_period_len
            cnt = cnt + 1

        if len(ret_series) == 0:
            return [0.0]
        else:
            return ret_series

    def compute_one_feature(self, feature_name, start_pos, end_pos):
        """
        It computes and returns a vector of one feature.
        :param feature_name: str | the name of the feature.
        :param start_pos: int | the start index of the current iteration(week).
        :param end_pos: int | the end index of the current iteration(week).
        :return: a list<float> | the feature vector.
        """
        series = self.series
        timestamps = self.timestamps
        feature_values = []

        # Indexical features
        if feature_name == "Hour":
            for i in range(start_pos, end_pos):
                feature_values.append(timestamps[i].hour)
            return feature_values
        if feature_name == "Day":
            for i in range(start_pos, end_pos):
                feature_values.append(timestamps[i].dayofweek)
            return feature_values

        # numerical features: KPI values
        if feature_name == "Raw":
            for i in range(start_pos, end_pos):
                feature_values.append(series[i])
            return feature_values

        # numerical features: <window, operator>
        win, period_grain, operation = feature_name.split("_")
        win = int(win)

        # calculate the features for each instance from the start_pos to end_pos
        for idx in range(start_pos, end_pos):
            feature_period_len = self.compute_feature_period_len(period_grain)
            vs = self.get_sametime_instances(current_index=idx,
                                             feature_period_len=feature_period_len,
                                             ts_period_len=self.ts_period_len,
                                             instance_count=win)
            if operation == "Mean":
                value = np.mean(vs)
            if operation == "Median":
                value = np.median(vs)
            if operation == "Wma":
                operator = WMA()
                value = operator.wma(vs)
            if operation == "Ewma":
                operator = EWMA()
                value = operator.ewma(vs)
            feature_values.append(value)

        return feature_values

    def compute_features(self, start_pos, end_pos):
        """
        It computes all the feature vectors in a batch manner, and returns the feature matrix.
        :param start_pos: int | the start index of the current iteration(week).
        :param end_pos: int | the end index of the current iteration(week).
        :return: array<float> | a two-dimension array storing the feature matrix.
        """
        feature_matrix = pd.DataFrame()
        for feature_name in self.feature_list:
            feature_values = np.array(self.compute_one_feature(feature_name, start_pos, end_pos))
            if len(feature_values) == 0:
                continue
            feature_matrix[feature_name] = feature_values
        return np.array(feature_matrix)

