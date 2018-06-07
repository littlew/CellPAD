# Copyright (c) 2017 Wu Jun (littlewj187@gmail.com)

import numpy as np
from CellPAD.statsmodels.local_correlation_score import LCS
from CellPAD.statsmodels.smoothing_algorithms import WMA, EWMA, HW


class StatisticalMeasurement:
    """
    A class to call statistical measuring algorithm, e.g. local correlation tracking.
    """
    def __init__(self, algorithm_name):
        """
        It initiates the class of the corresponding algorithm .
        :param predictor: str, the tag of the algorithm's name.
        """
        if algorithm_name == "LCS":
            self.measure = LCS()
            self.algorithm_name = "LCS"

    def measure(self, series1, series2, labels, period_len=168):
        """
        It calls self.measure (LCS model) to measure the LCS score of the series 1 and series 2.
        LCS will skip the detected anomalies with "True" label when measures the current instance.
        :param series1: list<float>, the series 1.
        :param series2: list<float>, the series 2.
        :param labels: list<bool>,the labels of the previous detected points.
        :param period_len: int, the number of instance in one period.
        :return: list<float>, a list of the measuring results.
        """
        measure_score = self.measure.measure_current_period(series1, series2, labels, period_len)
        return measure_score


class StatisticalPredictor:
    """
    A class to call statistical smoothing prediction algorithm, e.g. WMA, EWMA, HW.
    """
    def __init__(self, algorithm_name):
        """
        It initiates the class of the corresponding algorithm.
        :param predictor: str | the tag of the algorithm's name.
        """
        if algorithm_name == "WMA":
            self.predictor = WMA()
            self.predictor_name = "WMA"
        if algorithm_name == "EWMA":
            self.predictor = EWMA()
            self.predictor_name = "EWMA"
        if algorithm_name == "HW":
            self.predictor = HW()
            self.predictor_name = "HW"

    def predict(self, series, labels, period_len):
        """
        It calls the self.predictor(a smoothing model) to predict the reasonable values.
        It will skip the detected anomalies with the "True" label.
        :param series: list<float> | the series
        :param labels: list<bool> | the labels of the previous detected points.
        :param period_len: int | the number of instance in one period.
        :return: list<float> | a list of the predicted results.
        """
        if self.algorithm_name == "WMA" or self.algorithm_name == "EWMA":
            predicted_series = self.predictor.predict_next_period(series, labels, period_len)
        if self.algorithm_name == "HW":
            predicted_series = self.predictor.predict_next_period(series=series,
                                                                  period_len=period_len,
                                                                  mtype='multiplicative')
        return predicted_series


class RegressionPredictor:
    """
    A class to call regression algorithm, e.g. RF, RT, SLR, HR
    """
    def __init__(self, algorithm_name):
        """
        It initiates the class of the corresponding algorithm.
        :param predictor: str, the tag of the algorithm's name.
        """
        if algorithm_name == 'RF':
            from sklearn.ensemble import RandomForestRegressor
            self.reg = RandomForestRegressor(n_estimators=100, criterion="mse")
        if algorithm_name == 'RT':
            from sklearn.tree import DecisionTreeRegressor
            self.reg = DecisionTreeRegressor(criterion="mse")
        if algorithm_name == 'SLR':
            from sklearn.linear_model import LinearRegression
            self.reg = LinearRegression()
        if algorithm_name == 'HR':
            from sklearn.linear_model import HuberRegressor
            self.reg = HuberRegressor(fit_intercept=True, alpha=1.35, max_iter=100)

    def train(self, train_features, train_response):
        """
        It makes the self.reg which is the regression model to fit using the features and the response variables.
        :param train_features: array<float> | the features.
        :param train_response: list<float> | the response variables.
        :return:
        """
        self.reg.fit(train_features, train_response)

    def predict(self, predict_features):
        """
        It calls the model(self.reg) to making prediction using the features.
        :param predict_features: array<float> | the features.
        :return: list<float> | the predicted response variables.
        """
        predicted_series = self.reg.predict(np.array(predict_features))
        return predicted_series
