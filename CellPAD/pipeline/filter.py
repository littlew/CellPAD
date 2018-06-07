# Copyright (c) 2017 Wu Jun (littlewj187@gmail.com)

import numpy as np


class DropAnomalyFilter:
    def __init__(self, rule='gauss', coef=2.5):
        """
        DropAnomalyFilter is a class for measuring the drop degree.
        DropAnomalyFilter implements three anomaly detection methods,
        - gauss: based on gauss distribution assumption, which filters the outliers with high drop ratios as anomalies.
        - threshold: the operator could set a threshold of drop ratio by his or her domain knowledge.
        - proportion: the operator could set an assumed proportion of anomalies.

        :param rule: str | the approach to detect and filter out anomalies.
        :param coef: float | the coefficient for a specific approach(rule).
        """
        if rule == 'gauss':
            self.rule = rule
            self.sigma = coef
        if rule == 'threshold':
            self.rule = rule
            self.threshold = coef
        if rule == 'proportion':
            self.rule = rule
            self.proportion = coef
        return

    def detect_anomaly(self, predicted_series, practical_series):
        """
        It calculates the drop ratio of each point by comparing the predicted value and practical value.
        Then it runs filter_anomaly() function to filter out the anomalies by the parameter "rule".
        :param predicted_series: the predicted values of a KPI series
        :param practical_series: the practical values of a KPI series
        :return: drop_ratios, drop_labels and drop_scores
        """
        drop_ratios = []
        for i in range(len(practical_series)):
            dp = (practical_series[i] - predicted_series[i]) / (predicted_series[i] + 1e-7)
            drop_ratios.append(dp)
        drop_scores = []
        for r in drop_ratios:
            if r < 0:
                drop_scores.append(-r)
            else:
                drop_scores.append(0.0)

        drop_labels = self.filter_anomaly(drop_ratios)
        return drop_ratios, drop_labels, drop_scores

    def filter_by_threshold(self, drop_scores, threshold):
        """
        It judges whether a point is an anomaly by comparing its drop score and the threshold.
        :param drop_scores: list<float> | a measure of predicted drop anomaly degree.
        :param threshold: float | the threshold to filter out anomalies.
        :return: list<bool> | a list of labels where a point with a "true" label is an anomaly.
        """
        drop_labels = []
        for r in drop_scores:
            if r < threshold:
                drop_labels.append(True)
            else:
                drop_labels.append(False)
        return drop_labels

    def filter_anomaly(self, drop_ratios):
        """
        It calculates the threshold for different approach(rule) and then calls filter_by_threshold().
        - gauss: threshold = mean - self.sigma * std
        - threshold: the given threshold variable
        - proportion: threshold = sort_scores[threshold_index]
        :param drop_ratios: list<float> | a measure of predicted drop anomaly degree
        :return: list<bool> | the drop labels
        """
        if self.rule == 'gauss':
            mean = np.mean(drop_ratios)
            std = np.std(drop_ratios)
            threshold = mean - self.sigma * std
            drop_labels = self.filter_by_threshold(drop_ratios, threshold)
            return drop_labels

        if self.rule == "threshold":
            threshold = self.threshold
            drop_labels = self.filter_by_threshold(drop_ratios, threshold)
            return drop_labels

        if self.rule == "proportion":
            sort_scores = sorted(np.array(drop_ratios))
            threshold_index = int(len(drop_ratios) * self.proportion)
            threshold = sort_scores[threshold_index]
            drop_labels = self.filter_by_threshold(drop_ratios, threshold)
            return drop_labels


class ChangeAnomalyFilter:
    def __init__(self, rule='gauss', coef=3.0):
        """
        ChangeAnomalyFilter is a class for measuring the change degree.
        ChangeAnomalyFilter implements three anomaly detection methods,
        - gauss: based on gauss distribution assumption, which filters the outliers with high change ratios as anomalies.
        - threshold: the operator could set a threshold of change ratio by his or her domain knowledge.
        - proportion: the operator could set an assumed proportion of anomalies.
        :param rule: str | the approach to detect and filter out anomalies.
        :param coef: float | the coefficient for a specific approach(rule).
        """
        if rule == 'gauss':
            self.rule = rule
            self.sigma = coef
        if rule == 'threshold':
            self.rule = rule
            self.threshold = coef
        if rule == 'proportion':
            self.rule = rule
            self.proportion = coef
        return

    def detect_anomaly_lcs(self, lcs_scores):
        """
        It detects the anomalies which are measured by local correlation tracking method.
        - gauss: threshold = 0.0 + self.sigma * std
        - threshold: the given threshold variable
        - proportion: threshold = sort_scores[threshold_index]
        :param lcs_scores: list<float> | the list of local correlation scores
        :return:
        """
        if self.rule == "gauss":
            mean = 0.0
            std = np.std(lcs_scores)
            threshold = mean + self.sigma * std
            change_labels = []
            for lcs in range(len(lcs_scores)):
                if lcs > threshold:
                    change_labels.append(True)
                else:
                    change_labels.append(False)
            return change_labels, lcs_scores
        if self.rule == "threshold":
            threshold = self.threshold
            change_labels = []
            for lcs in range(len(lcs_scores)):
                if lcs > threshold:
                    change_labels.append(True)
                else:
                    change_labels.append(False)
            return change_labels, lcs_scores
        if self.rule == "proportion":
            sort_scores = sorted(np.array(lcs_scores))
            threshold_index = int(len(lcs_scores) * (1.0 - self.proportion))
            threshold = sort_scores[threshold_index]
            change_labels = []
            for lcs in range(len(lcs_scores)):
                if lcs > threshold:
                    change_labels.append(True)
                else:
                    change_labels.append(False)
            return change_labels, lcs_scores

    def detect_anomaly_regression(self, predicted_series1, practical_series1, predicted_series2, practical_series2):
        """
        It calculates the drop ratio of each point by comparing the predicted value and practical value.
        Then it runs filter_anomaly() function to filter out the anomalies by the parameter "rule".
        :param predicted_series1: list<float> | the predicted values of the KPI series 1.
        :param practical_series1: list<float> | the practical values of the KPI series 1.
        :param predicted_series2: list<float> | the predicted values of the KPI series 2.
        :param practical_series2: list<float> | the practical values of the KPI series 2.
        :return:
        """
        change_ratios1 = []
        change_ratios2 = []
        change_scores = []
        for i in range(len(practical_series1)):
            c1 = (practical_series1[i] - predicted_series1[i]) / (predicted_series1[i] + 1e-7)
            c2 = (practical_series2[i] - predicted_series2[i]) / (predicted_series2[i] + 1e-7)
            change_ratios1.append(c1)
            change_ratios2.append(c2)
            s = (abs(c1) + abs(c2)) / 2.0
            change_scores.append(s)

        change_labels = self.filter_anomaly(change_ratios1, change_ratios2, change_scores)
        return change_ratios1, change_ratios2, change_labels, change_scores

    def filter_by_threshold(self, change_ratios, threshold1, threshold2):
        """
        It filter out the too deviated points as anomalies.
        :param change_ratios: list<float> | the change ratios.
        :param threshold1: float | the negative threshold standing for a drop deviation.
        :param threshold2: float | the positive threshold standing for a rise deviation.
        :return: list<bool> | the list of the labels where "True" stands for an anomaly.
        """
        change_labels = []
        for r in change_ratios:
            if r < threshold1 or r > threshold2:
                change_labels.append(True)
            else:
                change_labels.append(False)
        return change_labels

    def filter_anomaly(self, change_ratios1, change_ratios2, change_scores):
        """
        It detects the anomalies which are measured by regression method.
        - gauss: threshold1 = mean - self.sigma * std, threshold2 = mean + self.sigma * std
        - threshold: the given threshold variable
        - proportion: threshold = sort_scores[threshold_index]
        :param change_ratios1: list<float> | the change ratios of the KPI1.
        :param change_ratios2: list<float> | the change ratios of the KPI2.
        :param change_scores: list<float> | the average of the change anomaly degree of the two change ratios.
        :return: list<bool> | the list of the labels where "True" stands for an anomaly.
        """
        if self.rule == 'gauss':
            mean = np.mean(change_ratios1)
            std = np.std(change_ratios1)
            threshold1 = mean - self.sigma * std
            threshold2 = mean + self.sigma * std
            change_labels1 = self.filter_by_threshold(change_ratios1, threshold1, threshold2)
            mean = np.mean(change_ratios2)
            std = np.std(change_ratios2)
            threshold1 = mean - self.sigma * std
            threshold2 = mean + self.sigma * std
            change_labels2 = self.filter_by_threshold(change_ratios2, threshold1, threshold2)
            change_labels = list(np.array(change_labels1) + np.array(change_labels2))
            return change_labels

        if self.rule == "threshold":
            threshold = self.threshold
            change_labels1 = self.filter_by_threshold(change_ratios1, -threshold, threshold)
            change_labels2 = self.filter_by_threshold(change_ratios2, -threshold, threshold)
            change_labels = list(np.array(change_labels1) + np.array(change_labels2))
            return change_labels

        if self.rule == "proportion":
            sort_scores = sorted(np.array(change_scores))
            threshold_index = int(len(change_scores) * (1.0 - self.proportion))
            threshold = sort_scores[threshold_index]
            change_labels = []
            for i in range(len(change_scores)):
                if change_scores[i] > threshold:
                    change_labels.append(True)
                else:
                    change_labels.append(False)
            return change_labels
