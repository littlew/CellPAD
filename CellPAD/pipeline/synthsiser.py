# Copyright (c) 2017 Wu Jun (littlewj187@gmail.com)

import numpy as np
import copy


class DropSynthesiser:

    def __init__(self, raw_series, period_len=168):
        """
        For injecting sudden drops into the raw time series and output the labels and synthesized series;

        We inject isolated drop points and short-lived continuous
        degradations into the raw KPI time series;

        Finally, we apply a rule-based method to filter out the obvious anomalies.

        :param raw_series: list<float> | the raw KPI time series;
        :param period_len: int | the period of the time series,
                                 e.g. 168 for a weekly period and 24 for a daily period in one hour basis.
        """

        self.period_len = period_len
        self.syn_series = copy.deepcopy(raw_series)
        self.instance_cnt = len(raw_series)
        self.point_labels = np.array([False] * self.instance_cnt)
        self.segment_labels = np.array([False] * self.instance_cnt)
        self.rule_labels = np.array([False] * self.instance_cnt)
        self.syn_labels = np.array([False] * self.instance_cnt)

    def add_point_anomalies(self, anomaly_fraction, lowest_drop_ratio):
        """
        It inject the single anomalies(called point anomalies) into the raw data.
        :param anomaly_fraction: float | the fraction of the anomalies.
        :param lowest_drop_ratio: float | the lowest drop ratio.
        :return:
        """
        import random
        random.seed(1)

        point_positions = np.random.uniform(0.0, 1.0-1e-7, int(self.instance_cnt * anomaly_fraction))
        index_positions = []
        for pos in point_positions:
            index = int(pos * self.instance_cnt)
            index_positions.append(index)

        for idx in index_positions:
            drop_ratio = np.random.uniform(lowest_drop_ratio,1.0)
            self.syn_series[idx] = self.syn_series[idx] * (1 - drop_ratio)
            self.point_labels[idx] = True

    def add_segment_anomalies(self, segment_cnt, lowest_drop_ratio, shortest_period, longest_period):
        """
        It inject several continous anomalies(called segment anomalies) into the raw data.
        :param segment_cnt: int | the count of the segment anomalies.
        :param lowest_drop_ratio: float | the lowest drop ratio.
        :param shortest_period: float | the shortest duration of the segment anomalies.
        :param longest_period: float | the longest duration of the segment anomalies.
        :return:
        """
        import random
        random.seed(1)
        start_positions = np.random.uniform(0.0, 1.0 - 1e-7, segment_cnt)
        start_index_positions = []
        for pos in start_positions:
            index = int(pos * self.instance_cnt)
            start_index_positions.append(index)
        for st_idx in start_index_positions:
            length = random.randint(shortest_period, longest_period)
            for pos in range(st_idx, min(st_idx + length, self.instance_cnt)):
                drop_ratio = np.random.uniform(lowest_drop_ratio, 1)
                self.syn_series[pos] = self.syn_series[pos] * (1 - drop_ratio)
                self.segment_labels[pos] = True

    def filter_by_rule(self):
        """
        It filters out the obvious drop anomalies by a rule approach.
        :return:
        """
        period = self.period_len
        for i in range(period * 2, self.instance_cnt):
            value1 = self.syn_series[i - period]
            value2 = self.syn_series[i - 2 * period]
            drop_ratio1 = (value1 - self.syn_series[i]) / (value1 + 1e-7)
            drop_ratio2 = (value2 - self.syn_series[i]) / (value2 + 1e-7)
            if drop_ratio1 >= 0.75 or drop_ratio2 > 0.75:
                self.rule_labels[i] = True
   
    def syn_drop(self,
                 point_fraction=0.015, lowest_drop_ratio=0.3,
                 segment_cnt=3, shortest_period=3, longest_period=24):
        """
        It implements a workflow to inject and filters out anomalies.
        :param point_fraction: float | the fraction of the anomalies.
        :param lowest_drop_ratio: float | the lowest drop ratio of an anomaly.
        :param segment_cnt: int | the count of the segment anomalies.
        :param shortest_period: float | the shortest duration of the segment anomalies.
        :param longest_period: float | the longest duration of the segment anomalies.
        :return:
        """
        self.add_point_anomalies(point_fraction, lowest_drop_ratio)
        self.add_segment_anomalies(segment_cnt, lowest_drop_ratio,
                                   shortest_period, longest_period)
        self.filter_by_rule()

        for i in range(self.instance_cnt):
            if self.point_labels[i] or self.segment_labels[i] or self.rule_labels[i]:
                self.syn_labels[i] = True

        return self.syn_series, self.syn_labels


class ChangeSynthesiser:
    def __init__(self, raw_series1, raw_series2, period_len=168):
        """
        For injecting change anomalies into the data and output the labels and synthesized series;

        We inject isolated change points and short-lived continuous
        anomalies into into the two raw time series.

        Finally, we apply a rule-based method to filter out the obvious anomalies.

        :param raw_series1: list<float> | the raw KPI time series 1;
        :param raw_series2: list<float> | the raw KPI time series 2;
        :param period_len: int | the period of the time series,
                                 e.g. 168 for a weekly period and 24 for a daily period in one hour basis.
        """
        self.period_len = period_len
        self.syn_series1 = copy.deepcopy(raw_series1)
        self.syn_series2 = copy.deepcopy(raw_series2)
        self.instance_cnt = len(raw_series1)
        self.point_labels = np.array([False] * self.instance_cnt)
        self.segment_labels = np.array([False] * self.instance_cnt)
        self.rule_labels = np.array([False] * self.instance_cnt)
        self.syn_labels = np.array([False] * self.instance_cnt)

    def add_point_anomalies(self,
                            anomaly_fraction, lowest_change_ratio,
                            largest_increase_ratio, largest_decrease_ratio):
        """
        It inject the single anomalies(called point anomalies) into the raw data.
        :param anomaly_fraction: float | the fraction of the anomalies.
        :param lowest_change_ratio: float | the lowest absolute value of change ratio.
        :param largest_increase_ratio: float | the largest increasing ratio.
        :param largest_decrease_ratio: float | the largest decreasing ratio.
        :return:
        """
        import random
        random.seed(1)
        point_positions = np.random.uniform(0.0, 1.0 - 1e-7,
                                            int(self.instance_cnt * anomaly_fraction))
        index_positions = []
        for pos in point_positions:
            index = int(pos * self.instance_cnt)
            index_positions.append(index)

        for idx in index_positions:
            index_kpi = random.randint(1, 2)
            change_direction = random.randint(1, 2)
            decrease_noise = np.random.uniform(lowest_change_ratio, largest_decrease_ratio)
            increase_noise = np.random.uniform(lowest_change_ratio, largest_increase_ratio)
            if index_kpi == 1 and change_direction == 1:
                self.syn_series1[idx] = self.syn_series1[idx] * (1 + increase_noise)
            if index_kpi == 1 and change_direction == 2:
                self.syn_series1[idx] = self.syn_series1[idx] * (1 - decrease_noise)
            if index_kpi == 2 and change_direction == 1:
                self.syn_series2[idx] = self.syn_series2[idx] * (1 + increase_noise)
            if index_kpi == 2 and change_direction == 2:
                self.syn_series2[idx] = self.syn_series2[idx] * (1 - decrease_noise)
            self.point_labels[idx] = True

    def add_segment_anomalies(self,
                              segment_cnt, shortest_period, longest_period,
                              lowest_change_ratio, largest_increase_ratio, largest_decrease_ratio):
        """
        It inject the continous anomalies(called segment anomalies) into the raw data.
        :param segment_cnt: int | the count of the segment anomalies.
        :param shortest_period: float | the shortest duration of the segment anomalies.
        :param longest_period: float | the longest duration of the segment anomalies.
        :param lowest_change_ratio: float | the lowest absolute value of change ratio.
        :param largest_increase_ratio: float | the largest increasing ratio.
        :param largest_decrease_ratio: float | the largest decreasing ratio.
        :return:
        """
        import random
        random.seed(1)
        start_positions = np.random.uniform(0.0, 1.0 - 1e-7, segment_cnt)
        start_index_positions = []
        for pos in start_positions:
            index = int(pos * self.instance_cnt)
            start_index_positions.append(index)
        for st_idx in start_index_positions:
            length = random.randint(shortest_period, longest_period)
            for idx in range(st_idx, min(self.instance_cnt, st_idx + length)):
                index_kpi = random.randint(1, 2)
                change_direction = random.randint(1, 2)
                decrease_noise = np.random.uniform(lowest_change_ratio, largest_decrease_ratio)
                increase_noise = np.random.uniform(lowest_change_ratio, largest_increase_ratio)
                if index_kpi == 1 and change_direction == 1:
                    self.syn_series1[idx] = self.syn_series1[idx] * (1 + increase_noise)
                if index_kpi == 1 and change_direction == 2:
                    self.syn_series1[idx] = self.syn_series1[idx] * (1 - decrease_noise)
                if index_kpi == 2 and change_direction == 1:
                    self.syn_series2[idx] = self.syn_series2[idx] * (1 + increase_noise)
                if index_kpi == 2 and change_direction == 2:
                    self.syn_series2[idx] = self.syn_series2[idx] * (1 - decrease_noise)
                self.segment_labels[idx] = True

    def filter_by_rule(self, rule_fraction):
        """
        It filters out the obvious drop anomalies by a rule approach.
        :return:
        """
        triples = []
        for i in range(self.instance_cnt):
            p = self.syn_series1[i] / (self.syn_series2[i] + 0.000000001)
            triples.append([p, i])
        sorted_triples = sorted(triples)
        ab_cnt = int(rule_fraction * self.instance_cnt)
        for i in range(ab_cnt):
            tri = sorted_triples[i]
            self.rule_labels[tri[1]] = True
        for i in range(self.instance_cnt - ab_cnt, self.instance_cnt):
            tri = sorted_triples[i]
            self.rule_labels[tri[1]] = True

    def syn_change(self,
                   point_anomaly_fraction=0.01, segment_cnt=3, rule_fraction=0.005,
                   shortest_period=3, longest_period=24,
                   lowest_change_ratio=0.3, largest_increase_ratio=1.0, largest_decrease_ratio=1.0):
        """
        It implements a workflow to inject and filters out anomalies.
        """
        self.add_point_anomalies(point_anomaly_fraction, lowest_change_ratio,
                                 largest_increase_ratio, largest_decrease_ratio)

        self.add_segment_anomalies(segment_cnt, shortest_period, longest_period,
                                   lowest_change_ratio, largest_increase_ratio,
                                   largest_decrease_ratio)

        self.filter_by_rule(rule_fraction)

        for i in range(self.instance_cnt):
            if self.point_labels[i] or self.segment_labels[i] or self.rule_labels[i]:
                self.syn_labels[i] = True
        return self.syn_series1, self.syn_series2, self.syn_labels
