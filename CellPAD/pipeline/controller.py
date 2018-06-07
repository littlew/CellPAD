# Copyright (c) 2017 Wu Jun (littlewj187@gmail.com)

import numpy as np
import pandas as pd
from CellPAD.pipeline.algorithm import RegressionPredictor, StatisticalPredictor, StatisticalMeasurement
from CellPAD.pipeline.feature import FeatureTools
from CellPAD.pipeline.preprocessor import Preprocessor
from CellPAD.pipeline.filter import DropAnomalyFilter, ChangeAnomalyFilter


class DropController:
    def __init__(self,
                 timestamps,
                 series,
                 period_len=168,
                 feature_types=["Indexical", "Numerical"],
                 feature_time_grain=["Weekly"],
                 feature_operations=["Mean","Median","Wma","Ewma"],
                 bootstrap_period_cnt=2,
                 to_remove_trend=True,
                 trend_remove_method="past_mean",
                 anomaly_filter_method="gauss",
                 anomaly_filter_coefficient=3.0):
        """
        DropController implements an automatical pipeline of time series degradations detection.
        It supports flexible setting interfaces of parameters .
        - the prediction algorithm, which is statistical or machine learning based.
        - the types of features, which is indexical or numerical.
        - the numerical operations for extracting different statistical features.
        - the approach for detecting and filtering anomalies in each periodical iteration.

        :param timestamps: list<str> | The timestamps of the time series(ts).
        :param series: list<float> | The values of the time series.
        :param period_len: int | The number of the elements in one period.
        :param feature_types:  list<str> | The tags of the feature types.
        :param feature_time_grain: str | The tag of the time grain when it calculates the numerical features.
        :param feature_operations: list<str> | the tags of operations for calculating numerical features.
        :param bootstrap_period_cnt: int | the counts of the initial periods for a bootstrap of training
        :param to_remove_trend: bool | the handle decides whether remove the trend from the time series.
        :param anomaly_filter_method: str | the tag decides the approach on detecting anomalies for each period.
        :param anomaly_filter_coefficient: float | the coefficient used in the anomaly detection stage.
        """

        # the dict of the attributes of the time series.
        dict_series = {}
        if not to_remove_trend:
            dict_series["detected_series"] = np.array(series)
        else:
            preprocessor = Preprocessor()
            dict_series["detected_series"] = preprocessor.remove_trend(series, period_len, method=trend_remove_method)
        dict_series["timestamps"] = timestamps
        dict_series["series_len"] = len(dict_series["detected_series"])
        dict_series["period_len"] = period_len
        self.dict_series = dict_series

        # the dict of features related variables.
        dict_feature = {}
        dict_feature["operations"] = feature_operations
        dict_feature["time_grain"] = feature_time_grain
        dict_feature["feature_types"] = feature_types
        dict_feature["feature_tool"] = FeatureTools()
        dict_feature["feature_list"] = dict_feature["feature_tool"].set_feature_names(dict_feature["feature_types"],
                                                                                      dict_feature["time_grain"],
                                                                                      dict_feature["operations"])
        self.dict_feature = dict_feature

        # the dict of the bootstrap parameters.
        dict_bootstrap = {}
        dict_bootstrap["period_cnt"] = bootstrap_period_cnt
        dict_bootstrap["bootstrap_series_len"] = bootstrap_period_cnt * period_len
        dict_bootstrap["bootstrap_series"] = self.dict_series["detected_series"][:dict_bootstrap["bootstrap_series_len"]]
        self.dict_bootstrap = dict_bootstrap

        # the dict of anomaly filter parameters.
        dict_filter = {}
        dict_filter["method"] = anomaly_filter_method
        dict_filter["coefficient"] = anomaly_filter_coefficient
        self.dict_filter = dict_filter

        # the dict of the storage for training data.
        dict_storage = {}
        dict_storage["normal_features_matrix"] = pd.DataFrame()
        dict_storage["normal_response_series"] = []
        self.dict_storage = dict_storage

    def __init_bootstrap(self):
        """
        It initiates the result variables.
        :return:
        """
        # the dict of prediction results
        dict_result = {}
        dict_result["drop_ratios"] = [0.0] * self.dict_bootstrap["bootstrap_series_len"]
        dict_result["drop_scores"] = [0.0] * self.dict_bootstrap["bootstrap_series_len"]
        dict_result["drop_labels"] = [False] * self.dict_bootstrap["bootstrap_series_len"]
        dict_result["predicted_series"] = self.dict_bootstrap["bootstrap_series"]
        self.dict_result = dict_result

    def __store_features_response(self, this_features_matrix, this_response_series, this_labels):
        """
        It stores the features and the response variables of each iteration and reuses them in the future.
        It only stores the instances with a "False" label, which means this instance is normal.
        :param this_features_matrix: array<float> | the matrix of features.
        :param this_response_series: list<float> | the corresponding series of the response variable.
        :param this_labels: list<bool> | the labels to indicates whether the point is an anomaly.
        :return:
        """
        for idx in range(len(this_labels)):
            top_matrix = self.dict_storage["normal_features_matrix"]
            bottom_line = this_features_matrix[idx]
            if not this_labels[idx]:
                self.dict_storage["normal_features_matrix"] = np.row_stack((top_matrix, bottom_line))
                self.dict_storage["normal_response_series"].append(this_response_series[idx])

    def __store_this_results(self, this_predicted_series, this_drop_ratios, this_drop_labels, this_drop_scores):
        """
        It stores the results in each period(iteration).
        :param this_predicted_series: list<float> | the predicted values.
        :param this_drop_ratios: list<float> | the drop ratios.
        :param this_drop_labels: list<bool> | the drop labels
        :param this_drop_scores: list<float> | the drop scores
        :return:
        """
        self.dict_result["predicted_series"] = np.append(self.dict_result["predicted_series"], this_predicted_series)
        self.dict_result["drop_ratios"] = np.append(self.dict_result["drop_ratios"], this_drop_ratios)
        self.dict_result["drop_labels"] = np.append(self.dict_result["drop_labels"], this_drop_labels)
        self.dict_result["drop_scores"] = np.append(self.dict_result["drop_scores"], this_drop_scores)

    def __detect_by_statistical_smoothing(self, predictor):
        """
        It run a workflow using statistical smoothing algorithm to predict and detect drop anomalies.
        :param predictor: the tag(identification) of the statistical prediction algorithm selected by the user.
        :return:
        """
        self.__init_bootstrap()
        model = StatisticalPredictor(predictor)
        series_len = self.dict_series["series_len"]
        period_len = self.dict_series["period_len"]
        round_cnt = int(np.ceil(series_len / period_len))
        for rod in range(self.dict_bootstrap["period_cnt"], round_cnt):
            st = rod * period_len
            ed = min(series_len, st + period_len)
            this_practical_series = self.dict_series["detected_series"][st:ed]
            # predict the next week
            this_predicted_series = model.predict(series=self.dict_series["detected_series"][:st],
                                                  labels=self.dict_result["drop_labels"][:st],
                                                  period_len=period_len)
            this_drop_ratios, this_drop_labels, this_drop_scores = \
                                self.__filter_anomaly(predicted_series=this_predicted_series,
                                                    practical_series=this_practical_series)
            self.__store_this_results(this_predicted_series, this_drop_ratios,
                                    this_drop_labels, this_drop_scores)

    def __detect_by_regression(self, predictor, n_esimators=100):
        """
        It run a workflow using regression algorithm to predict and detect drop anomalies.
        :param predictor: str | the tag(identification) of the algorithm selected by the user.
        :param n_esimators: int | the number of esimators when random forest approach is used.
        :return:
        """
        self.__init_bootstrap()
        # regression model
        model = RegressionPredictor(predictor)

        # extract features
        first_train_features = self.dict_feature["feature_tool"].compute_feature_matrix(
                                     timestamps=self.dict_series["timestamps"],
                                     series=self.dict_bootstrap["bootstrap_series"],
                                     labels=[False] * self.dict_bootstrap["bootstrap_series_len"],
                                     ts_period_len=self.dict_series["period_len"],
                                     feature_list=self.dict_feature["feature_list"],
                                     start_pos=0,
                                     end_pos=self.dict_bootstrap["bootstrap_series_len"])
        first_train_response = self.dict_bootstrap["bootstrap_series"]
        # train the regression model
        model.train(first_train_features, first_train_response)

        # store the bootstrapping features and the predicted instances
        self.dict_storage["normal_features_matrix"] = first_train_features
        self.dict_storage["normal_response_series"] = list(first_train_response)

        # interactively training
        round_cnt = int(np.ceil(self.dict_series["series_len"] / self.dict_series["period_len"]))
        for rod in range(self.dict_bootstrap["period_cnt"], round_cnt):
            st = rod * self.dict_series["period_len"]
            ed = min(self.dict_series["series_len"], st + self.dict_series["period_len"])
            #extract features
            this_predicted_features = self.dict_feature["feature_tool"].compute_feature_matrix(
                                         timestamps=self.dict_series["timestamps"],
                                         series=self.dict_series["detected_series"][:ed],
                                         labels=self.dict_result["drop_labels"],
                                         ts_period_len=self.dict_series["period_len"],
                                         feature_list=self.dict_feature["feature_list"],
                                         start_pos=st,
                                         end_pos=ed)
            # predict the series in the current period
            this_predicted_series = model.predict(this_predicted_features)
            this_practical_series = self.dict_series["detected_series"][st:ed]

            # compare the practical and predicted values and filter anomalies
            this_drop_ratios, this_drop_labels, this_drop_scores = \
                                    self.__filter_anomaly(predicted_series=this_predicted_series,
                                                        practical_series=this_practical_series)
            # store the detected results
            self.__store_this_results(this_predicted_series, this_drop_ratios,
                                    this_drop_labels, this_drop_scores)

            # store the features and values of the new normal instances in the current period.
            self.__store_features_response(this_features_matrix=this_predicted_features,
                                         this_response_series=this_practical_series,
                                         this_labels=this_drop_labels)
            # update the model
            model.train(np.array(self.dict_storage["normal_features_matrix"]),
                        np.array(self.dict_storage["normal_response_series"]))

    def __filter_anomaly(self, predicted_series, practical_series):
        """
        It detects and filters the anomalies in each period(iteration).
        :param predicted_series: list<float> | the series of the predicted values.
        :param practical_series: list<float> |the series of the practical values(containing anomalies).
        :return: the list of the drop ratios
                 the drop labels to identity the anomalies.
                 the drop scores measuring the anomaly degree.
        """
        anomaly_filter = DropAnomalyFilter(rule=self.dict_filter["method"],
                                           coef=self.dict_filter["coefficient"])
        drop_ratios, drop_labels, drop_scores = \
            anomaly_filter.detect_anomaly(predicted_series=predicted_series,
                                          practical_series=practical_series)
        return drop_ratios, drop_labels, drop_scores

    def detect(self, predictor):
        """
        It calls the algorithm named as "predictor".
        :param predictor:
        :return:
        """
        if predictor == "WMA" or predictor == "EWMA" or predictor == "HW":
            self.__detect_by_statistical_smoothing(predictor=predictor)
        if predictor == "RT" or predictor == "RF" or predictor == "SLR" or predictor == "HR":
            self.__detect_by_regression(predictor=predictor)


    def get_results(self):
        """
        It returns the results including the predicted series, the drop labels and drop ratios.
        :return: the detected_series(where the trend is removed or not)
                 the predicted_series
        """
        return self.dict_result


class ChangeController:
    def __init__(self,
                 timestamps,
                 series1,
                 series2,
                 period_len=168,
                 feature_types=["Numerical"],
                 feature_time_grain="Weekly",
                 feature_operations=["Raw"],
                 bootstrap_period_cnt=2,
                 to_remove_trend=False,
                 trend_remove_method="center_mean",
                 anomaly_filter_method="gauss",
                 anomaly_filter_coefficient=3.0):
        """
        ChangeController implements an automatical pipeline of time series correlation changes detection.
        It supports flexible setting interfaces of parameters .
        - the prediction algorithm, which is statistical or machine learning based.
        - the types of features, which is indexical or numerical.
        - the numerical operations for extracting different statistical features.
        - the approach for detecting and filtering anomalies in each periodical iteration.

        :param timestamps: list<str> | The timestamps of the time series(ts).
        :param series1: list<float> | The values of the time series 1.
        :param series2: list<float> | The values of the time series 2.
        :param period_len: int | The number of the elements in one period.
        :param feature_types:  list<str> | The tags of the feature types.
        :param feature_time_grain: str | The tag of the time grain when it calculates the numerical features.
        :param feature_operations: list<str> | the tags of operations for calculating numerical features.
        :param bootstrap_period_cnt: int | the counts of the initial periods for a bootstrap of training
        :param to_remove_trend: bool | the handle decides whether remove the trend from the time series.
        :param anomaly_filter_method: str | the tag decides the approach on detecting anomalies for each period.
        :param anomaly_filter_coefficient: float | the coefficient used in the anomaly detection stage.
        """

        # time series
        dict_series = {}
        if not to_remove_trend:
            dict_series["detected_series1"] = np.array(series1)
            dict_series["detected_series2"] = np.array(series2)
        else:
            preprocessor = Preprocessor()
            dict_series["detected_series1"] = preprocessor.remove_trend(series1, period_len, method=trend_remove_method)
            dict_series["detected_series2"] = preprocessor.remove_trend(series2, period_len, method=trend_remove_method)
        dict_series["timestamps"] = timestamps
        dict_series["series_len"] = len(dict_series["detected_series1"])
        dict_series["period_len"] = period_len
        self.dict_series = dict_series

        # feature
        dict_feature = {}
        dict_feature["operations"] = feature_operations
        dict_feature["time_grain"] = feature_time_grain
        dict_feature["feature_types"] = feature_types
        dict_feature["feature_tool"] = FeatureTools()
        dict_feature["feature_list"] = dict_feature["feature_tool"].set_feature_names(dict_feature["feature_types"],
                                                                                      dict_feature["time_grain"],
                                                                                      dict_feature["operations"])
        self.dict_feature = dict_feature

        # bootstrap parameters
        dict_bootstrap = {}
        dict_bootstrap["period_cnt"] = bootstrap_period_cnt
        dict_bootstrap["bootstrap_series_len"] = bootstrap_period_cnt * period_len
        dict_bootstrap["bootstrap_series1"] = self.dict_series["detected_series1"][
                                             :dict_bootstrap["bootstrap_series_len"]]
        dict_bootstrap["bootstrap_series2"] = self.dict_series["detected_series2"][
                                              :dict_bootstrap["bootstrap_series_len"]]
        self.dict_bootstrap = dict_bootstrap

        # anomaly filter parameters
        dict_filter = {}
        dict_filter["method"] = anomaly_filter_method
        dict_filter["coefficient"] = anomaly_filter_coefficient
        self.dict_filter = dict_filter

        # storage for training data
        dict_storage = {}
        dict_storage["normal_features_matrix1"] = pd.DataFrame()
        dict_storage["normal_features_matrix2"] = pd.DataFrame()
        dict_storage["normal_response_series1"] = []
        dict_storage["normal_response_series2"] = []
        self.dict_storage = dict_storage

    def __init_bootstrap(self):
        """
        It initiates the result variables.
        :return:
        """
        dict_result = {}
        dict_result["predicted_series1"] = self.dict_bootstrap["bootstrap_series1"]
        dict_result["predicted_series2"] = self.dict_bootstrap["bootstrap_series2"]
        dict_result["change_ratios1"] = [0.0] * self.dict_bootstrap["bootstrap_series_len"]
        dict_result["change_ratios2"] = [0.0] * self.dict_bootstrap["bootstrap_series_len"]
        dict_result["change_scores"] = [0.0] * self.dict_bootstrap["bootstrap_series_len"]
        dict_result["change_labels"] = [False] * self.dict_bootstrap["bootstrap_series_len"]

        self.dict_result = dict_result

    def __store_features_response(self,
                                this_features_matrix1, this_response_series1,
                                this_features_matrix2, this_response_series2,
                                this_labels):
        """
        It stores the features and the response variables of each iteration and reuses them in the future.
        It only stores the instances with a "False" label, which means this instance is normal.
        :param this_features_matrix1: array<float> | the matrix of features of time series 1.
        :param this_response_series1: list<float> | the corresponding series of the response variable 1.
        :param this_features_matrix2: array<float> | the matrix of features of time series 2.
        :param this_response_series2: list<float> | the corresponding series of the response variable 2.
        :param this_labels: list<bool> | the labels to indicates whether the point is an anomaly.
        :return:
        """
        for idx in range(len(this_labels)):
            top_matrix = self.dict_storage["normal_features_matrix1"]
            bottom_line = this_features_matrix1[idx]
            if not this_labels[idx]:
                self.dict_storage["normal_features_matrix1"] = np.row_stack((top_matrix, bottom_line))
                self.dict_storage["normal_response_series1"].append(this_response_series1[idx])

            top_matrix = self.dict_storage["normal_features_matrix2"]
            bottom_line = this_features_matrix2[idx]
            if not this_labels[idx]:
                self.dict_storage["normal_features_matrix2"] = np.row_stack((top_matrix, bottom_line))
                self.dict_storage["normal_response_series2"].append(this_response_series2[idx])

    def __store_this_results(self,
                           this_predicted_series1, this_predicted_series2,
                           this_change_ratios1, this_change_ratios2,
                           this_change_labels, this_change_scores):
        """
        It stores the results in each period(iteration).
        :param this_predicted_series1: list<float> | the predicted values for time series 1.
        :param this_predicted_series2: list<float> | the predicted values for time series 2.
        :param this_change_ratios1: list<float> | the change ratios for time series 1.
        :param this_change_ratios2: list<float> | the change ratios for time series 2.
        :param this_change_labels: list<bool> | the change labels.
        :param this_change_scores: list<float> | the change scores.
        :return:
        """
        self.dict_result["predicted_series1"] = np.append(self.dict_result["predicted_series1"], this_predicted_series1)
        self.dict_result["predicted_series2"] = np.append(self.dict_result["predicted_series2"], this_predicted_series2)
        self.dict_result["change_ratios1"] = np.append(self.dict_result["change_ratios1"], this_change_ratios1)
        self.dict_result["change_ratios2"] = np.append(self.dict_result["change_ratios2"], this_change_ratios2)
        self.dict_result["change_labels"] = np.append(self.dict_result["change_labels"], this_change_labels)
        self.dict_result["change_scores"] = np.append(self.dict_result["change_scores"], this_change_scores)

    def __detect_by_regression(self, predictor):
        """
        It run a workflow using regression algorithm to detect change anomalies.
        :param predictor: str | the tag(identification) of the algorithm selected by the user.
        :param n_esimators: int | the number of esimators when random forest approach is used.
        :return:
        """
        self.__init_bootstrap()
        # extract features
        first_train_features1 = self.dict_feature["feature_tool"].compute_feature_matrix(
                                                 timestamps=self.dict_series["timestamps"],
                                                 series=self.dict_bootstrap["bootstrap_series1"],
                                                 labels=[False] * self.dict_bootstrap["bootstrap_series_len"],
                                                 ts_period_len=self.dict_series["period_len"],
                                                 feature_list=self.dict_feature["feature_list"],
                                                 start_pos=0,
                                                 end_pos=self.dict_bootstrap["bootstrap_series_len"])
        first_train_response1 = self.dict_bootstrap["bootstrap_series2"]

        first_train_features2 = self.dict_feature["feature_tool"].compute_feature_matrix(
                                                 timestamps=self.dict_series["timestamps"],
                                                 series=self.dict_bootstrap["bootstrap_series2"],
                                                 labels=[False] * self.dict_bootstrap["bootstrap_series_len"],
                                                 ts_period_len=self.dict_series["period_len"],
                                                 feature_list=self.dict_feature["feature_list"],
                                                 start_pos=0,
                                                 end_pos=self.dict_bootstrap["bootstrap_series_len"])
        first_train_response2 = self.dict_bootstrap["bootstrap_series1"]

        # train the regression model
        model1 = RegressionPredictor(predictor)
        model1.train(first_train_features1, first_train_response1)
        model2 = RegressionPredictor(predictor)
        model2.train(first_train_features2, first_train_response2)
        # store the bootstrapping features and the predicted instances
        self.dict_storage["normal_features_matrix1"] = first_train_features1
        self.dict_storage["normal_features_matrix2"] = first_train_features2
        self.dict_storage["normal_response_series1"] = list(first_train_response1)
        self.dict_storage["normal_response_series2"] = list(first_train_response2)

        # interactively training
        round_cnt = int(np.ceil(self.dict_series["series_len"] / self.dict_series["period_len"]))
        for rod in range(self.dict_bootstrap["period_cnt"], round_cnt):
            st = rod * self.dict_series["period_len"]
            ed = min(self.dict_series["series_len"], st + self.dict_series["period_len"])

            this_predicted_features1 = self.dict_feature["feature_tool"].compute_feature_matrix(
                                                 timestamps=self.dict_series["timestamps"],
                                                 series=self.dict_series["detected_series1"][:ed],
                                                 labels=self.dict_result["change_labels"],
                                                 ts_period_len=self.dict_series["period_len"],
                                                 feature_list=self.dict_feature["feature_list"],
                                                 start_pos=st,
                                                 end_pos=ed)

            this_predicted_features2 = self.dict_feature["feature_tool"].compute_feature_matrix(
                                                  timestamps=self.dict_series["timestamps"],
                                                  series=self.dict_series["detected_series2"][:ed],
                                                  labels=self.dict_result["change_labels"],
                                                  ts_period_len=self.dict_series["period_len"],
                                                  feature_list=self.dict_feature["feature_list"],
                                                  start_pos=st,
                                                  end_pos=ed)

            this_predicted_series1 = model2.predict(this_predicted_features2)
            this_practical_series1 = self.dict_series["detected_series1"][st:ed]
            this_predicted_series2 = model1.predict(this_predicted_features1)
            this_practical_series2 = self.dict_series["detected_series2"][st:ed]

            # compare the practical and predicted values and filter anomalies
            this_change_ratios1, this_change_ratios2, this_change_labels, this_change_scores = \
                self.__filter_anomaly_regression(predicted_series1=this_predicted_series1,
                                               practical_series1=this_practical_series1,
                                               predicted_series2=this_predicted_series2,
                                               practical_series2=this_predicted_series2)
            # store the detected results
            self.__store_this_results(this_predicted_series1, this_predicted_series2,
                                    this_change_ratios1, this_change_ratios2,
                                    this_change_labels, this_change_scores)

            # store the features and values of the new normal instances in the current period.
            self.__store_features_response(this_features_matrix1=this_predicted_features1,
                                         this_response_series1=this_practical_series2,
                                         this_features_matrix2=this_predicted_features2,
                                         this_response_series2=this_practical_series1,
                                         this_labels=this_change_labels)
            # update the model
            model1.train(np.array(self.dict_storage["normal_features_matrix1"]),
                         np.array(self.dict_storage["normal_response_series1"]))
            model2.train(np.array(self.dict_storage["normal_features_matrix2"]),
                         np.array(self.dict_storage["normal_response_series2"]))

    def __detect_by_lcs(self, predictor):
        """
        It run a workflow using local correlation tracking algorithm to measure the change degree.
        :param predictor: str | the tag(identification) of the statistical prediction algorithm selected by the user.
        :return:
        """
        self.__init_bootstrap()
        model = StatisticalMeasurement(predictor)
        round_cnt = int(np.ceil(self.dict_series["series_len"] / self.dict_series["period_len"]))
        for rod in range(self.dict_bootstrap["period_cnt"], round_cnt):
            st = rod * self.dict_series["period_len"]
            ed = min(self.dict_series["series_len"], st + self.dict_series["period_len"])
            # predict the next week
            lcs_score = model.measure(series1=self.dict_series["detected_series1"][:ed],
                                      series2=self.dict_series["detected_series2"][:ed],
                                      labels=self.dict_result["change_labels"][:st],
                                      period_len=self.dict_series["period_len"])

            this_change_labels, this_change_scores = \
                self.__filter_anomaly_lcs(lcs_score)
            self.__store_this_results([], [], [], [], this_change_labels, this_change_scores)

    def __filter_anomaly_regression(self,
                                  predicted_series1, practical_series1,
                                  predicted_series2, practical_series2):
        """
        It detects and filters the anomalies in each period(iteration).
        The measure is the deviation(change ratio) between the predicted value and the practical value.
        :param predicted_series1: list<float> | the predicted values of the time series 1.
        :param practical_series1: list<float> | the practical values of the time series 1.
        :param predicted_series2: list<float> | the predicted values of the time series 2.
        :param practical_series2: list<float> | the practical values of the time series 2.
        :return: the change ratios of time series 1, the change ratios of time series 2,
                 the change labels to identity the anomalies.
                 the change scores measuring the anomaly degree.
        """
        anomaly_filter = ChangeAnomalyFilter(rule=self.dict_filter["method"],
                                             coef=self.dict_filter["coefficient"])
        change_ratios1, change_ratios2, change_labels, change_scores = \
            anomaly_filter.detect_anomaly_regression(predicted_series1=predicted_series1,
                                                     practical_series1=practical_series1,
                                                     predicted_series2=predicted_series2,
                                                     practical_series2=practical_series2)
        return change_ratios1, change_ratios2, change_labels, change_scores

    def __filter_anomaly_lcs(self, lcs_scores):
        """
        It detects and filters the anomalies in each period(iteration).
        The measure is local correlation score.
        :param lcs_scores: list<float> | the list of the local correlation scores.
        :return: the change labels to identity the anomalies.
                 the change scores measuring the anomaly degree.
        """
        anomaly_filter = ChangeAnomalyFilter(rule=self.dict_filter["method"],
                                             coef=self.dict_filter["coefficient"])
        change_labels, change_scores = \
            anomaly_filter.detect_anomaly_lcs(lcs_scores)
        return change_labels, change_scores

    def detect(self, predictor):
        """
        It calls the algorithm named as "predictor".
        :param predictor:
        :param n_esimators:
        :return:
        """
        if predictor == "LCS":
            self.__detect_by_lcs(predictor=predictor)
        if predictor in ["RF", "RT", "SLR", "HR"]:
            self.__detect_by_regression(predictor=predictor)

    def get_results(self):
        """
        It return the predicted values, scores and labels of all the weeks.
        :return:
        """
        return self.dict_result
