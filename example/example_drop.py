from CellPAD.pipeline.evaluator import evaluate
from CellPAD.pipeline.controller import DropController
from CellPAD.pipeline.synthsiser import DropSynthesiser
import pandas as pd
import platform


def test():
    # read KPI
    sysstr = platform.system()
    if sysstr == "Windows":
        data_path = "..\\data\\sd.csv"
    else:
        data_path = "../data/sd.csv"
    df = pd.read_csv(data_path)
    timestamps, series = df["Time"].values, df["KPI"].values

    # inject anomalies
    syner = DropSynthesiser(raw_series=series, period_len=168)
    syn_series, syn_labels = syner.syn_drop()

    # detect drop
    controller = DropController(timestamps=timestamps,
                                series=syn_series,
                                period_len=168,
                                feature_types=["Indexical", "Numerical"],
                                feature_time_grain=["Weekly"],
                                feature_operations=["Wma", "Ewma", "Mean", "Median"],
                                bootstrap_period_cnt=2,
                                to_remove_trend=True,
                                trend_remove_method="center_mean",
                                anomaly_filter_method="gauss",
                                anomaly_filter_coefficient=3.0)
    controller.detect(predictor="RF")
    results = controller.get_results()

    auc, prauc = evaluate(results["drop_scores"][2*168:], syn_labels[2*168:])

    print("front_mean", "auc", auc, "prauc", prauc)


test()