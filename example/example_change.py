from CellPAD.pipeline.evaluator import evaluate
from CellPAD.pipeline.controller import ChangeController
from CellPAD.pipeline.synthsiser import ChangeSynthesiser
import pandas as pd
import platform


def test():
    # read KPI
    sysstr = platform.system()
    if sysstr == "Windows":
        data_path = "..\\data\\cc.csv"
    else:
        data_path = "../data/cc.csv"
    df = pd.read_csv(data_path)
    timestamps, series1, series2 = df["Time"].values, df["KPI1"].values, df["KPI2"].values
    # inject anomalies
    syner = ChangeSynthesiser(raw_series1=series1, raw_series2=series2, period_len=168)
    syn_series1, syn_series2, syn_labels = syner.syn_change()
    # detect drop
    controller = ChangeController(timestamps=timestamps,
                                   series1=syn_series1, series2=syn_series2,
                                   period_len=168,
                                   feature_types=["Numerical"],
                                   feature_time_grain=["Weekly"],
                                   feature_operations=["Raw"],
                                   bootstrap_period_cnt=2,
                                   to_remove_trend=True,
                                   anomaly_filter_method="gauss",
                                   anomaly_filter_coefficient=3.0)

    controller.detect(predictor="HR")
    results = controller.get_results()

    auc, prauc = evaluate(results["change_scores"][168 * 2:], syn_labels[168 * 2:])
    print("auc", auc, "prauc", prauc)


test()