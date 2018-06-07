# Copyright (c) 2017 Wu Jun (littlewj187@gmail.com)


def evaluate(scores, labels):
    """
    It retures the auc and prauc scores.
    :param scores: list<float> | the anomaly scores predicted by CellPAD.
    :param labels: list<float> | the true labels.
    :return: the auc, prauc.
    """
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    pruc = metrics.auc(recall, precision)
    return auc, pruc
