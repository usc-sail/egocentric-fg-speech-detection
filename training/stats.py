import numpy as np
from scipy import stats
from sklearn import metrics
import torch

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """


    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    targets = np.argmax(target, 1)
    outputs = np.argmax(output, 1)
    acc = metrics.accuracy_score(targets, outputs)
    prec = metrics.precision_score(targets, outputs)
    recall = metrics.recall_score(targets, outputs)
    f1_score = metrics.f1_score(targets, outputs)
    auc = metrics.roc_auc_score(targets, outputs)
    
    stats = {'acc': acc, 'precision':prec, 'recall': recall, 'f1': f1_score, 'auc': auc}
    return stats

