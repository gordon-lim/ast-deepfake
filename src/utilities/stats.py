import numpy as np
from scipy import stats
from sklearn import metrics
import torch

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_eer(target, output):
    """Calculate Equal Error Rate (EER) from all target and output scores."""
    # Compute False Positive Rate, True Positive Rate, and thresholds
    fpr, tpr, thresholds = metrics.roc_curve(target, output)
    fnr = 1 - tpr  # False Negative Rate

    # Find the threshold where FPR == FNR (EER point)
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    return eer

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 1d array, (samples_num, )
      target: 1d array, (samples_num, )

    Returns:
      stats: list of statistics.
    """

    # Flatten target and output for dataset-level evaluation
    target_flat = target.flatten()
    output_flat = output.flatten()

    # Calculate metrics
    acc = metrics.accuracy_score(np.argmax(target, axis=1), np.argmax(output, axis=1))
    avg_precision = metrics.average_precision_score(target_flat, output_flat)
    auc = metrics.roc_auc_score(target_flat, output_flat)
    eer = calculate_eer(target_flat, output_flat)

    stats = {
        'accuracy': acc,
        'average_precision': avg_precision,
        'auc': auc,
        'eer': eer
    }

    return stats

