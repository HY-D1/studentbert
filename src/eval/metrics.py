"""Evaluation metrics for KT: AUC-ROC and ECE (Expected Calibration Error)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC-ROC. y_true in {0,1}, y_prob in [0,1]. Returns nan if one class only."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_prob)))


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins.

    Splits [0,1] into n_bins, compares avg predicted prob vs actual accuracy
    in each bin, weights by bin population. Lower is better-calibrated.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])  # bin index 0..n_bins-1
    total = len(y_prob)
    err = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        conf = y_prob[sel].mean()
        acc = y_true[sel].mean()
        err += (sel.sum() / total) * abs(conf - acc)
    return float(err)
