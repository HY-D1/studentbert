"""H-score on frozen features for next-step correctness (access regime F).

H(f) = tr(cov(f)^+ cov(E[f | y])): the share of feature covariance that separates the label
classes, relative to the total (Bao et al., 2019; verify the citation before any paper use).
Higher is better. The plain form inverts the empirical covariance by pseudo-inverse. The shrunk
form uses the Ledoit-Wolf covariance, a regularized variant for ill-conditioned features that
the transferability literature has also proposed (unverified here). For binary labels H equals
p0 p1 (m1 - m0)' cov^+ (m1 - m0), a Mahalanobis class separation, which the tests check.
"""

from __future__ import annotations

import numpy as np


def ledoit_wolf(xc: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrunk covariance of centered rows xc, and the shrinkage used."""
    n, d = xc.shape
    x2 = xc ** 2
    emp_trace = x2.sum(axis=0) / n
    mu = emp_trace.sum() / d
    beta_ = float(np.sum(x2.T @ x2))
    delta_ = float(np.sum((xc.T @ xc) ** 2)) / n ** 2
    beta = (beta_ / n - delta_) / (d * n)
    delta = (delta_ - 2.0 * mu * emp_trace.sum() + d * mu ** 2) / d
    beta = min(beta, delta)
    shrink = 0.0 if beta == 0 else beta / delta
    cov = (1.0 - shrink) * (xc.T @ xc / n) + shrink * mu * np.eye(d)
    return cov, float(shrink)


def _pinv_psd(cov: np.ndarray, rcond: float = 1e-10) -> tuple[np.ndarray, int]:
    w, v = np.linalg.eigh(cov)
    keep = w > rcond * max(float(w.max()), 0.0)
    if not keep.any():
        raise ValueError("feature covariance is zero; H-score undefined")
    return (v[:, keep] / w[keep]) @ v[:, keep].T, int(keep.sum())


def hscore(features: np.ndarray, labels: np.ndarray, *, shrink: bool = False) -> tuple[float, dict]:
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("H-score needs at least two label classes")
    xc = f - f.mean(axis=0)
    n = xc.shape[0]
    if shrink:
        cov, s = ledoit_wolf(xc)
    else:
        cov, s = xc.T @ xc / n, 0.0
    between = np.zeros_like(cov)
    for c in classes:
        idx = y == c
        m = xc[idx].mean(axis=0)
        between += (idx.sum() / n) * np.outer(m, m)
    inv, rank = _pinv_psd(cov)
    return float(np.trace(inv @ between)), {"rank": rank, "shrinkage": s, "n": int(n)}
