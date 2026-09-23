"""LogME (You et al., ICML 2021) for frozen StudentBERT features, numpy only.

Replaces the evidence computation in scripts/compute_logme.py, which subtracted the prior term
alpha/2 * ||m||^2 twice (its line 60 already contains it). The gap is exactly
-alpha * ||m||^2 / (2N) = -gamma / (2N) per sample: candidate-dependent and bounded by D / (2N),
0.0026 at the old defaults (D=256, N=50,000), which exceeds the 0.0017 margin by which the old
run placed in-domain above EdNet on ASSISTments 2017.

Evidence is evaluated exactly at the fixed point, per group g sharing hyperparameters (a, b):
  L_g = D/2 log a + N_g/2 log b - 1/2 log|A_g| - b/2 ||y_g - F_g m_g||^2 - a/2 ||m_g||^2
        - N_g/2 log 2pi,   A_g = a I + b F_g^T F_g,   m_g = b A_g^-1 F_g^T y_g.
log|A_g| includes the D - rank(F_g) directions that carry only the prior; the reference code
omits them, which matters only when a group has fewer samples than feature dimensions.

The fixed point stops on the evidence, not on lambda = a/b: when features carry no signal, a
drifts towards infinity (1e152 in testing) while the evidence is already stable to 1e-12.

Binary labels: LogME's classification form averages one-hot columns. After centring, the two
columns for {0, 1} are negatives of each other with identical evidence, so one column is exact.
"""

from __future__ import annotations

import math

import numpy as np

_LOG2PI = math.log(2.0 * math.pi)


def _prepare(F: np.ndarray, y: np.ndarray) -> dict:
    n, d = F.shape
    Fc = F - F.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    y2 = float(yc @ yc)
    if n == 1:
        sigma, z2 = np.zeros(0), np.zeros(0)
    else:
        U, s, _ = np.linalg.svd(Fc, full_matrices=False)
        keep = s > s[0] * 1e-10 if s[0] > 0 else np.zeros(s.shape, dtype=bool)
        sigma = s[keep] ** 2
        z2 = (U[:, keep].T @ yc) ** 2
    perp2 = max(y2 - float(z2.sum()), 0.0)  # part of y outside the column span of F
    return {"n": n, "d": d, "sigma": sigma, "z2": z2, "perp2": perp2, "y2": y2,
            "rank": int(sigma.size)}


def _terms(parts: list[dict], a: float, b: float) -> tuple[float, float, float, float]:
    """Total log evidence, and the summed gamma, ||m||^2 and residual needed by the updates."""
    ev = gamma = m2 = res2 = 0.0
    for p in parts:
        sig, z2 = p["sigma"], p["z2"]
        den = a + b * sig
        g = float((b * sig / den).sum())
        mm = float((b * b * sig * z2 / den**2).sum())
        rr = float((a * a * z2 / den**2).sum()) + p["perp2"]
        logdet = float(np.log(den).sum()) + (p["d"] - p["rank"]) * math.log(a)
        ev += (0.5 * p["d"] * math.log(a) + 0.5 * p["n"] * math.log(b) - 0.5 * logdet
               - 0.5 * b * rr - 0.5 * a * mm - 0.5 * p["n"] * _LOG2PI)
        gamma, m2, res2 = gamma + g, m2 + mm, res2 + rr
    return ev, gamma, m2, res2


def _fit(parts: list[dict], max_iter: int, tol: float) -> tuple[float, dict]:
    n_total = sum(p["n"] for p in parts)
    y2 = sum(p["y2"] for p in parts)
    perp2 = sum(p["perp2"] for p in parts)
    if y2 <= 0.0:
        raise ValueError("labels are constant after centring")
    if perp2 <= 1e-12 * y2:
        # Every label vector lies in its feature span, so the noise precision runs to infinity
        # and the evidence has no finite maximum: refuse rather than report a number.
        raise ValueError("labels can be interpolated exactly; evidence is unbounded")
    a, b = 1.0, 1.0
    prev = None
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        ev, gamma, m2, res2 = _terms(parts, a, b)
        if prev is not None and abs(ev - prev) <= tol * n_total:
            converged = True
            break
        prev = ev
        a = max(gamma, 1e-300) / max(m2, 1e-300)
        b = max(n_total - gamma, 1e-300) / max(res2, 1e-300)
    ev, gamma, m2, res2 = _terms(parts, a, b)
    return ev / n_total, {"alpha": a, "beta": b, "gamma": gamma, "iterations": iterations,
                          "converged": converged, "n": n_total}


def logme(F, y, *, max_iter: int = 1000, tol: float = 1e-12) -> tuple[float, dict]:
    """Per-sample log evidence of y given features F (nats). Higher is better."""
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if F.ndim != 2 or F.shape[0] != y.shape[0]:
        raise ValueError(f"F {F.shape} and y {y.shape} disagree")
    if F.shape[0] < 2:
        raise ValueError("LogME needs at least two samples")
    part = _prepare(F, y)
    score, info = _fit([part], max_iter, tol)
    info.update(dim=part["d"], rank=part["rank"])
    return score, info


def logme_per_group(F, y, groups, *, min_group: int = 20, max_iter: int = 1000,
                    tol: float = 1e-12) -> tuple[float, dict]:
    """Evidence of one linear readout per group with a shared prior and noise level, per sample.

    The KT head reads the logit of skill[t+1] from its own row (finetune_edubert.py lines 128
    to 129), so a readout per next skill is the frozen-feature analogue of fine-tuning. Sharing
    (a, b) across groups keeps small groups finite without dropping them. Groups below min_group
    are pooled into one residual readout so tiny groups do not each get a free intercept. The
    grouping depends only on the target sample, so every candidate sees the same partition.
    """
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    if not (F.shape[0] == y.shape[0] == groups.shape[0]):
        raise ValueError("F, y and groups must have the same length")
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    parts, sizes = [], []
    for g in np.flatnonzero(counts >= min_group):
        idx = np.flatnonzero(inv == g)
        parts.append(_prepare(F[idx], y[idx]))
        sizes.append(int(idx.size))
    small = np.isin(inv, np.flatnonzero(counts < min_group))
    pooled = int(small.sum())
    if pooled:
        parts.append(_prepare(F[small], y[small]))
    score, info = _fit(parts, max_iter, tol)
    info.update(groups_total=int(uniq.size), groups_own_readout=len(sizes),
                groups_pooled=int((counts < min_group).sum()), pooled_positions=pooled,
                constant_label_readouts=sum(1 for p in parts if p["y2"] == 0.0),
                min_group=min_group)
    return score, info
