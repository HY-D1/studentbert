"""PARC on frozen features (access regime F), as defined by Bolya, Mittapalli and Hoffman (NeurIPS
2021, Eqs. 7 and 8; settings from their Sec. 3.1 and Table 4).

On a probe set of n target examples, PARC compares two n x n distance matrices,

    D_f = 1 - corrcoef(PCA_f(features)),   D_y = 1 - corrcoef(onehot(labels)),

and scores the candidate by the Spearman correlation of their upper triangles: features that
place differently labelled examples far apart score high. Higher is better. Settings fixed in
docs/parc_design.md before any score existed: a probe of 500 examples with at least two per class,
PCA to 32 dimensions fitted on the probe, and no depth heuristic (the paper's "+l" ensemble is a
separate add-on, not part of PARC). The probe is drawn with numpy default_rng(seed) from the rows
given, so candidates scored on the same rows at the same seed share one probe.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

N_PROBE = 500
DIM = 32


def probe_indices(labels: np.ndarray, n_probe: int, seed: int) -> np.ndarray:
    """n_probe rows drawn at random, then topped up so every class has at least two."""
    y = np.asarray(labels)
    n = y.size
    rng = np.random.default_rng(seed)
    if n <= n_probe:
        return np.arange(n)
    idx = rng.choice(n, size=n_probe, replace=False)
    for c in np.unique(y):
        have = int(np.sum(y[idx] == c))
        if have >= 2:
            continue
        pool = np.setdiff1d(np.flatnonzero(y == c), idx)
        add = rng.choice(pool, size=min(2 - have, pool.size), replace=False)
        if add.size == 0:
            continue
        counts = {k: int(np.sum(y[idx] == k)) for k in np.unique(y[idx])}
        major = max(counts, key=lambda k: (counts[k], str(k)))
        drop = np.flatnonzero(y[idx] == major)[-add.size:]
        idx = np.concatenate([np.delete(idx, drop), add])
    return np.sort(idx)


def parc(features: np.ndarray, labels: np.ndarray, *, n_probe: int = N_PROBE, dim: int = DIM,
         seed: int = 0) -> tuple[float, dict]:
    F = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("PARC needs at least two label classes")
    idx = probe_indices(y, n_probe, seed)
    X = F[idx] - F[idx].mean(axis=0)
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    k = int(min(dim, int(np.sum(s > 1e-10 * s.max())))) if s.size and s.max() > 0 else 0
    if k == 0:
        raise ValueError("probe features have no variance; PARC undefined")
    Z = X @ vt[:k].T
    if np.any(Z.std(axis=1) == 0):
        raise ValueError("a probe example has constant reduced features; PARC undefined")
    onehot = (y[idx][:, None] == classes[None, :]).astype(np.float64)
    iu = np.triu_indices(idx.size, k=1)
    df = (1.0 - np.corrcoef(Z))[iu]
    dy = (1.0 - np.corrcoef(onehot))[iu]
    score = float(np.corrcoef(rankdata(df), rankdata(dy))[0, 1])
    return score, {"n_probe": int(idx.size), "dim": k, "classes": int(classes.size),
                   "probe_positive_share": float(np.mean(y[idx] == classes[-1]))}
