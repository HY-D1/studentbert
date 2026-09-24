"""NLEEP-style transferability on frozen features (access regime F).

LEEP (Nguyen et al., 2020) scores a source classifier by how well its soft labels predict the
target labels; NLEEP (Li et al., 2021) replaces the source classifier with a Gaussian mixture
fitted to the candidate's own target features (verify both citations before any paper use).
Higher is better, and the score is at most 0. Our settings, not a claim about the original:
PCA to `dim` components, a diagonal-covariance mixture with K components fitted by EM from
k-means++ starts under a fixed seed. Callers report more than one K so the choice is visible.
"""

from __future__ import annotations

import numpy as np


def pca(x: np.ndarray, dim: int) -> np.ndarray:
    xc = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[: min(dim, vt.shape[0])].T


def _log_gauss_diag(x, means, var):
    # (n, K): log N(x | mean_k, diag(var_k)), summed over dimensions
    diff = x[:, None, :] - means[None, :, :]
    return -0.5 * (np.log(2 * np.pi * var)[None].sum(-1) + (diff ** 2 / var[None]).sum(-1))


def _logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def gmm_diag(x: np.ndarray, k: int, *, seed: int, max_iter: int = 200, tol: float = 1e-6,
             reg: float = 1e-6) -> dict:
    """EM for a diagonal Gaussian mixture. Returns responsibilities and the log-likelihood trace."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    if k == 1:
        return {"resp": np.ones((n, 1)), "loglik": [], "iterations": 0, "converged": True}
    idx = [int(rng.integers(n))]
    dist = ((x - x[idx[0]]) ** 2).sum(1)
    for _ in range(1, k):
        p = dist / dist.sum() if dist.sum() > 0 else np.full(n, 1.0 / n)
        idx.append(int(rng.choice(n, p=p)))
        dist = np.minimum(dist, ((x - x[idx[-1]]) ** 2).sum(1))
    means = x[idx].copy()
    var = np.tile(x.var(axis=0) + reg, (k, 1))
    w = np.full(k, 1.0 / k)
    trace, converged = [], False
    for it in range(1, max_iter + 1):
        logp = np.log(w)[None] + _log_gauss_diag(x, means, var)
        norm = _logsumexp(logp, axis=1)
        resp = np.exp(logp - norm[:, None])
        trace.append(float(norm.mean()))
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) <= tol * abs(trace[-2]):
            converged = True
            break
        nk = resp.sum(0) + 1e-12
        w = nk / n
        means = (resp.T @ x) / nk[:, None]
        var = np.maximum((resp.T @ x ** 2) / nk[:, None] - means ** 2, 0.0) + reg
    return {"resp": resp, "loglik": trace, "iterations": it, "converged": converged}


def leep_from_resp(resp: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels)
    classes = np.unique(y)
    pz = resp.mean(0)
    joint = np.stack([resp[y == c].sum(0) / len(y) for c in classes])  # (C, K)
    cond = joint / np.maximum(pz[None], 1e-300)                        # P(y | z)
    pos = np.searchsorted(classes, y)
    return float(np.mean(np.log(np.maximum((cond[pos] * resp).sum(1), 1e-300))))


def nleep(features: np.ndarray, labels: np.ndarray, *, k: int = 8, dim: int = 32,
          seed: int = 0) -> tuple[float, dict]:
    x = pca(np.asarray(features, dtype=np.float64), dim)
    fit = gmm_diag(x, k, seed=seed)
    score = leep_from_resp(fit["resp"], labels)
    return score, {"k": k, "dim": int(x.shape[1]), "iterations": fit["iterations"],
                   "converged": fit["converged"],
                   "final_loglik": fit["loglik"][-1] if fit["loglik"] else None}
