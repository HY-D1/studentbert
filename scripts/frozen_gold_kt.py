from __future__ import annotations

# MRAP failure review, diagnostic D6: frozen-feature KT performance ("frozen gold") on StudentBERT,
# the analogue of the published frozen column of Bassignana et al. (2022) that D1 used on NLP.
# For each candidate encoder, the causal features the estimators read (src/estimators/features.py:
# the fine-tune's learner draw, the 50,000-position cap, vocabulary tensors loaded by intent, the
# fine-tune's random start for reset tensors) train a logistic readout, which is then scored on
# TEST learners. This script reads the test split on purpose: it measures a gold-side quantity,
# like the fine-tune's own test AUC, and is not an estimator. Its records therefore carry no
# "estimator", "score" or "metadata" field, so evaluate_estimators.py cannot load them as scores,
# and nothing here may feed a transferability estimate. Readouts, fixed before any result:
#   frozen_linear            logit = w.h_t + c                   (the label the estimators score)
#   frozen_linear_skillbias  logit = w.h_t + c + b[skill_{t+1}]  (adds the next skill's base rate,
#                                                                 which the KT head can express
#                                                                 and a shared readout cannot)
# Features are standardized with train statistics only; an L2 penalty lam applies to w and b, not
# to c; lam 1e-4 is primary and 1e-2 a sensitivity check. Test learners: a fixed draw (seed 0) of
# up to 3,000 learners and 200,000 positions, the same for every candidate and seed. AUC is
# src/eval/metrics.auc, the metric of the fine-tune gold. Plain H-score on the same train features
# is recorded as an alignment check against the 7 x 7 scores. Rerun-safe: finished (candidate,
# seed) pairs are skipped, and each pair's records are written together.
#
#   PYTHONPATH=. python scripts/frozen_gold_kt.py --target_dir ../processed/algebra2006 \
#       --candidates scratch ../checkpoints/edubert_algebra2006_pretrain_full_encoder.pt \
#       --n_students 3000 --seeds 42 1 2 --out frozen_gold_kt_algebra2006.jsonl

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

READOUTS = ("frozen_linear", "frozen_linear_skillbias")
LAMS = (1e-4, 1e-2)
TEST_SEED = 0


def standardize(xtr: np.ndarray, xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale both sets with the TRAIN mean and SD only."""
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (xtr - mu) / sd, (xte - mu) / sd


def fit_logistic(x: np.ndarray, y: np.ndarray, *, groups: np.ndarray | None = None,
                 n_groups: int = 0, lam: float = 1e-4, max_iter: int = 1000) -> tuple[dict, dict]:
    """Mean log loss plus lam/2 (|w|^2 + |b|^2), minimized with L-BFGS on analytic gradients."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    k = n_groups if groups is not None else 0

    def split(t):
        return t[:d], t[d], t[d + 1:]

    def f(t):
        w, c, b = split(t)
        z = x @ w + c + (b[groups] if k else 0.0)
        loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * (w @ w + b @ b)
        r = (expit(z) - y) / n
        gb = np.bincount(groups, weights=r, minlength=k) + lam * b if k else np.empty(0)
        return loss, np.concatenate([x.T @ r + lam * w, [r.sum()], gb])

    res = minimize(f, np.zeros(d + 1 + k), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "gtol": 1e-7})
    w, c, b = split(res.x)
    return ({"w": w, "c": float(c), "b": b},
            {"converged": bool(res.success), "iterations": int(res.nit),
             "final_loss": float(res.fun)})


def predict(params: dict, x: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) @ params["w"] + params["c"]
    if groups is not None and params["b"].size:
        z = z + params["b"][groups]
    return expit(z)


def test_learners(n_total: int, n_max: int) -> list[int]:
    """Candidate- and seed-independent draw of test learners."""
    order = np.random.default_rng(TEST_SEED).permutation(n_total)[:n_max]
    return sorted(int(i) for i in order)


def readout_records(ftr, ytr, gtr, fte, yte, gte, n_groups: int) -> list[dict]:
    from src.eval.metrics import auc

    ztr, zte = standardize(ftr, fte)
    out = []
    for readout in READOUTS:
        use_b = readout == "frozen_linear_skillbias"
        for lam in LAMS:
            t0 = time.perf_counter()
            params, info = fit_logistic(ztr, ytr, groups=gtr if use_b else None,
                                        n_groups=n_groups, lam=lam)
            out.append({"readout": readout, "lam": lam,
                        "frozen_test_auc": auc(yte, predict(params, zte, gte if use_b else None)),
                        "train_auc": auc(ytr, predict(params, ztr, gtr if use_b else None)),
                        "fit_s": time.perf_counter() - t0, **info})
    return out


def done_pairs(path: Path) -> set:
    if not path.exists():
        return set()
    seen: dict = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            seen.setdefault((r["candidate"], r["seed"]), set()).add((r["readout"], r["lam"]))
    full = {(ro, lam) for ro in READOUTS for lam in LAMS}
    return {k for k, v in seen.items() if v == full}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--n_students", type=int, default=None)
    ap.add_argument("--max_positions", type=int, default=50000)
    ap.add_argument("--test_students", type=int, default=3000)
    ap.add_argument("--test_max_positions", type=int, default=200000)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    from torch.utils.data import Subset

    from src.data.dataset import InteractionDataset
    from src.estimators.features import (build_backbone, cap_positions, kt_features,
                                         load_candidate, sample_target, target_num_skills)
    from src.estimators.hscore import hscore

    target = Path(a.target_dir).name
    k = target_num_skills(a.target_dir) + 1
    out = Path(a.out)
    skip = done_pairs(out)
    test_ds = InteractionDataset(str(a.target_dir), "test", a.max_seq_len)
    te_idx = test_learners(len(test_ds), a.test_students)
    test_sub = Subset(test_ds, te_idx)
    test_rows = {test_ds.rows[i] for i in te_idx}
    for seed in a.seeds:
        for c in a.candidates:
            name = "scratch" if c == "scratch" else Path(c).name
            if (name, seed) in skip:
                print(f"skip (done): {target} {name} seed {seed}", flush=True)
                continue
            t0 = time.perf_counter()
            bb = build_backbone(k - 1, seed=seed, max_len=a.max_seq_len)
            load = {"source": "none", "in_domain": False}
            if c != "scratch":
                load = load_candidate(bb, c, target)
            bb.to(a.device)
            sub, rows, fp = sample_target(a.target_dir, a.n_students, seed, a.max_seq_len)
            if test_rows & set(rows):
                raise SystemExit(f"ABORT: train draw and test learners overlap on {target}")
            with torch.no_grad():
                f, y, g = kt_features(bb, sub, a.device)
                fte, yte, gte = kt_features(bb, test_sub, a.device)
            sel = cap_positions(len(y), a.max_positions, seed)
            tsel = cap_positions(len(yte), a.test_max_positions, TEST_SEED)
            ftr, ytr, gtr = f[sel], y[sel], g[sel]
            fte, yte, gte = fte[tsel], yte[tsel], gte[tsel]
            extract_s = time.perf_counter() - t0
            hs, _ = hscore(ftr, ytr)
            recs = readout_records(ftr, ytr, gtr, fte, yte, gte, k)
            common = {"target": target, "candidate": name, "seed": seed, "gold_side": True,
                      "details": {"load": {kk: load.get(kk) for kk in ("source", "in_domain")},
                                  "train_fingerprint": fp, "n_students_requested": a.n_students,
                                  "n_train_positions": int(ytr.size),
                                  "n_test_learners": len(te_idx),
                                  "n_test_positions": int(yte.size),
                                  "hscore_on_train_features": float(hs),
                                  "extract_s": extract_s}}
            with open(out, "a") as fh:
                for r in recs:
                    fh.write(json.dumps({**common, **r}) + "\n")
            best = [r for r in recs if r["lam"] == LAMS[0]]
            print(f"FROZEN target={target} cand={name} seed={seed} "
                  + " ".join(f"{r['readout']}={r['frozen_test_auc']:.5f}" for r in best)
                  + f" hscore={hs:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
