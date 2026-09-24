#!/usr/bin/env python3
"""Task 2 frozen-feature estimators on one target: H-score (plain and shrunk) and NLEEP.

Features are the leakage-safe causal KT features the LogME scorer uses, extracted by the same
functions in the same order (same learner draw, same position cap), so every estimator here and
LogME score identical inputs for a given (candidate, seed). Access regime F: target inputs and
labels, no fine-tuning.

  PYTHONPATH=. python scripts/score_task2.py --target_dir ../processed/junyi \\
      --candidates scratch ../checkpoints/edubert_junyi_pretrain_full_encoder.pt \\
      --n_students 3000 --seeds 42 1 2 --out tg1_task2_kt_junyi.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.estimators.base import EstimatorResult
from src.estimators.features import (build_backbone, cap_positions, kt_features, load_candidate,
                                     sample_target, target_num_skills)
from src.estimators.hscore import hscore
from src.estimators.kt_logme import _peak_mb
from src.estimators.nleep import nleep


def features_like_logme(candidate, target_dir, *, seed, n_students, max_positions, device):
    """The exact feature path of src/estimators/kt_logme.score_kt_logme."""
    target = Path(target_dir).name
    t0 = time.perf_counter()
    bb = build_backbone(target_num_skills(target_dir), seed=seed)
    load = ({"source": "none", "in_domain": False, "loaded": 0, "total": 0, "skipped": []}
            if candidate is None else load_candidate(bb, candidate, target, device="cpu"))
    bb.to(device)
    subset, rows, fingerprint = sample_target(target_dir, n_students, seed)
    feats, y, _ = kt_features(bb, subset, device)
    n_all = int(y.size)
    sel = cap_positions(n_all, max_positions, seed)
    meta = {"sample_fingerprint": fingerprint, "n_students": len(rows),
            "n_students_requested": n_students, "positions_total": n_all,
            "positions_used": int(sel.size), "load": load, "device": device,
            "candidate_path": candidate or "scratch"}
    return feats[sel], y[sel], meta, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--n_students", type=int, default=None)
    ap.add_argument("--max_positions", type=int, default=50000)
    ap.add_argument("--nleep_k", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--nleep_dim", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target = Path(a.target_dir).name
    with open(a.out, "a") as fh:
        for seed in a.seeds:
            for c in a.candidates:
                ck = None if c == "scratch" else c
                feats, y, meta, t_ext = features_like_logme(
                    ck, a.target_dir, seed=seed, n_students=a.n_students,
                    max_positions=a.max_positions, device=device)
                jobs = [("hscore_kt_causal", lambda: hscore(feats, y)),
                        ("hscore_shrunk_kt_causal", lambda: hscore(feats, y, shrink=True))]
                jobs += [(f"nleep_k{k}_kt_causal",
                          lambda k=k: nleep(feats, y, k=k, dim=a.nleep_dim, seed=seed))
                         for k in a.nleep_k]
                line = []
                for est, fn in jobs:
                    s0 = time.perf_counter()
                    score, info = fn()
                    r = EstimatorResult(
                        estimator=est, access_regime="F",
                        candidate="scratch" if ck is None else Path(c).name, target=target,
                        seed=seed, score=float(score), score_direction="higher_is_better",
                        n_target_examples=int(y.size), n_target_labels=int(y.size),
                        feature_extraction_time=t_ext, scoring_time=time.perf_counter() - s0,
                        peak_memory_mb=_peak_mb(device), uncertainty_signals=info,
                        metadata={**meta, "feature_dim": int(feats.shape[1])})
                    fh.write(json.dumps(r.to_json()) + "\n")
                    line.append(f"{est.replace('_kt_causal', '')}={score:.5f}")
                fh.flush()
                print(f"TASK2 target={target} seed={seed} cand={Path(c).name} " + " ".join(line),
                      flush=True)


if __name__ == "__main__":
    main()
