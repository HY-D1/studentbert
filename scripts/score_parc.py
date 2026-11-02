#!/usr/bin/env python3
"""PARC on one StudentBERT target, on the same leakage-safe causal KT features as every other
estimator (scripts/score_task2.features_like_logme: same learner draw, same position cap).

Settings are fixed in docs/parc_design.md (probe 500 with at least two per class, PCA to 32, no
depth heuristic); the probe seed is the estimator seed, so every candidate at a seed shares one
probe. Records follow the EstimatorResult schema, estimator name parc_kt_causal.

  PYTHONPATH=. python scripts/score_parc.py --target_dir ../processed/junyi \\
      --candidates scratch ../checkpoints/edubert_junyi_pretrain_full_encoder.pt \\
      --n_students 3000 --seeds 42 1 2 --out tg1_parc_kt_junyi.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_task2 import features_like_logme  # noqa: E402

from src.estimators.base import EstimatorResult  # noqa: E402
from src.estimators.kt_logme import _peak_mb  # noqa: E402
from src.estimators.parc import DIM, N_PROBE, parc  # noqa: E402
from src.estimators.protocol import add_protocol_args, resolve_protocol  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--n_students", type=int, default=None)
    ap.add_argument("--max_positions", type=int, default=50000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    add_protocol_args(ap)
    a = ap.parse_args()
    split, n_draw, extra = resolve_protocol(a)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target = Path(a.target_dir).name
    with open(a.out, "a") as fh:
        for seed in a.seeds:
            for c in a.candidates:
                ck = None if c == "scratch" else c
                feats, y, meta, t_ext = features_like_logme(
                    ck, a.target_dir, seed=seed, n_students=n_draw,
                    max_positions=a.max_positions, device=device, split=split)
                s0 = time.perf_counter()
                score, info = parc(feats, y, n_probe=N_PROBE, dim=DIM, seed=seed)
                r = EstimatorResult(
                    estimator="parc_kt_causal", access_regime="F",
                    candidate="scratch" if ck is None else Path(c).name, target=target,
                    seed=seed, score=float(score), score_direction="higher_is_better",
                    n_target_examples=int(y.size), n_target_labels=int(info["n_probe"]),
                    feature_extraction_time=t_ext, scoring_time=time.perf_counter() - s0,
                    peak_memory_mb=_peak_mb(device), uncertainty_signals=info,
                    metadata={**meta, **extra, "feature_dim": int(feats.shape[1])})
                fh.write(json.dumps(r.to_json()) + "\n")
                fh.flush()
                print(f"PARC target={target} seed={seed} cand={Path(c).name} parc={score:.5f}",
                      flush=True)


if __name__ == "__main__":
    main()
