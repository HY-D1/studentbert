"""Leakage-safe knowledge-tracing LogME for one candidate on one target (access regime F).

Two scores per call, from the same features: plain LogME on hidden[t] (the literature form, blind
to which skill comes next) and LogME with one readout per next skill (the form of the KT head).
Reporting both keeps the modelling choice visible instead of buried in one number.
"""

from __future__ import annotations

import resource
import time
from pathlib import Path

import torch

from src.estimators.base import EstimatorResult, rank_candidates
from src.estimators.features import (build_backbone, cap_positions, kt_features, load_candidate,
                                     sample_target, target_num_skills)
from src.estimators.logme import logme, logme_per_group

PLAIN = "logme_kt_causal"
PER_SKILL = "logme_kt_causal_per_skill"


def _peak_mb(device: str) -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux reports KB
    if device.startswith("cuda") and torch.cuda.is_available():
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
    return peak


def score_kt_logme(candidate: str | None, target_dir: str | Path, *, seed: int,
                   n_students: int | None, max_positions: int | None = 50000,
                   min_group: int = 20, device: str | None = None, d_model: int = 256,
                   n_layers: int = 6, max_seq_len: int = 512,
                   batch_size: int = 64, split: str = "train") -> list[EstimatorResult]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    target = Path(target_dir).name
    t0 = time.perf_counter()
    backbone = build_backbone(target_num_skills(target_dir), seed=seed, d_model=d_model,
                              n_layers=n_layers, max_len=max_seq_len)
    if candidate is None:
        load = {"source": "none", "in_domain": False, "loaded": 0, "total": 0, "skipped": []}
    else:
        load = load_candidate(backbone, candidate, target, device="cpu")
    backbone.to(device)
    subset, rows, fingerprint = sample_target(target_dir, n_students, seed, max_seq_len,
                                              split=split)
    F, y, nxt = kt_features(backbone, subset, device, batch_size=batch_size)
    n_all = int(y.size)
    sel = cap_positions(n_all, max_positions, seed)
    F, y, nxt = F[sel], y[sel], nxt[sel]
    t1 = time.perf_counter()
    s_plain, i_plain = logme(F, y)
    t2 = time.perf_counter()
    s_skill, i_skill = logme_per_group(F, y, nxt, min_group=min_group)
    t3 = time.perf_counter()

    name = "scratch" if candidate is None else Path(candidate).name
    shared = {"sample_fingerprint": fingerprint, "n_students": len(rows),
              "n_students_requested": n_students, "positions_total": n_all,
              "positions_used": int(y.size), "positive_rate": float(y.mean()),
              "load": load, "feature_dim": int(F.shape[1]), "device": device,
              "candidate_path": candidate or "scratch", "score_split": split}
    peak = _peak_mb(device)
    out = []
    for est, score, info, t_score in ((PLAIN, s_plain, i_plain, t2 - t1),
                                      (PER_SKILL, s_skill, i_skill, t3 - t2)):
        out.append(EstimatorResult(
            estimator=est, access_regime="F", candidate=name, target=target, seed=seed,
            score=float(score), score_direction="higher_is_better",
            n_target_examples=int(y.size), n_target_labels=int(y.size),
            feature_extraction_time=t1 - t0, scoring_time=t_score, peak_memory_mb=peak,
            uncertainty_signals={k: info[k] for k in ("alpha", "beta", "gamma", "iterations",
                                                      "converged")},
            metadata={**shared, **{k: v for k, v in info.items()
                                   if k not in ("alpha", "beta", "gamma", "iterations",
                                                "converged")}}))
    return out


def ranks_by_seed(results: list[EstimatorResult]) -> dict:
    """{(estimator, target, seed): [candidate, ...] best first}, for quick inspection."""
    groups: dict = {}
    for r in results:
        groups.setdefault((r.estimator, r.target, r.seed), []).append(r)
    return {k: [r.candidate for r in rank_candidates(v)] for k, v in sorted(groups.items())}

