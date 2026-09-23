#!/usr/bin/env python3
"""Diagnose where the leakage-safe KT LogME loses the transfer signal (MRAP Section T).

Two questions, each answered from the same learners and positions the main estimator scores:
  layer         Does an earlier layer carry what the final layer loses? Causal features are
                captured at the embedding output (L0) and after every encoder layer (L1 to L6)
                with forward hooks, so each layer is exactly what the fine-tune computes.
  skill table   Is in-domain credited for its trained skill table, which fine-tuning would learn
                anyway? In-domain candidates are scored a second time with that table left at the
                fine-tune's random start, as every cross-domain candidate already is
                (estimator names ending _skillrand).
The final layer must reproduce scripts/score_transferability.py exactly; the script stops if it
does not.

  PYTHONPATH=. python scripts/diagnose_logme.py --target_dir ../processed/junyi \\
      --candidates scratch ../checkpoints/edubert_junyi_pretrain_full_encoder.pt \\
      --n_students 3000 --seeds 42 1 2 --out tg1_logmediag_junyi.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import collate_fn
from src.estimators.base import EstimatorResult
from src.estimators.features import (build_backbone, cap_positions, encode_causal, kt_features,
                                     load_candidate, sample_target, target_num_skills)
from src.estimators.logme import logme, logme_per_group


def n_positions(subset) -> int:
    return sum(max(int(subset[i]["length"]) - 1, 0) for i in range(len(subset)))


@torch.no_grad()
def layer_features(backbone, subset, sel: np.ndarray, device: str, batch_size: int = 64):
    """Per-layer features at the globally selected KT positions, plus labels and next skills."""
    backbone.eval()
    store: list = []
    hooks = [layer.register_forward_hook(lambda _m, _i, out: store.append(out))
             for layer in backbone.encoder.layers]
    n_layers = len(backbone.encoder.layers) + 1
    feats = [[] for _ in range(n_layers)]
    labels, nexts = [], []
    offset = 0
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    try:
        for batch in loader:
            skill = batch["skill"].to(device)
            correct = batch["correct"].to(device)
            tb = batch["time_bin"].to(device)
            mask = batch["mask"].to(device)
            B, L = skill.shape
            pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
            emb = backbone.emb_drop(backbone.emb_norm(
                backbone.skill_emb(skill) + backbone.outcome_emb(correct)
                + backbone.time_emb(tb) + backbone.pos_emb(pos)))
            store.clear()
            h = encode_causal(backbone, skill, correct, tb, ~mask)
            if len(store) != n_layers - 1 or not torch.equal(store[-1], h):
                raise RuntimeError("hooked last layer differs from encode_causal output")
            nv = mask[:, 1:] & mask[:, :-1]
            count = int(nv.sum())
            lo, hi = np.searchsorted(sel, [offset, offset + count])
            keep = torch.as_tensor(sel[lo:hi] - offset, device=device)
            for i, x in enumerate([emb, *store]):
                feats[i].append(x[:, :-1][nv][keep].float().cpu().numpy())
            labels.append(correct[:, 1:][nv][keep].cpu().numpy())
            nexts.append(skill[:, 1:][nv][keep].cpu().numpy())
            offset += count
    finally:
        for hk in hooks:
            hk.remove()
    return ([np.concatenate(f) for f in feats], np.concatenate(labels).astype(np.int64),
            np.concatenate(nexts).astype(np.int64))


def score_layers(candidate, target_dir, *, seed, n_students, max_positions, min_group, device,
                 randomize_skill_table=False, d_model=256, n_layers=6, max_seq_len=512):
    target = Path(target_dir).name
    t0 = time.perf_counter()
    bb = build_backbone(target_num_skills(target_dir), seed=seed, d_model=d_model,
                        n_layers=n_layers, max_len=max_seq_len)
    load = {"source": "none", "in_domain": False, "loaded": 0, "total": 0, "skipped": []}
    if candidate is not None:
        # Randomizing means loading as if cross-domain: the vocabulary tensors keep the
        # fine-tune's random start at this seed, exactly what a foreign candidate gets.
        load = load_candidate(bb, candidate, "__not_" + target if randomize_skill_table
                              else target)
    bb.to(device)
    subset, rows, fp = sample_target(target_dir, n_students, seed, max_seq_len)
    sel = cap_positions(n_positions(subset), max_positions, seed)
    feats, y, nxt = layer_features(bb, subset, sel, device)
    t1 = time.perf_counter()
    out = []
    suffix = "_skillrand" if randomize_skill_table else ""
    for L, F in enumerate(feats):
        for est, fn in ((f"logme_kt_causal_L{L}{suffix}", lambda F=F: logme(F, y)),
                        (f"logme_kt_causal_per_skill_L{L}{suffix}",
                         lambda F=F: logme_per_group(F, y, nxt, min_group=min_group))):
            s0 = time.perf_counter()
            score, info = fn()
            out.append(EstimatorResult(
                estimator=est, access_regime="F",
                candidate="scratch" if candidate is None else Path(candidate).name,
                target=target, seed=seed, score=float(score), score_direction="higher_is_better",
                n_target_examples=int(y.size), n_target_labels=int(y.size),
                feature_extraction_time=t1 - t0, scoring_time=time.perf_counter() - s0,
                peak_memory_mb=None,
                uncertainty_signals={k: info[k] for k in ("alpha", "beta", "gamma", "converged")},
                metadata={"sample_fingerprint": fp, "n_students_requested": n_students,
                          "layer": L, "skill_table_randomized": randomize_skill_table,
                          "load": load, "positions_used": int(y.size)}))
    return out, (feats[-1], y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--n_students", type=int, default=None)
    ap.add_argument("--max_positions", type=int, default=50000)
    ap.add_argument("--min_group", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target = Path(a.target_dir).name
    with open(a.out, "a") as fh:
        for seed in a.seeds:
            for c in a.candidates:
                ck = None if c == "scratch" else c
                variants = [False]
                if ck is not None:
                    cfg = torch.load(ck, map_location="cpu", weights_only=False).get("config") or {}
                    if Path(cfg.get("processed_dir", "")).name == target:
                        variants.append(True)
                for rnd in variants:
                    rs, (f_last, y) = score_layers(ck, a.target_dir, seed=seed,
                                                   n_students=a.n_students,
                                                   max_positions=a.max_positions,
                                                   min_group=a.min_group, device=device,
                                                   randomize_skill_table=rnd)
                    if not rnd:
                        # The final layer must equal the main estimator's features exactly.
                        bb = build_backbone(target_num_skills(a.target_dir), seed=seed)
                        if ck is not None:
                            load_candidate(bb, ck, target)
                        bb.to(device)
                        sub, _, _ = sample_target(a.target_dir, a.n_students, seed)
                        F, yy, _ = kt_features(bb, sub, device)
                        sel = cap_positions(len(yy), a.max_positions, seed)
                        # GPU kernels need not repeat bit for bit, hence a tolerance, not equality.
                        if not (np.allclose(F[sel], f_last, rtol=0, atol=1e-6)
                                and np.array_equal(yy[sel], y)):
                            raise SystemExit(f"final layer differs from the estimator for {c} "
                                             f"seed {seed}")
                    for r in rs:
                        fh.write(json.dumps(r.to_json()) + "\n")
                    fh.flush()
                    plain = [r for r in rs if "per_skill" not in r.estimator]
                    print(f"DIAG target={target} seed={seed} cand={Path(c).name} skillrand={rnd} "
                          + " ".join(f"L{r.metadata['layer']}={r.score:.5f}" for r in plain),
                          flush=True)


if __name__ == "__main__":
    main()
