#!/usr/bin/env python3
"""Few-shot fine-tuning proxy for transferability (access regime L).

Fine-tunes each candidate briefly on a small labelled set and scores it by AUC on a second small
set, both drawn from the target's TRAINING split under the seed; the validation and test splits
are never read, so the proxy cannot see the gold. The model class, epoch loop and learning-rate
schedule are imported from scripts/finetune_edubert.py and the encoder is loaded by that script's
own shape rule, so the proxy runs the real fine-tune in miniature; with the same seed it starts
from the same initialization as the real run. Every candidate at a seed sees the same learners.

  PYTHONPATH=. python scripts/fewshot_proxy.py --target_dir ../processed/junyi \\
      --candidates scratch ../checkpoints/edubert_junyi_pretrain_full_encoder.pt \\
      --n_fit 50 200 --n_eval 100 --epochs 5 --seeds 42 1 2 --out tg1_fewshot_kt_junyi.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import InteractionDataset, collate_fn
from src.estimators.base import EstimatorResult
from src.eval.metrics import auc
from src.utils import set_seed

_FT = Path(__file__).resolve().parent / "finetune_edubert.py"
_spec = importlib.util.spec_from_file_location("finetune_edubert", _FT)
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)


def draw_learners(n_train: int, n_fit: int, n_eval: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint fit and scoring indices into the training split, fixed by the seed alone."""
    if n_fit + n_eval > n_train:
        raise ValueError(f"need {n_fit} + {n_eval} learners, the training split has {n_train}")
    order = np.random.default_rng([seed, 7]).permutation(n_train)
    return order[:n_fit], order[n_fit:n_fit + n_eval]


def load_like_finetune(model, ckpt_path: str, device: str) -> dict:
    """finetune_edubert.py's rule: keep every tensor whose shape matches the target model."""
    ck = torch.load(ckpt_path, map_location=device)
    state = ck["model_state"]
    tgt = model.backbone.state_dict()
    keep = {k: v for k, v in state.items() if k in tgt and v.shape == tgt[k].shape}
    model.backbone.load_state_dict(keep, strict=False)
    return {"loaded": len(keep), "total": len(state), "skipped": sorted(set(state) - set(keep))}


def run_proxy(candidate: str | None, target_dir: str, *, seed: int, n_fit: int, n_eval: int,
              epochs: int, batch_size: int = 64, lr: float = 1e-3, warmup_frac: float = 0.1,
              device: str = "cpu") -> tuple[float, dict]:
    train = InteractionDataset(target_dir, "train", 512)
    fit_idx, eval_idx = draw_learners(len(train), n_fit, n_eval, seed)
    set_seed(seed)  # as finetune_edubert.main: the model is the next consumer of the torch RNG
    model = ft.EduBERTForKT(num_skills=ft.infer_num_skills(target_dir)).to(device)
    load = {"loaded": 0, "total": 0, "skipped": []}
    if candidate is not None:
        load = load_like_finetune(model, candidate, device)
    fit = DataLoader(Subset(train, fit_idx.tolist()), batch_size=batch_size, shuffle=True,
                     collate_fn=collate_fn)
    ev = DataLoader(Subset(train, eval_idx.tolist()), batch_size=batch_size, shuffle=False,
                    collate_fn=collate_fn)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=ft.make_lr_lambda(epochs * max(1, len(fit)), warmup_frac))
    curve = []
    for _ in range(epochs):
        ft.run_epoch(model, fit, device, opt, sched)
        _, yt, yp = ft.run_epoch(model, ev, device)
        curve.append(float(auc(yt, yp)))
    fit_positions = sum(max(int(train[i]["length"]) - 1, 0) for i in fit_idx.tolist())
    return max(curve), {"eval_auc_by_epoch": curve, "best_epoch": int(np.argmax(curve)) + 1,
                        "fit_positions": int(fit_positions),
                        "load": load, "fit_learners": int(n_fit), "eval_learners": int(n_eval),
                        "eval_positions": int(yt.size), "train_split_size": len(train)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--n_fit", nargs="+", type=int, default=[50, 200])
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--target_budget", default="n3000",
                    help="the benchmark cell this proxy is scored against")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target = Path(a.target_dir).name
    with open(a.out, "a") as fh:
        for n_fit in a.n_fit:
            est = f"fewshot_ft_fit{n_fit}_e{a.epochs}"
            for seed in a.seeds:
                for c in a.candidates:
                    ck = None if c == "scratch" else c
                    t0 = time.perf_counter()
                    score, info = run_proxy(ck, a.target_dir, seed=seed, n_fit=n_fit,
                                            n_eval=a.n_eval, epochs=a.epochs, device=device)
                    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    if device.startswith("cuda"):
                        peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 20)
                    r = EstimatorResult(
                        estimator=est, access_regime="L",
                        candidate="scratch" if ck is None else Path(c).name, target=target,
                        seed=seed, score=float(score), score_direction="higher_is_better",
                        n_target_examples=n_fit + a.n_eval,
                        n_target_labels=int(info["fit_positions"] + info["eval_positions"]),
                        feature_extraction_time=0.0, scoring_time=time.perf_counter() - t0,
                        peak_memory_mb=peak,
                        uncertainty_signals={"eval_auc_by_epoch": info["eval_auc_by_epoch"]},
                        metadata={**info, "target_budget": a.target_budget, "epochs": a.epochs,
                                  "candidate_path": c})
                    fh.write(json.dumps(r.to_json()) + "\n")
                    fh.flush()
                    print(f"FEWSHOT target={target} {est} seed={seed} cand={Path(c).name} "
                          f"auc={score:.4f} epochs={info['eval_auc_by_epoch']}", flush=True)


if __name__ == "__main__":
    main()
