from __future__ import annotations

# MRAP Task 4: frozen last-layer features of one encoder on one classification task, for H-score and
# LogME scoring on the CPU partition. The sample depends only on the task, its size and the seed,
# so every encoder is scored on the same training examples. Features are the first token ([CLS] or
# <s>) and the mean over non-padding subwords, the two inputs of Bassignana et al. (2022). Needs
# transformers and datasets (the envs/t4 environment); runs on GPU.
#
#   python scripts/task4_extract.py --model bert-base-uncased --task rte --seed 42 \
#       --out_dir /projects/algl/dai.hany/task4/features

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# task -> (hub id, config, (first text column, second text column or None))
TASKS = {
    "agnews": ("fancyzhx/ag_news", None, ("text", None)),
    "mnli": ("nyu-mll/glue", "mnli", ("premise", "hypothesis")),
    "qnli": ("nyu-mll/glue", "qnli", ("question", "sentence")),
    "rte": ("nyu-mll/glue", "rte", ("sentence1", "sentence2")),
}
MODELS = ("bert-base-uncased", "roberta-base", "distilbert-base-uncased",
          "emilyalsentzer/Bio_ClinicalBERT", "dmis-lab/biobert-v1.1",
          "cardiffnlp/twitter-roberta-base", "allenai/scibert_scivocab_uncased")


def sample_indices(n_total: int, n: int, seed: int) -> np.ndarray:
    """Candidate-independent: a function of the task size, the sample size and the seed only."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))


def slug(model: str) -> str:
    return model.replace("/", "__")


def out_path(out_dir: str, task: str, model: str, seed: int) -> Path:
    return Path(out_dir) / task / f"{slug(model)}_s{seed}.npz"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)

    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    out = out_path(a.out_dir, a.task, a.model, a.seed)
    if out.exists():
        print(f"skip (exists): {out}")
        return 0
    hub, cfg, (col_a, col_b) = TASKS[a.task]
    ds = load_dataset(hub, cfg, split="train")
    labels = np.asarray(ds["label"])
    if (labels < 0).any():
        sys.exit(f"ABORT: {a.task} train split has unlabeled rows")
    idx = sample_indices(len(ds), a.n, a.seed)
    sub = ds.select(idx.tolist())
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(a.device).eval()
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    cls_parts, mean_parts = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(sub), a.batch_size):
            batch = sub[i:i + a.batch_size]
            enc = tok(batch[col_a], batch[col_b] if col_b else None, truncation=True,
                      max_length=a.max_length, padding=True, return_tensors="pt").to(a.device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            cls_parts.append(h[:, 0].float().cpu().numpy())
            mean_parts.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if a.device.startswith("cuda") else 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, cls=np.concatenate(cls_parts), mean=np.concatenate(mean_parts),
             labels=labels[idx], idx=idx)
    tmp.rename(out)
    meta = {"model": a.model, "task": a.task, "seed": a.seed, "n": int(idx.size),
            "n_train": len(ds), "max_length": a.max_length, "extract_s": secs,
            "peak_gpu_mb": peak, "transformers": transformers.__version__,
            "torch": torch.__version__, "hub": hub, "config": cfg}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"ok: {out} n={idx.size} dim={cls_parts[0].shape[1]} extract_s={secs:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
