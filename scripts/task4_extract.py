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
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

# task -> (hub id, config, (first text column, second text column or None)); hub "local"
# reads the split pinned in LOCAL instead of the Hugging Face hub
TASKS = {
    "agnews": ("fancyzhx/ag_news", None, ("text", None)),
    "mnli": ("nyu-mll/glue", "mnli", ("premise", "hypothesis")),
    "qnli": ("nyu-mll/glue", "qnli", ("question", "sentence")),
    "rte": ("nyu-mll/glue", "rte", ("sentence1", "sentence2")),
    "airline": ("local", None, ("text", None)),
}
# Local tasks read a train split made outside this repo: task -> (file, md5, text column,
# labels). airline: the authors' own converter (mainlp/logme-nlp, commit 0046c725,
# sentiment/convert.py -rs 4012) on Kaggle's Tweets.csv (md5
# 2fa808ea99b32814ccd64d7097d935f2), run in a separate clone because that code is GPL-3.0;
# only its output is read here. SciERC is excluded: its entity-marked splits were never
# released (mrap_task4_crossdomain.md).
LOCAL = {
    "airline": ("/projects/algl/dai.hany/task4/data/airline/airline-train.csv",
                "b0fa4865e8d442b5e2146715e7a84c10", "text", (0, 1, 2)),
}
MODELS = ("bert-base-uncased", "roberta-base", "distilbert-base-uncased",
          "emilyalsentzer/Bio_ClinicalBERT", "dmis-lab/biobert-v1.1",
          "cardiffnlp/twitter-roberta-base", "allenai/scibert_scivocab_uncased")


def sample_indices(n_total: int, n: int, seed: int) -> np.ndarray:
    """Candidate-independent: a function of the task size, the sample size and the seed only."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))


def read_local(path: str, md5: str, col: str,
               allowed: tuple[int, ...]) -> tuple[list[str], np.ndarray]:
    """Texts and integer labels of a local CSV split; aborts on a different or malformed file."""
    p = Path(path)
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing")
    raw = p.read_bytes()
    got = hashlib.md5(raw).hexdigest()
    if got != md5:
        sys.exit(f"ABORT: {p} md5 {got}, expected {md5}")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    texts = [r[col] for r in rows]
    labels = np.asarray([int(r["label"]) for r in rows])
    if not rows or any(not t for t in texts) or not set(labels.tolist()) <= set(allowed):
        sys.exit(f"ABORT: {p} has no rows, an empty text or a label outside {allowed}")
    return texts, labels


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
    from datasets import Dataset, load_dataset
    from transformers import AutoModel, AutoTokenizer

    out = out_path(a.out_dir, a.task, a.model, a.seed)
    if out.exists():
        print(f"skip (exists): {out}")
        return 0
    hub, cfg, (col_a, col_b) = TASKS[a.task]
    source = {"hub": hub, "config": cfg}
    if hub == "local":
        path, md5, col, allowed = LOCAL[a.task]
        texts, labels = read_local(path, md5, col, allowed)
        ds = Dataset.from_dict({col: texts, "label": labels.tolist()})
        source.update(data_file=path, data_md5=md5)
    else:
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
            "torch": torch.__version__, **source}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"ok: {out} n={idx.size} dim={cls_parts[0].shape[1]} extract_s={secs:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
