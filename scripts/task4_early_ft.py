from __future__ import annotations

# Option 1: a short fine-tune of one encoder on one Task 4 classification task, logging validation
# accuracy after every epoch, for the recommendation-level reversal test on the NLP benchmark
# (docs/option1_nlp_design.md, approved by Prof. Hazra). The training examples are exactly the
# estimators' sample (scripts/task4_extract.py sample_indices, same task, size and seed), so the
# frozen scores and the early dynamics describe the same data. The encoder is fine-tuned in full
# with a linear head on its first-token or mean-pooled output, the two inputs of Bassignana et al.
# (2022), at their tuned learning rate (5e-5) and batch size (32). Validation: the official
# validation split where one exists (GLUE validation; Airline's dev file from the authors'
# converter), otherwise (AG News) rows of the training split outside the sample; at most 2,000
# rows, drawn once per task and seed and shared by every encoder. No test split is read. Needs
# torch, transformers and datasets (envs/t4); runs on GPU.
#
#   python scripts/task4_early_ft.py --model bert-base-uncased --task rte --pooling cls \
#       --seed 42 --out_dir /projects/algl/dai.hany/task4/early_ft

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "task4_extract", Path(__file__).resolve().parent / "task4_extract.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)

N_CLASSES = {"agnews": 4, "mnli": 3, "qnli": 2, "rte": 2, "airline": 3}
VAL_SPLIT = {"mnli": "validation_matched", "qnli": "validation", "rte": "validation"}
# Airline's dev split from the authors' converter (sentiment/convert.py -rs 4012, job 10709670);
# only its md5 prefix was recorded on 2026-09-30, so the prefix is what is checked.
AIRLINE_DEV = ("/projects/algl/dai.hany/task4/data/airline/airline-dev.csv", "3d396e48")
N_VAL = 2000
VAL_SEED = 0


def out_path(out_dir: str, task: str, model: str, pooling: str, seed: int) -> Path:
    return Path(out_dir) / task / f"{ex.slug(model)}_{pooling}_s{seed}.json"


def val_indices(n_val_pool: int, exclude: np.ndarray | None, seed: int) -> np.ndarray:
    """Validation rows: at most N_VAL of the pool, minus any training rows, the same for every
    encoder (a function of the pool, the excluded rows and the seed only)."""
    pool = np.arange(n_val_pool)
    if exclude is not None:
        pool = np.setdiff1d(pool, exclude)
    rng = np.random.default_rng(VAL_SEED + 1000 * seed)
    return np.sort(rng.choice(pool, size=min(N_VAL, pool.size), replace=False))


def check_epochs(record: dict, epochs: int) -> None:
    got = [e["epoch"] for e in record["epochs"]]
    if got != list(range(1, epochs + 1)):
        raise ValueError(f"epochs {got}, expected 1..{epochs}")
    for e in record["epochs"]:
        if not 0.0 <= e["val_acc"] <= 1.0 or e["train_loss"] != e["train_loss"]:
            raise ValueError(f"bad epoch record {e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ex.MODELS)
    ap.add_argument("--task", required=True, choices=sorted(ex.TASKS))
    ap.add_argument("--pooling", required=True, choices=("cls", "mean"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)

    import torch
    import transformers
    from datasets import Dataset, load_dataset
    from transformers import AutoModel, AutoTokenizer

    out = out_path(a.out_dir, a.task, a.model, a.pooling, a.seed)
    if out.exists():
        print(f"skip (exists): {out}")
        return 0
    hub, cfg, (col_a, col_b) = ex.TASKS[a.task]
    if hub == "local":
        path, md5, col, allowed = ex.LOCAL[a.task]
        texts, labels = ex.read_local(path, md5, col, allowed)
        train = Dataset.from_dict({col: texts, "label": labels.tolist()})
        dpath, dprefix = AIRLINE_DEV
        if not hashlib.md5(Path(dpath).read_bytes()).hexdigest().startswith(dprefix):
            sys.exit(f"ABORT: {dpath} is not the recorded dev split")
        dtexts, dlabels = ex.read_local(dpath, hashlib.md5(Path(dpath).read_bytes()).hexdigest(),
                                        col, allowed)
        valpool = Dataset.from_dict({col: dtexts, "label": dlabels.tolist()})
        exclude = None
    else:
        train = load_dataset(hub, cfg, split="train")
        if a.task in VAL_SPLIT:
            valpool, exclude = load_dataset(hub, cfg, split=VAL_SPLIT[a.task]), None
        else:
            valpool = train
    labels = np.asarray(train["label"])
    if (labels < 0).any():
        sys.exit(f"ABORT: {a.task} train split has unlabeled rows")
    idx = ex.sample_indices(len(train), a.n, a.seed)
    if a.task not in VAL_SPLIT and hub != "local":
        exclude = idx
    vidx = val_indices(len(valpool), exclude, a.seed)
    tr, va = train.select(idx.tolist()), valpool.select(vidx.tolist())
    vlab = np.asarray(va["label"])
    if (vlab < 0).any():
        sys.exit(f"ABORT: {a.task} validation rows are unlabeled")
    majority = float(np.bincount(vlab).max() / vlab.size)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = AutoModel.from_pretrained(a.model).to(dev)
    head = torch.nn.Linear(enc.config.hidden_size, N_CLASSES[a.task]).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=a.lr,
                            weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()

    def logits(batch):
        x = tok(batch[col_a], batch[col_b] if col_b else None, truncation=True,
                max_length=a.max_length, padding=True, return_tensors="pt").to(dev)
        h = enc(**x).last_hidden_state
        if a.pooling == "cls":
            z = h[:, 0]
        else:
            m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
            z = (h * m).sum(1) / m.sum(1)
        return head(z)

    gen = torch.Generator().manual_seed(a.seed)
    record = {"model": a.model, "task": a.task, "pooling": a.pooling, "seed": a.seed,
              "n_train": len(tr), "n_val": len(va), "val_majority": majority, "train_idx_md5":
              hashlib.md5(idx.tobytes()).hexdigest(), "val_idx_md5":
              hashlib.md5(vidx.tobytes()).hexdigest(), "lr": a.lr, "batch_size": a.batch_size,
              "max_length": a.max_length, "transformers": transformers.__version__,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "epochs": []}
    t_all = time.time()
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        enc.train()
        head.train()
        order = torch.randperm(len(tr), generator=gen).tolist()
        losses = []
        for i in range(0, len(order), a.batch_size):
            b = tr.select(order[i:i + a.batch_size])
            y = torch.tensor(b["label"], device=dev)
            loss = torch.nn.functional.cross_entropy(logits(b[:]), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        t_train = time.time() - t0
        enc.eval()
        head.eval()
        correct, vloss = 0, 0.0
        with torch.no_grad():
            for i in range(0, len(va), 64):
                b = va[i:i + 64]
                y = torch.tensor(b["label"], device=dev)
                lg = logits(b)
                vloss += torch.nn.functional.cross_entropy(lg, y, reduction="sum").item()
                correct += (lg.argmax(-1) == y).sum().item()
        record["epochs"].append({"epoch": ep, "train_loss": float(np.mean(losses)),
                                 "val_acc": correct / len(va), "val_loss": vloss / len(va),
                                 "train_s": t_train, "epoch_s": time.time() - t0})
        print(f"epoch {ep}: train_loss={np.mean(losses):.4f} val_acc={correct / len(va):.4f} "
              f"epoch_s={time.time() - t0:.1f}", flush=True)
    record["total_s"] = time.time() - t_all
    record["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
    check_epochs(record, a.epochs)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.rename(out)
    print(f"ok: {out} total_s={record['total_s']:.1f} peak_gpu_mb={record['peak_gpu_mb']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
