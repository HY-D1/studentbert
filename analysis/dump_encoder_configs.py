#!/usr/bin/env python3
"""Write every encoder checkpoint's saved recipe, loss and md5 to one TSV.

pretrain_edubert.py stores vars(args) as "config" inside each encoder, so the recipe of every
candidate is recoverable from the file even where no job definition was committed (the two EdNet
objective encoders, for instance). An empty objective means the file predates --objective, when
the loss was always skill plus correctness, so it is reported as full (implied).

Run under srun: a torch process on the login node was killed on 2026-09-22.
  srun --mem=8G --time=00:20:00 /projects/algl/dai.hany/envs/sb/bin/python \
      analysis/dump_encoder_configs.py --ckpt-dir ../checkpoints --out encoder_configs.tsv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import torch

FIELDS = ("objective", "epochs", "batch_size", "lr", "warmup_frac", "dropout", "mask_ratio",
          "max_seq_len", "d_model", "n_layers", "seed", "n_students", "max_interactions",
          "processed_dir", "run_type")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="../checkpoints")
    ap.add_argument("--out", default="encoder_configs.tsv")
    a = ap.parse_args()
    paths = sorted(Path(a.ckpt_dir).glob("*_encoder.pt"))
    if not paths:
        raise SystemExit(f"no *_encoder.pt under {a.ckpt_dir}")
    cols = ["name", "mtime_utc", "bytes", "md5", "epoch", "mlm_loss", "num_skills",
            "objective_effective", *FIELDS]
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for p in paths:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            cfg = ck.get("config") or {}
            obj = cfg.get("objective")
            row = [p.name, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)),
                   p.stat().st_size, md5sum(p), ck.get("epoch"), repr(ck.get("mlm_loss")),
                   ck.get("num_skills"), obj if obj else "full (implied: predates --objective)"]
            row += [cfg.get(k, "") for k in FIELDS]
            w.writerow(["" if v is None else v for v in row])
            print(f"{p.name}  epoch={ck.get('epoch')}  objective={row[7]}  "
                  f"batch={cfg.get('batch_size')}  warmup={cfg.get('warmup_frac')}", flush=True)
            del ck
    print(f"wrote {a.out} ({len(paths)} encoders)")


if __name__ == "__main__":
    main()
