from __future__ import annotations

"""Frozen features of the Quick-Tune candidates on the Meta-Album micro images
(docs/qt_hscore_design.md). One .npz per (task, model) with the pooled pre-classifier features of
every image of the task, its labels and file names, plus one JSON line per (task, model) with the
time, GPU and peak memory. Existing outputs are skipped, so a job can be resumed.

  python analysis/qt_extract_features.py \
      --images /projects/algl/dai.hany/quicktune/metaalbum_micro \
      --quicktune /projects/algl/dai.hany/quicktune \
      --out /projects/algl/dai.hany/quicktune/features --tasks BCT
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def model_names(quicktune: Path) -> list[str]:
    cols = next(csv.reader((quicktune / "micro" / "micro" / "unnormalized_args_table.csv").open()))
    names = sorted(c[len("cat__model_"):] for c in cols if c.startswith("cat__model_"))
    if len(names) != 24:
        sys.exit(f"ABORT: {len(names)} Quick-Tune models, expected 24")
    return names


def task_images(root: Path, code: str) -> tuple[list[Path], list[str]]:
    labels = sorted((root / code).rglob("labels.csv"))
    if len(labels) != 1:
        sys.exit(f"ABORT: {code} has {len(labels)} labels.csv files")
    rows = sorted(csv.DictReader(labels[0].open()), key=lambda r: r["FILE_NAME"])
    paths = [labels[0].parent / "images" / r["FILE_NAME"] for r in rows]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        sys.exit(f"ABORT: {code}: {len(missing)} listed images missing, e.g. {missing[0]}")
    return paths, [r["CATEGORY"] for r in rows]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--quicktune", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=[], help="task codes; default all 30")
    ap.add_argument("--models", nargs="*", default=[], help="default all 24")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args(argv)
    import timm
    import torch
    from PIL import Image
    from timm.data import create_transform, resolve_data_config

    root, out = Path(a.images), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks or sorted(p.name for p in root.iterdir() if p.is_dir())
    names = a.models or model_names(Path(a.quicktune))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    log = out / "extract_log.jsonl"
    for code in tasks:
        paths, cats = task_images(root, code)
        classes = sorted(set(cats))
        y = np.array([classes.index(c) for c in cats], dtype=np.int64)
        for name in names:
            dst = out / code / f"{name}.npz"
            if dst.is_file():
                continue
            m = timm.create_model(name, pretrained=True, num_classes=0).eval().to(dev)
            cfg = resolve_data_config({}, model=m)
            tf = create_transform(**cfg)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            feats = []
            with torch.no_grad():
                for i in range(0, len(paths), a.batch):
                    x = torch.stack([tf(Image.open(p).convert("RGB"))
                                     for p in paths[i:i + a.batch]]).to(dev)
                    f = m.forward_head(m.forward_features(x), pre_logits=True)
                    feats.append(f.float().cpu().numpy())
            if dev == "cuda":
                torch.cuda.synchronize()
            secs = time.perf_counter() - t0
            F = np.concatenate(feats)
            if F.shape[0] != len(paths) or not np.isfinite(F).all():
                sys.exit(f"ABORT: {code} {name}: features {F.shape}, finite {np.isfinite(F).all()}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.stem + ".part.npz")
            np.savez(tmp, features=F, labels=y, files=np.array([p.name for p in paths]),
                     classes=np.array(classes))
            tmp.rename(dst)
            pc = m.pretrained_cfg
            rec = {"task": code, "model": name,
                   "pretrained": f"{pc.get('architecture')}.{pc.get('tag')}",
                   "input_size": list(cfg["input_size"]),
                   "n": int(F.shape[0]), "dim": int(F.shape[1]), "classes": len(classes),
                   "seconds": secs, "gpu": gpu, "batch": a.batch,
                   "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                                   if dev == "cuda" else 0.0),
                   "timm": timm.__version__, "torch": torch.__version__,
                   "md5": hashlib.md5(dst.read_bytes()).hexdigest()}
            with log.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{code} {name}: dim {F.shape[1]}, {secs:.1f} s, "
                  f"peak {rec['peak_gpu_mb']:.0f} MB", flush=True)
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
