from __future__ import annotations

# MRAP Task 4: score saved features with the same H-score and LogME code as Task 3
# (src/estimators, numpy only), on the CPU partition. Multiclass LogME is the mean evidence over
# one-hot label columns, as in You et al. (2021) and the Bassignana et al. toolkit. Records are
# appended to --out as JSON lines; a feature file already scored is skipped, so reruns are safe.
#
#   PYTHONPATH=. python scripts/task4_score.py --features /projects/algl/dai.hany/task4/features \
#       --out /projects/algl/dai.hany/task4/task4_scores.jsonl

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.estimators.hscore import hscore
from src.estimators.logme import logme


def logme_multiclass(F: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("LogME needs at least two label classes")
    parts = [logme(F, (y == c).astype(np.float64)) for c in classes]
    return float(np.mean([p[0] for p in parts])), {
        "classes": int(classes.size), "converged": all(p[1].get("converged", True) for p in parts)}


ESTIMATORS = {
    "hscore": lambda F, y: hscore(F, y),
    "hscore_shrunk": lambda F, y: hscore(F, y, shrink=True),
    "logme": logme_multiclass,
}


def score_file(path: Path) -> list[dict]:
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path)
    y = z["labels"]
    recs = []
    for pooling in ("cls", "mean"):
        F = z[pooling]
        if not np.isfinite(F).all():
            sys.exit(f"ABORT: non-finite features in {path} ({pooling})")
        for est, fn in ESTIMATORS.items():
            t0 = time.time()
            score, info = fn(F, y)
            recs.append({"estimator": est, "task": meta["task"], "model": meta["model"],
                         "pooling": pooling, "seed": meta["seed"], "score": float(score),
                         "score_direction": "higher_is_better", "n": int(y.size),
                         "dim": int(F.shape[1]), "scoring_s": time.time() - t0,
                         "extract_s": meta["extract_s"], "peak_gpu_mb": meta["peak_gpu_mb"],
                         "converged": bool(info.get("converged", True)),
                         "file": f"{path.parent.name}/{path.name}"})
    return recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(ln)["file"] for ln in out.read_text().splitlines() if ln.strip()}
    files = sorted(p for p in Path(a.features).glob("*/*.npz") if ".tmp" not in p.name)
    if not files:
        sys.exit(f"ABORT: no feature files under {a.features}")
    new = 0
    with out.open("a") as fh:
        for p in files:
            if f"{p.parent.name}/{p.name}" in done:
                continue
            for r in score_file(p):
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            new += 1
    print(f"{len(files)} feature files, {new} newly scored, {len(done)} already scored; out {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
