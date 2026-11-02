from __future__ import annotations

# PARC on the saved NLP features of Task 4 (the same files H-score and LogME scored), on the CPU.
# Settings are fixed in docs/parc_design.md (probe 500 with at least two per class, PCA to 32, no
# depth heuristic); the probe seed is the feature file's sample seed, so every encoder at a seed
# shares one probe. Records match scripts/task4_score.py with estimator "parc"; a feature file
# already scored is skipped, so reruns are safe.
#
#   PYTHONPATH=. python scripts/task4_score_parc.py \
#       --features /projects/algl/dai.hany/task4/features \
#       --out /projects/algl/dai.hany/task4/task4_parc_scores.jsonl

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.estimators.parc import DIM, N_PROBE, parc


def score_file(path: Path) -> list[dict]:
    meta = json.loads(path.with_suffix(".json").read_text())
    z = np.load(path)
    y = z["labels"]
    recs = []
    for pooling in ("cls", "mean"):
        F = z[pooling]
        if not np.isfinite(F).all():
            sys.exit(f"ABORT: non-finite features in {path} ({pooling})")
        t0 = time.time()
        score, info = parc(F, y, n_probe=N_PROBE, dim=DIM, seed=int(meta["seed"]))
        recs.append({"estimator": "parc", "task": meta["task"], "model": meta["model"],
                     "pooling": pooling, "seed": meta["seed"], "score": float(score),
                     "score_direction": "higher_is_better", "n": int(y.size),
                     "n_probe": info["n_probe"], "dim": int(F.shape[1]), "pca_dim": info["dim"],
                     "scoring_s": time.time() - t0, "extract_s": meta["extract_s"],
                     "peak_gpu_mb": meta["peak_gpu_mb"], "converged": True,
                     "file": f"{path.parent.name}/{path.name}"})
    return recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    files = sorted(p for p in Path(a.features).glob("*/*.npz") if ".tmp" not in p.name)
    if not files:
        sys.exit(f"ABORT: no feature files under {a.features}")
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {json.loads(ln)["file"] for ln in out.read_text().splitlines() if ln.strip()}
    n = 0
    with out.open("a") as fh:
        for p in files:
            if f"{p.parent.name}/{p.name}" in done:
                continue
            for r in score_file(p):
                fh.write(json.dumps(r) + "\n")
                n += 1
    print(f"wrote {n} PARC records from {len(files)} feature files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
