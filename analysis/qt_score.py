from __future__ import annotations

"""Score the Quick-Tune micro features with the NLP benchmark's estimators
(docs/qt_hscore_design.md, docs/trajectory_amendment2.md). Candidates are each task's rule A
models in trajectories.tsv; for seeds 42, 1 and 2 the estimators see a stratified half of every
class, the same images for every candidate. One JSON line per (task, seed, model, estimator).

  PYTHONPATH=. python analysis/qt_score.py --features /projects/algl/dai.hany/quicktune/features \
      --trajectories benchmark_final/trajectories.tsv \
      --out /projects/algl/dai.hany/quicktune/qt_scores.jsonl
"""

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

SEEDS = (42, 1, 2)
TRAJ_MD5 = "e93329cbc69ae8a3af843c544d43793b"


def estimators() -> dict:
    path = Path(__file__).resolve().parents[1] / "scripts" / "task4_score.py"
    spec = importlib.util.spec_from_file_location("task4_score", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ESTIMATORS


def rule_a_candidates(trajectories: Path) -> dict[str, list[str]]:
    out = collections.defaultdict(set)
    for r in csv.DictReader(trajectories.read_text().splitlines(), delimiter="\t"):
        if r["benchmark"] == "quicktune_micro":
            out[r["target"].rsplit("/", 1)[1]].add(r["candidate"])
    return {t: sorted(c) for t, c in out.items()}


def half(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    idx = []
    for c in sorted(set(labels.tolist())):
        members = [i for i, y in enumerate(labels.tolist()) if y == c]
        idx += rng.sample(members, len(members) // 2)
    return np.array(sorted(idx))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pin", action="store_true", help="tests only: skip the md5 pin")
    a = ap.parse_args(argv)
    tp = Path(a.trajectories)
    got = hashlib.md5(tp.read_bytes()).hexdigest()
    if not a.no_pin and got != TRAJ_MD5:
        sys.exit(f"ABORT: {tp} md5 {got}, expected {TRAJ_MD5}")
    est = estimators()
    cands = rule_a_candidates(tp)
    if len(cands) != 30:
        sys.exit(f"ABORT: {len(cands)} micro tasks with rule A runs, expected 30")
    out = Path(a.out)
    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["seed"], r["model"], r["estimator"]))
    feats = Path(a.features)
    with out.open("a") as fh:
        for task, models in sorted(cands.items()):
            z = {m: np.load(feats / task / f"{m}.npz") for m in models}
            ref = z[models[0]]
            for m in models[1:]:
                if not (np.array_equal(z[m]["files"], ref["files"])
                        and np.array_equal(z[m]["labels"], ref["labels"])):
                    sys.exit(f"ABORT: {task} {m}: files or labels differ from {models[0]}")
            y_all = ref["labels"]
            for seed in SEEDS:
                idx = half(y_all, seed)
                y = y_all[idx]
                for m in models:
                    F = z[m]["features"][idx].astype(np.float64)
                    for name, fn in est.items():
                        if (task, seed, m, name) in done:
                            continue
                        t0 = time.perf_counter()
                        try:
                            score, info = fn(F, y)
                            score = float(score)
                        except ValueError as e:
                            # LogME is undefined when the features interpolate the labels
                            # exactly (feature dimension at least the number of images).
                            score, info = None, {"error": str(e)}
                        fh.write(json.dumps({"task": task, "seed": seed, "model": m,
                                             "estimator": name, "score": score,
                                             "n": int(F.shape[0]), "dim": int(F.shape[1]),
                                             "seconds": time.perf_counter() - t0,
                                             "info": {k: v for k, v in info.items()
                                                      if isinstance(v, (int, float, bool, str))}})
                                 + "\n")
                fh.flush()
            print(f"{task}: {len(models)} candidates scored at seeds {SEEDS}", flush=True)
    print(f"wrote {out.name} md5 {hashlib.md5(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
