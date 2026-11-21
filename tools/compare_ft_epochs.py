from __future__ import annotations

"""Compare 10-epoch NLP early fine-tunes with the recorded 3-epoch runs they rerun
(docs/nlp_budget_extension_design.md). Read-only. For every run in --new it checks that the run has
epochs 1 to --epochs and the same data and settings as the recorded run in --old (same relative
path), and reports how its first 3 epochs compare (identical validation accuracy or not, and the
largest difference) and its time per epoch. Differences in accuracy are reported, never fixed: they
come from GPU nondeterminism and do not stop anything. A missing epoch, a missing recorded run or a
change in data or settings stops with ABORT, since then the run is not a rerun of the recorded one.

  python tools/compare_ft_epochs.py --new /projects/algl/dai.hany/task4/early_ft10 \
      --old /projects/algl/dai.hany/task4/early_ft --expect 5
"""

import argparse
import json
import statistics as st
import sys
from pathlib import Path

SAME = ("model", "task", "pooling", "seed", "n_train", "n_val", "train_idx_md5", "val_idx_md5",
        "lr", "batch_size", "max_length")
OVERLAP = 3


def compare(new: dict, old: dict, epochs: int) -> dict:
    got = [e["epoch"] for e in new["epochs"]]
    if got != list(range(1, epochs + 1)):
        raise ValueError(f"epochs {got}, expected 1..{epochs}")
    if [e["epoch"] for e in old["epochs"]][:OVERLAP] != list(range(1, OVERLAP + 1)):
        raise ValueError("the recorded run lacks epochs 1 to 3")
    diff = [k for k in SAME if new.get(k) != old.get(k)]
    if diff:
        raise ValueError(f"data or settings differ from the recorded run: {diff}")
    d = [abs(a["val_acc"] - b["val_acc"]) for a, b in
         zip(new["epochs"][:OVERLAP], old["epochs"][:OVERLAP], strict=True)]
    return {"identical": max(d) == 0.0, "max_diff": max(d),
            "epoch_s": st.mean(e["epoch_s"] for e in new["epochs"]),
            "run_s": sum(e["epoch_s"] for e in new["epochs"]), "gpu": new.get("gpu", "")}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--expect", type=int, required=True, help="number of runs expected in --new")
    a = ap.parse_args(argv)
    new_dir, old_dir = Path(a.new), Path(a.old)
    files = sorted(new_dir.glob("*/*.json"))
    if len(files) != a.expect:
        sys.exit(f"ABORT: {len(files)} runs in {new_dir}, expected {a.expect}")
    rows = []
    for f in files:
        rel = f.relative_to(new_dir)
        old = old_dir / rel
        if not old.is_file():
            sys.exit(f"ABORT: no recorded run {old}")
        try:
            r = compare(json.loads(f.read_text()), json.loads(old.read_text()), a.epochs)
        except ValueError as e:
            sys.exit(f"ABORT: {rel}: {e}")
        rows.append((rel, r))
    print("| run | epochs 1 to 3 identical | largest val_acc difference | mean epoch s | gpu |")
    print("|---|---|---|---|---|")
    for rel, r in rows:
        print(f"| {rel} | {'yes' if r['identical'] else 'no'} | {r['max_diff']:.4f} | "
              f"{r['epoch_s']:.1f} | {r['gpu']} |")
    same = sum(r["identical"] for _, r in rows)
    print(f"runs: {len(rows)}, epochs 1 to 3 identical: {same}/{len(rows)}, largest difference "
          f"{max(r['max_diff'] for _, r in rows):.4f}; GPU-hours of these runs: "
          f"{sum(r['run_s'] for _, r in rows) / 3600:.3f}; GPUs: "
          f"{', '.join(sorted({r['gpu'] for _, r in rows}))}")


if __name__ == "__main__":
    main()
