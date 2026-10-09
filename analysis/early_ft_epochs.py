from __future__ import annotations

"""Per-candidate validation AUC at every fine-tuning epoch, for the reversal-aware pipeline.

Step 2 of the 2026-10-06 task needs early fine-tuning signals for every candidate on every held-out
target. Every Track B fine-tune (N=3000, 20 epochs) already logs its validation AUC per epoch, so
no new runs are needed: this script writes those values, one row per (target, candidate, seed,
epoch). It reuses load_runs() from analysis/finetune_trajectories.py, the parser D8 and the
2026-10-08 seven-target run (traj7) already used, with the same checks: one primary row per run,
every epoch present, the log's test AUC equal to the benchmark's, and the W&B best validation AUC
equal to the log's at 4 dp. Only validation learners appear here; the test value is carried for
reference and the features built from this file must not read it.

  PYTHONPATH=. python analysis/early_ft_epochs.py --executions benchmark_final/executions.tsv \
      --logdir . --wandb wandb_by_id.jsonl --out benchmark_final/early_ft_epochs.tsv
"""

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

from analysis.finetune_trajectories import load_runs, load_wandb

TARGETS = ("assist2017", "ednet", "junyi", "algebra2005", "bridge2006", "assist2009", "algebra2006")
CANDIDATES = 8
SEEDS = 6


def rows_from(runs: dict, epochs: int) -> list[dict]:
    out = []
    for (t, c, s), r in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        for e in range(1, epochs + 1):
            out.append({"target": t, "candidate": c, "finetune_seed": s, "epoch": e,
                        "val_auc": str(r["val"][e]), "best_val_auc": str(r["best_val"]),
                        "best_val_source": r["best_val_source"],
                        "test_auc_reference": str(r["test"]),
                        "log_file": r["log"]})
    return out


def check(runs: dict, targets: tuple) -> None:
    per_t = Counter(k[0] for k in runs)
    per_tc = Counter(k[:2] for k in runs)
    bad = [t for t in targets if per_t[t] != CANDIDATES * SEEDS]
    bad += [f"{t}/{c}" for (t, c), n in per_tc.items() if n != SEEDS]
    if bad:
        sys.exit(f"ABORT: expected {CANDIDATES} candidates x {SEEDS} seeds per target; off: {bad}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executions", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--wandb", default=None, help="wandb_by_id.jsonl, for full-precision best val")
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--budget", default="n3000")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = load_runs(a.executions, a.logdir, tuple(a.targets), a.budget, a.epochs,
                     load_wandb(a.wandb))
    check(runs, tuple(a.targets))
    rows = rows_from(runs, a.epochs)
    out = Path(a.out)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(runs)} runs, {len(rows)} rows; wrote {out} md5 "
          f"{hashlib.md5(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
