"""Early-dropout label for ASSISTments (leakage-free version).

Label = bottom-quantile total engagement, computed ONLY among students with
>= min_interactions total. This pairs with the early-prefix input in
downstream_edubert.py (model sees first K interactions; must predict eventual
disengagement). Computing the quantile on the eligible cohort prevents the
label from being a trivial function of "had a short sequence".

NOTE: the literal "30-day inactivity" rule is degenerate on this 981-day
dataset (flags 61% of students; median last-activity gap 428 days = collection
simply ended), so we use cohort-relative bottom-quantile total interactions as
a standard operationalization of disengagement.

Output: ../processed/assist2017/dropout_labels.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--quantile", type=float, default=0.25)
    ap.add_argument("--min_interactions", type=int, default=20,
                    help="must match K (prefix length) in downstream_edubert.py")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=["studentId", "skill", "correct", "startTime"])
    df = df.dropna(subset=["studentId", "skill", "correct", "startTime"])

    totals_all = df.groupby("studentId").size()
    eligible = totals_all[totals_all >= args.min_interactions]   # cohort with >= K
    thresh = eligible.quantile(args.quantile)
    dropped = (eligible <= thresh).astype(int)                   # bottom-quantile of eligible

    npz = np.load(Path(args.processed_dir) / "sequences.npz")
    npz_ids = set(int(s) for s in npz["student_ids"])
    # only label eligible students that are in the npz; others get no entry (excluded downstream)
    labels = {str(int(sid)): int(dropped.loc[sid]) for sid in dropped.index
              if int(sid) in npz_ids}

    pos = sum(labels.values()); n = len(labels)
    meta = {"definition": f"bottom-{args.quantile:.0%} total interactions among "
                          f">= {args.min_interactions}-interaction cohort",
            "min_interactions": args.min_interactions,
            "threshold_interactions": float(thresh),
            "n_eligible_labeled": n, "n_students_in_npz": len(npz_ids),
            "positive_count": pos, "positive_rate": pos / n if n else 0.0}
    (Path(args.processed_dir) / "dropout_labels.json").write_text(
        json.dumps({**labels, "_meta": meta}))

    print(f"eligible students (>= {args.min_interactions} interactions): {len(eligible)} "
          f"of {len(totals_all)} total")
    print(f"labeled {n} eligible students in npz; "
          f"positive rate = {pos/n:.4f} ({pos} disengaged)")
    print(f"threshold = <= {thresh:.0f} total interactions (bottom {args.quantile:.0%} of eligible)")
    print(f"eligible totals: median={eligible.median():.0f}, "
          f"p25={eligible.quantile(0.25):.0f}, p75={eligible.quantile(0.75):.0f}")


if __name__ == "__main__":
    main()
