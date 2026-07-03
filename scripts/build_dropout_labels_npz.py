# Dataset-agnostic early-dropout label builder, computed from the processed
# sequences.npz (identical format across ASSISTments/EdNet/Junyi), so no
# raw-CSV schema needed. Label = bottom-quantile TOTAL interactions among
# students with >= min_interactions, restricted to students in the npz.
# Pairs with the early-prefix input in downstream_edubert.py.
#
# Usage:
#   python scripts/build_dropout_labels_npz.py --processed_dir ../processed/ednet \
#          --min_interactions 5 --quantile 0.25
# Writes <processed_dir>/dropout_labels.json
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--quantile", type=float, default=0.25)
    ap.add_argument("--min_interactions", type=int, default=5,
                    help="must be <= smallest K used downstream")
    args = ap.parse_args()

    d = Path(args.processed_dir)
    npz = np.load(d / "sequences.npz")
    ids = npz["student_ids"]          # (N,)
    off = npz["offsets"]              # (N+1,) CSR
    totals = (off[1:] - off[:-1]).astype(np.int64)   # per-student total interactions

    elig_mask = totals >= args.min_interactions
    elig_tot = totals[elig_mask]
    if len(elig_tot) == 0:
        raise SystemExit("no eligible students; lower --min_interactions")
    thresh = np.quantile(elig_tot, args.quantile)
    # bottom-quantile of the eligible cohort = disengaged
    labels = {}
    for sid, tot, ok in zip(ids, totals, elig_mask):
        if not ok:
            continue
        labels[str(int(sid))] = int(tot <= thresh)

    pos = sum(labels.values()); n = len(labels)
    meta = {"definition": f"bottom-{args.quantile:.0%} total interactions among "
                          f">= {args.min_interactions}-interaction cohort (from npz)",
            "min_interactions": args.min_interactions,
            "threshold_interactions": float(thresh),
            "n_eligible_labeled": n, "n_students_in_npz": int(len(ids)),
            "positive_count": pos, "positive_rate": pos / n if n else 0.0}
    (d / "dropout_labels.json").write_text(json.dumps({**labels, "_meta": meta}))

    print(f"eligible (>= {args.min_interactions}): {int(elig_mask.sum())} of {len(ids)}")
    print(f"labeled {n}; positive rate = {pos/n:.4f} ({pos} disengaged)")
    print(f"threshold = <= {thresh:.0f} total interactions (bottom {args.quantile:.0%})")
    med = float(np.median(elig_tot))
    print(f"eligible totals: median={med:.0f}, "
          f"p25={np.quantile(elig_tot,0.25):.0f}, p75={np.quantile(elig_tot,0.75):.0f}")


if __name__ == "__main__":
    main()
