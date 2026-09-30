from __future__ import annotations

"""How much does an interaction's own response-time bin reveal about its own correctness?

SAINT+ as implemented here sees the time bin of the step it is predicting (the bin enters the
encoder unshifted and the causal mask admits the diagonal). This measures, per dataset, how
predictive that bin is on its own: the AUC of "faster bin -> correct" over the positions SAINT+
is scored on (test split, most recent max_seq_len interactions per learner, position 0 dropped,
since the loss starts at t=1). It is a bound on the information available, not a measurement of
how much SAINT+ used.

AUC is computed exactly from the 5-bin contingency counts (ties count one half), so there is no
sampling and no sklearn dependency.

Run from the code directory (EdNet needs a compute node for memory):
  srun --mem=16G --time=00:30:00 /projects/algl/dai.hany/envs/sb/bin/python analysis/timebin_label_signal.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DATASETS = ["assist2017", "ednet", "junyi", "algebra2005", "bridge2006", "assist2009", "algebra2006"]


def auc_fast_is_correct(counts_pos: np.ndarray, counts_neg: np.ndarray) -> float:
    # score = -bin, so a positive beats a negative when the positive's bin is SMALLER
    P, N = counts_pos.sum(), counts_neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    neg_above = np.cumsum(counts_neg[::-1])[::-1]  # negatives in bin >= b
    neg_strictly_above = neg_above - counts_neg     # negatives in bin > b
    wins = (counts_pos * neg_strictly_above).sum() + 0.5 * (counts_pos * counts_neg).sum()
    return float(wins / (P * N))


def one(ds_dir: Path, split: str, max_len: int) -> dict:
    data = np.load(ds_dir / "sequences.npz")
    sids, offsets = data["student_ids"], data["offsets"]
    correct, tbin = data["correct"], data["time_bin"]
    wanted = set(json.loads((ds_dir / "splits.json").read_text())[split])
    rows = [i for i, s in enumerate(sids) if int(s) in wanted]
    pos = np.zeros(6, dtype=np.int64)
    neg = np.zeros(6, dtype=np.int64)
    for r in rows:
        s, e = int(offsets[r]), int(offsets[r + 1])
        s = max(s, e - max_len)
        c, b = correct[s + 1:e], tbin[s + 1:e]
        pos += np.bincount(b[c == 1].astype(np.int64), minlength=6)[:6]
        neg += np.bincount(b[c == 0].astype(np.int64), minlength=6)[:6]
    n = int(pos.sum() + neg.sum())
    real = slice(1, 6)  # bin 0 is PAD / missing; excluded
    return {"learners": len(rows), "n": n, "bin0": int(pos[0] + neg[0]),
            "correct_rate": float(pos[real].sum() / max(1, pos[real].sum() + neg[real].sum())),
            "share": (pos[real] + neg[real]) / max(1, (pos[real] + neg[real]).sum()),
            "rate": pos[real] / np.maximum(1, pos[real] + neg[real]),
            "auc": auc_fast_is_correct(pos[real], neg[real])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="../processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    a = ap.parse_args()
    print(f"split={a.split}  max_seq_len={a.max_seq_len}  positions from t=1 (SAINT+ scoring)\n")
    print("dataset        learners  scored_n    correct  AUC(fast->correct)  "
          "share of bins 1..5                correct rate in bins 1..5")
    ok = True
    for ds in a.datasets:
        d = Path(a.processed) / ds
        if not (d / "sequences.npz").exists():
            print(f"{ds:14s} MISSING {d}")
            ok = False
            continue
        r = one(d, a.split, a.max_seq_len)
        share = " ".join(f"{x:.3f}" for x in r["share"])
        rate = " ".join(f"{x:.3f}" for x in r["rate"])
        print(f"{ds:14s} {r['learners']:8d} {r['n']:9d}   {r['correct_rate']:.3f}    "
              f"{r['auc']:.4f}             {share}   {rate}"
              + (f"   (bin0={r['bin0']})" if r["bin0"] else ""))
    print("\nAUC 0.5 = the bin says nothing about the step's own correctness; "
          "further from 0.5 = more label information in scope for SAINT+.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
