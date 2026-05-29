"""Preprocess the ASSISTments 2017 competition dataset into student interaction
sequences of (skill_idx, correct, time_bin) tuples for StudentBERT.

Input  : anonymized_full_release_competition_dataset.csv
Output : <out_dir>/
           sequences.npz       packed sequences + offsets (compact, fast to load)
           skill_vocab.json    {skill_name: skill_idx}
           splits.json         {"train": [...], "val": [...], "test": [...]}  (student ids)
           vocab_stats.md      human-readable dataset statistics

Usage:
    python scripts/preprocess_assistments2017.py \
        --csv ~/Downloads/anonymized_full_release_competition_dataset.csv \
        --out_dir data/sample/assist2017 \
        --sample_students 1000      # omit for full dataset

Columns used (others ignored):
    studentId  -> student grouping key
    skill      -> skill name (string)
    correct    -> 0/1 outcome
    startTime  -> Unix timestamp, used to order interactions within a student
    timeTaken  -> response time in seconds, bucketed into time_bin

Time complexity : O(N log N) dominated by the per-student sort (N = #interactions).
Space           : O(N) ints held in memory; output is packed once.

Edge cases handled:
    - rows with missing skill / correct / startTime are dropped (logged)
    - students with < MIN_INTERACTIONS interactions are dropped (configurable)
    - time bins are fixed, documented cutoffs so all 3 datasets can share a scheme
    - skill vocab is built only from the (post-filter) training portion would be
      ideal, but here we build it over all retained rows and reserve idx 0 = PAD;
      this is standard for KT and avoids unseen-skill crashes at eval. (Flagged.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed, documented response-time cutoffs (seconds). Shared across datasets.
# Bin 0 is reserved for PAD; real bins are 1..5.
TIME_BIN_EDGES = [5.0, 15.0, 60.0, 300.0]  # -> 5 buckets: <5, 5-15, 15-60, 60-300, >300
PAD_IDX = 0
MIN_INTERACTIONS = 10


def bin_time(seconds: np.ndarray) -> np.ndarray:
    """Map response times (s) to bins 1..5. np.digitize returns 0..4, we shift +1."""
    # values < 5 -> 0, [5,15) -> 1, ... ; +1 so real bins are 1..5 and 0 stays PAD
    return np.digitize(seconds, TIME_BIN_EDGES, right=False).astype(np.int16) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--sample_students", type=int, default=None,
                    help="If set, keep only this many random students (for local dev).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_interactions", type=int, default=MIN_INTERACTIONS)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load only the columns we need (keeps memory low on the full file) ---
    usecols = ["studentId", "skill", "correct", "startTime", "timeTaken"]
    df = pd.read_csv(args.csv, usecols=usecols)
    n_raw = len(df)

    # --- Clean ---
    df = df.dropna(subset=["studentId", "skill", "correct", "startTime"])
    df = df[df["correct"].isin([0, 1])]
    df["timeTaken"] = pd.to_numeric(df["timeTaken"], errors="coerce").fillna(0.0).clip(lower=0.0)
    n_clean = len(df)

    # --- Optional student subsample (for M1 dev) ---
    all_students = df["studentId"].unique()
    if args.sample_students is not None and args.sample_students < len(all_students):
        keep = rng.choice(all_students, size=args.sample_students, replace=False)
        df = df[df["studentId"].isin(keep)]

    # --- Drop short students ---
    counts = df.groupby("studentId").size()
    keep_students = counts[counts >= args.min_interactions].index
    df = df[df["studentId"].isin(keep_students)]

    # --- Build skill vocab (idx 0 reserved for PAD) ---
    skills = sorted(df["skill"].unique())
    skill_vocab = {name: i + 1 for i, name in enumerate(skills)}  # 1..K
    df["skill_idx"] = df["skill"].map(skill_vocab).astype(np.int32)
    df["time_bin"] = bin_time(df["timeTaken"].to_numpy())
    df["correct"] = df["correct"].astype(np.int8)

    # --- Sort by student then time, build packed sequences ---
    df = df.sort_values(["studentId", "startTime"], kind="stable")

    student_ids: list[int] = []
    skill_arr: list[int] = []
    correct_arr: list[int] = []
    time_arr: list[int] = []
    offsets: list[int] = [0]  # CSR-style: sequence i is [offsets[i]:offsets[i+1]]
    seq_lengths: list[int] = []

    for sid, g in df.groupby("studentId", sort=False):
        student_ids.append(int(sid))
        skill_arr.extend(g["skill_idx"].tolist())
        correct_arr.extend(g["correct"].tolist())
        time_arr.extend(g["time_bin"].tolist())
        offsets.append(len(skill_arr))
        seq_lengths.append(len(g))

    # --- Student-level train/val/test split (no student in two splits) ---
    n_students = len(student_ids)
    perm = rng.permutation(n_students)
    n_train = int(0.8 * n_students)
    n_val = int(0.1 * n_students)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    sid_array = np.array(student_ids)
    splits = {
        "train": sid_array[train_idx].tolist(),
        "val": sid_array[val_idx].tolist(),
        "test": sid_array[test_idx].tolist(),
    }

    # --- Save packed arrays ---
    np.savez_compressed(
        out / "sequences.npz",
        student_ids=np.array(student_ids, dtype=np.int64),
        skill=np.array(skill_arr, dtype=np.int32),
        correct=np.array(correct_arr, dtype=np.int8),
        time_bin=np.array(time_arr, dtype=np.int16),
        offsets=np.array(offsets, dtype=np.int64),
    )
    (out / "skill_vocab.json").write_text(json.dumps(skill_vocab, indent=2))
    (out / "splits.json").write_text(json.dumps(splits, indent=2))

    # --- Vocab stats ---
    seq_lengths_np = np.array(seq_lengths)
    frac_correct = float(np.mean(correct_arr)) if correct_arr else 0.0
    stats = f"""# ASSISTments 2017 — vocabulary statistics

Source CSV rows (raw)       : {n_raw:,}
Rows after cleaning         : {n_clean:,}
Rows after all filters      : {len(df):,}
Students retained           : {n_students:,}
  train / val / test        : {len(splits['train']):,} / {len(splits['val']):,} / {len(splits['test']):,}
Distinct skills (vocab K)   : {len(skill_vocab):,}  (skill_idx 1..{len(skill_vocab)}; 0 = PAD)
Fraction correct            : {frac_correct:.4f}

## Sequence length (interactions per student)
min     : {int(seq_lengths_np.min())}
median  : {int(np.median(seq_lengths_np))}
mean    : {seq_lengths_np.mean():.1f}
p95     : {int(np.percentile(seq_lengths_np, 95))}
max     : {int(seq_lengths_np.max())}

## Time bins (seconds), idx 1..5
1: <5    2: 5-15    3: 15-60    4: 60-300    5: >300
(idx 0 reserved for PAD)

Min interactions filter     : {args.min_interactions}
Sample students             : {args.sample_students if args.sample_students else 'ALL'}
Seed                        : {args.seed}
"""
    (out / "vocab_stats.md").write_text(stats)
    print(stats)
    print(f"Wrote outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
