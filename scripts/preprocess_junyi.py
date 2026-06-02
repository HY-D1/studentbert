"""Preprocess Junyi Academy problem logs into student interaction sequences of
(skill_idx, correct, time_bin) tuples for StudentBERT.

IMPORTANT — there are TWO different Junyi releases with DIFFERENT columns:
  (A) EduData / 2015 'junyi_ProblemLog_original.csv':
        columns include: user_id, exercise, correct, time_done, time_taken
  (B) Kaggle 2020 'Junyi Academy Online Learning Activity Dataset':
        problem logs use different names (e.g. uuid, ucid, is_correct, timestamp...)

This script is written for (A)-style columns and exposes column names as CLI args
so you can adapt to (B) WITHOUT editing code. ALWAYS run `head -3 <file>` first
and map the real columns via the flags below.

Output (same format as the other two scripts):
  sequences.npz, skill_vocab.json, splits.json, vocab_stats.md

Usage (defaults match EduData 2015 schema):
    python scripts/preprocess_junyi.py \
        --csv ~/Downloads/junyi_ProblemLog_original.csv \
        --out_dir data/sample/junyi \
        --sample_students 1000

    # If your file uses different names (Kaggle 2020), override:
    python scripts/preprocess_junyi.py --csv ... --out_dir ... \
        --col_user uuid --col_skill ucid --col_correct is_correct \
        --col_time timestamp --col_taken total_sec_taken

Time complexity : O(N log N). Space: O(N).

Edge cases:
  - correct may be bool / 'true'/'false' / 0-1 -> normalized to {0,1}
  - time_taken may be ms or sec depending on release -> see --time_taken_unit
  - missing skill/correct/time -> row dropped
  - students < min_interactions dropped
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TIME_BIN_EDGES_SEC = [5.0, 15.0, 60.0, 300.0]
MIN_INTERACTIONS = 10


def bin_time_seconds(seconds: np.ndarray) -> np.ndarray:
    return np.digitize(seconds, TIME_BIN_EDGES_SEC, right=False).astype(np.int16) + 1


def normalize_correct(s: pd.Series) -> pd.Series:
    """Map various truthy encodings to {0,1}."""
    if s.dtype == bool:
        return s.astype(np.int8)
    m = {
        "true": 1, "false": 0, "1": 1, "0": 0,
        "1.0": 1, "0.0": 0, "yes": 1, "no": 0,
    }
    return (
        s.astype(str).str.strip().str.lower().map(m)
        .fillna(pd.to_numeric(s, errors="coerce"))
        .fillna(0)
        .astype(np.int8)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_students", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_interactions", type=int, default=MIN_INTERACTIONS)
    # column-name flags (defaults = EduData 2015 schema). VERIFY with head -3.
    ap.add_argument("--col_user", default="user_id")
    ap.add_argument("--col_skill", default="exercise")
    ap.add_argument("--col_correct", default="correct")
    ap.add_argument("--col_time", default="time_done")
    ap.add_argument("--col_taken", default="time_taken")
    ap.add_argument("--time_taken_unit", choices=["sec", "ms"], default="sec",
                    help="Units of the response-time column.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    usecols = [args.col_user, args.col_skill, args.col_correct, args.col_time, args.col_taken]
    df = pd.read_csv(args.csv, usecols=usecols, low_memory=False)
    df = df.rename(columns={
        args.col_user: "user", args.col_skill: "skill", args.col_correct: "correct",
        args.col_time: "time", args.col_taken: "taken",
    })
    n_raw = len(df)

    df = df.dropna(subset=["user", "skill", "correct", "time"])
    df["correct"] = normalize_correct(df["correct"])
    df["taken"] = pd.to_numeric(df["taken"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if args.time_taken_unit == "ms":
        df["taken"] = df["taken"] / 1000.0
    n_clean = len(df)

    students = df["user"].unique()
    if args.sample_students is not None and args.sample_students < len(students):
        keep = rng.choice(students, size=args.sample_students, replace=False)
        df = df[df["user"].isin(keep)]

    counts = df.groupby("user").size()
    keep_students = counts[counts >= args.min_interactions].index
    df = df[df["user"].isin(keep_students)]

    skills = sorted(df["skill"].astype(str).unique())
    skill_vocab = {name: i + 1 for i, name in enumerate(skills)}  # 0 = PAD
    df["skill_idx"] = df["skill"].astype(str).map(skill_vocab).astype(np.int32)
    df["time_bin"] = bin_time_seconds(df["taken"].to_numpy())

    df = df.sort_values(["user", "time"], kind="stable")

    student_ids, skill_arr, correct_arr, time_arr = [], [], [], []
    offsets, seq_lengths = [0], []
    # user ids may be strings; map to stable ints for storage
    uid_map: dict = {}
    for u, g in df.groupby("user", sort=False):
        if u not in uid_map:
            uid_map[u] = len(uid_map) + 1
        student_ids.append(uid_map[u])
        skill_arr.extend(g["skill_idx"].tolist())
        correct_arr.extend(g["correct"].tolist())
        time_arr.extend(g["time_bin"].tolist())
        offsets.append(len(skill_arr))
        seq_lengths.append(len(g))

    n_students = len(student_ids)
    perm = rng.permutation(n_students)
    n_train, n_val = int(0.8 * n_students), int(0.1 * n_students)
    sid = np.array(student_ids)
    splits = {
        "train": sid[perm[:n_train]].tolist(),
        "val": sid[perm[n_train:n_train + n_val]].tolist(),
        "test": sid[perm[n_train + n_val:]].tolist(),
    }

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
    # also save the user-id mapping so you can trace back
    (out / "uid_map.json").write_text(json.dumps({str(k): v for k, v in uid_map.items()}, indent=2))

    sl = np.array(seq_lengths)
    frac_correct = float(np.mean(correct_arr)) if correct_arr else 0.0
    stats = f"""# Junyi Academy — vocabulary statistics

Raw rows                  : {n_raw:,}
Rows after cleaning       : {n_clean:,}
Rows after all filters    : {len(df):,}
Students retained         : {n_students:,}
  train / val / test      : {len(splits['train']):,} / {len(splits['val']):,} / {len(splits['test']):,}
Distinct skills (K)       : {len(skill_vocab):,}  (idx 1..{len(skill_vocab)}; 0 = PAD)
Fraction correct          : {frac_correct:.4f}

## Sequence length (interactions per student)
min {int(sl.min())} | median {int(np.median(sl))} | mean {sl.mean():.1f} | p95 {int(np.percentile(sl,95))} | max {int(sl.max())}

## Time bins (seconds) idx 1..5: <5 / 5-15 / 15-60 / 60-300 / >300 (0=PAD)
Columns used: user={args.col_user} skill={args.col_skill} correct={args.col_correct} time={args.col_time} taken={args.col_taken} ({args.time_taken_unit})
Min interactions filter   : {args.min_interactions}
Sample students           : {args.sample_students if args.sample_students else 'ALL'}
Seed                      : {args.seed}
"""
    (out / "vocab_stats.md").write_text(stats)
    print(stats)
    print(f"Wrote outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
