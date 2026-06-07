"""Preprocess EdNet-KT1 into student interaction sequences of
(skill_idx, correct, time_bin) tuples for StudentBERT.

EdNet differs from ASSISTments in three important ways:
  1. Data is ONE CSV PER STUDENT in a KT1/ folder; filename (minus .csv) is user_id.
  2. KT1 logs do NOT contain correctness or skill directly. Each row has:
        timestamp (ms), solving_id, question_id (e.g. 'q1234'),
        user_answer (a/b/c/d), elapsed_time (ms)
  3. Correctness + skill (tag) must be JOINED from a questions metadata file
     (contents/questions.csv), which has columns:
        question_id, bundle_id, explanation_id, correct_answer, part, tags
     - correct = (user_answer == correct_answer)
     - skill   = the question's 'tags' field (may be multi-tag, e.g. '1;2;179')
                 We use the FIRST tag as the primary skill (documented choice).

Output (same format as ASSISTments preprocessing):
  sequences.npz, skill_vocab.json, splits.json, vocab_stats.md

Usage:
    python scripts/preprocess_ednet.py \
        --kt1_dir ~/Downloads/EdNet-KT1/KT1 \
        --questions_csv ~/Downloads/EdNet-contents/contents/questions.csv \
        --out_dir data/sample/ednet \
        --sample_students 1000

Time complexity : O(N log N) per student sort, summed over students.
Space           : O(N) ints in memory.

Edge cases:
  - questions with empty/NaN tags -> skill 'UNK'
  - user_answer not in a-d or missing -> row dropped
  - multi-tag questions -> first tag used (flagged; revisit for multi-KC handling)
  - students < min_interactions dropped
  - reading thousands of tiny CSVs is slow; we read only needed columns
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TIME_BIN_EDGES_SEC = [5.0, 15.0, 60.0, 300.0]  # same scheme as ASSISTments
MIN_INTERACTIONS = 10


def bin_time_seconds(seconds: np.ndarray) -> np.ndarray:
    return np.digitize(seconds, TIME_BIN_EDGES_SEC, right=False).astype(np.int16) + 1


def load_question_meta(questions_csv: str) -> dict:
    """Return {question_id: (correct_answer, first_tag)}."""
    # VERIFY column names against your real questions.csv header.
    qm = pd.read_csv(questions_csv)
    # Expected columns: question_id, correct_answer, tags (others ignored)
    out = {}
    for qid, ans, tags in zip(qm["question_id"], qm["correct_answer"], qm["tags"]):
        first_tag = "UNK"
        if isinstance(tags, str) and tags.strip():
            # tags look like '1;2;179' or '1' — take first
            first_tag = tags.replace(",", ";").split(";")[0].strip() or "UNK"
        out[str(qid)] = (str(ans).strip(), first_tag)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kt1_dir", required=True, help="Folder of per-student KT1 CSVs.")
    ap.add_argument("--questions_csv", required=True, help="contents/questions.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_students", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_interactions", type=int, default=MIN_INTERACTIONS)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    qmeta = load_question_meta(args.questions_csv)

    kt1 = Path(args.kt1_dir)
    files = sorted(kt1.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {kt1}")

    if args.sample_students is not None and args.sample_students < len(files):
        idx = rng.choice(len(files), size=args.sample_students, replace=False)
        files = [files[i] for i in idx]

    # Build skill vocab on the fly (tag string -> idx, 0 = PAD)
    skill_vocab: dict[str, int] = {}

    student_ids: list[int] = []
    skill_arr: list[int] = []
    correct_arr: list[int] = []
    time_arr: list[int] = []
    offsets: list[int] = [0]
    seq_lengths: list[int] = []

    n_raw_rows = 0
    n_kept_rows = 0

    for f in files:
        # filename without .csv is the user_id; EdNet uses 'u1234' or just an int
        uid_str = f.stem.lstrip("u") or f.stem
        try:
            uid = int(uid_str)
        except ValueError:
            uid = abs(hash(f.stem)) % (10**9)  # fallback stable-ish id

        if f.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(f, usecols=["timestamp", "question_id", "user_answer", "elapsed_time"])
        except pd.errors.EmptyDataError:
            continue
        n_raw_rows += len(df)
        if df.empty:
            continue

        # Map question -> (correct_answer, tag); drop questions we have no meta for
        ca = df["question_id"].astype(str).map(lambda q: qmeta.get(q, (None, None))[0])
        tg = df["question_id"].astype(str).map(lambda q: qmeta.get(q, (None, "UNK"))[1])
        df = df.assign(correct_answer=ca, tag=tg)
        df = df.dropna(subset=["correct_answer", "user_answer", "timestamp"])
        if df.empty:
            continue

        df["correct"] = (
            df["user_answer"].astype(str).str.strip() == df["correct_answer"].astype(str).str.strip()
        ).astype(np.int8)
        # elapsed_time is ms -> seconds
        df["elapsed_sec"] = pd.to_numeric(df["elapsed_time"], errors="coerce").fillna(0.0) / 1000.0
        df["elapsed_sec"] = df["elapsed_sec"].clip(lower=0.0)

        if len(df) < args.min_interactions:
            continue

        df = df.sort_values("timestamp", kind="stable")

        # assign skill indices
        sidx = []
        for t in df["tag"].tolist():
            t = t if isinstance(t, str) and t else "UNK"
            if t not in skill_vocab:
                skill_vocab[t] = len(skill_vocab) + 1  # 1.. ; 0 = PAD
            sidx.append(skill_vocab[t])

        student_ids.append(uid)
        skill_arr.extend(sidx)
        correct_arr.extend(df["correct"].tolist())
        time_arr.extend(bin_time_seconds(df["elapsed_sec"].to_numpy()).tolist())
        offsets.append(len(skill_arr))
        seq_lengths.append(len(df))
        n_kept_rows += len(df)

    if not student_ids:
        raise SystemExit("No students passed filters — check paths / metadata join.")

    # student-level split
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

    sl = np.array(seq_lengths)
    frac_correct = float(np.mean(correct_arr)) if correct_arr else 0.0
    stats = f"""# EdNet-KT1 — vocabulary statistics

Per-student CSV files scanned : {len(files):,}
Raw rows read                 : {n_raw_rows:,}
Rows kept (after meta join)   : {n_kept_rows:,}
Students retained             : {n_students:,}
  train / val / test          : {len(splits['train']):,} / {len(splits['val']):,} / {len(splits['test']):,}
Distinct skills/tags (K)      : {len(skill_vocab):,}  (idx 1..{len(skill_vocab)}; 0 = PAD)
Fraction correct              : {frac_correct:.4f}

## Sequence length (interactions per student)
min {int(sl.min())} | median {int(np.median(sl))} | mean {sl.mean():.1f} | p95 {int(np.percentile(sl,95))} | max {int(sl.max())}

## Time bins (seconds) idx 1..5: <5 / 5-15 / 15-60 / 60-300 / >300 (0=PAD)
Min interactions filter       : {args.min_interactions}
Sample students               : {args.sample_students if args.sample_students else 'ALL'}
Seed                          : {args.seed}

NOTE: correctness derived as (user_answer == correct_answer); skill = first tag of question.
"""
    (out / "vocab_stats.md").write_text(stats)
    print(stats)
    print(f"Wrote outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
