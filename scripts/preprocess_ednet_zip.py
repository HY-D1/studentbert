"""Preprocess EdNet-KT1 into student interaction sequences of
(skill_idx, correct, time_bin) tuples for StudentBERT — reading DIRECTLY FROM
THE ZIP (no extraction of 784K tiny files).

Identical output to preprocess_ednet.py: sequences.npz, skill_vocab.json,
splits.json, vocab_stats.md. Only the file-iteration changed: instead of
globbing an extracted KT1/ dir, we stream each member CSV out of EdNet-KT1.zip
via zipfile.open(). This avoids the Lustre/GPFS small-file metadata stall.

Usage (cluster):
    PYTHONPATH=. python scripts/preprocess_ednet_zip.py \
        --kt1_zip ../raw/EdNet-KT1.zip \
        --questions_csv ../raw/contents/questions.csv \
        --out_dir ../processed/ednet

Time complexity : one sequential pass over the 1.2GB zip + O(N log N) per-student sort.
Space           : O(N) ints in memory; one student's CSV in memory at a time.

Edge cases (unchanged):
  - empty / 0-byte member -> skipped (try/except EmptyDataError)
  - questions with empty/NaN tags -> skill 'UNK'
  - user_answer or correct_answer missing -> row dropped
  - multi-tag questions -> first tag used
  - students < min_interactions dropped
  - members inside zip may be named 'KT1/u123.csv' or 'u123.csv'; both handled
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TIME_BIN_EDGES_SEC = [5.0, 15.0, 60.0, 300.0]  # same scheme as ASSISTments
MIN_INTERACTIONS = 10


def bin_time_seconds(seconds: np.ndarray) -> np.ndarray:
    return np.digitize(seconds, TIME_BIN_EDGES_SEC, right=False).astype(np.int16) + 1


def load_question_meta(questions_csv: str) -> dict:
    """Return {question_id: (correct_answer, first_tag)}."""
    qm = pd.read_csv(questions_csv)
    out = {}
    for qid, ans, tags in zip(qm["question_id"], qm["correct_answer"], qm["tags"]):
        first_tag = "UNK"
        if isinstance(tags, str) and tags.strip():
            first_tag = tags.replace(",", ";").split(";")[0].strip() or "UNK"
        out[str(qid)] = (str(ans).strip(), first_tag)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kt1_zip", required=True, help="Path to EdNet-KT1.zip (not extracted).")
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

    zf = zipfile.ZipFile(args.kt1_zip, "r")
    # member CSVs only; ignore dirs and __MACOSX junk
    members = [
        m for m in zf.namelist()
        if m.endswith(".csv") and not m.startswith("__MACOSX") and "/." not in m
    ]
    members.sort()
    if not members:
        raise SystemExit(f"No CSV members found in {args.kt1_zip}")

    if args.sample_students is not None and args.sample_students < len(members):
        idx = rng.choice(len(members), size=args.sample_students, replace=False)
        members = [members[i] for i in idx]

    skill_vocab: dict[str, int] = {}

    student_ids: list[int] = []
    skill_arr: list[int] = []
    correct_arr: list[int] = []
    time_arr: list[int] = []
    offsets: list[int] = [0]
    seq_lengths: list[int] = []

    n_raw_rows = 0
    n_kept_rows = 0
    n_scanned = 0

    for m in members:
        n_scanned += 1
        # member basename without .csv is the user_id; e.g. 'KT1/u123.csv' -> 'u123'
        stem = Path(m).stem
        uid_str = stem.lstrip("u") or stem
        try:
            uid = int(uid_str)
        except ValueError:
            uid = abs(hash(stem)) % (10**9)

        # read this member straight from the zip (no extraction)
        try:
            raw = zf.read(m)
            if not raw:  # 0-byte member (e.g. u299008.csv)
                continue
            df = pd.read_csv(
                io.BytesIO(raw),
                usecols=["timestamp", "question_id", "user_answer", "elapsed_time"],
            )
        except pd.errors.EmptyDataError:
            continue
        except Exception:
            # malformed member -> skip rather than abort whole run
            continue

        n_raw_rows += len(df)
        if df.empty:
            continue

        ca = df["question_id"].astype(str).map(lambda q: qmeta.get(q, (None, None))[0])
        tg = df["question_id"].astype(str).map(lambda q: qmeta.get(q, (None, "UNK"))[1])
        df = df.assign(correct_answer=ca, tag=tg)
        df = df.dropna(subset=["correct_answer", "user_answer", "timestamp"])
        if df.empty:
            continue

        df["correct"] = (
            df["user_answer"].astype(str).str.strip()
            == df["correct_answer"].astype(str).str.strip()
        ).astype(np.int8)
        df["elapsed_sec"] = pd.to_numeric(df["elapsed_time"], errors="coerce").fillna(0.0) / 1000.0
        df["elapsed_sec"] = df["elapsed_sec"].clip(lower=0.0)

        if len(df) < args.min_interactions:
            continue

        df = df.sort_values("timestamp", kind="stable")

        sidx = []
        for t in df["tag"].tolist():
            t = t if isinstance(t, str) and t else "UNK"
            if t not in skill_vocab:
                skill_vocab[t] = len(skill_vocab) + 1
            sidx.append(skill_vocab[t])

        student_ids.append(uid)
        skill_arr.extend(sidx)
        correct_arr.extend(df["correct"].tolist())
        time_arr.extend(bin_time_seconds(df["elapsed_sec"].to_numpy()).tolist())
        offsets.append(len(skill_arr))
        seq_lengths.append(len(df))
        n_kept_rows += len(df)

        # lightweight progress: every 50k students, print a heartbeat to the log
        if n_scanned % 50000 == 0:
            print(f"...scanned {n_scanned:,} / {len(members):,} members, "
                  f"kept {len(student_ids):,} students", flush=True)

    zf.close()

    if not student_ids:
        raise SystemExit("No students passed filters — check paths / metadata join.")

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

Member CSVs scanned (in zip)  : {len(members):,}
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
Source                        : read directly from {Path(args.kt1_zip).name} (no extraction)

NOTE: correctness derived as (user_answer == correct_answer); skill = first tag of question.
"""
    (out / "vocab_stats.md").write_text(stats)
    print(stats)
    print(f"Wrote outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
