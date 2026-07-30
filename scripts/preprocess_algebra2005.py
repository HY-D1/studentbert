# Preprocess Algebra2005 (KDD Cup 2010) into the SAME sequences.npz format as the
# other three datasets (ASSIST/EdNet/Junyi). Mirrors preprocess_junyi.py exactly:
# same schema, time bins, min-10 filter, 80/10/10 split by student, seed 42.
#
# KDD Cup file is TAB-delimited. Field mapping:
#   user    <- 'Anon Student Id'
#   skill   <- first KC from 'KC(Default)' (multi-KC rows split on ~~, take first)
#   correct <- 'Correct First Attempt' (already 0/1)
#   time    <- ordering by 'First Transaction Time' (fallback 'Step Start Time')
#   taken   <- 'Step Duration (sec)' -> time_bin 1-5
# Rows with empty KC or missing correct/duration are dropped. Students <10 dropped.
#
# Usage:
#   python preprocess_algebra2005.py --txt ../raw/algebra2005/algebra_2005_2006_train.txt \
#       --out_dir ../processed/algebra2005
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TIME_BIN_EDGES_SEC = [5.0, 15.0, 60.0, 300.0]
MIN_INTERACTIONS = 10

def bin_time_seconds(seconds: np.ndarray) -> np.ndarray:
    return np.digitize(seconds, TIME_BIN_EDGES_SEC, right=False).astype(np.int16) + 1

def first_kc(kc_str: str) -> str:
    # KC(Default) may hold multiple KCs joined by '~~'; take the first as the skill.
    # Each KC is a descriptive [SkillRule: ...] block; the whole first block is the id.
    if not isinstance(kc_str, str) or kc_str.strip() == "":
        return ""
    return kc_str.split("~~")[0].strip()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True, help="KDD Cup tab-delimited file (train has full data)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_interactions", type=int, default=MIN_INTERACTIONS)
    ap.add_argument("--col_user", default="Anon Student Id")
    ap.add_argument("--col_kc", default="KC(Default)")
    ap.add_argument("--col_correct", default="Correct First Attempt")
    ap.add_argument("--col_order", default="First Transaction Time")
    ap.add_argument("--col_dur", default="Step Duration (sec)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    usecols = [args.col_user, args.col_kc, args.col_correct, args.col_order, args.col_dur]
    df = pd.read_csv(args.txt, sep="\t", usecols=usecols, low_memory=False,
                     encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={
        args.col_user: "user", args.col_kc: "kc", args.col_correct: "correct",
        args.col_order: "order", args.col_dur: "taken",
    })
    n_raw = len(df)

    # skill = first KC; drop empty-KC rows
    df["skill"] = df["kc"].map(first_kc)
    df = df[df["skill"] != ""]
    # correct: already 0/1, drop missing
    df = df.dropna(subset=["user", "correct"])
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct"])
    df["correct"] = df["correct"].astype(np.int8)
    # duration -> seconds (already sec), fill/clip
    df["taken"] = pd.to_numeric(df["taken"], errors="coerce").fillna(0.0).clip(lower=0.0)
    # order key for stable sort within student
    _dt = pd.to_datetime(df["order"], errors="coerce")
    if _dt.notna().any():
        # datetime64 -> int64 nanoseconds (replaces removed .view("int64"))
        df["order"] = _dt.astype("int64")
        # NaT becomes a large negative sentinel after astype; fix by forward-fill on valid
        df.loc[_dt.isna(), "order"] = np.nan
        df["order"] = df["order"].ffill().fillna(0)
    else:
        # order column unparseable as datetime -> keep original row order
        df["order"] = np.arange(len(df))
    n_clean = len(df)

    # min interactions per student
    counts = df.groupby("user").size()
    keep_students = counts[counts >= args.min_interactions].index
    df = df[df["user"].isin(keep_students)]

    # skill vocab: idx 1..K, 0=PAD
    skills = sorted(df["skill"].astype(str).unique())
    skill_vocab = {name: i + 1 for i, name in enumerate(skills)}
    df["skill_idx"] = df["skill"].astype(str).map(skill_vocab).astype(np.int32)
    df["time_bin"] = bin_time_seconds(df["taken"].to_numpy())

    df = df.sort_values(["user", "order"], kind="stable")

    student_ids, skill_arr, correct_arr, time_arr = [], [], [], []
    offsets, seq_lengths = [0], []
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
    (out / "uid_map.json").write_text(json.dumps({str(k): v for k, v in uid_map.items()}, indent=2))

    sl = np.array(seq_lengths)
    frac_correct = float(np.mean(correct_arr)) if correct_arr else 0.0
    stats = (
        f"# {Path(args.out_dir).name} vocab stats\n\n"
        f"- raw rows: {n_raw}\n"
        f"- after cleaning (KC+correct present): {n_clean}\n"
        f"- students (>= {args.min_interactions} interactions): {n_students}\n"
        f"- skills (first-KC): {len(skill_vocab)}\n"
        f"- total interactions: {len(skill_arr)}\n"
        f"- median seq len: {float(np.median(sl)):.1f}\n"
        f"- mean seq len: {float(sl.mean()):.1f}\n"
        f"- PROCESSED correct base rate: {frac_correct:.4f}\n"
        f"- split: train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])} (seed {args.seed})\n"
    )
    (out / "vocab_stats.md").write_text(stats)
    print(stats)
    print(f"RAW base rate was 0.7665 (locked prediction). PROCESSED base rate: {frac_correct:.4f}")

if __name__ == "__main__":
    main()
