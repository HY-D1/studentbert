# Preprocess ASSISTments2009 skill_builder_data.csv into the SAME sequences.npz format
# as the other datasets. CSV (comma-delimited), ISO-8859 encoding.
# Field mapping:
#   user    <- user_id
#   skill   <- skill_id (numeric skill/KC id; drop rows with missing skill_id)
#   correct <- correct (0/1)
#   order   <- order_id (chronological, per the dataset docs)
#   time    <- ms_first_response (milliseconds -> seconds -> time_bin 1-5)
# Rows with missing skill_id dropped (skill_id has ~16% missing per EduData docs).
# Students < min_interactions dropped. 80/10/10 split by student, seed 42.
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TIME_BIN_EDGES_SEC = [5.0, 15.0, 60.0, 300.0]
MIN_INTERACTIONS = 10

def bin_time_seconds(seconds: np.ndarray) -> np.ndarray:
    return np.digitize(seconds, TIME_BIN_EDGES_SEC, right=False).astype(np.int16) + 1

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_interactions", type=int, default=MIN_INTERACTIONS)
    ap.add_argument("--col_user", default="user_id")
    ap.add_argument("--col_skill", default="skill_id")
    ap.add_argument("--col_correct", default="correct")
    ap.add_argument("--col_order", default="order_id")
    ap.add_argument("--col_time", default="ms_first_response")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    usecols = [args.col_user, args.col_skill, args.col_correct, args.col_order, args.col_time]
    # ASSISTments2009 is ISO-8859-15 encoded (skill_name has special chars)
    df = pd.read_csv(args.csv, usecols=usecols, low_memory=False,
                     encoding="ISO-8859-15", encoding_errors="replace")
    df = df.rename(columns={
        args.col_user:"user", args.col_skill:"skill", args.col_correct:"correct",
        args.col_order:"order", args.col_time:"taken_ms",
    })
    n_raw = len(df)

    # drop rows with missing skill_id (multi-skill collapsed rows / no-KC rows)
    df = df.dropna(subset=["user","skill","correct","order"])
    # skill_id may be float (e.g. 1.0) -> normalize to int string
    df["skill"] = df["skill"].astype(float).astype("int64").astype(str)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct"])
    df["correct"] = df["correct"].clip(0,1).astype(np.int8)
    # ms_first_response -> seconds
    df["taken"] = pd.to_numeric(df["taken_ms"], errors="coerce").fillna(0.0).clip(lower=0.0) / 1000.0
    df["order"] = pd.to_numeric(df["order"], errors="coerce")
    df = df.dropna(subset=["order"])
    n_clean = len(df)

    # min interactions per student
    counts = df.groupby("user").size()
    keep = counts[counts >= args.min_interactions].index
    df = df[df["user"].isin(keep)]

    skills = sorted(df["skill"].unique())
    skill_vocab = {name:i+1 for i,name in enumerate(skills)}  # 0=PAD
    df["skill_idx"] = df["skill"].map(skill_vocab).astype(np.int32)
    df["time_bin"] = bin_time_seconds(df["taken"].to_numpy())
    df = df.sort_values(["user","order"], kind="stable")

    student_ids, skill_arr, correct_arr, time_arr = [], [], [], []
    offsets, seq_lengths = [0], []
    uid_map = {}
    for u,g in df.groupby("user", sort=False):
        if u not in uid_map: uid_map[u]=len(uid_map)+1
        student_ids.append(uid_map[u])
        skill_arr.extend(g["skill_idx"].tolist())
        correct_arr.extend(g["correct"].tolist())
        time_arr.extend(g["time_bin"].tolist())
        offsets.append(len(skill_arr)); seq_lengths.append(len(g))

    n_students = len(student_ids)
    perm = rng.permutation(n_students)
    n_train, n_val = int(0.8*n_students), int(0.1*n_students)
    sid = np.array(student_ids)
    splits = {"train":sid[perm[:n_train]].tolist(),
              "val":sid[perm[n_train:n_train+n_val]].tolist(),
              "test":sid[perm[n_train+n_val:]].tolist()}

    np.savez_compressed(out/"sequences.npz",
        student_ids=np.array(student_ids,dtype=np.int64),
        skill=np.array(skill_arr,dtype=np.int32),
        correct=np.array(correct_arr,dtype=np.int8),
        time_bin=np.array(time_arr,dtype=np.int16),
        offsets=np.array(offsets,dtype=np.int64))
    (out/"skill_vocab.json").write_text(json.dumps(skill_vocab,indent=2))
    (out/"splits.json").write_text(json.dumps(splits,indent=2))
    (out/"uid_map.json").write_text(json.dumps({str(k):v for k,v in uid_map.items()},indent=2))

    sl = np.array(seq_lengths)
    br = float(np.mean(correct_arr)) if correct_arr else 0.0
    med = float(np.median(sl)); nsk = len(skill_vocab)
    pps = med/nsk
    stats = (f"# ASSISTments2009 vocab stats\n\n"
             f"- raw rows: {n_raw}\n- after cleaning (skill+correct present): {n_clean}\n"
             f"- students (>= {args.min_interactions}): {n_students}\n- skills: {nsk}\n"
             f"- total interactions: {len(skill_arr)}\n"
             f"- median seq len: {med:.1f}\n- mean seq len: {sl.mean():.1f}\n"
             f"- correct base rate: {br:.4f}\n"
             f"- PRACTICE-PER-SKILL (median_seq_len/n_skills): {pps:.3f}\n"
             f"- split: train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}\n")
    (out/"vocab_stats.md").write_text(stats)
    print(stats)
    print(f">>> PRACTICE-PER-SKILL = {pps:.3f}  (boundary is 1.41)")
    print(f">>> pps {'>' if pps>1.41 else '<'} 1.41  => predict {'SKILL' if pps>1.41 else 'CORRECTNESS'}-driven")

if __name__ == "__main__":
    main()
