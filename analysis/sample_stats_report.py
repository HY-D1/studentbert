#!/usr/bin/env python3
# Report the EXACT per-(seed, N) fine-tuning samples used by the next-skill
# sweep, without rerunning anything. Replicates scripts/downstream_edubert.py
# verbatim: row construction (set-iteration order over splits.json train ids)
# and sampling via random.Random(seed).sample(rows, N). Prints per-cell
# students / interactions / median / mean sequence length / distinct skills,
# a LaTeX-ready appendix table (seed-averaged), and an empirical nestedness
# check across N. CPU-only, seconds, read-only.
#
# Usage:
#   python analysis/sample_stats_report.py ../processed/assist2017
#   python analysis/sample_stats_report.py ../processed/assist2017 --seeds 42 1 2 --ns 25 50 100 200 500 1000

import argparse
import json
import random
import statistics as st
from pathlib import Path

import numpy as np


def build_rows(d: Path, split: str):
    """Verbatim replication of StudentSeqDataset row construction."""
    data = np.load(d / "sequences.npz")
    offsets = data["offsets"]
    skill = data["skill"]
    student_ids = data["student_ids"]
    splits = json.loads((d / "splits.json").read_text())
    wanted = set(int(s) for s in splits[split])           # set, as in the class
    id_to_row = {int(sid): i for i, sid in enumerate(student_ids)}
    rows = [id_to_row[s] for s in wanted if s in id_to_row]
    return rows, offsets, skill


def sample_rows(rows, n, seed):
    """Verbatim replication of the training-time subsample."""
    if n and n < len(rows):
        rng = random.Random(seed)
        return rng.sample(rows, n)
    return list(rows)


def cell_stats(sel, offsets, skill, max_seq_len=512):
    lens, skills = [], set()
    n_inter = 0
    for r in sel:
        s, e = int(offsets[r]), int(offsets[r + 1])
        L = e - s
        lens.append(L)
        n_inter += min(L, max_seq_len)   # what the model actually sees
        skills.update(np.unique(skill[s:e]).tolist())
    skills.discard(0)
    return {"students": len(sel), "interactions": n_inter,
            "median_len": st.median(lens), "mean_len": st.mean(lens),
            "skills": len(skills)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed_dir")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 1, 2])
    ap.add_argument("--ns", type=int, nargs="*", default=[25, 50, 100, 200, 500, 1000])
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--task", default="next_skill")  # documentation only
    args = ap.parse_args()

    d = Path(args.processed_dir)
    rows, offsets, skill = build_rows(d, "train")
    print(f"train partition: {len(rows)} learners\n")

    per_cell = {}
    print("=== per (N, seed) sample statistics (sequence lengths BEFORE the "
          f"{args.max_seq_len}-interaction cap; interactions AFTER the cap) ===")
    print(f"{'N':>5} {'seed':>5} {'students':>9} {'interactions':>13} "
          f"{'median_len':>11} {'mean_len':>9} {'skills':>7}")
    for N in args.ns:
        for s in args.seeds:
            sel = sample_rows(rows, N, s)
            c = cell_stats(sel, offsets, skill, args.max_seq_len)
            per_cell[(N, s)] = (set(sel), c)
            print(f"{N:>5} {s:>5} {c['students']:>9} {c['interactions']:>13,} "
                  f"{c['median_len']:>11.1f} {c['mean_len']:>9.1f} {c['skills']:>7}")

    print("\n=== seed-averaged (LaTeX appendix table body) ===")
    print("N & Interactions & Median len & Mean len & Skills covered \\\\")
    for N in args.ns:
        cs = [per_cell[(N, s)][1] for s in args.seeds]
        inter = st.mean(c["interactions"] for c in cs)
        med = st.mean(c["median_len"] for c in cs)
        mean = st.mean(c["mean_len"] for c in cs)
        sk = st.mean(c["skills"] for c in cs)
        print(f"{N} & {inter:,.0f} & {med:.0f} & {mean:.0f} & {sk:.1f} \\\\")

    print("\n=== nestedness check (is sample(N_i) a subset of sample(N_{i+1})?) ===")
    for s in args.seeds:
        flags = []
        for a, b in zip(args.ns, args.ns[1:]):
            nested = per_cell[(a, s)][0].issubset(per_cell[(b, s)][0])
            flags.append(f"{a}\u2286{b}:{'yes' if nested else 'NO'}")
        print(f"seed {s}: " + "  ".join(flags))
    print("\n(Report whatever this says, verbatim, in the Method; nestedness depends on"
          " an internal branch of random.sample and must be read off, not assumed."
          " Pairing across conditions at fixed (seed, N) holds regardless.)")


if __name__ == "__main__":
    main()
