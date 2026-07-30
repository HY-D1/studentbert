#!/usr/bin/env python3
# Regenerate the skill-frequency skew documentation for any processed dataset.
# READ-ONLY: touches nothing but sequences.npz / splits.json / skill_vocab.json.
# CPU only, runs in seconds on the login node.
#
# Usage:
#   python skill_freq_report.py ../processed/assist2017
#   python skill_freq_report.py ../processed/assist2017 --split test --out skill_freq_assist2017.md

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed_dir")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--out", default=None, help="optional markdown output path")
    ap.add_argument("--top", type=int, default=10, help="how many top/rare skills to list")
    args = ap.parse_args()

    d = Path(args.processed_dir)
    data = np.load(d / "sequences.npz")
    student_ids = data["student_ids"]
    skill = data["skill"]
    offsets = data["offsets"]
    vocab = json.loads((d / "skill_vocab.json").read_text())   # {name: idx}, idx 1..K
    K = len(vocab)
    idx_to_name = {v: k for k, v in vocab.items()}

    # select rows for the requested split
    id_to_row = {int(s): i for i, s in enumerate(student_ids)}
    if args.split == "all":
        rows = list(range(len(student_ids)))
    else:
        splits = json.loads((d / "splits.json").read_text())
        rows = [id_to_row[int(s)] for s in splits[args.split] if int(s) in id_to_row]

    counts = np.zeros(K + 1, dtype=np.int64)   # index 0 = PAD, unused
    n_inter = 0
    for r in rows:
        s, e = offsets[r], offsets[r + 1]
        seg = skill[s:e]
        n_inter += len(seg)
        np.add.at(counts, seg, 1)

    present = counts[1:]                        # drop PAD slot
    nonzero = present[present > 0]
    med = float(np.median(nonzero)) if nonzero.size else 0.0
    mx = int(nonzero.max()) if nonzero.size else 0

    lines = []
    lines.append(f"# Skill-frequency skew: {d.name} ({args.split} split)\n")
    lines.append(f"- Students in split: {len(rows):,}")
    lines.append(f"- Interactions in split: {n_inter:,}")
    lines.append(f"- Skills in vocabulary (K): {K}")
    lines.append(f"- Skills PRESENT in split: {int((present > 0).sum())}")
    lines.append(f"- Skills ABSENT from split: {int((present == 0).sum())}")
    for t in (50, 20, 10):
        lines.append(f"- Skills with fewer than {t} instances: {int(((present > 0) & (present < t)).sum())}")
    lines.append(f"- Max instance count: {mx:,}")
    lines.append(f"- Median instance count (present skills): {med:,.1f}")
    lines.append(f"- Max/median skew: {mx / med:.1f}x" if med else "- Max/median skew: n/a")
    lines.append("")
    order = np.argsort(-present)
    lines.append(f"## Top {args.top} most frequent skills\n")
    lines.append("| skill_idx | count | name |")
    lines.append("|---|---|---|")
    for i in order[: args.top]:
        lines.append(f"| {i + 1} | {int(present[i]):,} | {idx_to_name.get(int(i + 1), '?')} |")
    lines.append("")
    rare = [i for i in order[::-1] if present[i] > 0][: args.top]
    lines.append(f"## {args.top} rarest PRESENT skills\n")
    lines.append("| skill_idx | count | name |")
    lines.append("|---|---|---|")
    for i in rare:
        lines.append(f"| {i + 1} | {int(present[i]):,} | {idx_to_name.get(int(i + 1), '?')} |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
