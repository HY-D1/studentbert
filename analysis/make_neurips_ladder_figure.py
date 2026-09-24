#!/usr/bin/env python3
"""NeurIPS Figure (fig:srcscale): paired KT gain over scratch against source size.

Filled markers joined by lines are the Section 5.3 ladder (EdNet KT1 subsampled, source identity
fixed): each point is the mean over three independently pretrained encoders of their 6-seed
paired gains, and the bar spans the three. Open markers are the six cross-dataset conditions of
Table 3 (N=3000) with 95% intervals. Every value is computed here from the benchmark's per-seed
W&B values at full precision; nothing is typed in. Intervals use the recipe of
analysis/paired_bootstrap_pair.py (seeds paired and sorted, rng seed 42, 20,000 resamples,
endpoints at indices 500 and 19,499), so they match Table 3.

Replaces the earlier figure, whose 353,597 ASSISTments point and bar carried a draw copied from
the 150,000 rung (corrected 2026-09-23).

  python3 analysis/make_neurips_ladder_figure.py --bench benchmark_final --out source_scale_gain.pdf
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SIZES = (5000, 15000, 49153, 150000, 353597)
TRAIN = {"assist2017": 1366, "junyi": 49153, "ednet": 353597}
SHAPE = {"assist2017": "o", "ednet": "s", "junyi": "^"}
PRETTY = {"assist2017": "ASSISTments", "ednet": "EdNet", "junyi": "Junyi"}


def load(bench: str) -> dict:
    """{(track, target, candidate): {seed: value}} for the N=3000 knowledge-tracing cells."""
    v: dict = defaultdict(dict)
    with open(f"{bench}/executions.tsv", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r["valid_for_primary_analysis"] == "yes" and r["metric"] == "test_auc"
                    and r["track"] in ("B", "S11") and r["budget"] == "n3000"):
                val = r["value_wandb"] or r["value"]
                v[(r["track"], r["target_dataset"], r["candidate"])][int(r["finetune_seed"])] = float(val)
    return v


def paired(a: dict, b: dict) -> tuple[float, float, float]:
    shared = sorted(set(a) & set(b))
    gaps = [a[s] - b[s] for s in shared]
    n = len(gaps)
    rng = random.Random(42)
    means = sorted(sum(gaps[rng.randrange(n)] for _ in range(n)) / n for _ in range(20000))
    return sum(gaps) / n, means[500], means[19499]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="benchmark directory with executions.tsv")
    ap.add_argument("--out", default="source_scale_gain.pdf")
    a = ap.parse_args()
    v = load(a.bench)
    fig, ax = plt.subplots(figsize=(5.667, 3.374))
    ax.axvspan(1200, 4300, color="0.9", zorder=0)
    for tgt, style in (("assist2017", "-"), ("junyi", ":")):
        scratch = v[("B", tgt, "scratch")]
        xs, ms, lo, hi = [], [], [], []
        for size in SIZES:
            draws = []
            for d in (42, 1, 2):
                key = ("B", tgt, "src:ednet") if (size == 353597 and d == 42) else \
                    ("S11", tgt, f"ednet_n{size}_d{d}")
                draws.append(paired(v[key], scratch)[0])
            m = sum(draws) / 3
            xs.append(size)
            ms.append(m)
            lo.append(m - min(draws))
            hi.append(max(draws) - m)
            print(f"ladder {tgt:<10} {size:>7}: mean {m:+.4f} draws "
                  + " ".join(f"{g:+.4f}" for g in draws))
        ax.errorbar(xs, ms, yerr=[lo, hi], fmt=SHAPE[tgt] + style, color="black", ms=6,
                    capsize=3, lw=1.2, zorder=3,
                    label=f"EdNet subsampled, target {PRETTY[tgt]}")
    for tgt in ("assist2017", "ednet", "junyi"):
        pts = []
        for src in ("assist2017", "junyi", "ednet"):
            if src == tgt:
                continue
            m, l95, h95 = paired(v[("B", tgt, f"src:{src}")], v[("B", tgt, "scratch")])
            pts.append((TRAIN[src], m, l95, h95))
            print(f"real   {src:>10} -> {tgt:<10}: {m:+.4f} [{l95:+.4f}, {h95:+.4f}]")
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    fmt=SHAPE[tgt], mfc="white", color="0.45", ms=6, capsize=2, lw=0.9,
                    zorder=4, label=f"three real sources, target {PRETTY[tgt]}")
    ax.set_xscale("log")
    ticks = [1366, 5000, 15000, 49153, 150000, 353597]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks], fontsize=7)
    ax.minorticks_off()
    ax.set_xlim(1050, 470000)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)
    ax.set_xlabel("pretraining source size (training-split learners)", fontsize=8)
    ax.set_ylabel("paired gain in test AUC\nover scratch", fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out, metadata={"CreationDate": None})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
