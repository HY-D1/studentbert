"""Two-panel gains figure for the LAK paper: performance_by_data_regime.pdf.

Panel (a): top-1 accuracy gain over scratch, N=25..1000.
Panel (b): macro-OVR AUC gain over scratch, same N.
Shows individual seed values (dots) and the three-seed mean (line) for the
in-domain, Junyi, and EdNet conditions, paired by seed against scratch.

Validates every computed mean gain against the verified values from
RESULTS.md sections 3.1/3.2 before writing the figure (exit 2 on mismatch).

Run from the repo code root:
  python analysis/make_performance_figure.py \
      --csv nextskill_results_long.csv --out performance_by_data_regime.pdf
"""
import argparse
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

LEVELS = [25, 50, 100, 200, 500, 1000]
CONDS = ["indomain", "junyi", "ednet"]
LABEL = {"indomain": "In-domain", "junyi": "Junyi", "ednet": "EdNet"}
COLOR = {"indomain": "#0072B2", "junyi": "#009E73", "ednet": "#D55E00"}
JITTER = {"indomain": 0.94, "junyi": 1.0, "ednet": 1.065}

EXPECTED = {
    "top1": {
        "indomain": [0.0922, 0.0140, 0.0063, 0.0067, 0.0115, 0.0120],
        "ednet": [0.0340, -0.0046, -0.0015, 0.0027, 0.0089, 0.0120],
        "junyi": [0.0858, 0.0116, 0.0066, 0.0077, 0.0137, 0.0145],
    },
    "macro_auc": {
        "indomain": [0.0887, 0.0270, 0.0116, 0.0059, 0.0035, 0.0033],
        "ednet": [0.0456, 0.0081, 0.0008, 0.0022, 0.0027, 0.0027],
        "junyi": [0.0781, 0.0251, 0.0103, 0.0056, 0.0043, 0.0036],
    },
}


def load(csv_path):
    v = {}
    seeds = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "assist2017":
                continue
            if row["metric"] not in ("top1", "macro_auc"):
                continue
            key = (row["condition"], int(row["N"]), int(row["seed"]), row["metric"])
            v[key] = float(row["value"])
            seeds.add(int(row["seed"]))
    return v, sorted(seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nextskill_results_long.csv")
    ap.add_argument("--out", default="performance_by_data_regime.pdf")
    ap.add_argument("--skip-check", action="store_true")
    a = ap.parse_args()

    v, seeds = load(a.csv)
    print(f"loaded {len(v)} values, seeds {seeds}", flush=True)

    gains = {}
    for m in ("top1", "macro_auc"):
        for c in CONDS:
            for N in LEVELS:
                g = []
                for s in seeds:
                    g.append(v[(c, N, s, m)] - v[("scratch", N, s, m)])
                gains[(c, N, m)] = g

    bad = 0
    for m in ("top1", "macro_auc"):
        for c in CONDS:
            means = [float(np.mean(gains[(c, N, m)])) for N in LEVELS]
            exp = EXPECTED[m][c]
            ok = all(abs(x - e) <= 6e-4 for x, e in zip(means, exp))
            tag = "MATCH" if ok else "MISMATCH vs RESULTS.md"
            print(f"{m:10s} {c:9s} " + " ".join(f"{x:+.4f}" for x in means)
                  + f"  [{tag}]", flush=True)
            if not ok:
                bad += 1
    if bad and not a.skip_check:
        print(f"{bad} series disagree with the verified means; figure NOT written.")
        sys.exit(2)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
    panels = [("top1", "Top-1 accuracy gain over scratch", "(a)"),
              ("macro_auc", "Macro-OVR AUC gain over scratch", "(b)")]
    for ax, (m, ylab, tag) in zip(axes, panels):
        ax.axhline(0.0, color="0.55", lw=0.9, ls="--", zorder=1)
        for c in CONDS:
            xs = np.array(LEVELS, dtype=float)
            means = [float(np.mean(gains[(c, N, m)])) for N in LEVELS]
            ax.plot(xs, means, "-o", color=COLOR[c], lw=1.8, ms=4.5,
                    label=LABEL[c], zorder=3)
            for i, N in enumerate(LEVELS):
                xj = N * JITTER[c]
                ax.scatter([xj] * len(gains[(c, N, m)]), gains[(c, N, m)],
                           s=13, color=COLOR[c], alpha=0.5, linewidths=0,
                           zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(LEVELS)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("Target-data size $N$ (learners)")
        ax.set_ylabel(ylab)
        ax.text(0.02, 0.97, tag, transform=ax.transAxes, va="top",
                fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print(f"figure written to {a.out}")


if __name__ == "__main__":
    main()
