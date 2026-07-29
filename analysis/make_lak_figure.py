#!/usr/bin/env python3
# Build the LAK "break-even N" figures from nextskill_results_agg.csv
# (the file parse_nextskill_full.py writes). No new experiments needed.
#
#   break_even_n_curve.pdf/.png   -> 2-panel (top-1 | macro-OVR AUC) vs N, full width
#   break_even_macro_single.*     -> single-panel macro-AUC, half width (clean option)
#
# Usage:  python make_lak_figure.py [--csv nextskill_results_agg.csv] [--dataset assist2017]

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COND_ORDER = ["scratch", "indomain", "ednet", "junyi"]
COND_LABEL = {"scratch": "Scratch", "indomain": "In-domain (ASSISTments)",
              "ednet": "EdNet", "junyi": "Junyi"}
# colorblind-safe; scratch is the gray dashed baseline
STYLE = {
    "scratch":  dict(color="#7f7f7f", marker="o", ls="--", lw=1.8),
    "indomain": dict(color="#1f77b4", marker="s", ls="-",  lw=2.0),
    "ednet":    dict(color="#d62728", marker="^", ls="-",  lw=2.0),
    "junyi":    dict(color="#2ca02c", marker="D", ls="-",  lw=2.0),
}


def load(csv_path, dataset):
    # (metric, cond) -> {N: (mean, std)}
    d = defaultdict(dict)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["dataset"] != dataset:
                continue
            d[(r["metric"], r["condition"])][int(r["N"])] = (float(r["mean"]), float(r["std"]))
    return d


def series(d, metric, cond):
    pts = sorted(d[(metric, cond)].items())
    Ns = [n for n, _ in pts]
    mu = [m for _, (m, _) in pts]
    sd = [s for _, (_, s) in pts]
    return Ns, mu, sd


def panel(ax, d, metric, ylabel, low_n=75, legend=False):
    for cond in COND_ORDER:
        if (metric, cond) not in d:
            continue
        Ns, mu, sd = series(d, metric, cond)
        ax.errorbar(Ns, mu, yerr=sd, label=COND_LABEL[cond], capsize=2.5,
                    markersize=5, **STYLE[cond])
    ax.set_xscale("log")
    all_N = sorted({n for (m, c), v in d.items() if m == metric for n in v})
    ax.set_xticks(all_N)
    ax.set_xticklabels([str(n) for n in all_N])
    ax.axvspan(min(all_N) * 0.85, low_n, color="#f0c000", alpha=0.10, lw=0)
    ax.set_xlabel("Target training set size (student sequences, log scale)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.margins(x=0.03)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="lower right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nextskill_results_agg.csv")
    ap.add_argument("--dataset", default="assist2017")
    args = ap.parse_args()
    d = load(args.csv, args.dataset)

    plt.rcParams.update({"font.size": 10, "font.family": "serif",
                         "axes.spines.top": False, "axes.spines.right": False})

    # --- main 2-panel figure ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.7))
    panel(a1, d, "top1", "Next-skill top-1 accuracy", legend=True)
    panel(a2, d, "macro_auc", "Next-skill macro-OVR AUC")
    a1.set_title("(a) Top-1 accuracy", fontsize=10)
    a2.set_title("(b) Macro-OVR AUC", fontsize=10)
    fig.tight_layout()
    fig.savefig("break_even_n_curve.pdf", bbox_inches="tight")
    fig.savefig("break_even_n_curve.png", dpi=150, bbox_inches="tight")

    # --- single-panel macro-AUC (half-width, cleanest convergence) ---
    fig2, ax = plt.subplots(figsize=(5.2, 3.6))
    panel(ax, d, "macro_auc", "Next-skill macro-OVR AUC", legend=True)
    fig2.tight_layout()
    fig2.savefig("break_even_macro_single.pdf", bbox_inches="tight")
    fig2.savefig("break_even_macro_single.png", dpi=150, bbox_inches="tight")
    print("wrote break_even_n_curve.{pdf,png} and break_even_macro_single.{pdf,png}")


if __name__ == "__main__":
    main()
