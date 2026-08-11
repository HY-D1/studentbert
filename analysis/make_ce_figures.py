#!/usr/bin/env python3
"""Two figures for the C&E: AI manuscript. CPU only, matplotlib, seconds.

FIGURE 1  truncation_sweep.pdf
    ASSISTments 2017 test AUC against the truncation length K, four conditions.
    This is the causal panel. It shows the scratch control across the whole
    sweep including K=512, which was run 2026-08-11 (jobs 9081378-9081383,
    6 seeds, mean 0.6676). The endpoint is where correct_only falls BELOW
    scratch, which is what rules out a generic data-quantity explanation.

FIGURE 2  pps_vs_objective_gap.pdf
    Practice-per-skill against the skill_only minus correct_only gap for all
    seven datasets, with 95% paired-bootstrap intervals. Uses the EFFECTIVE
    pps definition, min(median length, 512) / n_skills, because that is the
    quantity the encoder actually sees and the quantity Figure 1 manipulates.

PROVENANCE
    Truncation values: RESULTS.md section 5 truncation sweep, parsed from
        w8_trunc_*.log; the K=512 scratch cell from w8_trunc_scratch_k512_s*.
    Gaps and intervals: analysis/paired_bootstrap_objective.py over
        objabl_perseed.csv, 6 seeds, 20000 resamples, rng seed 0.
    Dataset statistics: RESULTS.md section 0.1.
    Every value is printed to stdout when the script runs, so the figure can
    be checked against the tables without opening the PDF.

USAGE (from the repo root)
    python3 analysis/make_ce_figures.py
    python3 analysis/make_ce_figures.py --outdir figures --png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# K -> (full, skill_only, correct_only, scratch); scratch at 512 added 2026-08-11
TRUNC = {
    10:  (0.6357, 0.6362, 0.6483, 0.6347),
    20:  (0.6410, 0.6399, 0.6426, 0.6476),
    40:  (0.6438, 0.6415, 0.6504, 0.6423),
    80:  (0.6496, 0.6450, 0.6509, 0.6523),
    160: (0.6609, 0.6559, 0.6605, 0.6586),
    320: (0.6614, 0.6589, 0.6599, 0.6597),
    512: (0.6941, 0.6920, 0.6655, 0.6676),
}

# dataset -> (median_len, n_skills, gap_mean, ci_lo, ci_hi, regime)
GAPS = {
    "Junyi":        (  87.0, 1326, -0.0122, -0.0136, -0.0110, "correctness"),
    "EdNet":        (  30.0,  142, -0.0069, -0.0078, -0.0055, "correctness"),
    "ASSIST 2009":  (  40.0,  123, +0.0030, +0.0023, +0.0038, "skill"),
    "Bridge 2006":  (1373.0,  492, +0.0105, +0.0093, +0.0116, "skill"),
    "Algebra 2006": (1168.5,  484, +0.0186, +0.0148, +0.0228, "skill"),
    "ASSIST 2017":  ( 441.0,  102, +0.0240, +0.0222, +0.0261, "skill"),
    "Algebra 2005": ( 581.0,  109, +0.0228, +0.0173, +0.0281, "skill"),
}

CAP = 512
ASSIST_SKILLS = 102
ASSIST_MEDIAN = 441.0

STYLE = {
    "full":         ("#1b3a6b", "o", "-",  "Full objective"),
    "skill_only":   ("#2e7d4f", "s", "-",  "Skill only"),
    "correct_only": ("#a83232", "^", "-",  "Correctness only"),
    "scratch":      ("#6b6b6b", "D", "--", "From scratch"),
}


def effective_pps(median_len, n_skills, cap=CAP):
    return min(median_len, cap) / n_skills


def figure_truncation(path, dpi):
    Ks = sorted(TRUNC)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, key in enumerate(["full", "skill_only", "correct_only", "scratch"]):
        colour, marker, ls, label = STYLE[key]
        ys = [TRUNC[k][i] for k in Ks]
        ax.plot(Ks, ys, color=colour, marker=marker, linestyle=ls,
                linewidth=1.6, markersize=5, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("Truncation length $K$ (interactions retained per learner)")
    ax.set_ylabel("Test AUC")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

    sec = ax.secondary_xaxis(
        "top",
        functions=(lambda k: k, lambda k: k))
    sec.set_xticks(Ks)
    sec.set_xticklabels([f"{min(k, ASSIST_MEDIAN) / ASSIST_SKILLS:.2f}" for k in Ks],
                        fontsize=8)
    sec.set_xlabel("Effective practice-per-skill", fontsize=9)

    ax.annotate("correctness-only\nfalls below scratch",
                xy=(512, TRUNC[512][2]), xytext=(150, 0.6425),
                fontsize=8, color="#a83232",
                arrowprops=dict(arrowstyle="->", color="#a83232", lw=0.9))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def figure_gaps(path, dpi):
    rows = sorted(GAPS.items(), key=lambda kv: effective_pps(kv[1][0], kv[1][1]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for name, (med, ns, mean, lo, hi, regime) in rows:
        x = effective_pps(med, ns)
        colour = "#a83232" if regime == "correctness" else "#2e7d4f"
        marker = "^" if regime == "correctness" else "s"
        ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]],
                    fmt=marker, color=colour, markersize=6,
                    capsize=3, elinewidth=1.2, linewidth=0)
        ax.annotate(name, xy=(x, mean), xytext=(0, 9),
                    textcoords="offset points", fontsize=8,
                    ha="center", color="#333333")

    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.axvspan(0.325, 1.041, color="#cccccc", alpha=0.35, zorder=0)
    ax.annotate("no dataset\nin this range", xy=(0.58, -0.0155),
                fontsize=8, ha="center", color="#666666")

    ax.set_xscale("log")
    ax.set_xlabel("Effective practice-per-skill, "
                  r"$\min(\mathrm{median\ length},\ 512)\ /\ \mathrm{skills}$")
    ax.set_ylabel("Skill-only minus correctness-only (test AUC)")
    ax.grid(True, alpha=0.25, linewidth=0.6)

    handles = [
        plt.Line2D([], [], color="#2e7d4f", marker="s", linewidth=0,
                   label="Skill-driven"),
        plt.Line2D([], [], color="#a83232", marker="^", linewidth=0,
                   label="Correctness-driven"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--png", action="store_true",
                    help="also write PNG copies for slides")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Figure 1 values (ASSISTments 2017 truncation sweep):")
    print("    K   effective pps    full   skill   correct  scratch")
    for k in sorted(TRUNC):
        f, s, c, sc = TRUNC[k]
        print(f"  {k:4d}   {min(k, ASSIST_MEDIAN)/ASSIST_SKILLS:12.2f}  "
              f"{f:.4f}  {s:.4f}  {c:.4f}  {sc:.4f}")
    g320 = TRUNC[320][1] - TRUNC[320][2]
    g512 = TRUNC[512][1] - TRUNC[512][2]
    print(f"  skill minus correct: K=320 {g320:+.4f}, K=512 {g512:+.4f}")
    print(f"  K=320 to K=512 change: scratch {TRUNC[512][3]-TRUNC[320][3]:+.4f}, "
          f"full {TRUNC[512][0]-TRUNC[320][0]:+.4f}, "
          f"skill {TRUNC[512][1]-TRUNC[320][1]:+.4f}, "
          f"correct {TRUNC[512][2]-TRUNC[320][2]:+.4f}")

    print("\nFigure 2 values (skill_only minus correct_only, 6 seeds):")
    print("  dataset          eff pps    raw pps      gap      95% CI")
    for name, (med, ns, mean, lo, hi, reg) in sorted(
            GAPS.items(), key=lambda kv: effective_pps(kv[1][0], kv[1][1])):
        print(f"  {name:14s}  {effective_pps(med, ns):7.3f}  {med/ns:9.3f}  "
              f"{mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {reg}")

    f1 = os.path.join(args.outdir, "truncation_sweep.pdf")
    f2 = os.path.join(args.outdir, "pps_vs_objective_gap.pdf")
    figure_truncation(f1, args.dpi)
    figure_gaps(f2, args.dpi)
    written = [f1, f2]
    if args.png:
        p1 = f1.replace(".pdf", ".png")
        p2 = f2.replace(".pdf", ".png")
        figure_truncation(p1, args.dpi)
        figure_gaps(p2, args.dpi)
        written += [p1, p2]

    print("\nwrote:")
    for w in written:
        print(f"  {w}  ({os.path.getsize(w):,} bytes)")


if __name__ == "__main__":
    main()
