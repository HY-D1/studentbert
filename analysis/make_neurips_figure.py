#!/usr/bin/env python3
# Figure for the NeurIPS writeup: knowledge-tracing gain over scratch at N=3000,
# plotted against the number of students the pretraining source was trained on.
#
# The numbers are PARSED OUT OF RESULTS.md section 2.1, never typed in here.
# If a cell in that table is ever corrected, this figure changes with it, so the
# figure and Table 2 in the paper cannot silently disagree.
#
# Source sizes are training-partition student counts, because pretraining reads
# the training split only (scripts/pretrain_edubert.py:106). They come from
# processed/<ds>/vocab_stats.md and are asserted against RESULTS.md section 0.1
# below so a stale constant cannot survive.
#
# Usage, from the repo root on the cluster:
#   python3 analysis/make_neurips_figure.py
#   python3 analysis/make_neurips_figure.py --results RESULTS.md --out source_scale_gain.pdf

from __future__ import annotations

import argparse
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# training-partition students, from processed/<ds>/vocab_stats.md
TRAIN_STUDENTS = {"assist2017": 1366, "junyi": 49153, "ednet": 353597}
# which source each column corresponds to; "indomain" resolves to the target
COL_SOURCE = {"indomain": None, "fromednet": "ednet",
              "fromjunyi": "junyi", "fromassist": "assist2017"}
PRETTY = {"assist2017": "ASSISTments 2017", "ednet": "EdNet KT1",
          "junyi": "Junyi Academy"}
MEAN = re.compile(r"^\s*([0-9]*\.[0-9]+)")
GAIN = re.compile(r"\(([+-][0-9]*\.[0-9]+)\)")
TOTAL = re.compile(r"\|\s*([A-Za-z0-9 ]+?)\s*\|\s*([0-9,]+)\s*\|")


def parse_section(path, heading):
    """Return (header, rows) for the first markdown table under `heading`."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(heading):
            start = i
            break
    if start is None:
        sys.exit("heading not found in %s: %s" % (path, heading))
    table = []
    for ln in lines[start:]:
        if ln.startswith("|"):
            table.append(ln)
        elif table:
            break
    if len(table) < 3:
        sys.exit("no table found under %s" % heading)
    header = [c.strip() for c in table[0].strip("|").split("|")]
    rows = [[c.strip() for c in r.strip("|").split("|")] for r in table[2:]]
    return header, rows


def check_source_sizes(path):
    """Cross-check TRAIN_STUDENTS against the 80% share implied by section 0.1."""
    _, rows = parse_section(path, "### 0.1")
    totals = {}
    for r in rows:
        name = r[0].lower()
        key = None
        if name.startswith("assistments 2017"):
            key = "assist2017"
        elif name.startswith("ednet"):
            key = "ednet"
        elif name.startswith("junyi"):
            key = "junyi"
        if key:
            totals[key] = int(r[1].replace(",", ""))
    for k, train_n in TRAIN_STUDENTS.items():
        if k not in totals:
            sys.exit("could not read total students for %s from section 0.1" % k)
        share = train_n / totals[k]
        if not 0.79 <= share <= 0.81:
            sys.exit("TRAIN_STUDENTS[%s]=%d is not ~80%% of %d (got %.3f)"
                     % (k, train_n, totals[k], share))
    print("source sizes cross-checked against section 0.1:", TRAIN_STUDENTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="RESULTS.md")
    ap.add_argument("--out", default="source_scale_gain.pdf")
    args = ap.parse_args()

    check_source_sizes(args.results)
    header, rows = parse_section(args.results, "### 2.1")
    cols = {name: i for i, name in enumerate(header)}
    for need in ("Target", "scratch"):
        if need not in cols:
            sys.exit("column %r missing from section 2.1 header: %s" % (need, header))

    data = {}
    for r in rows:
        target = r[cols["Target"]].strip()
        m = MEAN.match(r[cols["scratch"]])
        if not m:
            sys.exit("could not read scratch mean for target %s" % target)
        scratch = float(m.group(1))
        noise = 0.0
        nm = re.search(r"[0-9]\.[0-9]+\D+([0-9]*\.[0-9]+)", r[cols["scratch"]])
        if nm:
            noise = float(nm.group(1))
        points = []
        for col, src in COL_SOURCE.items():
            if col not in cols:
                continue
            cell = r[cols[col]]
            g = GAIN.search(cell)
            if not g:
                continue
            source = src if src is not None else target
            if source not in TRAIN_STUDENTS:
                sys.exit("unknown source %r for column %r" % (source, col))
            points.append({"source": source, "n": TRAIN_STUDENTS[source],
                           "gain": float(g.group(1)), "indomain": src is None})
        if points:
            data[target] = {"scratch": scratch, "noise": noise,
                            "points": sorted(points, key=lambda p: p["n"])}

    order = [t for t in ("assist2017", "ednet", "junyi") if t in data]
    if len(order) != 3:
        sys.exit("expected 3 targets in section 2.1, found %s" % list(data))

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    for ax, target in zip(axes, order):
        d = data[target]
        cross = [p for p in d["points"] if not p["indomain"]]
        indom = [p for p in d["points"] if p["indomain"]]
        if d["noise"]:
            ax.axhspan(-d["noise"], d["noise"], color="0.85", zorder=0)
        ax.axhline(0.0, color="0.4", lw=0.8, ls="--", zorder=1)
        if cross:
            ax.plot([p["n"] for p in cross], [p["gain"] for p in cross],
                    "-o", color="#1f77b4", mfc="white", ms=6, zorder=3,
                    label="cross-dataset source")
        for p in indom:
            ax.plot([p["n"]], [p["gain"]], "s", color="#d62728", ms=7,
                    zorder=4, label="in-domain source")
        ax.set_xscale("log")
        ax.set_title("target: %s" % PRETTY[target], fontsize=9)
        ax.set_xlabel("source pretraining students", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.margins(x=0.25)
    axes[0].set_ylabel("KT test AUC gain\nover scratch", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for a, b in zip(h, l):
        if b not in seen:
            seen.add(b)
            hh.append(a)
            ll.append(b)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.legend(hh, ll, fontsize=8, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.savefig(args.out)
    print("wrote %s" % args.out)
    for target in order:
        d = data[target]
        cross = [p for p in d["points"] if not p["indomain"]]
        mono = all(cross[i]["gain"] < cross[i + 1]["gain"] for i in range(len(cross) - 1))
        print("%-11s scratch %.4f  band +/-%.4f  cross-dataset %s  monotone_in_source_size=%s"
              % (target, d["scratch"], d["noise"],
                 [(p["source"], p["gain"]) for p in cross], mono))


if __name__ == "__main__":
    main()
