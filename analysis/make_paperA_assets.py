from __future__ import annotations

"""Paper A ("How Much Pretraining Data Is Enough?") tables and figures, generated, never typed.

Every number comes from RESULTS.md (sections 0.1, 2.1, 11 and 11.1) or from the control summary that
RESULTS.md 11.1 was rendered from (per-draw gains for the figure), and the script stops if the two
disagree at 4 dp. Each table file starts with a provenance comment. Outputs:

  tables/datasets.tex, tables/source_comparison.tex, tables/scale_ladder.tex,
  tables/update_matched.tex, tables/matched_scale.tex,
  figures/source_scale_curve.pdf (two panels: learners and optimizer updates),
  figures/loss_vs_transfer.pdf

  PYTHONPATH=. python analysis/make_paperA_assets.py --results RESULTS.md \
      --summary benchmark_final/budget_control_summary.tsv --out paperA_assets
"""

import argparse
import csv
import hashlib
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

PINNED = {"RESULTS.md": "e0837732eabff7f87d67201183280a91",
          "budget_control_summary.tsv": "c1b14438e4d6a547af7952ebbc4b9fe4"}
TRAIN = {"ASSISTments 2017": "1,366", "EdNet KT1": "353,597", "Junyi Academy": "49,153"}
LADDER = (5000, 15000, 49153, 150000, 353597)
TGT = ("assist2017", "junyi")
NUM = r"[+-]?\d+\.\d+"


def q(x, sign: bool = True) -> str:
    d = Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def tex(s: str) -> str:
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def cell(g: str, lo: str, hi: str) -> str:
    """Gain over its interval, stacked, so wide tables need not be shrunk."""
    return f"\\makecell{{{tex(g)}\\\\{{\\scriptsize [{tex(lo)}, {tex(hi)}]}}}}"


def section(text: str, head: str, nxt: str) -> str:
    i = text.index(head)
    j = text.index(nxt, i + len(head))
    return text[i:j]


def parse(results: str) -> dict:
    out: dict = {}
    s0 = section(results, "## 0", "## 1.")
    out["datasets"] = {}
    for name in TRAIN:
        m = re.search(rf"^\| {re.escape(name)} \| ([\d,]+) \| ([\d,]+) \| ([\d,]+) \| ([\d.]+) \| "
                      rf"[\d.]+ \| ([\d.]+) \|", s0, re.M)
        if not m:
            sys.exit(f"ABORT: RESULTS.md 0.1 has no row for {name}")
        out["datasets"][name] = m.groups()
    for v in TRAIN.values():
        if v not in results:
            sys.exit(f"ABORT: RESULTS.md never states the training split {v}")
    s21 = section(results, "### 2.1", "### 2.2")
    cell = re.compile(rf"({NUM}) ±{NUM} \(({NUM}) \[({NUM}),({NUM})\] (\d/\d)\)")
    out["kt"] = {}
    for t in ("assist2017", "ednet", "junyi"):
        m = re.search(rf"^\| {t} \| ({NUM}) ±{NUM} \|(.*)\|\s*$", s21, re.M)
        if not m:
            sys.exit(f"ABORT: RESULTS.md 2.1 has no row for {t}")
        cols = [c.strip() for c in m.group(2).split("|")]
        if len(cols) != 4:
            sys.exit(f"ABORT: RESULTS.md 2.1 row {t} has {len(cols)} condition columns")
        out["kt"][t] = {"scratch": m.group(1),
                        "cells": [cell.match(c).groups() if cell.match(c) else None for c in cols]}
    s11 = section(results, "## 11.", "### 11.1")
    out["loss10"] = {}
    for n in LADDER:
        m = re.search(rf"^\| {n:,} \| ([\d.]+) / ([\d.]+) / ([\d.]+) \|", s11, re.M)
        if not m:
            sys.exit(f"ABORT: RESULTS.md 11 has no loss row for {n:,}")
        out["loss10"][n] = [float(x) for x in m.groups()]
    s111 = section(results, "### 11.1", "## 12.")
    out["r111"] = {}
    row = re.compile(rf"^\| (assist2017|junyi) \| ([\d,]+) \| (\d+) \| ([\d,]+) \| ({NUM}) "
                     rf"\[({NUM}), ({NUM})\] \| (\d/\d) \| ([\d.]+) \| (.*) \|$", re.M)
    for m in row.finditer(s111):
        t, n, ep = m.group(1), int(m.group(2).replace(",", "")), int(m.group(3))
        out["r111"][(t, n, ep)] = m.groups()
    jm = re.search(rf"^\| assist2017 \| 49,153 \(Junyi encoder\) \| 20 \| 7,700 \| ({NUM}) "
                   rf"\[({NUM}), ({NUM})\] \| (\d/\d) \|", s111, re.M)
    if not jm or len(out["r111"]) != 16:
        sys.exit(f"ABORT: RESULTS.md 11.1 table not as expected ({len(out['r111'])} rows)")
    out["junyi_enc"] = jm.groups()
    out["diff"] = {}
    for m in re.finditer(rf"^\| (assist2017|junyi) \| (.+?) minus (.+?) \| ({NUM}) \[({NUM}), "
                         rf"({NUM})\] \| (\d/\d) \|$", s111, re.M):
        out["diff"][(m.group(1), m.group(2), m.group(3))] = m.groups()[3:]
    if len(out["diff"]) != 12:
        sys.exit(f"ABORT: RESULTS.md 11.1 has {len(out['diff'])} differences, expected 12")
    return out


def check_summary(R: dict, rows: list[dict]) -> dict:
    """Per-draw gains from the summary; every mean and interval must equal RESULTS.md 11.1."""
    per = {}
    for r in rows:
        if r["condition"] not in ("ladder", "matched"):
            continue
        k = (r["target"], int(r["source_learners"]), int(r["epochs"]))
        res = R["r111"].get(k)
        if res is None or (q(r["gain"]), q(r["ci_lo"]), q(r["ci_hi"])) != res[4:7]:
            sys.exit(f"ABORT: summary {k} disagrees with RESULTS.md 11.1")
        per[k] = {"gain": float(r["gain"]), "lo": float(r["ci_lo"]), "hi": float(r["ci_hi"]),
                  "draws": [float(x.split(":")[1]) for x in r["per_draw"].split(";")]}
    return per


def table(path: Path, caption: str, label: str, spec: str, header: str, body: list[str],
          prov: str) -> None:
    L = [f"% GENERATED by analysis/make_paperA_assets.py; do not edit by hand. {prov}",
         "\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}",
         f"\\label{{{label}}}",
         "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
         f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header, "\\midrule", *body,
         "\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    path.write_text("\n".join(L) + "\n")


def tables(R: dict, out: Path, prov: str) -> None:
    body = []
    for name, (n, sk, inter, med, cr) in R["datasets"].items():
        body.append(f"{name} & {n} & {TRAIN[name]} & {sk} & {inter} & {med.rstrip('0').rstrip('.')}"
                    f" & {cr} \\\\")
    table(out / "datasets.tex", "Dataset characteristics. Source scale is the number of learners "
          "in the training split, the only data used for pretraining.", "tab:datasets", "lrrrrrr",
          "Dataset & Learners & Train learners & Skills & Interactions & Median length & "
          "Correct rate \\\\", body, prov + " RESULTS.md 0.1.")
    names = {"assist2017": "ASSISTments 2017", "ednet": "EdNet KT1", "junyi": "Junyi Academy"}
    body = []
    for t, v in R["kt"].items():
        cells = []
        for c in v["cells"]:
            cells.append("--" if c is None else cell(c[1], c[2], c[3]))
        body.append(f"{names[t]} & {v['scratch']} & " + " & ".join(cells) + " \\\\")
    table(out / "source_comparison.tex", "Knowledge-tracing transfer at a target budget of "
          "N=3000 learners: scratch test AUC and gain over scratch [95\\% paired-bootstrap "
          "interval], 6 seeds. In-domain pretraining uses the target's own training split; -- "
          "marks the in-domain source.", "tab:sourcecomparison", "lrcccc",
          "Target & Scratch AUC & In-domain & EdNet source & Junyi source & ASSISTments source "
          "\\\\", body, prov + " RESULTS.md 2.1.")
    body = []
    for n in LADDER:
        a, j = R["r111"][("assist2017", n, 10)], R["r111"][("junyi", n, 10)]
        body.append(f"{n:,} & {a[3]} & {cell(a[4], a[5], a[6])} & "
                    f"{cell(j[4], j[5], j[6])} & {a[8]} / {j[8]} \\\\")
    table(out / "scale_ladder.tex", "Knowledge-tracing gain over scratch as the number of EdNet "
          "source learners grows, every encoder pretrained for 10 epochs at batch 128 [95\\% "
          "interval over the 6 fine-tuning seeds of the mean over 3 encoder builds]. Draw spread "
          "is the range of the per-build gains (encoder-build variance).", "tab:scaleladder",
          "rrccr", "Source learners & Updates & ASSISTments 2017 gain & Junyi gain & "
          "Draw spread \\\\", body, prov + " RESULTS.md 11.1.")
    body = []
    groups = (("Fixed 10 epochs", ((5000, 10), (15000, 10), (49153, 10))),
              ("Updates matched to 49,153 $\\times$ 10", ((5000, 96), (15000, 33))),
              ("Twice the updates", ((49153, 20),)))
    for gi, (gname, items) in enumerate(groups):
        if gi:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{gname}}}}} \\\\")
        for n, ep in items:
            a, j = R["r111"][("assist2017", n, ep)], R["r111"][("junyi", n, ep)]
            body.append(f"{n:,} & {ep} & {a[3]} & {cell(a[4], a[5], a[6])} & "
                        f"{cell(j[4], j[5], j[6])} \\\\")
    body.append("\\midrule")
    body.append("\\multicolumn{5}{l}{\\emph{Paired differences against 49,153 $\\times$ 10}} \\\\")
    X = " $\\times$ "
    for a_lab, txt in (("matched 15000 33", f"15,000{X}33"), ("matched 5000 96", f"5,000{X}96"),
                       ("matched 49153 20", f"49,153{X}20")):
        da = R["diff"][("assist2017", a_lab, "ladder 49153 10")]
        dj = R["diff"][("junyi", a_lab, "ladder 49153 10")]
        body.append(f"\\multicolumn{{3}}{{l}}{{{txt} minus 49,153{X}10}} & "
                    f"{cell(da[0], da[1], da[2])} & {cell(dj[0], dj[1], dj[2])} \\\\")
    table(out / "update_matched.tex", "Separating data size from optimization budget (EdNet "
          "source, gain over scratch [95\\% interval]). Updates are ceil(learners / 128) times "
          "epochs. With the number of updates held at the 49,153-learner encoder's, 15,000 "
          "learners reach its gain and 5,000 learners fall short; doubling the updates at 49,153 "
          "learners adds further gain.", "tab:updatematched", "rrrcc",
          "Source learners & Epochs & Updates & ASSISTments 2017 gain & Junyi gain \\\\", body,
          prov + " RESULTS.md 11.1.")
    e10, e20 = R["r111"][("assist2017", 49153, 10)], R["r111"][("assist2017", 49153, 20)]
    # The Junyi row shows the RESULTS.md 2.1 cell, the same number as Table 2; 11.1 resamples the
    # same six seeds with another random stream and agrees at 4 dp on the mean.
    jy = R["kt"]["assist2017"]["cells"][2][1:4]
    if jy[0] != R["junyi_enc"][0]:
        sys.exit("ABORT: the Junyi gain differs between RESULTS.md 2.1 and 11.1")
    d20 = R["diff"][("assist2017", "junyi_source 49153 20", "matched 49153 20")]
    d10 = R["diff"][("assist2017", "junyi_source 49153 20", "ladder 49153 10")]
    body = [f"EdNet & 49,153 & 10 & 3,850 & {cell(e10[4], e10[5], e10[6])} \\\\",
            f"EdNet & 49,153 & 20 & 7,700 & {cell(e20[4], e20[5], e20[6])} \\\\",
            f"Junyi & 49,153 & 20 & 7,700 & {cell(*jy)} \\\\",
            "\\midrule",
            f"\\multicolumn{{4}}{{l}}{{Junyi minus EdNet, both 20 epochs}} & "
            f"{cell(d20[0], d20[1], d20[2])} \\\\",
            f"\\multicolumn{{4}}{{l}}{{Junyi minus EdNet at 10 epochs}} & "
            f"{cell(d10[0], d10[1], d10[2])} \\\\"]
    table(out / "matched_scale.tex", "Matched-scale comparison on ASSISTments 2017: EdNet cut to "
          "the size of the Junyi training split against the Junyi encoder (gain over scratch "
          "[95\\% interval]). At matched learners and matched updates the two sources tie.",
          "tab:matchedscale", "lrrrc", "Source & Learners & Epochs & Updates & Gain \\\\", body,
          prov + " RESULTS.md 11.1.")


def figures(R: dict, per: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"assist2017": "#1f77b4", "junyi": "#d62728"}
    lab = {"assist2017": "ASSISTments 2017", "junyi": "Junyi"}
    upd = {k: int(v[3].replace(",", "")) for k, v in R["r111"].items()}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for t in TGT:
        xs = list(LADDER)
        g = [per[(t, n, 10)]["gain"] for n in xs]
        err = [[per[(t, n, 10)]["gain"] - per[(t, n, 10)]["lo"] for n in xs],
               [per[(t, n, 10)]["hi"] - per[(t, n, 10)]["gain"] for n in xs]]
        for ax, xv in ((axes[0], xs), (axes[1], [upd[(t, n, 10)] for n in xs])):
            ax.errorbar(xv, g, yerr=err, color=col[t], marker="o", ms=4, lw=1.2, capsize=2,
                        label=f"{lab[t]}, 10 epochs")
            for x, n in zip(xv, xs):
                ax.scatter([x] * 3, per[(t, n, 10)]["draws"], color=col[t], s=6, alpha=0.4)
        for n, ep, mk, name in ((15000, 33, "D", "update-matched"), (5000, 96, "D", None),
                                (49153, 20, "s", "twice the updates")):
            p = per[(t, n, ep)]
            e = [[p["gain"] - p["lo"]], [p["hi"] - p["gain"]]]
            for ax, xv in ((axes[0], n), (axes[1], upd[(t, n, ep)])):
                ax.errorbar([xv], [p["gain"]], yerr=e, color=col[t], marker=mk, ms=6,
                            mfc="white", capsize=2, ls="none",
                            label=f"{lab[t]}, {name}" if name else None)
    for ax, xl in ((axes[0], "EdNet source learners (training split)"),
                   (axes[1], "Pretraining optimizer updates")):
        ax.set_xscale("log")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel(xl)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("KT test AUC gain over scratch")
    axes[0].set_title("(a) by number of learners", fontsize=10)
    axes[1].set_title("(b) by number of updates", fontsize=10)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "source_scale_curve.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    def mloss(t, n, ep):
        if ep == 10:
            return sum(R["loss10"][n]) / 3
        cells = R["r111"][(t, n, ep)][9].split(" / ")
        return sum(float(c) for c in cells) / 3

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for t in TGT:
        xs = [mloss(t, n, 10) for n in LADDER]
        ax.plot(xs, [per[(t, n, 10)]["gain"] for n in LADDER], color=col[t], marker="o", ms=4,
                lw=1.2, label=f"{lab[t]}, 10 epochs")
        for n, x in zip(LADDER, xs):
            if t == "assist2017":
                ax.annotate(f"{n // 1000}k" if n < 353597 else "354k",
                            (x, per[(t, n, 10)]["gain"]), textcoords="offset points",
                            xytext=(4, 4), fontsize=7)
        for n, ep in ((15000, 33), (5000, 96), (49153, 20)):
            ax.scatter([mloss(t, n, ep)], [per[(t, n, ep)]["gain"]], color=col[t],
                       marker="D" if ep != 20 else "s", facecolor="white", s=30)
            if t == "assist2017":
                ax.annotate(f"{n // 1000}k \u00d7 {ep}", (mloss(t, n, ep), per[(t, n, ep)]["gain"]),
                            textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.invert_xaxis()
    ax.set_xlabel("Best pretraining loss (training, mean of 3 builds)")
    ax.set_ylabel("KT test AUC gain over scratch")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "loss_vs_transfer.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for p in (a.results, a.summary):
        got = hashlib.md5(Path(p).read_bytes()).hexdigest()
        if got != PINNED[Path(p).name]:
            sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[Path(p).name]}")
    R = parse(Path(a.results).read_text())
    with open(a.summary, newline="") as fh:
        per = check_summary(R, list(csv.DictReader(fh, delimiter="\t")))
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    prov = (f"Inputs: RESULTS.md {PINNED['RESULTS.md']}, budget_control_summary.tsv "
            f"{PINNED['budget_control_summary.tsv']}.")
    tables(R, out / "tables", prov)
    figures(R, per, out / "figures")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(hashlib.md5(p.read_bytes()).hexdigest(), p)


if __name__ == "__main__":
    main()
