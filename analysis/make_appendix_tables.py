#!/usr/bin/env python3
"""Emit Appendix C and D tables as LaTeX, straight from the source files.

WHY THIS EXISTS
    Appendix C holds 384 per-seed values across four experiment families.
    Transcribing those by hand into LaTeX is the single most error-prone task
    left in the paper, and this project has already been bitten by numbers that
    were assumed rather than measured. This script reads the source files and
    writes the tables, so no value is ever retyped. It also prints every cell
    mean it computes, which is what you cross-check against the tables already
    in the manuscript.

INPUTS (all in the repo root, all regenerable, none tracked by git)
    objabl_perseed.csv     126 rows  prefix,objective,seed,auc
    trunc_perseed.csv      168 rows  trunc,objective,K,seed,auc
    baseline_perseed.csv    48 rows  base,dataset,model,seed,auc
    probe_perseed.csv       42 rows  probe,dataset,condition,seed,acc
    embedding_analysis/coherence_results.json
    embedding_analysis_matched/coherence_results.json

    A missing input is reported and skipped rather than guessed at.

NOT COVERED, deliberately
    Appendix A is generated from vocab_stats_all.txt, whose format differs
    between the three original datasets and the four added later; a loose parser
    there would be a liability rather than a safeguard, so those values are
    transcribed with the file open alongside. Appendix B has no data file at
    all, only settings already verified from the sbatch scripts and the source.

USAGE (from the repo root)
    python3 analysis/make_appendix_tables.py
    python3 analysis/make_appendix_tables.py --out appendix_tables.tex
"""
import argparse
import csv
import json
import os
import statistics as st

PREFIX_TO_NAME = {
    "w7_objabl": "ASSISTments 2017",
    "w8_regime_ednet": "EdNet KT1",
    "w7_objabl2": "Junyi Academy",
    "w8_algabl": "Algebra 2005",
    "w8_bridgeabl": "Bridge 2006",
    "w8_a09abl": "ASSISTments 2009",
    "w8_alg06abl": "Algebra 2006",
}
OBJ_ORDER = ["full", "skill_only", "correct_only"]
OBJ_LABEL = {"full": "Full", "skill_only": "Skill only",
             "correct_only": "Correct. only", "scratch": "Scratch"}
SEEDS = ["42", "1", "2", "3", "4", "5"]
SEEDS3 = ["42", "1", "2"]
KS = ["10", "20", "40", "80", "160", "320", "512"]
DS_ORDER = ["assist2017", "ednet", "junyi", "algebra2005",
            "bridge2006", "assist2009", "algebra2006"]
DS_LABEL = {"assist2017": "ASSISTments 2017", "ednet": "EdNet KT1",
            "junyi": "Junyi Academy", "algebra2005": "Algebra 2005",
            "bridge2006": "Bridge 2006", "assist2009": "ASSISTments 2009",
            "algebra2006": "Algebra 2006"}


def load_rows(path, ncol):
    if not os.path.exists(path):
        print(f"  MISSING {path}, skipping its table")
        return None
    out = []
    with open(path, newline="") as fh:
        for r in csv.reader(fh):
            if len(r) == ncol and r[-1].strip():
                out.append([c.strip() for c in r])
    print(f"  read {path}: {len(out)} usable rows")
    return out


def fmt(v):
    return f"{v:.4f}"


def stat_cell(vals):
    if not vals:
        return "--"
    if len(vals) == 1:
        return fmt(vals[0])
    return f"{st.mean(vals):.4f} $\\pm$ {st.pstdev(vals):.4f}"


def table_objective(rows, W, audit):
    lookup = {(r[0], r[1], r[2]): float(r[3]) for r in rows}
    W("\\begin{table}[t]\\centering")
    W("\\caption{\\harry{Per-seed knowledge-tracing test AUC for the pretraining"
      " objective comparison. Six seeds per cell.}}")
    W("\\label{app:tab:objseeds}\\small")
    W("\\begin{tabular}{llcccccc c}")
    W("\\toprule")
    W("Target & Objective & " + " & ".join("s" + s for s in SEEDS)
      + " & Mean $\\pm$ SD \\\\")
    W("\\midrule")
    for pre in PREFIX_TO_NAME:
        for j, obj in enumerate(OBJ_ORDER):
            vals = [lookup.get((pre, obj, s)) for s in SEEDS]
            got = [v for v in vals if v is not None]
            name = PREFIX_TO_NAME[pre] if j == 0 else ""
            cells = " & ".join(fmt(v) if v is not None else "--" for v in vals)
            W(f"{name} & {OBJ_LABEL[obj]} & {cells} & {stat_cell(got)} \\\\")
            if got:
                audit.append(f"objective {PREFIX_TO_NAME[pre]:18s} "
                             f"{obj:13s} mean {st.mean(got):.4f} (n={len(got)})")
        W("\\addlinespace")
    W("\\bottomrule\\end{tabular}\\end{table}")
    W("")


def table_truncation(rows, W, audit):
    lookup = {(r[1], r[2], r[3]): float(r[4]) for r in rows}
    W("\\begin{table}[t]\\centering")
    W("\\caption{\\harry{Per-seed knowledge-tracing test AUC for the"
      " ASSISTments 2017 sequence-truncation sweep. Six seeds per cell.}}")
    W("\\label{app:tab:truncseeds}\\small")
    W("\\begin{tabular}{llcccccc c}")
    W("\\toprule")
    W("$K$ & Condition & " + " & ".join("s" + s for s in SEEDS)
      + " & Mean $\\pm$ SD \\\\")
    W("\\midrule")
    for k in KS:
        for j, obj in enumerate(OBJ_ORDER + ["scratch"]):
            vals = [lookup.get((obj, k, s)) for s in SEEDS]
            got = [v for v in vals if v is not None]
            label = k if j == 0 else ""
            cells = " & ".join(fmt(v) if v is not None else "--" for v in vals)
            W(f"{label} & {OBJ_LABEL[obj]} & {cells} & {stat_cell(got)} \\\\")
            if got:
                audit.append(f"truncation K={k:>3s} {obj:13s} "
                             f"mean {st.mean(got):.4f} (n={len(got)})")
        W("\\addlinespace")
    W("\\bottomrule\\end{tabular}\\end{table}")
    W("")


def table_baseline(rows, W, audit):
    lookup = {(r[1], r[2], r[3]): float(r[4]) for r in rows}
    ds = [d for d in DS_ORDER if any(r[1] == d for r in rows)]
    W("\\begin{table}[t]\\centering")
    W("\\caption{\\harry{Per-seed knowledge-tracing test AUC for the baseline"
      " models. Three seeds for DKT and AKT, six for the scratch encoder.}}")
    W("\\label{app:tab:baseseeds}\\small")
    W("\\begin{tabular}{llcccccc c}")
    W("\\toprule")
    W("Dataset & Model & " + " & ".join("s" + s for s in SEEDS)
      + " & Mean $\\pm$ SD \\\\")
    W("\\midrule")
    for d in ds:
        for j, m in enumerate(["dkt", "akt", "scratch"]):
            vals = [lookup.get((d, m, s)) for s in SEEDS]
            got = [v for v in vals if v is not None]
            label = DS_LABEL.get(d, d) if j == 0 else ""
            cells = " & ".join(fmt(v) if v is not None else "--" for v in vals)
            W(f"{label} & {m.upper()} & {cells} & {stat_cell(got)} \\\\")
            if got:
                audit.append(f"baseline {DS_LABEL.get(d, d):18s} {m:8s} "
                             f"mean {st.mean(got):.4f} (n={len(got)})")
        W("\\addlinespace")
    W("\\bottomrule\\end{tabular}\\end{table}")
    W("")


def table_probe(rows, W, audit):
    lookup = {(r[1], r[2], r[3]): float(r[4]) for r in rows}
    W("\\begin{table}[t]\\centering")
    W("\\caption{\\harry{Per-seed masked-skill probe accuracy, pretrained"
      " against randomly initialized encoders. Three seeds per cell. Absolute"
      " values are not comparable across datasets.}}")
    W("\\label{app:tab:probeseeds}\\small")
    W("\\begin{tabular}{lccc c ccc c c}")
    W("\\toprule")
    W(" & \\multicolumn{3}{c}{Pretrained} & & \\multicolumn{3}{c}{Scratch} & &"
      " \\\\")
    W("\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}")
    W("Dataset & " + " & ".join("s" + s for s in SEEDS3) + " & & "
      + " & ".join("s" + s for s in SEEDS3) + " & & Difference \\\\")
    W("\\midrule")
    for d in DS_ORDER:
        pre = [lookup.get((d, "full", s)) for s in SEEDS3]
        scr = [lookup.get((d, "scratch", s)) for s in SEEDS3]
        gp = [v for v in pre if v is not None]
        gs = [v for v in scr if v is not None]
        if not gp or not gs:
            continue
        diff = st.mean(gp) - st.mean(gs)
        pc = " & ".join(fmt(v) if v is not None else "--" for v in pre)
        sc = " & ".join(fmt(v) if v is not None else "--" for v in scr)
        W(f"{DS_LABEL.get(d, d)} & {pc} & & {sc} & & ${diff:+.4f}$ \\\\")
        audit.append(f"probe {DS_LABEL.get(d, d):18s} pretrained "
                     f"{st.mean(gp):.4f}  scratch {st.mean(gs):.4f}  "
                     f"diff {diff:+.4f}  separation "
                     f"{'complete' if min(gp) > max(gs) else 'INCOMPLETE'}")
    W("\\bottomrule\\end{tabular}\\end{table}")
    W("")


def table_coherence(raw, matched, W, audit):
    by = {r["dataset"]: r for r in raw}
    bm = {r["dataset"]: r for r in matched}
    W("\\begin{table}[t]\\centering")
    W("\\caption{\\harry{Skill-embedding coherence, computed on each dataset's"
      " own in-domain encoder. The matched column restricts to the 100 most"
      " frequent skills in every dataset, which removes the vocabulary-size"
      " confound. The separation between regimes present in the raw column does"
      " not survive that control.}}")
    W("\\label{app:tab:coherence}\\small")
    W("\\begin{tabular}{lrrrr}")
    W("\\toprule")
    W("Dataset & Skills & Regime & Raw $\\rho$ & Matched $\\rho$ \\\\")
    W("\\midrule")
    for d in DS_ORDER:
        if d not in by or d not in bm:
            continue
        r, m = by[d], bm[d]
        reg = "Correctness" if r["regime"].startswith("corr") else "Skill"
        W(f"{DS_LABEL.get(d, d)} & {r['n_skills']:,} & {reg} & "
          f"{r['coherence_rho_cooccur']:+.4f} & "
          f"{m['coherence_rho_cooccur']:+.4f} \\\\")
        audit.append(f"coherence {DS_LABEL.get(d, d):18s} raw "
                     f"{r['coherence_rho_cooccur']:+.4f}  matched "
                     f"{m['coherence_rho_cooccur']:+.4f}  "
                     f"pairs {m['n_cooccur_pairs']}")
    W("\\bottomrule\\end{tabular}\\end{table}")
    W("")
    xs = [bm[d]["n_skills"] for d in DS_ORDER if d in bm]
    ys = [bm[d]["coherence_rho_cooccur"] for d in DS_ORDER if d in bm]
    if len(xs) > 2:
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs)
               * sum((b - my) ** 2 for b in ys)) ** 0.5
        audit.append(f"coherence vs n_skills, matched file, "
                     f"coherence_rho_cooccur: r = {num / den:+.4f} (n={len(xs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="appendix_tables.tex")
    args = ap.parse_args()

    lines, audit = [], []
    W = lines.append
    W("% Generated by analysis/make_appendix_tables.py. Do not hand-edit;")
    W("% regenerate instead. Every value is read from a source file.")
    W("")

    print("reading inputs:")
    obj = load_rows("objabl_perseed.csv", 4)
    tru = load_rows("trunc_perseed.csv", 5)
    bas = load_rows("baseline_perseed.csv", 5)
    pro = load_rows("probe_perseed.csv", 5)

    raw = matched = None
    for p, tag in [("embedding_analysis/coherence_results.json", "raw"),
                   ("embedding_analysis_matched/coherence_results.json", "matched")]:
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"  read {p}: {len(d)} datasets")
            if tag == "raw":
                raw = d
            else:
                matched = d
        else:
            print(f"  MISSING {p}, skipping its table")

    if obj:
        table_objective(obj, W, audit)
    if tru:
        table_truncation(tru, W, audit)
    if bas:
        table_baseline(bas, W, audit)
    if pro:
        table_probe(pro, W, audit)
    if raw and matched:
        table_coherence(raw, matched, W, audit)

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\nAUDIT, cross-check these against the tables already in the paper:")
    for a in audit:
        print("  " + a)
    print(f"\nwrote {args.out}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
