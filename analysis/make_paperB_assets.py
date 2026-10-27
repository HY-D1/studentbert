from __future__ import annotations

"""Paper B tables and figures, generated from RESULTS.md (no value is typed by hand).

Each asset reads one RESULTS.md section and checks that section's text against a pinned md5, so a
later section added elsewhere in RESULTS.md does not stop the script, while any change to the
numbers it uses does. Output files carry a provenance comment. Tables so far:

  tables/datasets.tex   Table 1, the seven datasets (RESULTS.md 0.1; the practice-per-skill column
                        is left out, because it belongs to another paper)
  tables/sources.tex    Table 2, the best source per target (RESULTS.md 12.1; the column on the
                        largest source is left out, because source scale is not this paper's claim)
  tables/estimators.tex Table 3, the estimator comparison (RESULTS.md 12.3)
  tables/recipes.tex    Appendix table, each source encoder's pretraining recipe, read from the
                        encoders' saved configurations (configs/paperB_encoder_recipes.log, the
                        output of the checkpoint query of job 11396056); written with --recipes
  tables/frozen.tex     Table 4, estimators against the frozen and the fine-tuned gold, and the
                        frozen oracle (RESULTS.md 12.10)
  tables/nlp.tex        Table 5, the estimators on the NLP benchmark of Bassignana et al. (2022),
                        10 settings, with the frozen oracle and the best-average rule (RESULTS.md
                        13.1)
  tables/warning.tex    Table 6, the warning rule on StudentBERT (retrospective) and on the NLP
                        benchmark (the prospective test and two fresh-seed replications) at
                        epochs 1 to 3 (RESULTS.md 12.15, 12.17, 12.18 and 12.19)
  Table 3 also gets Kendall tau-b, pairwise accuracy and seed stability with --summary, from
  benchmark_final/estimators_summary.tsv (md5 a82f6da1 in RESULTS.md 12.7), after its other
  columns are checked against RESULTS.md 12.3.
  figures/trajectories.pdf  Figure 1, the winner minus the own encoder during fine-tuning on the
                        two failure targets (RESULTS.md 12.8); written with --figures (needs
                        matplotlib)

  python analysis/make_paperB_assets.py --results RESULTS.md \
      --recipes configs/paperB_encoder_recipes.log \
      --summary benchmark_final/estimators_summary.tsv --figures --out paperB_assets
"""

import argparse
import ast
import csv
import hashlib
import sys
from pathlib import Path

SECTIONS = {"0.1": ("### 0.1 Dataset characterization", "\n---\n",
                    "7429722373c4661a86577b277c574cb0"),
            "12.1": ("### 12.1 Source selection", "\n### 12.2", "867ebb6c1e6528501851b184983c0029"),
            "12.3": ("### 12.3 Transferability estimators", "\n### 12.4",
                     "cc58ad3b5f534409c8f459ae2fc57a64"),
            "12.8": ("### 12.8 Fine-tuning trajectories", "\n### 12.9",
                     "ff7bebbb8f1ba1770b41c6bb71dd7fba"),
            "12.10": ("### 12.10 Frozen gold", "\n### 12.11", "af544688f9830440f355bd8136d9cce1"),
            "12.15": ("### 12.15 Option 1", "### 12.16", "a30942c60bbb418813e7b2e53c1fe022"),
            "12.17": ("### 12.17 Frozen state", "### 12.18", "8eae065b957bd2524b473bb8f6557d7a"),
            "12.18": ("### 12.18 Phase 2", "### 12.19", "b592641247dec5199fd3444364ff5f64"),
            "12.19": ("### 12.19 Phase 3", "## 13. Cross-domain check",
                      "80424d9df51432eb5f669db193c159cb"),
            "13.1": ("### 13.1 Airline added", "\n_End of consolidated results._",
                     "147f9cf9b1cdba7b2b37108993a3a64d")}
SLUG = {"assist2017": "ASSISTments 2017", "ednet": "EdNet KT1", "junyi": "Junyi Academy",
        "algebra2005": "Algebra 2005", "bridge2006": "Bridge to Algebra 2006",
        "assist2009": "ASSISTments 2009", "algebra2006": "Algebra 2006"}
SUMMARY_MD5 = "a82f6da1cd995a4d13fd4c6a825969a6"
ESTIMATORS = {"hscore_kt_causal": "H-score", "logme_kt_causal": "LogME",
              "largest source": "Largest source",
              "fewshot_ft_fit200_e5": "Short FT (200)",
              "fewshot_ft_fit50_e5": "Short FT (50)",
              "logme_kt_causal_per_skill": "LogME, per skill",
              "nleep_k4_kt_causal": "NLEEP, $K=4$", "nleep_k8_kt_causal": "NLEEP, $K=8$",
              "nleep_k16_kt_causal": "NLEEP, $K=16$"}
NAMES = {"ASSISTments 2017": "ASSISTments 2017", "EdNet KT1": "EdNet KT1",
         "Junyi Academy": "Junyi Academy", "Algebra 2005 (KDD Cup)": "Algebra 2005",
         "Bridge to Algebra 2006 (KDD Cup)": "Bridge to Algebra 2006",
         "ASSISTments 2009": "ASSISTments 2009", "Algebra 2006-2007 (KDD Cup)": "Algebra 2006"}


def section(results: str, key: str) -> str:
    head, end, md5 = SECTIONS[key]
    i = results.find(head)
    if i < 0:
        sys.exit(f"ABORT: RESULTS.md has no section {key}")
    j = results.find(end, i)
    text = results[i:j]
    got = hashlib.md5(text.encode()).hexdigest()
    if got != md5:
        sys.exit(f"ABORT: RESULTS.md {key} changed (md5 {got}, expected {md5}); nothing written")
    return text


def datasets(results: str) -> str:
    rows = [ln for ln in section(results, "0.1").splitlines() if ln.startswith("| ")]
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    if head != ["Dataset", "Students", "Skills", "Interactions", "Median len", "Mean len",
                "Correct rate", "pps"]:
        sys.exit(f"ABORT: unexpected columns {head}")
    body = []
    for ln in rows[1:]:
        c = [x.strip() for x in ln.strip("|").split("|")]
        if c[0] not in NAMES:
            sys.exit(f"ABORT: unexpected dataset {c[0]}")
        med = c[4][:-2] if c[4].endswith(".0") else c[4]
        body.append(f"{NAMES[c[0]]} & {c[1]} & {c[2]} & {c[3]} & {med} & {c[6]} \\\\")
    if len(body) != 7:
        sys.exit(f"ABORT: {len(body)} datasets, expected 7")
    return "\n".join([
        "% Generated by analysis/make_paperB_assets.py from RESULTS.md 0.1; do not edit by hand.",
        "\\begin{table}[t]", "\\centering",
        "\\caption{The seven educational datasets. Each provides one pretrained source encoder "
        "and one knowledge-tracing target. Counts after preprocessing (learners with at least 10 "
        "interactions); correct rate is the share of correct responses.}",
        "\\label{tab:datasets}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Dataset & Learners & Skills & Interactions & Median length & Correct \\\\",
        " & & & & & rate \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def rows(text: str) -> list[list[str]]:
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in text.splitlines()
            if ln.startswith("| ") and not ln.startswith("| ---")][1:]


def tex_num(s: str) -> str:
    return s.replace("-", "$-$") if s.startswith("-") else s


def pretty_sources(cell: str) -> str:
    out = []
    for part in [x.strip() for x in cell.split(",")]:
        words = part.split()
        name = "training from scratch" if words[0] == "scratch" else SLUG[words[0]]
        rest = [w for w in words[1:] if w != "(in-domain)"]
        own = " (own)" if "(in-domain)" in words else ""
        out.append(name + own + (" " + tex_num(rest[0]) if rest else ""))
    return ", ".join(out)


def sources(results: str) -> str:
    body = []
    for c in rows(section(results, "12.1")):
        if c[0] not in SLUG or len(c) != 4:
            sys.exit(f"ABORT: unexpected 12.1 row {c}")
        neg = "none" if c[3] == "none" else pretty_sources(c[3])
        body.append(f"{SLUG[c[0]]} & {pretty_sources(c[1])} & {neg} \\\\")
    if len(body) != 7:
        sys.exit(f"ABORT: {len(body)} targets in 12.1, expected 7")
    return "\n".join([
        "% Generated by analysis/make_paperB_assets.py from RESULTS.md 12.1; do not edit by hand.",
        "\\begin{table}[t]", "\\centering",
        "\\caption{The best source for each target in knowledge tracing (3{,}000 target learners "
        "or the whole training split; six fine-tuning seeds). Best: the candidates tied with the "
        "best one, with their mean AUC gain over training from scratch. Below scratch: sources "
        "whose "
        "gain has a 95\\% interval entirely below zero.}",
        "\\label{tab:sources}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabularx}{\\linewidth}{@{}l>{\\raggedright\\arraybackslash}X"
        ">{\\raggedright\\arraybackslash}X@{}}", "\\toprule",
        "Target & Best (top-equivalent) & Below scratch \\\\", "\\midrule", *body,
        "\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])


def load_summary(path: str) -> dict:
    p = Path(path)
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != SUMMARY_MD5:
        sys.exit(f"ABORT: {p} md5 {got}, expected {SUMMARY_MD5} (RESULTS.md 12.7)")
    rows_ = csv.DictReader(p.read_text().splitlines(), delimiter="\t")
    return {r["estimator"]: r for r in rows_ if r["track"] == "B" and r["view"] == "pretrained"}


def estimators(results: str, summary: dict | None = None) -> str:
    text = section(results, "12.3")
    body = []
    for c in rows(text):
        if c[0] not in ESTIMATORS or len(c) != 8:
            sys.exit(f"ABORT: unexpected 12.3 row {c}")
        mean, mx = [x.strip() for x in c[5].split("/")]
        extra = ""
        if summary is not None:
            if c[0] == "largest source":
                extra = " & -- & -- & --"
            else:
                s = summary[c[0]]
                same = (f"{float(s['rho']):+.4f}" == c[2]
                        and f"{s['top1']}/{s['rankings']}" == c[3]
                        and f"{s['equivalent']}/{s['rankings']}" == c[4]
                        and f"{float(s['mean_regret']):.4f}" == mean
                        and f"{float(s['max_regret']):.4f}" == mx and s["negative_choices"] == c[6])
                if not same:
                    sys.exit(f"ABORT: the summary disagrees with RESULTS.md 12.3 for {c[0]}")
                tau = f"{float(s['tau']):+.3f}"
                extra = (f" & {tex_num(tau)} & {float(s['pair_acc_decisive']):.3f} & "
                         f"{float(s['seed_stability_tau']):.3f}")
        body.append(f"{ESTIMATORS[c[0]]} & {tex_num(c[2])} & {c[3]} & {c[4]} & {mean} & {mx} & "
                    f"{c[6]} & {c[7]}{extra} \\\\")
    if len(body) != 9:
        sys.exit(f"ABORT: {len(body)} estimator rows in 12.3, expected 9")
    note = "hscore_shrunk makes the same pick as hscore in all 21 rankings"
    if note not in " ".join(text.split()):
        sys.exit("ABORT: the 12.3 note on the shrunk H-score moved")
    return "\n".join([
        "% Generated by analysis/make_paperB_assets.py from RESULTS.md 12.3; do not edit by hand.",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Transferability estimators and practical baselines against the fine-tuned gold "
        "(7 targets $\\times$ 3 estimator seeds; the largest-source rule gives one ranking per "
        "target). Largest source: the largest-source rule. Short FT: five epochs of fine-tuning "
        "on 200 or 50 target learners. $\\rho$: mean "
        "Spearman correlation. Top-1 and tied: the top-ranked candidate is "
        "the best one, or tied with it. Regret in test AUC. "
        + ("Neg." if summary else "Negative") + ": picks that transfer "
        "negatively. Own: picks of the target's own encoder. The shrunk H-score makes the same "
        "pick as H-score in all 21 rankings." + (
            " $\\tau_b$: mean Kendall correlation. Pair.: accuracy on the "
            "candidate pairs the gold separates. Stab.: mean $\\tau_b$ between the rankings at "
            "different "
            "estimator seeds; the largest-source rule has one ranking per target." if summary
            else "") + "}",
        "\\label{tab:estimators}", "\\small" if summary is None else "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}" if summary is None else "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular}{lrrrrrrr" + ("rrr" if summary else "") + "}", "\\toprule",
        "Method & $\\rho$ & Top-1 & Tied & \\multicolumn{2}{c}{Regret} & "
        + ("Neg. & Own & $\\tau_b$ & Pair. & Stab." if summary else "Negative & Own")
        + " \\\\",
        "\\cmidrule(lr){5-6}", " & & & & mean & max & &" + (" & & &" if summary else "")
        + " \\\\", "\\midrule",
        *body, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def recipes(log: str) -> str:
    seen = {}
    for ln in log.splitlines():
        if not ln.startswith("edubert_"):
            continue
        name, cfg = ln.split(" ", 1)
        ds = name.removeprefix("edubert_").removesuffix("_pretrain_full_encoder.pt")
        seen[ds] = ast.literal_eval(cfg)
    if sorted(seen) != sorted(SLUG):
        sys.exit(f"ABORT: the recipe log covers {sorted(seen)}, not the seven encoders")
    for ds, c in seen.items():
        if c["lr"] != 0.001 or c["mask_ratio"] != 0.15 or c["objective"] not in (None, "full"):
            sys.exit(f"ABORT: {ds} recipe {c} is not the shared one the caption states")
    body = [f"{SLUG[ds]} & {seen[ds]['epochs']} & {seen[ds]['batch_size']} & "
            f"{seen[ds]['warmup_frac']:g} \\\\" for ds in SLUG]
    return "\n".join([
        "% Generated by analysis/make_paperB_assets.py from the encoders' saved configurations",
        "% (configs/paperB_encoder_recipes.log); do not edit by hand.",
        "\\begin{table}[h]", "\\centering",
        "\\caption{Pretraining recipe of each source encoder, read from its saved configuration. "
        "All seven use the masked-interaction objective with a 15\\% mask ratio and a learning "
        "rate of $10^{-3}$; the three original encoders predate the option that selects the "
        "objective, whose default is this one.}",
        "\\label{tab:recipes}", "\\small", "\\begin{tabular}{lrrr}", "\\toprule",
        "Source encoder & Epochs & Batch size & Warm-up share \\\\", "\\midrule", *body,
        "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def frozen(results: str) -> str:
    body, mean = [], None
    for c in rows(section(results, "12.10")):
        if len(c) != 9:
            sys.exit(f"ABORT: unexpected 12.10 row {c}")
        hs = [x.strip() for x in c[7].split("/")]
        lm = [x.strip() for x in c[8].split("/")]
        if c[0] == "mean over 7":
            mean = (c[5], c[6], hs, lm)
            continue
        if c[0] not in SLUG:
            sys.exit(f"ABORT: unexpected 12.10 target {c[0]}")
        short = {"ednet": "EdNet", "junyi": "Junyi"}
        fb, tb = ("own" if x.removeprefix("src:") == c[0] else short.get(x.removeprefix("src:"),
                  SLUG[x.removeprefix("src:")]) for x in (c[1], c[3]))
        body.append(f"{SLUG[c[0]]} & {fb} & {tb} & {c[6]} & {tex_num(hs[0])} & {tex_num(hs[1])} & "
                    f"{tex_num(lm[0])} & {tex_num(lm[1])} \\\\")
    if len(body) != 7 or mean is None or mean[0] != "5/7":
        sys.exit("ABORT: 12.10 does not hold 7 targets and the mean row with 5/7")
    body.append("\\midrule")
    body.append(f"Mean over 7 & & & {mean[1]} & {tex_num(mean[2][0])} & {tex_num(mean[2][1])} & "
                f"{tex_num(mean[3][0])} & {tex_num(mean[3][1])} \\\\")
    return "\n".join([
        "% Generated by analysis/make_paperB_assets.py from RESULTS.md 12.10; do not edit by hand.",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Frozen against fine-tuned performance (3 estimator seeds). Best, frozen: the "
        "candidate with the best frozen performance, a logistic readout of the estimators' own "
        "features scored on test learners; own: the target's own encoder. The frozen oracle "
        "ranks by that performance; it "
        f"chooses the fine-tuned best on {mean[0]} targets, and its regret is in test AUC. "
        "$\\rho$: mean Spearman correlation of each estimator with the frozen and with the "
        "fine-tuned gold.}",
        "\\label{tab:frozen}", "\\small", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lllrrrrr}", "\\toprule",
        " & \\multicolumn{2}{c}{Best} & Oracle & \\multicolumn{2}{c}{H-score $\\rho$} & "
        "\\multicolumn{2}{c}{LogME $\\rho$} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
        "Target & frozen & tuned & regret & frozen & tuned & frozen & tuned \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def nlp(results: str) -> str:
    text = section(results, "13.1")
    a = text.index("All five tasks run, 10 settings x 3 seeds")
    b = text.index("Reference rankings from the published table on the same 10 settings")
    names = {"hscore": "H-score", "hscore_shrunk": "H-score, shrunk", "logme": "LogME",
             "frozen_oracle": "Frozen oracle", "loto_best_average": "Best-average rule"}
    body = []
    for c in rows(text[a:b]):
        if len(c) != 9 or c[0] not in names:
            sys.exit(f"ABORT: unexpected 13.1 row {c}")
        mean, mx = [x.strip() for x in c[7].split("/")]
        body.append(f"{names[c[0]]} & {c[1]} & {tex_num(c[2])} & {tex_num(c[3])} & {c[4]} & "
                    f"{c[5]} & {c[6]} & {mean} & {mx} & {c[8]} \\\\")
    if len(body) != 6:
        sys.exit(f"ABORT: {len(body)} estimator rows in 13.1, expected 6")
    body.append("\\midrule")
    refs = 0
    for c in rows(text[b:text.index("SciERC is excluded", b)]):
        if len(c) == 8 and c[0] in ("frozen_oracle", "loto_best_average") and c[1] == "tuned":
            mean, mx = [x.strip() for x in c[7].split("/")]
            body.append(f"{names[c[0]]} & {c[1]} & {tex_num(c[2])} & {tex_num(c[3])} & {c[4]} & "
                        f"{c[5]} & {c[6]} & {mean} & {mx} & -- \\\\")
            refs += 1
    if refs != 2:
        sys.exit("ABORT: the frozen oracle and best-average rows against tuned gold are missing")
    return "\n".join([
        "% Generated by analysis/make_paperB_assets.py from RESULTS.md 13.1; do not edit by hand.",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Encoder selection on the NLP benchmark of Bassignana et al.: 7 encoders, 5 "
        "tasks with first-token and mean pooling (10 settings), 3 sample seeds (30 rankings). "
        "Gold: the published frozen or fully fine-tuned performance. Regret in points of the task "
        "metric. Stability: mean Kendall $\\tau_b$ between seeds. The frozen oracle ranks by the "
        "published frozen performance and the best-average rule by the mean fine-tuned rank on the "
        "other tasks; each gives one ranking per setting (10).}",
        "\\label{tab:nlp}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.6pt}",
        "\\begin{tabular}{llrrrrrrrr}", "\\toprule",
        "Method & Gold & $\\rho$ & $\\tau_b$ & Pairwise & Top-1 & Tied & "
        "\\multicolumn{2}{c}{Regret} & Stability \\\\", "\\cmidrule(lr){8-9}",
        " & & & & & & & mean & max & \\\\", "\\midrule", *body, "\\bottomrule",
        "\\end{tabular}", "\\end{table}", ""])


def warning(results: str) -> str:
    def grab(key: str, pattern: str) -> list[str]:
        import re
        m = re.search(pattern, section(results, key), re.M)
        if not m:
            sys.exit(f"ABORT: RESULTS.md {key} has no line matching {pattern}")
        return list(m.groups())

    out = []
    # StudentBERT: 12.17 (epochs 1 and 3, with selective regret) and 12.18 part B (epoch 2)
    rule = r"\| (\d+)/21 \| (\d+)/7 \| (\d+) \| (\d\.\d{4}) \|$"
    sb = {1: grab("12.17", r"^\| fixed epoch-1 rule " + rule),
          3: grab("12.17", r"^\| fixed epoch-3 rule " + rule)}
    e2 = grab("12.18", r"^\| H-score \+ abstention, epoch 2 \| (\d+)/21 \| (\d+)/(\d+) \| "
                       r"(\d+)/(\d+) \|$")
    sb[2] = [e2[0], e2[1], str(int(e2[2]) - int(e2[1])), "--"]
    head = "\\multicolumn{6}{@{}p{0.97\\linewidth}}{\\emph{"
    out.append(head + "StudentBERT, retrospective: 21 recommendations, 7 "
               "overtaken; always following H-score 0.0012 AUC}} \\\\")
    for k in (1, 2, 3):
        fol, caught, fa, reg = sb[k]
        out.append(f"\\quad epoch {k} & {caught}/7 & {fa}/14 & {fol}/21 & {reg} & -- \\\\")
    # NLP, original seeds: 12.15 epochs table
    out.append(head + "NLP, prospective (seeds 42, 1, 2): 30 recommendations, "
               "20 overtaken; always following H-score 2.70 points}} \\\\")
    for k in (1, 2, 3):
        g = grab("12.15", rf"^\| {k} \| (\d+\.\d) \| (\d+)/30 \| (\d+)/20 \| (\d+) \| "
                          r"(\d\.\d{4}) \|")
        out.append(f"\\quad epoch {k} & {g[2]}/20 & {g[3]}/10 & {g[1]}/30 & {float(g[4]):.2f} & "
                   f"{g[0]} \\\\")
    # fresh seeds: the abstention rows of 12.18 and 12.19 (the all-task tables come first)
    for key, seeds, always in (("12.18", "3, 4, 5", "2.37"), ("12.19", "6, 7, 8", "2.70")):
        found = []
        for k in (1, 2, 3):
            g = grab(key, rf"^\| H-score \+ abstention, epoch {k} \| (\d+)/30 \| (\d+)/(\d+) \| "
                          r"(\d+)/(\d+) \| (\d\.\d{2}) \[")
            found.append(g)
        n_rev = int(found[0][1]) + int(found[0][3])
        if any(int(g[1]) + int(g[3]) != n_rev for g in found):
            sys.exit(f"ABORT: {key} abstention rows disagree on the number overtaken")
        out.append(head + f"NLP, fresh seeds {seeds}: 30 "
                   f"recommendations, {n_rev} overtaken; always following H-score {always} "
                   f"points}}}} \\\\")
        for k, g in zip((1, 2, 3), found):
            fa = int(g[2]) - int(g[1])
            cost = grab(key, rf"^\| {k} epochs? \| (\d+\.\d) \|$")[0]
            out.append(f"\\quad epoch {k} & {g[1]}/{n_rev} & {fa}/{30 - n_rev} & {g[0]}/30 & "
                       f"{g[5]} & {cost} \\\\")
    for key, line in (("12.18", "| H-score alone | 12/30 | 12/30 | 2.37 ["),
                      ("12.19", "| H-score alone | 10/30 | 10/30 | 2.70 [")):
        if line not in section(results, key):
            sys.exit(f"ABORT: H-score's regret line is not in RESULTS.md {key}")
    return "\n".join([
        "% Generated by analysis/make_paperB_assets.py from RESULTS.md 12.15, 12.17, 12.18 and "
        "12.19;", "% do not edit by hand.",
        "\\begin{table}[t]", "\\centering",
        "\\caption{The warning rule: flag H-score's recommendation when its choice is not the "
        "validation leader after $k$ epochs of fine-tuning every candidate. Caught: overtaken "
        "recommendations flagged. False alarms: other recommendations flagged. Followed: "
        "recommendations not flagged, and their mean regret (AUC on StudentBERT, points of the "
        "task metric on NLP). Cost: GPU-minutes per recommendation for the short fine-tunes of "
        "the seven encoders.}",
        "\\label{tab:warning}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Budget & Caught & False alarms & Followed & Regret, followed & Cost \\\\", "\\midrule",
        *out, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def trajectories(results: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    for c in rows(section(results, "12.8")):
        if len(c) == 5 and c[0] in ("assist2017", "algebra2006") and c[1].isdigit():
            series.setdefault(c[0], []).append((int(c[1]), float(c[3])))
    if sorted(series) != ["algebra2006", "assist2017"] or any(len(v) != 7 for v in series.values()):
        sys.exit("ABORT: 12.8 does not hold 7 epochs for both failure targets")
    label = {"assist2017": "ASSISTments 2017: EdNet minus own encoder",
             "algebra2006": "Algebra 2006: Junyi Academy minus own encoder"}
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for key, marker in (("assist2017", "o"), ("algebra2006", "s")):
        xs, ys = zip(*series[key])
        ax.plot(xs, ys, marker=marker, linewidth=1.4, markersize=4, label=label[key])
    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xlabel("fine-tuning epoch")
    ax.set_ylabel("validation AUC difference")
    ax.set_xticks([1, 2, 3, 5, 10, 15, 20])
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--recipes", default=None, help="the checkpoint-configuration log")
    ap.add_argument("--figures", action="store_true", help="also draw the figures (matplotlib)")
    ap.add_argument("--summary", default=None, help="benchmark_final/estimators_summary.tsv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    results = Path(a.results).read_text(encoding="utf-8")
    out = Path(a.out) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    made = {"datasets.tex": datasets(results), "sources.tex": sources(results),
            "estimators.tex": estimators(results, load_summary(a.summary) if a.summary else None),
            "frozen.tex": frozen(results), "nlp.tex": nlp(results), "warning.tex": warning(results)}
    if a.recipes:
        made["recipes.tex"] = recipes(Path(a.recipes).read_text(encoding="utf-8"))
    for name, text in made.items():
        p = out / name
        p.write_text(text)
        print(f"{hashlib.md5(p.read_bytes()).hexdigest()}  {p}")
    if a.figures:
        p = Path(a.out) / "figures" / "trajectories.pdf"
        trajectories(results, p)
        print(f"{hashlib.md5(p.read_bytes()).hexdigest()}  {p}")


if __name__ == "__main__":
    main()
