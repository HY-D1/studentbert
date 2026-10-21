from __future__ import annotations

"""Phase 2 supplement: descriptive additions to the fresh-seed NLP result, and StudentBERT.

Added after the phase 2 results (analysis/phase2_eval.py, job 11271624) were seen. It changes no
rule, threshold or choice; part A describes the saved choices further, part B applies the phase 2
methods retrospectively to the 7 StudentBERT targets, as the design (section 5) specifies.

A. NLP, from benchmark_final/p2_eval_choices.tsv: maximum regret per method; each method's regret
   minus H-score's per recommendation, averaged, with a task-cluster bootstrap interval (20,000
   resamples, random.Random(0)), for all five tasks and for AG News, MNLI and QNLI; and the warning
   signal (flag when H-score's pick is not the epoch-k leader) as reversals caught and false alarms.
B. StudentBERT, from benchmark_final/reversal_pairs.tsv (21 recommendations, 8 candidates, gold =
   mean test AUC at N=3000, the benchmark's tie rule through each pair's label): H-score alone,
   LogME alone, the early leader at epochs 1 to 3 (validation AUC of the fine-tune whose seed equals
   the estimator seed; ties to H-score), the full framework at epochs 1 to 3 (switch only when the
   leader's validation AUC exceeds the pick's by more than 0.001, the benchmark's practical tie
   margin) and H-score + abstention. Regret in AUC, with a target-cluster bootstrap. Retrospective:
   these targets were examined before.

  PYTHONPATH=. python analysis/phase2_supplement.py \
      --choices benchmark_final/p2_eval_choices.tsv --recs benchmark_final/p2_head_recs.tsv \
      --pairs benchmark_final/reversal_pairs.tsv --out benchmark_final/p2_supplement
"""

import argparse
import csv
import hashlib
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from analysis import phase2_eval as ev
from analysis import task4_evaluate as te

PINNED = {"p2_eval_choices.tsv": "21df978b07059d31198046609f88e1d0",
          "p2_head_recs.tsv": "a776f71b11a673d815dc6f486e4eb668",
          "reversal_pairs.tsv": "8f57558d142869b3e34a35ad18d095f7"}
NLP_METHODS = ["logme", "early_1", "early_2", "early_3", "full_1", "full_2", "full_3",
               "best_average"]
SB_MARGIN = 0.001
BOOTS = 20000


def read_pinned(path: str) -> list[dict]:
    p = Path(path)
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != PINNED[p.name]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[p.name]}")
    return list(csv.DictReader(p.read_text().splitlines(), delimiter="\t"))


def cluster_ci(by: dict) -> tuple[float, float, float]:
    keys = sorted(k for k in by if by[k])
    rng = random.Random(0)
    vals = []
    for _ in range(BOOTS):
        pick = [rng.choice(keys) for _ in keys]
        vals.append(st.mean(x for k in pick for x in by[k]))
    vals.sort()
    mean = st.mean(x for k in keys for x in by[k])
    return mean, vals[int(0.025 * BOOTS)], vals[int(0.975 * BOOTS)]


def part_a(choices: list[dict], recs: list[dict], gold: dict) -> list[str]:
    rev = {(r["task"], r["pooling"], r["seed"]): r["reversed"] == "1" for r in recs}

    def reg(r: dict, m: str) -> float:
        return ev.outcome(gold, r["task"], r["pooling"], r[m], r["hscore"])["regret"]

    L = ["## A. NLP fresh seeds: descriptive additions", "",
         "| method | maximum regret | minus H-score, all tasks [95% CI] | minus H-score, AG News, "
         "MNLI, QNLI [95% CI] |", "|---|---|---|---|",
         f"| H-score alone | {max(reg(r, 'hscore') for r in choices):.2f} | | |"]
    for m in NLP_METHODS:
        out = []
        for sub in (None, ev.FRESH_TASKS):
            by = defaultdict(list)
            for r in choices:
                if sub is None or r["task"] in sub:
                    by[r["task"]].append(reg(r, m) - reg(r, "hscore"))
            mean, lo, hi = cluster_ci(by)
            out.append(f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        L.append(f"| {ev.NAMES[m]} | {max(reg(r, m) for r in choices):.2f} | {out[0]} | "
                 f"{out[1]} |")
    L += ["", "| warning signal | reversals caught | false alarms |", "|---|---|---|"]
    n_rev = sum(rev.values())
    for k in ev.EPOCHS:
        caught = sum(1 for r in choices if r[f"abstain_{k}"] == ""
                     and rev[(r["task"], r["pooling"], r["seed"])])
        fa = sum(1 for r in choices if r[f"abstain_{k}"] == ""
                 and not rev[(r["task"], r["pooling"], r["seed"])])
        L.append(f"| epoch {k} | {ev.pct(caught, n_rev)} | {ev.pct(fa, len(choices) - n_rev)} |")
    return L + [""]


def sb_recs(pairs: list[dict]) -> list[dict]:
    """One record per StudentBERT recommendation with every candidate's scores and gold."""
    label = {}
    for r in pairs:
        label[(r["target"], r["est_seed"], frozenset((r["cand_a"], r["cand_b"])))] = r["label_main"]
    by = defaultdict(list)
    for r in pairs:
        if r["frozen_rank_a"] == "1":
            by[(r["target"], r["est_seed"])].append(r)
    out = []
    for (t, s), rs in sorted(by.items()):
        h = rs[0]["cand_a"]
        c = {h: {"hs": float(rs[0]["hscore_a"]), "lm": float(rs[0]["logme_a"]),
                 "gold": float(rs[0]["gold_a"]),
                 **{f"v{k}": float(rs[0][f"val_e{k}_a"]) for k in ev.EPOCHS}}}
        for r in rs:
            c[r["cand_b"]] = {"hs": float(r["hscore_b"]), "lm": float(r["logme_b"]),
                              "gold": float(r["gold_b"]),
                              **{f"v{k}": float(r[f"val_e{k}_b"]) for k in ev.EPOCHS}}
        if len(c) != 8:
            sys.exit(f"ABORT: {t} seed {s} has {len(c)} candidates")
        rec = {"target": t, "seed": s, "hscore": h, "cand": c,
               "reversed": any(r["label_main"] == "reversal" for r in rs),
               "tie": lambda x, y, t=t, s=s: x == y or label.get(
                   (t, s, frozenset((x, y)))) == "near_tie",
               "worse_than_pick": lambda x, t=t, s=s, h=h: x != h and label.get(
                   (t, s, frozenset((x, h)))) == "stable"}
        rec["logme"] = max(c, key=lambda m: (c[m]["lm"], c[m]["hs"]))
        for k in ev.EPOCHS:
            lead = max(c, key=lambda m: (c[m][f"v{k}"], c[m]["hs"]))
            rec[f"early_{k}"] = lead
            rec[f"full_{k}"] = lead if c[lead][f"v{k}"] - c[h][f"v{k}"] > SB_MARGIN else h
            rec[f"abstain_{k}"] = h if lead == h else ""
        out.append(rec)
    return out


def part_b(recs: list[dict]) -> list[str]:
    methods = (["hscore", "logme"] + [f"early_{k}" for k in ev.EPOCHS]
               + [f"full_{k}" for k in ev.EPOCHS])
    n_rev = sum(r["reversed"] for r in recs)
    L = ["## B. StudentBERT, retrospective (21 recommendations, 7 with H-score's pick overtaken)",
         "", "| method | top-1 | tied with best | mean regret (AUC) [95% CI] | reversal recovery | "
         "false corrections | stable across seeds |", "|---|---|---|---|---|---|---|"]
    for m in methods:
        by, top, tied, rec_k, fc_k = defaultdict(list), 0, 0, 0, 0
        for r in recs:
            c = r["cand"]
            best = max(c, key=lambda x: c[x]["gold"])
            ch = r[m]
            by[r["target"]].append(c[best]["gold"] - c[ch]["gold"])
            top += ch == best
            tb = r["tie"](ch, best)
            tied += tb
            if r["reversed"]:
                rec_k += tb
            else:
                fc_k += r["worse_than_pick"](ch)
        mean, lo, hi = cluster_ci(by)
        per = defaultdict(set)
        for r in recs:
            per[r["target"]].add(r[m])
        stable = sum(len(v) == 1 for v in per.values())
        if m == "hscore" and (top, tied, f"{mean:.4f}") != (11, 14, "0.0012"):
            sys.exit("ABORT: H-score on StudentBERT does not reproduce RESULTS.md 12.3")
        L.append(f"| {ev.NAMES[m]} | {top}/21 | {tied}/21 | {mean:.4f} [{lo:.4f}, {hi:.4f}] | "
                 f"{ev.pct(rec_k, n_rev)} | {ev.pct(fc_k, 21 - n_rev)} | {stable}/7 |")
    L += ["", "| abstention rule | followed | reversed among flagged | reversed among followed |",
          "|---|---|---|---|"]
    for k in ev.EPOCHS:
        fol = [r for r in recs if r[f"abstain_{k}"]]
        fl = [r for r in recs if not r[f"abstain_{k}"]]
        L.append(f"| H-score + abstention, epoch {k} | {len(fol)}/21 | "
                 f"{sum(r['reversed'] for r in fl)}/{len(fl)} | "
                 f"{sum(r['reversed'] for r in fol)}/{len(fol)} |")
    return L + [""]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--choices", required=True)
    ap.add_argument("--recs", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gold = te.load_gold()
    L = ["# Phase 2 supplement (added after the phase 2 results; describes the same choices, and "
         "StudentBERT retrospectively)", ""]
    L += part_a(read_pinned(a.choices), read_pinned(a.recs), gold)
    L += part_b(sb_recs(read_pinned(a.pairs)))
    L.append("Every interval here is coarse: 5 NLP tasks and 7 StudentBERT targets.")
    rp = Path(a.out + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print(f"wrote {rp.name} md5 {hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
