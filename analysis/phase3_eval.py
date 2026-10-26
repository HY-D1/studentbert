from __future__ import annotations

"""Phase 3: the final fresh-seed test of the three-epoch framework.

Design: docs/phase3_final_test_design.md.

Committed with the design before any phase 3 job runs. It reuses the phase 2 machinery
(analysis/phase2_eval.py) on seeds 6, 7 and 8 and adds the pre-declared primary test:

  primary   mean over the 30 recommendations of (full framework at epoch 3 regret - H-score regret),
            with a task-cluster bootstrap interval (5 tasks, 20,000 resamples, random.Random(0));
            the three-epoch version holds if the 95% interval lies entirely below zero
  secondary the same difference for the other methods and budgets, the phase 2 outcome table, the
            warning signal at epochs 1 to 3, AG News, MNLI and QNLI alone, and phase 2 and 3 pooled
            for the primary difference (60 recommendations; the phase 2 half has been seen)

  PYTHONPATH=. python analysis/phase3_eval.py \
      --early_ft /projects/algl/dai.hany/task4/early_ft_p3 --head benchmark_final/p3_head \
      --p2_choices benchmark_final/p2_eval_choices.tsv \
      --out benchmark_final/p3_eval
"""

import argparse
import csv
import hashlib
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from analysis import phase2_eval as ev
from analysis import task4_evaluate as te

SEEDS = (6, 7, 8)
P2_CHOICES_MD5 = "21df978b07059d31198046609f88e1d0"
BOOTS = 20000
OTHERS = ["early_3", "full_1", "early_1", "full_2", "early_2", "logme", "best_average"]


def load_head(prefix: str) -> tuple[list[dict], list[dict], str]:
    """The phase 3 head pairs, checked against the md5s their own report lists."""
    rp = Path(prefix + "_report.md")
    if not rp.is_file():
        sys.exit(f"ABORT: {rp} is missing")
    listed = dict(re.findall(r"(\S+\.tsv) ([0-9a-f]{32})", rp.read_text()))
    out = []
    for name in ("pairs", "recs"):
        p = Path(f"{prefix}_{name}.tsv")
        if hashlib.md5(p.read_bytes()).hexdigest() != listed.get(p.name):
            sys.exit(f"ABORT: {p} does not match the md5 in {rp.name}")
        out.append(list(csv.DictReader(p.read_text().splitlines(), delimiter="\t")))
    if sorted({int(r["seed"]) for r in out[1]}) != list(SEEDS) or len(out[1]) != 30:
        sys.exit("ABORT: the head pairs are not the 30 recommendations of seeds 6, 7 and 8")
    return out[0], out[1], hashlib.md5(rp.read_bytes()).hexdigest()


def diff_ci(rows: list[dict], gold: dict, m: str) -> tuple[float, float, float]:
    by = defaultdict(list)
    for r in rows:
        a = ev.outcome(gold, r["task"], r["pooling"], r[m], r["hscore"])["regret"]
        b = ev.outcome(gold, r["task"], r["pooling"], r["hscore"], r["hscore"])["regret"]
        by[r["task"]].append(a - b)
    tasks = sorted(by)
    rng = random.Random(0)
    vals = []
    for _ in range(BOOTS):
        pick = [rng.choice(tasks) for _ in tasks]
        vals.append(st.mean(x for t in pick for x in by[t]))
    vals.sort()
    mean = st.mean(x for t in tasks for x in by[t])
    return mean, vals[int(0.025 * BOOTS)], vals[int(0.975 * BOOTS)]


def holds(ci: tuple[float, float, float]) -> bool:
    return ci[2] < 0


def warning(rows: list[dict], recs: dict) -> list[str]:
    n_rev = sum(r["reversed"] == "1" for r in recs.values())
    L = ["| warning signal | reversals caught | false alarms |", "|---|---|---|"]
    for k in ev.EPOCHS:
        flag = [r for r in rows if r[f"abstain_{k}"] == ""]
        caught = sum(recs[(r["task"], r["pooling"], r["seed"])]["reversed"] == "1" for r in flag)
        L.append(f"| epoch {k} | {ev.pct(caught, n_rev)} | "
                 f"{ev.pct(len(flag) - caught, len(rows) - n_rev)} |")
    return L


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--early_ft", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--p2_choices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ev.SEEDS = SEEDS
    pairs, recl, head_md5 = load_head(a.head)
    gold = te.load_gold()
    models = sorted(gold[("rte", "cls")])
    runs = ev.load_early(a.early_ft, models)
    recs = {(r["task"], r["pooling"], int(r["seed"])): r for r in recl}
    rows = ev.choices(pairs, runs, gold)
    if len(rows) != 30 or any(r["hscore"] != recs[(r["task"], r["pooling"], r["seed"])]["pick"]
                              for r in rows):
        sys.exit("ABORT: the choices do not line up with the 30 phase 3 recommendations")
    p2 = Path(a.p2_choices)
    if hashlib.md5(p2.read_bytes()).hexdigest() != P2_CHOICES_MD5:
        sys.exit(f"ABORT: {p2} is not the phase 2 choices file")
    p2rows = list(csv.DictReader(p2.read_text().splitlines(), delimiter="\t"))

    prim = diff_ci(rows, gold, "full_3")
    nrev = sum(r["reversed"] == "1" for r in recl)
    gpu_h = sum(r["total_s"] for r in runs.values()) / 3600
    L = ["# Phase 3: the three-epoch framework on fresh seeds 6, 7 and 8", "",
         f"30 recommendations, {nrev} with H-score's pick overtaken; head-pair report md5 "
         f"{head_md5}. Early fine-tunes: {len(runs)} runs, all present, 3 epochs, V100-SXM2, "
         f"identical rows across encoders; {gpu_h:.1f} GPU-hours. Regret in published points.", "",
         "## Primary test (fixed before the run)", "",
         f"Full framework at epoch 3 minus H-score: {prim[0]:+.2f} points, 95% task-cluster "
         f"interval [{prim[1]:+.2f}, {prim[2]:+.2f}]. The three-epoch version "
         f"{'HOLDS' if holds(prim) else 'DOES NOT HOLD'} (criterion: the interval lies entirely "
         "below zero).", "", "## Secondary: other methods minus H-score", "",
         "| method | all five tasks [95% CI] | AG News, MNLI, QNLI [95% CI] |", "|---|---|---|"]
    sub = [r for r in rows if r["task"] in ev.FRESH_TASKS]
    for m in ["full_3"] + OTHERS:
        x, y = diff_ci(rows, gold, m), diff_ci(sub, gold, m)
        L.append(f"| {ev.NAMES[m]} | {x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}] | "
                 f"{y[0]:+.2f} [{y[1]:+.2f}, {y[2]:+.2f}] |")
    pooled = diff_ci(rows + p2rows, gold, "full_3")
    L += ["", f"Phase 2 and phase 3 pooled (60 recommendations, seeds 3 to 8; secondary): full "
          f"framework at epoch 3 minus H-score {pooled[0]:+.2f} [{pooled[1]:+.2f}, "
          f"{pooled[2]:+.2f}].", ""]
    L += ev.table(rows, recs, gold, "Outcomes, all five tasks")
    L += ev.table(sub, recs, gold, "Outcomes, AG News, MNLI and QNLI")
    L += warning(rows, recs)
    L += ["", "## Stability across the 3 seeds", "",
          "| method | same choice on all seeds | mean spread of regret (SD) |", "|---|---|---|"]
    for m in ev.METHODS:
        same, n, sd = ev.stability(rows, gold, m)
        L.append(f"| {ev.NAMES[m]} | {same}/{n} | {sd:.2f} |")
    L += ["", "| budget | GPU-minutes per recommendation |", "|---|---|"]
    for k in ev.EPOCHS:
        L.append(f"| {k} epoch{'s' if k > 1 else ''} | {ev.cost_minutes(runs, models, k):.1f} |")
    L += ["", "## Failures of the full framework at epoch 3: choice not tied with the best", "",
          "| task | pooling | seed | H-score's pick | epoch-3 leader | epoch-3 gap | delta | "
          "choice | regret |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        o = ev.outcome(gold, r["task"], r["pooling"], r["full_3"], r["hscore"])
        if not o["tied_best"]:
            L.append(f"| {r['task']} | {r['pooling']} | {r['seed']} | {r['hscore']} | "
                     f"{r['early_3']} | {r['gap_3']:.4f} | {r['delta_3']:.4f} | {r['full_3']} | "
                     f"{o['regret']:.2f} |")
    L += ["", "With 5 tasks and 30 recommendations every estimate here is coarse."]
    out = Path(a.out)
    tp = out.with_name(out.name + "_choices.tsv")
    with tp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    L += ["", f"Files: {tp.name} {hashlib.md5(tp.read_bytes()).hexdigest()}"]
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print(f"primary: full framework at epoch 3 minus H-score {prim[0]:+.2f} [{prim[1]:+.2f}, "
          f"{prim[2]:+.2f}], {'holds' if holds(prim) else 'does not hold'}; wrote {tp.name} and "
          f"{rp.name} md5 {hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
