from __future__ import annotations

"""Phase 2 evaluation on the fresh estimator seeds (docs/phase2_complete_rule_design.md and
docs/phase2_amendment1.md). Committed before it first reads the fresh early fine-tunes.

Inputs: the fresh head pairs and recommendations (analysis/nlp_head_pairs.py on the fresh scores;
the published tuned gold with the Task 4 tie rule), the 210 fresh early fine-tunes
(scripts/task4_early_ft.py, seeds 3, 4 and 5) and the published table. Every method chooses one
encoder per recommendation, except H-score + abstention, which may abstain:

  hscore        H-score's pick (the ablation without the early signal)
  logme         LogME's own top pick
  early_k       the validation leader after k epochs (one-epoch fine-tuning alone at k = 1; the
                ablation without H-score); ties broken by H-score
  full_k        the full framework: H-score's pick, switched to the epoch-k leader only when the
                leader's validation accuracy exceeds the pick's by more than
                delta = 1.96 x sqrt(2 p (1 - p) / n), p their mean accuracy, n the validation rows
  abstain_k     H-score + abstention: follow the pick when it leads at epoch k, otherwise abstain
  best_average  the Task 4 leave-one-task-out rule (it does not depend on the seed)

Outcomes: top-1, tied with the best, mean regret in published points (task-cluster bootstrap,
20,000 resamples, random.Random(0)), reversal recovery and false corrections (Wilson 95%), switches
to an encoder tied with the pick, wins, ties and losses against H-score, coverage and calibration of
the abstention rule, stability across the 3 seeds of each setting, compute per recommendation, the
failures of the full framework and its switches between near-tied encoders; everything for all 5
tasks and for AG News, MNLI and QNLI alone. Primary: full_1 against hscore on mean regret.

  PYTHONPATH=. python analysis/phase2_eval.py \
      --early_ft /projects/algl/dai.hany/task4/early_ft_fresh \
      --head benchmark_final/p2_head --out benchmark_final/p2_eval
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from analysis import task4_evaluate as te
from analysis.task4_ceiling import loto_scores

TASKS = ("agnews", "mnli", "qnli", "rte", "airline")
FRESH_TASKS = ("agnews", "mnli", "qnli")
POOLINGS = ("cls", "mean")
SEEDS = (3, 4, 5)
EPOCHS = (1, 2, 3)
HEAD_REPORT_MD5 = "62702fa6156404a02f6d5f34cf856d1d"
BOOTS = 20000


def slug(m: str) -> str:
    return m.replace("/", "__")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def delta(a: float, b: float, n: int) -> float:
    p = (a + b) / 2
    return 1.96 * math.sqrt(2 * p * (1 - p) / n)


def load_head(prefix: str) -> tuple[list[dict], list[dict]]:
    rp = Path(prefix + "_report.md")
    if hashlib.md5(rp.read_bytes()).hexdigest() != HEAD_REPORT_MD5:
        sys.exit(f"ABORT: {rp} is not the fresh head-pair report")
    listed = dict(re.findall(r"(\S+\.tsv) ([0-9a-f]{32})", rp.read_text()))
    out = []
    for name in ("pairs", "recs"):
        p = Path(f"{prefix}_{name}.tsv")
        if hashlib.md5(p.read_bytes()).hexdigest() != listed.get(p.name):
            sys.exit(f"ABORT: {p} does not match the md5 in {rp.name}")
        out.append(list(csv.DictReader(p.read_text().splitlines(), delimiter="\t")))
    return out[0], out[1]


def load_early(root: str, models: list[str]) -> dict:
    runs, fails = {}, []
    for t in TASKS:
        for p in POOLINGS:
            for s in SEEDS:
                for m in models:
                    f = Path(root) / t / f"{slug(m)}_{p}_s{s}.json"
                    if not f.is_file():
                        fails.append(f"missing {f}")
                        continue
                    r = json.loads(f.read_text())
                    if [e["epoch"] for e in r["epochs"]] != list(EPOCHS):
                        fails.append(f"{f}: epochs")
                    if "V100-SXM2" not in r.get("gpu", ""):
                        fails.append(f"{f}: gpu {r.get('gpu')}")
                    runs[(t, p, s, m)] = r
    for t in TASKS:
        for s in SEEDS:
            rs = [runs[(t, p, s, m)] for p in POOLINGS for m in models if (t, p, s, m) in runs]
            for k in ("train_idx_md5", "val_idx_md5", "n_val"):
                if len({r[k] for r in rs}) > 1:
                    fails.append(f"{t} seed {s}: encoders differ in {k}")
    if fails:
        sys.exit("ABORT, fresh early fine-tunes incomplete or inconsistent:\n  "
                 + "\n  ".join(fails[:20]))
    return runs


def choices(pairs: list[dict], runs: dict, gold: dict) -> list[dict]:
    """One row per fresh recommendation with every method's choice."""
    by = defaultdict(list)
    for r in pairs:
        by[(r["task"], r["pooling"], int(r["seed"]))].append(r)
    rows = []
    for (t, p, s), rs in sorted(by.items()):
        h = rs[0]["pick"]
        models = [h] + [r["alternative"] for r in rs]
        hs = {h: float(rs[0]["hscore_pick"]), **{r["alternative"]: float(r["hscore_alt"])
                                                  for r in rs}}
        lm = {h: float(rs[0]["logme_pick"]), **{r["alternative"]: float(r["logme_alt"])
                                                 for r in rs}}
        row = {"task": t, "pooling": p, "seed": s, "hscore": h,
               "logme": max(models, key=lambda m: (lm[m], hs[m])),
               "best_average": te.judge(loto_scores(gold, t, p),
                                        {m: gold[(t, p)][m]["tuned"] for m in gold[(t, p)]})
               ["choice"]}
        for k in EPOCHS:
            acc = {m: runs[(t, p, s, m)]["epochs"][k - 1]["val_acc"] for m in models}
            n = runs[(t, p, s, h)]["n_val"]
            lead = max(models, key=lambda m: (acc[m], hs[m]))
            d = delta(acc[lead], acc[h], n)
            row[f"early_{k}"] = lead
            row[f"full_{k}"] = lead if acc[lead] - acc[h] > d else h
            row[f"abstain_{k}"] = h if lead == h else ""
            row[f"gap_{k}"] = acc[lead] - acc[h]
            row[f"delta_{k}"] = d
        rows.append(row)
    return rows


def outcome(gold: dict, t: str, p: str, c: str, h: str) -> dict:
    perf = {m: gold[(t, p)][m]["tuned"] for m in gold[(t, p)]}
    best = max(perf, key=lambda m: perf[m][0])
    return {"regret": perf[best][0] - perf[c][0], "top1": c == best,
            "tied_best": c == best or te.tied(perf[c], perf[best]),
            "vs_pick": "same" if c == h else ("tie" if te.tied(perf[c], perf[h]) else
                                              ("better" if perf[c][0] > perf[h][0] else "worse"))}


def boot(vals: dict) -> tuple[float, float]:
    """Task-cluster bootstrap of a mean; vals maps task -> list of values."""
    tasks = [t for t in vals if vals[t]]
    rng = random.Random(0)
    out = []
    for _ in range(BOOTS):
        pick = [rng.choice(tasks) for _ in tasks]
        xs = [x for t in pick for x in vals[t]]
        if xs:
            out.append(st.mean(xs))
    out.sort()
    return out[int(0.025 * len(out))], out[min(len(out) - 1, int(0.975 * len(out)))]


def summarize(rows: list[dict], recs: dict, gold: dict, method: str) -> dict:
    res = {"n": len(rows), "followed": 0, "regret": [], "by_task": defaultdict(list), "top1": 0,
           "tied": 0, "rec_k": 0, "rec_n": 0, "fc_k": 0, "fc_n": 0, "harmless": 0,
           "wins": 0, "ties": 0, "losses": 0, "flag_rev": 0, "flag_n": 0, "fol_rev": 0}
    for r in rows:
        c, h = r[method], r["hscore"]
        rev = recs[(r["task"], r["pooling"], r["seed"])]["reversed"] == "1"
        if not c:
            res["flag_n"] += 1
            res["flag_rev"] += rev
            continue
        o = outcome(gold, r["task"], r["pooling"], c, h)
        oh = outcome(gold, r["task"], r["pooling"], h, h)
        res["followed"] += 1
        res["fol_rev"] += rev
        res["regret"].append(o["regret"])
        res["by_task"][r["task"]].append(o["regret"])
        res["top1"] += o["top1"]
        res["tied"] += o["tied_best"]
        if rev:
            res["rec_n"] += 1
            res["rec_k"] += o["tied_best"]
        else:
            res["fc_n"] += 1
            res["fc_k"] += o["vs_pick"] == "worse"
            res["harmless"] += o["vs_pick"] == "tie"
        if o["vs_pick"] == "same" or abs(o["regret"] - oh["regret"]) < 1e-12:
            res["ties"] += 1
        elif o["regret"] < oh["regret"]:
            res["wins"] += 1
        else:
            res["losses"] += 1
    res["mean"] = st.mean(res["regret"]) if res["regret"] else float("nan")
    res["ci"] = boot(res["by_task"]) if res["regret"] else (float("nan"), float("nan"))
    return res


def stability(rows: list[dict], gold: dict, method: str) -> tuple[int, int, float]:
    by = defaultdict(list)
    for r in rows:
        by[(r["task"], r["pooling"])].append(r)
    same, sds = 0, []
    for (t, p), rs in by.items():
        cs = [r[method] for r in rs]
        same += len(set(cs)) == 1
        regs = [outcome(gold, t, p, c, r["hscore"])["regret"] for r, c in zip(rs, cs) if c]
        if len(regs) > 1:
            sds.append(st.pstdev(regs))
    return same, len(by), st.mean(sds) if sds else float("nan")


def cost_minutes(runs: dict, models: list[str], k: int) -> float:
    per = [sum(sum(e["epoch_s"] for e in runs[(t, p, s, m)]["epochs"][:k]) for m in models) / 60
           for t in TASKS for p in POOLINGS for s in SEEDS]
    return st.mean(per)


def pct(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{100 * lo:.0f}%, {100 * hi:.0f}%]"


METHODS = (["hscore", "logme"] + [f"early_{k}" for k in EPOCHS] + [f"full_{k}" for k in EPOCHS]
           + ["best_average"])
NAMES = {"hscore": "H-score alone", "logme": "LogME alone", "best_average": "best-average rule",
         **{f"early_{k}": f"early leader, epoch {k}" for k in EPOCHS},
         **{f"full_{k}": f"full framework, epoch {k}" for k in EPOCHS},
         **{f"abstain_{k}": f"H-score + abstention, epoch {k}" for k in EPOCHS}}


def table(rows: list[dict], recs: dict, gold: dict, title: str) -> list[str]:
    L = [f"## {title}", "", "| method | top-1 | tied with best | mean regret [95% CI] | "
         "reversal recovery | false corrections | harmless switches | "
         "wins/ties/losses vs H-score |",
         "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        s = summarize(rows, recs, gold, m)
        L.append(f"| {NAMES[m]} | {s['top1']}/{s['n']} | {s['tied']}/{s['n']} | {s['mean']:.2f} "
                 f"[{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] | {pct(s['rec_k'], s['rec_n'])} | "
                 f"{pct(s['fc_k'], s['fc_n'])} | {s['harmless']} | "
                 f"{s['wins']}/{s['ties']}/{s['losses']} |")
    L += ["", "| abstention rule | followed (coverage) | reversed among flagged | reversed among "
          "followed | selective regret [95% CI] |", "|---|---|---|---|---|"]
    for k in EPOCHS:
        s = summarize(rows, recs, gold, f"abstain_{k}")
        L.append(f"| {NAMES[f'abstain_{k}']} | {s['followed']}/{s['n']} | "
                 f"{s['flag_rev']}/{s['flag_n']} | {s['fol_rev']}/{s['followed']} | "
                 f"{s['mean']:.2f} [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}] |")
    return L + [""]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--early_ft", required=True)
    ap.add_argument("--head", required=True, help="prefix of the fresh head-pair files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pairs, recl = load_head(a.head)
    gold = te.load_gold()
    models = sorted(gold[("rte", "cls")])
    runs = load_early(a.early_ft, models)
    recs = {(r["task"], r["pooling"], int(r["seed"])): r for r in recl}
    rows = choices(pairs, runs, gold)
    if len(rows) != 30 or any(r["hscore"] != recs[(r["task"], r["pooling"], r["seed"])]["pick"]
                              for r in rows):
        sys.exit("ABORT: the choices do not line up with the 30 fresh recommendations")
    hs = summarize(rows, recs, gold, "hscore")
    if abs(hs["mean"] - st.mean(float(r["regret_points"]) for r in recl)) > 1e-9:
        sys.exit("ABORT: H-score's regret does not match the fresh head-pair recommendations")
    nrev = sum(r["reversed"] == "1" for r in recl)
    gpu_h = sum(r["total_s"] for r in runs.values()) / 3600
    L = ["# Phase 2: the full framework on fresh seeds", "",
         f"30 fresh recommendations (seeds 3, 4, 5), {nrev} with H-score's pick overtaken. "
         f"Early fine-tunes: {len(runs)} runs, all present, 3 epochs, V100-SXM2, identical rows "
         f"across encoders; {gpu_h:.1f} GPU-hours. Regret in published fine-tuned points.", ""]
    L += table(rows, recs, gold, "All five tasks")
    sub = [r for r in rows if r["task"] in FRESH_TASKS]
    L += table(sub, recs, gold, "AG News, MNLI and QNLI (new samples)")
    L += ["## Stability across the 3 fresh seeds", "",
          "| method | settings with the same choice on all seeds | mean spread of regret (SD) |",
          "|---|---|---|"]
    for m in METHODS:
        same, n, sd = stability(rows, gold, m)
        L.append(f"| {NAMES[m]} | {same}/{n} | {sd:.2f} |")
    L += ["", "## Compute", "", "| budget | GPU-minutes per recommendation |", "|---|---|"]
    for k in EPOCHS:
        L.append(f"| {k} epoch{'s' if k > 1 else ''} | {cost_minutes(runs, models, k):.1f} |")
    L += ["", "## Failures of the full framework (epoch 1): choice not tied with the best", "",
          "| task | pooling | seed | H-score's pick | epoch-1 leader | epoch-1 gap | delta | "
          "choice | regret |", "|---|---|---|---|---|---|---|---|---|"]
    near = 0
    for r in rows:
        o = outcome(gold, r["task"], r["pooling"], r["full_1"], r["hscore"])
        near += r["full_1"] != r["hscore"] and o["vs_pick"] == "tie"
        if not o["tied_best"]:
            L.append(f"| {r['task']} | {r['pooling']} | {r['seed']} | {r['hscore']} | "
                     f"{r['early_1']} | {r['gap_1']:.4f} | {r['delta_1']:.4f} | {r['full_1']} | "
                     f"{o['regret']:.2f} |")
    L += ["", f"Switches of the full framework (epoch 1) between encoders the published results "
          f"cannot separate: {near}.", "", "With 5 tasks and 30 recommendations every estimate "
          "here is coarse."]
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
    f1 = summarize(rows, recs, gold, "full_1")
    print(f"full framework, epoch 1: mean regret {f1['mean']:.2f} against {hs['mean']:.2f} for "
          f"H-score; wrote {tp.name} and {rp.name} md5 {hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
