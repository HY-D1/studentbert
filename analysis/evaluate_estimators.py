#!/usr/bin/env python3
"""Evaluate transferability estimators against the benchmark gold (MRAP Task 3, Section Q).

Gold is rebuilt from the per-seed primary rows of executions.tsv (build_transfer_benchmark.py),
with the builder's own paired-bootstrap statistics, so estimator and benchmark cannot disagree on
what "best" or "tied" means. Two views are kept apart, as Section Q requires:
  pretrained  ranking among pretrained candidates only;
  practical   scratch is an action too, so choosing transfer where scratch is safer counts.
A choice is near-tie-aware correct if it is in the top-equivalent set: statistically (the paired
CI of best minus choice includes zero) or practically (the gap is within --margin, fixed from the
measured rerun effect before any held-out evaluation).

Estimators:
  LogME and any other scorer   JSON lines from scripts/score_transferability.py (--scores)
  largest source               pretraining training-split learners (retrospective heuristic)
  masked-skill probe           probe2 rows of executions.tsv (retrospective heuristic)
  Track A policies             always full, always skill_only, always correct_only, and the
                               leave-one-dataset-out threshold rules of analysis/lodo_regime.py
None of these fits anything on a held-out target's gold: LogME and the probe use target inputs
and labels only, the constants fit nothing, and the LODO rules are refit without the held-out
dataset in every fold. The held-out requirement of Task 3 step 1 is therefore met by construction.

  PYTHONPATH=. python3 analysis/evaluate_estimators.py \\
      --executions benchmark_20260922/executions.tsv --scores tg1_logme_kt_*.jsonl \\
      --out-prefix benchmark_20260922/estimators
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

try:
    from analysis import lodo_regime
    from analysis.build_transfer_benchmark import (TRACK_METRIC, cell_stats, kendall_tau_b, q4,
                                                   split_draw_cells)
except ModuleNotFoundError:  # run without PYTHONPATH=.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis import lodo_regime
    from analysis.build_transfer_benchmark import (TRACK_METRIC, cell_stats, kendall_tau_b, q4,
                                                   split_draw_cells)

# Pretraining scale as the papers quote it (training-split learners); splits.json must agree.
DOCUMENTED_TRAIN = {"assist2017": 1366, "junyi": 49153, "ednet": 353597}
CKPT = re.compile(r"^edubert_([a-z0-9]+)_pretrain_full_encoder\.pt$")


def load_gold(path: str) -> tuple[dict, dict]:
    """(cells, probes): cells[(track, target, budget)][candidate][seed] = value."""
    cells: dict = defaultdict(lambda: defaultdict(dict))
    probes: dict = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["valid_for_primary_analysis"] != "yes":
                continue
            t = r["track"]
            if t not in ("A", "B", "C-ns", "C-drop", "probe2") or r["metric"] != TRACK_METRIC[t]:
                continue
            if not r["finetune_seed"]:
                raise SystemExit(f"primary {t} row without a seed: {r['run_id']} "
                                 f"in {r['log_file']}")
            seed, v = int(r["finetune_seed"]), float(r["value"])
            if t in ("A", "B", "C-ns", "C-drop"):
                cells[(t, r["target_dataset"], r["budget"])][r["candidate"]][seed] = v
            elif t == "probe2":
                probes[r["target_dataset"]][r["candidate"]][seed] = v
    cells, _ = split_draw_cells(cells)
    return cells, probes


def train_split_sizes(root: str, sources: set[str]) -> dict:
    """{"src:<ds>": training-split learners} read from <root>/<ds>/splits.json.

    The encoders pretrain on the train partition only, so that is the size the rule ranks by.
    A missing file stops the run instead of silently dropping a candidate from the ranking.
    """
    out = {}
    for src in sorted(sources):
        ds = src.split(":", 1)[1]
        f = Path(root) / ds / "splits.json"
        if not f.exists():
            raise SystemExit(f"largest source: no {f}; pass --processed-root")
        n = len(json.loads(f.read_text())["train"])
        if ds in DOCUMENTED_TRAIN and n != DOCUMENTED_TRAIN[ds]:
            raise SystemExit(f"{f} has {n} training learners, RESULTS.md scale says "
                             f"{DOCUMENTED_TRAIN[ds]}")
        out[src] = n
    return out


def annotate(values: dict, margin: float, boots: int) -> dict:
    """Gold for exactly these candidates: means, best and ties all over their shared seeds.

    Computed per view, so "tied with best" in the pretrained view means tied with the best
    pretrained candidate, not with scratch (v2 read 10/11 top-1 but 9/11 tied, 2026-09-23).
    """
    stats = {r["candidate"]: r for r in cell_stats(values, boots, random.Random(0))}
    common = sorted(set.intersection(*(set(v) for v in values.values())))
    for c, r in stats.items():
        r["gold"] = st.mean(values[c][s] for s in common) if common else r["mean"]
    best = max(stats, key=lambda c: stats[c]["gold"])
    for c, r in stats.items():
        r["practical_tie"] = stats[best]["gold"] - r["gold"] <= margin
        r["equivalent"] = r["top_equivalent"] in ("best", "yes") or r["practical_tie"]
    return stats


def ranks(scores: dict) -> dict:
    order = sorted(scores, key=lambda c: -scores[c])
    out, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(a: dict, b: dict) -> float:
    ra, rb = ranks(a), ranks(b)
    keys = list(a)
    ma, mb = st.mean(ra[k] for k in keys), st.mean(rb[k] for k in keys)
    num = sum((ra[k] - ma) * (rb[k] - mb) for k in keys)
    den = math.sqrt(sum((ra[k] - ma) ** 2 for k in keys) * sum((rb[k] - mb) ** 2 for k in keys))
    return num / den if den else float("nan")


def judge(scores: dict, stats: dict, signs: dict, margin: float, ranking: bool = True) -> dict:
    """Metrics of one ranking against gold; ranking=False scores a bare choice (a policy).

    stats is the gold of the whole view (every candidate in it, scored or not), so a choice is
    judged against the best available action; an estimator that cannot score a candidate does
    not get to leave it out. Rank correlations use the scored candidates, reported with the
    coverage. signs is the full cell's, so negative transfer is always judged against scratch.
    """
    keys = [c for c in scores if c in stats]
    allg = {c: r["gold"] for c, r in stats.items()}
    gold = {c: allg[c] for c in keys}
    choice = max(keys, key=lambda c: scores[c])
    best, worst = max(allg, key=allg.get), min(allg, key=allg.get)
    out = {"choice": choice, "top1": choice == best, "equivalent": stats[choice]["equivalent"],
           "regret": allg[best] - allg[choice],
           "norm_regret": (allg[best] - allg[choice]) / (allg[best] - allg[worst])
           if allg[best] > allg[worst] else 0.0,
           "negative_choice": signs[choice].get("transfer_sign") == "negative",
           "coverage": f"{len(keys)}/{len(allg)}"}
    if ranking and len(keys) > 2:
        out["rho"] = spearman({c: scores[c] for c in keys}, gold)
        out["tau"] = kendall_tau_b([scores[c] for c in keys], [gold[c] for c in keys])
        pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
        decisive = [(a, b) for a, b in pairs if abs(gold[a] - gold[b]) > margin]
        agree = [(scores[a] - scores[b]) * (gold[a] - gold[b]) > 0 for a, b in decisive]
        out["pair_acc_decisive"] = sum(agree) / len(agree) if agree else float("nan")
        out["n_decisive"] = len(decisive)
    return out


def logme_scores(paths: list[str]) -> dict:
    """{(estimator, target, budget): {seed: {candidate: (score, result)}}}."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for ln in open(p):
            r = json.loads(ln)
            cand = "scratch" if r["candidate"] == "scratch" else None
            if cand is None and (m := CKPT.match(r["candidate"])):
                cand = f"src:{m.group(1)}"
            if cand is None:
                continue
            n = r["metadata"].get("n_students_requested")
            out[(r["estimator"], r["target"], f"n{n}" if n else "full_split")][r["seed"]][cand] = \
                (r["score"], r)
    return out


def lodo_policies() -> dict:
    """{policy: fn(target, budget) -> objective} for lodo_regime.py's single-feature rules.

    Each rule is refit without the held-out dataset. On a truncation cell (budget trunc_K) the
    held-out dataset is seen at cap K, so its effective pps is min(median, K) / skills, exactly
    lodo_regime's definition; the six training datasets stay at their natural cap of 512.
    Features that do not depend on the cap give the same prediction at every K, as they should.
    """
    rows = lodo_regime.build(512)
    out = {}
    for feat in ("pps_effective", "n_skills", "n_students"):
        def policy(target, budget, feat=feat):
            train = [r for r in rows if r[0] != target]
            if len(train) == len(rows):
                return None
            t, d, _, _ = lodo_regime.fit_threshold([r[1][feat] for r in train],
                                                   [r[2] for r in train])
            cap = int(budget.split("_K")[1]) if budget.startswith("trunc_K") else 512
            held = {r[0]: r[1] for r in lodo_regime.build(cap)}[target][feat]
            return "full" if lodo_regime.predict(held, t, d) == "skill" else "correct_only"
        out[f"LODO {feat}"] = policy
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--executions", required=True)
    ap.add_argument("--scores", nargs="*", default=[])
    ap.add_argument("--margin", type=float, default=0.001)
    ap.add_argument("--boots", type=int, default=20000)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--processed-root", default="../processed",
                    help="where <dataset>/splits.json live, for the largest-source rule")
    a = ap.parse_args()

    cells, probes = load_gold(a.executions)
    stats = {k: annotate(v, a.margin, a.boots) for k, v in cells.items() if len(v) >= 2}
    view_cache: dict = {}
    rows = []

    def record(est, cell, seed, view, scores, extra=None, ranking=True):
        s = stats[cell]
        cands = [c for c in s if view == "practical" or c != "scratch"]
        keep = {c: v for c, v in scores.items() if c in cands}
        if view == "practical" and "scratch" not in keep:
            return
        if not keep or len(cands) < 2:
            return
        vkey = (cell, frozenset(cands))
        if vkey not in view_cache:
            view_cache[vkey] = annotate({c: cells[cell][c] for c in cands}, a.margin, a.boots)
        rows.append({"estimator": est, "track": cell[0], "target": cell[1], "budget": cell[2],
                     "seed": seed, "view": view,
                     **judge(keep, view_cache[vkey], s, a.margin, ranking),
                     "_scores": keep, **(extra or {})})

    lscores = logme_scores(a.scores)
    for (est, tgt, budget), by_seed in lscores.items():
        cell = ("B", tgt, budget)
        if cell not in stats:
            continue
        for seed, cand in by_seed.items():
            sc = {c: v[0] for c, v in cand.items()}
            cost = {"extract_s": st.mean(v[1]["feature_extraction_time"] for v in cand.values()),
                    "score_s": st.mean(v[1]["scoring_time"] for v in cand.values()),
                    "peak_mb": max(v[1]["peak_memory_mb"] or 0 for v in cand.values())}
            for view in ("pretrained", "practical"):
                record(est, cell, seed, view, sc, cost)
    b_sources = {c for k in stats if k[0] == "B" for c in stats[k] if c.startswith("src:")}
    train = train_split_sizes(a.processed_root, b_sources) if b_sources else {}
    for cell in [k for k in stats if k[0] == "B"]:
        sizes = {c: train[c] for c in stats[cell] if c in train}
        for view in ("pretrained", "practical"):
            record("largest source", cell, "-", view, {**sizes, "scratch": 0})
        pr = probes.get(cell[1], {})
        for seed in sorted({s for v in pr.values() for s in v}):
            sc = {c: v[seed] for c, v in pr.items() if seed in v}
            for view in ("pretrained", "practical"):
                record("masked-skill probe", cell, seed, view, sc)
    policies = {f"always {o}": {} for o in ("full", "skill_only", "correct_only")}
    policies.update(lodo_policies())
    for cell in [k for k in stats if k[0] in ("A", "A-r128")]:
        for name, pol in policies.items():
            obj = name.split()[-1] if name.startswith("always") else pol(cell[1], cell[2])
            if obj and obj in stats[cell]:
                pick = {obj: 1.0, **{c: 0.0 for c in stats[cell] if c != obj}}
                record(name, cell, "-", "pretrained", pick, ranking=False)
                record(name, cell, "-", "practical", pick, ranking=False)

    stab = []
    by = defaultdict(list)
    for r in rows:
        if r["seed"] != "-" and "rho" in r:
            key = (r["estimator"], r["track"], r["target"], r["budget"], r["view"])
            by[key].append(r["_scores"])
    for (est, _track, tgt, _budget, view), orders in by.items():
        if len(orders) < 2:
            continue
        common = sorted(set.intersection(*(set(o) for o in orders)))
        taus = [kendall_tau_b([x[c] for c in common], [y[c] for c in common])
                for i, x in enumerate(orders) for y in orders[i + 1:]]
        stab.append({"estimator": est, "target": tgt, "view": view, "seed_tau_mean": st.mean(taus)})

    cols = ["estimator", "track", "target", "budget", "seed", "view", "choice", "top1",
            "equivalent", "regret", "norm_regret", "negative_choice", "coverage", "rho", "tau",
            "pair_acc_decisive", "n_decisive", "extract_s", "score_s", "peak_mb"]
    with open(f"{a.out_prefix}_cells.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    summary = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["estimator"], r["track"], r["view"])].append(r)
    for (est, track, view), rs in sorted(groups.items()):
        def m(k):
            v = [r[k] for r in rs if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))]
            return st.mean(v) if v else ""
        stb = [s["seed_tau_mean"] for s in stab if s["estimator"] == est and s["view"] == view]
        summary.append({"estimator": est, "track": track, "view": view, "rankings": len(rs),
                        "targets": len({(r["target"], r["budget"]) for r in rs}),
                        "rho": m("rho"), "tau": m("tau"),
                        "pair_acc_decisive": m("pair_acc_decisive"),
                        "top1": sum(r["top1"] for r in rs),
                        "equivalent": sum(r["equivalent"] for r in rs),
                        "mean_regret": m("regret"), "max_regret": max(r["regret"] for r in rs),
                        "negative_choices": sum(r["negative_choice"] for r in rs),
                        "seed_stability_tau": st.mean(stb) if stb else "",
                        "extract_s": m("extract_s"), "score_s": m("score_s")})
    with open(f"{a.out_prefix}_summary.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]), delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(summary)

    L = [f"# Estimator evaluation (margin {a.margin}, {a.boots} resamples)\n",
         "| estimator | track | view | rankings | rho | tau | decisive pairs | top-1 | "
         "tied-with-best | mean regret | max regret | negative picks | seed stability |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s['estimator']} | {s['track']} | {s['view']} | {s['rankings']} | "
                 f"{q4(s['rho'])} | {q4(s['tau'])} | {q4(s['pair_acc_decisive'])} | "
                 f"{s['top1']}/{s['rankings']} | {s['equivalent']}/{s['rankings']} | "
                 f"{q4(s['mean_regret'])} | {q4(s['max_regret'])} | {s['negative_choices']} | "
                 f"{q4(s['seed_stability_tau'])} |")
    Path(f"{a.out_prefix}_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
