from __future__ import annotations

"""Option 1 evaluation on the NLP benchmark (docs/option1_nlp_design.md, approved by Prof. Hazra).

Committed before it first reads the full set of early fine-tunes. It collects the 210 runs of
scripts/task4_early_ft.py, checks them (all present, 3 epochs each, V100-SXM2, and every encoder of
a task and seed trained and validated on exactly the same rows), and compares, on the 30 H-score
recommendations of step A (analysis/nlp_head_pairs.py, published tuned gold, Task 4 tie rule):

  always H-score          follow every recommendation
  fixed rule (epoch k)    abstain when H-score's pick is not the validation leader among the 7
                          encoders at epoch k; k = 3 is the rule fixed on 2026-10-13, k = 1 and 2
                          are the minimum-budget analysis; nothing is fitted
  learned head-pair       version 2's model (analysis/reversal_model_v2.py): L2 logistic regression
                          on the head pairs, noisy-OR recommendation risk, penalty and threshold
                          chosen inside each fold from the training tasks only (fixed 0.5 beside
                          it); features frozen + early fine-tuning up to epoch k + history
  best-average rule       the Task 4 leave-one-task-out recommender (analysis/task4_ceiling.py);
                          it never abstains, so it is reported by its regret
  frozen margin           H-score's margin to its runner-up, as an abstention order (AURC only)

Leave-one-task-out: both poolings of a task are held out together, because they share data. Regret
is in published tuned accuracy points. Compute cost is the measured GPU time of the early
fine-tunes a recommendation needs (7 encoders, epochs 1 to k). Primary comparisons (fixed in the
design): the epoch-3 rule against always following H-score on selective regret and coverage, and
the learned model against the rule. With 5 tasks and 30 recommendations every estimate is coarse.

  PYTHONPATH=. python analysis/nlp_option1_eval.py \
      --early_ft /projects/algl/dai.hany/task4/early_ft \
      --pairs benchmark_final/nlp_head_pairs.tsv --recs benchmark_final/nlp_head_recs.tsv \
      --out benchmark_final/nlp_option1
"""

import argparse
import csv
import hashlib
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from analysis import reversal_model_v2 as v2
from analysis import task4_evaluate as te
from analysis.task4_ceiling import loto_scores

TASKS = ("agnews", "mnli", "qnli", "rte", "airline")
POOLINGS = ("cls", "mean")
SEEDS = (42, 1, 2)
EPOCHS = (1, 2, 3)
PINNED = {"nlp_head_pairs.tsv": "d023dd9346eb1020afa11d1dacbaf579",
          "nlp_head_recs.tsv": "e533ea40073f9ff4926af3194570008c"}
BEST_AVERAGE_REGRET = "1.37"  # RESULTS.md 13.1, 10 settings
HSCORE_REGRET = "2.70"  # RESULTS.md 13.1, H-score against tuned gold, 30 rankings


def slug(m: str) -> str:
    return m.replace("/", "__")


def pinned_tsv(path: str) -> list[dict]:
    p = Path(path)
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != PINNED[p.name]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[p.name]}")
    return list(csv.DictReader(p.read_text().splitlines(), delimiter="\t"))


def load_early(root: str, models: list[str]) -> dict:
    """runs[(task, pooling, seed, model)] = record; aborts on any gap or inconsistency."""
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
                        fails.append(f"{f}: epochs {[e['epoch'] for e in r['epochs']]}")
                    if "V100-SXM2" not in r.get("gpu", ""):
                        fails.append(f"{f}: gpu {r.get('gpu')}")
                    runs[(t, p, s, m)] = r
    for t in TASKS:
        for s in SEEDS:
            rs = [runs[(t, p, s, m)] for p in POOLINGS for m in models if (t, p, s, m) in runs]
            for k in ("train_idx_md5", "val_idx_md5", "n_train", "n_val"):
                if len({r[k] for r in rs}) > 1:
                    fails.append(f"{t} seed {s}: encoders differ in {k}")
    if fails:
        sys.exit("ABORT, early fine-tunes incomplete or inconsistent:\n  "
                 + "\n  ".join(fails[:20]))
    return runs


def acc(runs: dict, t: str, p: str, s: int, m: str, k: int) -> float:
    return runs[(t, p, s, m)]["epochs"][k - 1]["val_acc"]


def ranks(score: dict) -> dict:
    order = sorted(score, key=lambda m: (-score[m], m))
    return {m: i + 1 for i, m in enumerate(order)}


def build_rows(pairs: list[dict], recs: list[dict], runs: dict, gold: dict, k: int) -> list[dict]:
    """Head-pair rows in the schema of analysis/reversal_model_v2.py, features up to epoch k."""
    reg = {(r["task"], r["pooling"], r["seed"]): r["regret_points"] for r in recs}
    by = defaultdict(list)
    for r in pairs:
        by[(r["task"], r["pooling"], r["seed"])].append(r)
    rows = []
    for (t, p, s), rs in sorted(by.items()):
        pick = rs[0]["pick"]
        models = [pick] + [r["alternative"] for r in rs]
        hs = {pick: float(rs[0]["hscore_pick"]), **{r["alternative"]: float(r["hscore_alt"])
                                                     for r in rs}}
        sd = st.pstdev(hs.values()) or 1.0
        hrank = ranks(hs)
        hist = {m: -v for m, v in loto_scores(gold, t, p).items()}
        e = {j: {m: acc(runs, t, p, int(s), m, j) for m in models} for j in EPOCHS}
        er = {j: ranks(e[j]) for j in EPOCHS}
        lead = max(e[k].values())
        for r in rs:
            a, b = pick, r["alternative"]
            row = {"target": t, "est_seed": f"{p}:{s}", "cand_a": a, "cand_b": b,
                   "frozen_rank_a": "1", "frozen_rank_b": str(hrank[b]),
                   "label_main": r["label"], "label_robust": r["label"],
                   "target_regret_practical": reg[(t, p, s)],
                   "margin_hscore_z": repr((hs[a] - hs[b]) / sd),
                   "logme_agrees": str(int(float(r["logme_pick"]) > float(r["logme_alt"]))),
                   "hist_rank_a": repr(hist[a]), "hist_rank_b": repr(hist[b]),
                   "hist_rank_diff": repr(hist[a] - hist[b]),
                   "pick_rank_at_k": str(er[k][a])}
            for j in EPOCHS[:k]:
                row[f"val_e{j}_a"], row[f"val_e{j}_b"] = repr(e[j][a]), repr(e[j][b])
                row[f"val_diff_e{j}"] = repr(e[j][a] - e[j][b])
            row.update({f"rank_e{k}_a": str(er[k][a]), f"rank_e{k}_b": str(er[k][b]),
                        f"gap_leader_e{k}_a": repr(lead - e[k][a]),
                        f"gap_leader_e{k}_b": repr(lead - e[k][b])})
            if k >= 2:
                row.update({"slope_a": repr(e[k][a] - e[1][a]), "slope_b": repr(e[k][b] - e[1][b]),
                            "slope_diff": repr((e[k][a] - e[1][a]) - (e[k][b] - e[1][b])),
                            "rank_gain_a": str(er[1][a] - er[k][a]),
                            "rank_gain_b": str(er[1][b] - er[k][b]),
                            "b_closing": repr((e[k][a] - e[k][b]) - (e[1][a] - e[1][b]))})
            rows.append(row)
    return rows


def feature_cols(k: int) -> list[str]:
    cols = ["margin_hscore_z", "logme_agrees", "frozen_rank_b", "hist_rank_a", "hist_rank_b",
            "hist_rank_diff"]
    for j in EPOCHS[:k]:
        cols += [f"val_e{j}_a", f"val_e{j}_b", f"val_diff_e{j}"]
    cols += [f"rank_e{k}_a", f"rank_e{k}_b", f"gap_leader_e{k}_a", f"gap_leader_e{k}_b"]
    if k >= 2:
        cols += ["slope_a", "slope_b", "slope_diff", "rank_gain_a", "rank_gain_b", "b_closing"]
    return cols


def rule_recs(rows: list[dict]) -> list[dict]:
    risk = {v2.rm.key(r): 0.0 if r["pick_rank_at_k"] == "1" else 1.0 for r in rows}
    return v2.rec_rows(rows, risk, "label_main")


def cost_minutes(runs: dict, models: list[str], k: int) -> float:
    """Mean GPU-minutes of the early fine-tunes a recommendation needs (7 encoders, epochs 1..k)."""
    per = [sum(sum(e["epoch_s"] for e in runs[(t, p, s, m)]["epochs"][:k]) for m in models) / 60
           for t in TASKS for p in POOLINGS for s in SEEDS]
    return st.mean(per)


def f4(x) -> str:
    return "nan" if x != x else f"{x:.4f}"


def table_rows(name: str, e: dict, rule: str, n: int) -> str:
    s = e[rule]
    return (f"| {name} | {s['followed']}/{n} | {s['caught']}/{e['reversed']} | {s['false_alarms']} "
            f"| {f4(s['selective_regret'])} |")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--early_ft", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--recs", required=True)
    ap.add_argument("--out", required=True, help="prefix for the _recs and _report files")
    a = ap.parse_args(argv)
    pairs, recs = pinned_tsv(a.pairs), pinned_tsv(a.recs)
    gold = te.load_gold()
    models = sorted(gold[("rte", "cls")])
    runs = load_early(a.early_ft, models)
    gpu_h = sum(r["total_s"] for r in runs.values()) / 3600

    ba = []
    for t in TASKS:
        for p in POOLINGS:
            perf = {m: gold[(t, p)][m]["tuned"] for m in gold[(t, p)]}
            ba.append(te.judge(loto_scores(gold, t, p), perf)["regret"])
    ba_mean = st.mean(ba)
    hs_mean = st.mean(float(r["regret_points"]) for r in recs)
    if f"{hs_mean:.2f}" != HSCORE_REGRET:
        sys.exit(f"ABORT: H-score regret {hs_mean:.2f} does not reproduce RESULTS.md 13.1")
    if f"{ba_mean:.2f}" != BEST_AVERAGE_REGRET:
        sys.exit(f"ABORT: best-average regret {ba_mean:.2f} does not reproduce RESULTS.md 13.1")

    L = ["# Option 1: the recommendation-level test on the NLP benchmark", "",
         f"Early fine-tunes: {len(runs)} runs, all present, 3 epochs, V100-SXM2, identical "
         "training "
         f"and validation rows across encoders; {gpu_h:.1f} GPU-hours in total. Recommendations: "
         f"30 (5 tasks x 2 poolings x 3 seeds), {sum(int(r['reversed']) for r in recs)} reversed.",
         ""]
    out_recs = []
    summary = {}
    for k in EPOCHS:
        rows = build_rows(pairs, recs, runs, gold, k)
        rr = rule_recs(rows)
        er = v2.evaluate(rr, "risk_noisy_or", None)
        pred, fold = v2.loto(rows, feature_cols(k), "label_main", head_only=True)
        lr = v2.rec_rows(rows, pred, "label_main")
        el = v2.evaluate(lr, "risk_noisy_or", fold)
        summary[k] = {"rule": er, "learned": el, "rule_recs": rr, "learned_recs": lr,
                      "fold": fold, "cost": cost_minutes(runs, models, k)}
        for r in rr:
            out_recs.append({"epochs": k, "method": "fixed rule", "tau_fold": 0.5, **r})
        for r in lr:
            out_recs.append({"epochs": k, "method": "learned head-pair",
                             "tau_fold": fold[r["target"]]["tau_noisy_or"], **r})
    n = summary[3]["rule"]["n"]
    always = st.mean(r["regret"] for r in summary[3]["rule_recs"])
    marg = v2.rm.aurc(summary[3]["rule_recs"], lambda r: -r["margin_to_runner_up_z"])
    s3 = summary[3]
    L += ["## Primary comparison (epoch 3)", "",
          "| method | followed | caught | false alarms | selective regret (points) |",
          "|---|---|---|---|---|",
          f"| always follow H-score | {n}/{n} | 0/{s3['rule']['reversed']} | 0 | {f4(always)} |",
          table_rows("fixed epoch-3 rule", s3["rule"], "fixed 0.5", n),
          table_rows("learned head-pair, fold-chosen threshold", s3["learned"], "fold-chosen", n),
          table_rows("learned head-pair, threshold 0.5", s3["learned"], "fixed 0.5", n), "",
          f"Best-average rule (never abstains): mean regret {ba_mean:.4f} points over the 10 "
          f"settings (reproduces RESULTS.md 13.1). AURC: learned {f4(s3['learned']['aurc'])}, "
          f"frozen margin {f4(marg)}, random {f4(always)}. Learned model Brier "
          f"{f4(s3['learned']['brier'])}.", "",
          "| learned risk (epoch 3) | recommendations | mean predicted | observed reversal rate |",
          "|---|---|---|---|"]
    for lo, hi, c, mp, ob in s3["learned"]["bins"]:
        L.append(f"| {lo:.2f} to {hi:.2f} | {c} | {f4(mp)} | {f4(ob)} |")
    L += ["", "## Epochs 1 to 3: the minimum useful budget", "",
          "| epochs | GPU-minutes per recommendation | rule followed | rule caught | rule false "
          "alarms | rule selective regret | learned followed | learned caught | learned false "
          "alarms | learned selective regret | learned AURC |", "|---|---|---|---|---|---|---|---|"
          "---|---|---|"]
    for k in EPOCHS:
        r_, l_ = summary[k]["rule"]["fixed 0.5"], summary[k]["learned"]["fold-chosen"]
        L.append(f"| {k} | {summary[k]['cost']:.1f} | {r_['followed']}/{n} | "
                 f"{r_['caught']}/{summary[k]['rule']['reversed']} | {r_['false_alarms']} | "
                 f"{f4(r_['selective_regret'])} | {l_['followed']}/{n} | "
                 f"{l_['caught']}/{summary[k]['learned']['reversed']} | {l_['false_alarms']} | "
                 f"{f4(l_['selective_regret'])} | {f4(summary[k]['learned']['aurc'])} |")
    L += ["", "## Per held-out task (epoch 3)", "",
          "| task | recommendations | reversed | rule abstains | rule caught | learned abstains | "
          "learned caught |", "|---|---|---|---|---|---|---|"]
    for t in TASKS:
        rr = [r for r in s3["rule_recs"] if r["target"] == t]
        lr = [r for r in s3["learned_recs"] if r["target"] == t]
        tau = s3["fold"][t]["tau_noisy_or"]
        rab = [r for r in rr if r["risk_noisy_or"] > 0.5]
        lab = [r for r in lr if r["risk_noisy_or"] > tau]
        L.append(f"| {t} | {len(rr)} | {sum(r['reversed'] for r in rr)} | {len(rab)} | "
                 f"{sum(r['reversed'] for r in rab)} | {len(lab)} | "
                 f"{sum(r['reversed'] for r in lab)} |")
    no_air = [r for r in s3["rule_recs"] if r["target"] != "airline"]
    ea = v2.evaluate(no_air, "risk_noisy_or", None)["fixed 0.5"]
    L += ["", f"Without Airline (24 recommendations): the epoch-3 rule follows {ea['followed']}, "
          f"catches {ea['caught']} of {sum(r['reversed'] for r in no_air)} with "
          f"{ea['false_alarms']} false alarms, selective regret {f4(ea['selective_regret'])}, "
          f"against {f4(st.mean(r['regret'] for r in no_air))} for always following H-score.", "",
          "With 5 tasks and 30 recommendations every estimate here is coarse."]
    out = Path(a.out)
    p = out.with_name(out.name + "_recs.tsv")
    m = v2.write_tsv(p, out_recs)
    L += ["", f"Files: {p.name} {m}"]
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    r3 = s3["rule"]["fixed 0.5"]
    print(f"epoch-3 rule: followed {r3['followed']}/{n}, caught {r3['caught']}, false alarms "
          f"{r3['false_alarms']}, selective regret {f4(r3['selective_regret'])} (always "
          f"{f4(always)}); wrote {p.name} and {rp.name} md5 "
          f"{hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
