from __future__ import annotations

# MRAP failure review, diagnostic D6 report: the StudentBERT analogue of the D1 check on NLP.
# Reads the frozen-gold records of scripts/frozen_gold_kt.py (a logistic readout on the estimators'
# own causal features, scored on test learners) and judges, per target and readout:
#   frozen oracle   the ranking by mean frozen test AUC, against the fine-tuned gold, with the
#                   Task 3 annotate() and judge() (pretrained view, 0.001 margin)
#   H-score, LogME  the 7 x 7 train-draw scores (3 seeds), by rank correlation with the frozen gold
#                   and with the fine-tuned gold
# Only complete targets are judged (every candidate on every seed, every readout and penalty);
# incomplete ones are listed and skipped. Every record's H-score on its train features must equal
# the 7 x 7 H-score of the same target, candidate and seed within --align_tol, or the run aborts,
# so the frozen gold is known to use the estimators' exact features. Gold-side only.
#
#   PYTHONPATH=. python analysis/frozen_gold_report.py \
#       --executions benchmark_final/executions.tsv \
#       --frozen /projects/algl/dai.hany/frozen_gold/frozen_gold_kt_*.jsonl \
#       --task2 benchmark_final/tg1_task2_kt_*.jsonl \
#       --logme benchmark_final/tg1_logme_kt_7x7_*.jsonl \
#       --out benchmark_final/frozen_gold

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

try:
    from analysis.evaluate_estimators import annotate, judge, load_gold, spearman
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.evaluate_estimators import annotate, judge, load_gold, spearman

READOUTS = (("frozen_linear", 1e-4), ("frozen_linear", 1e-2), ("frozen_linear_skillbias", 1e-4),
            ("frozen_linear_skillbias", 1e-2))
PRIMARY = READOUTS[0]
SEEDS = (42, 1, 2)


def q(x, places: str = "0.0001", sign: bool = True) -> str:
    d = Decimal(repr(float(x))).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def cand(name: str) -> str:
    return "scratch" if name == "scratch" else "src:" + name.split("_")[1]


def load_scores(paths: list[str], estimator: str) -> dict:
    """{(target, candidate, seed): score} for the default train-draw protocol only."""
    out = {}
    for p in paths:
        for ln in open(p):
            r = json.loads(ln)
            md = r.get("metadata", {})
            if (r["estimator"] != estimator or md.get("score_tag")
                    or md.get("score_split", "train") != "train"):
                continue
            out[(r["target"], cand(r["candidate"]), r["seed"])] = r["score"]
    return out


def load_frozen(paths: list[str]) -> dict:
    """{target: {(readout, lam): {candidate: {seed: frozen test AUC}}}} plus the H-score check."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    hs = {}
    for p in paths:
        for ln in open(p):
            r = json.loads(ln)
            c = cand(r["candidate"])
            out[r["target"]][(r["readout"], r["lam"])][c][r["seed"]] = r["frozen_test_auc"]
            hs[(r["target"], c, r["seed"])] = r["details"]["hscore_on_train_features"]
    return out, hs


def complete(fz: dict, cands: list[str]) -> bool:
    return all(set(fz.get(k, {}).get(c, {})) == set(SEEDS) for k in READOUTS for c in cands)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executions", required=True)
    ap.add_argument("--frozen", nargs="+", required=True)
    ap.add_argument("--task2", nargs="+", required=True)
    ap.add_argument("--logme", nargs="+", required=True)
    ap.add_argument("--margin", type=float, default=0.001)
    ap.add_argument("--boots", type=int, default=20000)
    ap.add_argument("--align_tol", type=float, default=1e-6)
    ap.add_argument("--out", required=True, help="prefix for _cells.tsv and _report.md")
    a = ap.parse_args(argv)
    cells, _ = load_gold(a.executions)
    frozen, hs_rec = load_frozen(a.frozen)
    hs7, lm7 = load_scores(a.task2, "hscore_kt_causal"), load_scores(a.logme, "logme_kt_causal")
    worst = 0.0
    for k, v in hs_rec.items():
        if k not in hs7:
            sys.exit(f"ABORT: no 7 x 7 H-score for {k}")
        worst = max(worst, abs(v - hs7[k]))
    if worst > a.align_tol:
        sys.exit(f"ABORT: frozen-gold features differ from the estimators' "
                 f"(H-score diff {worst:.2e})")
    rows, skipped = [], []
    for t in sorted(frozen):
        cell = ("B", t, "n3000")
        if cell not in cells:
            sys.exit(f"ABORT: no Track B N=3000 gold for {t}")
        cands = sorted(cells[cell])
        if not complete(frozen[t], cands):
            skipped.append(t)
            continue
        srcs = [c for c in cands if c.startswith("src:")]
        full = annotate(cells[cell], a.margin, a.boots)
        view = annotate({c: cells[cell][c] for c in srcs}, a.margin, a.boots)
        tuned = {c: view[c]["gold"] for c in srcs}
        for ro, lam in READOUTS:
            fz = {c: st.mean(frozen[t][(ro, lam)][c][s] for s in SEEDS) for c in srcs}
            j = judge(fz, view, full, a.margin)
            order = sorted(srcs, key=lambda c: (-fz[c], c))
            row = {"target": t, "readout": ro, "lam": lam, "frozen_best": order[0],
                   "own_rank": order.index(f"src:{t}") + 1 if f"src:{t}" in order else "",
                   "tuned_best": max(tuned, key=tuned.get), "oracle_rho": j["rho"],
                   "oracle_top1": j["top1"], "oracle_tied": j["equivalent"],
                   "oracle_regret": j["regret"]}
            for name, sc in (("hscore", hs7), ("logme", lm7)):
                per = [{c: sc[(t, c, s)] for c in srcs} for s in SEEDS]
                row[f"{name}_rho_frozen"] = st.mean(spearman(p, fz) for p in per)
                row[f"{name}_rho_tuned"] = st.mean(spearman(p, tuned) for p in per)
            rows.append(row)
    if not rows:
        sys.exit(f"ABORT: no complete target; incomplete: {skipped}")
    with open(f"{a.out}_cells.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    prim = [r for r in rows if (r["readout"], r["lam"]) == PRIMARY]
    L = ["# Frozen gold on StudentBERT (failure review D6)", "",
         f"Complete targets: {len(prim)}; incomplete and skipped: {', '.join(skipped) or 'none'}. "
         f"H-score on the frozen-gold features matches the 7 x 7 H-score within {worst:.1e}.", "",
         "## Main readout (frozen_linear, lam 1e-4)", "",
         "| target | frozen best | own-encoder rank | tuned best | oracle rho | oracle top-1 | "
         "oracle regret | H-score rho, frozen / tuned | LogME rho, frozen / tuned |",
         "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['target']} | {r['frozen_best']} | {r['own_rank']} | {r['tuned_best']} | "
          f"{q(r['oracle_rho'])} | {r['oracle_top1']} | {q(r['oracle_regret'], sign=False)} | "
          f"{q(r['hscore_rho_frozen'])} / {q(r['hscore_rho_tuned'])} | "
          f"{q(r['logme_rho_frozen'])} / {q(r['logme_rho_tuned'])} |" for r in prim]
    if prim:
        L += [f"| mean over {len(prim)} | | | | {q(st.mean(r['oracle_rho'] for r in prim))} | "
              f"{sum(r['oracle_top1'] for r in prim)}/{len(prim)} | "
              f"{q(st.mean(r['oracle_regret'] for r in prim), sign=False)} | "
              f"{q(st.mean(r['hscore_rho_frozen'] for r in prim))} / "
              f"{q(st.mean(r['hscore_rho_tuned'] for r in prim))} | "
              f"{q(st.mean(r['logme_rho_frozen'] for r in prim))} / "
              f"{q(st.mean(r['logme_rho_tuned'] for r in prim))} |"]
    L += ["", "## Readout sensitivity (frozen best, own-encoder rank, oracle rho and regret)", "",
          "| target | readout | lam | frozen best | own rank | oracle rho | oracle regret |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {r['target']} | {r['readout']} | {r['lam']:g} | {r['frozen_best']} | "
          f"{r['own_rank']} | {q(r['oracle_rho'])} | {q(r['oracle_regret'], sign=False)} |"
          for r in rows]
    Path(f"{a.out}_report.md").write_text("\n".join(L) + "\n")
    print(f"{len(prim)} complete targets, skipped {skipped}; H-score alignment {worst:.1e}; wrote "
          f"{a.out}_cells.tsv and {a.out}_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
