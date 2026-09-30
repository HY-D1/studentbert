from __future__ import annotations

# MRAP failure review, diagnostic D1 (and the simple-baseline part of D2): how well can any
# frozen-feature estimator do against the fully fine-tuned gold of Bassignana et al. (2022)?
# Everything is computed from configs/task4/bassignana2022_table2.tsv with the Task 4 judge(), so
# ties, decisive pairs and regret follow the rules fixed in analysis/task4_evaluate.py.
#   frozen_oracle      ranks the 7 models by their published frozen means, i.e. an estimator that
#                      knows frozen performance exactly; its score against tuned gold is the ceiling
#                      for any estimator whose target is frozen performance
#   published_logme    the paper's own LogME values, for comparison with our reproduction
#   loto_best_average  mean tuned rank over the other published tasks, same pooling (Section S,
#                      "globally best development strategy"); rule set before it was computed
# Each setting (task, pooling) is one deterministic ranking. Means are rounded once, in Decimal,
# half away from zero, only for display.
#
#   PYTHONPATH=. python analysis/task4_ceiling.py --out /projects/algl/dai.hany/task4/task4_ceiling
# --also_run airline adds a section for the tasks run so far plus Airline (2026-09-30); without
# it the report is byte-identical to the 2026-09-29 one, so review 7.5's targets stay checkable.

import argparse
import csv
import statistics as st
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

try:
    from analysis.task4_evaluate import GOLD, judge, load_gold
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.task4_evaluate import GOLD, judge, load_gold

RUN_TASKS = ("agnews", "mnli", "qnli", "rte")
ESTIMATORS = ("frozen_oracle", "published_logme", "loto_best_average")


def q(x: float, places: str = "0.0001") -> str:
    return str(Decimal(repr(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP))


def loto_scores(gold: dict, task: str, pooling: str) -> dict:
    """Higher is better: minus the mean tuned rank (1 = best) over the other tasks."""
    models = sorted(gold[(task, pooling)])
    others = sorted(t for (t, p) in gold if p == pooling and t != task)
    if not others:
        sys.exit(f"ABORT: no other task for {task} {pooling}")
    ranks: dict = {m: [] for m in models}
    for t in others:
        g = gold[(t, pooling)]
        if set(g) != set(models):
            sys.exit(f"ABORT: {t} {pooling} has a different model set")
        for i, m in enumerate(sorted(models, key=lambda m: (-g[m]["tuned"][0], m))):
            ranks[m].append(i + 1)
    return {m: -st.mean(v) for m, v in ranks.items()}


def cells(gold: dict) -> list[dict]:
    out = []
    for (task, pooling), g in sorted(gold.items()):
        scores = {"frozen_oracle": {m: g[m]["frozen"][0] for m in g},
                  "published_logme": {m: g[m]["logme"] for m in g},
                  "loto_best_average": loto_scores(gold, task, pooling)}
        for regime in ("frozen", "tuned"):
            perf = {m: g[m][regime] for m in g}
            for est in ESTIMATORS:
                if est == "frozen_oracle" and regime == "frozen":
                    continue
                out.append({"estimator": est, "task": task, "pooling": pooling, "regime": regime,
                            **judge(scores[est], perf)})
    return out


def summarize(rows: list[dict], tasks: tuple[str, ...] | None) -> list[dict]:
    out = []
    for regime in ("frozen", "tuned"):
        for est in ESTIMATORS:
            rs = [r for r in rows if r["estimator"] == est and r["regime"] == regime
                  and (tasks is None or r["task"] in tasks)]
            if not rs:
                continue
            pa = [r["pair_acc"] for r in rs if r["pair_acc"] == r["pair_acc"]]
            out.append({"estimator": est, "regime": regime, "rankings": len(rs),
                        "rho": q(st.mean(r["rho"] for r in rs)),
                        "tau": q(st.mean(r["tau"] for r in rs)),
                        "pair_acc": q(st.mean(pa)) if pa else "nan",
                        "top1": f"{sum(r['top1'] for r in rs)}/{len(rs)}",
                        "tied_with_best": f"{sum(r['tied_with_best'] for r in rs)}/{len(rs)}",
                        "mean_regret": q(st.mean(r["regret"] for r in rs), "0.01"),
                        "max_regret": q(max(r["regret"] for r in rs), "0.01")})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--out", required=True, help="prefix for _cells.tsv and _report.md")
    ap.add_argument("--also_run", nargs="*", default=[], help="later tasks, extra section")
    a = ap.parse_args(argv)
    gold = load_gold(Path(a.gold))
    extra = tuple(a.also_run)
    if set(extra) - {t for t, _p in gold} or set(extra) & set(RUN_TASKS):
        sys.exit(f"ABORT: --also_run {list(extra)} must name published tasks not in {RUN_TASKS}")
    rows = cells(gold)
    with open(f"{a.out}_cells.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    L = ["# Task 4 ceiling check against the published gold (Bassignana et al. 2022)", ""]
    sections = [("Tasks run in Task 4 (AGNews, MNLI, QNLI, RTE)", RUN_TASKS)]
    if extra:
        sections.append((f"Tasks run in Task 4 plus {', '.join(extra)}", RUN_TASKS + extra))
    sections.append(("All six published tasks", None))
    for title, tasks in sections:
        s = summarize(rows, tasks)
        cols = list(s[0])
        L += [f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        L += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in s] + [""]
    L += ["## Frozen oracle against tuned gold, per setting", "",
          "| task | pooling | rho | choice | best | regret | tied |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {r['task']} | {r['pooling']} | {q(r['rho'])} | {r['choice']} | {r['best']} | "
          f"{q(r['regret'], '0.01')} | {r['tied_with_best']} |"
          for r in rows if r["estimator"] == "frozen_oracle"]
    Path(f"{a.out}_report.md").write_text("\n".join(L) + "\n")
    for r in summarize(rows, RUN_TASKS):
        print(" ".join(f"{k}={v}" for k, v in r.items()))
    print(f"wrote {a.out}_cells.tsv and {a.out}_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
