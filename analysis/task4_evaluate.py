from __future__ import annotations

# MRAP Task 4: judge the estimators' encoder rankings against the published gold of Bassignana et
# al. (2022), frozen and fully fine-tuned, with the Task 3 metrics. Rules fixed on 2026-09-28,
# before any Task 4 score existed (mrap_task4_crossdomain.md, Section 3):
#   tie with the best   gap <= 0.1 points, or inside a two-sided 95% interval built from the
#                       published SDs with 5 seeds each (normal approximation)
#   decisive pair       a pair that is not tied by the same rule
#   regret              best mean minus the chosen model's mean, in points
# The LogME reproduction check compares our LogME ranking with the published LogME per task and
# pooling (Spearman), before any estimator comparison is read.
#
#   PYTHONPATH=. python analysis/task4_evaluate.py \
#       --scores /projects/algl/dai.hany/task4/task4_scores.jsonl \
#       --out /projects/algl/dai.hany/task4/task4_eval

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

try:
    from analysis.build_transfer_benchmark import kendall_tau_b
    from analysis.evaluate_estimators import spearman
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.build_transfer_benchmark import kendall_tau_b
    from analysis.evaluate_estimators import spearman

GOLD = Path(__file__).resolve().parents[1] / "configs" / "task4" / "bassignana2022_table2.tsv"
MARGIN = 0.1
Z = 1.96
N_SEEDS = 5


def load_gold(path: Path = GOLD) -> dict:
    """{(task, pooling): {model: {logme, frozen: (mean, sd), tuned: (mean, sd)}}}"""
    gold: dict = defaultdict(dict)
    with open(path, newline="") as fh:
        rows = csv.DictReader((ln for ln in fh if not ln.startswith("#")), delimiter="\t")
        for r in rows:
            gold[(r["dataset"], r["pooling"])][r["model"]] = {
                "logme": float(r["logme"]),
                "frozen": (float(r["frozen_mean"]), float(r["frozen_sd"])),
                "tuned": (float(r["tuned_mean"]), float(r["tuned_sd"]))}
    return gold


def tied(a: tuple[float, float], b: tuple[float, float]) -> bool:
    gap = abs(a[0] - b[0])
    se = math.sqrt(a[1] ** 2 / N_SEEDS + b[1] ** 2 / N_SEEDS)
    return gap <= MARGIN or gap <= Z * se


def judge(scores: dict, perf: dict) -> dict:
    """scores {model: score}, perf {model: (mean, sd)} over the same models."""
    models = sorted(scores)
    best = max(models, key=lambda m: perf[m][0])
    choice = max(models, key=lambda m: scores[m])
    decisive = [(a, b) for a, b in combinations(models, 2) if not tied(perf[a], perf[b])]
    agree = [(scores[a] - scores[b]) * (perf[a][0] - perf[b][0]) > 0 for a, b in decisive]
    return {"choice": choice, "best": best, "top1": choice == best,
            "tied_with_best": choice == best or tied(perf[choice], perf[best]),
            "regret": perf[best][0] - perf[choice][0],
            "rho": spearman(scores, {m: perf[m][0] for m in models}),
            "tau": kendall_tau_b([scores[m] for m in models], [perf[m][0] for m in models]),
            "pair_acc": sum(agree) / len(agree) if agree else float("nan"),
            "n_decisive": len(decisive)}


def load_scores(path: str) -> dict:
    """{(estimator, task, pooling): {seed: {model: score}}}"""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for ln in Path(path).read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            s = r["score"] if r["score_direction"] == "higher_is_better" else -r["score"]
            out[(r["estimator"], r["task"], r["pooling"])][r["seed"]][r["model"]] = s
    return out


def evaluate(scores: dict, gold: dict) -> tuple[list[dict], list[dict]]:
    rows, repro = [], []
    for (est, task, pooling), by_seed in sorted(scores.items()):
        g = gold.get((task, pooling))
        if g is None:
            continue
        seeds = sorted(by_seed)
        for seed in seeds:
            sc = by_seed[seed]
            if set(sc) != set(g):
                sys.exit(f"ABORT: {est} {task} {pooling} seed {seed} scores {len(sc)} of 7 models")
            for regime in ("frozen", "tuned"):
                rows.append({"estimator": est, "task": task, "pooling": pooling, "seed": seed,
                             "regime": regime, **judge(sc, {m: g[m][regime] for m in g})})
        taus = [kendall_tau_b([by_seed[a][m] for m in g], [by_seed[b][m] for m in g])
                for a, b in combinations(seeds, 2)]
        for r in rows:
            if (r["estimator"], r["task"], r["pooling"]) == (est, task, pooling):
                r["seed_stability"] = st.mean(taus) if taus else float("nan")
        if est == "logme":
            mean_sc = {m: st.mean(by_seed[s][m] for s in seeds) for m in g}
            repro.append({"task": task, "pooling": pooling,
                          "spearman_vs_published": spearman(mean_sc, {m: g[m]["logme"] for m in g}),
                          "seeds": len(seeds)})
    return rows, repro


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["estimator"], r["regime"])].append(r)

    def m(rs, k):
        v = [r[k] for r in rs if not (isinstance(r[k], float) and math.isnan(r[k]))]
        return st.mean(v) if v else float("nan")

    return [{"estimator": e, "regime": g, "rankings": len(rs), "rho": m(rs, "rho"),
             "tau": m(rs, "tau"), "pair_acc": m(rs, "pair_acc"),
             "top1": sum(r["top1"] for r in rs),
             "tied_with_best": sum(r["tied_with_best"] for r in rs),
             "mean_regret": m(rs, "regret"), "max_regret": max(r["regret"] for r in rs),
             "seed_stability": m(rs, "seed_stability")}
            for (e, g), rs in sorted(groups.items())]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--out", required=True, help="prefix for _cells.tsv, _summary.tsv, _report.md")
    a = ap.parse_args(argv)
    rows, repro = evaluate(load_scores(a.scores), load_gold(Path(a.gold)))
    if not rows:
        sys.exit("ABORT: no scores match a published task and pooling")
    summary = summarize(rows)
    for name, data in (("cells", rows), ("summary", summary), ("repro", repro)):
        if data:
            with open(f"{a.out}_{name}.tsv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0]), delimiter="\t",
                                   lineterminator="\n")
                w.writeheader()
                w.writerows(data)
    f = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)  # noqa: E731
    cols = list(summary[0])
    L = ["# Task 4 evaluation against Bassignana et al. (2022)", "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    L += ["| " + " | ".join(f(s[c]) for c in cols) + " |" for s in summary]
    L += ["", "## LogME reproduction (Spearman with the published LogME)", "",
          "| task | pooling | spearman | seeds |", "|---|---|---|---|"]
    L += [f"| {r['task']} | {r['pooling']} | {f(r['spearman_vs_published'])} | {r['seeds']} |"
          for r in repro]
    Path(f"{a.out}_report.md").write_text("\n".join(L) + "\n")
    print(f"{len(rows)} judged rankings; wrote {a.out}_cells.tsv, _summary.tsv, _repro.tsv, "
          "_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
