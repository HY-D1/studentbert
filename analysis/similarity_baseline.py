from __future__ import annotations

# MRAP Task 3: the simple dataset-similarity baseline (project constraints, Section S), chosen on
# 2026-09-28 in place of OTCE. For each Track B target at N=3000 it ranks the FOREIGN sources by
# how close their dataset profile is to the target's, and judges the ranking with
# evaluate_estimators.py's own gold, annotate() and judge(). The target's own encoder is left out
# of this view because a profile distance puts it at zero by construction; so that the comparison
# is like for like, every scored estimator and the largest-source rule are judged in the same
# foreign-only view. top-1, ties and regret therefore refer to the best FOREIGN source.
#
# Profiles are dataset statistics from analysis/lodo_regime.py; no result enters them.
#   similarity_inputs   log learners, log median length, effective practice-per-skill (regime U)
#   similarity_labels   the same plus the correct rate, a statistic of the target's labels
# Each feature is standardized over the seven datasets (population SD); distance is Euclidean.
# Learner counts are whole-dataset counts; they order the sources exactly as the train splits do.
#
#   PYTHONPATH=. python analysis/similarity_baseline.py \
#       --executions benchmark_final/executions.tsv --scores tg1_logme_kt_7x7_*.jsonl ... \
#       --out benchmark_final/similarity

import argparse
import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

try:
    from analysis import lodo_regime
    from analysis.evaluate_estimators import annotate, judge, load_gold, logme_scores
except ModuleNotFoundError:  # run without PYTHONPATH=.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis import lodo_regime
    from analysis.evaluate_estimators import annotate, judge, load_gold, logme_scores

VARIANTS = {
    "similarity_inputs": ("log_students", "log_median_len", "pps_effective"),
    "similarity_labels": ("log_students", "log_median_len", "pps_effective", "correct_rate"),
}


def profiles(cap: int = 512) -> dict:
    out = {}
    for name, f, _regime in lodo_regime.build(cap):
        out[name] = {"students": f["n_students"],
                     "log_students": math.log10(f["n_students"]),
                     "log_median_len": math.log10(f["median_len"]),
                     "pps_effective": f["pps_effective"], "correct_rate": f["correct_rate"]}
    return out


def standardize(prof: dict, feats: tuple) -> dict:
    z: dict = defaultdict(dict)
    for k in feats:
        vals = [p[k] for p in prof.values()]
        mu, sd = st.mean(vals), st.pstdev(vals)
        for name, p in prof.items():
            z[name][k] = (p[k] - mu) / sd if sd else 0.0
    return z


def similarity_scores(target: str, sources: list[str], feats: tuple, prof: dict) -> dict:
    """Higher is closer: minus the Euclidean distance in standardized profile space."""
    z = standardize(prof, feats)
    return {f"src:{s}": -math.sqrt(sum((z[target][k] - z[s][k]) ** 2 for k in feats))
            for s in sources}


def evaluate(cells: dict, lscores: dict, prof: dict, margin: float, boots: int) -> list[dict]:
    rows = []
    b = {k: v for k, v in cells.items() if k[0] == "B" and k[2] == "n3000" and len(v) >= 2}
    if not b:
        sys.exit("ABORT: no Track B N=3000 cells in the executions table")
    for cell in sorted(b):
        _track, tgt, budget = cell
        foreign = sorted(c for c in b[cell] if c.startswith("src:") and c != f"src:{tgt}")
        if len(foreign) < 2:
            continue
        missing = [c for c in foreign + [f"src:{tgt}"] if c[4:] not in prof]
        if missing:
            sys.exit(f"ABORT: no dataset profile for {missing}")
        full = annotate(b[cell], margin, boots)
        view = annotate({c: b[cell][c] for c in foreign}, margin, boots)
        best = max(view, key=lambda c: view[c]["gold"])
        srcs = [c[4:] for c in foreign]
        base = {"target": tgt, "budget": budget, "best_foreign": best}
        for name, feats in VARIANTS.items():
            rows.append({"estimator": name, "seed": "-", **base,
                         **judge(similarity_scores(tgt, srcs, feats, prof), view, full, margin)})
        rows.append({"estimator": "largest source", "seed": "-", **base,
                     **judge({f"src:{s}": prof[s]["students"] for s in srcs}, view, full, margin)})
        for (est, t2, b2), by_seed in sorted(lscores.items()):
            if (t2, b2) != (tgt, budget):
                continue
            for seed in sorted(by_seed):
                sc = {c: v[0] for c, v in by_seed[seed].items() if c in foreign}
                if len(sc) >= 2:
                    rows.append({"estimator": est, "seed": seed, **base,
                                 **judge(sc, view, full, margin)})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["estimator"]].append(r)

    def m(rs, k):
        v = [r[k] for r in rs if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))]
        return st.mean(v) if v else ""

    return [{"estimator": est, "rankings": len(rs), "targets": len({r["target"] for r in rs}),
             "rho": m(rs, "rho"), "tau": m(rs, "tau"),
             "pair_acc_decisive": m(rs, "pair_acc_decisive"),
             "top1": sum(r["top1"] for r in rs), "equivalent": sum(r["equivalent"] for r in rs),
             "mean_regret": m(rs, "regret"), "max_regret": max(r["regret"] for r in rs),
             "negative_choices": sum(r["negative_choice"] for r in rs)}
            for est, rs in sorted(groups.items(), key=lambda kv: _rho_key(m(kv[1], "rho")))]


def _rho_key(x) -> float:
    return -x if isinstance(x, float) else 9.0


def fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def write(prefix: str, rows: list[dict], summary: list[dict]) -> None:
    cols = ["estimator", "target", "budget", "seed", "best_foreign", "choice", "top1",
            "equivalent", "regret", "norm_regret", "negative_choice", "coverage", "rho", "tau",
            "pair_acc_decisive", "n_decisive"]
    with open(f"{prefix}_cells.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        w.writerows([[r.get(c, "") for c in cols] for r in rows])
    scols = list(summary[0])
    with open(f"{prefix}_summary.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(scols)
        w.writerows([[s[c] for c in scols] for s in summary])
    L = ["# Dataset-similarity baseline, foreign-source view (Track B, KT, N=3000)", "",
         "Every row is judged against the best FOREIGN source; the target's own encoder is out.",
         "",
         "| " + " | ".join(scols) + " |", "|" + "---|" * len(scols)]
    L += ["| " + " | ".join(fmt(s[c]) for c in scols) + " |" for s in summary]
    L += ["", "## Similarity picks per target", "",
          "| target | best foreign source | similarity_inputs | similarity_labels |",
          "|---|---|---|---|"]
    picks = defaultdict(dict)
    for r in rows:
        if r["estimator"] in VARIANTS:
            picks[r["target"]][r["estimator"]] = (r["choice"], r["best_foreign"], r["equivalent"])
    for tgt, p in sorted(picks.items()):
        cell = [f"{p[v][0]}{'' if p[v][2] else ' (not tied)'}" for v in VARIANTS]
        L.append(f"| {tgt} | {p['similarity_inputs'][1]} | {cell[0]} | {cell[1]} |")
    Path(f"{prefix}_report.md").write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executions", required=True)
    ap.add_argument("--scores", nargs="*", default=[])
    ap.add_argument("--margin", type=float, default=0.001)
    ap.add_argument("--boots", type=int, default=20000)
    ap.add_argument("--out", required=True,
                    help="prefix for _cells.tsv, _summary.tsv and _report.md")
    a = ap.parse_args(argv)
    cells, _ = load_gold(a.executions)
    rows = evaluate(cells, logme_scores(a.scores) if a.scores else {}, profiles(), a.margin,
                    a.boots)
    summary = summarize(rows)
    write(a.out, rows, summary)
    print(f"{len(rows)} rankings over {len({r['target'] for r in rows})} targets; "
          f"wrote {a.out}_cells.tsv, {a.out}_summary.tsv, {a.out}_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
