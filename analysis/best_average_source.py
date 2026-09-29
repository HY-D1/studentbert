from __future__ import annotations

# MRAP failure review, diagnostic D9: the "globally best development strategy" baseline of the
# project constraints (Section S) for Track B source selection, as a leave-one-target-out rule.
# For each held-out target, every candidate source is scored by minus its mean rank as a FOREIGN
# source on the other N=3000 targets (rank 1 = best of the sources there, by gold mean over their
# common seeds). A source's rank on its own dataset is never used, so the in-domain candidate is
# judged by the same kind of evidence as the others, and the held-out target's results never
# enter its own scores (tests). Judged with the Task 3 annotate() and judge() in the pretrained
# view and in the foreign-only view of similarity_baseline.py. The largest-source rule is judged
# alongside as a harness check: it must reproduce the recorded Task 3 rows (pretrained rho
# +0.4031, foreign-only +0.3959). Display values are rounded once, in Decimal, half away from zero.
#
#   PYTHONPATH=. python analysis/best_average_source.py \
#       --executions benchmark_final/executions.tsv --out benchmark_final/best_average

import argparse
import csv
import statistics as st
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

try:
    from analysis.evaluate_estimators import annotate, judge, load_gold
    from analysis.similarity_baseline import profiles
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.evaluate_estimators import annotate, judge, load_gold
    from analysis.similarity_baseline import profiles

VIEWS = ("pretrained", "foreign")


def q(x: float, places: str = "0.0001") -> str:
    return str(Decimal(repr(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP))


def gold_means(vals: dict, cands: list[str]) -> dict:
    common = sorted(set.intersection(*(set(vals[c]) for c in cands)))
    if not common:
        sys.exit(f"ABORT: no common seeds among {cands}")
    return {c: st.mean(vals[c][s] for s in common) for c in cands}


def best_average_scores(b: dict, tgt: str, cands: list[str]) -> tuple[dict, dict]:
    """Higher is better: minus the mean foreign rank on the other targets; also the rank counts."""
    ranks: dict = {c: [] for c in cands}
    for (_track, t2, _budget), vals in sorted(b.items()):
        if t2 == tgt:
            continue
        srcs = sorted(c for c in vals if c.startswith("src:"))
        means = gold_means(vals, srcs)
        order = sorted(srcs, key=lambda c: (-means[c], c))
        pos = {c: i + 1 for i, c in enumerate(order)}
        for c in cands:
            if c in pos and c != f"src:{t2}":
                ranks[c].append(pos[c])
    empty = [c for c, v in ranks.items() if not v]
    if empty:
        sys.exit(f"ABORT: no foreign rank outside {tgt} for {empty}")
    return {c: -st.mean(v) for c, v in ranks.items()}, {c: len(v) for c, v in ranks.items()}


def evaluate(cells: dict, prof: dict, margin: float, boots: int) -> list[dict]:
    b = {k: v for k, v in cells.items() if k[0] == "B" and k[2] == "n3000" and len(v) >= 2}
    if not b:
        sys.exit("ABORT: no Track B N=3000 cells in the executions table")
    rows = []
    for cell in sorted(b):
        tgt = cell[1]
        full = annotate(b[cell], margin, boots)
        for view in VIEWS:
            cands = sorted(c for c in b[cell] if c.startswith("src:")
                           and (view == "pretrained" or c != f"src:{tgt}"))
            vstats = annotate({c: b[cell][c] for c in cands}, margin, boots)
            best = max(vstats, key=lambda c: vstats[c]["gold"])
            sc, _n = best_average_scores(b, tgt, cands)
            sizes = {c: prof[c[4:]]["students"] for c in cands}
            for est, scores in (("best average", sc), ("largest source", sizes)):
                rows.append({"estimator": est, "view": view, "target": tgt, "best": best,
                             **judge(scores, vstats, full, margin)})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for view in VIEWS:
        for est in ("best average", "largest source"):
            rs = [r for r in rows if r["estimator"] == est and r["view"] == view]
            out.append({"estimator": est, "view": view, "rankings": len(rs),
                        "rho": q(st.mean(r["rho"] for r in rs)),
                        "tau": q(st.mean(r["tau"] for r in rs)),
                        "pair_acc_decisive": q(st.mean(r["pair_acc_decisive"] for r in rs)),
                        "top1": f"{sum(r['top1'] for r in rs)}/{len(rs)}",
                        "equivalent": f"{sum(r['equivalent'] for r in rs)}/{len(rs)}",
                        "mean_regret": q(st.mean(r["regret"] for r in rs)),
                        "max_regret": q(max(r["regret"] for r in rs)),
                        "negative_choices": sum(r["negative_choice"] for r in rs)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executions", required=True)
    ap.add_argument("--margin", type=float, default=0.001)
    ap.add_argument("--boots", type=int, default=20000)
    ap.add_argument("--out", required=True, help="prefix for _cells.tsv and _report.md")
    a = ap.parse_args(argv)
    cells, _probes = load_gold(a.executions)
    rows = evaluate(cells, profiles(), a.margin, a.boots)
    cols = ["estimator", "view", "target", "best", "choice", "top1", "equivalent", "regret",
            "negative_choice", "rho", "tau", "pair_acc_decisive", "n_decisive"]
    with open(f"{a.out}_cells.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    s = summarize(rows)
    keys = list(s[0])
    L = ["# Best-average source rule, leave one target out (Track B, KT, N=3000)", "",
         "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    L += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in s]
    L += ["", "## Picks per target", "", "| view | target | best | choice | regret |",
          "|---|---|---|---|---|"]
    L += [f"| {r['view']} | {r['target']} | {r['best']} | {r['choice']} | {q(r['regret'])} |"
          for r in rows if r["estimator"] == "best average"]
    Path(f"{a.out}_report.md").write_text("\n".join(L) + "\n")
    for r in s:
        print(" ".join(f"{k}={v}" for k, v in r.items()))
    print(f"wrote {a.out}_cells.tsv and {a.out}_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
