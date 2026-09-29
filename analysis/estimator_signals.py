from __future__ import annotations

# MRAP Task 3, step 4: confidence and variance signals per estimator per target, kept as raw
# material for the uncertainty-aware method. Score files are read with evaluate_estimators.py's
# own loader, so grouping (estimator@tag, target, budget) and candidate names match the evaluation
# exactly. Signals, per group, over the estimator seeds:
#   pick agreement   seeds whose top candidate is the modal top candidate
#   margin           top score minus runner-up score, mean over seeds (estimator's own units)
#   seed SD          per-candidate population SD of the score across seeds, averaged
#   margin / SD      the margin in units of seed noise; below 1 the pick is within seed noise
#   not converged    records whose uncertainty_signals report converged = False
# With --cells, the mean regret of the same group (evaluate_estimators.py *_cells.tsv) is joined,
# so a reader can see whether low-margin groups are the ones that pick badly. Nothing is fitted.
#
#   PYTHONPATH=. python analysis/estimator_signals.py --scores tg1_*.jsonl \
#       --cells benchmark_final/estimators_cells.tsv --view pretrained --out signals

import argparse
import csv
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from analysis.evaluate_estimators import logme_scores
except ImportError:  # run from analysis/ without PYTHONPATH
    from evaluate_estimators import logme_scores


def oriented(result: dict) -> float:
    s = float(result["score"])
    return s if result.get("score_direction", "higher_is_better") == "higher_is_better" else -s


def signals(scores: dict, pretrained_only: bool) -> list[dict]:
    rows = []
    for (est, tgt, budget), by_seed in sorted(scores.items()):
        tops, margins, per_cand, not_conv = [], [], defaultdict(list), 0
        for seed in sorted(by_seed):
            cand = {c: v for c, v in by_seed[seed].items()
                    if not (pretrained_only and c == "scratch")}
            if len(cand) < 2:
                continue
            vals = {c: oriented(r) for c, (_s, r) in cand.items()}
            order = sorted(vals, key=lambda c: (-vals[c], c))
            tops.append(order[0])
            margins.append(vals[order[0]] - vals[order[1]])
            for c, v in vals.items():
                per_cand[c].append(v)
            not_conv += sum(1 for _s, r in cand.values()
                            if (r.get("uncertainty_signals") or {}).get("converged") is False)
        if not tops:
            continue
        modal, hits = Counter(tops).most_common(1)[0]
        sds = [st.pstdev(v) for v in per_cand.values() if len(v) > 1]
        sd = st.mean(sds) if sds else 0.0
        margin = st.mean(margins)
        rows.append({"estimator": est, "target": tgt, "budget": budget, "seeds": len(tops),
                     "modal_pick": modal, "pick_agreement": f"{hits}/{len(tops)}",
                     "margin": margin, "seed_sd": sd,
                     "margin_over_sd": (margin / sd) if sd > 0 else "",
                     "not_converged": not_conv})
    return rows


def join_regret(rows: list[dict], cells_path: str, view: str) -> None:
    reg: dict = defaultdict(list)
    with open(cells_path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("view") == view and r.get("regret") not in (None, ""):
                reg[(r["estimator"], r["target"], r["budget"])].append(float(r["regret"]))
    for row in rows:
        v = reg.get((row["estimator"], row["target"], row["budget"]))
        row["mean_regret"] = st.mean(v) if v else ""


def fmt(x) -> str:
    return f"{x:.6f}" if isinstance(x, float) else str(x)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--cells", default=None, help="evaluate_estimators.py *_cells.tsv")
    ap.add_argument("--view", default="pretrained", choices=("pretrained", "practical"))
    ap.add_argument("--out", required=True, help="prefix: writes <out>.tsv and <out>.md")
    a = ap.parse_args(argv)
    rows = signals(logme_scores(a.scores), pretrained_only=a.view == "pretrained")
    if not rows:
        sys.exit("ABORT: no scored groups found in the score files")
    if a.cells:
        join_regret(rows, a.cells, a.view)
    cols = list(rows[0])
    with open(f"{a.out}.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        w.writerows([[fmt(r[c]) for c in cols] for r in rows])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    Path(f"{a.out}.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} groups from {len(a.scores)} score files; view {a.view}; "
          f"wrote {a.out}.tsv and {a.out}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
