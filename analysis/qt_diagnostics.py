from __future__ import annotations

"""Post-hoc diagnostics for the Quick-Tune H-score result (docs/trajectory_amendment2.md). Added
after the pre-specified analysis returned a mean regret of 56.5 points; it changes no rule and no
result, and only describes what drives the estimators' choices. Read-only.

For each estimator: which models it picks and how often; the mean final top-1 of its picks, of
the task's best and of a random candidate; how often its pick's run ended near chance; and the
mean Spearman correlation of its scores with the models' feature dimension. Also: how often a
rule A run ends near chance, by model, and how the final top-1 relates to dimension.

  python analysis/qt_diagnostics.py --trajectories benchmark_final/trajectories.tsv \
      --scores /projects/algl/dai.hany/quicktune/qt_scores.jsonl \
      --extract-log /projects/algl/dai.hany/quicktune/features/extract_log.jsonl
"""

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

NEAR_CHANCE = 10.0  # top-1 points; 20 classes give 5 points by chance


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--extract-log", required=True)
    a = ap.parse_args(argv)
    final = {}
    for r in csv.DictReader(Path(a.trajectories).read_text().splitlines(), delimiter="\t"):
        if r["benchmark"] == "quicktune_micro" and r["epoch"] == "50":
            final[(r["target"].rsplit("/", 1)[1], r["candidate"])] = float(r["score"])
    dim = {}
    for line in Path(a.extract_log).read_text().splitlines():
        r = json.loads(line)
        dim[r["model"]] = r["dim"]
    cands = collections.defaultdict(list)
    for (t, c) in final:
        cands[t].append(c)
    scores = collections.defaultdict(dict)
    for line in Path(a.scores).read_text().splitlines():
        r = json.loads(line)
        if r["score"] is not None:
            scores[(r["estimator"], r["task"], r["seed"])][r["model"]] = r["score"]
    print(f"Rule A runs: {len(final)}; ending near chance (top-1 <= {NEAR_CHANCE:g} at epoch 50): "
          f"{sum(v <= NEAR_CHANCE for v in final.values())}")
    by_model = collections.defaultdict(list)
    for (t, c), v in final.items():
        by_model[c].append(v)
    print("Near-chance share and mean final top-1 by model (dimension):")
    for c in sorted(by_model, key=lambda c: dim[c]):
        v = by_model[c]
        print(f"  {c} ({dim[c]}): {sum(x <= NEAR_CHANCE for x in v)}/{len(v)} near chance, "
              f"mean final {st.mean(v):.1f}")
    rho_fd = [spearman([dim[c] for c in sorted(cands[t])],
                       [final[(t, c)] for c in sorted(cands[t])]) for t in sorted(cands)]
    print(f"Mean Spearman of final top-1 with feature dimension, per task: {st.mean(rho_fd):+.2f}")
    rand = st.mean(st.mean(max(final[(t, x)] for x in cands[t]) - final[(t, c)] for c in cands[t])
                   for t in cands)
    print(f"Mean regret of a random candidate: {rand:.2f} points")
    for est in ("hscore_shrunk", "hscore", "logme"):
        picks, pf, near, rho_sd = collections.Counter(), [], 0, []
        for (e, t, s), sc in sorted(scores.items()):
            if e != est or len(sc) < 2:
                continue
            hi = max(sc.values())
            p = min(c for c in sc if sc[c] == hi)
            picks[p] += 1
            pf.append(final[(t, p)])
            near += final[(t, p)] <= NEAR_CHANCE
            order = sorted(sc)
            rho_sd.append(spearman([sc[c] for c in order], [dim[c] for c in order]))
        print(f"{est}: picks {picks.most_common(6)}; mean final of picks {st.mean(pf):.1f}; "
              f"picks near chance {near}/{len(pf)}; mean Spearman of score with dimension "
              f"{st.mean(rho_sd):+.2f}")
    best = st.mean(max(final[(t, c)] for c in cands[t]) for t in cands)
    print(f"Mean best final top-1 per task: {best:.1f}")


if __name__ == "__main__":
    main()
