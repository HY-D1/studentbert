from __future__ import annotations

"""Option 1, step A: the H-score recommendation and its head pairs on the NLP benchmark.

No new runs. For every task, pooling and estimator seed, H-score's pick among the 7 encoders is
paired with each of the 6 alternatives, and each pair is labelled against the published tuned gold
of Bassignana et al. (2022) with the Task 4 tie rule (analysis/task4_evaluate.py: gap <= 0.1
points, or inside a two-sided 95% interval from the published SDs, 5 seeds): near_tie, stable (the
pick stays ahead) or reversal (the alternative overtakes the pick). A recommendation is reversed
when at least one alternative overtakes its pick. The counts per task decide whether Option 1 has
enough reversal events to be worth new fine-tuning runs (option1_nlp_scoping.md, step A).

  PYTHONPATH=. python analysis/nlp_head_pairs.py \
      --scores /projects/algl/dai.hany/task4/task4_scores.jsonl --out benchmark_final/nlp_head
"""

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from analysis import task4_evaluate as te

TASKS = ("agnews", "mnli", "qnli", "rte", "airline")
POOLINGS = ("cls", "mean")


def build(scores: dict, gold: dict) -> tuple[list[dict], list[dict]]:
    pairs, recs = [], []
    for task in TASKS:
        for pooling in POOLINGS:
            g = gold.get((task, pooling))
            hs = scores.get(("hscore", task, pooling))
            lm = scores.get(("logme", task, pooling))
            if not g or not hs or not lm:
                raise SystemExit(f"ABORT: missing gold or scores for {task} {pooling}")
            for seed in sorted(hs):
                sc = hs[seed]
                if set(sc) != set(g) or set(lm[seed]) != set(g):
                    raise SystemExit(f"ABORT: {task} {pooling} seed {seed} lacks a model")
                perf = {m: g[m]["tuned"] for m in g}
                j = te.judge(sc, perf)
                pick = j["choice"]
                n_rev = 0
                for b in sorted(m for m in g if m != pick):
                    if te.tied(perf[pick], perf[b]):
                        lab = "near_tie"
                    else:
                        lab = "reversal" if perf[b][0] > perf[pick][0] else "stable"
                    n_rev += lab == "reversal"
                    pairs.append({"task": task, "pooling": pooling, "seed": seed, "pick": pick,
                                  "alternative": b, "hscore_pick": sc[pick], "hscore_alt": sc[b],
                                  "logme_pick": lm[seed][pick], "logme_alt": lm[seed][b],
                                  "tuned_pick": perf[pick][0], "tuned_alt": perf[b][0],
                                  "gap_points": perf[b][0] - perf[pick][0], "label": lab})
                if not j["tied_with_best"] and n_rev == 0:
                    raise SystemExit(f"ABORT: {task} {pooling} {seed}: the pick trails the best "
                                     "decisively but no pair is a reversal")
                recs.append({"task": task, "pooling": pooling, "seed": seed, "pick": pick,
                             "best": j["best"], "tied_with_best": int(j["tied_with_best"]),
                             "regret_points": j["regret"], "reversed": int(n_rev > 0),
                             "n_overtaking": n_rev})
    return pairs, recs


def summary(pairs: list[dict], recs: list[dict]) -> list[str]:
    by_t = defaultdict(lambda: {"recs": 0, "rev": 0, "pairs_rev": 0, "pairs_tie": 0})
    for r in recs:
        by_t[r["task"]]["recs"] += 1
        by_t[r["task"]]["rev"] += r["reversed"]
    for p in pairs:
        by_t[p["task"]]["pairs_rev"] += p["label"] == "reversal"
        by_t[p["task"]]["pairs_tie"] += p["label"] == "near_tie"
    L = ["| task | recommendations | reversed | reversal pairs | near-tie pairs |",
         "|---|---|---|---|---|"]
    for t in TASKS:
        s = by_t[t]
        L.append(f"| {t} | {s['recs']} | {s['rev']} | {s['pairs_rev']} | {s['pairs_tie']} |")
    tot_rev = sum(r["reversed"] for r in recs)
    events = sum(by_t[t]["rev"] > 0 for t in TASKS)
    L += ["", f"Reversed recommendations: {tot_rev} of {len(recs)}, from {events} of "
          f"{len(TASKS)} tasks. Head pairs: {len(pairs)} ("
          f"{sum(p['label'] == 'stable' for p in pairs)} stable, "
          f"{sum(p['label'] == 'near_tie' for p in pairs)} near-tie, "
          f"{sum(p['label'] == 'reversal' for p in pairs)} reversal)."]
    return L


def write_tsv(path: Path, rows: list[dict]) -> str:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    return hashlib.md5(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True, help="prefix for the _pairs, _recs, _report files")
    a = ap.parse_args(argv)
    pairs, recs = build(te.load_scores(a.scores), te.load_gold())
    out = Path(a.out)
    files = {out.name + "_pairs.tsv": write_tsv(out.with_name(out.name + "_pairs.tsv"), pairs),
             out.name + "_recs.tsv": write_tsv(out.with_name(out.name + "_recs.tsv"), recs)}
    L = ["# NLP head pairs (H-score's pick against each alternative, published tuned gold)", ""]
    L += summary(pairs, recs)
    L += ["", "Files: " + ", ".join(f"{k} {v}" for k, v in files.items())]
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print("\n".join(summary(pairs, recs)[-1:]))
    print(f"wrote {', '.join(files)} and {rp.name} md5 {hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
