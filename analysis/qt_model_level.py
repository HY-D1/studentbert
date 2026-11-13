from __future__ import annotations

"""Quick-Tune model-level outcome and dimension control (docs/trajectory_amendment3.md).

Outcome per (model, micro task): the median (primary), mean and best final top-1 over all complete
50-epoch runs, for models with at least 3 such runs. Estimators: the raw scores of qt_score.py and
rescored features projected on their top 128 principal components. Reported per estimator and
feature variant: agreement with each outcome, the choice's regret, the score's correlation with
feature dimension, and, for raw features, the partial correlation controlling for log-dimension.

  PYTHONPATH=. python analysis/qt_model_level.py --quicktune /projects/algl/dai.hany/quicktune \
      --features /projects/algl/dai.hany/quicktune/features \
      --scores /projects/algl/dai.hany/quicktune/qt_scores.jsonl \
      --extract-log /projects/algl/dai.hany/quicktune/features/extract_log.jsonl \
      --out benchmark_final/qt_model_level
"""

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

from analysis.qt_reversal import spearman
from analysis.qt_score import SEEDS, estimators, half

TIE = 0.5
EPOCHS = 50
MIN_RUNS = 3
PCA_DIM = 128
PRIMARY = "hscore_shrunk"
ESTIMATORS = ("hscore_shrunk", "hscore", "logme")
OUTCOMES = ("median", "mean", "best")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def outcomes(root: Path) -> tuple[dict, int, int]:
    d = root / "micro" / "micro"
    table = list(csv.DictReader((d / "unnormalized_args_table.csv").open()))
    idc = list(table[0])[0]
    top1 = json.load((d / "eval_top1.json").open())
    task_of = {rid: t for t, runs in top1.items() for rid in runs}
    finals = collections.defaultdict(list)
    for r in table:
        m = [k[len("cat__model_"):] for k, v in r.items()
             if k.startswith("cat__model_") and float(v or 0)]
        t = task_of.get(r[idc])
        if len(m) != 1 or t is None:
            sys.exit(f"ABORT: run {r[idc]} has no single model or no curve")
        curve = top1[t][r[idc]]
        if len(curve) >= EPOCHS:
            finals[(t.rsplit("/", 1)[1], m[0])].append(float(curve[EPOCHS - 1]))
    out, excluded = {}, 0
    for k, v in finals.items():
        if len(v) < MIN_RUNS:
            excluded += 1
            continue
        out[k] = {"median": st.median(v), "mean": st.mean(v), "best": max(v), "runs": len(v)}
    return out, excluded, len(finals)


def pca_scores(features: Path, cands: dict[str, list[str]]) -> dict:
    est = estimators()
    out = {}
    for task, models in sorted(cands.items()):
        z = {m: np.load(features / task / f"{m}.npz") for m in models}
        y_all = z[models[0]]["labels"]
        for m in models[1:]:
            if not (np.array_equal(z[m]["labels"], y_all)
                    and np.array_equal(z[m]["files"], z[models[0]]["files"])):
                sys.exit(f"ABORT: {task} {m}: files or labels differ")
        for seed in SEEDS:
            idx = half(y_all, seed)
            y = y_all[idx]
            for m in models:
                F = z[m]["features"][idx].astype(np.float64)
                F = F - F.mean(axis=0)
                if F.shape[1] > PCA_DIM:
                    _, _, vt = np.linalg.svd(F, full_matrices=False)
                    F = F @ vt[:PCA_DIM].T
                for name, fn in est.items():
                    try:
                        s, _ = fn(F, y)
                        s = float(s)
                    except ValueError:
                        s = None
                    out[(name, task, seed, m)] = s
        print(f"{task}: {len(models)} models rescored at {PCA_DIM} dimensions", flush=True)
    return out


def summarize(scores: dict, outc: dict, dims: dict, cands: dict) -> list[dict]:
    rows = []
    for variant, sc_all in scores.items():
        for name in ESTIMATORS:
            for o in OUTCOMES:
                rho, part, sdim, top, reg = [], [], [], 0, []
                for task, models in sorted(cands.items()):
                    for seed in SEEDS:
                        sc = {m: sc_all.get((name, task, seed, m)) for m in models}
                        sc = {m: v for m, v in sc.items() if v is not None}
                        if len(sc) < 3:
                            continue
                        order = sorted(sc)
                        s = [sc[m] for m in order]
                        y = [outc[(task, m)][o] for m in order]
                        ld = [math.log(dims[m]) for m in order]
                        r_sy, r_sd, r_yd = spearman(s, y), spearman(s, ld), spearman(y, ld)
                        rho.append(r_sy)
                        sdim.append(r_sd)
                        den = ((1 - r_sd ** 2) * (1 - r_yd ** 2)) ** 0.5
                        if variant == "raw" and den > 1e-12:
                            part.append((r_sy - r_sd * r_yd) / den)
                        best = max(outc[(task, m)][o] for m in models)
                        hi = max(sc.values())
                        pick = min(m for m in sc if sc[m] == hi)
                        top += best - outc[(task, pick)][o] <= TIE
                        reg.append(best - outc[(task, pick)][o])
                def mean(v: list[float]) -> float:
                    v = [x for x in v if not math.isnan(x)]
                    return st.mean(v) if v else float("nan")
                rows.append({"variant": variant, "estimator": name, "outcome": o,
                             "rankings": len(reg), "spearman": mean(rho),
                             "partial": mean(part) if variant == "raw" else float("nan"),
                             "score_vs_dim": mean(sdim), "top1": top,
                             "regret": mean(reg)})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quicktune", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--extract-log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    outc, excluded, pairs = outcomes(Path(a.quicktune))
    cands = collections.defaultdict(list)
    for (t, m) in sorted(outc):
        cands[t].append(m)
    dims = {}
    for line in Path(a.extract_log).read_text().splitlines():
        r = json.loads(line)
        dims[r["model"]] = r["dim"]
    raw = {}
    for line in Path(a.scores).read_text().splitlines():
        r = json.loads(line)
        if (r["task"], r["model"]) in outc:
            raw[(r["estimator"], r["task"], r["seed"], r["model"])] = r["score"]
    missing = [(t, m) for (t, m) in outc
               if (PRIMARY, t, SEEDS[0], m) not in raw]
    if missing:
        sys.exit(f"ABORT: {len(missing)} included models lack raw scores, e.g. {missing[:3]}")
    pca = pca_scores(Path(a.features), cands)
    rows = summarize({"raw": raw, f"pca{PCA_DIM}": pca}, outc, dims, cands)
    rand = {o: st.mean(st.mean(max(outc[(t, x)][o] for x in ms) - outc[(t, m)][o] for m in ms)
                       for t, ms in cands.items()) for o in OUTCOMES}
    out_dim = {o: st.mean(spearman([outc[(t, m)][o] for m in sorted(ms)],
                                   [math.log(dims[m]) for m in sorted(ms)])
                          for t, ms in cands.items()) for o in OUTCOMES}
    runs = [v["runs"] for v in outc.values()]
    L = ["# Quick-Tune micro: model-level outcome and dimension control", "",
         f"Pre-specified in docs/trajectory_amendment3.md. Inputs: {Path(a.scores).name} "
         f"{md5(Path(a.scores))}, {Path(a.extract_log).name} {md5(Path(a.extract_log))}. "
         f"Model-task pairs with complete runs: {pairs}; included (at least {MIN_RUNS} complete "
         f"runs): {len(outc)} on {len(cands)} tasks; excluded: {excluded}. Complete runs per "
         f"included pair: median {st.median(runs):.0f}, range {min(runs)} to {max(runs)}.", ""]
    for o in OUTCOMES:
        L += [f"## Outcome: {o} final top-1 over complete runs"
              + (" (primary)" if o == "median" else ""), "",
              f"Random model: mean regret {rand[o]:.2f} points. Outcome against log-dimension: "
              f"mean Spearman {out_dim[o]:+.2f}.", "",
              "| features | estimator | rankings | Spearman with outcome | partial, controlling "
              "log-dimension | score with log-dimension | top-1 tied with the best | mean regret "
              "(points) |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["outcome"] != o:
                continue
            def fmt(x: float) -> str:
                return "n/a" if math.isnan(x) else f"{x:+.2f}"
            part = "" if r["variant"] != "raw" else fmt(r["partial"])
            L.append(f"| {r['variant']} | {r['estimator']} | {r['rankings']} | "
                     f"{fmt(r['spearman'])} | {part} | {fmt(r['score_vs_dim'])} | "
                     f"{r['top1']}/{r['rankings']} | {r['regret']:.2f} |")
        L.append("")
    out = Path(a.out)
    tsv = out.with_name(out.name + ".tsv")
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    L.append(f"Files: {tsv.name} {md5(tsv)}")
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {tsv.name} and {rp.name} md5 {md5(rp)}")


if __name__ == "__main__":
    main()
