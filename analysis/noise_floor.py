from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import statistics as st
import sys
from fractions import Fraction
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE = {_HERE / "nlp_budget_extension.py": "39d08a0959e4f99f5c01510912b2b840",
         _HERE / "reversal_table.py": "6f56ff64bd2bc6b86637c4a8eb464b51"}
for _p, _m in _CODE.items():
    if hashlib.md5(_p.read_bytes()).hexdigest() != _m:
        sys.exit(f"ABORT: {_p} md5 differs from {_m}")

from analysis import nlp_budget_extension as nbe  # noqa: E402
from analysis import regret_compute as rc  # noqa: E402
from analysis import reversal_table as rt  # noqa: E402
from analysis import selective_trust as stl  # noqa: E402

EXECUTIONS_MD5 = "92a3e5250b50727f911fe6f09d4b225b"
QT_DRAWS = 10000
GOLD_TOL = 1e-9


def method_list(bench: str, budgets: tuple[int, ...]) -> list[tuple[str, str, object]]:
    out = [("static H-score", "", rc.static)]
    for k in budgets:
        out += [("early leader", f"epoch {k}", lambda d, k=k: rc.early(d, k)),
                ("override rule", f"epoch {k}", lambda d, k=k: rc.override(d, k)),
                ("warning, abstain", f"epoch {k}", lambda d, k=k: rc.abstain(d, k))]
    for eta in (2, 3):
        name = "successive halving" + ("" if eta == 2 else " (factor 3)")
        if bench == "nlp":
            out.append((name, "rungs 1,2,3", lambda d, eta=eta: rc.halving(d, [1, 2, 3], eta)))
            starts, cap = (1, 2, 3), nbe.EPOCHS
        else:
            starts, cap = rc.SH_START[bench], None
        for r in starts:
            out.append((name, f"start {r}", lambda d, r=r, eta=eta, cap=cap: rc.halving(
                d, rc.geometric(r, eta, len(d.cands), cap or d.epochs), eta)))
    if bench == "studentbert":
        out.append(("full fine-tuning", "best validation", rc.full_best_val))
    return out


def run_methods(ds: list[rc.Decision], methods: list) -> list[dict]:
    rows = []
    for name, param, fn in methods:
        res = [fn(d) for d in ds]
        followed = [(d, c) for d, (c, _) in zip(ds, res, strict=True) if c is not None]
        regs = [d.regret(c) for d, c in followed]
        rows.append({"method": name, "param": param, "decisions": len(ds),
                     "chosen": len(followed),
                     "compute_gpu_h": st.mean(s for _, s in res) / 3600,
                     "regret": st.mean(regs) if regs else float("nan"),
                     "choices": {d.key: c for d, c in followed}})
    return rows


def check_methods(bench: str, ds: list[rc.Decision], rows: list[dict]) -> None:
    want = {(r["method"], r["param"]): r for r in rc.evaluate(bench, ds)}
    for r in rows:
        w = want.get((r["method"], r["param"]))
        if w is None:
            sys.exit(f"ABORT: {bench} {r['method']} {r['param']} is not a 12.27 row")
        both_nan = math.isnan(r["regret"]) and math.isnan(w["regret"])
        same = (r["chosen"], r["compute_gpu_h"]) == (w["chosen"], w["compute_gpu_h"]) and (
            both_nan or r["regret"] == w["regret"])
        if not same:
            sys.exit(f"ABORT: {bench} {r['method']} {r['param']} differs from regret_compute")


def full_floor(ds: list[rc.Decision]) -> dict[str, Fraction]:
    return {d.key: stl.regret(d, rc.full_best_val(d)[0]) for d in ds}


def argmax_by_name(scores: dict[str, float]) -> str:
    top = max(scores.values())
    return min(c for c, v in scores.items() if v == top)


def sb_seed_curves(traj: Path) -> dict:
    curves: dict = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in csv.DictReader(traj.read_text().splitlines(), delimiter="\t"):
        if r["benchmark"] == "studentbert":
            curves[(r["target"], int(r["seed"]))][r["candidate"]][int(r["epoch"])] = \
                float(r["score"])
    return curves


def sb_spread(sb: list[rc.Decision], curves: dict) -> dict[int, Fraction]:
    targets = {d.cluster: d for d in sb}
    seeds = sorted({s for _, s in curves})
    if len(seeds) != 6:
        sys.exit(f"ABORT: StudentBERT trajectories have seeds {seeds}, expected 6")
    out = {}
    for s in seeds:
        regs = []
        for t, d in sorted(targets.items()):
            cv = curves[(t, s)]
            if set(cv) != set(d.cands):
                sys.exit(f"ABORT: StudentBERT {t} seed {s}: candidates differ from the decision")
            regs.append(stl.regret(d, argmax_by_name({c: max(cv[c].values()) for c in cv})))
        out[s] = sum(regs, Fraction(0)) / len(regs)
    return out


def sb_seed_oracle(sb: list[rc.Decision], executions: Path) -> dict[int, Fraction]:
    gold = rt.load_gold(str(executions))
    targets = {d.cluster: d for d in sb}
    for t, d in targets.items():
        if set(gold[t]) != set(d.cands):
            sys.exit(f"ABORT: executions.tsv {t}: candidates differ from the decision")
        for c in d.cands:
            if abs(st.mean(gold[t][c].values()) - d.outcome[c]) > GOLD_TOL:
                sys.exit(f"ABORT: executions.tsv {t} {c}: 6-seed mean differs from the gold")
    seeds = sorted({s for t in gold for c in gold[t] for s in gold[t][c]})
    out = {}
    for s in seeds:
        regs = []
        for t, d in sorted(targets.items()):
            regs.append(stl.regret(d, argmax_by_name({c: gold[t][c][s] for c in d.cands})))
        out[s] = sum(regs, Fraction(0)) / len(regs)
    return out


def qt_runs(root: Path) -> dict[tuple[str, str], list[list[float]]]:
    d = root / "micro" / "micro"
    table = list(csv.DictReader((d / "unnormalized_args_table.csv").open()))
    idc = list(table[0])[0]
    top1 = json.load((d / "eval_top1.json").open())
    task_of = {rid: t for t, runs in top1.items() for rid in runs}
    runs = collections.defaultdict(list)
    for r in table:
        m = [k[len("cat__model_"):] for k, v in r.items()
             if k.startswith("cat__model_") and float(v or 0)]
        t = task_of[r[idc]]
        if len(top1[t][r[idc]]) >= rc.QT_EPOCHS:
            runs[(t.rsplit("/", 1)[1], m[0])].append(top1[t][r[idc]][:rc.QT_EPOCHS])
    return {k: v for k, v in runs.items() if len(v) >= rc.QT_MIN_RUNS}


def check_qt_runs(qt: list[rc.Decision], runs: dict) -> None:
    for d in qt:
        for m in d.cands:
            med = [st.median(r[e] for r in runs[(d.cluster, m)]) for e in range(rc.QT_EPOCHS)]
            if med != d.curves[m]:
                sys.exit(f"ABORT: Quick-Tune {d.cluster} {m}: run medians differ from the curve")


def qt_floor(qt: list[rc.Decision], runs: dict, draws: int) -> tuple[dict, list[float]]:
    rng = random.Random(0)
    task_dec = {}
    for d in qt:
        task_dec.setdefault(d.cluster, d)
    floor, per_draw = {}, [[] for _ in range(draws)]
    for t in sorted(task_dec):
        d = task_dec[t]
        best = {m: [max(r) for r in runs[(t, m)]] for m in sorted(d.cands)}
        total = Fraction(0)
        for i in range(draws):
            pick = argmax_by_name({m: rng.choice(v) for m, v in best.items()})
            r = stl.regret(d, pick)
            total += r
            per_draw[i].append(float(r))
        floor[t] = total / draws
    means = sorted(st.mean(x) for x in per_draw)
    return floor, [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def nlp_leave_one_out(nl: list[rc.Decision]) -> Fraction:
    groups = collections.defaultdict(list)
    for d in nl:
        t, p, _ = d.key.split("/")
        groups[(t, p)].append(d)
    regs = []
    for _, ds in sorted(groups.items()):
        for d in ds:
            v = {c: Fraction(d.curves[c][nbe.EPOCHS - 1]) for c in d.cands}
            pick = next(c for c in d.order if v[c] == max(v.values()))
            others = [o for o in ds if o is not d]
            m = {c: sum((Fraction(o.curves[c][nbe.EPOCHS - 1]) for o in others), Fraction(0))
                 / len(others) for c in d.cands}
            regs.append((max(m.values()) - m[pick]) * 100)
    return sum(regs, Fraction(0)) / len(regs)


def label(ci: tuple[float, float]) -> str:
    if ci[0] > 0:
        return "above the floor"
    if ci[1] < 0:
        return "below the floor"
    return "not distinguishable from the floor"


def relative(ds: list[rc.Decision], rows: list[dict], floor: dict[str, Fraction],
             norm: Fraction) -> list[dict]:
    by = {d.key: d for d in ds}
    static = next(r for r in rows if r["method"] == "static H-score")["choices"]
    out = []
    for r in rows:
        keys = [k for k in by if k in r["choices"]]
        if not keys:
            out.append({**r, "followed": 0})
            continue
        fd = [by[k] for k in keys]
        reg = [stl.regret(by[k], r["choices"][k]) for k in keys]
        diff = [g - floor[k] for g, k in zip(reg, keys, strict=True)]
        vs = [g - stl.regret(by[k], static[k]) for g, k in zip(reg, keys, strict=True)]
        ex = sum(diff, Fraction(0)) / len(diff)
        ci = rc.boot_ci(fd, [float(x) for x in diff])
        ci_vs = rc.boot_ci(fd, [float(x) for x in vs])
        out.append({**r, "followed": len(keys), "exact_regret": sum(reg, Fraction(0)) / len(reg),
                    "floor_same": sum((floor[k] for k in keys), Fraction(0)) / len(keys),
                    "excess": ex, "excess_ci": ci, "norm_excess": ex / norm,
                    "norm_excess_ci": (Fraction(ci[0]) / norm, Fraction(ci[1]) / norm),
                    "label": label(ci), "vs_static": sum(vs, Fraction(0)) / len(vs),
                    "vs_static_ci": ci_vs})
    return out


def ci_txt(x: Fraction, ci: tuple, nd: int) -> str:
    return f"{stl.dec(x, nd)} [{stl.dec(Fraction(ci[0]), nd)}, {stl.dec(Fraction(ci[1]), nd)}]"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--task2", nargs="+", required=True)
    ap.add_argument("--executions", required=True)
    ap.add_argument("--nlp-heads", nargs=3, required=True)
    ap.add_argument("--nlp-early", nargs=3, required=True)
    ap.add_argument("--nlp-early-new", nargs=3, required=True)
    ap.add_argument("--nlp-scores", nargs=3, required=True)
    ap.add_argument("--quicktune", required=True)
    ap.add_argument("--qt-scores", required=True)
    ap.add_argument("--qt-extract-log", required=True)
    ap.add_argument("--runs-per-group", type=int, default=nbe.RUNS_PER_GROUP)
    ap.add_argument("--qt-draws", type=int, default=QT_DRAWS)
    ap.add_argument("--no-pin", action="store_true", help="tests only: skip the md5 pins")
    a = ap.parse_args(argv)
    files = [Path(a.trajectories), Path(a.pairs), *map(Path, a.task2), *map(Path, a.nlp_heads),
             *map(Path, a.nlp_scores), Path(a.qt_scores), Path(a.qt_extract_log)]
    inputs = stl.pin(files, stl.PINS, a.no_pin)
    inputs += stl.pin([Path(a.executions)], {"executions.tsv": EXECUTIONS_MD5}, a.no_pin)
    pins = {**stl.DOCS, **nbe.DESIGN}
    docs = stl.pin([_HERE.parent / "docs" / n for n in pins], pins, a.no_pin)
    old_dirs, new_dirs = [Path(p) for p in a.nlp_early], [Path(p) for p in a.nlp_early_new]

    sb = rc.studentbert(Path(a.trajectories), Path(a.pairs), [Path(p) for p in a.task2])
    old = rc.nlp([Path(p) for p in a.nlp_heads], old_dirs, [Path(p) for p in a.nlp_scores])
    qt, _ = rc.quicktune(Path(a.quicktune), Path(a.qt_scores), Path(a.qt_extract_log))
    for b, ds in (("studentbert", sb), ("nlp", old), ("quicktune", qt)):
        stl.check_t0(b, ds, stl.normalizer(ds), rc.evaluate(b, ds))
    nbe.check_references(old)
    nl, comps = nbe.extend(old, new_dirs, old_dirs, nbe.EPOCHS)
    man = nbe.manifest(new_dirs, {c["file"] for c in comps}, a.runs_per_group)

    rows = {"studentbert": run_methods(sb, method_list("studentbert", rc.BUDGETS["studentbert"])),
            "nlp": run_methods(nl, method_list("nlp", nbe.BUDGETS)),
            "quicktune": run_methods(qt, method_list("quicktune", rc.BUDGETS["quicktune"]))}
    check_methods("studentbert", sb, rows["studentbert"])
    check_methods("quicktune", qt, rows["quicktune"])
    runs = qt_runs(Path(a.quicktune))
    check_qt_runs(qt, runs)

    qfloor, qspread = qt_floor(qt, runs, a.qt_draws)
    floors = {"studentbert": full_floor(sb), "nlp": full_floor(nl),
              "quicktune": {d.key: qfloor[d.cluster] for d in qt}}
    domains = {"studentbert": sb, "nlp": nl, "quicktune": qt}
    sb_sp = sb_spread(sb, sb_seed_curves(Path(a.trajectories)))
    sb_or = sb_seed_oracle(sb, Path(a.executions))
    nlp_seed = {}
    for d in nl:
        nlp_seed.setdefault(int(d.key.split("/")[2]), []).append(floors["nlp"][d.key])
    nlp_loo = nlp_leave_one_out(nl)

    L = ["# Noise floor per domain, and every method relative to it", "",
         "Evidence classes: the floor and the read-outs follow a design written on 2026-11-24 "
         "before this script, kept outside the repository at Harry's request, so no commit "
         "documents its date; the read-outs are descriptive. The StudentBERT floor at the "
         "estimator seeds is the RESULTS.md 12.27 row \"full fine-tuning, best validation\", "
         "known before. NLP uses the 10-epoch campaign of analysis/nlp_budget_extension.py. "
         "Floor: the regret, against the domain's outcome, of choosing by one seed's (or one "
         "run's) complete fine-tuning, the best validation score over its epochs. Excess: the "
         "mean over the decisions a method follows of its regret minus the decision's floor, "
         "with a 95% paired cluster bootstrap (targets or tasks, 10,000 resamples, "
         "random.Random(0)). The script was not run on real data before this run; its tests use "
         "synthetic data.", "",
         f"Inputs: {', '.join(inputs)}. Design files: {', '.join(docs)}. Code: "
         + ", ".join(f"{p.name} {m}" for p, m in _CODE.items())
         + f", regret_compute.py {stl.RC_MD5}. NLP 10-epoch runs: {len(comps)} (manifest md5 "
         f"{man}).", "",
         "Checks passed: the recorded decisions of all three domains reproduce every RESULTS.md "
         "12.27 \"warning, abstain\" row; every StudentBERT and Quick-Tune method row reproduces "
         "its 12.27 row; the NLP reference rows reproduce 12.27 on the recorded runs; the "
         "10-epoch runs match their recorded runs in data and settings; the Quick-Tune run "
         "medians reproduce the 12.27 curves; the executions.tsv 6-seed means reproduce the "
         "StudentBERT gold.", ""]
    tsv_rows = []
    for b, ds in domains.items():
        nd, norm, fl = stl.DIGITS[b], stl.normalizer(ds), floors[b]
        vals = [fl[d.key] for d in ds]
        fmean = sum(vals, Fraction(0)) / len(vals)
        fci = rc.boot_ci(ds, [float(v) for v in vals])
        L += [f"## {b} ({stl.UNITS[b]})", "",
              f"Primary floor: {ci_txt(fmean, fci, nd)} {stl.UNITS[b]}, normalized "
              f"{stl.dec(fmean / norm, 3)}; zero regret on {sum(v == 0 for v in vals)} of "
              f"{len(ds)} decisions.", ""]
        if b == "studentbert":
            L += ["Spread over the 6 fine-tuning seeds (each seed's pick, mean over the 7 "
                  "targets; ties to the candidate name): "
                  + ", ".join(f"seed {s} {stl.dec(v, nd)}" for s, v in sb_sp.items())
                  + f"; range {stl.dec(min(sb_sp.values()), nd)} to "
                  f"{stl.dec(max(sb_sp.values()), nd)}.", "",
                  "Secondary, the seed oracle (the best test AUC of one seed against the 6-seed "
                  "mean): " + ", ".join(f"seed {s} {stl.dec(v, nd)}" for s, v in sb_or.items())
                  + f"; mean {stl.dec(sum(sb_or.values(), Fraction(0)) / len(sb_or), nd)}.", ""]
        elif b == "nlp":
            per = {s: sum(v, Fraction(0)) / len(v) for s, v in sorted(nlp_seed.items())}
            L += ["Spread over the 9 estimator seeds (mean over 10 decisions each): "
                  + ", ".join(f"seed {s} {stl.dec(v, nd)}" for s, v in per.items())
                  + f"; range {stl.dec(min(per.values()), nd)} to {stl.dec(max(per.values()), nd)}"
                  ". The NLP outcome is the published fully fine-tuned performance, so this floor "
                  "includes the difference between these 10,000-example runs and the published "
                  "protocol.", "",
                  "Secondary, within the protocol (one seed's epoch-10 pick against the mean "
                  "epoch-10 validation accuracy of the other 8 seeds, validation points, not "
                  f"subtracted from the methods): {stl.dec(nlp_loo, nd)}.", ""]
        else:
            L += [f"Spread over the {a.qt_draws} draws of one run per model (2.5th and 97.5th "
                  f"percentiles of the mean floor): {stl.dec(Fraction(qspread[0]), nd)} to "
                  f"{stl.dec(Fraction(qspread[1]), nd)}. Ties go to the model name. Added after "
                  "the first run (2026-11-25): every Quick-Tune run has its own random "
                  "hyperparameters (docs/trajectory_design.md, rule A), so one run "
                  "per model measures the variability of a hyperparameter draw, not of a seed; "
                  "the methods read curves that are medians over the same runs that define the "
                  "outcome, so this floor is not comparable with them and the readings below are "
                  "not a seed-noise comparison.", ""]
        L += ["| method | setting | decisions followed | compute (GPU-h) | regret | floor on "
              "those decisions | excess over the floor [95% CI] | normalized excess [95% CI] | "
              "reading | difference from the static choice [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in relative(ds, rows[b], fl, norm):
            if r["followed"] == 0:
                L.append(f"| {r['method']} | {r['param']} | 0 | {r['compute_gpu_h']:.3f} | n/a | "
                         "n/a | n/a | n/a | n/a | n/a |")
                continue
            vs = "n/a" if r["method"] == "static H-score" else \
                ci_txt(r["vs_static"], r["vs_static_ci"], nd)
            L.append(f"| {r['method']} | {r['param']} | {r['followed']}/{r['decisions']} | "
                     f"{r['compute_gpu_h']:.3f} | {stl.dec(r['exact_regret'], nd)} | "
                     f"{stl.dec(r['floor_same'], nd)} | "
                     f"{ci_txt(r['excess'], r['excess_ci'], nd)} | "
                     f"{ci_txt(r['norm_excess'], r['norm_excess_ci'], 3)} | {r['label']} | {vs} |")
            tsv_rows.append({"benchmark": b, "method": r["method"], "param": r["param"],
                             "followed": r["followed"], "compute_gpu_h": r["compute_gpu_h"],
                             "regret": float(r["exact_regret"]),
                             "floor_same": float(r["floor_same"]),
                             "excess": float(r["excess"]), "excess_low": r["excess_ci"][0],
                             "excess_high": r["excess_ci"][1],
                             "norm_excess": float(r["norm_excess"]), "reading": r["label"]})
        L.append("")
    L += ["Limitations: StudentBERT has 7 targets and NLP 5 tasks, so intervals are wide; the NLP "
          "floor includes the protocol difference to the published outcome; the Quick-Tune floor "
          "measures hyperparameter-draw variability and is not comparable with the methods; the "
          "StudentBERT floor at the estimator seeds was known before (12.27).",
          ""]

    out = Path(a.out)
    tsv = out.with_name(out.name + ".tsv")
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tsv_rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in tsv_rows:
            w.writerow({k: repr(v) if isinstance(v, float) else v for k, v in r.items()})
    L.append(f"Files: {tsv.name} {rc.md5(tsv)}")
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("checks: 12.27 rows reproduced, Quick-Tune medians reproduced, StudentBERT gold "
          "reproduced, 10-epoch runs match their recorded runs")
    print(f"wrote {tsv.name} and {rp.name} md5 {rc.md5(rp)}")


if __name__ == "__main__":
    main()
