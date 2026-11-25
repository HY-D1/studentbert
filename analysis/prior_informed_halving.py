from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import statistics as st
import sys
from fractions import Fraction
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE = {_HERE / "noise_floor.py": "fcc5c16b485fc061a04ff43599851224"}
for _p, _m in _CODE.items():
    if hashlib.md5(_p.read_bytes()).hexdigest() != _m:
        sys.exit(f"ABORT: {_p} md5 differs from {_m}")

from analysis import nlp_budget_extension as nbe  # noqa: E402
from analysis import noise_floor as nf  # noqa: E402
from analysis import regret_compute as rc  # noqa: E402
from analysis import selective_trust as stl  # noqa: E402

ORDERS = 1000
VARIANTS = ("U", "I0", "I1a", "I1b", "I2", "I3")
INFORMED = VARIANTS[1:]
LABELS = {"U": "uninformed (random order)", "I0": "tie-break only",
          "I1a": "prune to ceil(n/2)", "I1b": "prune to ceil(n/4)",
          "I2": "prior in the ranking", "I3": "protected pick"}


def halving(d: rc.Decision, rungs: list[int], eta: int, order: list[str],
            prior: bool = False, protect: str | None = None) -> tuple[str, float]:
    alive, trained = list(order), {c: 0 for c in order}
    pos = {c: i for i, c in enumerate(order)}
    hs_rank = {c: i for i, c in enumerate(d.order)}
    first = True
    for e in rungs:
        e = min(e, d.epochs)
        for c in alive:
            trained[c] = max(trained[c], e)
        if len(alive) == 1:
            break
        if prior:
            val = sorted(alive, key=lambda c: (-d.val(c, e), pos[c]))
            vr = {c: i for i, c in enumerate(val)}
            ranked = sorted(alive, key=lambda c: (Fraction(vr[c] + hs_rank[c], 2), hs_rank[c]))
        else:
            ranked = sorted(alive, key=lambda c: (-d.val(c, e), pos[c]))
        keep = ranked[:math.ceil(len(alive) / eta)]
        if first and protect is not None and protect in alive and protect not in keep:
            keep.append(protect)
        first = False
        alive = keep
        if len(alive) == 1:
            break
    if len(alive) > 1:
        k = max(trained[c] for c in alive)
        top = max(d.val(c, k) for c in alive)
        alive = [next(c for c in order if c in alive and d.val(c, k) == top)]
    return alive[0], d.ft_cost({c: v for c, v in trained.items() if v > 0})


def schedules(bench: str, n: int, eta: int) -> list[tuple[str, list[int]]]:
    if bench == "nlp":
        out = [("rungs 1,2,3", [1, 2, 3])]
        out += [(f"start {r}", rc.geometric(r, eta, n, nbe.EPOCHS)) for r in (1, 2, 3)]
        return out
    cap = 20 if bench == "studentbert" else rc.QT_EPOCHS
    return [(f"start {r}", rc.geometric(r, eta, n, cap)) for r in rc.SH_START[bench]]


def run_variant(d: rc.Decision, variant: str, sched: str, eta: int,
                orders: list[list[str]]) -> tuple[Fraction, float]:
    if variant == "U":
        regs, costs = [], []
        for o in orders:
            rungs = dict(schedules(d.bench, len(o), eta))[sched]
            c, s = halving(d, rungs, eta, o)
            regs.append(stl.regret(d, c))
            costs.append(s)
        return sum(regs, Fraction(0)) / len(regs), st.mean(costs)
    order = list(d.order)
    if variant in ("I1a", "I1b"):
        m = math.ceil(len(order) / (2 if variant == "I1a" else 4))
        order = order[:m]
    rungs = dict(schedules(d.bench, len(order), eta))[sched]
    if len(order) == 1:
        c, s = order[0], 0.0
    else:
        c, s = halving(d, rungs, eta, order, prior=variant == "I2",
                       protect=d.pick if variant == "I3" else None)
    return stl.regret(d, c), s + d.static_axis


def check_i0(bench: str, ds: list[rc.Decision]) -> None:
    for d in ds:
        for eta in (2, 3):
            for _, rungs in schedules(bench, len(d.cands), eta):
                if halving(d, rungs, eta, list(d.order)) != rc.halving(d, rungs, eta):
                    sys.exit(f"ABORT: {bench} {d.key}: halving differs from regret_compute")


def random_orders(ds: list[rc.Decision], n: int) -> dict[str, list[list[str]]]:
    rng = random.Random(0)
    out = {}
    for d in sorted(ds, key=lambda d: d.key):
        out[d.key] = []
        for _ in range(n):
            o = list(d.cands)
            rng.shuffle(o)
            out[d.key].append(o)
    return out


def configs(ds: list[rc.Decision], orders: dict, floor: dict, norm: Fraction) -> list[dict]:
    bench = ds[0].bench
    rows = []
    for variant in VARIANTS:
        for eta in (2, 3):
            for sched, _ in schedules(bench, len(ds[0].cands), eta):
                res = [run_variant(d, variant, sched, eta, orders[d.key]) for d in ds]
                regs = [r for r, _ in res]
                mean = sum(regs, Fraction(0)) / len(regs)
                diff = [float(r - floor[d.key]) for d, r in zip(ds, regs, strict=True)]
                rows.append({"benchmark": bench, "variant": variant,
                             "method": f"{variant} factor {eta}", "param": sched, "eta": eta,
                             "decisions": len(ds), "chosen": len(ds),
                             "compute_gpu_h": st.mean(s for _, s in res) / 3600,
                             "regret": float(mean), "exact": mean, "norm": mean / norm,
                             "ci": rc.boot_ci(ds, [float(r) for r in regs]),
                             "zero": sum(r == 0 for r in regs),
                             "excess": mean - sum((floor[d.key] for d in ds), Fraction(0))
                             / len(ds), "excess_ci": rc.boot_ci(ds, diff), "per": regs})
    return rows


def matched_compute(ds: list[rc.Decision], rows: list[dict]) -> list[dict]:
    us = [r for r in rows if r["variant"] == "U"]
    out = []
    for r in rows:
        if r["variant"] == "U":
            continue
        u = min(us, key=lambda x: (abs(Fraction(x["compute_gpu_h"])
                                       - Fraction(r["compute_gpu_h"])), x["compute_gpu_h"]))
        diff = [a - b for a, b in zip(r["per"], u["per"], strict=True)]
        out.append({"row": r, "u": u, "diff": sum(diff, Fraction(0)) / len(diff),
                    "ci": rc.boot_ci(ds, [float(x) for x in diff])})
    return out


def matched_regret(rows: list[dict]) -> list[dict]:
    out = []
    for u in (r for r in rows if r["variant"] == "U"):
        for v in INFORMED:
            ok = [r for r in rows if r["variant"] == v and r["exact"] <= u["exact"]]
            best = min(ok, key=lambda r: (r["compute_gpu_h"], r["exact"]), default=None)
            out.append({"u": u, "variant": v, "best": best})
    return out


def static_replaced(ds: list[rc.Decision], rows: list[dict]) -> dict | None:
    static = sum((stl.regret(d, d.pick) for d in ds), Fraction(0)) / len(ds)
    ok = [r for r in rows if r["variant"] == "U" and r["exact"] <= static]
    best = min(ok, key=lambda r: (r["compute_gpu_h"], r["exact"]), default=None)
    return {"static": static, "best": best}


def txt(x: Fraction, ci: tuple, nd: int) -> str:
    return f"{stl.dec(x, nd)} [{stl.dec(Fraction(ci[0]), nd)}, {stl.dec(Fraction(ci[1]), nd)}]"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--task2", nargs="+", required=True)
    ap.add_argument("--nlp-heads", nargs=3, required=True)
    ap.add_argument("--nlp-early", nargs=3, required=True)
    ap.add_argument("--nlp-early-new", nargs=3, required=True)
    ap.add_argument("--nlp-scores", nargs=3, required=True)
    ap.add_argument("--quicktune", required=True)
    ap.add_argument("--qt-scores", required=True)
    ap.add_argument("--qt-extract-log", required=True)
    ap.add_argument("--runs-per-group", type=int, default=nbe.RUNS_PER_GROUP)
    ap.add_argument("--orders", type=int, default=ORDERS)
    ap.add_argument("--qt-draws", type=int, default=nf.QT_DRAWS)
    ap.add_argument("--no-pin", action="store_true", help="tests only: skip the md5 pins")
    a = ap.parse_args(argv)
    files = [Path(a.trajectories), Path(a.pairs), *map(Path, a.task2), *map(Path, a.nlp_heads),
             *map(Path, a.nlp_scores), Path(a.qt_scores), Path(a.qt_extract_log)]
    inputs = stl.pin(files, stl.PINS, a.no_pin)
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
    domains = {"studentbert": sb, "nlp": nl, "quicktune": qt}
    for b, ds in domains.items():
        check_i0(b, ds)
    runs = nf.qt_runs(Path(a.quicktune))
    nf.check_qt_runs(qt, runs)
    qfloor, _ = nf.qt_floor(qt, runs, a.qt_draws)
    floors = {"studentbert": nf.full_floor(sb), "nlp": nf.full_floor(nl),
              "quicktune": {d.key: qfloor[d.cluster] for d in qt}}

    L = ["# Prior-informed successive halving: how much fine-tuning compute the score saves", "",
         "Evidence classes: the variants, rungs, accounting and read-outs follow a design written "
         "on 2026-11-24 before this script, kept outside the repository at Harry's request, so "
         "no commit documents its date; the read-outs are descriptive. Successive halving at the "
         "12.27 rungs (I0) was known before and is reproduced here decision by decision. U "
         "averages each decision over random candidate orders; the informed variants pay the "
         "frozen scoring time (off the axis on Quick-Tune, as in 12.27). Regret is against each "
         "domain's outcome; normalized by a random candidate's mean regret; 95% cluster "
         "bootstrap over targets or tasks (10,000 resamples, random.Random(0)). Offline replay "
         "of recorded curves, not new training. The script was not run on real data before this "
         "run; its tests use synthetic data.", "",
         f"Inputs: {', '.join(inputs)}. Design files: {', '.join(docs)}. Code: "
         + ", ".join(f"{p.name} {m}" for p, m in _CODE.items())
         + f", regret_compute.py {stl.RC_MD5}. NLP 10-epoch runs: {len(comps)} (manifest md5 "
         f"{man}). Random orders per decision: {a.orders}.", "",
         "Checks passed: the recorded decisions reproduce every RESULTS.md 12.27 \"warning, "
         "abstain\" row; the halving here equals regret_compute's halving for every decision, "
         "rung schedule and factor; the 10-epoch runs match their recorded runs; the Quick-Tune "
         "run medians reproduce the 12.27 curves.", ""]
    tsv_rows = []
    for b, ds in domains.items():
        nd, norm = stl.DIGITS[b], stl.normalizer(ds)
        orders = random_orders(ds, a.orders)
        rows = configs(ds, orders, floors[b], norm)
        front = rc.frontier(rows, exclude_static_compute=False)
        L += [f"## {b} ({stl.UNITS[b]})", "",
              f"{len(ds)} decisions, {len(rows)} configurations ({len(VARIANTS)} variants: "
              + "; ".join(f"{v} {LABELS[v]}" for v in VARIANTS) + ").", "",
              "| variant | factor | rungs | compute (GPU-h) | regret [95% CI] | normalized | zero "
              "regret | excess over the noise floor [95% CI] | frontier |",
              "|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rows):
            L.append(f"| {r['variant']} | {r['eta']} | {r['param']} | {r['compute_gpu_h']:.3f} | "
                     f"{txt(r['exact'], r['ci'], nd)} | {stl.dec(r['norm'], 3)} | {r['zero']} | "
                     f"{txt(r['excess'], r['excess_ci'], nd)} | {'yes' if i in front else ''} |")
            tsv_rows.append({"benchmark": b, "variant": r["variant"], "factor": r["eta"],
                             "rungs": r["param"], "compute_gpu_h": r["compute_gpu_h"],
                             "regret": r["regret"], "norm_regret": float(r["norm"]),
                             "ci_low": r["ci"][0], "ci_high": r["ci"][1],
                             "excess": float(r["excess"]), "frontier": i in front})
        L += ["", "Regret at matched compute (each informed configuration against the U "
              "configuration at the nearest compute, ties to the lower; difference informed "
              "minus U, paired over decisions):", "",
              "| variant | factor | rungs | compute | U at nearest compute | U compute | regret "
              "difference [95% CI] |", "|---|---|---|---|---|---|---|"]
        for m in matched_compute(ds, rows):
            r, u = m["row"], m["u"]
            L.append(f"| {r['variant']} | {r['eta']} | {r['param']} | {r['compute_gpu_h']:.3f} | "
                     f"factor {u['eta']}, {u['param']} | {u['compute_gpu_h']:.3f} | "
                     f"{txt(m['diff'], m['ci'], nd)} |")
        L += ["", "Compute saved at matched regret (for each U configuration, the cheapest "
              "configuration of each informed variant with mean regret at most U's; saving is "
              "U's compute minus that configuration's):", "",
              "| U configuration | U compute | U regret | variant | cheapest matching "
              "configuration | its compute | saving (GPU-h) | share of U's compute |",
              "|---|---|---|---|---|---|---|---|"]
        for m in matched_regret(rows):
            u, v, best = m["u"], m["variant"], m["best"]
            head = (f"| factor {u['eta']}, {u['param']} | {u['compute_gpu_h']:.3f} | "
                    f"{stl.dec(u['exact'], nd)} | {v} | ")
            if best is None:
                L.append(head + "none | n/a | n/a | n/a |")
                continue
            save = u["compute_gpu_h"] - best["compute_gpu_h"]
            share = f"{save / u['compute_gpu_h']:.3f}" if u["compute_gpu_h"] else "n/a"
            L.append(head + f"factor {best['eta']}, {best['param']} | "
                     f"{best['compute_gpu_h']:.3f} | {save:.3f} | {share} |")
        sr = static_replaced(ds, rows)
        if sr["best"] is None:
            L += ["", f"The static H-score choice ({stl.dec(sr['static'], nd)}): no U "
                  "configuration reaches its mean regret.", ""]
        else:
            bb = sr["best"]
            L += ["", f"The static H-score choice ({stl.dec(sr['static'], nd)}): the cheapest U "
                  f"configuration with mean regret at most this is factor {bb['eta']}, "
                  f"{bb['param']} at {bb['compute_gpu_h']:.3f} GPU-h per decision "
                  f"({stl.dec(bb['exact'], nd)}).", ""]
    L += ["Limitations: offline replay of recorded curves; StudentBERT has 7 targets and NLP 5 "
          "tasks; on Quick-Tune the outcome comes from the same curves as the signal, its scoring "
          "is off the axis and its noise floor measures a hyperparameter draw, not a seed; "
          "several variants are compared without a multiplicity correction; successive halving "
          "at the 12.27 rungs was known before.", ""]

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
    print("checks: 12.27 rows reproduced, halving reproduced, 10-epoch runs match their "
          "recorded runs, Quick-Tune medians reproduced")
    print(f"wrote {tsv.name} and {rp.name} md5 {rc.md5(rp)}")


if __name__ == "__main__":
    main()
