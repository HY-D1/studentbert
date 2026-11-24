from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import random
import statistics as st
import sys
from fractions import Fraction
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODE = {_HERE / "selective_trust.py": "561f9d471f9df4686c35bf6f5c7129a0",
         _HERE.parent / "tools" / "compare_ft_epochs.py": "16fbc8a6b27de3bec529729e1dec032e"}
for _p, _m in _CODE.items():
    if hashlib.md5(_p.read_bytes()).hexdigest() != _m:
        sys.exit(f"ABORT: {_p} md5 differs from {_m}")

from analysis import regret_compute as rc  # noqa: E402
from analysis import selective_trust as stl  # noqa: E402
from tools import compare_ft_epochs as cf  # noqa: E402

DESIGN = {"nlp_budget_extension_design.md": "d82d64f47e3d23f35a94316446a0041b"}
EPOCHS = 10
BUDGETS = (1, 2, 3, 5, 10)
TRANSFER_BUDGETS = (5, 10)
RUNS_PER_GROUP = 210
SEED_GROUP = {42: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}
FLAT = Fraction(1, 50)
ND = stl.DIGITS["nlp"]


def run_file(root: Path, d: rc.Decision, m: str) -> Path:
    t, p, s = d.key.split("/")
    return root / t / f"{m.replace('/', '__')}_{p}_s{s}.json"


def extend(old: list[rc.Decision], new_dirs: list[Path], old_dirs: list[Path],
           epochs: int) -> tuple[list[rc.Decision], list[dict]]:
    out, comps = [], []
    for d in old:
        s = int(d.key.split("/")[2])
        if s not in SEED_GROUP:
            sys.exit(f"ABORT: {d.key}: seed {s} is in no seed group")
        g = SEED_GROUP[s]
        nd = copy.copy(d)
        nd.curves, nd.costs = {}, {}
        for m in d.cands:
            fn, fo = run_file(new_dirs[g], d, m), run_file(old_dirs[g], d, m)
            if not fn.is_file():
                sys.exit(f"ABORT: no 10-epoch run {fn}")
            new, rec = json.loads(fn.read_text()), json.loads(fo.read_text())
            if [e["val_acc"] for e in sorted(rec["epochs"], key=lambda e: e["epoch"])] != \
                    d.curves[m]:
                sys.exit(f"ABORT: {fo} is not the run behind decision {d.key}")
            try:
                c = cf.compare(new, rec, epochs)
            except ValueError as e:
                sys.exit(f"ABORT: {fn}: {e}")
            comps.append({"group": g, "file": fn, **c})
            ep = sorted(new["epochs"], key=lambda e: e["epoch"])
            nd.curves[m] = [e["val_acc"] for e in ep]
            nd.costs[m] = [sum(e["epoch_s"] for e in ep[:i + 1]) for i in range(epochs)]
        nd.epochs = min(len(v) for v in nd.curves.values())
        out.append(nd)
    return out, comps


def manifest(new_dirs: list[Path], used: set[Path], expect: int) -> str:
    lines = []
    for root in new_dirs:
        files = sorted(root.glob("*/*.json"))
        if len(files) != expect:
            sys.exit(f"ABORT: {len(files)} runs in {root}, expected {expect}")
        extra = [f for f in files if f not in used]
        if extra:
            sys.exit(f"ABORT: {len(extra)} runs in {root} belong to no decision, e.g. {extra[0]}")
        lines += [f"{root.name}/{f.relative_to(root)}\t{rc.md5(f)}" for f in files]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def ref_row(method: str, param: str, ds: list[rc.Decision], fn) -> dict:
    res = [fn(d) for d in ds]
    regs = [d.regret(c) for d, (c, _) in zip(ds, res, strict=True)]
    lo, hi = rc.boot_ci(ds, regs)
    return {"benchmark": ds[0].bench, "method": method, "param": param, "decisions": len(ds),
            "chosen": len(ds), "compute_gpu_h": st.mean(s for _, s in res) / 3600,
            "regret": st.mean(regs), "ci_low": lo, "ci_high": hi,
            "zero_regret": sum(r == 0 for r in regs), "choices": [c for c, _ in res]}


def references(ds: list[rc.Decision], budgets: tuple[int, ...], geometric: bool) -> list[dict]:
    rows = [ref_row("early leader", f"epoch {k}", ds, lambda d, k=k: rc.early(d, k))
            for k in budgets]
    for eta in (2, 3):
        name = "successive halving" + ("" if eta == 2 else " (factor 3)")
        rows.append(ref_row(name, "rungs 1,2,3", ds,
                            lambda d, eta=eta: rc.halving(d, [1, 2, 3], eta)))
        if geometric:
            for r in (1, 2, 3):
                rows.append(ref_row(name, f"start {r}", ds, lambda d, r=r, eta=eta: rc.halving(
                    d, rc.geometric(r, eta, len(d.cands), EPOCHS), eta)))
    return rows


def check_references(old: list[rc.Decision]) -> None:
    want = {(r["method"], r["param"]): r for r in rc.evaluate("nlp", old)}
    for r in references(old, (1, 2, 3), geometric=False):
        got = {k: v for k, v in r.items() if k != "choices"}
        if got != want[(r["method"], r["param"])]:
            sys.exit(f"ABORT: reference {r['method']} {r['param']} differs from regret_compute")


def exact_norm_regret(ds: list[rc.Decision], choices: list[str], norm: Fraction) -> Fraction:
    regs = [stl.regret(d, c) for d, c in zip(ds, choices, strict=True)]
    return sum(regs, Fraction(0)) / len(regs) / norm


def paired_boot(ds: list[rc.Decision], fa, fb) -> tuple[tuple[float, float] | None, int]:
    by = collections.defaultdict(list)
    for d in ds:
        by[d.cluster].append(d)
    keys = sorted(by)
    rng = random.Random(0)
    diffs, empty = [], 0
    for _ in range(rc.BOOT):
        sample = [d for k in (rng.choice(keys) for _ in keys) for d in by[k]]
        a = [v for d in sample if (v := fa(d)) is not None]
        b = [v for d in sample if (v := fb(d)) is not None]
        if not a or not b:
            empty += 1
            continue
        diffs.append(st.mean(a) - st.mean(b))
    if not diffs:
        return None, empty
    diffs.sort()
    n = len(diffs)
    return (diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]), empty


def readout(entries: list[dict]) -> dict:
    best = min(entries, key=lambda e: (e["R"], e["compute"]))
    flat = None
    for i, e in enumerate(entries):
        later = [x["R"] for x in entries[i + 1:]]
        if not later or e["R"] - min(later) <= FLAT:
            flat = e
            break
    worse = [e for e in entries[entries.index(best) + 1:] if e["R"] > best["R"]]
    return {"best": best, "flat": flat, "worse": worse}


def selective_value(d: rc.Decision, k: int, norm: Fraction) -> float | None:
    if not stl.is_trusted(d, k, Fraction(0)):
        return None
    return float(stl.regret(d, d.pick) / norm)


def selective_entries(ds: list[rc.Decision], norm: Fraction) -> list[dict]:
    out = []
    for k in (0, *BUDGETS):
        p = stl.point(ds, k, None if k == 0 else Fraction(0), norm)
        if p["trusted"] == 0:
            continue
        out.append({"label": "0 (trust all)" if k == 0 else f"epoch {k}", "k": k,
                    "compute": stl.compute_gpu_h(ds, k), "R": p["norm_regret"],
                    "trusted": p["trusted"],
                    "value": lambda d, k=k: selective_value(d, k, norm)})
    return out


def ref_entries(ds: list[rc.Decision], rows: list[dict], method: str,
                norm: Fraction) -> list[dict]:
    sel = sorted((r for r in rows if r["method"] == method),
                 key=lambda r: (r["compute_gpu_h"], r["param"]))
    out = []
    for r in sel:
        ch = dict(zip((d.key for d in ds), r["choices"], strict=True))
        out.append({"label": r["param"], "compute": r["compute_gpu_h"],
                    "R": exact_norm_regret(ds, r["choices"], norm),
                    "value": lambda d, ch=ch: float(stl.regret(d, ch[d.key]) / norm)})
    return out


def readout_lines(name: str, ds: list[rc.Decision], entries: list[dict]) -> list[str]:
    ro = readout(entries)
    b, f = ro["best"], ro["flat"]
    head = (f"- {name}: lowest mean normalized regret (k*) at {b['label']}, "
            f"{stl.dec(b['R'], 3)} at {b['compute']:.3f} GPU-h; flattening at {f['label']} "
            f"({stl.dec(f['R'], 3)}); ")
    if not ro["worse"]:
        return [head + "no larger budget has a higher mean normalized regret than k*."]
    lines = [head + "larger budgets with a higher mean normalized regret than k*:"]
    for e in ro["worse"]:
        ci, empty = paired_boot(ds, e["value"], b["value"])
        txt = "n/a" if ci is None else f"[{stl.dec(ci[0], 3)}, {stl.dec(ci[1], 3)}]"
        note = f", {empty} resamples without a trusted decision left out" if empty else ""
        lines.append(f"  - {e['label']}: {stl.dec(e['R'], 3)}, difference "
                     f"{stl.dec(e['R'] - b['R'], 3)} {txt}{note}")
    return lines


def budget_label(k: int) -> str:
    return "0 (trust all)" if k == 0 else f"epoch {k}"


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
    ap.add_argument("--runs-per-group", type=int, default=RUNS_PER_GROUP)
    ap.add_argument("--no-pin", action="store_true", help="tests only: skip the md5 pins")
    a = ap.parse_args(argv)
    files = [Path(a.trajectories), Path(a.pairs), *map(Path, a.task2), *map(Path, a.nlp_heads),
             *map(Path, a.nlp_scores), Path(a.qt_scores), Path(a.qt_extract_log)]
    inputs = stl.pin(files, stl.PINS, a.no_pin)
    pins = {**stl.DOCS, **DESIGN}
    docs = stl.pin([_HERE.parent / "docs" / n for n in pins], pins, a.no_pin)
    old_dirs, new_dirs = [Path(p) for p in a.nlp_early], [Path(p) for p in a.nlp_early_new]

    sb = rc.studentbert(Path(a.trajectories), Path(a.pairs), [Path(p) for p in a.task2])
    old = rc.nlp([Path(p) for p in a.nlp_heads], old_dirs, [Path(p) for p in a.nlp_scores])
    qt, _ = rc.quicktune(Path(a.quicktune), Path(a.qt_scores), Path(a.qt_extract_log))
    for b, ds in (("studentbert", sb), ("nlp", old), ("quicktune", qt)):
        stl.check_t0(b, ds, stl.normalizer(ds), rc.evaluate(b, ds))
    check_references(old)
    nl, comps = extend(old, new_dirs, old_dirs, EPOCHS)
    man = manifest(new_dirs, {c["file"] for c in comps}, a.runs_per_group)
    norm = stl.normalizer(nl)
    if norm != stl.normalizer(old):
        sys.exit("ABORT: the NLP normalizer changed with the new curves")
    rows = references(nl, BUDGETS, geometric=True)
    cache: dict = {}

    L = ["# NLP selective trust at 1, 2, 3, 5 and 10 epochs: coverage, regret and compute",
         "",
         "Evidence classes: the runs, rules and read-outs are fixed in "
         "docs/nlp_budget_extension_design.md (commit 29652a2), committed before any run; the "
         "results up to epoch 3 were known before (RESULTS.md 12.27 and 12.28); epochs 4 to 10, "
         "the budgets 5 and 10, the flattening and worsening read-outs and the transfer test at 5 "
         "and 10 are new; all read-outs are descriptive. Every NLP budget comes from the new "
         "10-epoch runs (one campaign). Rules as in RESULTS.md 12.28 (analysis/selective_trust.py, "
         "imported): trust when the normalized lead L >= t, t from -1.00 to 1.00 in steps of 0.05, "
         "t = 0 the common fixed rule, normalized regret over a random candidate's mean regret, "
         "targets 0.10 and 0.50, 95% cluster bootstrap over the 5 tasks (10,000 resamples, "
         "random.Random(0)). The script was not run on real data before this run; its tests use "
         "synthetic decisions.",
         "",
         f"Inputs: {', '.join(inputs)}. Design files: {', '.join(docs)}. Code: "
         + ", ".join(f"{p.name} {m}" for p, m in _CODE.items())
         + f", regret_compute.py {stl.RC_MD5}. New runs: {len(comps)} (manifest md5 {man}).", "",
         "Checks passed: the recorded NLP, StudentBERT and Quick-Tune decisions reproduce every "
         "RESULTS.md 12.27 \"warning, abstain\" row; the reference rows reproduce the 12.27 early "
         "leader and successive-halving rows on the recorded runs; every new run has epochs 1 to "
         "10 and the data and settings of its recorded run; the normalizer is unchanged.", ""]

    L += ["## The new runs against the recorded 3-epoch runs", "",
          "| seed group | runs | epochs 1 to 3 identical | largest validation-accuracy "
          "difference | GPU-hours (10 epochs) |", "|---|---|---|---|---|"]
    for g, name in enumerate(("42, 1, 2", "3, 4, 5", "6, 7, 8")):
        cs = [c for c in comps if c["group"] == g]
        L.append(f"| {name} | {len(cs)} | {sum(c['identical'] for c in cs)} | "
                 f"{max(c['max_diff'] for c in cs):.4f} | "
                 f"{sum(c['run_s'] for c in cs) / 3600:.2f} |")
    L += ["", "t = 0 at 1, 2 and 3 epochs, new runs beside RESULTS.md 12.28 (trusted decisions, "
          "mean selective regret in points):", "",
          "| budget | new runs | RESULTS.md 12.28 |", "|---|---|---|"]
    for k in (1, 2, 3):
        p = stl.point(nl, k, Fraction(0), norm)
        was = stl.EXPECTED_T0["nlp"][k]
        L.append(f"| epoch {k} | {p['trusted']}, {stl.dec(p['regret'], ND)} | "
                 f"{was[0]}, {was[1]} |")
    L.append("")

    L += ["## 1. Per budget", "",
          f"{len(nl)} decisions. Mean regret of a random candidate (the normalizer): "
          f"{stl.dec(norm, ND + 2)} points. At t = 0, with the area under the normalized "
          "selective-regret curve over all thresholds, the highest-coverage threshold that meets "
          "normalized selective regret <= 0.10 at that budget, and the references at the nearest "
          "compute (method setting: mean regret in points at GPU-h).", "",
          "| budget | compute per decision (GPU-h) | trusted, coverage [95% CI] | selective "
          "regret [95% CI] | normalized [95% CI] | area (from coverage) | best threshold "
          "[band]: trusted, coverage [95% CI]; normalized [95% CI] | early leader | successive "
          "halving | successive halving (factor 3) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    grid = []
    for k in (0, *BUDGETS):
        ts = (None,) if k == 0 else stl.THRESHOLDS
        pts = [stl.point(nl, k, t, norm) for t in ts]
        comp = stl.compute_gpu_h(nl, k)
        for t, p in zip(ts, pts, strict=True):
            grid.append({"benchmark": "nlp", "budget": k, "threshold": stl.thr(t), **{
                f: p[f] for f in ("decisions", "trusted", "coverage", "regret", "norm_regret",
                                  "zero_regret")}, "compute_gpu_h": comp})
        t0 = None if k == 0 else Fraction(0)
        p = stl.point(nl, k, t0, norm)
        cov, raw, nrm = stl.intervals(cache, "nlp", nl, k, t0, norm)
        ar, cmin = (None, None) if k == 0 else stl.area(pts)
        best = "n/a"
        if k > 0:
            c = stl.choose([(nl, norm)], k)
            if c is None:
                best = "none"
            else:
                pb = stl.point(nl, k, c["t"], norm)
                cb = stl.intervals(cache, "nlp", nl, k, c["t"], norm)
                best = (f"{stl.thr(c['t'])} [{stl.thr(c['t_low'])}, {stl.thr(c['t'])}]: "
                        f"{stl.cov_text(pb, cb[0])}; {stl.ci_text(pb['norm_regret'], cb[2], 3)}")
        refs = []
        for m in stl.REFERENCES:
            r = stl.nearest(rows, m, comp)
            refs.append(f"{r['param']}: {stl.dec(r['regret'], ND)} at {r['compute_gpu_h']:.3f}")
        area_txt = "n/a" if ar is None else f"{stl.dec(ar, 3)} (from {stl.dec(cmin, 3)})"
        L.append(f"| {budget_label(k)} | {comp:.3f} | {stl.cov_text(p, cov)} | "
                 f"{stl.ci_text(p['regret'], raw, ND)} | {stl.ci_text(p['norm_regret'], nrm, 3)} "
                 f"| {area_txt} | {best} | {' | '.join(refs)} |")
    L.append("")

    L += ["## 2. The domain-specific budget for NLP (k = 0, 1, 2, 3, 5, 10)", "",
          "The smallest budget (k = 0, trusting every decision, checked first) at which some "
          "threshold reaches normalized selective regret at most 0.10 with coverage at least "
          "0.50, with the highest-coverage such threshold (ties to the largest t).", ""]
    br = stl.budget_rule(nl, norm, BUDGETS)
    if br is None:
        L.append("Not reached at any budget up to 10 epochs.")
    else:
        p = stl.point(nl, br["k"], br["t"], norm)
        ci = stl.intervals(cache, "nlp", nl, br["k"], br["t"], norm)
        band = "n/a" if br["t"] is None else \
            f"{stl.thr(br['t'])} [{stl.thr(br['t_low'])}, {stl.thr(br['t'])}]"
        L.append(f"Reached at {budget_label(br['k'])}, threshold {band}: "
                 f"{stl.cov_text(p, ci[0])}, normalized "
                 f"{stl.ci_text(p['norm_regret'], ci[2], 3)}, "
                 f"{stl.compute_gpu_h(nl, br['k']):.3f} GPU-h per decision.")
    L.append("")

    sel = selective_entries(nl, norm)
    fams = {"selective trust at t = 0 (normalized selective regret over the trusted decisions)":
            sel,
            **{f"{m} (normalized regret over all {len(nl)} decisions)":
               ref_entries(nl, rows, m, norm) for m in stl.REFERENCES}}
    L += ["## 3. Does more budget help", "",
          "k* is the budget with the lowest mean normalized regret (ties to the lower compute). "
          "Flattening is the smallest budget k after which no larger budget lowers the mean "
          f"normalized regret by more than {stl.dec(FLAT, 2)}. Worsening lists every budget larger "
          "than k* with a higher mean normalized regret, with a 95% paired cluster-bootstrap "
          "interval of R(k) minus R(k*) on the same resampled tasks. For successive halving the "
          "configurations are ordered by compute per decision. Coverage at t = 0: "
          + ", ".join(f"{e['label']} {e['trusted']}/{len(nl)}" for e in sel) + ".", ""]
    for name, ents in fams.items():
        L += readout_lines(name, nl, ents)
    L.append("")

    L += ["## 4. The best budget against the references at similar compute", "",
          "| method | k* | compute (GPU-h) | normalized regret | early leader | successive "
          "halving | successive halving (factor 3) |", "|---|---|---|---|---|---|---|"]
    for name, ents in fams.items():
        b = readout(ents)["best"]
        cells = []
        for m in stl.REFERENCES:
            rr = stl.nearest(rows, m, b["compute"])
            lo, hi = Fraction(rr["ci_low"]) / norm, Fraction(rr["ci_high"]) / norm
            cells.append(f"{rr['param']}: {stl.dec(exact_norm_regret(nl, rr['choices'], norm), 3)}"
                         f" [{stl.dec(lo, 3)}, {stl.dec(hi, 3)}] at {rr['compute_gpu_h']:.3f}")
        L.append(f"| {name.split(' (')[0]} | {b['label']} | {b['compute']:.3f} | "
                 f"{stl.dec(b['R'], 3)} | {' | '.join(cells)} |")
    L += ["", "Selective regret is over the trusted decisions only; the references are over all "
          f"{len(nl)} decisions.", ""]

    domains = {"studentbert": sb, "nlp": nl, "quicktune": qt}
    norms = {b: stl.normalizer(ds) for b, ds in domains.items()}
    L += ["## 5. Secondary: leave-one-domain-out threshold transfer at k = 5 and 10", "",
          "As in RESULTS.md 12.28: the threshold is chosen on the two other domains and applied "
          "to the held-out one, beside its own best threshold. StudentBERT and Quick-Tune as in "
          "12.28, NLP from the 10-epoch runs. \"none\": no threshold qualifies.", "",
          "| budget | held out | rule | source threshold [band] | source coverage, normalized "
          "regret | held-out trusted, coverage [95% CI] | held-out normalized regret [95% CI] | "
          "own best threshold [band] | own trusted, coverage [95% CI] | own normalized regret "
          "[95% CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in TRANSFER_BUDGETS:
        for held, hds in domains.items():
            srcs = [(ds, norms[b]) for b, ds in domains.items() if b != held]
            own = stl.choose([(hds, norms[held])], k)
            if own is None:
                own_cells = ["none", "n/a", "n/a"]
            else:
                po = stl.point(hds, k, own["t"], norms[held])
                ci = stl.intervals(cache, held, hds, k, own["t"], norms[held])
                own_cells = [f"{stl.thr(own['t'])} [{stl.thr(own['t_low'])}, "
                             f"{stl.thr(own['t'])}]", stl.cov_text(po, ci[0]),
                             stl.ci_text(po["norm_regret"], ci[2], 3)]
            for rule, w in (("primary (by decision)", "decision"),
                            ("sensitivity analysis (equal domain weight)", "domain")):
                c = stl.choose(srcs, k, w)
                if c is None:
                    cells = ["none", "n/a", "n/a", "n/a"]
                else:
                    ph = stl.point(hds, k, c["t"], norms[held])
                    ci = stl.intervals(cache, held, hds, k, c["t"], norms[held])
                    cells = [f"{stl.thr(c['t'])} [{stl.thr(c['t_low'])}, {stl.thr(c['t'])}]",
                             f"{stl.dec(c['coverage'], 3)}, {stl.dec(c['norm_regret'], 3)}",
                             stl.cov_text(ph, ci[0]), stl.ci_text(ph["norm_regret"], ci[2], 3)]
                L.append(f"| epoch {k} | {held} | {rule} | {' | '.join(cells)} | "
                         f"{' | '.join(own_cells)} |")
    L += ["", "Limitations: five NLP tasks make the intervals wide; the outcome is the published "
          "fully fine-tuned performance, not epoch 10 of these 10,000-example runs; epochs 1 to 3 "
          "can differ slightly from the recorded runs (GPU nondeterminism), so every budget comes "
          "from the new runs; the NLP result up to epoch 3 was known before this extension was "
          "designed.", ""]

    out = Path(a.out)
    tsv = out.with_name(out.name + ".tsv")
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grid[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for g in grid:
            w.writerow({k: ("" if v is None else repr(float(v)) if isinstance(v, Fraction | float)
                            else v) for k, v in g.items()})
    rtsv = out.with_name(out.name + "_references.tsv")
    with rtsv.open("w", newline="") as fh:
        keys = [k for k in rows[0] if k != "choices"]
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: repr(row[k]) if isinstance(row[k], float) else row[k] for k in keys})
    L.append(f"Files: {tsv.name} {rc.md5(tsv)}, {rtsv.name} {rc.md5(rtsv)}")
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("checks: 12.27 rows reproduced, references reproduced, 10-epoch runs match their "
          "recorded runs, normalizer unchanged")
    print(f"wrote {tsv.name}, {rtsv.name} and {rp.name} md5 {rc.md5(rp)}")


if __name__ == "__main__":
    main()
