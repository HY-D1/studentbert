from __future__ import annotations

"""Selective trust in the static H-score recommendation (docs/selective_trust_design.md with
amendments 1 and 2) on StudentBERT, the NLP benchmark and Quick-Tune micro: coverage, selective
regret and compute over thresholds of the normalized lead, the area under the selective-regret
curve, the leave-one-domain-out threshold transfer (primary and equal-domain-weight sensitivity),
the domain-specific budget and the multi-fidelity references of RESULTS.md 12.27. Decisions,
curves and costs come from analysis/regret_compute.py, imported and pinned by md5; no model is run.

  PYTHONPATH=. python analysis/selective_trust.py --out benchmark_final/selective_trust \
      <the input arguments of analysis/regret_compute.py>
"""

import argparse
import csv
import functools
import hashlib
import statistics as st
import sys
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

RC_MD5 = "fd4288700ffd9af74fe92911371935ae"
_RC = Path(__file__).resolve().with_name("regret_compute.py")
# regret_compute.py is cited by md5 in RESULTS.md 12.27; the loaders are reused only in that version.
if hashlib.md5(_RC.read_bytes()).hexdigest() != RC_MD5:
    sys.exit(f"ABORT: {_RC} md5 differs from {RC_MD5}, the version cited in RESULTS.md 12.27")

from analysis import regret_compute as rc  # noqa: E402

# Every input of the 12.27 run, as listed in its report.
PINS = {"trajectories.tsv": "e93329cbc69ae8a3af843c544d43793b",
        "reversal_pairs.tsv": "8f57558d142869b3e34a35ad18d095f7",
        "tg1_task2_kt_algebra2005.jsonl": "268ff75c4c59a23b2fa4d9bcc85b60aa",
        "tg1_task2_kt_algebra2006.jsonl": "d0f758940ee42bfdd8275e00ff12fe49",
        "tg1_task2_kt_assist2009.jsonl": "5823dd2999f96921409b9cce5272d924",
        "tg1_task2_kt_assist2017.jsonl": "f8a29e8c85e022e3306350ee159e31ff",
        "tg1_task2_kt_bridge2006.jsonl": "38235d16b5d18d72d97bb9d8b9a9a0e0",
        "tg1_task2_kt_ednet.jsonl": "59096f8a8941896c18ca60515ffdd9f2",
        "tg1_task2_kt_junyi.jsonl": "bc1af53d5d042f25d57057952795dad0",
        "nlp_head_pairs.tsv": "d023dd9346eb1020afa11d1dacbaf579",
        "p2_head_pairs.tsv": "9123ad27c17496af7f5f51cdb1987bc1",
        "p3_head_pairs.tsv": "9bad0afce43c426c8105473a6b8f918c",
        "task4_scores.jsonl": "9dfe5d930681ec04d79cb52c149bbc4e",
        "task4_scores_fresh.jsonl": "20c403eb43ade91bac987af41538299e",
        "task4_scores_p3.jsonl": "34ef47d06f1e7d155c585547a51f9220",
        "qt_scores.jsonl": "b7f48f2f5c6ee9c45ce17201a6d2f9ea",
        "extract_log.jsonl": "50ea8a9ba8fd20713c25211fd30c6f8e"}
# The run follows exactly these versions of the design and its amendments.
DOCS = {"selective_trust_design.md": "f63bfdc6bc0df97a97fb2fc97ef0eddd",
        "selective_trust_amendment_1.md": "de8a55b63b442dd2d1f28bb7bb2c1c9e",
        "selective_trust_amendment_2.md": "e8c9ae36be77cfb3fd887b1a6f0a82f2"}

THRESHOLDS = tuple(Fraction(i, 20) for i in range(-20, 21))
REGRET_TARGET, COVERAGE_TARGET = Fraction(1, 10), Fraction(1, 2)
SHARED_BUDGETS = (1, 2, 3)
FULL_SHARE = ("studentbert", "quicktune")
DIGITS = {"studentbert": 4, "nlp": 2, "quicktune": 2}
UNITS = {"studentbert": "test AUC", "nlp": "points", "quicktune": "top-1 points"}
REFERENCES = ("early leader", "successive halving", "successive halving (factor 3)")
# RESULTS.md 12.27 "warning, abstain" rows: budget -> (decisions followed, mean regret as printed).
EXPECTED_T0 = {
    "studentbert": {1: (17, "0.0009"), 2: (12, "0.0003"), 3: (12, "0.0009"), 5: (10, "0.0010"),
                    10: (5, "0.0007"), 20: (10, "0.0009")},
    "nlp": {1: (25, "0.70"), 2: (20, "0.90"), 3: (24, "0.50")},
    "quicktune": {1: (3, "16.56"), 2: (0, "nan"), 3: (0, "nan"), 5: (3, "0.00"),
                  10: (6, "0.00"), 20: (6, "0.00"), 30: (6, "0.00"), 40: (6, "0.00"),
                  50: (6, "0.00")}}


def pin(paths: list[Path], pins: dict[str, str], skip: bool) -> list[str]:
    names = sorted(p.name for p in paths)
    if not skip and names != sorted(pins):
        sys.exit(f"ABORT: files {names} differ from the pinned set {sorted(pins)}")
    out = []
    for p in paths:
        h = rc.md5(p)
        if not skip and h != pins[p.name]:
            sys.exit(f"ABORT: {p} md5 {h}, expected {pins[p.name]}")
        out.append(f"{p.name} {h}")
    return out


@functools.cache
def lead(d: rc.Decision, k: int) -> Fraction:
    """Normalized lead of the static choice after k epochs, exact; 0 when every score is equal."""
    v = {c: Fraction(d.val(c, k)) for c in d.cands}
    spread = max(v.values()) - min(v.values())
    if spread == 0:
        return Fraction(0)
    return (v[d.pick] - max(x for c, x in v.items() if c != d.pick)) / spread


def regret(d: rc.Decision, c: str) -> Fraction:
    return Fraction(d.best) - Fraction(d.outcome[c])


def is_trusted(d: rc.Decision, k: int, t: Fraction | None) -> bool:
    # k = 0: no fine-tuning, every decision is trusted (design, Budgets).
    return k == 0 or lead(d, k) >= t


def normalizer(ds: list[rc.Decision]) -> Fraction:
    """Mean over decisions of the mean regret over candidates (a random candidate's regret)."""
    n = sum((sum((regret(d, c) for c in d.cands), Fraction(0)) / len(d.cands) for d in ds),
            Fraction(0)) / len(ds)
    if n == 0:
        sys.exit(f"ABORT: {ds[0].bench}: a random candidate has zero mean regret")
    return n


def point(ds: list[rc.Decision], k: int, t: Fraction | None, norm: Fraction) -> dict:
    tr = [d for d in ds if is_trusted(d, k, t)]
    regs = [regret(d, d.pick) for d in tr]
    sel = sum(regs, Fraction(0)) / len(regs) if regs else None
    return {"decisions": len(ds), "trusted": len(tr), "coverage": Fraction(len(tr), len(ds)),
            "regret": sel, "norm_regret": None if sel is None else sel / norm,
            "zero_regret": sum(r == 0 for r in regs)}


def compute_gpu_h(ds: list[rc.Decision], k: int) -> float:
    # Frozen scoring plus k epochs of every candidate; static_axis keeps Quick-Tune's scoring,
    # timed on other hardware, off the fine-tuning axis (as in 12.27).
    return st.mean(d.static_axis if k == 0 else rc.abstain(d, k)[1] for d in ds) / 3600


def full_gpu_h(ds: list[rc.Decision]) -> float:
    return st.mean(d.ft_cost({c: d.epochs for c in d.cands}) for d in ds) / 3600


def area(points: list[dict]) -> tuple[Fraction | None, Fraction | None]:
    """Trapezoid area under normalized selective regret against coverage (amendment 1, point 1),
    over the distinct points with coverage above zero; returns (area, smallest coverage)."""
    by: dict[Fraction, Fraction] = {}
    for p in points:
        if p["trusted"] == 0:
            continue
        if by.setdefault(p["coverage"], p["norm_regret"]) != p["norm_regret"]:
            sys.exit("ABORT: two thresholds with equal coverage and different selective regret")
    if not by:
        return None, None
    cs = sorted(by)
    a = sum(((c2 - c1) * (by[c1] + by[c2]) / 2 for c1, c2 in zip(cs, cs[1:], strict=False)), Fraction(0))
    return a, cs[0]


def choose(sources: list[tuple[list[rc.Decision], Fraction]], k: int,
           weighting: str = "decision") -> dict | None:
    """Highest-coverage threshold whose pooled normalized selective regret is at most the target.

    weighting "decision" pools decisions (amendment 1, point 3); "domain" gives each source equal
    weight and needs coverage above zero in every source (amendment 2). Ties go to the largest t
    of the band (amendment 1, point 2); None when no threshold qualifies."""
    qual = []
    for t in THRESHOLDS:
        ps = [point(ds, k, t, norm) for ds, norm in sources]
        if weighting == "decision":
            n_tr = sum(p["trusted"] for p in ps)
            if n_tr == 0:
                continue
            cov = Fraction(n_tr, sum(p["decisions"] for p in ps))
            reg = sum((p["norm_regret"] * p["trusted"] for p in ps if p["trusted"]),
                      Fraction(0)) / n_tr
        elif weighting == "domain":
            if any(p["trusted"] == 0 for p in ps):
                continue
            cov = sum((p["coverage"] for p in ps), Fraction(0)) / len(ps)
            reg = sum((p["norm_regret"] for p in ps), Fraction(0)) / len(ps)
        else:
            sys.exit(f"ABORT: unknown weighting {weighting!r}")
        if reg <= REGRET_TARGET:
            qual.append((t, cov, reg))
    if not qual:
        return None
    top = max(c for _, c, _ in qual)
    band = [(t, r) for t, c, r in qual if c == top]
    return {"t": band[-1][0], "t_low": band[0][0], "coverage": top, "norm_regret": band[-1][1]}


def budget_rule(ds: list[rc.Decision], norm: Fraction, budgets: tuple[int, ...]) -> dict | None:
    """Smallest budget (k = 0 first) where some threshold reaches the regret target with coverage
    at least the coverage target (design item 4, amendment 1 point 4); None when not reached."""
    if point(ds, 0, None, norm)["norm_regret"] <= REGRET_TARGET:
        return {"k": 0, "t": None, "t_low": None}
    for k in budgets:
        c = choose([(ds, norm)], k)
        if c is not None and c["coverage"] >= COVERAGE_TARGET:
            return {"k": k, "t": c["t"], "t_low": c["t_low"]}
    return None


def nearest(rows: list[dict], method: str, x: float) -> dict:
    """The row of a method whose compute is nearest x, ties to the lower compute (amendment 1,
    point 6)."""
    return min((r for r in rows if r["method"] == method),
               key=lambda r: (abs(Fraction(r["compute_gpu_h"]) - Fraction(x)),
                              r["compute_gpu_h"]))


def check_t0(bench: str, ds: list[rc.Decision], norm: Fraction, rc_rows: list[dict]) -> None:
    """t = 0 must be the 12.27 warning rule: same trusted decisions as regret_compute.abstain and
    the recorded rows, through both the 12.27 float path and the exact path."""
    got_rc = {r["param"]: (r["chosen"], f"{r['regret']:.{DIGITS[bench]}f}") for r in rc_rows
              if r["method"] == "warning, abstain"}
    for k in rc.BUDGETS[bench]:
        mine = [is_trusted(d, k, Fraction(0)) for d in ds]
        if mine != [rc.abstain(d, k)[0] is not None for d in ds]:
            sys.exit(f"ABORT: {bench} epoch {k}: t = 0 trusts other decisions than the warning rule")
        regs = [d.regret(d.pick) for d, m in zip(ds, mine, strict=True) if m]
        float_path = (sum(mine), f"{st.mean(regs):.{DIGITS[bench]}f}" if regs else "nan")
        p = point(ds, k, Fraction(0), norm)
        exact_path = (p["trusted"], dec(p["regret"], DIGITS[bench]) if p["trusted"] else "nan")
        want = EXPECTED_T0[bench][k]
        if not float_path == exact_path == got_rc[f"epoch {k}"] == want:
            sys.exit(f"ABORT: {bench} epoch {k}: t = 0 gives {float_path} (exact {exact_path}, "
                     f"regret_compute {got_rc[f'epoch {k}']}), RESULTS.md 12.27 has {want}")


def dec(x: Fraction | float | None, nd: int) -> str:
    """Round once, ties half away from zero."""
    if x is None:
        return "n/a"
    if isinstance(x, float) and x != x:
        return "nan"
    x = Fraction(x)
    with localcontext() as ctx:
        ctx.prec = 60
        v = Decimal(x.numerator) / Decimal(x.denominator)
    return str(v.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def intervals(cache: dict, bench: str, ds: list[rc.Decision], k: int, t: Fraction | None,
              norm: Fraction) -> tuple:
    """95% cluster bootstrap (amendment 1, point 5): coverage over all decisions, selective regret
    over the trusted ones, raw and normalized with the normalizer fixed."""
    key = (bench, k, None if k == 0 else t)
    if key not in cache:
        flags = [is_trusted(d, k, t) for d in ds]
        cov = rc.boot_ci(ds, [1.0 if f else 0.0 for f in flags])
        tr = [d for d, f in zip(ds, flags, strict=True) if f]
        if tr:
            raw = rc.boot_ci(tr, [float(regret(d, d.pick)) for d in tr])
            nrm = rc.boot_ci(tr, [float(regret(d, d.pick) / norm) for d in tr])
        else:
            raw = nrm = None
        cache[key] = (cov, raw, nrm)
    return cache[key]


def ci_text(x: Fraction | None, ci: tuple | None, nd: int) -> str:
    if x is None or ci is None:
        return "n/a"
    return f"{dec(x, nd)} [{dec(ci[0], nd)}, {dec(ci[1], nd)}]"


def cov_text(p: dict, ci: tuple) -> str:
    return f"{p['trusted']}/{p['decisions']}, {ci_text(p['coverage'], ci, 3)}"


def thr(t: Fraction | None) -> str:
    return "n/a" if t is None else dec(t, 2)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--task2", nargs="+", required=True)
    ap.add_argument("--nlp-heads", nargs=3, required=True)
    ap.add_argument("--nlp-early", nargs=3, required=True)
    ap.add_argument("--nlp-scores", nargs=3, required=True)
    ap.add_argument("--quicktune", required=True)
    ap.add_argument("--qt-scores", required=True)
    ap.add_argument("--qt-extract-log", required=True)
    ap.add_argument("--no-pin", action="store_true", help="tests only: skip the md5 pins")
    a = ap.parse_args(argv)
    files = [Path(a.trajectories), Path(a.pairs), *map(Path, a.task2), *map(Path, a.nlp_heads),
             *map(Path, a.nlp_scores), Path(a.qt_scores), Path(a.qt_extract_log)]
    inputs = pin(files, PINS, a.no_pin)
    docs = pin([_RC.parent.parent / "docs" / n for n in DOCS], DOCS, a.no_pin)
    sb = rc.studentbert(Path(a.trajectories), Path(a.pairs), [Path(p) for p in a.task2])
    nl = rc.nlp([Path(p) for p in a.nlp_heads], [Path(p) for p in a.nlp_early],
                [Path(p) for p in a.nlp_scores])
    qt, _ = rc.quicktune(Path(a.quicktune), Path(a.qt_scores), Path(a.qt_extract_log))
    domains = {"studentbert": sb, "nlp": nl, "quicktune": qt}
    norms = {b: normalizer(ds) for b, ds in domains.items()}
    rc_rows = {b: rc.evaluate(b, ds) for b, ds in domains.items()}
    for b, ds in domains.items():
        check_t0(b, ds, norms[b], rc_rows[b])
    cache: dict = {}

    L = ["# Selective trust in the static recommendation: coverage, selective regret and compute",
         "",
         "Evidence classes: the rule at t = 0 (trust the static choice unless another candidate is "
         "strictly ahead) was known before (RESULTS.md 12.17, 12.20 and 12.27) and is reproduced "
         "here exactly, decision by decision and in every 12.27 \"warning, abstain\" row. The "
         "threshold family, the normalized units, the area, the leave-one-domain-out transfer and "
         "its equal-domain-weight sensitivity analysis, the budget question and the reference "
         "points are pre-specified in docs/selective_trust_design.md (commit c8ca7dd), amendment 1 "
         "(commit 882de48) and amendment 2, committed before this script. The script was not run on real data "
         "before this run (the inputs are on the cluster only); its tests use synthetic decisions. "
         "Static choice: H-score (shrunk H-score on Quick-Tune). Signal: the normalized lead L "
         "(the static choice's validation score minus the best other candidate's, over the spread "
         "of all candidates at epoch k; 0 when the spread is 0); trust when L >= t, t from -1.00 to "
         "1.00 in steps of 0.05. Normalized regret: regret over the domain's mean regret of a "
         "random candidate. Targets: normalized selective regret at most 0.10, coverage at least "
         "0.50. Intervals: 95% cluster bootstrap over targets or tasks, 10,000 resamples, "
         "random.Random(0).",
         "",
         f"Inputs: {', '.join(inputs)}. Design files: {', '.join(docs)}. "
         f"regret_compute.py {RC_MD5}.", ""]
    grid = []
    for b, ds in domains.items():
        nd, norm = DIGITS[b], norms[b]
        full = full_gpu_h(ds) if b in FULL_SHARE else None
        L += [f"## {b} ({UNITS[b]})", "",
              f"{len(ds)} decisions. Mean regret of a random candidate (the normalizer): "
              f"{dec(norm, nd + 2)} {UNITS[b]}."
              + (f" Fine-tuning every candidate fully: {full:.3f} GPU-h per decision." if full
                 else " NLP curves stop at epoch 3, so there is no full-compute share."), "",
              "At t = 0 (the common fixed rule) at each budget, with the area under the normalized "
              "selective-regret curve over all thresholds and the multi-fidelity references of "
              "12.27 at the nearest compute (method setting: mean regret at GPU-h).", "",
              "| budget | compute per decision (GPU-h) | share of full compute | trusted, coverage "
              "[95% CI] | selective regret [95% CI] | normalized [95% CI] | area (from coverage) "
              "| early leader | successive halving | successive halving (factor 3) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k in (0, *rc.BUDGETS[b]):
            ts = (None,) if k == 0 else THRESHOLDS
            pts = [point(ds, k, t, norm) for t in ts]
            comp = compute_gpu_h(ds, k)
            for t, p in zip(ts, pts, strict=True):
                grid.append({"benchmark": b, "budget": k, "threshold": thr(t), **{
                    f: p[f] for f in ("decisions", "trusted", "coverage", "regret", "norm_regret",
                                      "zero_regret")}, "compute_gpu_h": comp})
            t0 = None if k == 0 else Fraction(0)
            p = point(ds, k, t0, norm)
            cov, raw, nrm = intervals(cache, b, ds, k, t0, norm)
            ar, cmin = (None, None) if k == 0 else area(pts)
            refs = []
            for m in REFERENCES:
                r = nearest(rc_rows[b], m, comp)
                refs.append(f"{r['param']}: {dec(r['regret'], nd)} at {r['compute_gpu_h']:.3f}")
            L.append(f"| {'0 (trust all)' if k == 0 else f'epoch {k}'} | {comp:.3f} | "
                     f"{f'{comp / full:.3f}' if full else 'n/a'} | {cov_text(p, cov)} | "
                     f"{ci_text(p['regret'], raw, nd)} | {ci_text(p['norm_regret'], nrm, 3)} | "
                     f"{'n/a' if ar is None else f'{dec(ar, 3)} (from {dec(cmin, 3)})'} | "
                     f"{' | '.join(refs)} |")
        L.append("")

    L += ["## Leave-one-domain-out threshold transfer (k = 1, 2, 3)", "",
          "The threshold is chosen on the two other domains (the highest pooled coverage with "
          "pooled normalized selective regret at most 0.10; ties to the largest t of the band) and "
          "applied to the held-out domain, beside the held-out domain's own best threshold under "
          "the same target. Primary rule: the source domains pooled by decision (amendment 1). "
          "Sensitivity analysis (amendment 2): each source domain weighted equally, coverage above "
          "zero required in both. \"none\": no threshold qualifies.", "",
          "| budget | held out | rule | source threshold [band] | source coverage, normalized "
          "regret | held-out trusted, coverage [95% CI] | held-out normalized regret [95% CI] | "
          "at the band's smallest t: coverage, normalized regret | own best threshold [band] | "
          "own trusted, coverage [95% CI] | own normalized regret [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in SHARED_BUDGETS:
        for held, hds in domains.items():
            srcs = [(ds, norms[b]) for b, ds in domains.items() if b != held]
            own = choose([(hds, norms[held])], k)
            if own is None:
                own_cells = ["none", "n/a", "n/a"]
            else:
                po = point(hds, k, own["t"], norms[held])
                ci = intervals(cache, held, hds, k, own["t"], norms[held])
                own_cells = [f"{thr(own['t'])} [{thr(own['t_low'])}, {thr(own['t'])}]",
                             cov_text(po, ci[0]), ci_text(po["norm_regret"], ci[2], 3)]
            for rule, w in (("primary (by decision)", "decision"),
                            ("sensitivity analysis (equal domain weight)", "domain")):
                c = choose(srcs, k, w)
                if c is None:
                    cells = ["none", "n/a", "n/a", "n/a", "n/a"]
                else:
                    ph = point(hds, k, c["t"], norms[held])
                    ci = intervals(cache, held, hds, k, c["t"], norms[held])
                    pl = point(hds, k, c["t_low"], norms[held])
                    cl = intervals(cache, held, hds, k, c["t_low"], norms[held])
                    cells = [f"{thr(c['t'])} [{thr(c['t_low'])}, {thr(c['t'])}]",
                             f"{dec(c['coverage'], 3)}, {dec(c['norm_regret'], 3)}",
                             cov_text(ph, ci[0]), ci_text(ph["norm_regret"], ci[2], 3),
                             f"{cov_text(pl, cl[0])}; {ci_text(pl['norm_regret'], cl[2], 3)}"]
                L.append(f"| epoch {k} | {held} | {rule} | {' | '.join(cells)} | "
                         f"{' | '.join(own_cells)} |")
    L.append("")

    L += ["## The domain-specific budget", "",
          "Per domain, the smallest budget (k = 0, trusting every decision, checked first) at which "
          "some threshold reaches normalized selective regret at most 0.10 with coverage at least "
          "0.50, with the highest-coverage such threshold (ties to the largest t).", "",
          "| domain | smallest budget | threshold [band] | trusted, coverage [95% CI] | normalized "
          "selective regret [95% CI] | compute per decision (GPU-h) | share of full compute |",
          "|---|---|---|---|---|---|---|"]
    for b, ds in domains.items():
        r = budget_rule(ds, norms[b], rc.BUDGETS[b])
        if r is None:
            L.append(f"| {b} | not reached | n/a | n/a | n/a | n/a | n/a |")
            continue
        p = point(ds, r["k"], r["t"], norms[b])
        ci = intervals(cache, b, ds, r["k"], r["t"], norms[b])
        comp = compute_gpu_h(ds, r["k"])
        band = "n/a" if r["t"] is None else f"{thr(r['t'])} [{thr(r['t_low'])}, {thr(r['t'])}]"
        share = f"{comp / full_gpu_h(ds):.3f}" if b in FULL_SHARE else "n/a"
        label = "0 (trust all)" if r["k"] == 0 else f"epoch {r['k']}"
        L.append(f"| {b} | {label} | {band} | "
                 f"{cov_text(p, ci[0])} | {ci_text(p['norm_regret'], ci[2], 3)} | {comp:.3f} | "
                 f"{share} |")
    L += ["", "Limitations: StudentBERT is retrospective; NLP stops at epoch 3; Quick-Tune's outcome "
          "shares its curves with the signal, so its selective regret at 50 epochs is zero by "
          "construction; the t = 0 results were known before. Quick-Tune's frozen scoring (H200) is "
          "off the fine-tuning compute axis, as in 12.27.", ""]

    out = Path(a.out)
    tsv = out.with_name(out.name + ".tsv")
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grid[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in grid:
            w.writerow({k: ("" if v is None else repr(float(v)) if isinstance(v, Fraction | float)
                            else v) for k, v in r.items()})
    L.append(f"Files: {tsv.name} {rc.md5(tsv)}")
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("reproduction: t = 0 equals every RESULTS.md 12.27 warning, abstain row")
    print(f"wrote {tsv.name} and {rp.name} md5 {rc.md5(rp)}")


if __name__ == "__main__":
    main()
