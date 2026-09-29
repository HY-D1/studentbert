from __future__ import annotations

# MRAP Task 4: check that configs/task4/bassignana2022_table2.tsv reproduces the paper's own
# reported correlations. The table was entered from the paper (no machine-readable release exists),
# so each (dataset, pooling, regime) setting is recomputed from its 7 rows: Pearson between LogME
# and mean performance, and weighted Kendall tau as scipy.stats.weightedtau computes it by default
# (the paper cites Vigna 2015; which weighting it used is not stated, so tau is a secondary check).
# A wrong digit in any LogME value or mean would move Pearson at the third decimal. The sd columns
# enter no reported statistic and cannot be checked this way.
#
#   python analysis/task4_gold_check.py

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "configs" / "task4"


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader((ln for ln in fh if not ln.startswith("#")), delimiter="\t"))


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


def weighted_tau(x: list[float], y: list[float]) -> float | None:
    try:
        from scipy.stats import weightedtau
    except ImportError:
        return None
    return float(weightedtau(x, y).statistic)


def check(table: Path, reported: Path, tol_rho: float, tol_tau: float) -> tuple[int, list[str]]:
    rows = read_tsv(table)
    rep = {(r["dataset"], r["pooling"], r["regime"]): r for r in read_tsv(reported)}
    lines, bad = [], 0
    for (ds, pool, regime), r in sorted(rep.items()):
        sel = [x for x in rows if x["dataset"] == ds and x["pooling"] == pool]
        if len(sel) != 7:
            sys.exit(f"ABORT: {ds} {pool} has {len(sel)} rows, expected 7")
        logme = [float(x["logme"]) for x in sel]
        perf = [float(x[f"{regime}_mean"]) for x in sel]
        rho, tau = pearson(logme, perf), weighted_tau(logme, perf)
        ok_rho = abs(rho - float(r["pearson"])) <= tol_rho
        ok_tau = tau is None or abs(tau - float(r["tau_w"])) <= tol_tau
        bad += not ok_rho
        tau_s = "n/a" if tau is None else f"{tau:+.3f} ({r['tau_w']}){'' if ok_tau else ' DIFF'}"
        lines.append(f"{ds:8s} {pool:4s} {regime:6s} pearson {rho:+.3f} ({r['pearson']})"
                     f"{'' if ok_rho else ' MISMATCH'}   tau_w {tau_s}")
    return bad, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(ROOT / "bassignana2022_table2.tsv"))
    ap.add_argument("--reported", default=str(ROOT / "bassignana2022_reported.tsv"))
    ap.add_argument("--tol-rho", type=float, default=0.0015,
                    help="the paper prints 3 dp; table rounding adds a little more")
    ap.add_argument("--tol-tau", type=float, default=0.0015)
    a = ap.parse_args(argv)
    bad, lines = check(Path(a.table), Path(a.reported), a.tol_rho, a.tol_tau)
    print("\n".join(lines))
    print(f"{len(lines)} settings; Pearson mismatches: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
