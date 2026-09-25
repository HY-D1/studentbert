from __future__ import annotations

# RESULTS.md 3.1 render-and-compare check (carried over since 2026-09-22).
# The collector refuses to overwrite RESULTS.md while hand-maintained sections exist, so 3.1
# is not regenerated in place. Instead every number it prints is recomputed here from the
# per-seed values in nextskill_results_long.csv (analysis/parse_nextskill_full.py) and
# compared cell by cell, the same way the benchmark builder cross-checks sections 2, 4 and 6.
#
# Arithmetic: Decimal from the CSV strings, rounded once, half away from zero, at 4 dp.
# The CSV holds 6 dp, so a mean can be off by at most 5e-7; a cell whose mean lies that close
# to a rounding boundary is reported as boundary-sensitive and accepted either way.
# The +- column is tested against both the population and the sample SD, because which one
# 3.1 prints was never recorded (parse_nextskill_full.py uses statistics.pstdev).
#
# Usage (standalone or through the collector):
#   python analysis/nextskill_31.py --long-csv nextskill_results_long.csv --results-md RESULTS.md
#   python analysis/collect_all_results.py --check-3-1 nextskill_results_long.csv

import argparse
import csv
import re
import sys
from decimal import ROUND_HALF_UP, Decimal, getcontext
from pathlib import Path

getcontext().prec = 40

DATASET = "assist2017"
CONDS = ("scratch", "indomain", "ednet", "junyi")
NS = (25, 50, 100, 200, 500, 1000)
SEEDS = (1, 2, 42)
METRICS = (
    ("top1", "Test top-1 accuracy"),
    ("top5", "Test top-5 accuracy"),
    ("macro_auc", "Test macro-OVR AUC"),
    ("weighted_auc", "Test weighted-OVR AUC"),
)
GAP_ROWS = tuple((c, m) for m in ("top1", "macro_auc") for c in ("indomain", "ednet", "junyi"))
GAP_LABEL = {"top1": "top-1", "macro_auc": "macro AUC"}
Q4 = Decimal("0.0001")
CSV_EPS = Decimal("0.0000005")


def q4(d: Decimal) -> Decimal:
    return d.quantize(Q4, rounding=ROUND_HALF_UP)


def boundary(d: Decimal) -> bool:
    """True when d lies within the 6 dp CSV error of a 4 dp half boundary."""
    frac = (abs(d) / Q4) % 1
    return abs(frac - Decimal("0.5")) * Q4 <= CSV_EPS


def load_long(path: str | Path, dataset: str = DATASET) -> dict:
    vals: dict = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        need = {"dataset", "condition", "N", "seed", "metric", "value"}
        if not need <= set(reader.fieldnames or []):
            sys.exit(f"ABORT {path}: expected columns {sorted(need)}, got {reader.fieldnames}")
        for row in reader:
            if row["dataset"] != dataset:
                continue
            key = (row["condition"], int(row["N"]), row["metric"])
            seed = int(row["seed"])
            cell = vals.setdefault(key, {})
            if seed in cell:
                sys.exit(f"ABORT {path}: duplicate row {key} seed {seed}")
            cell[seed] = Decimal(row["value"])
    return vals


def stats(cell: dict) -> dict:
    xs = [cell[s] for s in sorted(cell)]
    n = len(xs)
    mean = sum(xs) / n
    ss = sum((x - mean) ** 2 for x in xs)
    return {"n": n, "mean": mean, "seeds": sorted(cell),
            "pop": (ss / n).sqrt() if n else None,
            "sample": (ss / (n - 1)).sqrt() if n > 1 else None}


def render(vals: dict, sd: str = "pop") -> str:
    """Section 3.1 tables in RESULTS.md layout, from per-seed values."""
    out = []
    head = "| Condition | " + " | ".join(f"N={n}" for n in NS) + " |"
    rule = "|---|" + "---|" * len(NS)
    for metric, title in METRICS:
        out += [f"**{title} (mean ±std, n=3):**", "", head, rule]
        for c in CONDS:
            cells = []
            for n in NS:
                cell = vals.get((c, n, metric))
                if not cell:
                    cells.append("NOT FOUND")
                    continue
                s = stats(cell)
                cells.append(f"{q4(s['mean'])} ±{q4(s[sd])}")
            out.append(f"| {c} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["**Gap vs scratch (mean of per-seed means):**", "",
            "| Condition, metric | " + " | ".join(f"N={n}" for n in NS) + " |", rule]
    for c, metric in GAP_ROWS:
        cells = []
        for n in NS:
            a, b = vals.get((c, n, metric)), vals.get(("scratch", n, metric))
            if not a or not b:
                cells.append("NOT FOUND")
                continue
            g = q4(stats(a)["mean"] - stats(b)["mean"])
            cells.append(f"{'+' if g >= 0 else ''}{g}")
        out.append(f"| {c}, {GAP_LABEL[metric]} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def section_31(md_text: str) -> str:
    m = re.search(r"^### 3\.1 .*?(?=^### |^## |\Z)", md_text, flags=re.M | re.S)
    if not m:
        sys.exit("ABORT: RESULTS.md has no '### 3.1' section")
    return m.group(0)


def parse_31(sec: str) -> dict:
    """Displayed cells: ('mean'|'gap', cond, N, metric) -> (value, sd or None)."""
    shown: dict = {}
    current = None
    titles = {title: metric for metric, title in METRICS}
    for line in sec.splitlines():
        t = re.match(r"^\*\*(.+?) \(", line)
        if t:
            current = ("gap", None) if t.group(1).startswith("Gap vs scratch") else \
                ("mean", titles.get(t.group(1)))
            continue
        if not line.startswith("|") or current is None or line.startswith("|---"):
            continue
        cols = [x.strip() for x in line.strip().strip("|").split("|")]
        if cols[0] in ("Condition", "Condition, metric"):
            continue
        kind, metric = current
        if kind == "gap":
            cond, _, label = cols[0].partition(", ")
            metric = {v: k for k, v in GAP_LABEL.items()}.get(label)
        else:
            cond = cols[0]
        for n, cell in zip(NS, cols[1:]):
            mm = re.match(r"^([+-]?\d+\.\d+)(?:\s*±\s*(\d+\.\d+))?$", cell)
            shown[(kind, cond, n, metric)] = (
                (Decimal(mm.group(1)), Decimal(mm.group(2)) if mm.group(2) else None)
                if mm else (cell, None))
    return shown


def check(results_md: str | Path, long_csv: str | Path, parser_stdout: str | None = None,
          out: str | None = None) -> int:
    vals = load_long(long_csv)
    sec = section_31(Path(results_md).read_text(encoding="utf-8"))
    shown = parse_31(sec)
    problems, boundary_cells = [], []
    sd_hits = {"pop": 0, "sample": 0}
    n_mean = n_gap = 0

    for metric, _ in METRICS:
        for c in CONDS:
            for n in NS:
                key = ("mean", c, n, metric)
                cell = vals.get((c, n, metric))
                disp = shown.get(key)
                if disp is None:
                    problems.append(f"{c} N={n} {metric}: not printed in 3.1")
                    continue
                if not cell:
                    problems.append(f"{c} N={n} {metric}: NOT FOUND in CSV, 3.1 prints {disp[0]}")
                    continue
                n_mean += 1
                s = stats(cell)
                want = q4(s["mean"])
                if disp[0] != want:
                    if boundary(s["mean"]) and abs(disp[0] - want) == Q4:
                        boundary_cells.append(f"{c} N={n} {metric}")
                    else:
                        problems.append(f"{c} N={n} {metric}: 3.1 {disp[0]}, recomputed {want}")
                if disp[1] is not None:
                    for conv in ("pop", "sample"):
                        if s[conv] is not None and q4(s[conv]) == disp[1]:
                            sd_hits[conv] += 1
                    if s["n"] != 3:
                        problems.append(f"{c} N={n} {metric}: n={s['n']}, 3.1 says n=3")
                    if s["seeds"] != list(SEEDS):
                        problems.append(f"{c} N={n} {metric}: seeds {s['seeds']}")

    for c, metric in GAP_ROWS:
        for n in NS:
            disp = shown.get(("gap", c, n, metric))
            a, b = vals.get((c, n, metric)), vals.get(("scratch", n, metric))
            if disp is None or not a or not b:
                problems.append(f"gap {c} N={n} {metric}: missing (printed or CSV)")
                continue
            n_gap += 1
            diff = stats(a)["mean"] - stats(b)["mean"]
            want = q4(diff)
            if disp[0] != want:
                if boundary(diff) and abs(disp[0] - want) == Q4:
                    boundary_cells.append(f"gap {c} N={n} {metric}")
                else:
                    problems.append(f"gap {c} N={n} {metric}: 3.1 {disp[0]}, recomputed {want}")

    conv = max(sd_hits, key=lambda k: sd_hits[k])
    if sd_hits[conv] != n_mean:
        problems.append(f"SD: population matches {sd_hits['pop']}/{n_mean}, "
                        f"sample {sd_hits['sample']}/{n_mean}; no single convention fits")

    note = re.search(r"top-1 std (\d\.\d+)", sec)
    ed = vals.get(("ednet", 25, "top1"))
    if note and ed and q4(stats(ed)[conv]) != Decimal(note.group(1)):
        problems.append(f"note: ednet N=25 top-1 std {note.group(1)}, "
                        f"recomputed {q4(stats(ed)[conv])} ({conv})")

    keys = {(c, n, s) for (c, n, _m), cell in vals.items() for s in cell}
    mt1 = [k for k in vals if k[2] == "macro_top1"]
    if len(keys) != 72:
        problems.append(f"coverage: {len(keys)} unique (cond, N, seed), 3.1 says 72")
    if mt1 and "macro_top1 not logged" in sec:
        problems.append(f"coverage: CSV has {len(mt1)} macro_top1 cells, 3.1 says NOT FOUND")
    if parser_stdout:
        txt = Path(parser_stdout).read_text(errors="ignore")
        m = re.search(r"raw run records: (\d+)\s+unique \(ds,cond,N,seed\): (\d+)", txt)
        claim = re.search(r"(\d+) raw run records dedup to (\d+) unique", sec)
        if not m:
            problems.append("parser stdout: no 'raw run records' line found")
        elif claim and (m.group(1), m.group(2)) != claim.groups():
            problems.append(f"parser stdout: {m.group(1)} raw / {m.group(2)} unique, "
                            f"3.1 says {claim.group(1)} / {claim.group(2)}")

    if out:
        Path(out).write_text(render(vals, conv), encoding="utf-8")
    print(f"3.1 cross-check: {n_mean} mean cells, {n_gap} gap cells")
    print(f"SD convention: population matches {sd_hits['pop']}/{n_mean}, "
          f"sample matches {sd_hits['sample']}/{n_mean}")
    print(f"coverage: {len(keys)} unique (cond, N, seed); macro_top1 cells: {len(mt1)}")
    print(f"boundary-sensitive cells accepted: {len(boundary_cells)}")
    for b in boundary_cells:
        print(f"  boundary: {b}")
    print(f"problems: {len(problems)}")
    for p in problems:
        print(f"  {p}")
    print("VERDICT:", "PASS" if not problems else "FAIL")
    return 0 if not problems else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-csv", required=True)
    ap.add_argument("--results-md", default="RESULTS.md")
    ap.add_argument("--parser-stdout", default=None,
                    help="saved stdout of parse_nextskill_full.py, to check the 144/72 claim")
    ap.add_argument("--out", default=None, help="also write the rendered 3.1 tables here")
    a = ap.parse_args(argv)
    return check(a.results_md, a.long_csv, a.parser_stdout, a.out)


if __name__ == "__main__":
    sys.exit(main())
