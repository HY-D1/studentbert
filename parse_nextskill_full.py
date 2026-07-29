#!/usr/bin/env python3
# Parse the W5 next-skill sweep logs into a COMPLETE per-condition x N table.
#
# Why the ad-hoc parse only returned scratch + indomain:
#   1. It searched for condition tokens fromednet|fromjunyi|fromassist, but the
#      W5 next-skill --run_type values are scratch|indomain|ednet|junyi, so the
#      two cross-dataset conditions never matched.
#   2. It matched only the "_nextskill_" task token and skipped the "_nsauc_"
#      runs (the ones with AUC + the convergence-fair 60-epoch EdNet at N<=50).
#   3. Duplicate/timed-out job logs (resubmits) were counted twice, inflating
#      the in-domain seed count to 6.
#
# This parser fixes all three, tolerates BOTH metric-print formats (wandb-key
# "test/macro_auc 0.87" and human "test macro-OVR AUC: 0.87"), de-dupes by
# (dataset, condition, N, seed), and writes NOT FOUND for any missing cell.
# It never fabricates a number.

import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

DEFAULT_GLOBS = [
    "w5_nextskill_sweep_*.log",
    "w5_nextskill_auc_sweep_*.log",
    "w5_nsauc_*.log",
    "w5_ns*_*.log",          # widen; the run-name regex is the real filter
    "w4_nextskill*.log",
]

# --run_type condition tokens actually emitted by the W5 sbatch scripts,
# mapped to canonical names. fromassist == assist source into assist == in-domain.
COND_ALIASES = {
    "scratch": "scratch",
    "indomain": "indomain", "indom": "indomain", "fromassist": "indomain",
    "ednet": "ednet", "fromednet": "ednet",
    "junyi": "junyi", "fromjunyi": "junyi",
}
# longest-first so "indomain" wins over "indom", "fromednet" over "ednet", etc.
COND_TOKENS = "|".join(sorted(COND_ALIASES, key=len, reverse=True))
TASK_TOKENS = r"nextskill|nsauc|ns"

RUN_RE = re.compile(
    r"edubert_(?P<ds>[A-Za-z0-9]+)_(?P<cond>" + COND_TOKENS + r")_"
    r"(?P<task>" + TASK_TOKENS + r")_n(?P<N>\d+)_seed(?P<S>\d+)"
)

NUM = r"([0-9]*\.?[0-9]+)"
# each metric: try wandb-key style first, then human-print style.
METRIC_PATTERNS = {
    "top1":         [r"test/top1\s*[=:]?\s*" + NUM,         r"test top-1 acc\s*:?\s*" + NUM],
    "top5":         [r"test/top5\s*[=:]?\s*" + NUM,         r"test top-5 acc\s*:?\s*" + NUM],
    "macro_auc":    [r"test/macro_auc\s*[=:]?\s*" + NUM,    r"test macro-?OVR AUC\s*:?\s*" + NUM],
    "weighted_auc": [r"test/weighted_auc\s*[=:]?\s*" + NUM, r"test weighted-?OVR AUC\s*:?\s*" + NUM],
    "macro_top1":   [r"test/macro_top1\s*[=:]?\s*" + NUM,   r"test macro-?top1\s*:?\s*" + NUM],
}
METRIC_RE = {m: [re.compile(p) for p in pats] for m, pats in METRIC_PATTERNS.items()}

METRIC_ORDER = ["top1", "top5", "macro_auc", "weighted_auc", "macro_top1"]
COND_ORDER = ["scratch", "indomain", "ednet", "junyi"]
COND_LABEL = {"scratch": "scratch", "indomain": "in-domain",
              "ednet": "EdNet", "junyi": "Junyi"}
# prefer nsauc (full metric set + convergence-fair EdNet) over the older nextskill runs
TASK_PREF = {"nsauc": 0, "nextskill": 1, "ns": 2}


def iter_run_blocks(txt):
    """Yield (match, block_text) for each run. The run name is printed twice per
    run (run=... then === ... ===); collapse consecutive same-key hits so the
    block spans from a run's first mention to the NEXT distinct run's first
    mention, which is where that run's test metrics live."""
    hits = list(RUN_RE.finditer(txt))
    bounds, last_key = [], None
    for m in hits:
        key = (m.group("ds"), m.group("cond"), m.group("task"), m.group("N"), m.group("S"))
        if key != last_key:
            bounds.append((m, m.start()))
            last_key = key
    for i, (m, start) in enumerate(bounds):
        end = bounds[i + 1][1] if i + 1 < len(bounds) else len(txt)
        yield m, txt[start:end]


def extract_metrics(block):
    out = {}
    for metric, regs in METRIC_RE.items():
        for rx in regs:
            mm = rx.search(block)
            if mm:
                out[metric] = float(mm.group(1))
                break
    return out


def parse_dir(root, globs):
    files = []
    for g in globs:
        files.extend(sorted(glob.glob(os.path.join(root, g))))
    files = sorted(set(files))
    per_file_counts = {}
    # raw records keyed by full identity incl. task token and source file
    raw = []
    for fp in files:
        try:
            txt = open(fp, errors="ignore").read()
        except OSError:
            continue
        n = 0
        for m, block in iter_run_blocks(txt):
            rec = {
                "ds": m.group("ds"),
                "cond_raw": m.group("cond"),
                "cond": COND_ALIASES[m.group("cond")],
                "task": m.group("task"),
                "N": int(m.group("N")),
                "seed": int(m.group("S")),
                "file": os.path.basename(fp),
                "metrics": extract_metrics(block),
            }
            raw.append(rec)
            n += 1
        per_file_counts[os.path.basename(fp)] = n
    return files, per_file_counts, raw


def dedup(raw):
    """Merge to one record per (ds, cond, N, seed). Prefer nsauc-token values;
    fill any still-missing metric from a less-preferred duplicate."""
    groups = defaultdict(list)
    for r in raw:
        groups[(r["ds"], r["cond"], r["N"], r["seed"])].append(r)
    merged = {}
    dup_report = []  # (key, n_raw_records)
    for key, recs in groups.items():
        if len(recs) > 1:
            dup_report.append((key, len(recs)))
        recs = sorted(recs, key=lambda r: TASK_PREF.get(r["task"], 9))
        m = {}
        srcs = []
        for r in recs:
            srcs.append(f"{r['file']}:{r['task']}")
            for k, v in r["metrics"].items():
                if k not in m:  # first (most-preferred) non-missing wins
                    m[k] = v
        merged[key] = {"ds": key[0], "cond": key[1], "N": key[2], "seed": key[3],
                       "metrics": m, "sources": srcs}
    return merged, dup_report


def aggregate(merged, dataset):
    """(cond, N, metric) -> {mean, std, n, values, seeds}."""
    buckets = defaultdict(lambda: {"vals": [], "seeds": []})
    Ns = set()
    for key, rec in merged.items():
        if rec["ds"] != dataset:
            continue
        Ns.add(rec["N"])
        for metric, val in rec["metrics"].items():
            b = buckets[(rec["cond"], rec["N"], metric)]
            b["vals"].append(val)
            b["seeds"].append(rec["seed"])
    agg = {}
    for (cond, N, metric), b in buckets.items():
        vals = b["vals"]
        agg[(cond, N, metric)] = {
            "mean": st.mean(vals),
            "std": (st.pstdev(vals) if len(vals) > 1 else 0.0),
            "n": len(vals),
            "seeds": sorted(b["seeds"]),
            "values": [round(v, 5) for v in vals],
        }
    return agg, sorted(Ns)


def fmt_cell(entry):
    if entry is None:
        return "NOT FOUND"
    return f"{entry['mean']:.4f}\u00b1{entry['std']:.4f}(n={entry['n']})"


def print_table(agg, Ns, metric, title):
    print(f"\n=== {title}  [{metric}] ===")
    header = "cond".ljust(12) + "".join(f"N={n}".rjust(22) for n in Ns)
    print(header)
    for cond in COND_ORDER:
        row = [COND_LABEL[cond].ljust(12)]
        any_cell = False
        for n in Ns:
            e = agg.get((cond, n, metric))
            if e:
                any_cell = True
            row.append(fmt_cell(e).rjust(22))
        # still print the row even if all NOT FOUND, so gaps are explicit
        print("".join(row))


def print_gap_table(agg, Ns, metric, title):
    print(f"\n=== {title}  [gap vs scratch, {metric}] ===")
    header = "cond".ljust(12) + "".join(f"N={n}".rjust(14) for n in Ns)
    print(header)
    for cond in COND_ORDER:
        if cond == "scratch":
            continue
        row = [COND_LABEL[cond].ljust(12)]
        for n in Ns:
            e = agg.get((cond, n, metric))
            s = agg.get(("scratch", n, metric))
            if e and s:
                row.append(f"{e['mean'] - s['mean']:+.4f}".rjust(14))
            else:
                row.append("NOT FOUND".rjust(14))
        print("".join(row))


def coverage(agg, Ns):
    print("\n=== coverage (distinct seeds per cell) ===")
    print("cond".ljust(12) + "N".rjust(7) + "  " + "  ".join(m.rjust(12) for m in METRIC_ORDER))
    for cond in COND_ORDER:
        for n in Ns:
            cells = []
            has_any = False
            for metric in METRIC_ORDER:
                e = agg.get((cond, n, metric))
                if e:
                    has_any = True
                    cells.append(str(e["n"]).rjust(12))
                else:
                    cells.append("-".rjust(12))
            if has_any:
                print(cond.ljust(12) + str(n).rjust(7) + "  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory containing the .log files")
    ap.add_argument("--dataset", default="assist2017",
                    help="target dataset token; use 'all' to print every dataset found")
    ap.add_argument("--globs", nargs="*", default=None,
                    help="override log glob patterns")
    ap.add_argument("--out_prefix", default="nextskill_results")
    args = ap.parse_args()

    globs = args.globs if args.globs else DEFAULT_GLOBS
    files, per_file_counts, raw = parse_dir(args.dir, globs)

    print("=== files scanned (runs matched per file) ===")
    if not files:
        print("  (none matched globs:", globs, ")")
    for f in files:
        print(f"  {os.path.basename(f):40s} runs={per_file_counts.get(os.path.basename(f), 0)}")

    merged, dup_report = dedup(raw)
    print(f"\nraw run records: {len(raw)}   unique (ds,cond,N,seed): {len(merged)}")
    if dup_report:
        print("de-duplicated (key -> #raw copies, kept 1):")
        for key, c in sorted(dup_report):
            print(f"  {key} <- {c} copies")

    datasets = sorted({r["ds"] for r in raw}) if args.dataset == "all" else [args.dataset]

    full_json = {"files": [os.path.basename(f) for f in files],
                 "per_file_counts": per_file_counts,
                 "datasets": {}}
    csv_rows = ["dataset,condition,N,seed,metric,value"]
    agg_rows = ["dataset,condition,N,metric,mean,std,n,seeds"]

    for ds in datasets:
        agg, Ns = aggregate(merged, ds)
        if not Ns:
            print(f"\n### dataset '{ds}': NOT FOUND (no runs matched) ###")
            continue
        print(f"\n############## dataset: {ds} ##############")
        for metric in METRIC_ORDER:
            print_table(agg, Ns, metric, ds)
        for metric in ("macro_auc", "top1"):
            print_gap_table(agg, Ns, metric, ds)
        coverage(agg, Ns)

        # collect for machine-readable dumps
        full_json["datasets"][ds] = {}
        for (cond, N, metric), e in agg.items():
            full_json["datasets"][ds].setdefault(f"{cond}", {}).setdefault(str(N), {})[metric] = {
                "mean": round(e["mean"], 6), "std": round(e["std"], 6),
                "n": e["n"], "seeds": e["seeds"]}
            agg_rows.append(f"{ds},{cond},{N},{metric},{e['mean']:.6f},{e['std']:.6f},{e['n']},"
                            + "|".join(map(str, e["seeds"])))
        for key, rec in merged.items():
            if rec["ds"] != ds:
                continue
            for metric, val in rec["metrics"].items():
                csv_rows.append(f"{ds},{rec['cond']},{rec['N']},{rec['seed']},{metric},{val:.6f}")

    with open(args.out_prefix + ".json", "w") as f:
        json.dump(full_json, f, indent=2)
    with open(args.out_prefix + "_long.csv", "w") as f:
        f.write("\n".join(csv_rows) + "\n")
    with open(args.out_prefix + "_agg.csv", "w") as f:
        f.write("\n".join(agg_rows) + "\n")
    print(f"\nwrote {args.out_prefix}.json, {args.out_prefix}_long.csv, {args.out_prefix}_agg.csv")


if __name__ == "__main__":
    main()
