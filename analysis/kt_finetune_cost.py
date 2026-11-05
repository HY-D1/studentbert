from __future__ import annotations

"""Measured cost of fully fine-tuning every StudentBERT candidate (Prof. Hazra, Nov 2026).

Reads benchmark_final/executions.tsv and keeps the runs behind the Track B gold: task kt, valid
for the primary analysis, 20 epochs, budget n3000 or full_split. Every (target, candidate, seed)
was run twice (campaigns TG1 and N3000P, the known non-bit-exact reruns); a rerun is not part of
a method's cost, so each group counts once, at the median elapsed time of its copies. Elapsed is
the scheduler's wall-clock time for a one-GPU job (start-up, data loading, per-epoch validation and
test evaluation included), so it is an upper bound on pure training time. Stops unless every
target has exactly 8 candidates x 6 seeds.

  python analysis/kt_finetune_cost.py --executions benchmark_final/executions.tsv \
      --out benchmark_final/kt_finetune_cost
"""

import argparse
import collections
import csv
import hashlib
import statistics as st
import sys
from pathlib import Path


def seconds(elapsed: str) -> float:
    days, _, hms = elapsed.rpartition("-")
    parts = [float(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return (float(days) if days else 0.0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src = Path(a.executions)
    src_md5 = hashlib.md5(src.read_bytes()).hexdigest()
    rows = [r for r in csv.DictReader(src.read_text().splitlines(), delimiter="\t")
            if r["track"] == "B" and r["task"] == "kt" and r["valid_for_primary_analysis"] == "yes"
            and r["config_epochs"] == "20" and r["budget"] in ("n3000", "full_split")
            and r["elapsed"]]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["target_dataset"], r["candidate"], r["finetune_seed"])].append(r)
    by = collections.defaultdict(list)
    for (t, c, s), rs in groups.items():
        by[t].append({"candidate": c, "seed": s, "copies": len(rs),
                      "seconds": st.median(seconds(r["elapsed"]) for r in rs),
                      "gpus": sorted({r["gpu_type"] for r in rs}),
                      "campaigns": sorted({r["campaign"] for r in rs})})
    bad = {t: len(v) for t, v in by.items()
           if len(v) != 48 or len({x["candidate"] for x in v}) != 8
           or len({x["seed"] for x in v}) != 6}
    if len(by) != 7 or bad:
        sys.exit(f"ABORT: expected 7 targets with 8 candidates x 6 seeds; got {len(by)} targets, "
                 f"off: {bad}")
    out_rows = []
    for t in sorted(by):
        v = by[t]
        mins = [x["seconds"] / 60 for x in v]
        total = sum(x["seconds"] for x in v) / 3600
        out_rows.append({"target": t, "fine_tunes": len(v), "total_gpu_hours": total,
                         "per_seed_all_candidates_gpu_hours": total / 6,
                         "minutes_per_fine_tune_min": min(mins),
                         "minutes_per_fine_tune_max": max(mins),
                         "gpu_types": ",".join(sorted({g for x in v for g in x["gpus"]})),
                         "copies_per_group": ",".join(str(n) for n in
                                                      sorted({x["copies"] for x in v})),
                         "campaigns": ",".join(sorted({c for x in v for c in x["campaigns"]}))})
    out = Path(a.out)
    tsv = out.with_name(out.name + ".tsv")
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    grand = sum(r["total_gpu_hours"] for r in out_rows)
    L = ["# Measured cost of fully fine-tuning every StudentBERT candidate", "",
         f"Source: {src.name} (md5 {src_md5}). One run per target, candidate and seed (20 epochs, "
         "3,000 target learners or the whole training split), at the median elapsed time of its "
         "copies; wall-clock of a one-GPU job.", "",
         "| target | fine-tunes | all candidates x 6 seeds (GPU-h) | one seed, all 8 candidates "
         "(GPU-h) | minutes per fine-tune | GPU | copies per run |",
         "|---|---|---|---|---|---|---|"]
    for r in out_rows:
        L.append(f"| {r['target']} | {r['fine_tunes']} | {r['total_gpu_hours']:.2f} | "
                 f"{r['per_seed_all_candidates_gpu_hours']:.2f} | "
                 f"{r['minutes_per_fine_tune_min']:.1f} to {r['minutes_per_fine_tune_max']:.1f} | "
                 f"{r['gpu_types']} | {r['copies_per_group']} |")
    L += ["", f"Whole grid (7 targets x 8 candidates x 6 seeds): {grand:.1f} GPU-hours.", "",
          f"Files: {tsv.name} {hashlib.md5(tsv.read_bytes()).hexdigest()}"]
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print(f"wrote {tsv.name} and {rp.name} md5 {hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
