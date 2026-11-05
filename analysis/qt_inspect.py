from __future__ import annotations

"""Read-only inspection of the Quick-Tune meta-dataset (Pineda Arango et al., ICLR 2024).

Answers whether the released curves can simulate early model-selection decisions (Prof. Hazra,
Nov 2026): for each version (micro, mini, extended) it reports the runs per model and task, the
curve lengths and how many runs stop before 50 epochs, the per-epoch training and evaluation
time, and, the key question, whether a hyperparameter configuration is shared by every model on
a task, so that one comparable curve per model exists without choosing it by its final result.

A run's configuration is its row in unnormalized_args_table.csv; its model and task are the
one-hot cat__model_ and cat__dataset_ columns; the configuration signature is every other column
except those named in --ignore (by default the id column and columns that vary with the task,
such as the number of classes). The column list is printed so the signature can be checked.

  python analysis/qt_inspect.py --root /projects/algl/dai.hany/quicktune \
      --out /projects/algl/dai.hany/quicktune/qt_inspect_report.md
"""

import argparse
import collections
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

VERSIONS = ("micro", "mini", "extended")
DEFAULT_IGNORE = ("num_classes", "dataset_size", "num_samples", "image_size_of_dataset")


def one_hot(row: dict, prefix: str) -> str | None:
    hits = [k[len(prefix):] for k, v in row.items()
            if k.startswith(prefix) and float(v or 0) == 1.0]
    return hits[0] if len(hits) == 1 else None


def inspect(root: Path, version: str, ignore: tuple[str, ...]) -> list[str]:
    d = root / version / version
    rows = list(csv.DictReader((d / "unnormalized_args_table.csv").open()))
    cols = list(rows[0])
    idc = cols[0]
    curves = {"top1": json.load((d / "eval_top1.json").open()),
              "train": json.load((d / "train_time.json").open()),
              "eval": json.load((d / "eval_time.json").open())}
    run_task = {rid: task for task, runs in curves["top1"].items() for rid in runs}
    skip = {idc} | {c for c in cols if c.startswith("cat__dataset_") or c.startswith("cat__model_")}
    skip |= {c for c in cols if any(s in c for s in ignore)}
    sig_cols = [c for c in cols if c not in skip]
    L = [f"## {version}", "", f"Configurations in the table: {len(rows)}; tasks with curves: "
         f"{len(curves['top1'])}; runs with curves: {len(run_task)}.", "",
         "Signature columns (configuration without model and task): " + ", ".join(sig_cols), ""]
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    lengths, missing, unparsed = [], 0, 0
    sec_per_epoch = collections.defaultdict(list)
    for r in rows:
        rid = r[idc]
        model, task = one_hot(r, "cat__model_"), one_hot(r, "cat__dataset_")
        if model is None or task is None:
            unparsed += 1
            continue
        if rid not in run_task:
            missing += 1
            continue
        curve = curves["top1"][run_task[rid]][rid]
        lengths.append(len(curve))
        tt = curves["train"][run_task[rid]].get(rid, [])
        if tt:
            sec_per_epoch[model].append(st.median(tt))
        sig = tuple(r[c] for c in sig_cols)
        per[task][model].append((sig, len(curve)))
    models = sorted({m for t in per.values() for m in t})
    cnt = collections.Counter(lengths)
    full = sum(n for k, n in cnt.items() if k >= 50)
    L += [f"Models: {len(models)}. Rows without one model and one task: {unparsed}; rows without "
          f"a curve: {missing}.",
          f"Curve lengths: median {st.median(lengths):.0f} epochs, {full} of {len(lengths)} runs "
          f"reach 50; shortest lengths and counts: {sorted(cnt.items())[:6]}.", ""]
    rpm = [len(v) for t in per.values() for v in t.values()]
    L += [f"Runs per model and task: min {min(rpm)}, median {st.median(rpm):.0f}, max {max(rpm)}; "
          f"model-task pairs with no run: {len(per) * len(models) - len(rpm)}.", ""]
    shared_all, shared_best = 0, []
    for task, by_model in sorted(per.items()):
        sigs = collections.Counter()
        for m, runs in by_model.items():
            for s in {s for s, _ in runs}:
                sigs[s] += 1
        top = sigs.most_common(1)[0][1] if sigs else 0
        shared_best.append(top)
        shared_all += top == len(models)
    L += [f"Shared configuration check: on {shared_all} of {len(per)} tasks one configuration was "
          f"run for all {len(models)} models; the best-covered configuration per task reaches "
          f"min {min(shared_best)}, median {st.median(shared_best):.0f}, max {max(shared_best)} "
          f"models.", ""]
    L += ["Median training seconds per epoch by model: " + ", ".join(
        f"{m} {st.median(v):.1f}" for m, v in sorted(sec_per_epoch.items())), ""]
    return L


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ignore", nargs="*", default=list(DEFAULT_IGNORE))
    a = ap.parse_args(argv)
    root = Path(a.root)
    L = ["# Quick-Tune meta-dataset inspection", ""]
    for v in VERSIONS:
        z = root / f"{v}.zip"
        if z.is_file():
            L.append(f"{z.name} md5 {hashlib.md5(z.read_bytes()).hexdigest()}")
    L.append("")
    for v in VERSIONS:
        L += inspect(root, v, tuple(a.ignore))
    out = Path(a.out)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {out.name} md5 {hashlib.md5(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
