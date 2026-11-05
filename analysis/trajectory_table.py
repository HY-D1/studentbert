from __future__ import annotations

"""Per-epoch ranking trajectories and compute for every candidate (docs/trajectory_design.md).

One row per candidate per epoch, from the files already on disk; no model is run. Each source is
optional, so the table can be rebuilt as sources are added:

  python analysis/trajectory_table.py \
      --studentbert benchmark_final/early_ft_epochs.tsv \
      --executions benchmark_final/executions.tsv \
      --nlp /projects/algl/dai.hany/task4/early_ft /projects/algl/dai.hany/task4/early_ft_fresh \
            /projects/algl/dai.hany/task4/early_ft_p3 \
      --quicktune /projects/algl/dai.hany/quicktune \
      --out benchmark_final/trajectories
"""

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

FIELDS = ["benchmark", "target", "setting", "seed", "candidate", "epoch", "metric", "score",
          "rank", "epoch_s", "cumulative_s", "cost_source", "gpu", "run_id", "source"]
QT_VERSIONS = ("micro", "mini", "extended")
QT_EPOCHS = 50


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def seconds(elapsed: str) -> float:
    days, _, hms = elapsed.rpartition("-")
    parts = [float(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return (float(days) if days else 0.0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def studentbert(epochs_tsv: Path, executions: Path) -> tuple[list[dict], list[str]]:
    ex = {}
    for r in csv.DictReader(executions.read_text().splitlines(), delimiter="\t"):
        if r["log_file"] and r["elapsed"]:
            ex.setdefault(Path(r["log_file"]).name, r)
    rows, by_run, missing = [], collections.defaultdict(list), set()
    for r in csv.DictReader(epochs_tsv.read_text().splitlines(), delimiter="\t"):
        by_run[(r["target"], r["candidate"], r["finetune_seed"], r["log_file"])].append(r)
    for (t, c, s, log), rs in sorted(by_run.items()):
        e = ex.get(Path(log).name)
        if e is None:
            missing.add(log)
            continue
        rs.sort(key=lambda r: int(r["epoch"]))
        per = seconds(e["elapsed"]) / int(e["epochs_run"] or len(rs))
        for i, r in enumerate(rs, 1):
            rows.append({"benchmark": "studentbert", "target": t, "setting": "kt", "seed": s,
                         "candidate": c, "epoch": int(r["epoch"]), "metric": "val_auc",
                         "score": float(r["val_auc"]), "epoch_s": per, "cumulative_s": per * i,
                         "cost_source": "elapsed_even", "gpu": e["gpu_type"],
                         "run_id": e["run_id"], "source": epochs_tsv.name})
    if missing:
        sys.exit(f"ABORT: {len(missing)} StudentBERT runs have no execution record, e.g. "
                 f"{sorted(missing)[:3]}")
    return rows, [f"studentbert: {len(by_run)} runs from {epochs_tsv.name} (md5 "
                  f"{md5(epochs_tsv)}), cost from {executions.name} (md5 {md5(executions)})"]


def nlp(dirs: list[Path]) -> tuple[list[dict], list[str]]:
    rows, notes = [], []
    for d in dirs:
        files = sorted(d.glob("*/*.json"))
        if not files:
            sys.exit(f"ABORT: no early fine-tune records under {d}")
        for f in files:
            rec = json.loads(f.read_text())
            cum = 0.0
            for e in sorted(rec["epochs"], key=lambda e: e["epoch"]):
                cum += e["epoch_s"]
                rows.append({"benchmark": "nlp_early", "target": rec["task"],
                             "setting": rec["pooling"], "seed": rec["seed"],
                             "candidate": rec["model"], "epoch": e["epoch"],
                             "metric": "val_acc", "score": e["val_acc"], "epoch_s": e["epoch_s"],
                             "cumulative_s": cum, "cost_source": "measured", "gpu": rec["gpu"],
                             "run_id": f.stem, "source": f"{d.name}/{f.parent.name}"})
        notes.append(f"nlp_early: {len(files)} runs from {d}")
    return rows, notes


def quicktune(root: Path) -> tuple[list[dict], list[str]]:
    rows, notes = [], []
    for v in QT_VERSIONS:
        d = root / v / v
        table = list(csv.DictReader((d / "unnormalized_args_table.csv").open()))
        idc = list(table[0])[0]
        top1 = json.load((d / "eval_top1.json").open())
        tr = json.load((d / "train_time.json").open())
        ev = json.load((d / "eval_time.json").open())
        task_of = {rid: t for t, runs in top1.items() for rid in runs}
        chosen, skipped = {}, 0
        for r in table:
            rid = r[idc]
            model = [k[11:] for k, x in r.items() if k.startswith("cat__model_") and float(x or 0)]
            task = task_of.get(rid)
            if len(model) != 1 or task is None:
                sys.exit(f"ABORT: {v} run {rid} has no single model or no curve")
            if len(top1[task][rid]) < QT_EPOCHS:
                skipped += 1
                continue
            chosen.setdefault((task, model[0]), rid)
        for (task, model), rid in sorted(chosen.items()):
            cum = 0.0
            for i in range(QT_EPOCHS):
                es = tr[task][rid][i] + ev[task][rid][i]
                cum += es
                rows.append({"benchmark": f"quicktune_{v}", "target": task, "setting": "",
                             "seed": "", "candidate": model, "epoch": i + 1,
                             "metric": "val_top1_pct", "score": top1[task][rid][i],
                             "epoch_s": es, "cumulative_s": cum, "cost_source": "measured",
                             "gpu": "", "run_id": rid, "source": f"{v}.zip"})
        per_task = collections.Counter(t for t, _ in chosen)
        full = sum(n == 24 for n in per_task.values())
        z = root / f"{v}.zip"
        zmd5 = md5(z) if z.is_file() else "zip not found"
        notes.append(f"quicktune_{v}: rule A chose {len(chosen)} runs on {len(per_task)} tasks "
                     f"(models per task min {min(per_task.values())}, max "
                     f"{max(per_task.values())}; all 24 models on {full} tasks); runs shorter "
                     f"than {QT_EPOCHS} epochs not used: {skipped}; {z.name} md5 {zmd5}")
    return rows, notes


def add_ranks(rows: list[dict]) -> None:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["benchmark"], r["target"], r["setting"], str(r["seed"]), r["epoch"])].append(r)
    for g in groups.values():
        g.sort(key=lambda r: -r["score"])
        rank = 0
        for i, r in enumerate(g):
            if i == 0 or r["score"] < g[i - 1]["score"]:
                rank = i + 1
            r["rank"] = rank


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--studentbert", default=None)
    ap.add_argument("--executions", default=None)
    ap.add_argument("--nlp", nargs="*", default=[])
    ap.add_argument("--quicktune", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows, notes = [], []
    if a.studentbert:
        if not a.executions:
            sys.exit("ABORT: --studentbert needs --executions for the cost")
        r, n = studentbert(Path(a.studentbert), Path(a.executions))
        rows += r
        notes += n
    if a.nlp:
        r, n = nlp([Path(x) for x in a.nlp])
        rows += r
        notes += n
    if a.quicktune:
        r, n = quicktune(Path(a.quicktune))
        rows += r
        notes += n
    if not rows:
        sys.exit("ABORT: no source given")
    add_ranks(rows)
    out = Path(a.out)
    tsv = out.with_name(out.name + ".tsv")
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    counts = collections.Counter(r["benchmark"] for r in rows)
    L = ["# Per-epoch trajectories and compute", "",
         "Definitions in docs/trajectory_design.md. Rows per benchmark: "
         + ", ".join(f"{b} {n}" for b, n in sorted(counts.items())) + ".", ""]
    L += [f"- {n}" for n in notes]
    L += ["", f"Files: {tsv.name} {md5(tsv)}"]
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {tsv.name} and {rp.name} md5 {md5(rp)}")


if __name__ == "__main__":
    main()
