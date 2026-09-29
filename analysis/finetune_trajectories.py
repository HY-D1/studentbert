from __future__ import annotations

# MRAP failure review, diagnostic D8: how the candidate ranking evolves during fine-tuning, and
# whether the validation learners rank the candidates as the test learners do. Reads the Track B
# N=3000 fine-tunes of the chosen targets from the benchmark's own executions.tsv (valid primary
# rows only, so the runs are exactly the gold's), their logs for the per-epoch validation AUC
# (printed at 4 dp by scripts/finetune_edubert.py), and, with --wandb, the full-precision
# best/val_auc of each run from the W&B export (wandb_by_id.jsonl, looked up by run id; a record
# whose name differs from the log's run is refused). Test AUC is the benchmark's value, which is the
# full-precision W&B value where one exists. Every log must hold all its epochs and a test AUC equal
# to the benchmark's value_log, or the run aborts. Means are exact in Decimal and rounded once, half
# away from zero, for display; the winner is the best source by mean test AUC over the shared seeds.
#
#   PYTHONPATH=. python analysis/finetune_trajectories.py \
#       --executions benchmark_final/executions.tsv --logdir . --wandb wandb_by_id.jsonl \
#       --out benchmark_final/trajectories

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

EPOCH = re.compile(r"^epoch\s+(\d+)\s+train_loss=\S+\s+val_AUC=([0-9.]+)")
RUN = re.compile(r"run=(\S+)")


def q(x, places: str = "0.0001", sign: bool = False) -> str:
    d = Decimal(str(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def parse_log(text: str, epochs: int) -> dict:
    val, test, run = {}, None, None
    for ln in text.splitlines():
        s = ln.strip()
        m = EPOCH.match(s)
        if m:
            val[int(m.group(1))] = Decimal(m.group(2))
        if s.startswith("test AUC"):
            test = Decimal(s.split(":", 1)[1].strip())
        r = RUN.search(s)
        if r and run is None and s.startswith("device="):
            run = r.group(1)
    if sorted(val) != list(range(1, epochs + 1)):
        raise ValueError(f"epochs {sorted(val)}, expected 1 to {epochs}")
    if test is None:
        raise ValueError("no test AUC line")
    return {"val": val, "test_log": test, "run": run}


def load_wandb(path: str | None) -> dict:
    if not path:
        return {}
    out = {}
    for ln in Path(path).read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            if "error" not in r:
                out[r["id"]] = r
    return out


def load_runs(executions: str, logdir: str, targets: tuple, budget: str, epochs: int,
              wandb: dict) -> dict:
    runs = {}
    with open(executions, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if not (r["track"] == "B" and r["budget"] == budget and r["metric"] == "test_auc"
                    and r["target_dataset"] in targets
                    and r["valid_for_primary_analysis"] == "yes"):
                continue
            key = (r["target_dataset"], r["candidate"], int(r["finetune_seed"]))
            if key in runs:
                sys.exit(f"ABORT: two primary rows for {key}")
            p = Path(logdir) / r["log_file"]
            if not p.is_file():
                sys.exit(f"ABORT: missing log {p}")
            try:
                lg = parse_log(p.read_text(errors="ignore"), epochs)
            except ValueError as ex:
                sys.exit(f"ABORT: {p.name}: {ex}")
            if r["value_log"] and Decimal(r["value_log"]) != lg["test_log"]:
                sys.exit(f"ABORT: {p.name} test AUC {lg['test_log']} but the benchmark logged "
                         f"{r['value_log']}")
            best_val, source = max(lg["val"].values()), "log 4 dp"
            rec = wandb.get(r["wandb_run_id"]) if r["wandb_run_id"] else None
            if wandb:
                if rec is None:
                    sys.exit(f"ABORT: no W&B record for {p.name} ({r['wandb_run_id']})")
                if lg["run"] and rec.get("name") != lg["run"]:
                    sys.exit(f"ABORT: W&B id {r['wandb_run_id']} names {rec.get('name')}, "
                             f"the log ran {lg['run']}")
                bv = (rec.get("summary") or {}).get("best/val_auc")
                if bv is None:
                    sys.exit(f"ABORT: W&B record {r['wandb_run_id']} has no best/val_auc")
                if Decimal(repr(bv)).quantize(Decimal("0.0001"), ROUND_HALF_UP) != best_val:
                    sys.exit(f"ABORT: {p.name} best val {best_val} in the log, {bv} in W&B")
                best_val, source = Decimal(repr(bv)), "W&B full precision"
            runs[key] = {"val": lg["val"], "best_val": best_val, "best_val_source": source,
                         "test": Decimal(r["value"]), "log": p.name}
    return runs


def mean(xs) -> Decimal:
    xs = list(xs)
    return sum(xs, Decimal(0)) / Decimal(len(xs))


def analyse(runs: dict, targets: tuple, epochs: int) -> tuple[list[dict], list[dict]]:
    traj, cmp_ = [], []
    for t in targets:
        cands = sorted({k[1] for k in runs if k[0] == t})
        srcs = [c for c in cands if c.startswith("src:")]
        seeds = sorted(set.intersection(*({k[2] for k in runs if k[:2] == (t, c)} for c in cands)))
        if not srcs or len(seeds) < 2:
            sys.exit(f"ABORT: {t} has {len(srcs)} sources over {len(seeds)} shared seeds")
        test = {c: mean(runs[(t, c, s)]["test"] for s in seeds) for c in srcs}
        win, own = max(srcs, key=lambda c: test[c]), f"src:{t}"
        for e in range(1, epochs + 1):
            m = {c: mean(runs[(t, c, s)]["val"][e] for s in seeds) for c in srcs}
            order = sorted(srcs, key=lambda c: (-m[c], c))
            ahead = sum(runs[(t, win, s)]["val"][e] > runs[(t, own, s)]["val"][e] for s in seeds)
            traj.append({"target": t, "epoch": e, "leader": order[0], "own_val": q(m[own]),
                         "winner_val": q(m[win]), "winner_minus_own": q(m[win] - m[own], sign=True),
                         "seeds_winner_ahead": f"{ahead}/{len(seeds)}",
                         "own_rank": order.index(own) + 1, "winner_rank": order.index(win) + 1})
        for metric, get in (("best validation AUC", lambda k: runs[k]["best_val"]),
                            ("test AUC", lambda k: runs[k]["test"])):
            diffs = [get((t, win, s)) - get((t, own, s)) for s in seeds]
            cmp_.append({"target": t, "winner": win, "own": own, "metric": metric,
                         "winner_minus_own_mean": q(mean(diffs), sign=True),
                         "seeds_positive": f"{sum(d > 0 for d in diffs)}/{len(seeds)}",
                         "per_seed": " ".join(q(d, sign=True) for d in diffs),
                         "seeds": " ".join(map(str, seeds)),
                         "source": runs[(t, own, seeds[0])]["best_val_source"]
                         if metric.startswith("best") else "benchmark value"})
    return traj, cmp_


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executions", required=True)
    ap.add_argument("--logdir", default=".")
    ap.add_argument("--wandb", default=None, help="wandb_by_id.jsonl for full-precision best val")
    ap.add_argument("--targets", nargs="+", default=["assist2017", "algebra2006"])
    ap.add_argument("--budget", default="n3000")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", required=True,
                    help="prefix for _epochs.tsv, _valtest.tsv and _report.md")
    a = ap.parse_args(argv)
    targets = tuple(a.targets)
    runs = load_runs(a.executions, a.logdir, targets, a.budget, a.epochs, load_wandb(a.wandb))
    traj, cmp_ = analyse(runs, targets, a.epochs)
    for name, rows in (("epochs", traj), ("valtest", cmp_)):
        with open(f"{a.out}_{name}.tsv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    L = ["# Fine-tuning trajectories and the validation-against-test check", "",
         f"{len(runs)} runs; validation AUC per epoch from the logs (4 dp); means over shared "
         "seeds.",
         ""]
    for t in targets:
        L += [f"## {t}", "", "| epoch | leader | own | winner | winner minus own | seeds winner "
              "ahead | own rank | winner rank |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['epoch']} | {r['leader']} | {r['own_val']} | {r['winner_val']} | "
              f"{r['winner_minus_own']} | {r['seeds_winner_ahead']} | {r['own_rank']} | "
              f"{r['winner_rank']} |" for r in traj if r["target"] == t]
        L += [""]
    L += ["## Winner minus own encoder, best validation AUC against test AUC", "",
          "| target | winner | metric | mean | seeds positive | per seed | source |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {r['target']} | {r['winner']} | {r['metric']} | {r['winner_minus_own_mean']} | "
          f"{r['seeds_positive']} | {r['per_seed']} | {r['source']} |" for r in cmp_]
    Path(f"{a.out}_report.md").write_text("\n".join(L) + "\n")
    for r in cmp_:
        print(" ".join(f"{k}={v}" for k, v in r.items() if k != "per_seed"))
    print(f"wrote {a.out}_epochs.tsv, {a.out}_valtest.tsv and {a.out}_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
