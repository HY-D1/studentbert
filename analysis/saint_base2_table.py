from __future__ import annotations

"""SAINT+ rows of the uniform base2 baseline campaign, read from the logs and the W&B export.

Per dataset and seed it reports: the 4 dp test AUC from the log banner, the full-precision
test/auc from wandb_by_id.jsonl (matched by run id taken from the log, never by name, since
run names are not unique), the number of epochs the log actually trained, and the epochs,
dropout and batch size recorded in the W&B config. It then prints appendix-ready LaTeX rows
(4 dp per seed, mean and population SD from the unrounded values) and the 3 dp main-table
value, rounded once, half up, from the full-precision mean.

Run from the code directory:
  /projects/algl/dai.hany/envs/sb/bin/python analysis/saint_base2_table.py --logdirs . ../logs --wandb wandb_by_id.jsonl
Exit status is 1 if anything is missing, duplicated, or disagrees, so a clean run is exit 0.
"""

import argparse
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

DATASETS = [("assist2017", "ASSISTments 2017"), ("ednet", "EdNet"), ("junyi", "Junyi"),
            ("algebra2005", "Algebra 2005"), ("bridge2006", "Bridge 2006"),
            ("assist2009", "ASSISTments 2009"), ("algebra2006", "Algebra 2006")]
SEEDS = [42, 1, 2, 3, 4, 5]
# RESULTS.md section 1 means, used only as a cross-check on what this script computes.
RECORDED_4DP = {"assist2017": "0.6290", "ednet": "0.6860", "junyi": "0.7985",
                "algebra2005": "0.8265", "bridge2006": "0.8534", "assist2009": "0.8406",
                "algebra2006": "0.8748"}
LOG_TOL = 5.001e-5  # a 4 dp banner can differ from full precision by half a unit

NAME = re.compile(r"^base2_saint_([a-z0-9]+)_seed(\d+)_(\d+)\.log$")
EPOCH = re.compile(r"^epoch\s+(\d+)\s")
BANNER = re.compile(r"^===\s+SAINT\s+on\s+(\S+)\s+===")
TEST_AUC = re.compile(r"^test AUC\s*:\s*([0-9.]+)")
RUN_ID = re.compile(r"wandb\.ai/[^/\s]+/[^/\s]+/runs/([a-z0-9]+)")
RUN_NAME = re.compile(r"\brun=(\S+)")


def D(x) -> Decimal:
    # repr gives the shortest decimal that round-trips the stored float, the value W&B wrote
    return x if isinstance(x, Decimal) else Decimal(repr(x))


def q(x, places: str) -> Decimal:
    return D(x).quantize(Decimal(places), rounding=ROUND_HALF_UP)


def mean_sd(vals: list[float]) -> tuple[Decimal, Decimal]:
    d = [D(v) for v in vals]
    m = sum(d) / len(d)
    return m, (sum((x - m) ** 2 for x in d) / len(d)).sqrt()


def parse_log(path: Path) -> dict:
    rec = {"file": path.name, "epochs": [], "banner": None, "auc4": None, "id": None, "run": None}
    for line in path.read_text(errors="replace").splitlines():
        if (m := EPOCH.match(line)):
            rec["epochs"].append(int(m.group(1)))
        elif (m := BANNER.match(line)):
            rec["banner"] = m.group(1)
        elif (m := TEST_AUC.match(line)) and rec["banner"]:
            rec["auc4"] = float(m.group(1))
        if rec["id"] is None and (m := RUN_ID.search(line)):
            rec["id"] = m.group(1)
        if rec["run"] is None and (m := RUN_NAME.search(line)):
            rec["run"] = m.group(1)
    return rec


LINE_ID = re.compile(r'^\{"id":\s*"([A-Za-z0-9]+)"')


def load_wandb(path: Path, wanted: set[str]) -> tuple[dict, int]:
    # Stream the export and decode only the lines for runs we need. Loading every record
    # (full summaries and configs of the whole project) is what gets a login-node process killed.
    out, n = {}, 0
    if not path.exists():
        return out, n
    with path.open() as fh:
        for ln in fh:
            n += 1
            m = LINE_ID.match(ln)
            if not m or m.group(1) not in wanted:
                continue
            r = json.loads(ln)
            if "error" not in r:
                out[r["id"]] = r
    return out, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdirs", nargs="+", default=["."])
    ap.add_argument("--wandb", default="wandb_by_id.jsonl")
    ap.add_argument("--ids_out", default="saint_ids_needed.txt")
    a = ap.parse_args()

    runs: dict[tuple[str, int], list[dict]] = {}
    seen: set[str] = set()
    for d in a.logdirs:
        for p in sorted(Path(d).glob("base2_saint_*.log")):
            if p.name in seen:
                continue
            seen.add(p.name)
            m = NAME.match(p.name)
            if not m:
                continue
            rec = parse_log(p)
            rec["ds"], rec["seed"] = m.group(1), int(m.group(2))
            runs.setdefault((rec["ds"], rec["seed"]), []).append(rec)

    wanted = {r["id"] for recs in runs.values() for r in recs if r["id"]}
    wb, n_lines = load_wandb(Path(a.wandb), wanted)
    problems: list[str] = []
    need_ids: list[str] = []
    print(f"logs found: {len(seen)}   run ids in logs: {len(wanted)}   "
          f"W&B lines scanned: {n_lines}   matched: {len(wb)}\n")
    print("dataset      seed  epochs  auc(log)  auc(W&B full precision)   cfg_epochs cfg_dropout cfg_batch  id")
    table: dict[str, dict[int, float]] = {}
    for ds, _ in DATASETS:
        table[ds] = {}
        for s in SEEDS:
            recs = runs.get((ds, s), [])
            done = [r for r in recs if r["banner"] and r["auc4"] is not None]
            if not done:
                problems.append(f"{ds} seed {s}: no completed log ({len(recs)} log(s) without a results banner)")
                print(f"{ds:12s} {s:4d}  MISSING")
                continue
            if len(done) > 1:
                problems.append(f"{ds} seed {s}: {len(done)} completed logs: {[r['file'] for r in done]}")
            r = done[-1]
            ep = max(r["epochs"]) if r["epochs"] else 0
            if ep != 30 or len(r["epochs"]) != 30:
                problems.append(f"{ds} seed {s}: log trained {len(r['epochs'])} epochs (max {ep}), expected 30")
            rw = wb.get(r["id"]) if r["id"] else None
            full = cfg_ep = cfg_do = cfg_bs = None
            if rw:
                full = rw.get("summary", {}).get("test/auc")
                cfg = rw.get("config", {})
                cfg_ep, cfg_do, cfg_bs = cfg.get("epochs"), cfg.get("dropout"), cfg.get("batch_size")
                if cfg_ep != 30:
                    problems.append(f"{ds} seed {s}: W&B config epochs = {cfg_ep}")
                if rw.get("name") and r["run"] and rw["name"] != r["run"]:
                    problems.append(f"{ds} seed {s}: W&B name {rw['name']} != log run {r['run']}")
            else:
                if r["id"]:
                    need_ids.append(r["id"])
                problems.append(f"{ds} seed {s}: no full-precision W&B record (id {r['id']})")
            if full is not None and abs(full - r["auc4"]) > LOG_TOL:
                problems.append(f"{ds} seed {s}: log {r['auc4']:.4f} vs W&B {full!r}")
            if full is not None:
                table[ds][s] = full
            fs = repr(full) if full is not None else "-"
            print(f"{ds:12s} {s:4d}  {ep:6d}  {r['auc4']:.4f}    {fs:24s}  {cfg_ep!s:>10s} {cfg_do!s:>11s} {cfg_bs!s:>9s}  {r['id']}")

    print("\n--- appendix rows (4 dp per seed; mean and population SD from unrounded values) ---")
    for ds, label in DATASETS:
        vals = table[ds]
        if len(vals) != len(SEEDS):
            print(f"% {label}: incomplete ({len(vals)}/6 full-precision values), row not emitted")
            continue
        v = [vals[s] for s in SEEDS]
        mean, sd = mean_sd(v)
        cells = " & ".join(str(q(x, "0.0001")) for x in v)
        print(f"{label} & SAINT$+$ & {cells} & {q(mean, '0.0001')} $\\pm$ {q(sd, '0.0001')} \\\\")

    print("\n--- main-table values (3 dp, rounded once from the full-precision mean) ---")
    for ds, label in DATASETS:
        vals = table[ds]
        if len(vals) != len(SEEDS):
            continue
        mean, _ = mean_sd([vals[s] for s in SEEDS])
        m4 = q(mean, "0.0001")
        flag = "" if str(m4) == RECORDED_4DP[ds] else f"   <-- RESULTS.md records {RECORDED_4DP[ds]}"
        if flag:
            problems.append(f"{ds}: 4 dp mean {m4} disagrees with RESULTS.md {RECORDED_4DP[ds]}")
        print(f"{label:17s} mean {mean:.8f}  4dp {m4}  3dp {q(mean, '0.001')}{flag}")

    if need_ids:
        Path(a.ids_out).write_text("\n".join(need_ids) + "\n")
        print(f"\nwrote {len(need_ids)} ids without a W&B record to {a.ids_out}")

    print("\nPROBLEMS:" if problems else "\nno problems: 42 runs, 30 epochs each, log and W&B agree")
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
