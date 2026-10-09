from __future__ import annotations

"""Budget-matched source-scale controls for Paper A (Prof. Hazra's items 4 and 5, 2026-10-08).

The 108 fine-tunes of slurm/generators/gen_budget_control.sh: EdNet encoders whose number of
optimizer updates is matched (49,153 learners x 20 epochs = 7,700 updates, Junyi's; 15,000 x 33 and
5,000 x 96 = about 3,850, the 49,153 x 10 rung's), 3 encoder draws each, fine-tuned at N=3000 on
ASSISTments 2017 and Junyi with seeds 1 2 3 4 5 42.

  ids     parse every *_bud_* log, check that each of the 108 runs has exactly one finished log,
          and write the W&B run ids for analysis/wandb_export_by_id.py
  report  pair every run with the Track B scratch run of the same target and seed, add the
          10-epoch ladder (executions.tsv, track S11, and Track B src:ednet for 353,597 draw 42)
          and the Junyi encoder on ASSISTments 2017, and write per-run values, per-condition
          gains and paired differences

Statistics, as in RESULTS.md 11: a draw's gain is its 6-seed mean of (run - scratch at that seed);
a condition's gain is the mean over its draws; the draw spread is the encoder-build variance; the
95% interval resamples the 6 seeds of the across-draw mean (20,000 resamples, random.Random(0),
boot_mean_ci of analysis/build_transfer_benchmark.py), so seed and build variance are reported
apart. Differences between conditions pair the across-draw means by seed. Test AUC is the W&B
full-precision summary, checked against the log's 4 dp value.

  PYTHONPATH=. python analysis/budget_control.py ids --logdir . --out benchmark_final/budget_ids.txt
  PYTHONPATH=. python analysis/budget_control.py report --logdir . --wandb wandb_by_id.jsonl \
      --executions benchmark_final/executions.tsv --out benchmark_final/budget_control
"""

import argparse
import csv
import hashlib
import random
import re
import statistics as st
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from analysis.build_transfer_benchmark import WB_FINISH, boot_mean_ci
from analysis.finetune_trajectories import load_wandb, parse_log

SPECS = ((49153, 20), (15000, 33), (5000, 96))
DRAWS = (42, 1, 2)
SEEDS = (1, 2, 3, 4, 5, 42)
TARGETS = {"assist": "assist2017", "junyi": "junyi"}
LADDER = (5000, 15000, 49153, 150000, 353597)
BOOTS = 20000
NAME = re.compile(r"^kt_(assist|junyi)_fromednet_n(\d+)e(\d+)(?:d(\d+))?"
                  r"_bud_n3000_seed(\d+)_(\d+)\.log$")
# RESULTS.md 11 and 2.1, rounded once; the report stops if the inputs no longer reproduce them.
RESULTS_GAINS = {("assist2017", "ladder", 5000, 10): "-0.0018",
                 ("junyi", "ladder", 5000, 10): "-0.0037",
                 ("assist2017", "ladder", 15000, 10): "+0.0002",
                 ("junyi", "ladder", 15000, 10): "-0.0023",
                 ("assist2017", "ladder", 49153, 10): "+0.0177",
                 ("junyi", "ladder", 49153, 10): "+0.0014",
                 ("assist2017", "ladder", 150000, 10): "+0.0241",
                 ("junyi", "ladder", 150000, 10): "+0.0043",
                 ("assist2017", "ladder", 353597, 10): "+0.0259",
                 ("junyi", "ladder", 353597, 10): "+0.0062",
                 ("assist2017", "junyi_source", 49153, 20): "+0.0214"}


def q(x, sign: bool = True) -> str:
    d = Decimal(repr(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def collect(logdir: str) -> dict:
    """runs[(target, size, epochs, draw, seed)] = parsed finished log; aborts on gaps or clashes."""
    found: dict = {}
    for p in sorted(Path(logdir).glob("kt_*_fromednet_n*e*_bud_n3000_seed*_*.log")):
        m = NAME.match(p.name)
        if not m:
            continue
        tok, size, ep, draw, seed, job = m.groups()
        key = (TARGETS[tok], int(size), int(ep), int(draw or 42), int(seed))
        text = p.read_text(errors="ignore")
        try:
            lg = parse_log(text, 20)
        except ValueError:
            continue  # an unfinished attempt; the expected-set check below catches a gap
        wb = WB_FINISH.findall(text)
        if len(wb) != 1:
            sys.exit(f"ABORT: {p.name} has {len(wb)} W&B finish lines, expected 1")
        rec = {"log": p.name, "job": int(job), "test_log": lg["test_log"], "run": wb[0][0],
               "wandb_id": wb[0][3]}
        if key in found:
            old = found[key]
            if abs(old["test_log"] - rec["test_log"]) > Decimal("0.0005"):
                sys.exit(f"ABORT: {key} has two finished logs that disagree: {old['log']} "
                         f"{old['test_log']} and {p.name} {rec['test_log']}")
            rec = min(old, rec, key=lambda r: r["job"])
        found[key] = rec
    expected = {(t, s, e, d, sd) for t in TARGETS.values() for s, e in SPECS for d in DRAWS
                for sd in SEEDS}
    missing, extra = sorted(expected - set(found)), sorted(set(found) - expected)
    if missing or extra:
        sys.exit(f"ABORT: {len(found)} finished runs; missing {missing[:6]} extra {extra[:6]}")
    return found


def load_executions(path: str) -> tuple[dict, dict, dict]:
    """scratch[t][seed], ladder[(t, size, draw)][seed], and the Junyi encoder on ASSISTments."""
    scratch: dict = {t: {} for t in TARGETS.values()}
    ladder: dict = {}
    junyi: dict = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["metric"] != "test_auc" or r["valid_for_primary_analysis"] != "yes":
                continue
            t, s = r["target_dataset"], int(r["finetune_seed"] or 0)
            if t not in scratch:
                continue
            if r["track"] == "B" and r["budget"] == "n3000":
                if r["candidate"] == "scratch":
                    scratch[t][s] = float(r["value"])
                elif r["candidate"] == "src:ednet":
                    ladder.setdefault((t, 353597, 42), {})[s] = float(r["value"])
                elif r["candidate"] == "src:junyi" and t == "assist2017":
                    junyi[s] = float(r["value"])
            elif r["track"] == "S11":
                m = re.match(r"^ednet_n(\d+)_d(\d+)$", r["candidate"])
                if m and int(m.group(1)) in LADDER:
                    k = (t, int(m.group(1)), int(m.group(2)))
                    ladder.setdefault(k, {})[s] = float(r["value"])
    for t in TARGETS.values():
        if set(scratch[t]) != set(SEEDS):
            sys.exit(f"ABORT: scratch seeds for {t}: {sorted(scratch[t])}")
        for size in LADDER:
            for d in DRAWS:
                if set(ladder.get((t, size, d), {})) != set(SEEDS):
                    sys.exit(f"ABORT: ladder {t} {size} draw {d} is not 6 seeds")
    if set(junyi) != set(SEEDS):
        sys.exit("ABORT: Junyi encoder on assist2017 is not 6 seeds")
    return scratch, ladder, junyi


def condition(per_draw: dict, scratch: dict, idx: list) -> dict:
    """per_draw[draw][seed] = test AUC; gains against scratch at the same seed."""
    gains = {d: {s: v[s] - scratch[s] for s in SEEDS} for d, v in per_draw.items()}
    draw_mean = {d: st.mean(g.values()) for d, g in gains.items()}
    series = [st.mean(gains[d][s] for d in gains) for s in SEEDS]
    lo, hi = boot_mean_ci(series, idx)
    spread = max(draw_mean.values()) - min(draw_mean.values()) if len(draw_mean) > 1 else None
    return {"gain": st.mean(draw_mean.values()), "lo": lo, "hi": hi, "series": series,
            "per_draw": draw_mean, "spread": spread, "pos": sum(x > 0 for x in series)}


def report(a) -> None:
    runs = collect(a.logdir)
    wb = load_wandb(a.wandb)
    scratch, ladder, junyi = load_executions(a.executions)
    rng = random.Random(0)
    idx = [[rng.randrange(len(SEEDS)) for _ in SEEDS] for _ in range(BOOTS)]
    value = {}
    rows = []
    for key, r in sorted(runs.items()):
        rec = wb.get(r["wandb_id"])
        if not rec:
            sys.exit(f"ABORT: no W&B record for {r['log']} ({r['wandb_id']})")
        if rec.get("name") != r["run"]:
            sys.exit(f"ABORT: W&B {r['wandb_id']} is {rec.get('name')}, the log ran {r['run']}")
        v = (rec.get("summary") or {}).get("test/auc")
        if v is None or Decimal(repr(v)).quantize(Decimal("0.0001"),
                                                   ROUND_HALF_UP) != r["test_log"]:
            sys.exit(f"ABORT: {r['log']} test AUC {r['test_log']} in the log, {v} in W&B")
        value[key] = float(v)
        t, size, ep, d, s = key
        rows.append({"target": t, "source_learners": size, "epochs": ep,
                     "updates": (size + 127) // 128 * ep, "draw": d, "seed": s,
                     "test_auc": repr(float(v)), "scratch": repr(scratch[t][s]),
                     "gain": repr(float(v) - scratch[t][s]), "wandb_id": r["wandb_id"],
                     "log": r["log"]})
    C = {}
    for t in TARGETS.values():
        for size in LADDER:
            C[(t, "ladder", size, 10)] = condition({d: ladder[(t, size, d)] for d in DRAWS},
                                                  scratch[t], idx)
        for size, ep in SPECS:
            C[(t, "matched", size, ep)] = condition(
                {d: {s: value[(t, size, ep, d, s)] for s in SEEDS} for d in DRAWS}, scratch[t], idx)
    C[("assist2017", "junyi_source", 49153, 20)] = condition({42: junyi}, scratch["assist2017"],
                                                              idx)
    bad = [f"{k}: {q(C[k]['gain'])} != {want}" for k, want in RESULTS_GAINS.items()
           if q(C[k]["gain"]) != want]
    if bad:
        sys.exit("ABORT: the inputs no longer reproduce RESULTS.md: " + "; ".join(bad))

    def diff(k1, k2) -> dict:
        d = [x - y for x, y in zip(C[k1]["series"], C[k2]["series"])]
        lo, hi = boot_mean_ci(d, idx)
        return {"a": k1, "b": k2, "diff": st.mean(d), "lo": lo, "hi": hi,
                "pos": sum(x > 0 for x in d)}

    D = []
    for t in TARGETS.values():
        D += [diff((t, "matched", 49153, 20), (t, "ladder", 49153, 10)),
              diff((t, "matched", 15000, 33), (t, "ladder", 15000, 10)),
              diff((t, "matched", 5000, 96), (t, "ladder", 5000, 10)),
              diff((t, "matched", 15000, 33), (t, "ladder", 49153, 10)),
              diff((t, "matched", 5000, 96), (t, "ladder", 49153, 10))]
    D += [diff(("assist2017", "junyi_source", 49153, 20), ("assist2017", "matched", 49153, 20)),
          diff(("assist2017", "junyi_source", 49153, 20), ("assist2017", "ladder", 49153, 10))]

    out = Path(a.out)
    files = {}
    srows = []
    for k, c in sorted(C.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        t, kind, size, ep = k
        srows.append({"target": t, "condition": kind, "source_learners": size, "epochs": ep,
                      "updates": (size + 127) // 128 * ep, "gain": repr(c["gain"]),
                      "ci_lo": repr(c["lo"]), "ci_hi": repr(c["hi"]),
                      "seeds_positive": f"{c['pos']}/6",
                      "per_draw": ";".join(f"d{d}:{v:+.6f}"
                                           for d, v in sorted(c["per_draw"].items())),
                      "draw_spread": "" if c["spread"] is None else repr(c["spread"])})
    drows = [{"target": x["a"][0], "a": " ".join(map(str, x["a"][1:])),
              "b": " ".join(map(str, x["b"][1:])), "diff": repr(x["diff"]),
              "ci_lo": repr(x["lo"]),
              "ci_hi": repr(x["hi"]), "seeds_positive": f"{x['pos']}/6"} for x in D]
    for suffix, data in (("_runs.tsv", rows), ("_summary.tsv", srows), ("_diffs.tsv", drows)):
        p = out.with_name(out.name + suffix)
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]), delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(data)
        files[p.name] = hashlib.md5(p.read_bytes()).hexdigest()
    lines = ["# Budget-matched source-scale controls", "",
             "| target | condition | learners | epochs | updates | gain [95% CI] | seeds + | "
             "draw spread |", "|---|---|---|---|---|---|---|---|"]
    for r in srows:
        c = C[(r["target"], r["condition"], r["source_learners"], r["epochs"])]
        sp = "" if c["spread"] is None else q(c["spread"], sign=False)
        lines.append(f"| {r['target']} | {r['condition']} | {r['source_learners']:,} | "
                     f"{r['epochs']} | {r['updates']:,} | {q(c['gain'])} [{q(c['lo'])}, "
                     f"{q(c['hi'])}] | {r['seeds_positive']} | {sp} |")
    lines += ["", "| target | a minus b | difference [95% CI] | seeds + |", "|---|---|---|---|"]
    for x in D:
        lines.append(f"| {x['a'][0]} | {' '.join(map(str, x['a'][1:]))} minus "
                     f"{' '.join(map(str, x['b'][1:]))} | {q(x['diff'])} [{q(x['lo'])}, "
                     f"{q(x['hi'])}] | {x['pos']}/6 |")
    lines += ["", "Files: " + ", ".join(f"{k} {v}" for k, v in files.items())]
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} runs reproduce the log values in W&B; RESULTS.md 11 and 2.1 reproduce; "
          f"wrote {', '.join(files)} and {rp.name} md5 {hashlib.md5(rp.read_bytes()).hexdigest()}")


def ids(a) -> None:
    runs = collect(a.logdir)
    out = Path(a.out)
    out.write_text("".join(f"{r['wandb_id']} {r['run']}\n" for _k, r in sorted(runs.items())))
    print(f"{len(runs)} finished runs; wrote {len(runs)} W&B ids to {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("ids")
    p1.add_argument("--logdir", required=True)
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("report")
    p2.add_argument("--logdir", required=True)
    p2.add_argument("--wandb", required=True)
    p2.add_argument("--executions", required=True)
    p2.add_argument("--out", required=True, help="prefix for the _runs, _summary, _diffs files")
    a = ap.parse_args(argv)
    ids(a) if a.cmd == "ids" else report(a)


if __name__ == "__main__":
    main()
