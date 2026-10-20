from __future__ import annotations

"""Supplement to the Option 1 NLP evaluation (RESULTS.md 12.15), for the framework decision.

Added after the 12.15 results were seen, at Prof. Hazra's request to finish the comparison and
judge whether the early fine-tuning rule is strong enough for the framework. It changes no rule,
threshold or model; it describes the same decisions (benchmark_final/nlp_option1_recs.tsv) with
uncertainty and cost:

  - for the fixed rule and the learned head-pair model at epochs 1, 2 and 3: reversals caught,
    false alarms and coverage with Wilson 95% intervals, and selective regret with a task-cluster
    bootstrap interval (the 5 tasks resampled with replacement, 20,000 times, random.Random(0)),
    since recommendations from one task are not independent;
  - the rule's calibration as two observed rates: reversals among the recommendations it flags and
    among those it follows;
  - how often the rule's decisions agree between budgets;
  - compute: the measured GPU-minutes of the early fine-tunes per recommendation, against a lower
    bound for fine-tuning all 7 encoders in full, one epoch over each task's full training split
    (the measured training time per example on the 10,000-example sample, scaled to the split
    size; validation excluded). The published fine-tunes ran longer than one epoch, so the real
    saving is larger than this ratio shows.

Needs the datasets package for the training-split sizes (envs/t4).

  HF_HOME=/projects/algl/dai.hany/hf_cache PYTHONPATH=. envs/t4/bin/python \
      analysis/nlp_option1_supplement.py --early_ft /projects/algl/dai.hany/task4/early_ft \
      --recs benchmark_final/nlp_option1_recs.tsv --out benchmark_final/nlp_option1_supplement
"""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

RECS_MD5 = "8a146de30d1cc103d8df324f983d99cb"
TASKS = ("agnews", "mnli", "qnli", "rte", "airline")
HUB = {"agnews": ("ag_news", None), "mnli": ("glue", "mnli"), "qnli": ("glue", "qnli"),
       "rte": ("glue", "rte")}
AIRLINE_TRAIN = "/projects/algl/dai.hany/task4/data/airline/airline-train.csv"
BOOTS = 20000


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cluster_boot(recs: list[dict], abstain, boots: int = BOOTS) -> tuple[float, float]:
    """Percentile interval of the selective regret, resampling whole tasks."""
    by = {t: [r for r in recs if r["target"] == t] for t in TASKS}
    rng = random.Random(0)
    vals = []
    for _ in range(boots):
        pick = [rng.choice(TASKS) for _ in TASKS]
        fol = [float(r["regret"]) for t in pick for r in by[t] if not abstain(r)]
        if fol:
            vals.append(st.mean(fol))
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def summarize(recs: list[dict], abstain) -> dict:
    rev = [r for r in recs if r["reversed"] == "1"]
    ok = [r for r in recs if r["reversed"] == "0"]
    caught = sum(abstain(r) for r in rev)
    fa = sum(abstain(r) for r in ok)
    fol = [r for r in recs if not abstain(r)]
    flagged = [r for r in recs if abstain(r)]
    return {"caught": caught, "n_rev": len(rev), "fa": fa, "n_ok": len(ok),
            "followed": len(fol), "n": len(recs),
            "sel": st.mean(float(r["regret"]) for r in fol) if fol else float("nan"),
            "sel_ci": cluster_boot(recs, abstain),
            "rev_among_flagged": (sum(r["reversed"] == "1" for r in flagged), len(flagged)),
            "rev_among_followed": (sum(r["reversed"] == "1" for r in fol), len(fol))}


def split_sizes() -> dict:
    from datasets import load_dataset

    sizes = {t: len(load_dataset(h, c, split="train")) for t, (h, c) in HUB.items()}
    with open(AIRLINE_TRAIN, newline="") as fh:
        sizes["airline"] = sum(1 for _ in csv.DictReader(fh))
    return sizes


def costs(root: str, recs: list[dict], sizes: dict) -> dict:
    """Mean GPU-minutes per recommendation: early budgets k = 1..3, and one full-data epoch."""
    files = sorted(Path(root).glob("*/*.json"))
    runs = [json.loads(f.read_text()) for f in files]
    if len(runs) != 210:
        sys.exit(f"ABORT: {len(runs)} early fine-tunes, expected 210")
    by = {}
    for r in runs:
        by.setdefault((r["task"], r["pooling"], r["seed"]), []).append(r)
    early = {k: [] for k in (1, 2, 3)}
    full = []
    for key, rs in by.items():
        if len(rs) != 7:
            sys.exit(f"ABORT: {key} has {len(rs)} encoders")
        for k in early:
            early[k].append(sum(sum(e["epoch_s"] for e in r["epochs"][:k]) for r in rs) / 60)
        full.append(sum(st.mean(e["train_s"] for e in r["epochs"]) * sizes[r["task"]]
                        / r["n_train"] for r in rs) / 60)
    return {"early": {k: st.mean(v) for k, v in early.items()}, "full_epoch": st.mean(full)}


def pct(x: float) -> str:
    return "nan" if x != x else f"{100 * x:.0f}%"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--early_ft", required=True)
    ap.add_argument("--recs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = Path(a.recs)
    if hashlib.md5(p.read_bytes()).hexdigest() != RECS_MD5:
        sys.exit(f"ABORT: {p} is not the 12.15 recommendations file")
    allrecs = list(csv.DictReader(p.read_text().splitlines(), delimiter="\t"))
    sizes = split_sizes()
    rule_ab = lambda r: float(r["risk_noisy_or"]) > 0.5  # noqa: E731
    learned_ab = lambda r: float(r["risk_noisy_or"]) > float(r["tau_fold"])  # noqa: E731
    c = costs(a.early_ft, [r for r in allrecs if r["method"] == "fixed rule"], sizes)
    L = ["# Option 1 supplement: intervals, calibration, budget agreement and compute", "",
         "Added after the 12.15 results were seen; it describes the same decisions and changes no "
         "rule, threshold or model. Intervals: Wilson 95% for counts, task-cluster bootstrap for "
         "selective regret (5 tasks, so the intervals are wide).", "",
         "| method | epochs | caught [95% CI] | false alarms [95% CI] | followed | "
         "selective regret "
         "[95% CI] | GPU-minutes per recommendation |", "|---|---|---|---|---|---|---|"]
    rows, dec = [], {}
    for method, ab in (("fixed rule", rule_ab), ("learned head-pair", learned_ab)):
        for k in ("1", "2", "3"):
            recs = [r for r in allrecs if r["method"] == method and r["epochs"] == k]
            s = summarize(recs, ab)
            dec[(method, k)] = {(r["target"], r["est_seed"]): ab(r) for r in recs}
            cl, ch = wilson(s["caught"], s["n_rev"])
            fl, fh = wilson(s["fa"], s["n_ok"])
            L.append(f"| {method} | {k} | {s['caught']}/{s['n_rev']} [{pct(cl)}, {pct(ch)}] | "
                     f"{s['fa']}/{s['n_ok']} [{pct(fl)}, {pct(fh)}] | {s['followed']}/{s['n']} | "
                     f"{s['sel']:.2f} [{s['sel_ci'][0]:.2f}, {s['sel_ci'][1]:.2f}] | "
                     f"{c['early'][int(k)]:.1f} |")
            rows.append({"method": method, "epochs": k, **{x: s[x] for x in
                         ("caught", "n_rev", "fa", "n_ok", "followed", "n", "sel")},
                         "sel_lo": s["sel_ci"][0], "sel_hi": s["sel_ci"][1]})
    always = [r for r in allrecs if r["method"] == "fixed rule" and r["epochs"] == "3"]
    s_all = summarize(always, lambda r: False)
    L += ["", f"Always following H-score: regret {s_all['sel']:.2f} points "
          f"[{s_all['sel_ci'][0]:.2f}, {s_all['sel_ci'][1]:.2f}] over all 30 recommendations.", ""]
    L += ["Rule calibration (observed reversal rate):", "",
          "| epochs | among flagged | among followed |", "|---|---|---|"]
    for k in ("1", "2", "3"):
        recs = [r for r in allrecs if r["method"] == "fixed rule" and r["epochs"] == k]
        s = summarize(recs, rule_ab)
        a1, n1 = s["rev_among_flagged"]
        a2, n2 = s["rev_among_followed"]
        L.append(f"| {k} | {a1}/{n1} ({pct(a1 / n1 if n1 else float('nan'))}) | "
                 f"{a2}/{n2} ({pct(a2 / n2 if n2 else float('nan'))}) |")
    agree13 = sum(dec[("fixed rule", "1")][x] == dec[("fixed rule", "3")][x]
                  for x in dec[("fixed rule", "3")])
    agree23 = sum(dec[("fixed rule", "2")][x] == dec[("fixed rule", "3")][x]
                  for x in dec[("fixed rule", "3")])
    L += ["", f"Rule decisions agree between epoch 1 and epoch 3 on {agree13} of 30 "
          "recommendations, "
          f"between epoch 2 and epoch 3 on {agree23} of 30.", ""]
    L += ["Compute per recommendation (GPU-minutes, 7 encoders):", "",
          "| budget | GPU-minutes | share of one full-data epoch |", "|---|---|---|"]
    for k in (1, 2, 3):
        L.append(f"| early, {k} epoch{'s' if k > 1 else ''} on the 10,000-example sample | "
                 f"{c['early'][k]:.1f} | {pct(c['early'][k] / c['full_epoch'])} |")
    L.append(f"| full fine-tuning, one epoch over the full training split (lower bound) | "
             f"{c['full_epoch']:.1f} | 100% |")
    L += ["", "Training-split sizes: " + ", ".join(f"{t} {sizes[t]:,}" for t in TASKS) + ". The "
          "full-data figure scales the measured training time per example; the published "
          "fine-tunes "
          "ran several epochs with early stopping, so it understates their cost."]
    out = Path(a.out)
    tp = out.with_name(out.name + ".tsv")
    with tp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    L += ["", f"Files: {tp.name} {hashlib.md5(tp.read_bytes()).hexdigest()}"]
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print(f"wrote {tp.name} and {rp.name} md5 {hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
