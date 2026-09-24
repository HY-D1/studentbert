#!/usr/bin/env python3
"""Pretraining-exposure test for the in-domain preference (RESULTS.md 12.4 and 12.5).

Every Task 2 estimator scores target learners from the train split, which the in-domain encoder saw
during masked pretraining, while the gold is test AUC on learners no encoder saw. This report
compares three sample protocols on the same candidates and seeds:
  train       the earlier scores (tg1_logme_kt_7x7_*, tg1_task2_kt_*): up to 3,000 train learners
  trainmatch  a train draw with as many learners as the validation split, capped at 3,000
  val         the validation learners. No encoder saw them: pretrain_edubert.py reads the train
              split and keeps its checkpoint by train loss. The fine-tunes use them only to pick
              their best epoch, and the gold is the test split.
trainmatch against val separates exposure from sample size. The statistic is the in-domain margin:
the in-domain score minus the best other pretrained score, on one sample at one seed. It is a
difference between candidates, so a split shift that moves every candidate alike cancels. Scratch
is left out of the margin because the preference being tested is among pretrained encoders.

Decision rule, fixed in this file before any trainmatch or val score existed. Per estimator and
target, the in-domain encoder is "preferred" on a protocol when its margin is positive (strictly
top) on a majority of seeds (2 of 3):
  no preference   not preferred on train: nothing to explain on this target
  sample size     preferred on train, not on trainmatch: fewer learners alone removes it, so this
                  test cannot attribute it to exposure
  exposure        preferred on trainmatch, not on val
  representation  preferred on trainmatch and on val
Margins are compared within one estimator only; their units differ between estimators. Where a
trainmatch draw is the train draw (same sample fingerprint: EdNet and Junyi, whose validation
splits exceed 3,000 learners), both must give the same scores; the report prints the largest
difference as a determinism check.

  PYTHONPATH=. python3 analysis/exposure_report.py \\
      --scores tg1_logme_kt_7x7_*.jsonl tg1_task2_kt_*.jsonl tg1_exposure_*.jsonl \\
      --out-prefix exposure_20260925/exposure
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

PROTOCOLS = ("train", "trainmatch", "val")
SEEDS = {42, 1, 2}
CKPT = re.compile(r"^edubert_([a-z0-9]+)_pretrain_full_encoder\.pt$")
FAILURE_TARGETS = ("assist2017", "algebra2006")


def q(x: float | None, places: str = "0.000001") -> str:
    """Display rounding only (half away from zero, once); the TSVs keep full precision."""
    if x is None:
        return ""
    return f"{Decimal(repr(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP):+}"


def protocol_of(md: dict) -> str:
    """score_tag if the scorer set one, else the split: an untagged val score is still val."""
    tag = md.get("score_tag")
    if tag:
        return tag
    return md.get("score_split") or "train"


def load(paths: list[str], budget: str) -> dict:
    """{(estimator, target, protocol, seed): {candidate: (score, learners, positions, sample)}}.

    Duplicate lines (the 3-target files re-scored inside the 7 x 7 grid, or an appended rerun)
    are kept once when they agree to 1e-9 and stop the run when they do not: a duplicate is never
    chosen by value.
    """
    out: dict = defaultdict(dict)
    clash = []
    for p in paths:
        for ln in open(p):
            r = json.loads(ln)
            md = r["metadata"]
            n_req = md.get("n_students_requested")
            if (md.get("target_budget") or (f"n{n_req}" if n_req else "full_split")) != budget:
                continue
            if r["candidate"] == "scratch":
                cand = "scratch"
            elif m := CKPT.match(r["candidate"]):
                cand = f"src:{m.group(1)}"
            else:
                continue
            key = (r["estimator"], r["target"], protocol_of(md), int(r["seed"]))
            val = (float(r["score"]), md.get("n_students"), md.get("positions_used"),
                   md.get("sample_fingerprint"))
            old = out[key].get(cand)
            if old is not None and abs(old[0] - val[0]) > 1e-9 * max(1.0, abs(val[0])):
                clash.append(f"{key} {cand}: {old[0]!r} against {val[0]!r} ({p})")
            out[key].setdefault(cand, val)
    if clash:
        raise SystemExit("disagreeing duplicate scores:\n  " + "\n  ".join(clash))
    return out


def check_complete(data: dict, estimators: list[str], n_candidates: int) -> list[str]:
    """Every estimator x target x protocol must hold seeds 42, 1, 2 with n_candidates each."""
    targets = sorted({k[1] for k in data if k[0] in estimators})
    bad = []
    for est in estimators:
        for tgt in targets:
            for prot in PROTOCOLS:
                seeds = {k[3] for k in data if k[:3] == (est, tgt, prot)}
                if seeds != SEEDS:
                    bad.append(f"{est} {tgt} {prot}: seeds {sorted(seeds)}, expected [1, 2, 42]")
                    continue
                for s in sorted(seeds):
                    n = len(data[(est, tgt, prot, s)])
                    if n != n_candidates:
                        bad.append(f"{est} {tgt} {prot} seed {s}: {n} of {n_candidates} "
                                   "candidates")
    return bad


def margin_row(scores: dict, target: str) -> dict:
    """In-domain margin, rank and top pick among pretrained candidates at one seed."""
    own = f"src:{target}"
    pre = {c: v[0] for c, v in scores.items() if c != "scratch"}
    if own not in pre:
        raise SystemExit(f"{target}: the in-domain candidate {own} was not scored")
    others = {c: s for c, s in pre.items() if c != own}
    best_other = max(others, key=others.get)
    above = sum(s > pre[own] for s in others.values())
    ties = sum(s == pre[own] for s in others.values())
    top = max(pre, key=pre.get)
    return {"margin": pre[own] - others[best_other], "indomain_rank": 1 + above + ties / 2,
            "top_pick": top, "runner_up": best_other, "n_students": scores[own][1],
            "positions": scores[own][2]}


def verdict(preferred: dict) -> str:
    if not preferred["train"]:
        return "no preference"
    if not preferred["trainmatch"]:
        return "sample size"
    return "representation" if preferred["val"] else "exposure"


def analyse(data: dict, estimators: list[str]) -> tuple[list[dict], list[dict], list[str]]:
    per_seed, summary, determinism = [], [], []
    targets = sorted({k[1] for k in data if k[0] in estimators})
    need = len(SEEDS) // 2 + 1
    for est in estimators:
        for tgt in targets:
            rows = {}
            for prot in PROTOCOLS:
                for s in sorted(SEEDS):
                    r = margin_row(data[(est, tgt, prot, s)], tgt)
                    rows[(prot, s)] = r
                    per_seed.append({"estimator": est, "target": tgt, "protocol": prot,
                                     "seed": s, **r})
            k = {p: sum(rows[(p, s)]["margin"] > 0 for s in SEEDS) for p in PROTOCOLS}
            shrink = [rows[("trainmatch", s)]["margin"] - rows[("val", s)]["margin"]
                      for s in sorted(SEEDS)]
            summary.append({
                "estimator": est, "target": tgt,
                **{f"n_{p}": rows[(p, 42)]["n_students"] for p in PROTOCOLS},
                **{f"preferred_{p}": f"{k[p]}/{len(SEEDS)}" for p in PROTOCOLS},
                **{f"margin_{p}": st.mean(rows[(p, s)]["margin"] for s in SEEDS)
                   for p in PROTOCOLS},
                "shrink_mean": st.mean(shrink),
                "shrink_positive": f"{sum(x > 0 for x in shrink)}/{len(SEEDS)}",
                "verdict": verdict({p: k[p] >= need for p in PROTOCOLS})})
            for s in sorted(SEEDS):
                a, b = data[(est, tgt, "train", s)], data[(est, tgt, "trainmatch", s)]
                fp = a[f"src:{tgt}"][3]
                if fp and fp == b[f"src:{tgt}"][3]:
                    diff = max(abs(a[c][0] - b[c][0]) for c in a if c in b)
                    determinism.append(f"{est} {tgt} seed {s}: same sample {fp} "
                                       f"({a[f'src:{tgt}'][1]} learners), largest score "
                                       f"difference {diff:.3e}")
    return per_seed, summary, determinism


def write_tsv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--estimators", nargs="+", default=["hscore_kt_causal", "logme_kt_causal"])
    ap.add_argument("--budget", default="n3000")
    ap.add_argument("--n-candidates", type=int, default=8,
                    help="scratch plus the seven full encoders")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    data = load(a.scores, a.budget)
    bad = check_complete(data, a.estimators, a.n_candidates)
    if bad:
        print("INCOMPLETE, nothing written:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    per_seed, summary, determinism = analyse(data, a.estimators)
    Path(a.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    write_tsv(f"{a.out_prefix}_seeds.tsv", per_seed)
    write_tsv(f"{a.out_prefix}_summary.tsv", summary)

    L = [f"# Exposure test ({a.budget} cells, pretrained candidates, seeds 42 1 2)\n",
         "Margin: in-domain score minus the best other pretrained score (positive: in-domain "
         "preferred). Shrink: trainmatch margin minus val margin.\n",
         "| estimator | target | learners train / match / val | preferred train | match | val |"
         " margin train | match | val | shrink (seeds > 0) | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s['estimator']} | {s['target']} | {s['n_train']} / {s['n_trainmatch']} / "
                 f"{s['n_val']} | {s['preferred_train']} | {s['preferred_trainmatch']} | "
                 f"{s['preferred_val']} | {q(s['margin_train'])} | {q(s['margin_trainmatch'])} | "
                 f"{q(s['margin_val'])} | {q(s['shrink_mean'])} ({s['shrink_positive']}) | "
                 f"{s['verdict']} |")
    L.append("\n## The two failure targets of RESULTS.md 12.4\n")
    for est in a.estimators:
        got = {s["target"]: s["verdict"] for s in summary if s["estimator"] == est}
        L.append(f"- {est}: " + ", ".join(f"{t} {got.get(t, 'not scored')}"
                                          for t in FAILURE_TARGETS))
    L.append("\n## Determinism (trainmatch draws identical to the train draw)\n")
    L += [f"- {d}" for d in determinism] or ["- none: every trainmatch draw is smaller"]
    Path(f"{a.out_prefix}_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
