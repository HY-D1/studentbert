#!/usr/bin/env python3
"""Build the canonical transfer benchmark (MRAP Task 1) from the raw logs.

One row per execution and metric. Each row carries its log file, Slurm job id, campaign, job
state, node and GPU (sacct plus the sinfo GRES map), the W&B run id printed in that same log (so a
duplicated run name can never join to the wrong run), the encoder the log says it loaded, and on
the second pass the full-precision W&B value.

Filename order decides nothing. Every execution stays in executions.tsv, and each duplicate
family is resolved by a written rule (resolve_duplicates):
  1. a superseded campaign loses to its replacement: the pinned N=3000 grid over W6, and W6 over
     W5 for ASSISTments dropout (RESULTS.md 2.1 and 8.1);
  2. a copy whose job did not end COMPLETED loses to one that did;
  3. copies within --dup-tol keep the lowest job id; copies further apart are all excluded as
     unresolved. A copy is never chosen by its value.
Only rows marked valid_for_primary_analysis enter the per-track tables. Nothing is rounded here;
report.md prints 4 dp, half up.

The banner and metric regexes come from analysis/inventory_runs.py, so the tools cannot drift.

Pass 1 (stdlib only, fine on a login node):
  PYTHONPATH=. python3 analysis/build_transfer_benchmark.py --logdir . \\
      --sacct sacct_20260922.txt --gres nodes_gres_20260922.txt --results-md RESULTS.md \\
      --outdir benchmark_20260922
Pass 2 adds: --wandb wandb_by_id.jsonl --encoders encoder_configs.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

try:
    from analysis import inventory_runs as inv
    from analysis.paired_bootstrap_objective import percentile
except ModuleNotFoundError:  # run without PYTHONPATH=.: the repo root is the parent of analysis/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis import inventory_runs as inv
    from analysis.paired_bootstrap_objective import percentile

SEEDS6 = (42, 1, 2, 3, 4, 5)
SEEDS3 = (42, 1, 2)
TRACK_METRIC = {"A": "test_auc", "B": "test_auc", "S11": "test_auc", "C-ns": "test_macro_ovr_auc",
                "C-drop": "test_auc", "probe7": "test_probe_acc", "probe2": "test_probe_acc",
                "baseline": "test_auc"}
WANDB_KEY = {"test_auc": "test/auc", "test_ece": "test/ece", "test_probe_acc": "test/probe_acc",
             "test_top_1_acc": "test/top1", "test_top_5_acc": "test/top5",
             "test_macro_ovr_auc": "test/macro_auc", "test_macro_top1": "test/macro_top1",
             "test_weighted_ovr_auc": "test/weighted_auc", "test_f1_minority": "test/f1",
             "test_f1_rate_matched": "test/f1_rate", "test_pos_rate": "test/pos_rate"}
# Campaigns a duplicate can be superseded by, best first (RESULTS.md 2.1 "SUPERSEDES the earlier
# 3-seed table"; 8.1 is W6 only and W5 must never be mixed in).
PREFER = {"B": ("N3000P", "W6"), "C-ns": ("N3000P", "W6"), "C-drop": ("W6", "W5")}
SHORT = {"assist": "assist2017", "ednet": "ednet", "junyi": "junyi",
         "fromassist": "assist2017", "fromednet": "ednet", "fromjunyi": "junyi"}
OBJ_ENCODER = {"full": "edubert_ednet_pretrain_full_encoder.pt",
               "skill_only": "edubert_ednet_pretrain_ednet_skill_only_encoder.pt",
               "correct_only": "edubert_ednet_pretrain_ednet_correct_only_encoder.pt"}
LOG_TOL = 5.001e-5  # a 4 dp log value may differ from full precision by half a unit

DEVICE_KV = re.compile(r"(\w+)=(\S+)")
LOADED = re.compile(r"^loaded (\d+)/(\d+) (?:encoder )?tensors from (\S+)")
EPOCH = re.compile(r"^epoch\s+\d+\s")
WB_FINISH = re.compile(r"View run (\S+) at:? \S*wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([A-Za-z0-9]+)")
WB_SYNC = re.compile(r"Syncing run (\S+)")
WB_URL = re.compile(r"wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([A-Za-z0-9]+)")
JOB_ID = re.compile(r"_(\d+)\.log$")
SEED_S = re.compile(r"_s(\d+)$")

_A_ORIG = "(full|skill_only|correct_only|scratch)"
TRACK_A = [
    (re.compile(rf"^edubert_(assist2017)_(?:tg_)?objabl_{_A_ORIG}_n1000_seed(\d+)$"), "n1000"),
    (re.compile(rf"^edubert_(junyi)_(?:tg_)?objabl2_{_A_ORIG}_n1000_seed(\d+)$"), "n1000"),
    (re.compile(rf"^edubert_(ednet)_(?:tg_)?regime_ednet_{_A_ORIG}_n1000_seed(\d+)$"), "n1000"),
    (re.compile(r"^edubert_(algebra2005)_algabl_(full|skill_only|correct_only)_seed(\d+)$"),
     "full_split"),
    (re.compile(r"^edubert_(bridge2006)_bridgeabl_(full|skill_only|correct_only)_seed(\d+)$"),
     "full_split"),
    (re.compile(r"^edubert_(assist2009)_a09abl_(full|skill_only|correct_only)_seed(\d+)$"),
     "full_split"),
    (re.compile(r"^edubert_(algebra2006)_alg06abl_(full|skill_only|correct_only)_seed(\d+)$"),
     "full_split"),
    (re.compile(r"^edubert_(algebra2005|bridge2006|assist2009|algebra2006)_(scratch)_\1"
                r"_seed(\d+)$"), "full_split"),
]
_COND = "(scratch|indomain|fromednet|fromjunyi|fromassist)"
RX_B = re.compile(rf"^edubert_(assist2017|ednet|junyi)_kt_(assist|ednet|junyi)_{_COND}"
                  r"_n3000_seed(\d+)$")
RX_NS = re.compile(rf"^edubert_(assist2017|ednet|junyi)_ns_(assist|ednet|junyi)_{_COND}"
                   r"_n3000_seed(\d+)$")
RX_S11 = re.compile(r"^edubert_(assist2017|junyi)_kt_(assist|junyi)_fromednet_n(\d+)(?:d(\d+))?"
                    r"_src_n3000_seed(\d+)$")
RX_DROP = [
    re.compile(r"^edubert_(assist2017)_(scratch|indomain|ednet|junyi)_dropout_k(\d+)_seed(\d+)$"),
    re.compile(r"^edubert_(ednet)_drop_ednet_(scratch|indomain|fromassist|fromjunyi)_k(\d+)"
               r"_n\d+_seed(\d+)$"),
    re.compile(r"^edubert_(junyi)_drop_junyi_(scratch|indomain|fromassist|fromednet)_k(\d+)"
               r"_n\d+_seed(\d+)$"),
]
RX_P7 = re.compile(r"^edubert_([a-z0-9]+)_(?:tg_)?probe7_\1_(full|scratch|skill_only|correct_only)"
                   r"_s(\d+)$")
RX_P2A = re.compile(r"^edubert_(assist2017)_probe2_(ednet|junyi|indomain|scratch)_seed(\d+)$")
RX_P2B = re.compile(r"^edubert_(ednet|junyi)_probe2_\1_(fromassist|fromjunyi|fromednet|indomain"
                    r"|scratch)_seed(\d+)$")


# --------------------------------------------------------------------------- log parsing
def parse_banner(line: str, ctx: dict | None) -> dict | None:
    if m := inv.BASELINE_BANNER.match(line):
        run = ctx["kv"].get("run", "") if ctx else ""
        return {"kind": "baseline", "model": m.group(1).upper(), "dataset": m.group(2),
                "run": run, "init": ""}
    if m := inv.KT_BANNER.match(line):
        return {"kind": "kt", "model": "EduBERT-KT", "dataset": "", "run": m.group(1).strip(),
                "init": m.group(2)}
    if m := inv.TASK_BANNER.match(line):
        kind = m.group(1) + (f"({m.group(2).strip()})" if m.group(2) else "")
        return {"kind": kind, "model": "EduBERT", "dataset": "", "run": m.group(3).strip(),
                "init": m.group(4)}
    return None


def scan_file(path: Path) -> tuple[list[dict], dict]:
    """Executions (one per banner) in one log, plus file-level facts.

    Device, loaded and epoch lines are stdout and precede their own banner inside one process,
    so the latest device line is the banner's context; a run-name check guards that. W&B lines
    go to stderr and interleave arbitrarily, so ids are mapped by run name, never by position.
    """
    execs: list[dict] = []
    ctx = cur = pending = pretrain = None
    wb: dict[str, set] = defaultdict(set)
    with path.open(errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            s = line.strip()
            if m := WB_FINISH.search(line):
                wb[m.group(1)].add(m.groups()[1:])
            elif m := WB_SYNC.search(line):
                pending = m.group(1)
            elif pending and (m := WB_URL.search(line)):
                wb[pending].add(m.groups())
                pending = None
            if s.startswith("device="):
                ctx = {"kv": dict(DEVICE_KV.findall(s)), "encoder": "", "loaded": "", "epochs": 0}
                cur = None
                continue
            if ctx is not None:
                if m := LOADED.match(s):
                    ctx["encoder"], ctx["loaded"] = m.group(3), f"{m.group(1)}/{m.group(2)}"
                    continue
                if EPOCH.match(s):
                    ctx["epochs"] += 1
                    continue
                if s.startswith("best mlm_loss"):
                    pretrain = {"run": ctx["kv"].get("run", ""), "epochs": ctx["epochs"],
                                "best_mlm_loss": s.split(":", 1)[-1].strip()}
            banner = parse_banner(line, ctx)
            if banner is not None:
                cur = {**banner, "banner_line": lineno, "metrics": {}, "ctx": ctx}
                execs.append(cur)
                continue
            if line.startswith("==="):
                cur = None
                continue
            if cur is not None and (m := inv.METRIC.match(s)):
                cur["metrics"].setdefault(inv.slug(m.group(1)), float(m.group(2)))

    job = JOB_ID.search(path.name)
    for e in execs:
        c = e.pop("ctx") or {}
        kv = c.get("kv", {})
        ok = bool(c) and kv.get("run") == e["run"]
        e["ctx_status"] = "ok" if ok else ("no_device_line" if not c else "run_name_mismatch")
        e["train_students"] = kv.get("train_students", "") if ok else ""
        e["encoder"] = c.get("encoder", "") if ok else ""
        e["loaded"] = c.get("loaded", "") if ok else ""
        e["epochs_run"] = c.get("epochs", "") if ok else ""
        e["device_seed"] = kv.get("seed", "") if ok else ""
        ids = wb.get(e["run"], set())
        e["wandb_entity"], e["wandb_project"], e["wandb_id"] = (next(iter(ids)) if len(ids) == 1
                                                                 else ("", "", ""))
        e["wandb_id_status"] = {0: "not_in_log", 1: "from_log"}.get(len(ids), "ambiguous")
        e["file"] = path.name
        e["job_id"] = job.group(1) if job else ""
    facts = {"file": path.name, "job_id": job.group(1) if job else "", "banners": len(execs),
             "pretrain": pretrain}
    return execs, facts


def seed_of(e: dict) -> int | None:
    s = inv.seed_from(e["run"], None)
    if s is None and (m := SEED_S.search(e["run"])):
        s = int(m.group(1))
    if s is None and str(e.get("device_seed", "")).isdigit():
        s = int(e["device_seed"])
    return s


def campaign_of(fname: str) -> str:
    stem = re.sub(r"(_\d+)?\.log$", "", fname)
    if m := re.match(r"(w\d+)_", stem):
        return m.group(1).upper()
    if stem.startswith("tg1_"):
        return "TG1"
    if stem.startswith(("kt_", "ns_")):
        return "S11" if "_src_" in stem else "N3000P"
    if stem.startswith("srcscale_"):
        return "S11"
    if stem.startswith("base2_"):
        return "BASE2"
    return "OTHER:" + stem.split("_")[0]


# --------------------------------------------------------------------------- families
def _fam(track: str, target: str, candidate: str, seed: int, **kw) -> dict:
    base = {"track": track, "target": target, "candidate": candidate, "seed": seed,
            "budget": "", "K": "", "source": "", "objective": "", "size": "", "draw": ""}
    base.update(kw)
    return base


def _src(target: str, cond: str) -> str:
    if cond == "scratch":
        return ""
    return target if cond == "indomain" else SHORT[cond]


def classify(run: str, kind: str) -> dict | None:
    if kind == "baseline":
        return _fam("baseline", "", "", -1)
    for rx, budget in TRACK_A:
        if m := rx.match(run):
            tgt, cand, seed = m.group(1), m.group(2), int(m.groups()[-1])
            pre = cand != "scratch"
            return _fam("A", tgt, cand, seed, budget=budget, source="ednet" if pre else "",
                        objective=cand if pre else "")
    for rx, track in ((RX_B, "B"), (RX_NS, "C-ns")):
        if (m := rx.match(run)) and SHORT[m.group(2)] == m.group(1):
            tgt, cond, seed = m.group(1), m.group(3), int(m.group(4))
            src = _src(tgt, cond)
            return _fam(track, tgt, f"src:{src}" if src else "scratch", seed, budget="n3000",
                        source=src, objective="full" if src else "")
    if (m := RX_S11.match(run)) and SHORT[m.group(2)] == m.group(1):
        size, draw = int(m.group(3)), int(m.group(4) or 42)
        return _fam("S11", m.group(1), f"ednet_n{size}_d{draw}", int(m.group(5)),
                    budget="n3000", source="ednet", objective="full", size=size, draw=draw)
    for rx in RX_DROP:
        if m := rx.match(run):
            tgt, cond, k, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            src = _src(tgt, cond if cond.startswith(("from", "indomain", "scratch"))
                       else "from" + cond)
            return _fam("C-drop", tgt, f"src:{src}" if src else "scratch", seed, K=k,
                        budget=f"K{k}", source=src, objective="full" if src else "")
    if m := RX_P7.match(run):
        cand = m.group(2)
        return _fam("probe7", m.group(1), cand, int(m.group(3)),
                    source="" if cand == "scratch" else "ednet",
                    objective="" if cand == "scratch" else cand)
    for rx in (RX_P2A, RX_P2B):
        if m := rx.match(run):
            tgt, cond = m.group(1), m.group(2)
            src = _src(tgt, cond if cond.startswith(("from", "indomain", "scratch"))
                       else "from" + cond)
            return _fam("probe2", tgt, f"src:{src}" if src else "scratch", int(m.group(3)),
                        source=src, objective="full" if src else "")
    return None


def expected_encoder(f: dict) -> str | None:
    if f["candidate"] == "scratch" or f["track"] == "baseline":
        return None
    if f["track"] in ("A", "probe7"):
        return OBJ_ENCODER[f["objective"]]
    if f["track"] == "S11":
        if f["size"] == 353597 and f["draw"] == 42:
            return OBJ_ENCODER["full"]
        tag = f"n{f['size']}" + ("" if f["draw"] == 42 else f"d{f['draw']}")
        return f"edubert_ednet_pretrain_ednet_{tag}_encoder.pt"
    return f"edubert_{f['source']}_pretrain_full_encoder.pt"


# --------------------------------------------------------------------------- side inputs
def load_sacct(path: str | None) -> dict:
    jobs: dict = {}
    if not path:
        return jobs
    with open(path, errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("|")
        col = {h: i for i, h in enumerate(header)}
        lacking = [h for h in ("JobID", "State", "NodeList") if h not in col]
        if lacking:
            raise SystemExit(f"{path}: header lacks {lacking}; expected sacct -P output")
        for ln in fh:
            f = ln.rstrip("\n").split("|")
            if len(f) == len(header) and f[col["JobID"]].isdigit():
                jobs[f[col["JobID"]]] = {h: f[i] for h, i in col.items()}
    return jobs


def load_gres(path: str | None) -> dict:
    gpu: dict = {}
    if not path:
        return gpu
    for ln in open(path, errors="replace"):
        parts = ln.split()
        if len(parts) < 2:
            continue
        m = re.search(r"gpu:([A-Za-z][^:,()\s]*)", parts[1])
        kind = m.group(1) if m else ("none" if parts[1] == "(null)" else "untyped")
        if gpu.get(parts[0]) in (None, "none", "untyped"):
            gpu[parts[0]] = kind
    return gpu


def load_wandb(path: str | None) -> dict:
    recs: dict = {}
    if not path:
        return recs
    for ln in open(path, errors="replace"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        rid = r.get("id")
        if rid and not ("error" in r and rid in recs):
            recs[rid] = r
    return recs


def load_encoders(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, newline="") as fh:
        return {r["name"]: r for r in csv.DictReader(fh, delimiter="\t")}


# --------------------------------------------------------------------------- annotation
def exclude(e: dict, reason: str) -> None:
    if not e["exclusion"]:
        e["exclusion"] = reason


def annotate(execs: list[dict], sacct: dict, gres: dict, wandb: dict) -> None:
    for e in execs:
        e["seed"] = seed_of(e)
        e["campaign"] = campaign_of(e["file"])
        e["fam"] = classify(e["run"], e["kind"]) if (e["run"] or e["kind"] == "baseline") else None
        e["exclusion"] = ""
        job = sacct.get(e["job_id"], {})
        e["job_state"] = (job.get("State") or "").split(" ")[0]
        e["node"] = job.get("NodeList", "")
        e["elapsed"] = job.get("Elapsed", "")
        e["gpu_type"] = gres.get(e["node"], "unknown" if e["node"] else "")
        f = e["fam"]
        e["leakage_status"] = "clean"
        if e["kind"] in inv.QUARANTINE:
            e["leakage_status"] = "quarantined"
            exclude(e, inv.QUARANTINE[e["kind"]])
        elif f and f["track"] == "C-drop" and f["K"] > (50 if f["target"] == "assist2017" else 10):
            e["leakage_status"] = "leaked_K"
            exclude(e, "dropout label leaks at this K (clean: ASSISTments K<=50, EdNet and "
                       "Junyi K<=10)")
        if f is None:
            exclude(e, "no benchmark family matches this run name")
        elif f["track"] != "baseline" and e["seed"] is not None and f["seed"] != e["seed"]:
            exclude(e, f"seed parse disagrees: name {f['seed']}, parsed {e['seed']}")
        if f and f["track"] == "S11" and f["size"] == 1366:
            exclude(e, "1,366 rung is a characterized failure mode (RESULTS.md 11), never a "
                       "point estimate")
        exp = expected_encoder(f) if f else None
        got = Path(e["encoder"]).name if e["encoder"] else ""
        if f is None or f["track"] == "baseline":
            e["encoder_status"] = "n/a"
        elif exp is None:
            e["encoder_status"] = "ok" if not got else f"unexpected load of {got}"
        elif not got:
            e["encoder_status"] = "not_logged"
        else:
            e["encoder_status"] = "ok" if got == exp else f"mismatch: expected {exp}, loaded {got}"
        if e["encoder_status"].startswith(("mismatch", "unexpected")):
            exclude(e, e["encoder_status"])
        if e["metrics"].get("test_auc") == 1.0:
            exclude(e, "AUC of 1.0 is always a bug (hand-off 6.8)")

        rec = wandb.get(e["wandb_id"]) if e["wandb_id"] else None
        e["wandb_rec"] = rec if rec and "error" not in rec else None
        if e["wandb_rec"] and e["wandb_rec"].get("name") not in (None, e["run"]):
            exclude(e, f"W&B id {e['wandb_id']} names run {e['wandb_rec'].get('name')}")
        e["values"], e["value_src"], e["prov"] = {}, {}, {}
        metric = TRACK_METRIC.get(f["track"]) if f else None
        summary = (e["wandb_rec"] or {}).get("summary") or {}
        for name, v in e["metrics"].items():
            w = summary.get(WANDB_KEY.get(name, "\0"))
            if e["leakage_status"] == "quarantined":
                e["values"][name], e["value_src"][name], e["prov"][name] = v, "log", "quarantined"
            elif not isinstance(w, (int, float)) or isinstance(w, bool):
                e["values"][name], e["value_src"][name], e["prov"][name] = v, "log", "log_verified"
            elif abs(float(w) - v) <= LOG_TOL:
                e["values"][name], e["value_src"][name] = float(w), "wandb"
                e["prov"][name] = "wandb_verified"
            else:
                e["values"][name], e["value_src"][name] = v, "log"
                e["prov"][name] = "wandb_log_conflict"
                if name == metric:
                    exclude(e, f"W&B {w} and log {v} disagree beyond 4 dp rounding")


def resolve_duplicates(execs: list[dict], tol: float) -> list[dict]:
    groups: dict = defaultdict(list)
    for e in execs:
        e["duplicate_status"], e["duplicate_copies"] = "unique", 1
        if e["run"] and e["seed"] is not None:
            groups[(e["run"], e["seed"])].append(e)
    report = []
    for (run, seed), grp in sorted(groups.items()):
        if len(grp) < 2:
            continue
        f = next((e["fam"] for e in grp if e["fam"]), None)
        track = f["track"] if f else ""
        metric = TRACK_METRIC.get(track) or ("test_auc" if any("test_auc" in e["metrics"]
                                                               for e in grp) else "")
        vals = [e["metrics"].get(metric) for e in grp]
        same = len({v for v in vals if v is not None}) <= 1 and None not in vals
        for e in grp:
            e["duplicate_status"] = "identical" if same else "disagree"
            e["duplicate_copies"] = len(grp)
        live = [e for e in grp if not e["exclusion"]]
        rule = "already excluded upstream" if not live else ""
        pref = PREFER.get(track)
        if live and pref and len({e["campaign"] for e in live}) > 1:
            present = [c for c in pref if any(e["campaign"] == c for e in live)]
            if present:
                for e in live:
                    if e["campaign"] in pref and e["campaign"] != present[0]:
                        exclude(e, f"superseded campaign {e['campaign']}: {present[0]} wins "
                                   "(RESULTS.md 2.1 / 8.1)")
                rule = f"campaign preference {' > '.join(pref)}"
                live = [e for e in live if not e["exclusion"]]
        completed = [e for e in live if e["job_state"] == "COMPLETED"]
        if completed and len(completed) < len(live):
            for e in live:
                if e["job_state"] and e["job_state"] != "COMPLETED":
                    exclude(e, f"job ended {e['job_state']} while a COMPLETED copy exists")
            rule = (rule + "; " if rule else "") + "COMPLETED over other states"
            live = [e for e in live if not e["exclusion"]]
        if len(live) > 1:
            lv = [e["metrics"].get(metric) for e in live]
            if None in lv:
                for e in live:
                    exclude(e, "unresolved duplicate: a copy lacks the track metric")
                rule = (rule + "; " if rule else "") + "unresolved (metric missing)"
            else:
                spread = max(lv) - min(lv)
                if spread <= tol:
                    keep = min(live, key=lambda x: int(x["job_id"]) if x["job_id"] else 10**15)
                    for e in live:
                        if e is not keep:
                            exclude(e, f"duplicate within {tol} of job {keep['job_id']}")
                    rule = (rule + "; " if rule else "") + f"within tol, kept job {keep['job_id']}"
                else:
                    for e in live:
                        exclude(e, f"unresolved duplicate: copies differ by {spread:.4f} > {tol};"
                                   " investigate provenance (MRAP F)")
                    rule = (rule + "; " if rule else "") + f"unresolved, spread {spread:.4f}"
        report.append({"run": run, "seed": seed, "track": track, "metric": metric,
                       "copies": len(grp), "status": grp[0]["duplicate_status"], "rule": rule,
                       "kept": ",".join(e["job_id"] for e in grp if not e["exclusion"]) or "none",
                       "values": ";".join(f"{e['job_id']}:{e['campaign']}:{e['job_state'] or '?'}:"
                                          f"{e['gpu_type'] or '?'}:{e['metrics'].get(metric)}"
                                          for e in grp)})
    return report


def primary(execs: list[dict]) -> list[dict]:
    return [e for e in execs if not e["exclusion"] and e["fam"]]


def pair_scratch(execs: list[dict]) -> list[str]:
    """Attach the matched scratch value and gain for each pretrained primary execution."""
    notes, index = [], {}
    for e in primary(execs):
        f = e["fam"]
        if f["candidate"] == "scratch" and f["track"] in ("A", "B", "C-ns", "C-drop"):
            key = (f["track"], f["target"], f["budget"], e["seed"])
            if key in index:
                notes.append(f"two primary scratch executions for {key}; kept the lower job id")
                if int(e["job_id"] or 0) > int(index[key]["job_id"] or 0):
                    continue
            index[key] = e
    for e in execs:
        e["matched_scratch"], e["gain"] = "", ""
        f = e["fam"]
        if e["exclusion"] or not f or f["candidate"] == "scratch":
            continue
        if f["track"] not in ("A", "B", "C-ns", "C-drop", "S11"):
            continue
        key = ("B" if f["track"] == "S11" else f["track"], f["target"], f["budget"], e["seed"])
        s = index.get(key)
        metric = TRACK_METRIC[f["track"]]
        if s and metric in s["values"] and metric in e["values"]:
            e["matched_scratch"] = s["values"][metric]
            e["gain"] = e["values"][metric] - s["values"][metric]
    return notes


# --------------------------------------------------------------------------- statistics
def boot_mean_ci(diffs: list[float], idx: list[list[int]]) -> tuple[float, float]:
    n = len(diffs)
    means = sorted(sum(diffs[i] for i in ix) / n for ix in idx)
    return percentile(means, 0.025), percentile(means, 0.975)


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    conc = disc = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else float("nan")


def cell_stats(values: dict[str, dict[int, float]], boots: int, rng: random.Random) -> list[dict]:
    """Per-candidate statistics for one target: means, paired gains against scratch, P(best),
    top-equivalent set and cross-seed ranking stability, all over the seeds every candidate has."""
    cands = sorted(values)
    rows = []
    for c in cands:
        vs = [values[c][s] for s in sorted(values[c])]
        rows.append({"candidate": c, "n": len(vs), "mean": st.mean(vs),
                     "pstdev": st.pstdev(vs) if len(vs) > 1 else 0.0,
                     "seeds": ",".join(str(s) for s in sorted(values[c]))})
    common = sorted(set.intersection(*(set(values[c]) for c in cands))) if cands else []
    n = len(common)
    for r in rows:
        r.update(common_seeds=n, p_best="", top_equivalent="", gain_vs_scratch="", gain_ci="",
                 gain_k_of_n="", transfer_sign="", tau_b_mean="", top1_seed_agreement="")
    if n < 2 or len(cands) < 2:
        return rows
    idx = [[rng.randrange(n) for _ in range(n)] for _ in range(boots)]
    mat = {c: [values[c][s] for s in common] for c in cands}
    mean_c = {c: st.mean(mat[c]) for c in cands}
    best = max(cands, key=lambda c: mean_c[c])
    wins = Counter(max(cands, key=lambda c: sum(mat[c][i] for i in ix)) for ix in idx)
    per_seed_tau, top1_hits = [], 0
    for i in range(n):
        xs = [mat[c][i] for c in cands]
        per_seed_tau.append(kendall_tau_b(xs, [mean_c[c] for c in cands]))
        top1_hits += cands[xs.index(max(xs))] == best
    taus = [t for t in per_seed_tau if not math.isnan(t)]
    for r in rows:
        c = r["candidate"]
        r["p_best"] = wins[c] / boots
        if c == best:
            r["top_equivalent"] = "best"
        else:
            lo, _ = boot_mean_ci([mat[best][i] - mat[c][i] for i in range(n)], idx)
            r["top_equivalent"] = "yes" if lo <= 0 else "no"
        r["tau_b_mean"] = st.mean(taus) if taus else ""
        r["top1_seed_agreement"] = f"{top1_hits}/{n}"
        if "scratch" in mat and c != "scratch":
            d = [mat[c][i] - mat["scratch"][i] for i in range(n)]
            lo, hi = boot_mean_ci(d, idx)
            toward = sum(x > 0 for x in d) if st.mean(d) >= 0 else sum(x < 0 for x in d)
            r.update(gain_vs_scratch=st.mean(d), gain_ci=f"[{lo:+.6f}, {hi:+.6f}]",
                     gain_k_of_n=f"{toward}/{n}",
                     transfer_sign="positive" if lo > 0 else ("negative" if hi < 0
                                                              else "spans zero"))
    return rows


def q4(x) -> str:
    if x in ("", None):
        return ""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# --------------------------------------------------------------------------- RESULTS.md check
def results_md_cells(path: str | None) -> dict:
    """Means RESULTS.md reports for sections 4 (Track A), 2.1 (B), 2.2 (C-ns) and 6 (probe7)."""
    if not path or not Path(path).exists():
        return {}
    lines = Path(path).read_text().splitlines()

    def block(start: str) -> list[str]:
        i = next((k for k, ln in enumerate(lines) if re.match(start, ln)), None)
        if i is None:
            return []
        out = []
        for ln in lines[i + 1:]:
            if ln.startswith(("#", "**", "_")) and out:
                break
            if ln.startswith("|"):
                out.append(ln)
        return out

    def num(cell: str):
        m = re.search(r"[-+]?\d+\.\d+", cell)
        return float(m.group()) if m and "(=" not in cell else None

    cells: dict = {}
    for ln in block(r"^## 4\. Objective reversal"):
        c = [x.strip() for x in ln.strip("|").split("|")]
        if len(c) >= 4 and re.match(r"^[a-z]", c[0]) and c[0] != "Target":
            for cand, v in zip(("full", "skill_only", "correct_only"), c[1:4]):
                cells[("A", c[0], cand)] = num(v)
    for track, start in (("B", r"^### 2\.1 "), ("C-ns", r"^### 2\.2 ")):
        for ln in block(start):
            c = [x.strip() for x in ln.strip("|").split("|")]
            tgt = c[0].split()[0] if c else ""
            if tgt in ("assist2017", "ednet", "junyi") and len(c) >= 6:
                conds = ("scratch", "indomain", "fromednet", "fromjunyi", "fromassist")
                for cond, v in zip(conds, c[1:6]):
                    src = _src(tgt, cond)
                    cells[(track, tgt, f"src:{src}" if src else "scratch")] = num(v)
    for ln in block(r"^## 6\. Probe mechanism"):
        c = [x.strip() for x in ln.strip("|").split("|")]
        if len(c) >= 3 and re.match(r"^[a-z]", c[0]) and c[0] not in ("Dataset",):
            cells[("probe7", c[0], "full")] = num(c[1])
            cells[("probe7", c[0], "scratch")] = num(c[2])
    return {k: v for k, v in cells.items() if v is not None}


# --------------------------------------------------------------------------- outputs
EXEC_COLS = [
    "run_id", "wandb_run_id", "wandb_id_status", "slurm_job_id", "log_file", "campaign",
    "job_state", "node", "gpu_type", "elapsed", "git_commit", "track", "task", "metric",
    "target_dataset", "candidate", "source_dataset", "pretrain_objective", "source_n_students",
    "encoder_draw_id", "target_n_students", "budget", "dropout_K", "max_seq_len",
    "finetune_seed", "target_subsample_seed", "epochs_run", "config_epochs",
    "config_batch_size", "config_lr", "checkpoint", "checkpoint_hash", "encoder_recipe",
    "encoder_status", "value", "value_log", "value_wandb", "value_source",
    "matched_scratch_performance", "transfer_gain", "provenance_status", "leakage_status",
    "duplicate_status", "duplicate_copies", "valid_for_primary_analysis", "exclusion_reason",
]


def task_of(kind: str) -> str:
    return {"kt": "kt", "baseline": "kt_baseline", "next_skill": "next_skill",
            "dropout": "dropout"}.get(kind, "probe" if kind.startswith("probe") else kind)


def exec_rows(execs: list[dict], encoders: dict) -> list[dict]:
    rows = []
    for e in execs:
        f = e["fam"] or {}
        cfg = (e["wandb_rec"] or {}).get("config") or {}
        meta = (e["wandb_rec"] or {}).get("metadata") or {}
        ck = Path(e["encoder"]).name if e["encoder"] else ""
        enc = encoders.get(ck, {})
        draw = f.get("draw") or enc.get("seed", "")
        src_n = f.get("size") or (enc.get("n_students") or ("full split" if enc else ""))
        budget = f.get("budget", "")
        sub = ("none (full split)" if budget == "full_split" or f.get("track") == "C-drop"
               else ("= finetune_seed (coupled)" if budget in ("n1000", "n3000") else ""))
        recipe = (f"ep{enc.get('epochs')}/bs{enc.get('batch_size')}/wu{enc.get('warmup_frac')}"
                  f"/seed{enc.get('seed')}" if enc else "")
        for metric in sorted(e["metrics"]):
            primary_metric = metric == TRACK_METRIC.get(f.get("track", ""))
            rows.append({
                "run_id": e["run"], "wandb_run_id": e["wandb_id"],
                "wandb_id_status": e["wandb_id_status"], "slurm_job_id": e["job_id"],
                "log_file": e["file"], "campaign": e["campaign"], "job_state": e["job_state"],
                "node": e["node"], "gpu_type": e["gpu_type"], "elapsed": e["elapsed"],
                "git_commit": (meta.get("git") or {}).get("commit", "")
                if isinstance(meta.get("git"), dict) else "",
                "track": f.get("track", ""), "task": task_of(e["kind"]), "metric": metric,
                "target_dataset": f.get("target", "") or e.get("dataset", ""),
                "candidate": f.get("candidate", ""), "source_dataset": f.get("source", ""),
                "pretrain_objective": f.get("objective", ""), "source_n_students": src_n,
                "encoder_draw_id": draw, "target_n_students": e["train_students"]
                or cfg.get("n_students", ""), "budget": budget, "dropout_K": f.get("K", ""),
                "max_seq_len": cfg.get("max_seq_len", ""), "finetune_seed": e["seed"],
                "target_subsample_seed": sub, "epochs_run": e["epochs_run"],
                "config_epochs": cfg.get("epochs", ""),
                "config_batch_size": cfg.get("batch_size", ""), "config_lr": cfg.get("lr", ""),
                "checkpoint": ck, "checkpoint_hash": enc.get("md5", ""),
                "encoder_recipe": recipe, "encoder_status": e["encoder_status"],
                "value": e["values"][metric], "value_log": e["metrics"][metric],
                "value_wandb": e["values"][metric] if e["value_src"][metric] == "wandb" else "",
                "value_source": e["value_src"][metric],
                "matched_scratch_performance": e["matched_scratch"] if primary_metric else "",
                "transfer_gain": e["gain"] if primary_metric else "",
                "provenance_status": e["prov"][metric], "leakage_status": e["leakage_status"],
                "duplicate_status": e["duplicate_status"],
                "duplicate_copies": e["duplicate_copies"],
                "valid_for_primary_analysis": "yes" if not e["exclusion"] and f else "no",
                "exclusion_reason": e["exclusion"],
            })
    return rows


def write_tsv(path: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    cols = cols or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def expected_cells() -> list[tuple]:
    out = []
    for tgt in ("assist2017", "ednet", "junyi", "algebra2005", "bridge2006", "assist2009",
                "algebra2006"):
        for c in ("full", "skill_only", "correct_only", "scratch"):
            out += [("A", tgt, c, s) for s in SEEDS6]
        for c in ("full", "scratch", "skill_only", "correct_only"):
            out += [("probe7", tgt, c, s) for s in SEEDS3]
    for track in ("B", "C-ns"):
        for tgt in ("assist2017", "ednet", "junyi"):
            for c in ("scratch", "src:assist2017", "src:ednet", "src:junyi"):
                out += [(track, tgt, c, s) for s in SEEDS6]
    draws = {1366: (42, 1, 2, 3), 5000: (42, 1, 2), 15000: (42, 1, 2), 49153: (42, 1, 2),
             150000: (42, 1, 2), 353597: (1, 2)}
    for tgt in ("assist2017", "junyi"):
        for size, ds in draws.items():
            for d in ds:
                out += [("S11", tgt, f"ednet_n{size}_d{d}", s) for s in SEEDS6]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logdir", action="append", default=None)
    ap.add_argument("--sacct")
    ap.add_argument("--gres")
    ap.add_argument("--wandb", help="JSONL from analysis/wandb_export_by_id.py (pass 2)")
    ap.add_argument("--encoders", help="TSV from analysis/dump_encoder_configs.py (pass 2)")
    ap.add_argument("--results-md", default="RESULTS.md")
    ap.add_argument("--outdir", default="benchmark")
    ap.add_argument("--dup-tol", type=float, default=0.0005)
    ap.add_argument("--boots", type=int, default=20000)
    a = ap.parse_args()

    files = []
    for d in a.logdir or ["."]:
        files.extend(sorted(Path(d).glob("*.log")))
    if not files:
        raise SystemExit("no *.log files found; pass --logdir")
    execs, facts = [], []
    for p in files:
        ex, fc = scan_file(p)
        execs.extend(ex)
        facts.append(fc)
    sacct, gres = load_sacct(a.sacct), load_gres(a.gres)
    wandb, encoders = load_wandb(a.wandb), load_encoders(a.encoders)
    annotate(execs, sacct, gres, wandb)
    dups = resolve_duplicates(execs, a.dup_tol)
    pair_notes = pair_scratch(execs)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = exec_rows(execs, encoders)
    write_tsv(out / "executions.tsv", rows, EXEC_COLS)
    write_tsv(out / "duplicates.tsv", dups,
              ["run", "seed", "track", "metric", "copies", "status", "rule", "kept", "values"])

    # per-track gold tables over primary executions and the track metric
    rng = random.Random(0)
    prim = primary(execs)
    cellvals: dict = defaultdict(lambda: defaultdict(dict))
    collisions = []
    for e in prim:
        f = e["fam"]
        metric = TRACK_METRIC.get(f["track"])
        if f["track"] in ("A", "B", "C-ns", "C-drop") and metric in e["values"]:
            cell = (f["track"], f["target"], f["budget"])
            slot = cellvals[cell][f["candidate"]]
            if e["seed"] in slot:
                collisions.append(f"{cell} {f['candidate']} seed {e['seed']}: job {e['job_id']}")
                continue
            slot[e["seed"]] = e["values"][metric]
    gold = []
    for (track, tgt, budget), vals in sorted(cellvals.items()):
        for r in cell_stats(vals, a.boots, rng):
            ebv = ("measured (S11 353,597: three draws)" if track == "B"
                   and r["candidate"] == "src:ednet" and tgt in ("assist2017", "junyi")
                   else ("n/a" if r["candidate"] == "scratch" else "unmeasured: one encoder"))
            gold.append({"track": track, "target": tgt, "budget": budget, **r,
                         "encoder_build_variance": ebv})
    write_tsv(out / "gold_cells.tsv", gold)

    # S11: per-draw mean gains against the Track B scratch controls, and draw spreads
    s11: dict = defaultdict(lambda: defaultdict(dict))
    for e in prim:
        f = e["fam"]
        if e["gain"] == "":
            continue
        if f["track"] == "S11":
            s11[(f["target"], f["size"])][f["draw"]][e["seed"]] = e["gain"]
        elif f["track"] == "B" and f["candidate"] == "src:ednet" and f["target"] in ("assist2017",
                                                                                    "junyi"):
            s11[(f["target"], 353597)][42][e["seed"]] = e["gain"]
    s11_rows = []
    for (tgt, size), draws in sorted(s11.items()):
        per = {d: st.mean(v.values()) for d, v in draws.items()}
        s11_rows.append({"target": tgt, "source_students": size,
                         "draws": ",".join(str(d) for d in sorted(per)),
                         "per_draw_gain": ";".join(f"d{d}:{per[d]:+.6f}(n={len(draws[d])})"
                                                   for d in sorted(per)),
                         "mean_gain": st.mean(per.values()),
                         "draw_spread": max(per.values()) - min(per.values()) if len(per) > 1
                         else ""})
    write_tsv(out / "ladder_s11.tsv", s11_rows)

    probes: dict = defaultdict(dict)
    for e in prim:
        f = e["fam"]
        if f["track"] in ("probe7", "probe2") and "test_probe_acc" in e["values"]:
            key = (f["track"], f["target"], f["candidate"])
            probes[key][e["seed"]] = e["values"]["test_probe_acc"]
    write_tsv(out / "probe_cells.tsv",
              [{"track": k[0], "target": k[1], "candidate": k[2], "n": len(v),
                "seeds": ",".join(str(s) for s in sorted(v)), "mean": st.mean(v.values()),
                "pstdev": st.pstdev(v.values()) if len(v) > 1 else 0.0}
               for k, v in sorted(probes.items())])

    # expected design against what exists
    status: dict = {}
    for e in execs:
        f = e["fam"]
        if not f or e["seed"] is None:
            continue
        key = (f["track"], f["target"], f["candidate"], e["seed"])
        if not e["exclusion"]:
            status[key] = "primary"
        elif status.get(key) != "primary":
            status[key] = "excluded: " + e["exclusion"]
    missing = [{"track": t, "target": g, "candidate": c, "seed": s,
                "status": status.get((t, g, c, s), "missing")} for t, g, c, s in expected_cells()]
    write_tsv(out / "missing_cells.tsv", missing)

    ids = sorted({e["wandb_id"] for e in execs if e["wandb_id"] and e["fam"]})
    (out / "wandb_ids_needed.txt").write_text("\n".join(ids) + ("\n" if ids else ""))

    # RESULTS.md cross-check over primary means
    ours: dict = {}
    for (track, tgt, _b), vals in cellvals.items():
        for c, v in vals.items():
            ours[(track, tgt, c)] = (st.mean(v.values()), len(v))
    for (track, tgt, c), v in probes.items():
        ours[(track, tgt, c)] = (st.mean(v.values()), len(v))
    mism = []
    for key, rv in sorted(results_md_cells(a.results_md).items()):
        mine = ours.get(key)
        if mine is None:
            mism.append({"cell": "|".join(key), "results_md": rv, "builder": "", "n": "",
                         "diff": "", "note": "no primary executions"})
        elif abs(mine[0] - rv) > LOG_TOL:
            mism.append({"cell": "|".join(key), "results_md": rv, "builder": mine[0],
                         "n": mine[1], "diff": mine[0] - rv, "note": ""})
    write_tsv(out / "results_md_mismatches.tsv", mism,
              ["cell", "results_md", "builder", "n", "diff", "note"])

    pre = [fc["pretrain"] | {"file": fc["file"], "job_id": fc["job_id"]}
           for fc in facts if fc["pretrain"]]
    write_tsv(out / "pretrain_runs.tsv", pre, ["run", "file", "job_id", "best_mlm_loss", "epochs"])

    write_report(out, a, files, execs, facts, dups, gold, s11_rows, missing, mism, collisions,
                 pair_notes, ids, sacct, wandb, encoders)
    print(f"wrote {out}/ : executions.tsv ({len(rows)} rows), gold_cells.tsv, ladder_s11.tsv, "
          f"probe_cells.tsv, duplicates.tsv, missing_cells.tsv, results_md_mismatches.tsv, "
          f"pretrain_runs.tsv, wandb_ids_needed.txt ({len(ids)} ids), report.md")


def write_report(out, a, files, execs, facts, dups, gold, s11_rows, missing, mism, collisions,
                 pair_notes, ids, sacct, wandb, encoders) -> None:
    L = []
    W = L.append
    prim = primary(execs)
    W("# Transfer benchmark build report\n")
    W(f"Inputs: {len(files)} logs; sacct {'yes' if sacct else 'no'} ({len(sacct)} jobs); "
      f"W&B records {len(wandb)}; encoder configs {len(encoders)}. Duplicate tolerance "
      f"{a.dup_tol}, bootstrap {a.boots} resamples (random.Random(0)).\n")
    no_banner = [fc for fc in facts if fc["banners"] == 0]
    W(f"Executions: {len(execs)} across {len({e['file'] for e in execs})} logs; "
      f"{len(prim)} valid for primary analysis. Logs with no result banner: {len(no_banner)} "
      f"({sum(1 for fc in no_banner if fc['pretrain'])} pretraining logs, "
      f"{sum(1 for fc in no_banner if not fc['pretrain'])} died or never evaluated).\n")
    W("## Campaigns\n")
    for c, n in Counter(e["campaign"] for e in execs).most_common():
        W(f"- {c}: {n}")
    W("\n## Families\n")
    for t, n in Counter((e["fam"] or {}).get("track", "unassigned") for e in execs).most_common():
        W(f"- {t}: {n} executions, {sum(1 for e in prim if e['fam']['track'] == t)} primary")
    stems = Counter(re.sub(r"_(seed|s)\d+$", "", e["run"]) for e in execs if not e["fam"])
    if stems:
        W("\nUnassigned run stems (top 30; extend classify() if any belongs to a track):")
        for s, n in stems.most_common(30):
            W(f"- `{s or '(no run name)'}`: {n}")
    W("\n## Exclusions\n")
    if not any(e["exclusion"] for e in execs):
        W("none")
    for r, n in Counter(re.sub(r"\d+\.\d+|job \d+|\d{5,}", "#", e["exclusion"])
                        for e in execs if e["exclusion"]).most_common():
        W(f"- {n}: {r}")
    W("\n## Duplicates\n")
    W(f"{len(dups)} duplicated (run, seed) groups: "
      f"{sum(1 for d in dups if d['status'] == 'disagree')} disagree, "
      f"{sum(1 for d in dups if d['status'] == 'identical')} identical. By outcome:")
    for r, n in Counter(re.sub(r"\d+\.\d+|job \d+|kept job \d+", "#", d["rule"])
                        for d in dups).most_common():
        W(f"- {n}: {r or 'no rule needed'}")
    W("\n## Provenance and W&B coverage\n")
    ws = Counter(e["wandb_id_status"] for e in execs if e["fam"])
    W(f"W&B ids in logs (assigned families): {dict(ws)}; ids written: {len(ids)}.")
    pv = Counter(e["prov"].get(TRACK_METRIC.get(e['fam']['track'], ''), "no track metric")
                 for e in prim)
    W(f"Primary track-metric provenance: {dict(pv)}.")
    W(f"Encoder check (assigned families): "
      f"{dict(Counter(e['encoder_status'].split(':')[0] for e in execs if e['fam']))}.")
    W("\n## Hardware of primary executions, by track (informs gap-fill pinning)\n")
    for t in ("A", "B", "C-ns", "C-drop", "S11", "probe7"):
        c = Counter(e["gpu_type"] or "unknown" for e in prim if e["fam"]["track"] == t)
        if c:
            W(f"- {t}: {dict(c.most_common())}")
    keys = ("epochs", "batch_size", "lr", "warmup_frac", "dropout", "max_seq_len")
    for t in ("A", "probe7"):
        sig = Counter(tuple(((e["wandb_rec"] or {}).get("config") or {}).get(k, "?") for k in keys)
                      for e in prim if e["fam"]["track"] == t)
        if sig:
            W(f"- {t} fine-tune config {keys} (W&B; ? means pass 1 or not logged): "
              f"{dict(sig.most_common())}")
    W("\n## Gold cells (track metric; 4 dp display, half up)\n")
    W("| track | target | budget | candidate | n | mean | gain vs scratch | sign | P(best) | "
      "top-equiv | tau-b | top-1 agree | encoder build variance |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in gold:
        p_best = "" if g["p_best"] == "" else f"{g['p_best']:.3f}"
        W(f"| {g['track']} | {g['target']} | {g['budget']} | {g['candidate']} | {g['n']} | "
          f"{q4(g['mean'])} | {q4(g['gain_vs_scratch'])} {g['gain_k_of_n']} | "
          f"{g['transfer_sign']} | {p_best} | {g['top_equivalent']} | {q4(g['tau_b_mean'])} | "
          f"{g['top1_seed_agreement']} | {g['encoder_build_variance']} |")
    if s11_rows:
        W("\n## Section 11 ladder (gain against the Track B scratch controls)\n")
        for r in s11_rows:
            W(f"- {r['target']} n={r['source_students']}: mean {q4(r['mean_gain'])}, spread "
              f"{q4(r['draw_spread'])}, {r['per_draw_gain']}")
    W("\n## Missing and excluded cells of the expected design\n")
    by = defaultdict(Counter)
    for m in missing:
        by[m["track"]][m["status"].split(":")[0]] += 1
    for t, c in by.items():
        W(f"- {t}: {dict(c)}")
    W("\n## RESULTS.md cross-check\n")
    differ = [m for m in mism if m["builder"] != ""]
    absent = [m for m in mism if m["builder"] == ""]
    W(f"{len(differ)} RESULTS.md cells differ from the builder beyond 4 dp rounding; "
      f"{len(absent)} have no primary executions in this build (results_md_mismatches.tsv). "
      "A difference usually means another duplicate copy, campaign or precision, which the "
      "rules above now decide explicitly.")
    for m in differ[:40]:
        W(f"- {m['cell']}: RESULTS.md {m['results_md']}, builder {q4(m['builder'])} "
          f"(n={m['n']}, diff {m['diff']:+.6f})")
    for note in collisions + pair_notes:
        W(f"- {note}")
    (out / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
