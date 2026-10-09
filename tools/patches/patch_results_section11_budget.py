from __future__ import annotations

"""RESULTS.md 11.1: update-matched source-scale controls, and a dated correction of section 11.

Why this exists. Section 11's ladder trains every encoder for 10 epochs at batch 128, so the number
of optimizer updates grows with the corpus, and its reading says corpus size "is doing the work".
Prof. Hazra asked for a compute-matched control and for EdNet 49,153 retrained at Junyi's budget
(2026-10-08). The 108 fine-tunes of slurm/generators/gen_budget_control.sh finished on 2026-10-09
and analysis/budget_control.py reported them. This adds 11.1 (per-size intervals by a stated method,
the update-matched results, the matched-scale comparison at both budgets, the increment at full
precision, each source encoder's recipe) and appends a dated correction after section 11's reading.
The original text stays for provenance.

Every number is read from pinned or parsed files at run time, the prose states only what CHECKS
verify, and a different file aborts before anything is written. Values are rounded once, half away
from zero. Run from the code root (stdlib only, but as a short job, not on the login node). Harry,
2026-10-09.
"""

import ast
import csv
import glob
import hashlib
import json
import re
import sys
import textwrap
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "31b26cc5d108ab17c55550a6b25c2e06"
BF = Path("benchmark_final")
PINNED = {
    BF / "budget_control_summary.tsv": "c1b14438e4d6a547af7952ebbc4b9fe4",
    BF / "budget_control_diffs.tsv": "722e1164514e934329a671a1f9ef06fd",
    BF / "budget_control_runs.tsv": "27cb94677b1c1bc9f7adca203fb252c6",
    Path("wandb_budget_by_id.jsonl"): "b2166758dde5ec76272fb7e69155c5b1",
}
HEAD = "### 11.1 Update-matched controls"
MARK = "CORRECTION (2026-10-09, see 11.1)"
ANCHOR_SECTION = "\n\n## 12. Transfer benchmark and estimator evaluation"
ANCHOR_READ = ("benefit shrinks as target data grows._\n\nDIGIT COLLISION: this Spearman -0.83 is "
               "a THIRD quantity")
OLD_CI = "+0.0177 [+0.0162,+0.0194]"
TARGETS = ("assist2017", "junyi")
LADDER = (5000, 15000, 49153, 150000, 353597)
SPECS = ((49153, 20), (15000, 33), (5000, 96))
RECIPES = {
    "edubert_junyi_pretrain_full_encoder.pt": (20, 128, 0.001, 0.05, 20),
    "edubert_ednet_pretrain_full_encoder.pt": (10, 128, 0.001, 0.05, 10),
    "edubert_assist2017_pretrain_full_encoder.pt": (100, 64, 0.001, 0.1, 95),
    "edubert_ednet_pretrain_ednet_n49153_encoder.pt": (10, 128, 0.001, 0.05, 10),
}


def q(x, sign: bool = True, places: str = "0.0001") -> str:
    d = Decimal(str(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def ci(lo, hi) -> str:
    return f"[{q(lo)}, {q(hi)}]"


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def pinned(p: Path) -> bytes:
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    b = p.read_bytes()
    got = hashlib.md5(b).hexdigest()
    if got != PINNED[p]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[p]}; nothing written")
    return b


def tsv(p: Path) -> list[dict]:
    return list(csv.DictReader(pinned(p).decode().splitlines(), delimiter="\t"))


def recipes(fails: list) -> str:
    logs = sorted(glob.glob("recipe_check_*.log"))
    if len(logs) != 1:
        fails.append(f"expected one recipe_check_*.log, found {len(logs)}")
        return ""
    out = {}
    for ln in Path(logs[0]).read_text().splitlines():
        m = re.match(r"^(edubert_\S+\.pt) (\{.*\}) best epoch (\d+)$", ln.strip())
        if m:
            cfg = ast.literal_eval(m.group(2))
            out[m.group(1)] = (cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["warmup_frac"],
                               int(m.group(3)))
    if out != RECIPES:
        fails.append(f"{logs[0]} does not show the expected recipes: {out}")
    return logs[0]


def losses(fails: list) -> dict:
    out: dict = {}
    for p in sorted(glob.glob("budget_pretrain_ednet_n*e*_*.log")):
        m = re.match(r"^budget_pretrain_ednet_n(\d+)e(\d+)(?:d(\d+))?_(\d+)\.log$", Path(p).name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)), int(m.group(3) or 42))
        best = re.findall(r"best mlm_loss\s*:\s*([0-9.]+)", Path(p).read_text(errors="ignore"))
        if len(best) != 1:
            fails.append(f"{p} has {len(best)} 'best mlm_loss' lines")
            continue
        if key in out:
            fails.append(f"two finished pretraining logs for {key}")
        out[key] = best[0]
    want = {(s, e, d) for s, e in SPECS for d in (42, 1, 2)}
    if set(out) != want:
        fails.append(f"pretraining logs: {sorted(set(out) ^ want)} missing or unexpected")
    return out


def build(results: str) -> tuple[str, str]:
    fails: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    S = {(r["target"], r["condition"], int(r["source_learners"]), int(r["epochs"])): r
         for r in tsv(BF / "budget_control_summary.tsv")}
    D = {(r["target"], r["a"], r["b"]): r for r in tsv(BF / "budget_control_diffs.tsv")}
    runs = tsv(BF / "budget_control_runs.tsv")
    wb = [json.loads(x) for x in pinned(Path("wandb_budget_by_id.jsonl")).decode().splitlines()]
    need(len(runs) == 108 and len(wb) == 108, f"{len(runs)} runs and {len(wb)} W&B records")
    need(all(r.get("state") == "finished"
             and "V100-SXM2" in (r.get("metadata") or {}).get("gpu", "") for r in wb),
         "a W&B record is not finished on a V100-SXM2")
    log = recipes(fails)
    loss = losses(fails)
    pe = Path("scripts/pretrain_edubert.py").read_text()
    need("save encoder at best (lowest) mlm loss" in pe and '"train/mlm_loss"' in pe,
         "scripts/pretrain_edubert.py no longer saves at the best training loss")

    m15 = {t: D[(t, "matched 15000 33", "ladder 49153 10")] for t in TARGETS}
    m5 = {t: D[(t, "matched 5000 96", "ladder 49153 10")] for t in TARGETS}
    dbl = {t: D[(t, "matched 49153 20", "ladder 49153 10")] for t in TARGETS}
    js = D[("assist2017", "junyi_source 49153 20", "matched 49153 20")]
    js10 = D[("assist2017", "junyi_source 49153 20", "ladder 49153 10")]
    for t in TARGETS:
        need(float(m15[t]["ci_lo"]) <= 0 <= float(m15[t]["ci_hi"]),
             f"{t}: update-matched 15,000 is not within the 49,153 gain")
        need(float(m5[t]["ci_hi"]) < 0, f"{t}: update-matched 5,000 does not fall short")
        need(float(dbl[t]["ci_lo"]) > 0, f"{t}: doubling updates at 49,153 does not add")
    need(float(js["ci_lo"]) <= 0 <= float(js["ci_hi"]), "Junyi and EdNet 49,153 x 20 do not tie")
    need(float(js10["ci_lo"]) > 0, "Junyi does not beat EdNet 49,153 x 10")
    gain = {k: float(r["gain"]) for k, r in S.items()}
    inc = {t: gain[(t, "ladder", 353597, 10)] - gain[(t, "ladder", 150000, 10)] for t in TARGETS}
    need(q(inc["assist2017"]) == "+0.0018" and q(inc["junyi"]) == "+0.0019",
         f"the increments are {q(inc['assist2017'])} and {q(inc['junyi'])}")

    def draws(r: dict) -> list[float]:
        return [float(x.split(":")[1]) for x in r["per_draw"].split(";")]

    for t in TARGETS:
        need(min(draws(S[(t, "ladder", 353597, 10)])) > max(draws(S[(t, "ladder", 150000, 10)])),
             f"{t}: not every 353,597 draw exceeds every 150,000 draw")
    lad49 = S[("assist2017", "ladder", 49153, 10)]
    need(OLD_CI in results, "section 11 no longer quotes the old interval")
    need(q(lad49["gain"]) == "+0.0177", "the 49,153 gain on assist2017 is not +0.0177")
    spreads = [float(r["draw_spread"]) for r in S.values() if r["draw_spread"]]
    sp5 = float(S[("assist2017", "matched", 5000, 96)]["draw_spread"])
    need(sp5 == max(spreads), "the 5,000 x 96 draw spread is not the largest")
    up = {s: int(S[("assist2017", "ladder", s, 10)]["updates"]) for s in (150000, 353597)}
    ratio = (Decimal(up[353597]) / Decimal(up[150000])).quantize(Decimal("0.1"), ROUND_HALF_UP)
    need(str(ratio) == "2.4", f"the update ratio is {ratio}")
    if fails:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(fails))

    L = [f"{HEAD}: number of updates against number of learners (2026-10-09; "
         "analysis/budget_control.py, slurm/generators/gen_budget_control.sh)", ""]
    L += wrap(
        "Why. Every encoder in the table above trains for 10 epochs at batch 128, so the number of "
        "optimizer updates is ceil(learners / 128) x 10 and grows with the corpus, from 400 at "
        "5,000 learners to 27,630 at 353,597. The checkpoints record the source encoders' recipes "
        f"({log}): Junyi 20 epochs, batch 128, lr 1e-3, warm-up 0.05 (best epoch 20, so 7,700 "
        "updates over 49,153 learners); EdNet full corpus and EdNet 49,153 (draw 42) 10 epochs, "
        "batch 128, warm-up 0.05 (best epoch 10); ASSISTments 2017 100 epochs, batch 64, warm-up "
        "0.1 (best epoch 95). Each saved encoder is the one with the lowest training loss "
        "(scripts/pretrain_edubert.py), so the losses in the table above are training losses, not "
        "held-out losses.")
    L += [""] + wrap(
        "Design. Encoders edubert_ednet_pretrain_ednet_n{SIZE}e{EPOCHS}{,d1,d2}_encoder.pt with "
        "the "
        "section-11 recipe except the epoch count and the same draw seeds (42, 1, 2, so the same "
        "sampled learners): 49,153 x 20 epochs (7,700 updates, Junyi's), 15,000 x 33 (3,894) and "
        "5,000 x 96 (3,840), the last two matched to the 49,153 rung's 3,850. Fine-tunes "
        "kt_<t>_fromednet_n{SIZE}e{EPOCHS}{,d1,d2}_bud_n3000_seed{1,2,3,4,5,42} on assist2017 and "
        "junyi, N=3000, 20 epochs, gpu:v100-sxm2: 108 runs, paired with the scratch controls "
        "above. Test AUC is the W&B full-precision value (wandb_budget_by_id.jsonl), each checked "
        "against its log's 4 dp value; the training losses come from the "
        "budget_pretrain_ednet_n{SIZE}e{EPOCHS}*_<job>.log files.")
    L += [""] + wrap(
        "Method, for this subsection and the per-size intervals of the 10-epoch ladder: a draw's "
        "gain is its 6-seed mean of (run - scratch at that seed); a condition's gain is the mean "
        "over its draws; the 95% interval resamples the 6 seeds of the across-draw mean (20,000 "
        "resamples, random.Random(0), boot_mean_ci of analysis/build_transfer_benchmark.py); the "
        "draw spread (encoder-build variance) is reported beside it; differences pair the "
        "across-draw means by seed.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "| target | source learners | epochs | updates | gain [95% CI] | seeds + | "
          "draw spread | best training loss (d42 / d1 / d2) |",
          "|---|---|---|---|---|---|---|---|"]
    rows = [("ladder", s, 10) for s in LADDER] + [("matched", s, e) for s, e in SPECS]
    for t in TARGETS:
        for kind, size, ep in rows:
            r = S[(t, kind, size, ep)]
            ls = (" / ".join(loss[(size, ep, dd)] for dd in (42, 1, 2)) if kind == "matched"
                  else "section 11 table")
            L.append(f"| {t} | {size:,} | {ep} | {int(r['updates']):,} | {q(r['gain'])} "
                     f"{ci(r['ci_lo'], r['ci_hi'])} | {r['seeds_positive']} | "
                     f"{q(r['draw_spread'], sign=False)} | {ls} |")
    jr = S[("assist2017", "junyi_source", 49153, 20)]
    L.append(f"| assist2017 | 49,153 (Junyi encoder) | 20 | {int(jr['updates']):,} | "
             f"{q(jr['gain'])} {ci(jr['ci_lo'], jr['ci_hi'])} | {jr['seeds_positive']} | "
             "one encoder | not comparable (Junyi vocabulary) |")
    L += ["", "| target | a minus b | difference [95% CI] | seeds + |", "|---|---|---|---|"]
    for (t, a, b), r in D.items():
        L.append(f"| {t} | {a} minus {b} | {q(r['diff'])} {ci(r['ci_lo'], r['ci_hi'])} | "
                 f"{r['seeds_positive']} |")
    a15, j15 = m15["assist2017"], m15["junyi"]
    L += [""] + wrap(
        "_Read: with the number of updates matched to the 49,153 rung, 15,000 learners transfer "
        "as well as 49,153 learners trained for 10 epochs: the difference is "
        f"{q(a15['diff'])} {ci(a15['ci_lo'], a15['ci_hi'])} on assist2017 and {q(j15['diff'])} "
        f"{ci(j15['ci_lo'], j15['ci_hi'])} on junyi. At the same update count 5,000 learners fall "
        f"short, {q(m5['assist2017']['diff'])} and {q(m5['junyi']['diff'])} with both intervals "
        f"below zero. Doubling the updates at 49,153 learners adds {q(dbl['assist2017']['diff'])} "
        f"and {q(dbl['junyi']['diff'])}. At matched learners and updates the Junyi encoder and "
        f"EdNet 49,153 tie on assist2017 ({q(js['diff'])} {ci(js['ci_lo'], js['ci_hi'])}), while "
        f"against the 10-epoch EdNet 49,153 Junyi leads by {q(js10['diff'])}. So the sign change "
        "between 15,000 and 49,153 in the ladder is an effect of the number of updates, and a data "
        "shortfall remains at 5,000 learners at this update count. Going from 150,000 to 353,597 "
        f"learners at 10 epochs adds {q(inc['assist2017'])} on assist2017 and {q(inc['junyi'])} on "
        "junyi at full precision, and every 353,597 draw exceeds every 150,000 draw on both "
        f"targets; that step also multiplies the updates by {ratio} ({up[150000]:,} to "
        f"{up[353597]:,}), so it is not budget-matched either. Caveats: 96 epochs on 5,000 "
        "learners "
        "may overfit (only training loss is recorded); the matched-scale comparison has one "
        f"target; the 5,000 x 96 draw spread on assist2017 ({q(sp5, sign=False)}) is the largest "
        "in this section._")
    corr = "\n".join(wrap(
        f"{MARK}. Two statements in the reading above do not hold once the number of optimizer "
        "updates is matched. \"Corpus SIZE is therefore doing the work\": at the 49,153 rung's "
        "3,850 updates, 15,000 learners transfer as well as 49,153, so the sign flip between "
        "15,000 "
        "and 49,153 is an update effect; a shortfall remains at 5,000. \"(a) ... at matched size "
        "EdNet is if anything the weaker corpus per student\": the Junyi encoder trained for 20 "
        "epochs (7,700 updates) against 10 for EdNet 49,153 (3,850), and at matched learners and "
        "updates the two tie. The interval [+0.0162,+0.0194] has no recorded method; by the "
        f"method of 11.1 the 49,153 gain on assist2017 is {q(lad49['gain'])} "
        f"{ci(lad49['ci_lo'], lad49['ci_hi'])}, draw spread "
        f"{q(lad49['draw_spread'], sign=False)}. Points (b) and (c) keep their numbers, but (b)'s "
        "losses are training losses at a fixed 10 epochs, so they also vary with the number of "
        "updates."))
    return "\n" + "\n".join(L), corr


def apply(text: str, old: str, new: str, why: str) -> str:
    if new in text and (old not in text or old in new):
        print(f"skip (already applied): {why}")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1; nothing written")
    print(f"ok: {why}")
    return text.replace(old, new, 1)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src and MARK in src:
        print("skip (already applied): 11.1 and the section 11 correction")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if HEAD in src or MARK in src:
        sys.exit("ABORT: RESULTS.md is partly patched; nothing written")
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    section, corr = build(src)
    new_read = ANCHOR_READ.replace("grows._\n\nDIGIT COLLISION",
                                   "grows._\n\n" + corr + "\n\nDIGIT COLLISION")
    out = apply(src, ANCHOR_READ, new_read, "section 11: dated correction after the reading")
    out = apply(out, ANCHOR_SECTION, section + ANCHOR_SECTION, "section 11.1")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
