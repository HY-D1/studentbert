from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics as st
import sys
import textwrap
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# Why this exists: the failure-review diagnostics of 2026-09-29 (D8 fine-tuning trajectories and
# the validation check, D4 layers and skill table on Algebra 2006, D6 frozen gold, D10 learner
# bootstrap) had no RESULTS.md section, and three sentences they make inexact needed correcting:
# 12.4 said no diagnostic was run on Algebra 2006, 12.6 called that failure "neither exposure nor
# gold noise" (it is not exposure, but across evaluation learners the gold does not separate the
# two encoders), and 12.7 said the best-average rule is right on Algebra 2006 without the split.
# Every number is read from the pinned files at run time. Values a script already rounded once are
# printed as written; values rounded here are rounded once, half away from zero. The prose states
# only what CHECKS verify on the same files; a different or edited file aborts before anything is
# written. Harry, 2026-09-29.

TARGET = Path("RESULTS.md")
BF = Path("benchmark_final")
PINNED = {
    "trajectories_valtest.tsv": "9088a58c14422e0a415d7e276dacf65c",
    "trajectories_epochs.tsv": "b493726aeac64d6f2f53c86a93954af7",
    "layer_diag_algebra2006_report.md": "b40fdea796ce83f5ac146661ea7eafee",
    "layer_diag_algebra2006_cells.tsv": "e11f95f387d6eca99d6becf78e36df1b",
    "frozen_gold_cells.tsv": "46008bbac2b0e3fba6b62b7f52359d73",
    "frozen_gold_report.md": "10a1661f378626db33ebe8d537fd0f47",
    "split_bootstrap_algebra2006_summary.tsv": "65d86026e7623b8887487935c7a5fcdd",
    "split_bootstrap_algebra2006_checks.json": "f07e604fc0bea2225f144f66dbde0e76",
}
ANCHOR = "\n\n## 13. Cross-domain check"
PRIMARY = ("frozen_linear", "0.0001")
MARGIN = Decimal("0.001")

EDITS = [
    ("L6 +0.42, where L5 still picks in-domain on ASSISTments 2017. Neither diagnostic was run on "
     "Algebra\n2006. Per-skill LogME fails at every layer.",
     "L6 +0.42, where L5 still picks in-domain on ASSISTments 2017. On Algebra 2006 the same "
     "diagnostics\npoint the other way (12.9). Per-skill LogME fails at every layer.",
     "12.4 points to the Algebra 2006 diagnostic"),
    ("On Algebra 2006 the preference holds on every seed of every sample, its\nmargin grows on "
     "unseen learners, and Junyi is the only top-equivalent source (12.1), so that\nfailure is "
     "neither exposure nor gold noise. On Bridge 2006 the estimators disagree (exposure for",
     "On Algebra 2006 the preference holds on every seed of every sample and its\nmargin grows on "
     "unseen learners, so it is not exposure; Junyi is the only top-equivalent source\nacross "
     "fine-tuning seeds (12.1) but not across evaluation learners (12.8, 12.11). On\nBridge 2006 "
     "the estimators disagree (exposure for",
     "12.6 separates exposure from gold noise"),
    ("in the pretrained view: it is right on\nAlgebra 2006, has the highest rank correlation in the "
     "foreign-source view, and trails H-score in the",
     "in the pretrained view: it is right on\nAlgebra 2006's test learners (12.11), has the highest "
     "rank correlation in the foreign-source\nview, and trails H-score in the",
     "12.7 names the split"),
]


def q(x, places: str = "0.0001", sign: bool = True) -> str:
    d = Decimal(str(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def raw(name: str) -> bytes:
    p = BF / name
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    b = p.read_bytes()
    got = hashlib.md5(b).hexdigest()
    if got != PINNED[name]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[name]}; nothing written")
    return b


def tsv(name: str) -> list[dict]:
    return list(csv.DictReader(raw(name).decode().splitlines(), delimiter="\t"))


def md_tables(text: str) -> list[list[str]]:
    """Every markdown table in a report, as its raw lines."""
    out, cur = [], []
    for ln in text.splitlines():
        if ln.startswith("|"):
            cur.append(ln)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def build(results: str) -> str:
    fails: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    vt = tsv("trajectories_valtest.tsv")
    ep = tsv("trajectories_epochs.tsv")
    lrep = raw("layer_diag_algebra2006_report.md").decode()
    lc = tsv("layer_diag_algebra2006_cells.tsv")
    fg = tsv("frozen_gold_cells.tsv")
    frep = raw("frozen_gold_report.md").decode()
    sb = tsv("split_bootstrap_algebra2006_summary.tsv")
    ck = json.loads(raw("split_bootstrap_algebra2006_checks.json"))

    # 12.8 checks
    V = {(r["target"], r["metric"]): r for r in vt}
    E = {(r["target"], int(r["epoch"])): r for r in ep}
    need(all(r["source"] == "W&B full precision" for r in vt if r["metric"].startswith("best")),
         "best validation AUC is not from W&B at full precision")
    need(V[("assist2017", "best validation AUC")]["seeds_positive"] == "6/6"
         and V[("assist2017", "test AUC")]["seeds_positive"] == "6/6",
         "ASSISTments 2017 validation and test do not both favour EdNet on 6 of 6 seeds")
    a6v, a6t = V[("algebra2006", "best validation AUC")], V[("algebra2006", "test AUC")]
    need(a6v["seeds_positive"] == "3/6" and abs(Decimal(a6v["winner_minus_own_mean"])) < MARGIN
         and a6t["seeds_positive"] == "6/6", "Algebra 2006 validation is not a tie within the margin")
    need(E[("assist2017", 1)]["leader"] == "src:assist2017"
         and E[("algebra2006", 1)]["leader"] == "src:algebra2006",
         "the own encoder does not lead at epoch 1 on both targets")
    need(all(E[("assist2017", e)]["leader"] == "src:ednet" for e in range(2, 21)),
         "EdNet does not lead ASSISTments 2017 from epoch 2 on")
    need(all(E[("algebra2006", e)]["leader"] == "src:junyi" for e in range(3, 10))
         and all(E[("algebra2006", e)]["leader"] == "src:algebra2006" for e in range(10, 21)),
         "Algebra 2006 leaders are not Junyi at epochs 3 to 9 and the own encoder from 10")

    # 12.9 checks
    C = {(r["estimator"], int(r["layer"]), r["variant"]): [] for r in lc}
    for r in lc:
        C[(r["estimator"], int(r["layer"]), r["variant"])].append(r["pick"])
    T, R = "trained skill table", "skill table reset"
    for e in ("hscore_kt_causal", "logme_kt_causal"):
        need(C[(e, 6, T)] == ["src:algebra2006"] * 3, f"{e} L6 does not pick in-domain on 3 of 3")
    need(C[("hscore_kt_causal", 6, R)].count("src:junyi") == 2
         and C[("logme_kt_causal", 6, R)].count("src:junyi") == 1,
         "the reset L6 picks are not Junyi on 2 of 3 (H-score) and 1 of 3 (LogME)")
    need(all(C[("hscore_kt_causal", L, R)] == ["src:junyi"] * 3 for L in (2, 3, 4, 5))
         and all(C[("logme_kt_causal", L, R)] == ["src:junyi"] * 3 for L in (3, 4, 5)),
         "the reset middle layers do not pick Junyi on every seed")
    need(C[("hscore_kt_causal", 3, T)] == ["src:junyi"] * 3
         and all(C[("hscore_kt_causal", L, T)].count("src:junyi") == 1 for L in (2, 4, 5)),
         "the trained-table H-score picks by layer changed")
    need("Complete seeds: 1, 2, 42; incomplete: none." in lrep, "the D4 report is not complete")
    ltab = md_tables(lrep)
    need(len(ltab) == 2 and all(len(t) == 16 for t in ltab), "the D4 report tables changed shape")
    m = re.search(r"largest absolute difference: H-score (\S+), LogME (\S+)\.", lrep)
    need(m is not None, "the D4 report lacks its final-layer check")
    final = f"within {m.group(1)} for H-score and {m.group(2)} for LogME)" if m else ""

    # 12.10 checks
    prim = [r for r in fg if (r["readout"], r["lam"]) == PRIMARY]
    need(len(prim) == 7, f"frozen gold has {len(prim)} primary targets, expected 7")
    miss = sorted(r["target"] for r in prim if r["oracle_top1"] != "True")
    need(miss == ["algebra2006", "assist2017"], f"the frozen oracle misses {miss}")
    need(all(r["frozen_best"] == f"src:{r['target']}" for r in prim if r["target"] in miss),
         "frozen performance does not prefer the own encoder on the two failure targets")
    hs_f = st.mean(float(r["hscore_rho_frozen"]) for r in prim)
    hs_t = st.mean(float(r["hscore_rho_tuned"]) for r in prim)
    lm_f = st.mean(float(r["logme_rho_frozen"]) for r in prim)
    lm_t = st.mean(float(r["logme_rho_tuned"]) for r in prim)
    orho = st.mean(float(r["oracle_rho"]) for r in prim)
    oreg = st.mean(float(r["oracle_regret"]) for r in prim)
    need(f"| hscore_kt_causal | 21 | {q(hs_t)} |" in results,
         "H-score's rank correlation with fine-tuned gold does not reproduce 12.3")
    need(f"| logme_kt_causal | 21 | {q(lm_t)} |" in results,
         "LogME's rank correlation with fine-tuned gold does not reproduce 12.3")
    need(hs_f > hs_t and lm_f > lm_t, "the estimators do not track frozen gold better")
    need("H-score on the frozen-gold features matches the 7 x 7 H-score within" in frep,
         "the D6 report lacks its alignment line")
    m2 = re.search(r"matches the 7 x 7 H-score within (\S+)\.\n", frep)
    align = m2.group(1) if m2 else ""
    need(m2 is not None, "the D6 report lacks its alignment value")
    need(f"| hscore_kt_causal | 21 | {q(hs_t)} | 11/21 | 14/21 | 0.0012 / 0.0103 |" in results,
         "H-score's 12.3 regret is no longer 0.0012")
    ftab = md_tables(frep)
    need(len(ftab) == 2 and len(ftab[0]) == 10, "the D6 main table changed shape")

    # 12.11 checks
    S = {r["split"]: r for r in sb}
    need(len(ck) == 12 and max(c["abs_diff"] for c in ck) < 5e-5,
         "not every checkpoint reproduces its recorded test AUC")
    for sp in ("val", "test", "test minus val"):
        lo, hi = Decimal(S[sp]["ci_low"]), Decimal(S[sp]["ci_high"])
        need(lo < 0 < hi, f"the {sp} interval excludes zero")
    if fails:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(fails))

    L = ["### 12.8 Fine-tuning trajectories and the validation check (the two failure targets, "
         "N=3000, 6 seeds; analysis/finetune_trajectories.py)", ""]
    L += wrap("Validation AUC per epoch from the fine-tune logs (4 dp, as scripts/finetune_edubert.py "
              "prints it); best validation AUC from the W&B export at full precision, each value "
              "checked against its log; test AUC is the benchmark value. The winner is the best "
              "source by mean test AUC (EdNet on ASSISTments 2017, Junyi on Algebra 2006).")
    L += [""] + wrap("Files (md5): trajectories_valtest.tsv "
                     f"{PINNED['trajectories_valtest.tsv']}, trajectories_epochs.tsv "
                     f"{PINNED['trajectories_epochs.tsv']}.")
    L += ["", "| target | winner minus own encoder | mean | seeds positive | per seed |",
          "|---|---|---|---|---|"]
    L += [f"| {r['target']} | {r['metric']} | {r['winner_minus_own_mean']} | "
          f"{r['seeds_positive']} | {r['per_seed']} |" for r in vt]
    L += ["", "| target | epoch | leader | winner minus own | seeds winner ahead |",
          "|---|---|---|---|---|"]
    for t in ("assist2017", "algebra2006"):
        for e in (1, 2, 3, 5, 10, 15, 20):
            r = E[(t, e)]
            L.append(f"| {t} | {e} | {r['leader'].removeprefix('src:')} | "
                     f"{r['winner_minus_own']} | {r['seeds_winner_ahead']} |")
    L += [""] + wrap("_Read: at epoch 1 of fine-tuning the own encoder leads on both targets, as "
                     "the frozen scores say. EdNet leads ASSISTments 2017 from epoch 2 on, and its "
                     "validation and test learners agree (6 of 6 seeds on both). On Algebra 2006 "
                     "Junyi leads the validation learners at epochs 3 to 9 and the own encoder "
                     "from epoch 10 on; at the best checkpoints the validation learners tie the two "
                     f"({a6v['winner_minus_own_mean']}, {a6v['seeds_positive']} seeds) while the "
                     f"test learners favour Junyi ({a6t['winner_minus_own_mean']}, "
                     f"{a6t['seeds_positive']})._")

    L += ["", "### 12.9 Layer and skill-table diagnostic on Algebra 2006 (3 seeds, scratch and 7 "
          "encoders; scripts/diagnose_logme.py with H-score, analysis/layer_diag_report.py)", ""]
    L += wrap("Each ranking of the 7 sources is judged against the fine-tuned gold as in 12.3. "
              "'Skill table reset' re-scores the in-domain encoder with its skill table at the "
              "fine-tune's random start, the table every foreign encoder is scored with; nothing "
              f"else changes. The final layer reproduces the 7 x 7 scores ({final}.")
    L += [""] + wrap("Files (md5): layer_diag_algebra2006_report.md "
                     f"{PINNED['layer_diag_algebra2006_report.md']}, "
                     "layer_diag_algebra2006_cells.tsv "
                     f"{PINNED['layer_diag_algebra2006_cells.tsv']}.")
    for name, tab in zip(("H-score", "LogME"), ltab):
        L += ["", f"{name}:", ""] + tab
    L += [""] + wrap("_Read: on Algebra 2006, unlike ASSISTments 2017 (12.4), the trained skill "
                     "table carries much of the in-domain preference. With it reset, H-score's "
                     "final layer picks Junyi on 2 of 3 seeds and LogME's on 1 of 3, and at layers "
                     "L2 to L5 H-score picks Junyi on every seed, as LogME does from L3. With the "
                     "trained table only H-score's L3 picks Junyi on every seed, and the final "
                     "layer never does. The frozen view grants a trained skill table to the "
                     "in-domain encoder only; fine-tuning trains one for every candidate._")

    L += ["", "### 12.10 Frozen gold: logistic readouts on the estimators' features, scored on "
          "test learners (all 7 targets, 3 seeds; scripts/frozen_gold_kt.py, "
          "analysis/frozen_gold_report.py)", ""]
    L += wrap("A logistic readout of the estimators' own causal features (the same learner draw, "
              "position cap, vocabulary loading and random start), fit on the train draw and "
              "scored on a fixed draw of test learners, gives each candidate's frozen "
              "performance. The frozen oracle ranks the candidates by it and is judged against "
              "the fine-tuned gold as in 12.3. The main readout is shared across skills with an "
              "L2 penalty of 1e-4; the report also holds a 1e-2 penalty and a next-skill-bias "
              f"readout. The frozen-gold features match the estimators' within {align}.")
    L += [""] + wrap(f"Files (md5): frozen_gold_cells.tsv {PINNED['frozen_gold_cells.tsv']}, "
                     f"frozen_gold_report.md {PINNED['frozen_gold_report.md']}.")
    L += [""] + ftab[0]
    L += [""] + wrap(f"_Read: knowing frozen performance exactly picks the fine-tuned winner on "
                     f"{7 - len(miss)} of 7 targets (rank correlation {q(orho)}, mean regret "
                     f"{q(oreg, sign=False)}, against 0.0012 for H-score) and misses exactly on "
                     "ASSISTments 2017 and Algebra 2006, where frozen performance on test learners "
                     "prefers the own encoder. H-score and LogME track frozen performance "
                     f"({q(hs_f)}, {q(lm_f)}) far better than fine-tuned performance ({q(hs_t)}, "
                     f"{q(lm_t)}, the 12.3 values reproduced). The estimators measure frozen "
                     "performance well; frozen performance is an imperfect guide to fine-tuned "
                     "performance._")

    L += ["", "### 12.11 Learner-level bootstrap of the Algebra 2006 validation and test splits "
          "(in-domain against Junyi, 6 seeds; scripts/split_bootstrap_kt.py)", ""]
    L += wrap("The 12 saved best checkpoints are re-scored on both splits as the fine-tune scores "
              "its test set; every one reproduces its recorded test AUC within "
              f"{max(c['abs_diff'] for c in ck):.1e}. Learners are resampled 2,000 times (seed "
              "0), the same resample for every checkpoint; the statistic is the mean over seeds "
              "of Junyi's AUC minus the in-domain encoder's. The two splits hold disjoint "
              "learners and are resampled independently.")
    L += [""] + wrap("Files (md5): split_bootstrap_algebra2006_summary.tsv "
                     f"{PINNED['split_bootstrap_algebra2006_summary.tsv']}, "
                     "split_bootstrap_algebra2006_checks.json "
                     f"{PINNED['split_bootstrap_algebra2006_checks.json']}.")
    L += ["", "| split | learners | positions | Junyi minus in-domain | 95% interval | P(>0) | "
          "seeds positive |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['split']} | {r['learners']} | {r['positions']} | {r['b_minus_a']} | "
          f"[{r['ci_low']}, {r['ci_high']}] | {r['p_boot_gt0']} | {r['seeds_positive']} |"
          for r in sb]
    L += [""] + wrap("_Read: once learners are resampled, neither split separates Junyi from the "
                     "in-domain encoder: the test interval includes zero, and the gap between the "
                     "splits is within learner sampling. The Algebra 2006 difference of 12.1 holds "
                     "across fine-tuning seeds, whose intervals the gold uses, but not across "
                     "evaluation learners, which the gold does not resample._")
    return "\n" + "\n".join(L)


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
    heads = ("### 12.8 Fine-tuning trajectories", "### 12.9 Layer and skill-table",
             "### 12.10 Frozen gold", "### 12.11 Learner-level bootstrap")
    present = [h in src for h in heads]
    if all(present) and all(old not in src for old, _n, _w in EDITS):
        print("skip (already applied): 12.4, 12.6, 12.7 edits and 12.8 to 12.11")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if any(present) or "### 12.8" in src:
        sys.exit("ABORT: a 12.8 to 12.11 heading exists but not as this patcher writes it; "
                 "nothing written")
    sections = build(src)
    out = src
    for old, new, why in EDITS:
        out = apply(out, old, new, why)
    out = apply(out, ANCHOR, sections + ANCHOR, "sections 12.8 to 12.11")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
