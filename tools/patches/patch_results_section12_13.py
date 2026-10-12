from __future__ import annotations

"""RESULTS.md 12.13: the reversal-aware signal pipeline, first version (task of 2026-10-06).

Why this exists. Steps 1 to 10 of the task ran on 2026-10-09 (tools/reversal_pipeline.sh, jobs
10943035 and 10943165, identical logs). This records the design, the reproduction checks and the
first leave-one-target-out results, rendered from the pinned output files; the report's tables are
copied as they were written, and the prose states only what CHECKS verify. Nothing was tuned after
the results were seen. Run from the code root as a short job. Harry, 2026-10-12.
"""

import csv
import hashlib
import re
import statistics as st
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "e0837732eabff7f87d67201183280a91"
BF = Path("benchmark_final")
PINNED = {
    BF / "reversal_pairs.tsv": "8f57558d142869b3e34a35ad18d095f7",
    BF / "reversal_pairs_targets.tsv": "88746eb17c78250bd47a75d850ef8ec0",
    BF / "reversal_model_report.md": "d6684f62a402e4b7852e673bbde4d4e8",
    BF / "reversal_model_recs.tsv": "d7b874b710acc7a2d611e987917fc426",
    BF / "reversal_model_pairs.tsv": "aecb1dc3263881b08269010636c3be6a",
}
HEAD = "### 12.13 Reversal-aware signal pipeline"
ANCHOR = "\n\n## 13. Cross-domain check: NLP encoder selection"
TASK3_REGRET = {"algebra2005": "0.003445", "algebra2006": "0.001727", "assist2017": "0.002971",
                "bridge2006": "0.000527", "ednet": "0", "junyi": "0", "assist2009": "0"}


def pinned(p: Path) -> str:
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    b = p.read_bytes()
    got = hashlib.md5(b).hexdigest()
    if got != PINNED[p]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[p]}; nothing written")
    return b.decode()


def tsv(p: Path) -> list[dict]:
    return list(csv.DictReader(pinned(p).splitlines(), delimiter="\t"))


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def grab(report: str, pattern: str, what: str, fails: list) -> str:
    m = re.search(pattern, report)
    if not m:
        fails.append(f"report has no {what}")
        return ""
    return m.group(1)


def build() -> str:
    fails: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    logs = [Path(f"reversal_pipeline_{j}.log") for j in (10943035, 10943165)]
    need(all(x.is_file() for x in logs) and logs[0].read_bytes() == logs[1].read_bytes()
         and b"reversal pipeline: done" in logs[0].read_bytes(),
         "the two pipeline logs are missing, unfinished or differ")
    pairs = tsv(BF / "reversal_pairs.tsv")
    targets = tsv(BF / "reversal_pairs_targets.tsv")
    recs = tsv(BF / "reversal_model_recs.tsv")
    preds = tsv(BF / "reversal_model_pairs.tsv")
    report = pinned(BF / "reversal_model_report.md")

    lab = Counter(r["label_main"] for r in pairs)
    amb = sum(r["label_robust"] == "ambiguous" for r in pairs)
    need(len(pairs) == 588 and len(preds) == 588, "not 588 pair rows")
    need((lab["stable"], lab["near_tie"], lab["reversal"]) == (314, 162, 112), f"labels {lab}")
    need(amb == 3, f"{amb} ambiguous rows, expected 3 (the Algebra 2006 pair on 3 seeds)")
    reg = defaultdict(list)
    for r in targets:
        reg[r["target"]].append(float(r["target_regret_pretrained"]))
    for t, want in TASK3_REGRET.items():
        need(abs(st.mean(reg[t]) - float(want)) < 5e-7, f"{t} regret {st.mean(reg[t])} != {want}")
    mean_reg = st.mean(x for v in reg.values() for x in v)
    need(f"{mean_reg:.4f}" == "0.0012", f"mean regret {mean_reg}")

    by_t = defaultdict(list)
    for r in recs:
        by_t[r["target"]].append(r)
    need(len(recs) == 21, f"{len(recs)} recommendations")
    a17 = by_t["assist2017"]
    need(all(r["reversed"] == "1" and r["abstain"] == "0" for r in a17),
         "assist2017 is not reversed and followed on all 3 seeds")
    jy = by_t["junyi"]
    need(all(r["reversed"] == "0" and r["abstain"] == "1" and r["selected"] == "src:ednet"
             for r in jy), "junyi's EdNet pick is not flagged on all 3 seeds")
    caught = [r for r in recs if r["abstain"] == "1" and r["reversed"] == "1"]
    need(len(caught) == 1 and caught[0]["target"] == "algebra2006"
         and caught[0]["est_seed"] == "42", "the one caught reversal is not Algebra 2006 seed 42")

    auc = grab(report, r"Pair-level AUC (0\.\d{4}), Brier", "pair AUC", fails)
    sel = re.search(r"\| follow when risk <= 0\.5 \| (\d+) \| (0\.\d{4}) \| (\d+) \| (\d+) \| "
                    r"(\d+) \|", report)
    alw = grab(report, r"\| always follow H-score \| 21 \| (0\.\d{4}) \|", "always row", fails)
    aurc = re.search(r"AURC \(lower is better\): model (0\.\d{4}), frozen margin (0\.\d{4}), "
                     r"random (0\.\d{4}), oracle (0\.\d{4})\.", report)
    early = grab(report, r"\| early only \| (0\.\d{4}) \|", "early-only row", fails)
    layers = grab(report, r"\| layers only \| (0\.\d{4}) \|", "layers-only row", fails)
    frozen_aurc = grab(report, r"\| frozen only \| 0\.\d{4} \| 0\.\d{4} \| (0\.\d{4}) \|",
                       "frozen-only row", fails)
    top_bin = re.search(r"\| 0\.8 to 1\.0 \| (\d+) \| (0\.\d{4}) \| (0\.\d{4}) \|", report)
    need(sel is not None and sel.groups() == ("16", "0.0015", "1", "4", "6"),
         "the selective row is not 16, 0.0015, 1, 4, 6")
    need(alw == "0.0012", "always-follow regret is not 0.0012")
    need(aurc is not None and float(aurc.group(1)) > float(aurc.group(3)),
         "the model's AURC is not above random")
    need(float(frozen_aurc or 1) < float(aurc.group(3) if aurc else 0),
         "frozen-only AURC is not below random")
    need(auc == "0.8278" and early == "0.8848" and layers == "0.4978", "pair AUCs moved")
    need(top_bin is not None and float(top_bin.group(2)) > float(top_bin.group(3)),
         "the top calibration bin is not overconfident")
    if fails:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(fails))

    L = [f"{HEAD}, first version: pairs, leave-one-target-out reversal risk and a selective rule "
         "(2026-10-09; analysis/reversal_table.py, analysis/reversal_model.py, "
         "tools/reversal_pipeline.sh, jobs 10943035 and 10943165)", ""]
    L += wrap(
        "Task: Prof. Hazra's reversal-aware signal pipeline (Project_idea.pdf, added 2026-10-06), "
        "with her decisions of 2026-10-08: candidate pairs as rows, every pair of a held-out "
        "target held out together, near-ties their own class, the benchmark tie rule for the main "
        "label, Algebra 2006 ambiguous in the robustness label, the 7 KT targets only.")
    L += [""] + wrap(
        "Design. One row per target, estimator seed (42, 1, 2) and pair of the 8 candidates (7 "
        "encoders and scratch), a being the candidate H-score ranks higher: 588 rows. Label from "
        "the Track B N=3000 gold, 6 fine-tuning seeds: near_tie if the paired bootstrap interval "
        "of gold(a) - gold(b) includes zero or |d| <= 0.001 (the benchmark rule), otherwise "
        f"stable or reversal: {lab['stable']} stable, {lab['near_tie']} near-tie, "
        f"{lab['reversal']} reversal; the robustness label marks the Algebra 2006 in-domain/Junyi "
        f"pair ambiguous ({amb} rows). Features, none reading the held-out target's gold or any "
        "test split: frozen (H-score and LogME at the final layer, margins, ranks), early "
        "fine-tuning (validation AUC at epochs 1 to 3 of the fine-tune with the estimator's seed, "
        "slopes, ranks, gaps), history (each candidate as a foreign source on the other targets), "
        "layers (H-score at layer 3, layer agreement, skill-table reset). Model: L2-regularized "
        "logistic regression for P(reversal), fitted on stable and reversal pairs of the training "
        "targets, features standardized on the training fold, penalty chosen inside each fold by "
        "an inner leave-one-target-out. Recommendation: H-score's top pick, its risk the largest "
        "predicted risk over the pairs it heads, abstain above 0.5 (fixed before any result).")
    L += [""] + wrap(
        "Reproduction checks: H-score's pretrained regret is "
        f"{mean_reg:.4f} over 21 rankings with every per-target value equal to 12.3 and 12.12; the "
        "skill-table reset flips Algebra 2006 to Junyi on seeds 1 and 2 (12.9); the ASSISTments "
        "2017 in-domain encoder leads at epoch 1 and trails at epoch 3 (12.8). The two pipeline "
        "runs wrote identical logs.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/reversal_model.py (headings lowered one level):", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("#" * 2 + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += ["", "| target | selected (seeds 42 / 1 / 2) | reversal probability | abstained | "
          "reversed |", "|---|---|---|---|---|"]
    for t in sorted(by_t):
        rs = sorted(by_t[t], key=lambda r: {"42": 0, "1": 1, "2": 2}[r["est_seed"]])
        sel_c = " / ".join(r["selected"].removeprefix("src:") for r in rs)
        risk = " / ".join(f"{float(r['reversal_probability']):.2f}" for r in rs)
        L.append(f"| {t} | {sel_c} | {risk} | {sum(r['abstain'] == '1' for r in rs)}/3 | "
                 f"{sum(r['reversed'] == '1' for r in rs)}/3 |")
    L += [""] + wrap(
        f"_Read: reversals between candidate pairs are predictable out of target (pair AUC {auc}; "
        "calibration close in the lower bins and overconfident in the top one, "
        f"{top_bin.group(2)} predicted against {top_bin.group(3)} observed). The recommendation "
        f"rule does not yet help: abstaining above 0.5 follows {sel.group(1)} of 21 "
        f"recommendations with mean regret {sel.group(2)}, against {alw} for always following "
        f"H-score, catching {sel.group(3)} reversal (Algebra 2006, seed 42) with "
        f"{sel.group(4)} false alarms; it follows all three ASSISTments 2017 recommendations, "
        "which reverse, and abstains on all three Junyi recommendations, whose EdNet pick is "
        f"right. Its AURC ({aurc.group(1)}) is above random ({aurc.group(3)}), while the frozen "
        f"features alone give {frozen_aurc}. Early fine-tuning is the strongest group for pairs "
        f"(AUC {early} alone) and the layer group carries no signal ({layers} alone). With 7 "
        "targets and 21 recommendations every estimate here is coarse. Nothing was tuned after "
        "these results were seen; changing the model to fit them would be research-history "
        "leakage in the sense of the project constraints._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.13")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.13")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
