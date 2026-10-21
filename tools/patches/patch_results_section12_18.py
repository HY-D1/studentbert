from __future__ import annotations

"""RESULTS.md 12.18: phase 2, the full framework on fresh seeds.

Why this exists. Prof. Hazra's phase 2 tasks: the full decision rule (H-score, a one-epoch check,
keep or switch), evaluated on fresh seeds against H-score, LogME, one-epoch fine-tuning alone and
H-score + abstention, with the outcomes, ablations and failure analysis she listed. The design
(docs/phase2_complete_rule_design.md and docs/phase2_amendment1.md) and the evaluation
(analysis/phase2_eval.py) were committed before the fresh fine-tunes were read; the run is job
11271624. The supplement (analysis/phase2_supplement.py) was added after the results and is
labelled so. Nothing was tuned after the results. Run from the code root as a short job. Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "99919ec0c4f48d2de8b6b8f456a160e3"
BF = Path("benchmark_final")
PINNED = {
    Path("docs/phase2_complete_rule_design.md"): "254538daf4b91503fe1cfe767f32ce0f",
    Path("docs/phase2_amendment1.md"): "ce8e827bfe9ffee6a074b498915c29cc",
    Path("analysis/phase2_eval.py"): "d0ecd67479378b4d3dcbf3457253f5dc",
    Path("analysis/phase2_supplement.py"): "bf1756e7072fd077aad769a4f452b50c",
    BF / "p2_head_report.md": "62702fa6156404a02f6d5f34cf856d1d",
    BF / "p2_head_pairs.tsv": "9123ad27c17496af7f5f51cdb1987bc1",
    BF / "p2_head_recs.tsv": "a776f71b11a673d815dc6f486e4eb668",
    BF / "p2_eval_report.md": "205f72d95e20cf2208a51428175a2e86",
    BF / "p2_eval_choices.tsv": "21df978b07059d31198046609f88e1d0",
    BF / "p2_supplement_report.md": "fb051bca22405d4d505d06b46f2b695d",
}
LOG = Path("p2_eval_11271624.log")
HEAD = "### 12.18 Phase 2: the full framework on fresh seeds"
ANCHOR = "\n\n## 13. Cross-domain check: NLP encoder selection"
CHECKS = {
    BF / "p2_eval_report.md": [
        "| H-score alone | 12/30 | 12/30 | 2.37 [0.97, 3.56] |",
        "| full framework, epoch 1 | 12/30 | 16/30 | 1.95 [0.52, 3.45] | 4/18 [9%, 45%] | "
        "0/12 [0%, 24%] | 3 | 5/22/3 |",
        "| early leader, epoch 1 | 12/30 | 19/30 | 1.79 [0.55, 3.27] | 7/18 [20%, 61%] |",
        "| early leader, epoch 3 | 12/30 | 19/30 | 1.12 [0.45, 2.07] | 9/18 [29%, 71%] |",
        "| full framework, epoch 3 | 15/30 | 18/30 | 1.42 [0.33, 3.02] |",
        "| LogME alone | 15/30 | 18/30 | 2.08 [0.68, 3.47] |",
        "| best-average rule | 6/30 | 12/30 | 1.37 [0.51, 2.33] |",
        "| full framework, epoch 1 | 5/10 | 0.96 |", "| early leader, epoch 1 | 3/10 | 1.02 |",
        "Switches of the full framework (epoch 1) between encoders the published results cannot "
        "separate: 3."],
    BF / "p2_supplement_report.md": [
        "| full framework, epoch 1 | 6.91 | -0.42 [-1.96, +0.69] |",
        "| early leader, epoch 3 | 6.91 | -1.26 [-2.55, +0.01] |",
        "| best-average rule | 3.84 | -1.01 [-1.98, -0.07] |",
        "| epoch 1 | 14/18 [55%, 91%] | 6/12 [25%, 75%] |",
        "| epoch 2 | 17/18 [74%, 99%] | 5/12 [19%, 68%] |",
        "| H-score alone | 11/21 | 14/21 | 0.0012 [0.0003, 0.0023] |",
        "| early leader, epoch 1 | 8/21 | 12/21 | 0.0031 [0.0007, 0.0070] | 0/7 [0%, 35%] |",
        "| full framework, epoch 3 | 17/21 | 19/21 | 0.0006 [0.0001, 0.0016] | 6/7 [49%, 97%] |"],
}


def pinned(p: Path) -> str:
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != PINNED[p]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[p]}; nothing written")
    return p.read_text()


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def body(text: str) -> list[str]:
    out = [ln for ln in text.splitlines() if not ln.startswith("# ")]
    while out and out[-1] == "":
        out.pop()
    return ["#" * 2 + ln if ln.startswith("## ") else ln for ln in out]


def build() -> str:
    text = {p: pinned(p) for p in PINNED}
    missing = [f"{p.name}: {c}" for p, cs in CHECKS.items() for c in cs if c not in text[p]]
    log = LOG.read_text() if LOG.is_file() else ""
    if "full framework, epoch 1: mean regret 1.95 against 2.37 for H-score" not in log:
        missing.append(f"{LOG} does not show the primary result")
    if missing:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/phase2_eval.py, job 11271624; design "
         "docs/phase2_complete_rule_design.md "
         "and docs/phase2_amendment1.md, both committed before the fresh fine-tunes were read)", ""]
    L += wrap(
        "Task: Prof. Hazra's phase 2 list. Fresh estimator seeds 3, 4 and 5 (new samples on AG "
        "News, "
        "MNLI and QNLI; Airline's samples overlap almost entirely with the old ones and RTE's are "
        "identical, so only their fine-tuning randomness is new): new frozen features and scores, "
        "30 fresh H-score recommendations (18 overtaken after fine-tuning, from 4 of 5 tasks) and "
        "210 early fine-tunes of 3 epochs. The full framework keeps H-score's pick unless the "
        "epoch-k leader's validation accuracy exceeds it by more than 1.96 x sqrt(2 p (1 - p) / "
        "n); "
        "with a strict check it would equal the early leader alone, which is reported as the "
        "ablation without H-score. Primary, fixed before the run: the full framework at epoch 1 "
        "against H-score alone on mean regret. Four jobs ran twice after scheduler errors on "
        "explorer-02; each of the 210 outputs is one complete run, checked by the evaluation.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/phase2_eval.py:", ""]
    L += body(text[BF / "p2_eval_report.md"])
    L += ["", "Supplement, added after the results (analysis/phase2_supplement.py; part B is "
          "retrospective):", ""] + body(text[BF / "p2_supplement_report.md"])
    L += [""] + wrap(
        "_Read: the pre-declared primary comparison is inconclusive. The full framework at epoch 1 "
        "lowers mean regret from 2.37 to 1.95 points (5 wins, 22 ties, 3 losses against H-score), "
        "but the paired difference, -0.42, has a task-cluster interval from -1.96 to +0.69. It "
        "recovers 4 of 18 overtaken recommendations and makes no false correction; the early "
        "leader "
        "alone recovers 7 of 18, also without false corrections, with regret 1.79, so at one epoch "
        "H-score adds stability (5 of 10 settings choose the same encoder on every seed, against "
        "3) "
        "but not lower regret. More early fine-tuning helps correction: at epoch 3 the early "
        "leader "
        "reaches 1.12 (difference -1.26, interval -2.55 to +0.01) and the full framework 1.42. The "
        "best-average rule, which uses the published fine-tuned results of the other tasks, has "
        "1.37 and is the only method whose difference from H-score excludes zero (-1.01, -1.98 to "
        "-0.07). The warning signal replicates less cleanly on fresh seeds: 14 of 18 overtaken "
        "recommendations caught at epoch 1 with 6 of 12 false alarms (originally 19 of 20 and 4 of "
        "10), 17 of 18 with 5 at epoch 2. The full framework's remaining failures at epoch 1 are "
        "AG "
        "News (6), where H-score's pick leads after one epoch in 4 cases and is overtaken later; "
        "QNLI (5) and "
        "RTE (2), "
        "where the leader's advantage is inside the validation noise (RTE has 277 validation "
        "rows); "
        "and one wrong switch on MNLI. On StudentBERT (retrospective) one epoch makes things worse "
        "(regret 0.0031 against 0.0012, no reversal recovered), while three epochs recover 6 of 7 "
        "at regret 0.0006. In both domains, then, one epoch is not enough to correct H-score, and "
        "the evidence does not yet support a complete selection framework at that budget. Five "
        "tasks and seven targets make every interval wide._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.18")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.18")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
