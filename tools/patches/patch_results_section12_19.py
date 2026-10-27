from __future__ import annotations

"""RESULTS.md 12.19: phase 3, the final fresh-seed test of the three-epoch framework.

Why this exists. Prof. Hazra: the diagnosis and the warning signal are the main contribution, and
correction stays secondary unless the three-epoch version holds on further fresh seeds; one final
fresh-seed evaluation with the rule fixed before looking. analysis/phase3_eval.py (job 11395653)
ran it on seeds 6, 7 and 8. This records the result as declared, the exact timeline of the design
commit (which came five minutes after the first fine-tune job ended, so the design file's own
commit note is wrong on that point) and the facts behind the reading. Run from the code root as a
short job. Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "a981b9680b3aa6dd5e1012e382ec2a29"
BF = Path("benchmark_final")
PINNED = {
    Path("docs/phase3_final_test_design.md"): "c10e130c866e721ef96d7d0d66fda637",
    Path("slurm/generators/gen_phase3_jobs.sh"): "c4d16748aca88164f70219dace81b3d1",
    Path("analysis/phase3_eval.py"): "a143007d38fe0a72bbd6c80e833bf37b",
    BF / "p3_head_report.md": "84b62b7cbdc4bb91b17cbd4b008367a7",
    BF / "p3_head_pairs.tsv": "9bad0afce43c426c8105473a6b8f918c",
    BF / "p3_head_recs.tsv": "1aaaca3159db41dc1ad1deffcc6d3245",
    BF / "p3_eval_report.md": "0532591f84e64d2b2634b07faa7b0aee",
    BF / "p3_eval_choices.tsv": "e92570b7749f3fe4deab19f4250a7670",
}
LOG = Path("p3_eval_11395653.log")
HEAD = "### 12.19 Phase 3: the final fresh-seed test of the three-epoch framework"
ANCHOR = "\n\n## 13. Cross-domain check: NLP encoder selection"
CHECKS = [
    "Full framework at epoch 3 minus H-score: -1.09 points, 95% task-cluster interval [-3.02, "
    "+0.00]. The three-epoch version DOES NOT HOLD",
    "| early leader, epoch 3 | -1.95 [-3.67, -0.41] | -2.90 [-4.83, -0.57] |",
    "| early leader, epoch 2 | -1.59 [-3.25, -0.15] | -2.64 [-4.44, -0.68] |",
    "| LogME alone | -0.53 [-1.08, -0.04] |",
    "| best-average rule | -1.33 [-2.68, -0.07] |",
    "full framework at epoch 3 minus H-score -1.02 [-2.68, +0.21].",
    "| full framework, epoch 3 | 16/30 | 17/30 | 1.61 [0.28, 2.94] | 7/20 [18%, 57%] | "
    "0/10 [0%, 28%] | 1 | 7/22/1 |",
    "| early leader, epoch 3 | 18/30 | 22/30 | 0.75 [0.14, 1.57] | 12/20 [39%, 78%] | "
    "0/10 [0%, 28%] | 3 | 17/9/4 |",
    "| H-score alone | 10/30 | 10/30 | 2.70 [1.04, 4.15] |",
    "| epoch 1 | 17/20 [64%, 95%] | 5/10 [24%, 76%] |",
    "| epoch 2 | 19/20 [76%, 99%] | 4/10 [17%, 69%] |",
    "| epoch 3 | 18/20 [70%, 97%] | 3/10 [11%, 60%] |",
    "| full framework, epoch 3 | 7/10 | 0.82 |", "| early leader, epoch 3 | 4/10 | 0.58 |",
]


def pinned(p: Path) -> str:
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != PINNED[p]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[p]}; nothing written")
    return p.read_text()


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def build() -> str:
    text = {p: pinned(p) for p in PINNED}
    design = text[Path("docs/phase3_final_test_design.md")]
    report = " ".join(text[BF / "p3_eval_report.md"].split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in report]
    if hashlib.md5(design.encode()[:3567]).hexdigest() != "d4bd10936c2e4b142c057697ac5adb01":
        missing.append("the design's first 3,567 bytes are not the delivered version")
    log = LOG.read_text() if LOG.is_file() else ""
    if "primary: full framework at epoch 3 minus H-score -1.09 [-3.02, +0.00], does not hold" \
            not in log:
        missing.append(f"{LOG} does not show the primary result")
    if missing:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/phase3_eval.py, job 11395653; design "
         "docs/phase3_final_test_design.md)", ""]
    L += wrap(
        "Task: Prof. Hazra's decision after 12.18 (the diagnosis and the warning signal are the "
        "main contribution; correction stays secondary unless the three-epoch version holds on "
        "further fresh seeds). Rule: the full framework at epoch 3 of 12.18. Fresh seeds 6, 7 and "
        "8 (new samples on AG News, MNLI and QNLI; Airline's overlap almost entirely with earlier "
        "ones and RTE's are identical). Criterion, fixed in advance: the three-epoch version holds "
        "if the 95% task-cluster interval of its regret minus H-score's lies entirely below zero.")
    L += [""] + wrap(
        "Timeline, from git and sacct. The design, the generator and the evaluation were delivered "
        "before any phase 3 job, but their commit step did not run. The 210 fine-tunes were "
        "submitted at 17:29 on 2026-10-26; the first job ended at 17:40:13; the design was "
        "committed at 17:45:25 (687613d), five minutes later, and the evaluation read the "
        "fine-tunes afterwards (job 11395653). The design file's commit note says no fine-tune had "
        "finished by then; that is wrong, as one job (6 runs) had. No output was read before the "
        "commit, and the first 3,567 bytes of the design are the delivered version.")
    L += [""] + wrap(
        "H-score's picks on seeds 6 to 8 equal those on seeds 42, 1 and 2 in all 10 settings, "
        "although the scores differ (new samples), so the 30 fresh labels match the original ones: "
        "20 reversed, from 4 of 5 tasks.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/phase3_eval.py (headings lowered one level):", ""]
    for ln in text[BF / "p3_eval_report.md"].splitlines():
        if ln.startswith("# "):
            continue
        L.append("#" * 2 + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: by the criterion fixed in advance, the three-epoch framework does not hold: its "
        "regret minus H-score's is -1.09 points with an interval from -3.02 to +0.00, which does "
        "not lie below zero, so correction stays secondary. The noise margin is what holds it "
        "back: "
        "the framework never makes a false correction, but of its 13 failures, 11 keep H-score's "
        "pick because the early leader's advantage is inside the margin (AG News, QNLI, RTE) and "
        "in 2 (AG News) H-score's pick is itself the epoch-3 leader. The early "
        "leader alone at epoch 3, a secondary analysis fixed in the design, reduces regret by 1.95 "
        "points (-3.67 to -0.41) with no false correction, but its choice changes across seeds in "
        "6 of 10 settings; in phase 2 the same comparison gave -1.26 (-2.55 to +0.01). The best-"
        "average rule (-1.33) and LogME (-0.53) also improve on H-score here. The warning signal "
        "replicates: 17 of 20 overtaken recommendations flagged at epoch 1, 19 at epoch 2 and 18 "
        "at "
        "epoch 3, with 5, 4 and 3 false alarms out of 10. Pooled with phase 2, the framework at "
        "epoch 3 gives -1.02 (-2.68 to +0.21). Five tasks make every interval wide._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.19")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.19")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
