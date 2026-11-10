from __future__ import annotations

"""RESULTS.md 12.24: the pre-specified trajectory and compute analysis (StudentBERT, Quick-Tune).

Why this exists. Prof. Hazra (Nov 2026) asked for the pre-specified trajectory and compute
analysis on StudentBERT and Quick-Tune, to decide how strong the methods story is.
analysis/trajectory_analysis.py (commit 1790a07, committed with docs/trajectory_amendment1.md
before the run; job 11762159) reproduced RESULTS.md 12.17, 12.18 and 12.20 before reporting. This
records its report verbatim with the files it rests on. Run from the code root as a short job.
Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "b4d0f62bc0fe8103ea92f11ed132b1bc"
BF = Path("benchmark_final")
PINNED = {
    Path("docs/trajectory_design.md"): "9da80165f872ca8e572a0782afe474eb",
    Path("docs/trajectory_amendment1.md"): "9fb6a874c0d025e0678735381a26f5ce",
    Path("analysis/trajectory_analysis.py"): "4e524cb49c5da4fead0aea6ffa0417ac",
    BF / "trajectory_analysis_report.md": "0e04cb3dccee6bf136ac3ee8a376c763",
    BF / "trajectory_analysis.tsv": "b6fb4d2fb7d63ce1b2aa8ee7f57e5a85",
    BF / "trajectories.tsv": "e93329cbc69ae8a3af843c544d43793b",
    BF / "reversal_pairs.tsv": "8f57558d142869b3e34a35ad18d095f7",
}
HEAD = "### 12.24 Trajectory and compute analysis: StudentBERT and Quick-Tune"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = [
    "| 2 | 0.078 | 0.0010 | 12/21 | 0.0003 | 1 |",
    "| 3 | 0.118 | 0.0006 | 12/21 | 0.0009 | 1 |",
    "| 10 | 0.392 | 0.0021 | 5/21 | 0.0007 | 2 |",
    "| 20 | 0.784 | 0.0017 | 10/21 | 0.0009 | 4 |",
    "overtaken (7): the decision settles at epochs [7, 14, 20]; never within 20 epochs: 4.",
    "| 1 | 0.13 | 20.69 | 4/30 |", "| 5 | 0.64 | 5.58 | 17/30 |", "| 20 | 2.57 | 1.57 | 21/30 |",
    "| 1 | 0.39 | 18.06 | 5/30 |", "| 5 | 1.96 | 6.96 | 14/30 |", "| 20 | 7.86 | 1.31 | 24/30 |",
    "| 1 | 1.57 | 9.56 | 7/26 |", "| 5 | 7.85 | 5.53 | 11/26 |", "| 20 | 31.38 | 0.56 | 20/26 |",
    "by 10 on 10 and by 20 on 13 of 30 tasks; median settling epoch 26;",
    "by 10 on 15 and by 20 on 18 of 30 tasks; median settling epoch 12;",
    "by 10 on 14 and by 20 on 19 of 26 tasks; median settling epoch 10;",
    "median 0.49.", "median 0.25.", "median 0.20.",
]


def pinned(p: Path) -> str:
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != PINNED[p]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[p]}; nothing written")
    return p.read_text() if p.suffix in (".md", ".py") else ""


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def build() -> str:
    for p in PINNED:
        pinned(p)
    report = (BF / "trajectory_analysis_report.md").read_text()
    flat = " ".join(report.split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in flat]
    if missing:
        sys.exit("ABORT, the report does not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/trajectory_analysis.py, commit 1790a07, job 11762159)", ""]
    L += wrap(
        "Task: Prof. Hazra (Nov 2026) asked for the pre-specified trajectory and compute analysis "
        "on StudentBERT and Quick-Tune. Definitions: docs/trajectory_design.md and "
        "docs/trajectory_amendment1.md, committed with the script before the run; the script "
        "reproduced the warning-rule and early-leader results of 12.17, 12.18 and 12.20 at epochs "
        "1 to 3 before reporting. The analysis is descriptive: the StudentBERT outcomes were seen "
        "before, and the script was test-run on the StudentBERT scores with stand-in costs while "
        "it was written. Quick-Tune uses rule A (12.23), so its models are not compared under "
        "identical hyperparameters.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/trajectory_analysis.py (headings lowered two levels):",
          ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("##" + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: on StudentBERT (retrospective; costs are run times spread evenly over epochs), "
        "fine-tuning all 8 candidates at one seed costs 0.039 GPU-hours per epoch and 0.78 for all "
        "20. After 2 or 3 epochs, 10 to 15% of that, the warning rule follows 12 of 21 "
        "recommendations with mean regret 0.0003 and 0.0009 AUC, and after 3 to 5 epochs the early "
        "leader halves H-score's regret (0.0006 against 0.0012). A longer budget does not help: "
        "the early leader's regret rises to 0.0021 after 10 epochs and 0.0017 after 20, worse than "
        "H-score alone, which suggests that one seed's validation ranking late in training is a "
        "noisy guide to the six-seed test gold. The strict detection rule accordingly settles on "
        "only 3 of the 7 overtaken recommendations within 20 epochs; on Algebra 2006 and one "
        "ASSISTments 2017 seed the own encoder regains the validation lead late. On Quick-Tune, "
        "early leaders are poor: their mean regret is 21, 18 and 10 points after one epoch on "
        "micro, mini and extended, 5.5 to 7.0 after five, and 0.6 to 1.6 after twenty. The "
        "eventual winner is the stable early leader by epoch 10 on 10 of 30 micro, 15 of 30 mini "
        "and 14 of "
        "26 extended tasks, at a median of 49%, 25% and 20% of the compute of running every "
        "candidate for 50 epochs. The epoch-50 score defines the final ranking, so regret there is "
        "zero by construction. H-score's choice does not yet exist on Quick-Tune, so these numbers "
        "describe how fast the eventual winner emerges, not when an H-score reversal becomes "
        "detectable._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.24")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.24")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
