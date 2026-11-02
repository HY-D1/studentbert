from __future__ import annotations

"""RESULTS.md 12.20: the warning signal as a detector, and at matched coverage.

Why this exists. Prof. Hazra (2026-10-27) asked for a more formal evaluation of the warning signal
(precision and recall, AUROC and AUPRC across epochs). analysis/warning_metrics.py (commit 632d8df,
job 11421958) computed it after every warning result had been seen, so the section says so. This
records its report verbatim, the files it rests on, and a reading that states the weak NLP
discrimination and the matched-coverage result side by side. Run from the code root as a short
job. Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "60e07b50868c8c917d08391355aa9416"
BF = Path("benchmark_final")
PINNED = {
    Path("analysis/warning_metrics.py"): "fb52474862374e4eadc13f2890ac21cb",
    Path("tests/test_warning_metrics.py"): "f715ef05c0d5651fed07363034cb3560",
    BF / "warning_metrics_report.md": "bd6d07d9a0ea2c604bed8a0030167f8c",
    BF / "warning_metrics.tsv": "e0f40b73114194830b854e155a47c4b7",
    BF / "warning_metrics_coverage.tsv": "4865675cb63b56d50ed18ccb719bb505",
    BF / "reversal_pairs.tsv": "8f57558d142869b3e34a35ad18d095f7",
}
LOG = Path("wm3_run_11421958.log")
HEAD = "### 12.20 The warning signal as a detector, and at matched coverage"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = [
    "| NLP, all three seed sets | epoch 3 | 51/66 (0.77) | 51/58 (0.88) | 0.68 [0.26, 0.93] | "
    "0.69 [0.45, 0.94] |",
    "| NLP, all three seed sets | frozen margin | -- | -- | 0.65 [0.20, 0.88] | "
    "0.78 [0.34, 0.97] |",
    "| StudentBERT (retrospective) | epoch 2 | 6/9 (0.67) | 6/7 (0.86) | 0.84 [0.46, 1.00] |",
    "| StudentBERT (retrospective) | frozen margin | -- | -- | 0.56 [0.05, 0.91] |",
    "| NLP, seeds 42, 1, 2 (prospective) | epoch 3 | 8/30 | 2.70 | 0.54 (2) | 3.32 (4) |",
    "| NLP, seeds 3, 4, 5 | epoch 3 | 6/30 | 2.37 | 0.72 (2) | 3.45 (3) |",
    "| NLP, seeds 6, 7, 8 | epoch 3 | 9/30 | 2.70 | 0.31 (2) | 3.39 (5) |",
    "| NLP, seeds 42, 1, 2 (prospective) | epoch 2 | 5/30 | 2.70 | 2.76 (2) | 5.31 (4) |",
    "| StudentBERT (retrospective) | epoch 3 | 12/21 | 0.0012 | 0.0009 (1) | 0.0014 (3) |",
    "NLP, seeds 42, 1, 2 (prospective), epoch 3: agnews cls seed 1 (overtaken; flagged by the "
    "original rule only)",
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
    report = text[BF / "warning_metrics_report.md"]
    flat = " ".join(report.split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in flat]
    log = LOG.read_text() if LOG.is_file() else ""
    if ("wrote warning_metrics.tsv and warning_metrics_report.md md5 "
            "bd6d07d9a0ea2c604bed8a0030167f8c") not in log:
        missing.append(f"{LOG} does not show the run that wrote the pinned report")
    if missing:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/warning_metrics.py, commit 632d8df, job 11421958)", ""]
    L += wrap(
        "Task: Prof. Hazra (2026-10-27) asked for a more formal evaluation of the warning signal, "
        "with precision and recall and AUROC and AUPRC across epochs. The analysis was defined and "
        "run after every warning result in 12.15 to 12.19 had been seen; it changes no rule and no "
        "decision. Graded score: the best alternative's validation score after k epochs minus that "
        "of H-score's pick, so the fixed rule flags exactly when it is above zero. Comparison "
        "without fine-tuning: H-score's standardized margin over its runner-up, negated. The "
        "matched-coverage comparison follows the rule's own number of recommendations and takes "
        "the same number with the largest frozen margins.")
    L += [""] + wrap(
        "Ties: the original NLP evaluation (analysis/nlp_option1_eval.py) ranked by validation "
        "accuracy and then by encoder name; the replications (analysis/phase2_eval.py) and "
        "StudentBERT break a tie in H-score's favour. The script checks each set against its own "
        "implementation, and its counts reproduce 12.15 to 12.19. Only one tie changes a reported "
        "count: AG News, first-token pooling, seed 1, epoch 3, where the original rule's 18 of 20 "
        "becomes 17 of 20 under the replications' tie handling.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/warning_metrics.py (headings lowered two levels):", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("##" + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: as a threshold-free detector of overtaken recommendations, the early signal is "
        "moderate on NLP (pooled AUROC 0.64 to 0.68, every interval including 0.5) and no better "
        "than the frozen margin (0.65; AUPRC 0.78 against 0.67 to 0.69). Its precision at the "
        "rule's threshold, 0.67 to 0.86, sits only modestly above the share overtaken, 0.60 to "
        "0.67. On StudentBERT it separates well after two and three epochs (AUROC 0.84 and 0.83 "
        "against 0.56), but only three targets have an overtaken recommendation, so the intervals "
        "are uninformative, and this evidence is retrospective. At matched coverage the two "
        "signals differ sharply: after three epochs the recommendations the rule follows have mean "
        "regret 0.31 to 0.72 points on the three NLP seed sets, against 3.32 to 3.45 for the same "
        "number with the largest frozen margins, more than following every recommendation (2.37 "
        "to 2.70); on StudentBERT 0.0009 AUC against 0.0014. The exception is the prospective set "
        "after two epochs (2.76 points, two overtaken among five followed). The early signal is "
        "therefore not a sharper detector than the frozen margin on NLP, but the recommendations "
        "it lets through are the ones whose errors are cheap._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.20")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.20")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
