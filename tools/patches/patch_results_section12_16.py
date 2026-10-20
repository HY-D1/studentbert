from __future__ import annotations

"""RESULTS.md 12.16: uncertainty, calibration, budget agreement and compute for the NLP test.

Why this exists. Prof. Hazra asked to finish the comparison of 12.15 and to compare epochs 1 to 3
so the early fine-tuning rule can be judged for the framework. analysis/nlp_option1_supplement.py
(job 11263574) describes the same decisions with Wilson and task-cluster bootstrap intervals, the
rule's observed calibration, how often the budgets agree, and compute against a full fine-tuning
lower bound. It was added after the 12.15 results were seen and changes no rule, threshold or model.
Run from the code root as a short job. Harry.
"""

import hashlib
import re
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "972e43434cda8b6180356c7a56050ebc"
BF = Path("benchmark_final")
PINNED = {
    Path("analysis/nlp_option1_supplement.py"): "b6de6a5e5859dfe02fb9b5df9cdc182c",
    BF / "nlp_option1_recs.tsv": "8a146de30d1cc103d8df324f983d99cb",
    BF / "nlp_option1_supplement_report.md": "d404e124397cd141a82c431874bb0df7",
    BF / "nlp_option1_supplement.tsv": "58b7229abe69d88bb0dea920e77e7c93",
}
HEAD = "### 12.16 Option 1 supplement: intervals, calibration, budget agreement and compute"
ANCHOR = "\n\n## 13. Cross-domain check: NLP encoder selection"
CHECKS = [
    "| fixed rule | 1 | 19/20 [76%, 99%] | 4/10 [17%, 69%] | 7/30 | 0.31 [0.00, 2.15] | 5.1 |",
    "| fixed rule | 3 | 18/20 [70%, 97%] | 4/10 [17%, 69%] | 8/30 | 0.54 [0.00, 2.15] | 15.2 |",
    "| fixed rule | 2 | 18/20 [70%, 97%] | 7/10 [40%, 89%] | 5/30 | 2.76 [0.00, 6.90] | 10.2 |",
    "| learned head-pair | 3 | 3/20 [5%, 36%] | 3/10 [11%, 60%] | 24/30 | 2.51 [0.46, 4.49] | "
    "15.2 |",
    "Always following H-score: regret 2.70 points [1.11, 4.15] over all 30 recommendations.",
    "| 3 | 18/22 (82%) | 2/8 (25%) |",
    "| 1 | 19/23 (83%) | 1/7 (14%) |",
    "agree between epoch 1 and epoch 3 on 29 of 30 recommendations, between epoch 2 and epoch 3 "
    "on 23 of 30.",
    "| full fine-tuning, one epoch over the full training split (lower bound) | 72.5 | 100% |",
    "| early, 1 epoch on the 10,000-example sample | 5.1 | 7% |",
    "| early, 3 epochs on the 10,000-example sample | 15.2 | 21% |",
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
    for p in PINNED:
        pinned(p)
    report = pinned(BF / "nlp_option1_supplement_report.md")
    flat = " ".join(report.split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in flat]
    if missing:
        sys.exit("ABORT, the report does not show:\n  " + "\n  ".join(missing))
    if not re.search(r"agnews 120,000, mnli 392,702, qnli 104,743, rte 2,490, "
                     r"airline 10,248", flat):
        sys.exit("ABORT: the training-split sizes moved")

    L = [f"{HEAD} (analysis/nlp_option1_supplement.py, job 11263574; added after the 12.15 "
         "results, describing the same decisions)", ""]
    L += wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/nlp_option1_supplement.py:", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append(ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: as a detector the fixed rule is strong on these data. At epoch 3 it flags 18 of 20 "
        "wrong recommendations (95% interval 70% to 97%); 82% of the recommendations it flags are "
        "wrong, against 25% of those it follows; and those it follows have regret 0.54 points "
        "(task-cluster "
        "interval 0.00 to 2.15) against 2.70 (1.11 to 4.15) for always following H-score. Its "
        "weaknesses are the false-alarm rate, 4 of 10 correct recommendations (17% to 69%), and "
        "coverage, 8 of 30. Epoch 1 makes the same decision as epoch 3 on 29 of 30 recommendations "
        "at a third of the cost (5.1 against 15.2 GPU-minutes per recommendation, 7% and 21% of "
        "one epoch of full fine-tuning over the full training splits, a lower bound because the "
        "authors' tuned setting allows up to 50 epochs with early stopping); epoch 2 agrees on "
        "only 23, so "
        "the budgets are not ordered by the counts alone. The learned head-pair model catches 3 of "
        "20 (5% to 36%) at every budget. Five tasks make every interval wide._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.16")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.16")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
