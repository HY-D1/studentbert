from __future__ import annotations

"""RESULTS.md 12.17: the frozen state before phase 2, and the one-epoch rule on StudentBERT.

Why this exists. Prof. Hazra's phase 2 task 1: freeze the current StudentBERT and NLP results in
RESULTS.md, including the fixed one-epoch rule. The NLP results are 12.15 and 12.16; this records
the freeze and adds the fixed rule on StudentBERT at both budgets (analysis/fixed_rule_eval.py,
epoch 3 as fixed on 2026-10-13 and, with --epoch 1, the one-epoch budget carried into phase 2), both
retrospective. Run from the code root as a short job. Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "55766e53de928ab1c88ca8dd015ed10e"
BF = Path("benchmark_final")
PINNED = {
    Path("analysis/fixed_rule_eval.py"): "94fc5f277d758c6b64fc8ebd5e169de2",
    BF / "reversal_pairs.tsv": "8f57558d142869b3e34a35ad18d095f7",
    BF / "fixed_rule_report.md": "9c9d0608a9c5f75e205b6f9e92aa34d3",
    BF / "fixed_rule_recs.tsv": "f2373c453120d3301a724410ccaa3706",
    BF / "fixed_rule_e1_report.md": "67e44aacb5ef638cad2a7b19c412c9fb",
    BF / "fixed_rule_e1_recs.tsv": "bae76c73c9558791b4e67c91d604f19f",
    Path("docs/pipeline_results_20261020.md"): "68f5ec603cd8e523974b621282e3d569",
    Path("docs/phase2_complete_rule_design.md"): "254538daf4b91503fe1cfe767f32ce0f",
    Path("docs/phase2_amendment1.md"): "ce8e827bfe9ffee6a074b498915c29cc",
}
HEAD = ("### 12.17 Frozen state before phase 2, and the fixed rule on StudentBERT at one and three "
        "epochs")
ANCHOR = "\n\n## 13. Cross-domain check: NLP encoder selection"
CHECKS = {
    BF / "fixed_rule_report.md": ["| always follow H-score | 21/21 | 0/7 | 0 | 0.0012 |",
                                  "| fixed epoch-3 rule | 12/21 | 6/7 | 3 | 0.0009 |"],
    BF / "fixed_rule_e1_report.md": ["| always follow H-score | 21/21 | 0/7 | 0 | 0.0012 |",
                                     "| fixed epoch-1 rule | 17/21 | 1/7 | 3 | 0.0009 |",
                                     "| algebra2005 | 42 | src:ednet | 0 | 1 | 1 |",
                                     "| assist2017 | 42 | src:assist2017 | 1 | 0 | 1 |",
                                     "| algebra2006 | 42 | src:algebra2006 | 1 | 0 | 1 |"],
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


def build() -> str:
    text = {p: pinned(p) for p in PINNED}
    missing = [f"{p.name}: {c}" for p, cs in CHECKS.items() for c in cs if c not in text[p]]
    if missing:
        sys.exit("ABORT, the reports do not show:\n  " + "\n  ".join(missing))

    def body(p: Path) -> list[str]:
        out = [ln for ln in text[p].splitlines() if not ln.startswith("# ")]
        while out and out[-1] == "":
            out.pop()
        return ["#" * 2 + ln if ln.startswith("## ") else ln for ln in out]

    L = [HEAD, ""]
    L += wrap(
        "Freeze. Sections 12.13 to 12.16 are frozen together with the write-up "
        "docs/pipeline_results_20261020.md and its checksum manifest, and the repository is tagged "
        "pipeline-results-20261020 (commit bcb20aa, 2026-10-20). Later results go into new "
        "sections. "
        "The rule carried into phase 2 is the same check after one epoch: flag H-score's "
        "recommendation when its pick is not the validation leader after one epoch of fine-tuning "
        "(docs/phase2_complete_rule_design.md and docs/phase2_amendment1.md).")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "The fixed rule on StudentBERT at epoch 3 (analysis/fixed_rule_eval.py, as fixed on "
          "2026-10-13):", ""] + body(BF / "fixed_rule_report.md")
    L += ["", "The same rule at epoch 1 (analysis/fixed_rule_eval.py --epoch 1):", ""]
    L += body(BF / "fixed_rule_e1_report.md")
    L += [""] + wrap(
        "_Read: both rows are retrospective, because the rule was suggested by diagnostic D8 on "
        "these targets. At epoch 3 the rule catches 6 of 7 reversed StudentBERT recommendations "
        "with 3 false alarms; at epoch 1 it catches 1 of 7 (Algebra 2005, seed 42), also with 3 "
        "false alarms, because the in-domain picks of ASSISTments 2017 and Algebra 2006 lead after "
        "one epoch on every seed and are overtaken later. On the NLP benchmark one epoch matches "
        "three (12.16: 19 of 20 caught at epoch 1, 18 of 20 at epoch 3). So the one-epoch budget "
        "is "
        "supported on NLP but not on StudentBERT, and the budget may depend on the domain._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.17")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.17")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
