from __future__ import annotations

"""RESULTS.md 12.21: PARC as an additional transferability baseline.

Why this exists. Prof. Hazra (2026-10-27) asked to include PARC (Bolya et al., NeurIPS 2021). Its
settings were fixed in docs/parc_design.md and committed with the code (6d841b9) before any PARC
job ran; analysis/parc_report.py summarized the runs and checked that the evaluators reproduce
RESULTS.md 12.7 and 13.1. This records the report verbatim with the files it rests on. Run from
the code root as a short job. Harry.
"""

import hashlib
import os
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "f456eb0ccabd7ff666f998f9a093a28e"
BF = Path("benchmark_final")
T4 = Path(os.environ.get("TASK4_DIR", "/projects/algl/dai.hany/task4"))
PINNED = {
    Path("docs/parc_design.md"): "359effa3f201685de2cfcd48dbcb5bc8",
    Path("src/estimators/parc.py"): "7a0cc68bd4b9392c70070b31190badc5",
    Path("scripts/score_parc.py"): "4bcfccd6a14b6add6b9a4af26d6c0293",
    Path("scripts/task4_score_parc.py"): "ff529ab75b1e1bb0a19ce2bbbe23f33d",
    Path("analysis/parc_report.py"): "2aeaf5ec71a244dccb7418d04cd0e331",
    BF / "parc_report.md": "9168d99796667a3cb8f05c512604027c",
    BF / "parc_estimators_summary.tsv": "d2a05f591c19c208778469ba542179ff",
    BF / "task4_parc_summary.tsv": "9f83da2b69115ea0d0d6d6d406ba102b",
    T4 / "task4_parc_scores.jsonl": "5ebda1c51835cba7eca0b45a4b33e6a6",
    Path("tg1_parc_kt_algebra2005.jsonl"): "79d7a17779247908f5f60aca72bcae6f",
    Path("tg1_parc_kt_algebra2006.jsonl"): "042568523fa3b24cd7826d8399c9eb8c",
    Path("tg1_parc_kt_assist2009.jsonl"): "e7b98c82b1ee1af19fcce299be4d5869",
    Path("tg1_parc_kt_assist2017.jsonl"): "50338441f0699dacd1f61279e957844e",
    Path("tg1_parc_kt_bridge2006.jsonl"): "ba9a9deea80a50e2fe2a17a687b731c8",
    Path("tg1_parc_kt_ednet.jsonl"): "10fef3f7fcb6ee9e6dbe0930c5d42eb5",
    Path("tg1_parc_kt_junyi.jsonl"): "b08e7ac34a877e28e760b2c77b083c17",
}
HEAD = "### 12.21 PARC as an additional transferability baseline"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = [
    "| pretrained | 21 | -0.2347 | -0.1973 | 0.3801 | 1/21 | 1/21 | 0.0067 | 0.0103 | 5 | "
    "0.3651 |",
    "| tuned | 30 | -0.0131 | -0.0063 | 0.5107 | 8/30 | 10/30 | 3.87 | 10.00 | 0.6317 |",
    "| frozen | 30 | +0.2821 | +0.1968 | 0.6360 | 10/30 | 14/30 | 1.65 | 5.00 | 0.6317 |",
    "Own-encoder picks: 2/21.",
    "Check: the other 28 rows of the same evaluation equal",
    "reproduces all 6 estimator rows of RESULTS.md 13.1.",
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
    report = text[BF / "parc_report.md"]
    flat = " ".join(report.split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in flat]
    if missing:
        sys.exit("ABORT, the report does not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/parc_report.py; design docs/parc_design.md, commit 6d841b9)", ""]
    L += wrap(
        "Task: Prof. Hazra (2026-10-27) asked to include PARC (Bolya, Mittapalli and Hoffman, "
        "NeurIPS 2021) as an additional baseline. Settings, fixed in the design file and committed "
        "with the code before any PARC job ran: a probe of 500 examples with at least two per "
        "class, PCA to 32 dimensions fitted on the probe, distances one minus the Pearson "
        "correlation of the features and of the one-hot labels, and the Spearman correlation of "
        "the two upper triangles; no depth heuristic. Features: the same frozen causal features as "
        "the other StudentBERT estimators (7 GPU jobs, 168 scores), and the saved NLP features of "
        "13.1 (210 scores). Evaluation: the existing evaluators, unchanged, whose other rows "
        "reproduce 12.7 and 13.1 exactly.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/parc_report.py (headings lowered two levels):", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("##" + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: PARC, in the version fixed before scoring, is weak on both benchmarks. On "
        "StudentBERT its rankings run against the fine-tuned gold (rho -0.23; top choice the best "
        "in 1 of 21 rankings; 5 negatively transferring picks; mean regret 0.0067 AUC, against "
        "H-score's +0.52, 11 of 21 and 0.0012), and it picks the target's own encoder in only 2 "
        "of 21 rankings, so it does not share the own-encoder preference of H-score and LogME. On "
        "the NLP benchmark it agrees only weakly with frozen performance (+0.28) and not with "
        "fine-tuned performance (-0.01; the fine-tuned best in 8 of 30 rankings; mean regret 3.87 "
        "points against H-score's 2.70). Its rankings also change more across seeds (stability "
        "0.37 on StudentBERT and 0.63 on NLP). Bolya et al. report PARC at its best with a depth "
        "ensemble on a benchmark of diverse architectures; these numbers characterize the core "
        "method without that ensemble, as fixed in the design._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.21")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.21")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
