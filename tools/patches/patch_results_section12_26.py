from __future__ import annotations

"""RESULTS.md 12.26: Quick-Tune micro with a model-level outcome and a dimension control.

Why this exists. Prof. Hazra's reply to 12.25 (Nov 2026) asked for a model-level outcome that
aggregates each model's complete runs (median or mean, best run only as a sensitivity analysis)
and for a check of the conclusions after controlling for representation dimensionality.
docs/trajectory_amendment3.md fixed both before analysis/qt_model_level.py (commit ef899da, job
11851877) ran. This records its report verbatim with the files it rests on. Run from the code
root as a short job. Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "0c457a8b572964ede14eb70c347f304b"
BF = Path("benchmark_final")
QT = Path("/projects/algl/dai.hany/quicktune")
PINNED = {
    Path("docs/trajectory_amendment3.md"): "68997af8cd801ebca198776b2626c181",
    Path("analysis/qt_model_level.py"): "498ccccb5d9ed6de7a71417d1a1c40b2",
    Path("analysis/qt_score.py"): "3ace18aa5aee44106002d00f7a78defb",
    Path("analysis/qt_reversal.py"): "1877ad66cf8d54f28fddf358aa9c2791",
    BF / "qt_model_level_report.md": "dfdeb25cd579b43651df73b1e7239f0f",
    BF / "qt_model_level.tsv": "a83777a3aebb5a13c9a2659f5acf6223",
    QT / "qt_scores.jsonl": "b7f48f2f5c6ee9c45ce17201a6d2f9ea",
    QT / "features" / "extract_log.jsonl": "50ea8a9ba8fd20713c25211fd30c6f8e",
    QT / "micro.zip": "ed189adccce30d8b7f4a6e52b8f0caae",
}
HEAD = "### 12.26 Quick-Tune micro: model-level outcome and dimension control"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = [
    "Model-task pairs with complete runs: 674; included (at least 3 complete runs): 614 on 30 "
    "tasks; excluded: 60. Complete runs per included pair: median 10, range 3 to 23.",
    "Random model: mean regret 43.76 points. Outcome against log-dimension: mean Spearman +0.22.",
    "| raw | hscore_shrunk | 90 | +0.21 | +0.04 | +0.93 | 6/90 | 40.56 |",
    "| raw | hscore | 90 | +0.27 | +0.18 | +0.90 | 7/90 | 26.23 |",
    "| raw | logme | 90 | +0.12 | +0.10 | +0.44 | 6/90 | 45.57 |",
    "| pca128 | hscore_shrunk | 90 | +0.20 |  | +0.67 | 4/90 | 47.89 |",
    "| pca128 | hscore | 90 | +0.22 |  | +0.70 | 9/90 | 40.91 |",
    "| pca128 | logme | 90 | +0.20 |  | +0.66 | 7/90 | 36.21 |",
    "Random model: mean regret 20.30 points.",
    "| pca128 | hscore_shrunk | 90 | +0.41 |  | +0.67 | 32/90 | 11.00 |",
    "| raw | hscore | 90 | +0.31 | +0.24 | +0.90 | 13/90 | 13.52 |",
]


def pinned(p: Path) -> None:
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != PINNED[p]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[p]}; nothing written")


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def build() -> str:
    for p in PINNED:
        pinned(p)
    report = (BF / "qt_model_level_report.md").read_text()
    flat = " ".join(report.split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in flat]
    if missing:
        sys.exit("ABORT, the report does not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/qt_model_level.py, commit ef899da, job 11851877)", ""]
    L += wrap(
        "Task: Prof. Hazra's reply to 12.25 (Nov 2026) asked for a model-level outcome that "
        "aggregates each model's complete runs, with best-run accuracy only as a sensitivity "
        "analysis, since the best single run rewards lucky hyperparameters and drifts toward "
        "selecting a model together with its tuning; and for a check of the conclusions after "
        "controlling for representation dimensionality. docs/trajectory_amendment3.md fixed both "
        "before the run. Outcome: each model's median (primary), mean or best final top-1 over all "
        "its complete 50-epoch runs on a task, for models with at least 3 such runs. Scores: the "
        "12.25 scores (raw), and the same estimators on each model's features projected on their "
        "top 128 principal components, computed from the scored images without labels "
        "(pca128). The recommendation is again the shrunk H-score's choice. This is a follow-up "
        "to 12.25, not a replacement; rule A is not used here, and the outcome instead averages "
        "over Quick-Tune's sampled hyperparameters. Remaining limitation: the estimators may see "
        "images on which Quick-Tune's curves were validated.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/qt_model_level.py (headings lowered two levels):", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("##" + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: against a model's typical fine-tuned accuracy (the median over 614 model-task "
        "pairs with 3 to 23 complete runs), every estimator agrees only weakly, with mean Spearman "
        "+0.12 to +0.27. Controlling for log feature dimension leaves almost nothing of the shrunk "
        "H-score's correlation (partial +0.04), and its choice does about as well as a random "
        "model (mean regret 40.56 against 43.76 points): its apparent signal was mostly feature "
        "size. Plain H-score keeps a modest signal (partial +0.18) and a clearly lower regret "
        "(26.23). Projecting every model on 128 principal components does not remove the link to "
        "size, since the scores still correlate +0.66 to +0.70 with the original dimension, and "
        "the outcome itself rises slightly with dimension (+0.22); size and transferability "
        "cannot be fully separated here, and the partial correlation is the cleaner control. With "
        "best-run accuracy, the sensitivity outcome, all correlations are higher (+0.28 to +0.41) "
        "and regrets lower, as expected when lucky settings are rewarded. The conclusion of 12.25 "
        "therefore holds with a model-level outcome and with dimension controlled: on Quick-Tune "
        "micro, H-score predicts a model's typical fine-tuned accuracy only weakly, and its "
        "shrunk form hardly at all once feature size is accounted for._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.26")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.26")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
