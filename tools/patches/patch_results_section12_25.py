from __future__ import annotations

"""RESULTS.md 12.25: H-score on Quick-Tune micro (pre-specified result and post-hoc diagnostic).

Why this exists. Prof. Hazra (Nov 2026) asked whether H-score can be computed on the Quick-Tune
tasks and whether their curves can show when a reversal becomes detectable. The design
(docs/qt_hscore_design.md) and amendment 2 were committed before any feature or score existed;
analysis/qt_reversal.py (commit c930810, job 11808894) produced the pre-specified result, and
analysis/qt_diagnostics.py (commit e748869, job 11809464) was added afterwards to explain it. This
records both verbatim, labelled as such, with the files they rest on. Run from the code root as a
short job. Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "772b940b4d65fe41d33dab6d83e3dfc2"
BF = Path("benchmark_final")
QT = Path("/projects/algl/dai.hany/quicktune")
PINNED = {
    Path("docs/qt_hscore_design.md"): "9131a8e9221faf387cb4ff3c9d87fcf1",
    Path("docs/trajectory_amendment2.md"): "8859f77870d589845fc5d373a6cae773",
    Path("analysis/qt_fetch_metaalbum.py"): "0850a7b55419bc5acd752956d136287f",
    Path("analysis/qt_extract_features.py"): "49f0d1fcdba7442465f3a306fb2b3f2b",
    Path("analysis/qt_score.py"): "3ace18aa5aee44106002d00f7a78defb",
    Path("analysis/qt_reversal.py"): "1877ad66cf8d54f28fddf358aa9c2791",
    Path("analysis/qt_diagnostics.py"): "8bb9857c7a93b2825c300085d902a585",
    BF / "trajectories.tsv": "e93329cbc69ae8a3af843c544d43793b",
    BF / "qt_reversal_report.md": "5b6c386718ae4818c81be9eb4410bf48",
    BF / "qt_reversal.tsv": "9fc1b8f63f4978e6b9728edc2d02a1cf",
    BF / "qt_diagnostics_11809464.log": "2507c1726127b85679ed7d9729214224",
    QT / "metaalbum_micro_report.json": "1700e621ccce6d64692db92068378006",
    QT / "features" / "extract_log.jsonl": "50ea8a9ba8fd20713c25211fd30c6f8e",
    QT / "qt_scores.jsonl": "b7f48f2f5c6ee9c45ce17201a6d2f9ea",
}
HEAD = "### 12.25 H-score on Quick-Tune micro"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = {
    BF / "qt_reversal_report.md": [
        "| hscore_shrunk | 2022/2022 | 90 | +0.11 | 5/90 | 56.54 |",
        "| hscore | 2022/2022 | 90 | +0.17 | 4/90 | 34.46 |",
        "| logme | 1146/2022 | 90 | +0.05 | 0/90 | 60.09 |",
        "by estimator: logme 876.",
        "## Warning rule on shrunk H-score's choice (90 recommendations, 85 overtaken)",
        "mean 2.9 GPU-minutes on NVIDIA H200"],
    BF / "qt_diagnostics_11809464.log": [
        "Rule A runs: 674; ending near chance (top-1 <= 10 at epoch 50): 257",
        "beit_large_patch16_512 (1024): 16/20 near chance, mean final 7.8",
        "Mean Spearman of final top-1 with feature dimension, per task: +0.12",
        "Mean regret of a random candidate: 50.88 points",
        "hscore_shrunk: picks [('tf_efficientnet_b6_ns', 27), ('beit_large_patch16_512', 25), "
        "('tf_efficientnet_b7_ns', 16), ('swinv2_base_window12to24_192to384_22kft1k', 14),",
        "mean Spearman of score with dimension +0.93",
        "hscore: picks [('volo_d3_448', 47), ('volo_d4_448', 15), ('volo_d5_448', 10), "
        "('volo_d5_512', 5), ('convnext_small_384_in22ft1k', 3), ('volo_d1_384', 2)]",
        "mean Spearman of score with dimension +0.88",
        "Mean best final top-1 per task: 84.2"],
}


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
    for p, checks in CHECKS.items():
        flat = " ".join(p.read_text().split())
        missing = [c for c in checks if " ".join(c.split()) not in flat]
        if missing:
            sys.exit(f"ABORT, {p.name} does not support the prose; nothing written:\n  "
                     + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/qt_reversal.py, commit c930810, job 11808894; post-hoc diagnostic "
         "analysis/qt_diagnostics.py, commit e748869, job 11809464)", ""]
    L += wrap(
        "Task: Prof. Hazra (Nov 2026) asked whether H-score can be computed on the Quick-Tune "
        "tasks and whether their curves can show when a reversal becomes detectable. Design: "
        "docs/qt_hscore_design.md and docs/trajectory_amendment2.md, both committed before any "
        "feature or score existed. The 30 Meta-Album micro tasks (40 images per class, 20 classes, "
        "19 on DOG and PNU) were fetched from OpenML; frozen features of the 24 models were "
        "extracted on an H200 with each model's own preprocessing; H-score, shrunk H-score and "
        "LogME (the NLP estimator code) scored each task's rule A candidates on stratified halves "
        "at seeds 42, 1 and 2. The recommendation, fixed in advance, is the shrunk H-score's "
        "choice; plain H-score and LogME are secondary rows. Limitations: rule A (12.23; models "
        "are not compared under identical hyperparameters) and the image split (the estimators "
        "may see images on which Quick-Tune's curves were validated).")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Pre-specified result, as written by analysis/qt_reversal.py (headings lowered two "
          "levels):", ""]
    for ln in (BF / "qt_reversal_report.md").read_text().splitlines():
        if ln.startswith("# "):
            continue
        L.append("##" + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += ["", "Post-hoc diagnostic, added after the result above to explain it; it changes no rule "
          "and no result (output of analysis/qt_diagnostics.py, job 11809464):", "", "```"]
    L += (BF / "qt_diagnostics_11809464.log").read_text().rstrip("\n").splitlines()
    L += ["```", ""]
    L += wrap(
        "_Read: under the pre-specified rule, shrunk H-score's choice is overtaken on 85 of 90 "
        "recommendations, with a mean regret of 56.5 points, more than the 50.9 of a random "
        "candidate. The post-hoc diagnostic shows two causes. First, 257 of the 674 rule A runs "
        "(38%) end near chance (top-1 at most 10 points at epoch 50), among them 16 of the 20 "
        "BEiT-large runs: under rule A a model's final accuracy mostly reflects whether its one "
        "sampled configuration worked, and it barely relates to feature dimension (mean Spearman "
        "+0.12). Second, with 400 images (380 on DOG and PNU) H-score largely tracks feature "
        "dimension (mean Spearman of score with dimension +0.93 for shrunk and +0.88 for plain "
        "H-score): shrunk H-score picks EfficientNet-B6, BEiT-large, EfficientNet-B7 or SwinV2 on "
        "82 of 90 recommendations. Plain H-score picks a VOLO model on at least 79 of 90 and does "
        "better (mean regret 34.5); the pre-specified primary is not changed after the fact. LogME "
        "is undefined for 876 of the 2,022 candidate scores, where the feature dimension is at "
        "least the number of images. Quick-Tune under rule A therefore cannot test whether an "
        "estimator's choice is overtaken after fine-tuning, and the warning and detection rows "
        "above are not interpreted. Another representation of each model, such as its best final "
        "accuracy over all its complete runs, would need its own pre-specified analysis._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.25")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.25")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
