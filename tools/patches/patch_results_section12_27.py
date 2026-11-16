from __future__ import annotations

"""RESULTS.md 12.27: regret against compute for static selection, the early leader, the override
rule, abstention and successive halving, on StudentBERT, NLP and Quick-Tune micro.

Why this exists. Prof. Hazra (Nov 2026) asked, as the key experiment for the methods question, for
a regret-versus-compute comparison of H-score, the early leader, our warning and override rule and
a simple multi-fidelity baseline such as successive halving, under the same pre-specified criteria,
with each method's place on the frontier. docs/regret_compute_design.md was committed (ef198fe)
before analysis/regret_compute.py existed; the script (commit 935ce90, job 11919022) reproduced
RESULTS.md 12.17, 12.18 and 12.24 on StudentBERT before reporting. This records its report with the
files it rests on. Run from the code root as a short job. Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "0c439ea10cd797ee9db6ec5bbefa51e9"
BF = Path("benchmark_final")
QT = Path("/projects/algl/dai.hany/quicktune")
PINNED = {
    Path("docs/regret_compute_design.md"): "4dc6393deea651bf59aa8144e1ce1608",
    Path("analysis/regret_compute.py"): "fd4288700ffd9af74fe92911371935ae",
    Path("analysis/regret_compute_figure.py"): "1ef0df36f93e72b3a2c9f09627f4bb20",
    BF / "regret_compute_report.md": "fe8155d94f0d79572f9df5af382f7908",
    BF / "regret_compute.tsv": "d77242c6243450571be358a5661c5ca6",
    BF / "trajectories.tsv": "e93329cbc69ae8a3af843c544d43793b",
    BF / "reversal_pairs.tsv": "8f57558d142869b3e34a35ad18d095f7",
    BF / "nlp_head_pairs.tsv": "d023dd9346eb1020afa11d1dacbaf579",
    BF / "p2_head_pairs.tsv": "9123ad27c17496af7f5f51cdb1987bc1",
    BF / "p3_head_pairs.tsv": "9bad0afce43c426c8105473a6b8f918c",
    QT / "qt_scores.jsonl": "b7f48f2f5c6ee9c45ce17201a6d2f9ea",
    QT / "features" / "extract_log.jsonl": "50ea8a9ba8fd20713c25211fd30c6f8e",
}
HEAD = "### 12.27 Regret against compute: static, early leader, override, abstention, halving"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = [
    "| static H-score |  | 0.008 | 21/21 | 0.0012 [0.0002, 0.0023] | 11 | yes |",
    "| early leader | epoch 3 | 0.118 | 21/21 | 0.0006 [0.0000, 0.0016] | 18 | yes |",
    "| override rule | epoch 5 | 0.204 | 21/21 | 0.0005 [0.0000, 0.0015] | 19 | yes |",
    "| warning, abstain | epoch 2 | 0.086 | 12/21 | 0.0003 [0.0000, 0.0008] | 8 |  |",
    "| early leader | epoch 10 | 0.392 | 21/21 | 0.0021 [0.0004, 0.0041] | 11 |  |",
    "| successive halving | start 1 | 0.078 | 21/21 | 0.0007 [0.0002, 0.0013] | 13 | yes |",
    "| successive halving | start 5 | 0.394 | 21/21 | 0.0013 [0.0004, 0.0024] | 9 |  |",
    "| full fine-tuning | best validation | 0.784 | 21/21 | 0.0003 [0.0000, 0.0006] | 15 | yes |",
    "| static H-score |  | 0.029 | 90/90 | 2.59 [1.00, 3.93] | 32 | yes |",
    "| early leader | epoch 3 | 0.253 | 90/90 | 0.97 [0.27, 2.03] | 48 | yes |",
    "| override rule | epoch 3 | 0.283 | 90/90 | 1.61 [0.39, 3.09] | 45 |  |",
    "| warning, abstain | epoch 3 | 0.283 | 24/90 | 0.50 [0.00, 1.56] | 17 |  |",
    "| successive halving | rungs 1,2,3 | 0.158 | 90/90 | 1.26 [0.43, 2.69] | 41 | yes |",
    "| static H-score |  | 0.040 | 90/90 | 40.56 [30.41, 51.01] | 6 |  |",
    "| static plain H-score |  | 0.039 | 90/90 | 26.23 [19.82, 33.08] | 6 | yes |",
    "| early leader | epoch 10 | 0.953 | 90/90 | 1.06 [0.33, 1.97] | 72 | yes |",
    "| override rule | epoch 10 | 0.953 | 90/90 | 1.06 [0.33, 1.97] | 72 | yes |",
    "| warning, abstain | epoch 2 | 0.189 | 0/90 | nan [nan, nan] | 0 |  |",
    "| successive halving | start 2 | 0.932 | 90/90 | 2.51 [0.72, 4.72] | 69 | yes |",
    "| early leader | epoch 50 | 4.946 | 90/90 | 0.00 [0.00, 0.00] | 90 | yes |",
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
    report = (BF / "regret_compute_report.md").read_text()
    flat = " ".join(report.split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in flat]
    if missing:
        sys.exit("ABORT, the report does not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/regret_compute.py, commit 935ce90, job 11919022)", ""]
    L += wrap(
        "Task: Prof. Hazra (Nov 2026) asked, as the key experiment for the methods question, for a "
        "regret-versus-compute comparison of H-score (static selection), the early leader, our "
        "warning and override rule, and successive halving, under the same pre-specified "
        "criteria, with each method's place on the frontier. docs/regret_compute_design.md was "
        "committed before the script existed. H-score's, the early leader's and the rule's "
        "results were known before (12.13 to 12.26); successive halving, the common compute "
        "accounting and the frontier are new, and the script was test-run on the StudentBERT "
        "scores with stand-in costs while it was written. Decisions: 21 on StudentBERT (8 "
        "candidates, six-seed test gold), 90 on NLP (7 encoders, published fine-tuned gold; every "
        "method stops at epoch 3, where the curves end) and 90 on Quick-Tune micro (models with "
        "at least 3 complete runs, median curves and the model-level median outcome of 12.26). "
        "Compute is GPU-hours per decision: StudentBERT's per-epoch costs are measured run times "
        "spread evenly over epochs and its scoring jobs' GPU is not recorded; NLP is V100-SXM2 "
        "time; Quick-Tune's fine-tuning time is its own on unrecorded hardware, so the static "
        "points' H200 scoring time is shown but counted as zero for the frontier. The frontier is "
        "the set of points that no other point beats on both mean compute and mean regret, among "
        "the methods that choose on every decision; abstention is reported beside it.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items())
                     + ". The frontier figure, benchmark_final/regret_compute_frontier.pdf, is "
                     "drawn from regret_compute.tsv by analysis/regret_compute_figure.py.")
    L += ["", "Report as written by analysis/regret_compute.py (headings lowered two levels):", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("##" + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: static selection is always the cheapest point and never good enough once a little "
        "fine-tuning is allowed. A simple multi-fidelity method matches or beats the override rule "
        "on every benchmark. On StudentBERT, successive halving started at one epoch reaches "
        "0.0007 AUC at 0.078 GPU-hours per decision, a tenth of fully fine-tuning every candidate "
        "(0.784 GPU-hours, 0.0003 when chosen by best validation), against 0.0012 for H-score "
        "alone; the early leader reaches 0.0006 at 3 epochs, and the override's best point, 0.0005 "
        "at 5 epochs, lies within the others' intervals. On NLP the override rule is never on the "
        "frontier: it keeps H-score's choice when the margin is not cleared, and its regret stays "
        "above the early leader's at every budget (1.61 against 0.97 points at 3 epochs); the "
        "frontier is H-score, the early leader at 1 and 3 epochs and successive halving. On "
        "Quick-Tune the override equals the early leader from 2 epochs on, because H-score's "
        "choice is always far behind, and the early leader and successive halving share the "
        "frontier (1.06 points at 10 epochs, 2.51 for halving at similar compute). What the rule "
        "adds is the warning: with abstention, the decisions it follows have the lowest regret "
        "(0.0003 AUC on 12 of 21 at 2 epochs on StudentBERT, 0.50 points on 24 of 90 at 3 epochs "
        "on NLP), at partial coverage, and on Quick-Tune it follows almost none (0 to 6 of 90; "
        "regret prints as nan where none is followed). On StudentBERT every method gets worse "
        "beyond about 5 epochs (the early leader 0.0021 at 10 epochs, halving 0.0013 when started "
        "at 5), the non-monotonicity of 12.24. StudentBERT is retrospective, NLP is truncated at 3 "
        "epochs, and Quick-Tune's outcome shares its curves with the methods' trajectories, so "
        "its regret at 50 epochs is zero by construction._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.27")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.27")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
