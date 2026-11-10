from __future__ import annotations

"""RESULTS.md 12.23: the per-epoch trajectory table (StudentBERT, NLP, Quick-Tune rule A).

Why this exists. Prof. Hazra (Nov 2026) asked to keep each candidate's per-epoch ranking
trajectory and compute cost for a regret-versus-compute analysis, approved rule A for Quick-Tune,
and asked that its limitation be documented clearly. analysis/trajectory_table.py (commit 9302502,
job 11634489) built the table after the design was committed; this records its report with the
files it rests on and the limitation. Run from the code root as a short job. Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "070b2162d36700183d2905c1649a8b3b"
BF = Path("benchmark_final")
QT = Path("/projects/algl/dai.hany/quicktune")
PINNED = {
    Path("docs/trajectory_design.md"): "9da80165f872ca8e572a0782afe474eb",
    Path("analysis/trajectory_table.py"): "5040a7fe5ea64f355ac875ed391cd88e",
    Path("analysis/qt_inspect.py"): "c811fac46ece2136aace4cff051c6ea8",
    BF / "trajectories_report.md": "5b0d4c16558db76ddc6f32d0e35f2b52",
    BF / "trajectories.tsv": "e93329cbc69ae8a3af843c544d43793b",
    BF / "early_ft_epochs.tsv": "602d8f8e997384835453edcc45e8e6da",
    BF / "executions.tsv": "92a3e5250b50727f911fe6f09d4b225b",
    QT / "micro.zip": "ed189adccce30d8b7f4a6e52b8f0caae",
    QT / "mini.zip": "0560bb028c2ea3c03cfb11165e9a6d87",
    QT / "extended.zip": "7619c2a3c6d5e13b410037037bfe577a",
}
HEAD = "### 12.23 Per-epoch trajectories and compute for every candidate"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = [
    "Rows per benchmark: nlp_early 1890, quicktune_extended 24250, "
    "quicktune_micro 33700, "
    "quicktune_mini 31900, studentbert 6720.",
    "quicktune_micro: rule A chose 674 runs on 30 tasks (models per task min 20, max 24; all 24 "
    "models on 4 tasks)",
    "quicktune_extended: rule A chose 485 runs on 26 tasks (models per task min 6, max 23; all "
    "24 models on 0 tasks)",
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
    report = (BF / "trajectories_report.md").read_text()
    flat = " ".join(report.split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in flat]
    if missing:
        sys.exit("ABORT, the report does not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/trajectory_table.py, commit 9302502, job 11634489)", ""]
    L += wrap(
        "Task: Prof. Hazra (Nov 2026) asked to keep, for every check, each candidate's per-epoch "
        "ranking trajectory and compute cost. One table holds, per candidate per epoch, the "
        "validation score, the rank among the candidates of its ranking and the epoch and "
        "cumulative seconds: StudentBERT (7 targets x 8 candidates x 6 seeds, epochs 1 to 20; cost "
        "the measured run time spread evenly over its epochs), the NLP short fine-tunes (three "
        "seed sets, epochs 1 to 3; cost measured per epoch) and Quick-Tune (Pineda Arango et "
        "al., ICLR 2024; epochs 1 to 50; cost the recorded training and evaluation time per "
        "epoch). The "
        "design was committed before the table was built.")
    L += [""] + wrap(
        "Quick-Tune rule A (Harry's choice, approved by Prof. Hazra): every Quick-Tune run uses "
        "its own random hyperparameters, and no configuration is shared across models on any task "
        "(analysis/qt_inspect.py), so each model on each task is represented by its first complete "
        "50-epoch run in file order, a rule that uses no result. Limitation: the models are "
        "therefore not compared under identical hyperparameters, and a model can rank low because "
        "its run drew poor settings rather than because it transfers worse. This limitation is "
        "stated wherever Quick-Tune results are used.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/trajectory_table.py:", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append(ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: rule A keeps 20 to 24 models on every micro task, 12 to 24 on mini and 6 to 23 on "
        "extended, where many runs end before 50 epochs; all 24 models have a complete run on 4 "
        "micro and 11 mini tasks and on no extended task. Candidate sets therefore differ by task, "
        "and every Quick-Tune regret is judged within its task's own set._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.23")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.23")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
