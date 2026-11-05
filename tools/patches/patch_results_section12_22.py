from __future__ import annotations

"""RESULTS.md 12.22: measured cost of fully fine-tuning every StudentBERT candidate.

Why this exists. Prof. Hazra (Nov 2026) asked for the measured cost of fully fine-tuning all
candidates on one KT target, toward a regret-versus-compute analysis. analysis/kt_finetune_cost.py
(commit 4c8e75c, job 11631444) computed it from executions.tsv; this records its report verbatim
with the files it rests on. Run from the code root as a short job. Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "793352815f0471d07acc0ada6788c7f9"
BF = Path("benchmark_final")
PINNED = {
    Path("analysis/kt_finetune_cost.py"): "f0470c7cd0b60990e8955bb9439d846d",
    BF / "executions.tsv": "92a3e5250b50727f911fe6f09d4b225b",
    BF / "kt_finetune_cost_report.md": "19400b9033fa169bb7f78d1269d624f2",
    BF / "kt_finetune_cost.tsv": "8d48cdbf826faab22a1cccce4fd81d89",
}
HEAD = "### 12.22 Measured cost of fully fine-tuning every StudentBERT candidate"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = [
    "| ednet | 48 | 19.83 | 3.30 | 23.3 to 35.9 | v100-sxm2 | 2 |",
    "| junyi | 48 | 5.56 | 0.93 | 6.5 to 9.5 | v100-sxm2 | 2 |",
    "| algebra2005 | 48 | 0.71 | 0.12 | 0.8 to 1.0 | v100-sxm2 | 2 |",
    "| assist2009 | 48 | 2.70 | 0.45 | 3.2 to 3.7 | v100-sxm2 | 2 |",
    "Whole grid (7 targets x 8 candidates x 6 seeds): 32.8 GPU-hours.",
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
    report = pinned(BF / "kt_finetune_cost_report.md")
    for p in PINNED:
        pinned(p)
    flat = " ".join(report.split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in flat]
    if missing:
        sys.exit("ABORT, the report does not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/kt_finetune_cost.py, commit 4c8e75c, job 11631444)", ""]
    L += wrap(
        "Task: Prof. Hazra (Nov 2026) asked for the measured cost of fully fine-tuning all "
        "candidates on one KT target, toward a regret-versus-compute analysis. Source: the runs "
        "behind the Track B gold in executions.tsv (task kt, valid for the primary analysis, 20 "
        "epochs, 3,000 target learners or the whole training split). Every target, candidate and "
        "seed was run twice (campaigns TG1 and N3000P); a rerun is not part of a method's cost, so "
        "each counts once, at the median elapsed time of its copies. Elapsed is the scheduler's "
        "wall-clock time for a one-GPU job, including start-up, data loading and evaluation, so it "
        "is an upper bound on training time.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/kt_finetune_cost.py (heading lowered two levels):", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append(ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: fully fine-tuning all 8 candidates once, at one seed, costs 0.12 to 0.45 "
        "GPU-hours on the five smaller targets, 0.93 on Junyi Academy and 3.30 on EdNet, all on "
        "V100-SXM2; the whole 7 by 8 by 6 grid took 32.8 GPU-hours. Candidates cost about the "
        "same within a target, so the target's size sets the cost. With 20 epochs of similar "
        "length, one epoch is roughly "
        "a twentieth of a full fine-tune, an approximation until per-epoch times are recorded._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.22")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.22")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
