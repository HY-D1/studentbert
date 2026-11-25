from __future__ import annotations

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "a1e94d20d398ab3cbf5efb0560776b4f"
BF = Path("benchmark_final")
T4 = Path("/projects/algl/dai.hany/task4")
QT = Path("/projects/algl/dai.hany/quicktune")
PINNED = {
    Path("analysis/noise_floor.py"): "fcc5c16b485fc061a04ff43599851224",
    Path("tests/test_noise_floor.py"): "5c20425bfe2eb6cee8d399aaa6f66d4f",
    Path("analysis/nlp_budget_extension.py"): "39d08a0959e4f99f5c01510912b2b840",
    Path("analysis/reversal_table.py"): "6f56ff64bd2bc6b86637c4a8eb464b51",
    Path("analysis/regret_compute.py"): "fd4288700ffd9af74fe92911371935ae",
    BF / "noise_floor_report.md": "c0b276810b4dfd6c3fc7ca05e9c996fc",
    BF / "noise_floor.tsv": "39a48176dd549371d5b56cd89a781834",
    BF / "executions.tsv": "92a3e5250b50727f911fe6f09d4b225b",
    BF / "trajectories.tsv": "e93329cbc69ae8a3af843c544d43793b",
    BF / "reversal_pairs.tsv": "8f57558d142869b3e34a35ad18d095f7",
    Path("tg1_task2_kt_algebra2005.jsonl"): "268ff75c4c59a23b2fa4d9bcc85b60aa",
    Path("tg1_task2_kt_algebra2006.jsonl"): "d0f758940ee42bfdd8275e00ff12fe49",
    Path("tg1_task2_kt_assist2009.jsonl"): "5823dd2999f96921409b9cce5272d924",
    Path("tg1_task2_kt_assist2017.jsonl"): "f8a29e8c85e022e3306350ee159e31ff",
    Path("tg1_task2_kt_bridge2006.jsonl"): "38235d16b5d18d72d97bb9d8b9a9a0e0",
    Path("tg1_task2_kt_ednet.jsonl"): "59096f8a8941896c18ca60515ffdd9f2",
    Path("tg1_task2_kt_junyi.jsonl"): "bc1af53d5d042f25d57057952795dad0",
    BF / "nlp_head_pairs.tsv": "d023dd9346eb1020afa11d1dacbaf579",
    BF / "p2_head_pairs.tsv": "9123ad27c17496af7f5f51cdb1987bc1",
    BF / "p3_head_pairs.tsv": "9bad0afce43c426c8105473a6b8f918c",
    T4 / "task4_scores.jsonl": "9dfe5d930681ec04d79cb52c149bbc4e",
    T4 / "task4_scores_fresh.jsonl": "20c403eb43ade91bac987af41538299e",
    T4 / "task4_scores_p3.jsonl": "34ef47d06f1e7d155c585547a51f9220",
    QT / "qt_scores.jsonl": "b7f48f2f5c6ee9c45ce17201a6d2f9ea",
    QT / "features" / "extract_log.jsonl": "50ea8a9ba8fd20713c25211fd30c6f8e",
}
REPORT = BF / "noise_floor_report.md"
HEAD = "### 12.30 Noise floor per domain, and every method relative to it"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = [
    ('Primary floor: 0.0003 [0.0000, 0.0006] test AUC, normalized 0.033; zero regret o'
     'n 15 of 21 decisions.'),
    ("Spread over the 6 fine-tuning seeds (each seed's pick, mean over the 7 targets; "
     'ties to the candidate name): seed 1 0.0004, seed 2 0.0009, seed 3 0.0001, seed 4'
     ' 0.0001, seed 5 0.0004, seed 42 0.0003; range 0.0001 to 0.0009.'),
    ('Secondary, the seed oracle (the best test AUC of one seed against the 6-seed mea'
     'n): seed 1 0.0002, seed 2 0.0002, seed 3 0.0000, seed 4 0.0001, seed 5 0.0000, s'
     'eed 42 0.0001; mean 0.0001.'),
    ('| static H-score |  | 21/21 | 0.008 | 0.0012 | 0.0003 | 0.0010 [0.0000, 0.0020] '
     '| 0.131 [0.003, 0.269] | above the floor | n/a |'),
    ('| early leader | epoch 1 | 21/21 | 0.039 | 0.0031 | 0.0003 | 0.0029 [0.0004, 0.0'
     '068] | 0.383 [0.050, 0.908] | above the floor | 0.0019 [0.0000, 0.0049] |'),
    ('| early leader | epoch 2 | 21/21 | 0.078 | 0.0010 | 0.0003 | 0.0008 [-0.0002, 0.'
     '0023] | 0.101 [-0.022, 0.309] | not distinguishable from the floor | -0.0002 [-0'
     '.0013, 0.0007] |'),
    ('| early leader | epoch 3 | 21/21 | 0.118 | 0.0006 | 0.0003 | 0.0004 [-0.0004, 0.'
     '0015] | 0.048 [-0.058, 0.200] | not distinguishable from the floor | -0.0006 [-0'
     '.0016, 0.0001] |'),
    ('| early leader | epoch 5 | 21/21 | 0.196 | 0.0006 | 0.0003 | 0.0003 [-0.0005, 0.'
     '0015] | 0.046 [-0.063, 0.198] | not distinguishable from the floor | -0.0006 [-0'
     '.0016, 0.0001] |'),
    ('| early leader | epoch 10 | 21/21 | 0.392 | 0.0021 | 0.0003 | 0.0019 [0.0000, 0.'
     '0039] | 0.249 [0.002, 0.522] | above the floor | 0.0009 [-0.0017, 0.0037] |'),
    ('| early leader | epoch 20 | 21/21 | 0.784 | 0.0017 | 0.0003 | 0.0015 [0.0003, 0.'
     '0032] | 0.197 [0.045, 0.427] | above the floor | 0.0005 [-0.0014, 0.0027] |'),
    ('| successive halving | start 1 | 21/21 | 0.078 | 0.0007 | 0.0003 | 0.0004 [-0.00'
     '02, 0.0011] | 0.058 [-0.022, 0.144] | not distinguishable from the floor | -0.00'
     '06 [-0.0011, -0.0001] |'),
    ('| successive halving | start 2 | 21/21 | 0.157 | 0.0007 | 0.0003 | 0.0005 [-0.00'
     '02, 0.0013] | 0.064 [-0.024, 0.167] | not distinguishable from the floor | -0.00'
     '05 [-0.0016, 0.0007] |'),
    ('| full fine-tuning | best validation | 21/21 | 0.784 | 0.0003 | 0.0003 | 0.0000 '
     '[0.0000, 0.0000] | 0.000 [0.000, 0.000] | not distinguishable from the floor | -'
     '0.0010 [-0.0020, -0.0000] |'),
    ('Primary floor: 0.93 [0.16, 1.97] points, normalized 0.278; zero regret on 54 of '
     '90 decisions.'),
    ('Spread over the 9 estimator seeds (mean over 10 decisions each): seed 1 1.12, se'
     'ed 2 1.22, seed 3 1.82, seed 4 0.72, seed 5 0.34, seed 6 0.33, seed 7 0.91, seed'
     ' 8 0.93, seed 42 1.03; range 0.33 to 1.82. The NLP outcome is the published full'
     'y fine-tuned performance, so this floor includes the difference between these 10'
     ',000-example runs and the published protocol.'),
    ("Secondary, within the protocol (one seed's epoch-10 pick against the mean epoch-"
     '10 validation accuracy of the other 8 seeds, validation points, not subtracted f'
     'rom the methods): 1.33.'),
    ('| static H-score |  | 90/90 | 0.029 | 2.59 | 0.93 | 1.65 [0.16, 3.21] | 0.492 [0'
     '.047, 0.953] | above the floor | n/a |'),
    ('| early leader | epoch 1 | 90/90 | 0.086 | 1.64 | 0.93 | 0.71 [-0.04, 1.45] | 0.'
     '210 [-0.012, 0.431] | not distinguishable from the floor | -0.95 [-2.53, 0.47] |'),
    ('| override rule | epoch 1 | 90/90 | 0.115 | 2.02 | 0.93 | 1.09 [0.13, 2.17] | 0.'
     '324 [0.039, 0.646] | above the floor | -0.57 [-2.13, 0.59] |'),
    ('| override rule | epoch 2 | 90/90 | 0.199 | 2.13 | 0.93 | 1.19 [0.08, 2.59] | 0.'
     '354 [0.023, 0.770] | above the floor | -0.46 [-2.40, 1.52] |'),
    ('| early leader | epoch 3 | 90/90 | 0.254 | 1.00 | 0.93 | 0.06 [-0.13, 0.21] | 0.'
     '018 [-0.039, 0.062] | not distinguishable from the floor | -1.59 [-3.21, -0.17] '
     '|'),
    ('| override rule | epoch 5 | 90/90 | 0.452 | 1.42 | 0.93 | 0.48 [0.01, 1.17] | 0.'
     '143 [0.002, 0.348] | above the floor | -1.17 [-2.81, -0.02] |'),
    ('| override rule | epoch 10 | 90/90 | 0.874 | 1.62 | 0.93 | 0.68 [0.04, 1.41] | 0'
     '.203 [0.011, 0.419] | above the floor | -0.97 [-2.60, 0.31] |'),
    ('| successive halving | start 3 | 90/90 | 0.498 | 1.10 | 0.93 | 0.16 [0.04, 0.29]'
     ' | 0.049 [0.011, 0.087] | above the floor | -1.49 [-3.05, -0.05] |'),
    ('Primary floor: 23.62 [19.58, 27.62] top-1 points, normalized 0.540; zero regret '
     'on 0 of 90 decisions.'),
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
    report = REPORT.read_text()
    flat = " ".join(report.split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in flat]
    if missing:
        sys.exit("ABORT, the report does not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/noise_floor.py, commit 6f269bf, job 12136043)", ""]
    L += wrap(
        'Task: Prof. Hazra (2026-11-24, before drafting Paper C): a noise-floor analysis per '
        'domain, to show whether the regret differences between methods are meaningfully above '
        'the variability from using one validation seed instead of the multi-seed final outcome, '
        'with methods reported relative to the floor and paired bootstrap uncertainty. The floor '
        'and read-outs follow a design written on 2026-11-24 before analysis/noise_floor.py; at '
        "Harry's request the design is kept outside the repository, so no commit documents its "
        'date, and the analysis is described as designed before the run, not as pre-specified. '
        'The script (commit 4d49821) was not run on real data before its first run (job '
        '12135833); after the audit its Quick-Tune wording was corrected, as marked in the '
        'report, and it was rerun (commit 6f269bf, job 12136043) with the same table '
        '(noise_floor.tsv has the same md5 in both runs). Floor: the regret of choosing by one '
        "seed's (or one run's) complete fine-tuning, the best validation score over its epochs; "
        'StudentBERT and Quick-Tune decisions as in 12.27, NLP from the 10-epoch campaign of '
        '12.29. The StudentBERT floor at the estimator seeds is the 12.27 row full fine-tuning, '
        'best validation, known before. Descriptive.')
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items())
                     + ". " + (
        'The design file is outside the repository and is not pinned.'))
    L += ["", "Report as written by "
          "analysis/noise_floor.py (headings lowered two levels):", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("##" + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: on StudentBERT and NLP the static choice's regret is above the seed floor, and "
        "most multi-fidelity rows are not distinguishable from it. StudentBERT's floor is 0.0003 "
        '[0.0000, 0.0006] test AUC (15 of 21 decisions at zero; 0.0001 to 0.0009 over the 6 '
        'seeds; the seed oracle, which no selection rule can remove, 0.0001); the static H-score '
        'choice is above it by 0.0010 [0.0000, 0.0020], the early leader at 2, 3 and 5 epochs and'
        ' successive halving started at 1 or 2 epochs are not distinguishable from it, and the '
        "early leader at 1, 10 and 20 epochs is above it. NLP's floor is 0.93 [0.16, 1.97] points"
        ' (0.33 to 1.82 over the 9 seeds; 1.33 validation points within the protocol); the static'
        ' choice is above it by 1.65 [0.16, 3.21], the early leader is not distinguishable from '
        'it at any budget (0.71 [-0.04, 1.45] at 1 epoch, 0.06 [-0.13, 0.21] at 3), the override '
        'rule is above it at 1, 2, 5 and 10 epochs, and successive halving is not distinguishable'
        " from it except at start 3 (0.16 [0.04, 0.29]). The static choice's excess over the "
        'floor (0.0010 AUC, 1.65 points) is larger than the floor itself (0.0003, 0.93). On '
        'Quick-Tune the floor (23.62 top-1 points) measures a hyperparameter draw, not a seed, '
        'and is not comparable with the methods, which read median curves over the same runs that'
        ' define the outcome; the Quick-Tune rows are not a seed-noise comparison. StudentBERT '
        'has 7 targets and NLP 5 tasks, and the NLP floor includes the protocol difference to the'
        ' published outcome._')
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.30")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.30")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
