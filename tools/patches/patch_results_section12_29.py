from __future__ import annotations

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "eeeb20becddcab752782e7f64e505fe5"
BF = Path("benchmark_final")
T4 = Path("/projects/algl/dai.hany/task4")
QT = Path("/projects/algl/dai.hany/quicktune")
PINNED = {
    Path("docs/nlp_budget_extension_design.md"): "d82d64f47e3d23f35a94316446a0041b",
    Path("analysis/nlp_budget_extension.py"): "39d08a0959e4f99f5c01510912b2b840",
    Path("tests/test_nlp_budget_extension.py"): "94a8f7bbd4fab16aff25b83450b3ffc5",
    Path("analysis/selective_trust.py"): "561f9d471f9df4686c35bf6f5c7129a0",
    Path("analysis/regret_compute.py"): "fd4288700ffd9af74fe92911371935ae",
    Path("tools/compare_ft_epochs.py"): "16fbc8a6b27de3bec529729e1dec032e",
    Path("slurm/generators/gen_nlp_ft10_jobs.sh"): "8038b2ea85aa01443c800a6a891e604b",
    Path("scripts/task4_early_ft.py"): "9c87a401c955fc90aca772e5c17cbf08",
    BF / "nlp_budget_extension_report.md": "6737d85651b7bedb453967c3674e1333",
    BF / "nlp_budget_extension.tsv": "adb591aa86fbaed5a606da575b2c288f",
    BF / "nlp_budget_extension_references.tsv": "2e655e0112e3d896d70a75f36579b0e0",
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
REPORT = BF / "nlp_budget_extension_report.md"
HEAD = "### 12.29 NLP selective trust at 1, 2, 3, 5 and 10 epochs: coverage, regret and compute"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = [
    '| 42, 1, 2 | 210 | 44 | 0.4730 | 25.33 |',
    '| 3, 4, 5 | 210 | 40 | 0.4475 | 25.38 |',
    '| 6, 7, 8 | 210 | 42 | 0.2094 | 25.31 |',
    '| epoch 1 | 26, 1.27 | 25, 0.70 |',
    '| epoch 2 | 20, 0.73 | 20, 0.90 |',
    '| epoch 3 | 21, 0.47 | 24, 0.50 |',
    ('90 decisions. Mean regret of a random candidate (the normalizer): 3.3633 points.'
     ' At t = 0, with the area under the normalized selective-regret curve over all th'
     'resholds, the highest-coverage threshold that meets normalized selective regret '
     '<= 0.10 at that budget, and the references at the nearest compute (method settin'
     'g: mean regret in points at GPU-h).'),
    ('| 0 (trust all) | 0.029 | 90/90, 1.000 [1.000, 1.000] | 2.59 [1.00, 3.93] | 0.77'
     '0 [0.296, 1.170] | n/a | n/a | epoch 1: 1.64 at 0.086 | rungs 1,2,3: 1.35 at 0.1'
     '58 | rungs 1,2,3: 1.56 at 0.122 |'),
    ('| epoch 1 | 0.115 | 26/90, 0.289 [0.111, 0.522] | 1.27 [0.13, 3.80] | 0.377 [0.0'
     '39, 1.129] | 0.514 (from 0.011) | 0.40 [0.35, 0.40]: 1/90, 0.011 [0.000, 0.033];'
     ' 0.000 [0.000, 0.000] | epoch 1: 1.64 at 0.086 | rungs 1,2,3: 1.35 at 0.158 | ru'
     'ngs 1,2,3: 1.56 at 0.122 |'),
    ('| epoch 2 | 0.199 | 20/90, 0.222 [0.056, 0.433] | 0.73 [0.00, 3.45] | 0.217 [0.0'
     '00, 1.026] | 0.479 (from 0.044) | 0.15 [0.15, 0.15]: 7/90, 0.078 [0.000, 0.211];'
     ' 0.091 [0.000, 0.639] | epoch 2: 1.21 at 0.170 | start 1: 1.43 at 0.183 | start '
     '1: 1.28 at 0.158 |'),
    ('| epoch 3 | 0.283 | 21/90, 0.233 [0.044, 0.467] | 0.47 [0.00, 1.48] | 0.139 [0.0'
     '00, 0.439] | 0.394 (from 0.011) | 0.15 [0.10, 0.15]: 11/90, 0.122 [0.000, 0.322]'
     '; 0.075 [0.000, 0.412] | epoch 3: 1.00 at 0.254 | start 2: 1.34 at 0.363 | start'
     ' 2: 1.32 at 0.315 |'),
    ('| epoch 5 | 0.452 | 21/90, 0.233 [0.100, 0.422] | 0.74 [0.00, 2.07] | 0.219 [0.0'
     '00, 0.614] | 0.432 (from 0.033) | 0.10 [0.10, 0.10]: 14/90, 0.156 [0.011, 0.389]'
     '; 0.046 [0.000, 0.639] | epoch 5: 1.00 at 0.423 | start 3: 1.10 at 0.498 | start'
     ' 3: 0.95 at 0.472 |'),
    ('| epoch 10 | 0.874 | 29/90, 0.322 [0.144, 0.556] | 0.75 [0.00, 1.81] | 0.223 [0.'
     '000, 0.540] | 0.410 (from 0.011) | 0.60 [0.45, 0.60]: 1/90, 0.011 [0.000, 0.033]'
     '; 0.000 [0.000, 0.000] | epoch 10: 1.02 at 0.845 | start 3: 1.10 at 0.498 | star'
     't 3: 0.95 at 0.472 |'),
    'Not reached at any budget up to 10 epochs.',
    ('- selective trust at t = 0 (normalized selective regret over the trusted decisio'
     'ns): lowest mean normalized regret (k*) at epoch 3, 0.139 at 0.283 GPU-h; flatte'
     'ning at epoch 3 (0.139); larger budgets with a higher mean normalized regret tha'
     'n k*:'),
    ('  - epoch 5: 0.219, difference 0.080 [-0.020, 0.410], 2 resamples without a trus'
     'ted decision left out'),
    ('  - epoch 10: 0.223, difference 0.084 [-0.010, 0.349], 2 resamples without a tru'
     'sted decision left out'),
    ('- early leader (normalized regret over all 90 decisions): lowest mean normalized'
     ' regret (k*) at epoch 3, 0.296 at 0.254 GPU-h; flattening at epoch 3 (0.296); la'
     'rger budgets with a higher mean normalized regret than k*:'),
    '  - epoch 5: 0.297, difference 0.000 [-0.065, 0.058]',
    '  - epoch 10: 0.302, difference 0.006 [-0.067, 0.065]',
    ('- successive halving (normalized regret over all 90 decisions): lowest mean norm'
     'alized regret (k*) at start 3, 0.327 at 0.498 GPU-h; flattening at start 3 (0.32'
     '7); no larger budget has a higher mean normalized regret than k*.'),
    ('- successive halving (factor 3) (normalized regret over all 90 decisions): lowes'
     't mean normalized regret (k*) at start 3, 0.282 at 0.472 GPU-h; flattening at st'
     'art 3 (0.282); no larger budget has a higher mean normalized regret than k*.'),
    ('| selective trust at t = 0 | epoch 3 | 0.283 | 0.139 | epoch 3: 0.296 [0.097, 0.'
     '602] at 0.254 | start 2: 0.399 [0.104, 0.883] at 0.363 | start 2: 0.392 [0.130, '
     '0.798] at 0.315 |'),
    ('| early leader | epoch 3 | 0.254 | 0.296 | epoch 3: 0.296 [0.097, 0.602] at 0.25'
     '4 | start 1: 0.426 [0.116, 0.892] at 0.183 | start 2: 0.392 [0.130, 0.798] at 0.'
     '315 |'),
    ('| successive halving | start 3 | 0.498 | 0.327 | epoch 5: 0.297 [0.118, 0.548] a'
     't 0.423 | start 3: 0.327 [0.132, 0.608] at 0.498 | start 3: 0.282 [0.082, 0.587]'
     ' at 0.472 |'),
    ('| successive halving (factor 3) | start 3 | 0.472 | 0.282 | epoch 5: 0.297 [0.11'
     '8, 0.548] at 0.423 | start 3: 0.327 [0.132, 0.608] at 0.498 | start 3: 0.282 [0.'
     '082, 0.587] at 0.472 |'),
    ('| epoch 5 | studentbert | primary (by decision) | 0.10 [0.10, 0.10] | 0.094, 0.0'
     '38 | 8/21, 0.381 [0.095, 0.714] | 0.171 [0.000, 0.685] | 0.15 [0.15, 0.15] | 7/2'
     '1, 0.333 [0.048, 0.667] | 0.000 [0.000, 0.000] |'),
    ('| epoch 5 | nlp | sensitivity analysis (equal domain weight) | -0.30 [-0.30, -0.'
     '30] | 0.493, 0.086 | 49/90, 0.544 [0.278, 0.811] | 0.442 [0.132, 0.982] | 0.10 ['
     '0.10, 0.10] | 14/90, 0.156 [0.011, 0.389] | 0.046 [0.000, 0.639] |'),
    ('| epoch 10 | quicktune | primary (by decision) | 0.60 [0.45, 0.60] | 0.009, 0.00'
     '0 | 0/90, 0.000 [0.000, 0.000] | n/a | -0.30 [-0.30, -0.30] | 12/90, 0.133 [0.03'
     '3, 0.267] | 0.050 [0.000, 0.107] |'),
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
    L = [f"{HEAD} (analysis/nlp_budget_extension.py, commit 6f269bf, job 12136024)", ""]
    L += wrap(
        'Task: Prof. Hazra (2026-11-20 and 11-23), for Paper C: extend the NLP early fine-tuning '
        'runs to 5 and 10 epochs with the same models, tasks, seeds, decision rule, threshold and'
        ' compute accounting, without new tuning, and report coverage, regret and compute at 1, '
        '2, 3, 5 and 10 epochs, whether NLP ever reaches the selective-trust target, where the '
        'curve flattens or gets worse, the best budget against successive halving and the early '
        'leader at similar compute, and whether the useful budget is still domain-dependent. '
        'docs/nlp_budget_extension_design.md (commit 29652a2) fixed the runs, rules and read-outs'
        ' before any run; the 630 runs (7 encoders x 5 tasks x 2 poolings x 9 seeds) were rerun '
        'from the pretrained encoders to 10 epochs with scripts/task4_early_ft.py unchanged '
        '(slurm/generators/gen_nlp_ft10_jobs.sh, commit 998efa3; 210 jobs, all COMPLETED, 76.02 '
        'GPU-hours on V100-SXM2), after a 5-run protocol check. analysis/nlp_budget_extension.py '
        '(commit 3b542c0) was not run on real data before its first run (job 12135671); a report '
        'label (the factor-3 row of item 4) and the limitation wording on epochs 1 to 3 were then'
        ' corrected (commit 6f269bf) and the analysis rerun (job 12136024) with identical tables '
        '(the two table files have the same md5 in both runs). Evidence classes: the results up '
        'to epoch 3 were known before (12.27, 12.28); epochs 4 to 10, the budgets 5 and 10 and '
        'the read-outs are pre-specified and descriptive. Every NLP budget, including 1 to 3, '
        'comes from the new runs, so 12.28 stays as recorded.')
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items())
                     + ". " + (
        'The 630 new runs are pinned by the manifest md5 in the report; the tables hold the full '
        'threshold grid and the reference rows.'))
    L += ["", "Report as written by "
          "analysis/nlp_budget_extension.py (headings lowered two levels):", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("##" + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        '_Read: more early fine-tuning does not bring NLP to the selective-trust target. At t = 0'
        ' the trusted decisions are 26/90, 20/90, 21/90, 21/90 and 29/90 at 1, 2, 3, 5 and 10 '
        'epochs, with normalized selective regret 0.377, 0.217, 0.139, 0.219 and 0.223, for 0.115'
        ' to 0.874 GPU-hours per decision. No budget up to 10 epochs reaches normalized selective'
        ' regret at most 0.10 with coverage at least 0.50; the highest coverage under the regret '
        'target is 14/90 (0.156) at 5 epochs. Selective trust and the early leader are lowest at '
        '3 epochs (0.139 and 0.296) and flatten there; at 5 and 10 epochs selective trust is '
        'higher by 0.080 [-0.020, 0.410] and 0.084 [-0.010, 0.349], and the early leader by 0.000'
        ' and 0.006, intervals that span zero, so a worsening is not shown. Successive halving is'
        ' lowest at its largest configuration (start 3: 0.327 at 0.498 GPU-hours; factor 3: 0.282'
        ' at 0.472). Near 0.25 to 0.28 GPU-hours, selective trust at 3 epochs has 0.139 over its '
        '21 trusted decisions, the early leader at 3 epochs 0.296 over all 90, and successive '
        'halving 0.392 to 0.399; selective regret counts only the trusted decisions, so these are'
        ' not like for like. With 12.28 (StudentBERT reaches the target at 2 epochs, Quick-Tune '
        "does not), NLP's answer stays not reached up to 10 epochs; at 5 and 10 epochs a "
        'threshold chosen on the two other domains still misses on some held-out domains '
        '(held-out StudentBERT 0.171 at 5 epochs, held-out NLP 0.442 under equal domain weight at'
        ' 5, held-out Quick-Tune 0/90 at 10). The reruns did not replay the recorded first epochs'
        ' exactly (44, 40 and 42 of 210 runs per seed group repeat epochs 1 to 3; the largest '
        'difference is 0.4730), and the t = 0 rows at 1 to 3 epochs moved with them (1.27 points '
        'at 1 epoch against 0.70 in 12.28). Five tasks make the intervals wide, and the outcome '
        'is the published fully fine-tuned performance._')
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.29")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.29")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
