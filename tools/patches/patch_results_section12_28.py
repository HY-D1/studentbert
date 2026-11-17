from __future__ import annotations

"""RESULTS.md 12.28: selective trust in the static recommendation (coverage, selective regret and
compute over thresholds of the normalized lead; leave-one-domain-out threshold transfer with an
equal-domain-weight sensitivity analysis; the domain-specific budget), for Paper C.

Why this exists. Prof. Hazra (Nov 2026) asked, for Paper C, when the static H-score
recommendation can be trusted, how much compute that decision needs and whether one threshold
transfers across domains. docs/selective_trust_design.md (c8ca7dd) and its amendments 1 (882de48)
and 2 (992bd44) were committed before analysis/selective_trust.py (b90163f); job 11951748 ran it,
reproducing the 12.27 warning rows at t = 0 before reporting. This records its report with the
files it rests on. Run from the code root as a short job. Harry.
"""

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "99ed91735c97c5ead1002b774d9c7a0c"
BF = Path("benchmark_final")
T4 = Path("/projects/algl/dai.hany/task4")
QT = Path("/projects/algl/dai.hany/quicktune")
PINNED = {
    Path("docs/selective_trust_design.md"): "f63bfdc6bc0df97a97fb2fc97ef0eddd",
    Path("docs/selective_trust_amendment_1.md"): "de8a55b63b442dd2d1f28bb7bb2c1c9e",
    Path("docs/selective_trust_amendment_2.md"): "e8c9ae36be77cfb3fd887b1a6f0a82f2",
    Path("analysis/selective_trust.py"): "561f9d471f9df4686c35bf6f5c7129a0",
    Path("tests/test_selective_trust.py"): "38cbc5781baa722f080d80fe7610d98a",
    Path("analysis/regret_compute.py"): "fd4288700ffd9af74fe92911371935ae",
    BF / "selective_trust_report.md": "6d8abb21cfb3fea5ea23b1c9bd3a7ee3",
    BF / "selective_trust.tsv": "afde26b12e7cf0a30472b35febc94e65",
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
HEAD = "### 12.28 Selective trust in the static recommendation: coverage, compute and transfer"
ANCHOR = "\n\n## 13. Cross-domain check"
CHECKS = [
    ("21 decisions. Mean regret of a random candidate (the normalizer): 0.007540 test "
     "AUC. Fine-tuning every candidate fully: 0.784 GPU-h per decision."),
    ("| 0 (trust all) | 0.008 | 0.010 | 21/21, 1.000 [1.000, 1.000] | 0.0012 [0.0002, "
     "0.0023] | 0.164 [0.033, 0.301] | n/a | epoch 1: 0.0031 at 0.039 | start 1: 0.000"
     "7 at 0.078 | start 1: 0.0008 at 0.068 |"),
    ("| epoch 1 | 0.047 | 0.060 | 17/21, 0.810 [0.619, 0.952] | 0.0009 [0.0001, 0.0018"
     "] | 0.122 [0.012, 0.244] | 0.080 (from 0.048) | epoch 1: 0.0031 at 0.039 | start"
     " 1: 0.0007 at 0.078 | start 1: 0.0008 at 0.068 |"),
    ("| epoch 2 | 0.086 | 0.110 | 12/21, 0.571 [0.238, 0.857] | 0.0003 [0.0000, 0.0008"
     "] | 0.037 [0.000, 0.100] | 0.056 (from 0.048) | epoch 2: 0.0010 at 0.078 | start"
     " 1: 0.0007 at 0.078 | start 1: 0.0008 at 0.068 |"),
    ("| epoch 20 | 0.792 | 1.010 | 10/21, 0.476 [0.190, 0.762] | 0.0009 [0.0000, 0.001"
     "9] | 0.115 [0.006, 0.248] | 0.112 (from 0.048) | epoch 20: 0.0017 at 0.784 | sta"
     "rt 5: 0.0013 at 0.394 | start 5: 0.0009 at 0.344 |"),
    ("| epoch 1 | 0.115 | n/a | 25/90, 0.278 [0.078, 0.533] | 0.70 [0.00, 2.08] | 0.20"
     "8 [0.000, 0.620] | 0.472 (from 0.011) | epoch 1: 1.64 at 0.085 | rungs 1,2,3: 1."
     "26 at 0.158 | rungs 1,2,3: 1.62 at 0.121 |"),
    ("| epoch 2 | 0.199 | n/a | 20/90, 0.222 [0.089, 0.444] | 0.90 [0.00, 3.45] | 0.26"
     "9 [0.000, 1.026] | 0.526 (from 0.011) | epoch 2: 1.50 at 0.169 | rungs 1,2,3: 1."
     "26 at 0.158 | rungs 1,2,3: 1.62 at 0.121 |"),
    ("| epoch 3 | 0.283 | n/a | 24/90, 0.267 [0.067, 0.522] | 0.50 [0.00, 1.56] | 0.14"
     "9 [0.000, 0.465] | 0.424 (from 0.022) | epoch 3: 0.97 at 0.253 | rungs 1,2,3: 1."
     "26 at 0.158 | rungs 1,2,3: 1.62 at 0.121 |"),
    ("| epoch 1 | 0.094 | 0.019 | 3/90, 0.033 [0.000, 0.100] | 16.56 [16.56, 16.56] | "
     "0.378 [0.378, 0.378] | 0.815 (from 0.033) | epoch 1: 33.58 at 0.094 | start 1: 9"
     ".30 at 0.417 | start 1: 11.65 at 0.287 |"),
    ("| epoch 10 | 0.953 | 0.193 | 6/90, 0.067 [0.000, 0.167] | 0.00 [0.00, 0.00] | 0."
     "000 [0.000, 0.000] | 0.438 (from 0.033) | epoch 10: 1.06 at 0.953 | start 2: 2.5"
     "1 at 0.932 | start 3: 3.43 at 0.987 |"),
    ("| epoch 1 | studentbert | primary (by decision) | 0.30 [0.30, 0.30] | 0.011, 0.0"
     "00 | 6/21, 0.286 [0.095, 0.476] | 0.050 [0.000, 0.149] | 6/21, 0.286 [0.095, 0.4"
     "76]; 0.050 [0.000, 0.149] | 0.20 [0.20, 0.20] | 9/21, 0.429 [0.190, 0.667] | 0.0"
     "85 [0.012, 0.235] |"),
    ("| epoch 2 | studentbert | primary (by decision) | 0.30 [0.25, 0.30] | 0.017, 0.0"
     "00 | 4/21, 0.190 [0.000, 0.476] | 0.017 [0.000, 0.070] | 5/21, 0.238 [0.000, 0.5"
     "24]; 0.028 [0.000, 0.070] | -0.10 [-0.25, -0.10] | 17/21, 0.810 [0.524, 1.000] |"
     " 0.099 [0.012, 0.217] |"),
    ("| epoch 3 | studentbert | primary (by decision) | 0.10 [0.10, 0.10] | 0.094, 0.0"
     "75 | 8/21, 0.381 [0.095, 0.714] | 0.171 [0.000, 0.685] | 8/21, 0.381 [0.095, 0.7"
     "14]; 0.171 [0.000, 0.685] | 0.15 [0.15, 0.15] | 5/21, 0.238 [0.000, 0.571] | 0.0"
     "00 [0.000, 0.000] |"),
    ("| epoch 1 | nlp | primary (by decision) | 0.20 [0.20, 0.20] | 0.081, 0.085 | 7/9"
     "0, 0.078 [0.022, 0.133] | 0.352 [0.000, 0.639] | 7/90, 0.078 [0.022, 0.133]; 0.3"
     "52 [0.000, 0.639] | 0.30 [0.30, 0.30] | 2/90, 0.022 [0.000, 0.067] | 0.000 [0.00"
     "0, 0.000] |"),
    ("| epoch 2 | nlp | primary (by decision) | -0.10 [-0.25, -0.10] | 0.153, 0.099 | "
     "29/90, 0.322 [0.200, 0.522] | 0.473 [0.084, 1.215] | 46/90, 0.511 [0.356, 0.733]"
     "; 0.534 [0.165, 1.089] | 0.30 [0.25, 0.30] | 3/90, 0.033 [0.000, 0.100] | 0.000 "
     "[0.000, 0.000] |"),
    ("| epoch 3 | nlp | primary (by decision) | 0.15 [0.15, 0.15] | 0.045, 0.000 | 13/"
     "90, 0.144 [0.044, 0.300] | 0.098 [0.000, 0.479] | 13/90, 0.144 [0.044, 0.300]; 0"
     ".098 [0.000, 0.479] | 0.10 [0.10, 0.10] | 17/90, 0.189 [0.044, 0.411] | 0.075 [0"
     ".000, 0.479] |"),
    ("| epoch 3 | nlp | sensitivity analysis (equal domain weight) | -0.35 [-0.35, -0."
     "35] | 0.493, 0.076 | 54/90, 0.600 [0.378, 0.822] | 0.570 [0.227, 1.104] | 54/90,"
     " 0.600 [0.378, 0.822]; 0.570 [0.227, 1.104] | 0.10 [0.10, 0.10] | 17/90, 0.189 ["
     "0.044, 0.411] | 0.075 [0.000, 0.479] |"),
    ("| epoch 1 | quicktune | primary (by decision) | 0.30 [0.30, 0.30] | 0.072, 0.037"
     " | 0/90, 0.000 [0.000, 0.000] | n/a | 0/90, 0.000 [0.000, 0.000]; n/a | none | n"
     "/a | n/a |"),
    ("| studentbert | epoch 2 | -0.10 [-0.25, -0.10] | 17/21, 0.810 [0.524, 1.000] | 0"
     ".099 [0.012, 0.217] | 0.086 | 0.110 |"),
    "| nlp | not reached | n/a | n/a | n/a | n/a | n/a |",
    "| quicktune | not reached | n/a | n/a | n/a | n/a | n/a |",
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
    report = (BF / "selective_trust_report.md").read_text()
    flat = " ".join(report.split())
    missing = [c for c in CHECKS if " ".join(c.split()) not in flat]
    if missing:
        sys.exit("ABORT, the report does not support the prose; nothing written:\n  "
                 + "\n  ".join(missing))
    L = [f"{HEAD} (analysis/selective_trust.py, commit b90163f, job 11951748)", ""]
    L += wrap(
        "Task: Prof. Hazra (Nov 2026), for Paper C: the promising part of the rule is the warning, "
        "or selective-trust, signal. She asked when the static (H-score) recommendation can be "
        "trusted, when to abstain or keep evaluating, how much compute that decision needs, and "
        "whether one common threshold transfers across domains or each domain needs its own "
        "budget, with the early leader and successive halving (12.27) as the multi-fidelity "
        "comparisons. docs/selective_trust_design.md (commit c8ca7dd) and its amendments 1 "
        "(882de48: area, ties, pooling, budget rule, intervals, references) and 2 (992bd44: an "
        "equal-domain-weight sensitivity analysis of the transfer test) were committed before "
        "analysis/selective_trust.py (b90163f), which was not run on real data before this job; "
        "its tests use synthetic decisions. Evidence classes: the rule at t = 0 was known before "
        "(12.17, 12.20, 12.27) and is reproduced exactly, decision by decision and in every 12.27 "
        "warning row (the run stops otherwise); the threshold family, the normalized units, the "
        "area, the transfer test and its sensitivity analysis, the budget question and the "
        "reference points are pre-specified. Decisions and costs are those of 12.27, loaded by its "
        "code: 21 on StudentBERT, 90 on NLP (curves to epoch 3) and 90 on Quick-Tune micro. "
        "Compute per decision is frozen scoring plus k epochs of every candidate; Quick-Tune's "
        "scoring stays off the fine-tuning axis, so its trust-all row shows 0.000 GPU-hours where "
        "12.27's static row shows 0.040, and the share of full compute divides by fine-tuning "
        "alone, so StudentBERT's share at 20 epochs is 1.010.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items())
                     + ". selective_trust.tsv holds the full grid (domain, budget and threshold: "
                     "coverage, selective regret raw and normalized, compute), 741 rows.")
    L += ["", "Report as written by analysis/selective_trust.py (headings lowered two levels):", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("##" + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: in these data no single threshold carries across domains, and only one domain "
        "reaches the targets within its budgets. At t = 0, StudentBERT trusts 17/21 decisions at "
        "1 epoch (normalized selective regret 0.122) and 12/21 at 2 epochs (0.037), NLP trusts "
        "20/90 to 25/90 (0.149 to 0.269), and Quick-Tune at most 6/90. Only StudentBERT meets the "
        "pre-specified targets (normalized selective regret at most 0.10 with coverage at least "
        "0.50): at 2 epochs with t = -0.10 it trusts 17/21 at 0.099 [0.012, 0.217], for 0.086 "
        "GPU-hours per decision, 0.110 of fully fine-tuning every candidate; the point sits at "
        "the target and its interval spans it. NLP, whose curves stop at 3 epochs, and "
        "Quick-Tune, up to 50, do not reach the targets. A threshold chosen on the two other "
        "domains misses on the held-out one: held-out Quick-Tune trusts 0/90 at every shared "
        "budget; held-out NLP has normalized selective regret 0.352 and 0.473 at 1 and 2 epochs "
        "and 0.098 on 13/90 at 3; held-out StudentBERT stays under the target at 1 and 2 epochs "
        "(0.050 and 0.017) but trusts fewer decisions than its own best threshold (6/21 against "
        "9/21, 4/21 against 17/21), and has 0.171 at 3 epochs. The equal-domain-weight "
        "sensitivity analysis finds no qualifying threshold in 5 of the 9 cases; where it differs "
        "from the primary rule (NLP held out at 3 epochs, t = -0.35), the held-out regret is "
        "0.570. At StudentBERT's operating point the multi-fidelity references choose on every "
        "decision at similar compute (the early leader 0.0010 AUC at 0.078 GPU-hours, successive "
        "halving 0.0007 at 0.078, against a normalizer of 0.007540), so at that compute selective "
        "trust is not shown to beat successive halving on regret; what it adds is knowing which "
        "decisions to evaluate further. Intervals are wide (7 StudentBERT targets, 5 NLP tasks). "
        "StudentBERT is retrospective, NLP stops at 3 epochs, Quick-Tune's outcome shares its "
        "curves with the signal, and the t = 0 results were known before._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.28")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.28")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
