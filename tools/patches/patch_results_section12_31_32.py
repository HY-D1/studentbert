from __future__ import annotations

import hashlib
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "c0bac2881e8b65049df2620f9e547887"
PINNED = {
    Path("scripts/task4_early_ft.py"): "9c87a401c955fc90aca772e5c17cbf08",
    Path("tools/compare_ft_epochs.py"): "16fbc8a6b27de3bec529729e1dec032e",
    Path("docs/nlp_budget_extension_design.md"): "d82d64f47e3d23f35a94316446a0041b",
    Path("docs/qt_hscore_design.md"): "9131a8e9221faf387cb4ff3c9d87fcf1",
    Path("docs/trajectory_amendment2.md"): "8859f77870d589845fc5d373a6cae773",
    Path("docs/selective_trust_design.md"): "f63bfdc6bc0df97a97fb2fc97ef0eddd",
    Path("analysis/regret_compute.py"): "fd4288700ffd9af74fe92911371935ae",
    Path("analysis/noise_floor.py"): "fcc5c16b485fc061a04ff43599851224",
    Path("docs/regret_compute_design.md"): "4dc6393deea651bf59aa8144e1ce1608",
    Path("docs/option1_nlp_design.md"): "198ce6a6ecede9a26db35adf0cce6059",
}
HEADS = ("### 12.31 Corrections and added limitations for 12.24 to 12.30 (audit of 2026-11-25)",
         '### 12.32 Correction to 12.20: what the label "(prospective)" covers (audit of '
         '2026-11-25)')
ANCHOR = "\n\n## 13. Cross-domain check"
QUOTES = [
    "| 42, 1, 2 | 210 | 44 | 0.4730 | 25.33 |",
    "| 3, 4, 5 | 210 | 40 | 0.4475 | 25.38 |",
    "| 6, 7, 8 | 210 | 42 | 0.2094 | 25.31 |",
    "the largest difference is 0.4730",
    ("epochs 1 to 3 differ from the recorded runs in most runs, by up to the largest difference "
     "in the table above (GPU nondeterminism)"),
    ("Design: docs/qt_hscore_design.md and docs/trajectory_amendment2.md, both committed before "
     "any feature or score existed."),
    "after 3 to 5 epochs the early leader halves H-score's regret (0.0006 against 0.0012)",
    ("A longer budget does not help: the early leader's regret rises to 0.0021 after 10 epochs "
     "and 0.0017 after 20, worse than H-score alone"),
    "Only StudentBERT meets the pre-specified targets",
    "at 2 epochs with t = -0.10 it trusts 17/21 at 0.099 [0.012, 0.217]",
    "the point sits at the target and its interval spans it",
    "StudentBERT reaches the target at 2 epochs",
    "held-out StudentBERT 0.171 at 5 epochs",
    "On StudentBERT every method gets worse beyond about 5 epochs",
    ("static selection is always the cheapest point and never good enough once a little "
     "fine-tuning is allowed"),
    "| early leader | epoch 5 | 0.196 | 21/21 | 0.0006 [0.0000, 0.0016] | 18 | yes |",
    "| early leader | epoch 10 | 0.392 | 21/21 | 0.0021 [0.0004, 0.0041] | 11 |  |",
    "| override rule | epoch 5 | 0.204 | 21/21 | 0.0005 [0.0000, 0.0015] | 19 | yes |",
    "| override rule | epoch 10 | 0.400 | 21/21 | 0.0017 [0.0003, 0.0034] | 13 |  |",
    "| warning, abstain | epoch 5 | 0.204 | 10/21 | 0.0010 [0.0000, 0.0026] | 9 |  |",
    "| warning, abstain | epoch 10 | 0.400 | 5/21 | 0.0007 [0.0000, 0.0017] | 3 |  |",
    "| successive halving | start 2 | 0.157 | 21/21 | 0.0007 [0.0001, 0.0015] | 15 |  |",
    "| successive halving | start 5 | 0.394 | 21/21 | 0.0013 [0.0004, 0.0024] | 9 |  |",
    "| successive halving (factor 3) | start 2 | 0.138 | 21/21 | 0.0014 [0.0002, 0.0034] | 11 |  |",
    "| successive halving (factor 3) | start 5 | 0.344 | 21/21 | 0.0009 [0.0004, 0.0014] | 11 |  |",
    ('| static H-score |  | 21/21 | 0.008 | 0.0012 | 0.0003 | 0.0010 [0.0000, 0.0020] | '
     '0.131 [0.003, 0.269] | above the floor | n/a |'),
    ('| early leader | epoch 1 | 21/21 | 0.039 | 0.0031 | 0.0003 | 0.0029 [0.0004, 0.0068] '
     '| 0.383 [0.050, 0.908] | above the floor | 0.0019 [0.0000, 0.0049] |'),
    ('| early leader | epoch 3 | 21/21 | 0.118 | 0.0006 | 0.0003 | 0.0004 [-0.0004, '
     '0.0015] | 0.048 [-0.058, 0.200] | not distinguishable from the floor | -0.0006 '
     '[-0.0016, 0.0001] |'),
    ('| early leader | epoch 5 | 21/21 | 0.196 | 0.0006 | 0.0003 | 0.0003 [-0.0005, '
     '0.0015] | 0.046 [-0.063, 0.198] | not distinguishable from the floor | -0.0006 '
     '[-0.0016, 0.0001] |'),
    ('| early leader | epoch 10 | 21/21 | 0.392 | 0.0021 | 0.0003 | 0.0019 [0.0000, '
     '0.0039] | 0.249 [0.002, 0.522] | above the floor | 0.0009 [-0.0017, 0.0037] |'),
    ('| early leader | epoch 20 | 21/21 | 0.784 | 0.0017 | 0.0003 | 0.0015 [0.0003, '
     '0.0032] | 0.197 [0.045, 0.427] | above the floor | 0.0005 [-0.0014, 0.0027] |'),
    ('| override rule | epoch 5 | 21/21 | 0.204 | 0.0005 | 0.0003 | 0.0003 [-0.0005, '
     '0.0014] | 0.034 [-0.070, 0.189] | not distinguishable from the floor | -0.0007 '
     '[-0.0016, -0.0000] |'),
    ('| successive halving | start 1 | 21/21 | 0.078 | 0.0007 | 0.0003 | 0.0004 [-0.0002, '
     '0.0011] | 0.058 [-0.022, 0.144] | not distinguishable from the floor | -0.0006 '
     '[-0.0011, -0.0001] |'),
    ('| successive halving (factor 3) | start 1 | 21/21 | 0.068 | 0.0008 | 0.0003 | 0.0006 '
     '[-0.0004, 0.0017] | 0.075 [-0.052, 0.228] | not distinguishable from the floor | '
     '-0.0004 [-0.0009, -0.0000] |'),
    ('| full fine-tuning | best validation | 21/21 | 0.784 | 0.0003 | 0.0003 | 0.0000 '
     '[0.0000, 0.0000] | 0.000 [0.000, 0.000] | not distinguishable from the floor | '
     '-0.0010 [-0.0020, -0.0000] |'),
    ('| static H-score |  | 90/90 | 0.029 | 2.59 | 0.93 | 1.65 [0.16, 3.21] | 0.492 '
     '[0.047, 0.953] | above the floor | n/a |'),
    ('| early leader | epoch 3 | 90/90 | 0.254 | 1.00 | 0.93 | 0.06 [-0.13, 0.21] | 0.018 '
     '[-0.039, 0.062] | not distinguishable from the floor | -1.59 [-3.21, -0.17] |'),
    ('| early leader | epoch 5 | 90/90 | 0.423 | 1.00 | 0.93 | 0.06 [-0.19, 0.34] | 0.019 '
     '[-0.057, 0.102] | not distinguishable from the floor | -1.59 [-3.09, -0.26] |'),
    ('| early leader | epoch 10 | 90/90 | 0.845 | 1.02 | 0.93 | 0.08 [-0.15, 0.33] | 0.024 '
     '[-0.044, 0.098] | not distinguishable from the floor | -1.57 [-3.02, -0.21] |'),
    ('| override rule | epoch 5 | 90/90 | 0.452 | 1.42 | 0.93 | 0.48 [0.01, 1.17] | 0.143 '
     '[0.002, 0.348] | above the floor | -1.17 [-2.81, -0.02] |'),
    ('| successive halving | start 3 | 90/90 | 0.498 | 1.10 | 0.93 | 0.16 [0.04, 0.29] | '
     '0.049 [0.011, 0.087] | above the floor | -1.49 [-3.05, -0.05] |'),
    ('| successive halving (factor 3) | start 3 | 90/90 | 0.472 | 0.95 | 0.93 | 0.02 '
     '[-0.31, 0.30] | 0.005 [-0.092, 0.091] | not distinguishable from the floor | -1.64 '
     '[-3.25, -0.26] |'),
    ('| early leader | epoch 1 | 90/90 | 0.094 | 33.58 | 23.62 | 9.96 [-0.27, 20.44] | '
     '0.228 [-0.006, 0.467] | not distinguishable from the floor | -6.98 [-20.84, 7.47] |'),
    ('| override rule | epoch 1 | 90/90 | 0.094 | 32.44 | 23.62 | 8.83 [-1.15, 19.27] | '
     '0.202 [-0.026, 0.440] | not distinguishable from the floor | -8.12 [-21.72, 6.17] |'),
    "Complete runs per included pair: median 10, range 3 to 23.",
    "and 90 on Quick-Tune micro (models with at least 3 complete runs",
    "seed 1 0.0004, seed 2 0.0009, seed 3 0.0001, seed 4 0.0001, seed 5 0.0004, seed 42 0.0003",
    "seed 1 0.0002, seed 2 0.0002, seed 3 0.0000, seed 4 0.0001, seed 5 0.0000, seed 42 0.0001",
    ("The static choice's excess over the floor (0.0010 AUC, 1.65 points) is larger than the "
     "floor itself (0.0003, 0.93)"),
    "docs/regret_compute_design.md was committed before the script existed.",
    ("the frontier are new, and the script was test-run on the StudentBERT scores with stand-in "
     "costs while it was written"),
    "| NLP, seeds 42, 1, 2 (prospective) | epoch 3 | 17/21 (0.81) | 17/20 (0.85) |",
    ("where the original rule's 18 of 20 becomes 17 of 20 under the replications' tie "
     "handling"),
    ("The analysis was defined and run after every warning result in 12.15 to 12.19 had been "
     "seen"),
    "The exception is the prospective set after two epochs",
    ("committed in 359b78d after the check and after the full set had been submitted, before any "
     "output of the full set was read"),
    "The rule was fixed on 2026-10-13 (analysis/fixed_rule_eval.py), before any NLP early fine-tune",
]
LINES = [
    (Path("scripts/task4_early_ft.py"), 61, 'if not 0.0 <= e["val_acc"] <= 1.0'),
    (Path("scripts/task4_early_ft.py"), 174, '"val_acc": correct / len(va)'),
    (Path("tools/compare_ft_epochs.py"), 35, 'd = [abs(a["val_acc"] - b["val_acc"])'),
    (Path("analysis/regret_compute.py"), 267, "curves[(code, m)] = [st.median(top1[t][rid][e]"),
    (Path("analysis/regret_compute.py"), 269, "per = [st.median(tr_t[t][rid][e] + ev_t[t][rid][e]"),
    (Path("analysis/regret_compute.py"), 271, "costs[(code, m)] = [sum(per[:e + 1])"),
    (Path("analysis/noise_floor.py"), 80, "def full_floor(ds: list[rc.Decision])"),
    (Path("analysis/noise_floor.py"), 81, "rc.full_best_val(d)[0]"),
    (Path("analysis/noise_floor.py"), 122, "st.mean(gold[t][c].values()) - d.outcome[c]"),
    (Path("analysis/noise_floor.py"), 123, "6-seed mean differs from the gold"),
]
SOURCES = [
    (Path("docs/nlp_budget_extension_design.md"),
     "Epochs 1 to 3 may differ slightly from the recorded runs (GPU nondeterminism)"),
    (Path("docs/trajectory_amendment2.md"),
     "Follows `docs/qt_hscore_design.md`, whose features now exist for all 30 micro tasks and 24 "
     "models (job 11805425; 720 files)."),
    (Path("docs/trajectory_amendment2.md"),
     "It is committed with `analysis/qt_score.py` and `analysis/qt_reversal.py` before either "
     "runs on the real features."),
    (Path("docs/selective_trust_design.md"),
     "on the two other domains, pooled, choose the threshold with the highest coverage whose "
     "normalized selective regret is at most 0.10; apply it to the held-out domain"),
    (Path("docs/selective_trust_design.md"),
     "4. The domain-specific budget: per domain, the smallest budget at which some threshold "
     "reaches a normalized selective regret of at most 0.10 with coverage of at least 0.50"),
    (Path("docs/option1_nlp_design.md"),
     "2. The fixed rule, fixed on 2026-10-13 before these data existed: follow H-score unless its "
     "pick is not the validation leader among the 7 encoders at epoch 3 of the fine-tune with the "
     "estimator's seed; otherwise abstain."),
    (Path("docs/option1_nlp_design.md"),
     "4. Reference rows: the best-average rule of Task 4 and the frozen margin to the runner-up."),
]


def flat(text: str) -> str:
    return " ".join(text.split())


def pinned(p: Path) -> None:
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != PINNED[p]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[p]}; nothing written")


def check(src: str) -> None:
    for p in PINNED:
        pinned(p)
    body = flat(src)
    bad = [q for q in QUOTES if body.count(flat(q)) != 1]
    if bad:
        sys.exit("ABORT, quoted RESULTS.md text not found exactly once; nothing written:\n  "
                 + "\n  ".join(bad))
    for p, n, s in LINES:
        lines = p.read_text(encoding="utf-8").splitlines()
        if n > len(lines) or s not in lines[n - 1]:
            sys.exit(f"ABORT: {p} line {n} does not hold {s!r}; nothing written")
    for p, s in SOURCES:
        if flat(s) not in flat(p.read_text(encoding="utf-8")):
            sys.exit(f"ABORT: {p} does not hold {s!r}; nothing written")


def wrap(text: str, first: str = "", rest: str = "") -> list[str]:
    return textwrap.wrap(text, width=100, initial_indent=first, subsequent_indent=rest,
                         break_long_words=False, break_on_hyphens=False)


def item(n: int, text: str) -> list[str]:
    return wrap(text, f"{n}. ", "   ") + [""]


def files(paths: list[Path]) -> str:
    return "Files (md5): " + ", ".join(f"{p} {PINNED[p]}" for p in paths) + "."


def section_31() -> list[str]:
    L = [HEADS[0], ""]
    L += wrap(
        "Task: an audit of the record (Harry, 2026-11-25) checked the readings of 12.24 to 12.30 "
        "against their tables, design files, code and commit history, and found eight places "
        "where a reading says more than the evidence or leaves out a limitation. This section "
        "corrects them. It changes no number and runs no analysis (descriptive). The earlier "
        "sections stay as recorded, because the record is append-only and the split files list "
        "each section's md5; read each of them with the item below that names it. Every number "
        "below is printed in the section it cites, except the accuracy points of item 1, which "
        "convert a printed fraction.") + [""]
    L += wrap(files([p for p in PINNED if p != Path("docs/option1_nlp_design.md")])) + [""]
    L += item(1, (
        '12.29, the reruns. The table "The new runs against the recorded 3-epoch runs" gives the '
        "largest validation-accuracy differences 0.4730, 0.4475 and 0.2094 without a unit, and "
        "the _Read_ repeats 0.4730. They are fractions of validation accuracy, that is 47.30, "
        "44.75 and 20.94 accuracy points (val_acc is correct / len(va), "
        "scripts/task4_early_ft.py line 174, checked to lie in [0, 1] at line 61; the difference "
        'is abs(a["val_acc"] - b["val_acc"]), tools/compare_ft_epochs.py line 35). The '
        'limitations name the cause "(GPU nondeterminism)". That is the cause the design '
        'expected ("Epochs 1 to 3 may differ slightly from the recorded runs (GPU '
        'nondeterminism)", docs/nlp_budget_extension_design.md), not a tested one: the check '
        "confirms only that each new run has the data and settings of its recorded run, and "
        "differences of up to 47.30 points are not slight. The decision of 12.29 to take every "
        "budget from the new runs is unaffected."))
    L += item(2, (
        '12.25, the design order. "Design: docs/qt_hscore_design.md and '
        'docs/trajectory_amendment2.md, both committed before any feature or score existed" '
        "holds for the design, committed in 404c1f7 (2026-11-10 17:06) as fixed before "
        "extraction, together with the feature extraction script. It does not hold for "
        "amendment 2, committed in c930810 (2026-11-10 20:38), which opens: \"Follows "
        "`docs/qt_hscore_design.md`, whose features now exist for all 30 micro tasks and 24 "
        'models (job 11805425; 720 files)." Amendment 2 was committed after the features existed '
        "and before any score: it is committed with analysis/qt_score.py and "
        "analysis/qt_reversal.py before either runs on the real features, as it states."))
    L += item(3, (
        '12.24, the early leader on StudentBERT. "after 3 to 5 epochs the early leader halves '
        "H-score's regret (0.0006 against 0.0012)\" and \"A longer budget does not help: the "
        "early leader's regret rises to 0.0021 after 10 epochs and 0.0017 after 20, worse than "
        'H-score alone" compare point estimates. The paired differences from the static choice '
        "in 12.30 span zero at each of these budgets: -0.0006 [-0.0016, 0.0001] at 3 and at 5 "
        "epochs, 0.0009 [-0.0017, 0.0037] at 10 and 0.0005 [-0.0014, 0.0027] at 20. Neither the "
        "halving nor the later worsening is shown."))
    L += item(4, (
        "12.28 and 12.29, the StudentBERT operating point. The budget and threshold at which "
        '"Only StudentBERT meets the pre-specified targets" (2 epochs, t = -0.10: 17/21 at '
        "0.099 [0.012, 0.217]) are chosen on the same 21 decisions on which they are scored "
        "(docs/selective_trust_design.md, Reported, item 4: the smallest budget at which some "
        "threshold reaches the targets), so this point is in-sample and optimistic. The "
        "design's out-of-sample test is the leave-one-domain-out transfer (Reported, item 3): "
        "there a "
        "threshold chosen on the two other domains gives held-out StudentBERT 0.171 at 5 epochs "
        '(12.29). 12.28 adds that "the point sits at the target and its interval spans it"; '
        'the phrase "StudentBERT reaches the target at 2 epochs" in 12.29 carries the same '
        "caveat."))
    L += item(5, (
        '12.27, the _Read_. (a) "On StudentBERT every method gets worse beyond about 5 epochs": '
        "in the 12.27 table the early leader (0.0006 at 5 epochs, 0.0021 at 10), the override "
        "rule (0.0005, 0.0017) and successive halving (0.0007 started at 2, 0.0013 started at 5) "
        "are higher later, but warning, abstain is not (0.0010 at 5, 0.0007 at 10), nor is "
        "successive halving with factor 3 (0.0014 started at 2, 0.0009 started at 5); all of "
        'these are point estimates (item 3). (b) "static selection is always the cheapest point '
        'and never good enough once a little fine-tuning is allowed": by the paired differences '
        "from the static choice in 12.30, on NLP the early leader at 3, 5 and 10 epochs (-1.59 "
        "[-3.21, -0.17], -1.59 [-3.09, -0.26], -1.57 [-3.02, -0.21]), the override rule at 5 "
        "epochs (-1.17 [-2.81, -0.02]) and successive halving started at 3 (-1.49 [-3.05, "
        "-0.05]; with factor 3, -1.64 [-3.25, -0.26]) have lower regret than static, and every "
        "other NLP row, including all at 1 and 2 epochs, has an interval that includes zero. On "
        "StudentBERT only successive halving started at 1 epoch is clearly lower (-0.0006 "
        "[-0.0011, -0.0001]); three intervals end at a printed -0.0000 (successive halving with "
        "factor 3 started at 1, the override rule at 5 epochs, full fine-tuning), and every "
        "early-leader interval includes or touches zero. On Quick-Tune the early leader and the "
        "override rule are lower from 2 epochs on but not at 1 epoch (-6.98 [-20.84, 7.47] and "
        "-8.12 [-21.72, 6.17]), and there the methods read the same runs that define the "
        "outcome (12.27, 12.30)."))
    L += item(6, (
        "Quick-Tune compute in 12.27 to 12.30. Each model's curve is the median over its "
        "complete runs (at least 3; median 10 per included pair, 12.26), while its compute is "
        "the median of one run's time per epoch (analysis/regret_compute.py lines 267 to 271). "
        "The Quick-Tune methods are therefore charged for one run per model while their curves "
        "summarize several, which understates their compute relative to the information they "
        "use. No number changes; this is a limitation of the Quick-Tune fine-tuning compute in "
        "12.27 to 12.30."))
    L += item(7, (
        "12.30, the StudentBERT floor. The floor chooses by one seed's best validation score "
        "(analysis/noise_floor.py lines 80 to 81), and that seed's test AUC is one of the six "
        "averaged into the gold (lines 122 to 123 check each decision's outcome against the "
        "6-seed mean; 12.30 lists the same six seeds, 1, 2, 3, 4, 5 and 42, for the spread and "
        "for the seed oracle). The choosing seed thus also enters the outcome, which biases the "
        "StudentBERT floor downward (optimistic); the NLP floor is measured against the "
        "published outcome, which our runs do not enter. Separately, \"The static choice's "
        "excess over the floor (0.0010 AUC, 1.65 points) is larger than the floor itself "
        '(0.0003, 0.93)" compares point estimates: the excess intervals [0.0000, 0.0020] and '
        "[0.16, 3.21] contain 0.0003 and 0.93."))
    L += item(8, (
        '12.27, the design order. "docs/regret_compute_design.md was committed before the '
        'script existed" should read "before the script was committed": the design is commit '
        "ef198fe (2026-11-16 18:16:56) and the script, 441 lines with 67 lines of tests, commit "
        "935ce90 seven minutes later (18:24:04). 12.27 already states that \"the script was "
        'test-run on the StudentBERT scores with stand-in costs while it was written".'))
    L += wrap(
        "_Read: no number in 12.24 to 12.30 changes. Items 3, 5 and 7 narrow three readings to "
        "what the intervals show: on StudentBERT no early-leader difference from the static "
        "choice excludes zero as printed, longer budgets are not shown to hurt, and the static "
        "choice's excess over the seed floor is not shown to exceed the floor. Item 4 marks "
        "StudentBERT's operating point as in-sample; items 6 and 7 add two limitations "
        "(Quick-Tune compute is charged for one run, the StudentBERT floor is optimistic); "
        "items 1, 2 and 8 correct statements of fact. What stands: on NLP the early leader at "
        "3, 5 and 10 epochs, the override rule at 5 and successive halving at its largest "
        "setting have lower regret than the static choice with intervals that exclude zero, "
        "and on StudentBERT successive halving started at 1 epoch does._")
    return L


def section_32() -> list[str]:
    L = [HEADS[1], ""]
    L += wrap(
        "Task: the audit of 12.31 also found that the tables of 12.20 label the seed set 42, 1, "
        '2 "(prospective)" in all of its rows, while only part of what they report was '
        "fixed before those seeds' fine-tunes existed. Descriptive; no number changes. The rule "
        'fixed in advance is the epoch-3 rule: "follow H-score unless its pick is not the '
        "validation leader among the 7 encoders at epoch 3 of the fine-tune with the "
        "estimator's seed; otherwise abstain\", fixed on 2026-10-13 before any NLP early "
        "fine-tune existed (docs/option1_nlp_design.md, Methods compared, item 2; committed in "
        "359b78d before any output of the full set was read, 12.15). The label holds for the "
        "epoch-3 flags on seeds 42, 1, 2, which 12.20 counts with the replications' tie "
        "handling (17 of 20 overtaken recommendations recalled, against 18 of 20 under the "
        "original rule; 12.20, Ties). The same flag at epochs 1 and 2 and the AUROC, AUPRC and "
        "matched-coverage read-outs, including those of the frozen margin (a reference row of "
        "the design, Methods compared, item 4), were not fixed in advance: 12.20 defined them "
        "after every warning "
        "result in 12.15 to 12.19 had been seen. For these the seed set 42, 1, 2 is post hoc "
        "like the other two, and so is the row that the _Read_ of 12.20 singles out (\"The "
        'exception is the prospective set after two epochs").') + [""]
    L += wrap(files([Path("docs/option1_nlp_design.md")])) + [""]
    L += wrap(
        '_Read: in 12.20, "(prospective)" names the seed set on which the epoch-3 rule was first '
        "tested; it describes that rule's flags at epoch 3, not the other budgets or the "
        "detector metrics. The 12.20 task paragraph already states that the analysis was "
        "defined and run after the results._")
    return L


def build(src: str) -> str:
    check(src)
    return "\n" + "\n".join(section_31() + section_32())


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    present = [h in src for h in HEADS]
    if all(present):
        print("skip (already applied): 12.31 and 12.32")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if any(present):
        sys.exit("ABORT: only one of 12.31 and 12.32 is present; nothing written")
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build(src) + ANCHOR, 1)
    print("ok: sections 12.31 and 12.32")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
