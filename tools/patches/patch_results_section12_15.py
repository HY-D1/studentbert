from __future__ import annotations

"""RESULTS.md 12.15: Option 1, the recommendation-level test on the NLP benchmark.

Why this exists. Prof. Hazra approved the NLP early fine-tunes and asked for H-score, the fixed
epoch-3 rule and the learned head-pair model to be compared with held-out-task evaluation, with
reversal detection, false alarms, regret and selective regret, calibration, coverage, compute cost
and epochs 1 to 3. analysis/nlp_option1_eval.py produced the report (job 11261879). This records it
with the design's timeline and the facts behind the reading, each checked against the pinned files.
Nothing was tuned after the results were seen. Run from the code root as a short job. Harry.
"""

import csv
import hashlib
import re
import statistics as st
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "5842bb467bdc0c208d21f6fa882357f2"
BF = Path("benchmark_final")
PINNED = {
    Path("docs/option1_nlp_design.md"): "198ce6a6ecede9a26db35adf0cce6059",
    Path("analysis/nlp_option1_eval.py"): "76f268952cc363be25271e4df738b5d7",
    BF / "nlp_head_pairs.tsv": "d023dd9346eb1020afa11d1dacbaf579",
    BF / "nlp_head_recs.tsv": "e533ea40073f9ff4926af3194570008c",
    BF / "nlp_option1_report.md": "011c052c9166199daa417efe0a6cf173",
    BF / "nlp_option1_recs.tsv": "8a146de30d1cc103d8df324f983d99cb",
}
LOG = Path("opt1_eval_11261879.log")
HEAD = "### 12.15 Option 1: the recommendation-level test on the NLP benchmark"
ANCHOR = "\n\n## 13. Cross-domain check: NLP encoder selection"


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
    fails: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    design = pinned(Path("docs/option1_nlp_design.md"))
    need(hashlib.md5(design.encode()[:4266]).hexdigest() == "2454fcd5ac5f7beb5e92d5d20194da30",
         "the design's first 4,266 bytes are not the version delivered before the check")
    pinned(Path("analysis/nlp_option1_eval.py"))
    pinned(BF / "nlp_head_pairs.tsv")
    pinned(BF / "nlp_head_recs.tsv")
    report = pinned(BF / "nlp_option1_report.md")
    recs = list(csv.DictReader(pinned(BF / "nlp_option1_recs.tsv").splitlines(), delimiter="\t"))
    log = LOG.read_text() if LOG.is_file() else ""
    need("epoch-3 rule: followed 8/30, caught 18, false alarms 4, selective regret 0.5375 "
         "(always 2.6967)" in log, f"{LOG} does not show the epoch-3 result")
    need("210 runs, all present, 3 epochs, V100-SXM2, identical training and validation rows"
         in report and "7.6 GPU-hours" in report, "the report's data check line moved")
    need("mean regret 1.3670 points over the 10 settings (reproduces RESULTS.md 13.1)" in report,
         "the best-average reproduction line moved")

    def get(k: str, method: str) -> list[dict]:
        return [r for r in recs if r["epochs"] == k and r["method"] == method]

    r3 = get("3", "fixed rule")
    rev = [r for r in r3 if r["reversed"] == "1"]
    ok = [r for r in r3 if r["reversed"] == "0"]
    flag = lambda r: float(r["risk_noisy_or"]) > 0.5  # noqa: E731
    fa = sorted((r["target"], r["est_seed"]) for r in ok if flag(r))
    miss = sorted((r["target"], r["est_seed"]) for r in rev if not flag(r))
    need(len(rev) == 20 and len(ok) == 10, "not 20 reversed and 10 kept")
    need(fa == [("airline", "cls:1"), ("rte", "cls:1"), ("rte", "cls:2"), ("rte", "cls:42")],
         f"false alarms {fa}")
    need(miss == [("agnews", "mean:1"), ("agnews", "mean:42")], f"missed {miss}")
    miss_reg = {round(float(r["regret"]), 2) for r in rev if not flag(r)}
    need(miss_reg == {2.15}, f"missed regrets {miss_reg}")
    l3 = get("3", "learned head-pair")
    lab = sorted({r["target"] for r in l3 if float(r["risk_noisy_or"]) > float(r["tau_fold"])})
    need(lab == ["rte"], f"the learned model abstains on {lab}")
    same = all([(r["target"], r["est_seed"], float(r["risk_noisy_or"]) > float(r["tau_fold"]))
                for r in get(k, "learned head-pair")] ==
               [(r["target"], r["est_seed"], float(r["risk_noisy_or"]) > float(r["tau_fold"]))
                for r in l3] for k in ("1", "2"))
    need(same, "the learned model's decisions differ across budgets")
    row = {}
    for k in ("1", "2", "3"):
        m = re.search(rf"^\| {k} \| (\d+\.\d) \| (\d+)/30 \| (\d+)/20 \| (\d+) \| "
                      rf"(\d\.\d{{4}}) \|", report, re.M)
        row[k] = m.groups() if m else ()
    need(row["1"] == ("5.1", "7", "19", "4", "0.3071")
         and row["2"][:4] == ("10.2", "5", "18", "7")
         and row["3"] == ("15.2", "8", "18", "4", "0.5375"), f"budget rows {row}")
    low = re.search(r"^\| 0\.00 to 0\.25 \| (\d+) \| (0\.\d{4}) \| (1\.0000) \|$", report, re.M)
    need(low is not None and low.group(1) == "5" and low.group(2).startswith("0.04"),
         "the low-risk calibration bin moved")
    if fails:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(fails))

    L = [f"{HEAD} (analysis/nlp_option1_eval.py, job 11261879; design "
         "docs/option1_nlp_design.md)", ""]
    L += wrap(
        "Task: Prof. Hazra approved 210 early fine-tunes (7 encoders x 5 tasks x 2 poolings x 3 "
        "estimator seeds, 3 epochs each on the estimators' own 10,000-example samples, "
        "scripts/task4_early_ft.py), starting with a 5-run timing and protocol check, and asked "
        "for the fixed epoch-3 rule to be kept fixed for a test on new data. Labels are the head "
        "pairs "
        "of step A (published tuned gold, Task 4 tie rule): 30 recommendations, 20 reversed, "
        "from 4 of 5 tasks. Evaluation is leave-one-task-out, both poolings of a task held out "
        "together.")
    L += [""] + wrap(
        "Timeline of the design. The rule was fixed on 2026-10-13 (analysis/fixed_rule_eval.py), "
        "before any NLP early fine-tune existed. The design file was delivered before the 5-run "
        "check and committed in 359b78d after the check and after the full set had been submitted, "
        "before any output of the full set was read; its first 4,266 bytes are the delivered "
        "version and its last section records this. The check ran one encoder per task, so it "
        "could not show whether H-score's pick led at epoch 3.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/nlp_option1_eval.py (headings lowered one level):", ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("#" * 2 + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        "_Read: the fixed epoch-3 rule, set before these data existed, flags 18 of the 20 reversed "
        "recommendations and 4 of the 10 that hold (Airline once, RTE on all three first-token "
        "rankings); it follows 8 of 30, and those have mean regret 0.5375 points against 2.6967 "
        "for always following H-score. The two reversals it follows are AG News with mean pooling "
        "(seeds 1 and 42, regret 2.15 points each). Its coverage is low: 22 of 30 recommendations "
        "fall back to full fine-tuning. The learned head-pair model does not beat the rule: at its "
        "fold-chosen thresholds it abstains only on RTE, so it catches 3 of 20, and the same "
        "decisions come out at every budget; its calibration is poor (5 recommendations given a "
        "risk near 0.04 were all reversed). On budget, the rule at epoch 1 flags 19 of 20 with 4 "
        "false alarms at 5.1 GPU-minutes per recommendation, against 15.2 at epoch 3; epoch 2 is "
        "worse (18 of 20, 7 false alarms), so these counts do not order the budgets reliably. The "
        "cost of fine-tuning all 7 encoders in full was not measured here, so the saving is not "
        "quantified. With 5 tasks and 30 recommendations every estimate is coarse._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.15")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.15")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
