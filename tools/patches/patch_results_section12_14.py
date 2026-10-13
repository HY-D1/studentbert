from __future__ import annotations

"""RESULTS.md 12.14: reversal-aware pipeline, version 2 (the recommendation as the target).

Why this exists. Prof. Hazra asked for a version aligned with the final decision (pairs headed by
H-score's pick only, a recommendation-level risk, the all-pairs model as baseline, no layer group,
no threshold tuning on the held-out target, recommendation-level evaluation). analysis/
reversal_model_v2.py implements it; its primary comparison was fixed in the commit that added it
(ccd1655), before its first run (job 11042633). This records the result as declared, including the
reproduction of 12.13, and three diagnostics computed from the pinned outputs. Nothing was tuned
after the results were seen. Run from the code root as a short job. Harry.
"""

import csv
import hashlib
import re
import sys
import textwrap
from pathlib import Path

TARGET = Path("RESULTS.md")
PRE = "86365a2cf9a24e4dd704f9a84f0c719f"
BF = Path("benchmark_final")
PINNED = {
    Path("analysis/reversal_model_v2.py"): "47d206c63749bdc2eb0cee6fdda62d7b",
    BF / "reversal_model_v2_report.md": "f145e45fee795193ad885100af7295be",
    BF / "reversal_model_v2_recs.tsv": "1f8d84f8a598452c4ba74404526b2438",
    BF / "reversal_model_v2_pairs.tsv": "34fff18c059bab71fc1c6ce29ab3bd34",
}
LOG = Path("v2_run_11042633.log")
HEAD = "### 12.14 Reversal-aware signal pipeline, version 2"
ANCHOR = "\n\n## 13. Cross-domain check: NLP encoder selection"
PRIMARY = "head: frozen + early + history"


def pinned(p: Path) -> str:
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != PINNED[p]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[p]}; nothing written")
    return p.read_text()


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def row(report: str, section: str, model: str, risk: str, rule: str) -> tuple:
    part = report.split(section, 1)[1]
    m = re.search(rf"^\| {re.escape(model)} \| {risk} \| {rule} \| (\d+)/(\d+) \| (\d+)/(\d+) \| "
                  r"(\d+) \| (\d\.\d{4}) \| (\d\.\d{4}) \| (\d\.\d{4}) \|$", part, re.M)
    return m.groups() if m else ()


def build() -> str:
    fails: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    pinned(Path("analysis/reversal_model_v2.py"))
    report = pinned(BF / "reversal_model_v2_report.md")
    recs = list(csv.DictReader(pinned(BF / "reversal_model_v2_recs.tsv").splitlines(),
                               delimiter="\t"))
    pinned(BF / "reversal_model_v2_pairs.tsv")
    log = LOG.read_text() if LOG.is_file() else ""
    need("primary: followed 9, caught 0, false alarms 12, selective regret 0.0027" in log,
         f"{LOG} does not show the primary result")

    main, rob = "## Main label", "## Robustness label"
    rep = row(report, main, "all pairs: version 1 (12.13)", "max", "fixed 0.5")
    need(rep[:6] == ("16", "21", "1", "7", "4", "0.0015"), f"12.13 is not reproduced: {rep}")
    pri = row(report, main, PRIMARY, "noisy-or", "fold-chosen")
    need(pri[:7] == ("9", "21", "0", "7", "12", "0.0027", "0.0027"), f"primary row {pri}")
    need("Always following H-score: mean regret 0.0012" in report.split(rob)[0],
         "always-follow regret is not 0.0012")
    prob = row(report, rob, PRIMARY, "noisy-or", "fold-chosen")
    need(prob[:5] == ("18", "18", "0", "4", "0"), f"robust primary row {prob}")

    pr = [r for r in recs if r["label"] == "main" and r["model"] == PRIMARY]
    pos = [float(r["risk_noisy_or"]) for r in pr if r["reversed"] == "1"]
    neg = [float(r["risk_noisy_or"]) for r in pr
           if r["reversed"] == "0" and r["ambiguous"] == "0"]
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    need(len(pos) == 7 and len(neg) == 14 and auc < 0.5, f"recommendation AUC {auc}")
    events = {t: sum(r["reversed"] == "1" for r in pr if r["target"] == t)
              for t in ("assist2017", "algebra2006", "algebra2005")}
    need(events == {"assist2017": 3, "algebra2006": 3, "algebra2005": 1},
         f"reversed recommendations per target {events}")
    pen = re.search(r"Primary model, fold-chosen thresholds \(noisy-OR\): (.*?); "
                    r"penalties: (.*?)\.\n", report.split(rob)[0])
    taus = dict(x.split(" ") for x in pen.group(1).split(", ")) if pen else {}
    lams = dict(x.split(" ") for x in pen.group(2).split(", ")) if pen else {}
    n_top = sum(v == "100.0" for v in lams.values())
    need(n_top == 6 and len(lams) == 7, f"{n_top} of {len(lams)} penalties at the grid maximum")
    need(taus.get("assist2017") == "1.0000" and taus.get("algebra2006") == "1.0000",
         "the folds holding out the reversal targets do not choose 1.0")
    if fails:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(fails))

    L = [f"{HEAD}: the H-score recommendation as the target (analysis/reversal_model_v2.py, design "
         "frozen in commit ccd1655 before its first run, job 11042633)", ""]
    L += wrap(
        "Task: Prof. Hazra's feedback on 12.13. Rows are only the pairs headed by H-score's pick, "
        "7 "
        "alternatives per recommendation (147 rows; 123 stable, 13 near-tie, 11 reversal on the "
        "main label); the target is whether the alternative overtakes the pick after fine-tuning; "
        "the pairwise risks are combined by noisy-OR into a recommendation-level risk (the largest "
        "pairwise risk reported beside it); features are frozen, early fine-tuning and history, "
        "without the layer group (early fine-tuning alone as a variant); the all-pairs model of "
        "12.13, and the same model with version 2's features, are the baselines. Leave-one-target-"
        "out throughout; the penalty and the threshold are chosen inside each fold from the "
        "training targets' out-of-fold predictions (the threshold by Youden's J), and a fixed 0.5 "
        "is reported beside it. Primary, fixed before the run: the head-pair model with frozen, "
        "early and history features, noisy-OR risk and the fold-chosen threshold.")
    L += [""] + wrap(
        "Reproduction check: the 12.13 all-pairs model, scored by its own rule (largest pairwise "
        f"risk, threshold 0.5), follows {rep[0]} of {rep[1]} recommendations, catches {rep[2]} of "
        f"{rep[3]} reversals with {rep[4]} false alarms, selective regret {rep[5]}, as in 12.13.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{p.name} {m}" for p, m in PINNED.items()) + ".")
    L += ["", "Report as written by analysis/reversal_model_v2.py (headings lowered one level):",
          ""]
    for ln in report.splitlines():
        if ln.startswith("# "):
            continue
        L.append("#" * 2 + ln if ln.startswith("## ") else ln)
    while L[-1] == "":
        L.pop()
    L += [""] + wrap(
        f"_Read: aligning the target with the recommendation did not improve the decision. The "
        f"primary rule follows {pri[0]} of {pri[1]} recommendations, catches {pri[2]} of {pri[3]} "
        f"reversals and raises {pri[4]} false alarms; the recommendations it follows have mean "
        f"regret {pri[5]}, against 0.0012 for always following H-score, and its AURC ({pri[6]}) "
        "is above the 0.0012 of a random order. Its held-out risks run against the outcome: as a "
        f"detector of reversed recommendations their AUC is {auc:.2f} ({len(pos)} reversed, "
        f"{len(neg)} not). The penalty sits at the top of the grid in {n_top} of 7 folds, and the "
        "folds that hold out ASSISTments 2017 and Algebra 2006 choose a threshold of 1.0 (follow "
        "everything), so those reversals are missed by construction. On the robustness label the "
        f"primary rule follows all {prob[0]} recommendations, which is always following H-score. "
        "The reversed recommendations come from three events (ASSISTments 2017 on three seeds, "
        "Algebra 2006 on three, Algebra 2005 on one), so holding out a target removes its event "
        "entirely, and the remaining events do not carry a signal that transfers across targets. "
        "With 7 targets this is a sample-size limit as much as a modelling result. Nothing was "
        "tuned after the run._")
    return "\n" + "\n".join(L)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src:
        print("skip (already applied): 12.14")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if hashlib.md5(src.encode()).hexdigest() != PRE:
        sys.exit(f"ABORT: RESULTS.md is not {PRE}; nothing written")
    if src.count(ANCHOR) != 1:
        sys.exit("ABORT: the section 13 anchor does not match exactly once; nothing written")
    out = src.replace(ANCHOR, build() + ANCHOR, 1)
    print("ok: section 12.14")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
