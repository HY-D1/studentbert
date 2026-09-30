from __future__ import annotations

import csv
import hashlib
import json
import statistics as st
import sys
import textwrap
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# Why this exists: failure-review limitation 7.7 said evaluation-learner uncertainty was measured
# for one pair only (Algebra 2006, 12.11). Diagnostic D11 (2026-09-30) runs the same learner-level
# bootstrap (scripts/split_bootstrap_kt.py, jobs 10707875 and 10708445) on the other two pairs
# that carry H-score regret on a small target: ASSISTments 2017 (in-domain against EdNet) and
# Algebra 2005 (EdNet, H-score's wrong pick on one seed, against Junyi). It also splits H-score's
# 21-ranking mean regret by target and evidence status, from benchmark_final/signals_main.tsv
# (6 dp values as analysis/estimator_signals.py writes them, printed as written; each share is
# that value over 7, rounded once, half away from zero). Every number is read from the pinned
# files at run time, the prose states only what CHECKS verify on them, and a different or edited
# file aborts before anything is written. Harry, 2026-09-30.

TARGET = Path("RESULTS.md")
BF = Path("benchmark_final")
PINNED = {
    "split_bootstrap_assist2017_summary.tsv": "854f586525daaa32b90af656c357912b",
    "split_bootstrap_assist2017_checks.json": "6fb893cae0c03e73d3b51664e0cde348",
    "split_bootstrap_algebra2005_summary.tsv": "5570b3f70342d562607f4995f0a99b3c",
    "split_bootstrap_algebra2005_checks.json": "761628948e0c744f4e65fa32b4c1e401",
    "signals_main.tsv": "a3f4c0d80490904618a69ede169f5858",
}
ANCHOR = "\n\n## 13. Cross-domain check"
HEAD = "### 12.12 Learner-level bootstrap of the other two regret-bearing pairs"
MARGIN = Decimal("0.001")
TOL = 5e-5
EST = "hscore_kt_causal"

EDITS = [
    ("Junyi, the best, and wrong for all three estimators on ASSISTments 2017 (EdNet best) and Algebra\n"
     "2006 (Junyi best). Ruled out as causes by the LogME diagnostics (scripts/diagnose_logme.py, three\n",
     "Junyi, the best, and wrong for all three estimators on ASSISTments 2017 (EdNet best, also across\n"
     "evaluation learners, 12.12) and Algebra 2006 (Junyi best). Ruled out as causes by the LogME\n"
     "diagnostics (scripts/diagnose_logme.py, three\n",
     "12.4 points to the ASSISTments 2017 learner check"),
]

NAMES = {"assist2017": "ASSISTments 2017", "algebra2005": "Algebra 2005",
         "algebra2006": "Algebra 2006", "bridge2006": "Bridge 2006",
         "assist2009": "ASSISTments 2009", "ednet": "EdNet", "junyi": "Junyi"}


def q(x, places: str = "0.0001", sign: bool = False) -> str:
    d = Decimal(str(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def raw(name: str) -> bytes:
    p = BF / name
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    b = p.read_bytes()
    got = hashlib.md5(b).hexdigest()
    if got != PINNED[name]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[name]}; nothing written")
    return b


def tsv(name: str) -> list[dict]:
    return list(csv.DictReader(raw(name).decode().splitlines(), delimiter="\t"))


def gold_gap(checks: list[dict], a_tag: str, b_tag: str) -> float:
    """Mean recorded test AUC of b minus a: the benchmark gold for the pair."""
    a = [c["recorded"] for c in checks if a_tag in c["run"]]
    b = [c["recorded"] for c in checks if b_tag in c["run"]]
    if len(a) != 6 or len(b) != 6:
        sys.exit(f"ABORT: expected 6 and 6 checkpoints for {b_tag} and {a_tag}; nothing written")
    return st.mean(b) - st.mean(a)


def build(results: str) -> str:
    fails: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    S = {t: {r["split"]: r for r in tsv(f"split_bootstrap_{t}_summary.tsv")}
         for t in ("assist2017", "algebra2005")}
    C = {t: json.loads(raw(f"split_bootstrap_{t}_checks.json")) for t in ("assist2017", "algebra2005")}
    sig = {r["target"]: r for r in tsv("signals_main.tsv") if r["estimator"] == EST}

    worst = max(c["abs_diff"] for t in C for c in C[t])
    need(all(len(C[t]) == 12 for t in C) and worst < TOL,
         "not every checkpoint reproduces its recorded test AUC")
    for t in S:
        for sp in ("val", "test"):
            r = S[t][sp]
            need(Decimal(r["ci_low"]) > 0 and r["seeds_positive"] == "6/6",
                 f"{t} {sp}: the interval does not exclude zero on 6 of 6 seeds")
        g = S[t]["test minus val"]
        need(Decimal(g["ci_low"]) < 0 < Decimal(g["ci_high"]), f"{t}: the split gap excludes zero")
    need(Decimal(S["assist2017"]["test"]["b_minus_a"]) > MARGIN,
         "ASSISTments 2017 test gap is not above the 0.001 margin")

    for sp, metric in (("val", "best validation AUC"), ("test", "test AUC")):
        r = S["assist2017"][sp]
        need(f"| assist2017 | {metric} | {r['b_minus_a']} | {r['seeds_positive']} | "
             f"{r['per_seed']} |" in results,
             f"ASSISTments 2017 {sp} per-seed differences do not equal 12.8")

    need(len(sig) == 7, f"signals_main.tsv has {len(sig)} {EST} targets, expected 7")
    a17, a05 = sig["assist2017"], sig["algebra2005"]
    need(a17["modal_pick"] == "src:assist2017" and a17["pick_agreement"] == "3/3",
         "H-score does not pick in-domain on 3 of 3 seeds on ASSISTments 2017")
    need(abs(float(a17["mean_regret"]) - gold_gap(C["assist2017"], "_indomain_", "_fromednet_"))
         < 1e-6, "ASSISTments 2017 regret does not equal EdNet minus in-domain")
    need(a05["modal_pick"] == "src:junyi" and a05["pick_agreement"] == "2/3",
         "H-score does not pick Junyi on 2 of 3 seeds on Algebra 2005")
    need(abs(3 * float(a05["mean_regret"]) - gold_gap(C["algebra2005"], "_fromednet_", "_fromjunyi_"))
         < 2e-6, "the one wrong Algebra 2005 seed does not carry Junyi minus EdNet (not EdNet)")
    for t in ("assist2009", "ednet", "junyi"):
        need(Decimal(sig[t]["mean_regret"]) == 0, f"H-score regret on {t} is not zero")
    total = sum(Decimal(r["mean_regret"]) for r in sig.values()) / 7
    need(q(total) == "0.0012", f"H-score's 21-ranking mean regret is {q(total)}, not 0.0012")
    need("| hscore_kt_causal | 21 |" in results and "| 0.0012 / 0.0103 |" in results,
         "12.3 no longer shows H-score's 0.0012 / 0.0103")
    need("| test | 131 | 58800 | +0.0017 | [-0.0001, +0.0035] |" in results,
         "12.11 no longer shows the Algebra 2006 test interval")
    need("| bridge2006 | junyi +0.0084, bridge2006 (in-domain) +0.0079 |" in results,
         "12.1 no longer shows the Bridge 2006 in-domain encoder tied with the best")
    need("| algebra2005 | junyi +0.0369 | +0.0266, rank 7 of 7 |" in results,
         "12.1 no longer shows EdNet last on Algebra 2005")
    if fails:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(fails))

    share = {t: Decimal(r["mean_regret"]) / 7 for t, r in sig.items()}
    L = [f"{HEAD} (ASSISTments 2017 in-domain against EdNet, Algebra 2005 EdNet against Junyi, 6 "
         "seeds; scripts/split_bootstrap_kt.py), and H-score's regret by evidence status", ""]
    L += wrap("The 12.11 design on the two other pairs where H-score is charged regret on a small "
              "target. On ASSISTments 2017, a is the in-domain encoder and b is EdNet; on Algebra "
              "2005, a is EdNet, H-score's pick on its one wrong seed, and b is Junyi. The 24 "
              "saved best checkpoints reproduce their recorded test AUC within "
              f"{worst:.1e}; 2,000 learner resamples (seed 0), the same for every checkpoint; the "
              "statistic is the mean over seeds of b minus a. Each checkpoint was selected on the "
              "validation learners, so the test interval is the clean one.")
    L += [""] + wrap("Files (md5): " + ", ".join(f"{n} {m}" for n, m in PINNED.items()) + ".")
    L += ["", "| target | b minus a | split | learners | positions | difference | 95% interval | "
          "P(>0) | seeds positive |", "|---|---|---|---|---|---|---|---|---|"]
    for t, lab in (("assist2017", "EdNet minus in-domain"), ("algebra2005", "Junyi minus EdNet")):
        for sp in ("val", "test", "test minus val"):
            r = S[t][sp]
            L.append(f"| {t} | {lab} | {sp} | {r['learners']} | {r['positions']} | "
                     f"{r['b_minus_a']} | [{r['ci_low']}, {r['ci_high']}] | {r['p_boot_gt0']} | "
                     f"{r['seeds_positive']} |")
    L += [""] + wrap(f"H-score's mean regret over the 21 rankings of 12.3 ({q(total)}) by target: "
                     "mean regret over its 3 seeds as benchmark_final/signals_main.tsv writes it, "
                     "and that value over 7.")
    status = {
        "assist2017": "robust: across fine-tuning seeds (12.1) and evaluation learners on both "
                      "splits (this section)",
        "algebra2005": "gold robust across learners (this section); the miss is one estimator "
                       "seed of 3",
        "algebra2006": "not separable across evaluation learners (12.11)",
        "bridge2006": "the pick is tied with the best (12.1)",
    }
    L += ["", "| target | H-score pick, seeds | mean regret | share of the 21-ranking mean | "
          "evidence |", "|---|---|---|---|---|"]
    for t in ("assist2017", "algebra2005", "algebra2006", "bridge2006", "assist2009", "ednet",
              "junyi"):
        r = sig[t]
        L.append(f"| {t} | {r['modal_pick'].removeprefix('src:')}, {r['pick_agreement']} | "
                 f"{r['mean_regret']} | {q(share[t])} | {status.get(t, 'no regret')} |")
    L.append(f"| all 7 | | | {q(total)} | |")
    v, s = S["assist2017"]["val"], S["assist2017"]["test"]
    w, u = S["algebra2005"]["val"], S["algebra2005"]["test"]
    L += [""] + wrap(
        f"_Read: on ASSISTments 2017 EdNet beats the in-domain encoder across evaluation learners "
        f"on both splits, {v['b_minus_a']} [{v['ci_low']}, {v['ci_high']}] on validation and "
        f"{s['b_minus_a']} [{s['ci_low']}, {s['ci_high']}] on test, 6 of 6 seeds each, and the "
        "test gap is above the 0.001 margin, so the two are not top-equivalent at learner level "
        "either; the re-scored differences equal 12.8 seed by seed. On Algebra 2005 Junyi beats "
        f"EdNet on both splits, {w['b_minus_a']} [{w['ci_low']}, {w['ci_high']}] and "
        f"{u['b_minus_a']} [{u['ci_low']}, {u['ci_high']}], so H-score's one wrong seed there is an "
        "estimator "
        f"error (EdNet, last of 7 there in 12.1), not gold noise. Of H-score's {q(total)}, {q(share['assist2017'])} comes from "
        f"ASSISTments 2017, {q(share['algebra2005'])} from the Algebra 2005 seed, "
        f"{q(share['algebra2006'])} from Algebra 2006, which the evaluation learners do not "
        f"separate, and {q(share['bridge2006'])} from the Bridge 2006 tie._")
    return "\n" + "\n".join(L)


def apply(text: str, old: str, new: str, why: str) -> str:
    if new in text and (old not in text or old in new):
        print(f"skip (already applied): {why}")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1; nothing written")
    print(f"ok: {why}")
    return text.replace(old, new, 1)


def main() -> None:
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src and all(old not in src for old, _n, _w in EDITS):
        print("skip (already applied): 12.4 edit and 12.12")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if HEAD in src or "### 12.12" in src:
        sys.exit("ABORT: a 12.12 heading exists but not as this patcher writes it; nothing written")
    section = build(src)
    out = src
    for old, new, why in EDITS:
        out = apply(out, old, new, why)
    out = apply(out, ANCHOR, section + ANCHOR, "section 12.12")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
