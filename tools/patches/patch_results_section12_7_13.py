from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import textwrap
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# Why this exists: the Task 3 additions of 2026-09-28 (confidence signals, the foreign-source
# view with the dataset-similarity baseline, the leave-one-target-out best-average rule) and the
# Task 4 cross-domain check had no RESULTS.md section, and two 12.x sentences were imprecise:
# 12.3 said hscore_shrunk equals hscore (same picks, but rho +0.5238 against +0.5221) and "3,000
# target learners" (the five smaller targets use their whole train split); 12.4 grouped
# ASSISTments 2009, where the in-domain encoder is the best source, with the Bridge 2006 tie.
# Every number is read from the files below at run time and rounded once, half away from zero;
# values a script already rounded once are printed as written, never rounded again. The prose
# states only what CHECKS verify on the same files, and every input md5 is pinned, so a different
# or edited file aborts the patch before anything is written. Harry, 2026-09-29.

TARGET = Path("RESULTS.md")
BF = Path("benchmark_final")
PINNED = {
    ("bf", "estimators_summary.tsv"): "a82f6da1cd995a4d13fd4c6a825969a6",
    ("bf", "estimators_cells.tsv"): "3a69311ffa9b0d90befdb7287f4ce632",
    ("bf", "signals_main.tsv"): "a3f4c0d80490904618a69ede169f5858",
    ("bf", "similarity_summary.tsv"): "14ef5a915ca203eed44649da767ec66d",
    ("bf", "similarity_cells.tsv"): "f1ca3ab393d465df3dab7ef8c6d28ddd",
    ("bf", "best_average_cells.tsv"): "37a2d6d02d2232da2d06b73248b19596",
    ("bf", "best_average_report.md"): "028f88245621a2bf83da55b9f346c4c9",
    ("t4", "task4_eval_summary.tsv"): "17d4dfd7116bd9d2a0a4642719210c8d",
    ("t4", "task4_eval_cells.tsv"): "d1f2c4fe25d7d51d0a253dc50f3a3260",
    ("t4", "task4_eval_repro.tsv"): "dab14daf29eb1ba817ab824f7b3e75f3",
    ("t4", "task4_scores.jsonl"): "52ed7805e369eb479f5ca45afcc1cb34",
    ("t4", "task4_ceiling_report.md"): "dc37d1cc158b39359a646340948a52ba",
    ("repo", "scripts/task4_extract.py"): "f6685d8e61157b1be71005921bc81e1f",
    ("repo", "slurm/generators/gen_task4_jobs.sh"): "e4316008887ff4a89aee46670eed4bf8",
    ("repo", "configs/task4/bassignana2022_table2.tsv"): "62dee49cdd471e280430885e28d6dd9d",
}
TARGETS = ("assist2017", "ednet", "junyi", "algebra2005", "bridge2006", "assist2009",
           "algebra2006")
ESTS = ("hscore_kt_causal", "logme_kt_causal")
FAIL_T = ("assist2017", "algebra2006")
NLI = ("mnli", "qnli", "rte")
BIOBERT = "dmis-lab/biobert-v1.1"
MARGIN = Decimal("0.001")
END = "_End of consolidated results._"
ANCHOR = "matched train draw is a different eighth of the train split at each seed._\n\n" + END

OLD_123 = ("hscore_shrunk equals hscore; the largest-source rule has one ranking per target. "
           "Scores use the\nleakage-safe causal features of src/estimators/features.py on 3,000 "
           "target learners, capped at\n50,000 positions.")
OLD_124 = ("H-score picks the target's own encoder in 15 of 21 rankings, LogME in 14 and the "
           "few-shot proxy in\n12. That is right on EdNet (in-domain best), Bridge 2006 and "
           "ASSISTments 2009 (in-domain tied for\nbest), and wrong for all three estimators on "
           "ASSISTments 2017 (EdNet best) and Algebra 2006 (Junyi\nbest). Ruled out as causes by "
           "the LogME diagnostics (scripts/diagnose_logme.py, three original\ntargets): the "
           "trained skill table, since re-scoring the in-domain encoder with its skill table "
           "at\nthe random start leaves the ASSISTments 2017 pick unchanged on 3 of 3 seeds; and "
           "layer choice, with\nrank correlation by layer L0 +0.20, L1 -0.40, L2 -0.40, L3 "
           "+0.04, L4 +0.24, L5 +0.62, L6 +0.42,\nwhere L5 still picks in-domain on ASSISTments "
           "2017. Per-skill LogME fails at every layer.")
NEW_124_TEXT = ("H-score picks the target's own encoder in 15 of 21 rankings, LogME in 14 and the "
                "few-shot proxy in 12. That is right on EdNet and ASSISTments 2009, where "
                "in-domain is the best source (on ASSISTments 2009 Junyi and scratch are tied "
                "with it), not an error on Bridge 2006, where in-domain is tied with Junyi, the "
                "best, and wrong for all three estimators on ASSISTments 2017 (EdNet best) and "
                "Algebra 2006 (Junyi best). Ruled out as causes by the LogME diagnostics "
                "(scripts/diagnose_logme.py, three original targets): the trained skill table, "
                "since re-scoring the in-domain encoder with its skill table at the random start "
                "leaves the ASSISTments 2017 pick unchanged on 3 of 3 seeds; and layer choice, "
                "with rank correlation by layer L0 +0.20, L1 -0.40, L2 -0.40, L3 +0.04, L4 "
                "+0.24, L5 +0.62, L6 +0.42, where L5 still picks in-domain on ASSISTments 2017. "
                "Neither diagnostic was run on Algebra 2006. Per-skill LogME fails at every "
                "layer.")


def q(x, places: str = "0.0001", sign: bool = True) -> str:
    d = Decimal(str(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def signed(s: str) -> str:
    """A value a script already rounded once: add the sign, never round again."""
    return s if s.startswith(("-", "+")) else f"+{s}"


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


class Inputs:
    def __init__(self, t4: Path) -> None:
        self.dirs = {"bf": BF, "t4": t4, "repo": Path(".")}

    def path(self, key: tuple[str, str]) -> Path:
        return self.dirs[key[0]] / key[1]

    def raw(self, key: tuple[str, str]) -> bytes:
        p = self.path(key)
        if not p.is_file():
            sys.exit(f"ABORT: {p} is missing; nothing written")
        b = p.read_bytes()
        got = hashlib.md5(b).hexdigest()
        if got != PINNED[key]:
            sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[key]}; nothing written")
        return b

    def tsv(self, key: tuple[str, str]) -> list[dict]:
        text = self.raw(key).decode("utf-8")
        return list(csv.DictReader(text.splitlines(), delimiter="\t"))

    def text(self, key: tuple[str, str]) -> str:
        return self.raw(key).decode("utf-8")


def md_table(text: str, heading: str | None = None) -> list[dict]:
    """The first markdown table after `heading` (or the first table), as dicts."""
    lines = text.splitlines()
    i = 0
    if heading is not None:
        i = lines.index(heading)
    while i < len(lines) and not lines[i].startswith("| "):
        i += 1
    head = [c.strip() for c in lines[i].strip("|").split("|")]
    rows = []
    for ln in lines[i + 2:]:
        if not ln.startswith("| "):
            break
        rows.append(dict(zip(head, [c.strip() for c in ln.strip("|").split("|")])))
    return rows


def build(inp: Inputs, results: str) -> tuple[str, str]:
    fails: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    esum = {r["estimator"]: r for r in inp.tsv(("bf", "estimators_summary.tsv"))
            if r["track"] == "B" and r["view"] == "pretrained"}
    cells = [r for r in inp.tsv(("bf", "estimators_cells.tsv"))
             if r["track"] == "B" and r["budget"] == "n3000" and r["view"] == "pretrained"]
    C = {(r["estimator"], r["target"], r["seed"]): r for r in cells}
    sig = {(r["estimator"], r["target"]): r for r in inp.tsv(("bf", "signals_main.tsv"))
           if r["budget"] == "n3000"}
    ssum = inp.tsv(("bf", "similarity_summary.tsv"))
    scells = inp.tsv(("bf", "similarity_cells.tsv"))
    bcells = inp.tsv(("bf", "best_average_cells.tsv"))
    brep = md_table(inp.text(("bf", "best_average_report.md")))
    B = {(r["estimator"], r["view"]): r for r in brep}
    tsum = {(r["estimator"], r["regime"]): r for r in inp.tsv(("t4", "task4_eval_summary.tsv"))}
    tcells = inp.tsv(("t4", "task4_eval_cells.tsv"))
    repro = inp.tsv(("t4", "task4_eval_repro.tsv"))
    ceil = md_table(inp.text(("t4", "task4_ceiling_report.md")),
                    "## Tasks run in Task 4 (AGNews, MNLI, QNLI, RTE)")
    K = {(r["estimator"], r["regime"]): r for r in ceil}
    scores = [json.loads(ln) for ln in inp.text(("t4", "task4_scores.jsonl")).splitlines() if ln]
    extract = inp.text(("repo", "scripts/task4_extract.py"))
    gen = inp.text(("repo", "slurm/generators/gen_task4_jobs.sh"))

    def seeds_of(est: str, t: str) -> list[dict]:
        return sorted((r for k, r in C.items() if k[0] == est and k[1] == t),
                      key=lambda r: r["seed"])

    def mean_regret(est: str, t: str) -> Decimal:
        rs = seeds_of(est, t)
        return sum(Decimal(r["regret"]) for r in rs) / Decimal(len(rs))

    # 12.3: shrunk H-score makes the same picks; the learner draw per target.
    h, s = esum["hscore_kt_causal"], esum["hscore_shrunk_kt_causal"]
    for k in ("rankings", "top1", "equivalent", "mean_regret", "max_regret", "negative_choices"):
        need(h[k] == s[k], f"hscore and hscore_shrunk differ on {k}")
    need(all(C[("hscore_kt_causal", t, sd)]["choice"] == C[("hscore_shrunk_kt_causal", t, sd)]
             ["choice"] for (_e, t, sd) in C if _e == "hscore_kt_causal"),
         "hscore_shrunk picks differ from hscore in some ranking")
    need(q(s["rho"]) != q(h["rho"]), "shrunk rho equals plain rho at 4 dp; the old note stands")
    need(f"| hscore_kt_causal | 21 | {q(h['rho'])} | {h['top1']}/21 |" in results,
         "the hscore row of 12.3 does not match estimators_summary.tsv")
    for t, n in (("assist2017", 1366), ("algebra2005", 453), ("bridge2006", 904),
                 ("assist2009", 2495), ("algebra2006", 1048), ("ednet", 3000),
                 ("junyi", 3000)):
        need(f"| hscore_kt_causal | {t} | {n} / " in results,
             f"12.6 does not list the {t} train draw as {n} learners")
    new_123 = "\n".join(wrap(
        f"hscore_shrunk makes the same pick as hscore in all {s['rankings']} rankings (top-1 "
        f"{s['top1']}/21, tied with best {s['equivalent']}/21, regret "
        f"{q(s['mean_regret'], sign=False)}"
        f" / {q(s['max_regret'], sign=False)}) at rho {q(s['rho'])}; the largest-source rule has "
        "one ranking per target. Scores use the leakage-safe causal features of "
        "src/estimators/features.py on the fine-tune's learner draw (3,000 learners on EdNet and "
        "Junyi, the whole train split on the five smaller targets), capped at 50,000 positions."))

    # 12.4: what is right, tied and wrong.
    for t, pick, top1, eq in (("ednet", "src:ednet", "True", "True"),
                              ("assist2009", "src:assist2009", "True", "True"),
                              ("bridge2006", "src:bridge2006", "False", "True"),
                              ("assist2017", "src:assist2017", "False", "False"),
                              ("algebra2006", "src:algebra2006", "False", "False")):
        need(all(r["choice"] == pick and r["top1"] == top1 and r["equivalent"] == eq
                 for r in seeds_of("hscore_kt_causal", t)),
             f"hscore on {t} is not {pick} with top1 {top1}, tied {eq} on every seed")
    need("| assist2009 | assist2009 (in-domain) +0.0010, junyi +0.0006, scratch |" in results,
         "12.1 no longer lists Junyi and scratch as tied with in-domain on assist2009")
    need("| bridge2006 | junyi +0.0084, bridge2006 (in-domain) +0.0079 |" in results,
         "12.1 no longer lists Junyi best and in-domain tied on bridge2006")
    diag = {r["target"] for r in cells if r["estimator"].startswith("logme_kt_causal_L")}
    need(diag == {"assist2017", "ednet", "junyi"},
         f"layer diagnostics exist for {sorted(diag)}, not only the three original targets")
    need(esum["hscore_kt_causal"]["rho"] > esum["logme_kt_causal"]["rho"],
         "H-score does not lead LogME on StudentBERT")

    # 12.7 signals.
    for est in ESTS:
        for t in TARGETS:
            need((est, t) in sig, f"signals lack {est} {t}")
            if (est, t) in sig:
                d = abs(Decimal(sig[(est, t)]["mean_regret"]) - mean_regret(est, t))
                need(d <= Decimal("0.0000005"),
                     f"signals mean regret disagrees with cells, {est} {t}")
        for t in FAIL_T:
            r = sig.get((est, t), {})
            need(r.get("modal_pick") == f"src:{t}" and r.get("pick_agreement") == "3/3",
                 f"{est} {t} is not the in-domain pick on 3 of 3 seeds")
    for t in FAIL_T:
        r = sig.get(("fewshot_ft_fit200_e5", t), {})
        need(r.get("modal_pick") == f"src:{t}" and r.get("pick_agreement") in ("2/3", "3/3"),
             f"the 200-learner proxy does not share the in-domain pick on {t}")
    h17 = sig.get(("hscore_kt_causal", "assist2017"), {}).get("margin_over_sd", "0")
    hj = sig.get(("hscore_kt_causal", "junyi"), {})
    need(Decimal(h17) > 2, "H-score's assist2017 margin is not above 2 seed SDs")
    need(Decimal(hj.get("margin_over_sd", "9")) < 1 and mean_regret("hscore_kt_causal", "junyi")
         == 0, "H-score's Junyi pick is not a correct pick inside seed noise")

    # 12.7 foreign-only view and the best-average rule.
    fs = {r["estimator"]: r for r in ssum}
    ba_f, ba_p, ls_p = (B.get(("best average", "foreign")), B.get(("best average", "pretrained")),
                        B.get(("largest source", "pretrained")))
    need(None not in (ba_f, ba_p, ls_p), "best_average_report.md lacks a summary row")
    if fails:
        sys.exit("ABORT, nothing written:\n  " + "\n  ".join(fails))
    need(f"| largest source | 7 | {signed(ls_p['rho'])} | {ls_p['top1']} | {ls_p['equivalent']} | "
         f"{ls_p['mean_regret']} / {ls_p['max_regret']} | {ls_p['negative_choices']} |" in results,
         "the largest-source rule re-judged by best_average_source.py does not reproduce 12.3")
    need(B[("largest source", "foreign")]["rho"] == q(fs["largest source"]["rho"], sign=False),
         "the foreign-view largest-source rule differs between the two scripts")
    need(Decimal(fs["largest source"]["rho"]) > Decimal(fs["hscore_kt_causal"]["rho"])
         and int(fs["hscore_kt_causal"]["top1"]) / 21 > int(fs["largest source"]["top1"]) / 7,
         "foreign view: largest source no longer orders better, or H-score no longer picks more")
    need(Decimal(ba_f["rho"]) > max(Decimal(r["rho"]) for r in ssum),
         "foreign view: the best-average rule is not the highest rank correlation")
    need(Decimal(ba_p["rho"]) < Decimal(h["rho"])
         and Decimal(ba_p["mean_regret"]) > Decimal(q(h["mean_regret"], sign=False)),
         "pretrained view: the rule does not trail H-score")
    sims = [r for r in ssum if r["estimator"].startswith("similarity")]
    need(all(Decimal(r["rho"]) < Decimal("0.2") for r in sims), "similarity rules are not weak")
    brule = {r["target"]: r for r in bcells if r["estimator"] == "best average"
             and r["view"] == "pretrained"}
    need(sorted(brule) == sorted(TARGETS), "best-average cells lack a target")
    need(all(r["choice"] == "src:junyi" for r in brule.values()),
         "the rule does not pick Junyi on every target in the pretrained view")
    need(brule["algebra2006"]["top1"] == "True", "the rule does not pick the winner on algebra2006")
    hs_t = {t: mean_regret("hscore_kt_causal", t) for t in TARGETS}
    ru_t = {t: Decimal(brule[t]["regret"]) for t in TARGETS}
    both = [t for t in TARGETS if hs_t[t] > MARGIN and ru_t[t] > MARGIN]
    need(both == ["assist2017"], f"targets both lose on: {both}")
    oracle = sum(min(hs_t[t], ru_t[t]) for t in TARGETS) / Decimal(len(TARGETS))
    fh = {(r["estimator"], r["target"], r["seed"]): r for r in scells if r["seed"] != "-"}
    worst = [r for r in scells if r["estimator"] in ("hscore_kt_causal", "hscore_shrunk_kt_causal",
                                                     "logme_kt_causal")
             and Decimal(r["regret"]) == Decimal(max(r2["regret"] for r2 in scells
                                                     if r2["estimator"] == r["estimator"]))]
    need(len(worst) == 3 and {(r["target"], r["seed"], r["choice"]) for r in worst}
         == {("assist2017", "42", "src:bridge2006")},
         "the foreign-view maximum is not one shared assist2017 seed-42 pick of bridge2006")
    need(all(fh[("hscore_kt_causal", "algebra2006", sd)]["choice"] == "src:junyi"
             for sd in ("1", "2", "42")),
         "foreign view: H-score does not pick Junyi on algebra2006")
    if fails:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(fails))

    L = ["### 12.7 Confidence signals, the foreign-source view and two simple rules (the seven "
         "N=3000 cells; analysis/estimator_signals.py, analysis/similarity_baseline.py, "
         "analysis/best_average_source.py)", ""]
    L += wrap("Signals from analysis/estimator_signals.py over the 3 estimator seeds of each "
              "target: the modal pick, the seeds that agree with it, and the margin of the pick "
              "over the runner-up divided by the per-candidate seed SD, as that script writes it "
              "(6 dp). Mean regret is recomputed from estimators_cells.tsv. The foreign-source "
              "view, from analysis/similarity_baseline.py, leaves the target's own encoder out "
              "and judges every choice against the best foreign source. The best-average rule "
              "(analysis/best_average_source.py, commit 5471edc) scores each source by minus its "
              "mean rank as a foreign source on the other six targets, so no target's results "
              "enter its own scores; it re-judges the largest-source rule as a check, which "
              "reproduces 12.3 exactly.")
    L += [""]
    L += wrap("Files (md5): " + ", ".join(f"{k[1]} {PINNED[k]}" for k in PINNED if k[0] == "bf")
              + ".")
    L += ["", "| estimator | target | modal pick | agreement | margin / seed SD | mean regret |",
          "|---|---|---|---|---|---|"]
    for est in ESTS:
        for t in TARGETS:
            r = sig[(est, t)]
            L.append(f"| {est} | {t} | {r['modal_pick'].removeprefix('src:')} | "
                     f"{r['pick_agreement']} | {r['margin_over_sd']} | "
                     f"{q(mean_regret(est, t), sign=False)} |")
    L += ["", "Foreign-source view (the target's own encoder out; 7 targets):", "",
          "| estimator | rankings | rho | top-1 | tied with best | regret mean / max | "
          "negative picks |", "|---|---|---|---|---|---|---|"]
    frows = [("best average (leave one target out)", ba_f["rankings"], signed(ba_f["rho"]),
              ba_f["top1"], ba_f["equivalent"], ba_f["mean_regret"], ba_f["max_regret"],
              ba_f["negative_choices"], Decimal(ba_f["rho"]))]
    for r in ssum:
        n = r["rankings"]
        frows.append((r["estimator"], n, q(r["rho"]), f"{r['top1']}/{n}", f"{r['equivalent']}/{n}",
                      q(r["mean_regret"], sign=False), q(r["max_regret"], sign=False),
                      r["negative_choices"], Decimal(r["rho"])))
    for fr in sorted(frows, key=lambda x: -x[-1]):
        L.append(f"| {fr[0]} | {fr[1]} | {fr[2]} | {fr[3]} | {fr[4]} | {fr[5]} / {fr[6]} | "
                 f"{fr[7]} |")
    L += [""] + wrap("Best-average rule against H-score, pretrained view (regret per target; "
                     "the rule has one ranking per target, H-score three):")
    L += ["", "| target | best-average pick | rule regret | H-score regret |", "|---|---|---|---|"]
    for t in TARGETS:
        L.append(f"| {t} | {brule[t]['choice'].removeprefix('src:')} | {q(ru_t[t], sign=False)} | "
                 f"{q(hs_t[t], sign=False)} |")
    L += [""]
    L += wrap(f"Pretrained view, all targets: best average rho {signed(ba_p['rho'])}, top-1 "
              f"{ba_p['top1']}, tied with best {ba_p['equivalent']}, regret {ba_p['mean_regret']}"
              f" / {ba_p['max_regret']}, {ba_p['negative_choices']} negative picks. Taking the "
              f"lower of the two regrets on each target, a bound that reads the gold, gives "
              f"{q(oracle, sign=False)}.")
    L += [""]
    L += wrap("_Read: the logged confidence signals do not flag the two wrong picks. On "
              "ASSISTments 2017 and Algebra 2006, H-score and LogME pick the in-domain encoder "
              "on 3 of 3 seeds, and the 200-learner few-shot proxy makes the same pick on at "
              f"least 2 of 3; H-score's ASSISTments 2017 margin is {h17} seed SDs, while its "
              f"correct Junyi pick has a margin of {hj['margin_over_sd']}. With the own encoder "
              "removed, H-score picks Junyi on Algebra 2006 on every seed; the largest-source "
              "rule orders the foreign sources better than H-score but H-score picks the "
              "winner more often; the dataset-similarity rules are weak. The best-average rule "
              "picks Junyi on every target in the pretrained view: it is right on Algebra 2006, "
              "has the highest rank "
              "correlation in the foreign-source view, and trails H-score in the pretrained view. "
              "H-score and the rule lose regret on different targets, and only ASSISTments "
              "2017 defeats both. The foreign-view maximum of H-score, its shrunk form and "
              "LogME comes from one ranking, ASSISTments 2017 at seed 42, where all three pick "
              "Bridge 2006._")
    sec127 = "\n".join(L)

    # 13: cross-domain check.
    ts = {k: tsum.get(k) for k in [(e, g) for e in ("hscore", "hscore_shrunk", "logme")
                                  for g in ("frozen", "tuned")]}
    need(None not in ts.values(), "task4_eval_summary.tsv lacks a row")
    if fails:
        sys.exit("ABORT, nothing written:\n  " + "\n  ".join(fails))
    need(all(r["rankings"] == "24" for r in ts.values()), "Task 4 rows are not 24 rankings")
    need(Decimal(ts[("hscore", "frozen")]["rho"]) > Decimal(ts[("logme", "frozen")]["rho"]),
         "H-score does not rank frozen performance above LogME")
    need((ts[("hscore", "tuned")]["top1"], ts[("logme", "tuned")]["top1"]) == ("4", "7"),
         "tuned top-1 is not 4 (H-score) and 7 (LogME)")
    need(all(Decimal(ts[(e, "tuned")]["rho"]) < Decimal("0.25") for e in ("hscore", "logme")),
         "tuned rank correlation is not about +0.21")
    need(K[("frozen_oracle", "tuned")]["top1"] == "2/8", "the frozen oracle does not pick 2 of 8")
    lt = K[("loto_best_average", "tuned")]
    need(Decimal(lt["mean_regret"]) < min(Decimal(q(ts[(e, "tuned")]["mean_regret"], "0.01",
                                                    sign=False)) for e in ("hscore", "logme"))
         and lt["top1"] == "0/8", "the best-average rule is not lowest in regret with no winner")
    need(Decimal(K[("published_logme", "tuned")]["rho"]) > Decimal(ts[("logme", "tuned")]["rho"]),
         "the published LogME does not rank tuned performance above ours")
    tuned = [r for r in tcells if r["regime"] == "tuned"]
    best = {(r["task"], r["pooling"]): r["best"] for r in tuned}
    need(all(b == "roberta-base" for (t, _p), b in best.items() if t in NLI)
         and all(b == "distilbert-base-uncased" for (t, _p), b in best.items() if t == "agnews"),
         "tuned winners are not RoBERTa on NLI and DistilBERT on AGNews")
    bio = Counter(r["estimator"] for r in tuned if r["choice"] == BIOBERT)
    need((bio["hscore"], bio["logme"]) == (11, 15) and BIOBERT not in best.values(),
         f"BioBERT picks {dict(bio)} or BioBERT is a tuned winner")
    rp = {(r["task"], r["pooling"]): Decimal(r["spearman_vs_published"]) for r in repro}
    need(len(rp) == 8 and all(v >= Decimal("0.8") for (t, _p), v in rp.items() if t == "agnews")
         and all(Decimal("0.4") < v < Decimal("0.7") for (t, _p), v in rp.items() if t in NLI),
         "LogME reproduction is not strong on AGNews and partial on the NLI tasks")
    ns = {}
    for r in scores:
        ns.setdefault(r["task"], set()).add(r["n"])
    need(len(scores) == 504 and ns == {"agnews": {10000}, "mnli": {10000}, "qnli": {10000},
                                       "rte": {2490}}, f"score count or sample sizes {ns}")
    table = inp.text(("repo", "configs/task4/bassignana2022_table2.tsv"))
    for phrase in ("analysis/task4_gold_check.py verifies every row against the paper's own",
                   "RTE mean-pooled rows give Pearson 0.605 and 0.581 against the printed 0.616 "
                   "and 0.597",
                   "the MNLI mean-pooled tuned sds of the last four models repeat SciERC's "
                   "exactly"):
        need(phrase in table, f"the published-table header no longer says: {phrase}")
    need('TASKS = {\n    "agnews"' in extract and '"scierc"' not in extract
         and '"airline"' not in extract, "task4_extract.py task list changed")
    need("so every encoder is scored on the same training examples" in extract
         and "Candidate-independent" in extract, "the extractor no longer states a shared sample")
    need("#SBATCH --partition=gpu" in gen and "#SBATCH --partition=short" in gen,
         "the generator no longer puts extraction on gpu and scoring on short")
    logs = sorted(p.name for p in (inp.dirs["t4"] / "logs").glob("*.log"))
    xid = sorted(int(n.rsplit("_", 1)[1][:-4]) for n in logs if n.startswith("t4x_"))
    sid = [int(n.rsplit("_", 1)[1][:-4]) for n in logs if n.startswith("t4s_")]
    need(xid == list(range(10673386, 10673393)) and sid == [10673657],
         f"Task 4 job logs are {xid} and {sid}")
    per = Counter((r["estimator"], r["task"], r["pooling"]) for r in tuned)
    need(len(per) == 24 and set(per.values()) == {3}, "Task 4 rankings are not 3 seeds each")
    need('"--n", type=int, default=10000' in extract
         and '"--max_length", type=int, default=256' in extract
         and "last_hidden_state" in extract, "task4_extract.py defaults changed")
    need("task4_extract.py --model $model --task \\$task --seed \\$seed --out_dir" in gen
         and "--max_length" not in gen and " --n " not in gen,
         "the job generator overrides the extraction defaults")
    if fails:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(fails))

    M = ["## 13. Cross-domain check: NLP encoder selection (Bassignana et al., EMNLP 2022; "
         "analysis/task4_evaluate.py, analysis/task4_ceiling.py; /projects/algl/dai.hany/task4/)",
         ""]
    M += wrap("Seven Hugging Face encoders (bert-base-uncased, roberta-base, "
              "distilbert-base-uncased, emilyalsentzer/Bio_ClinicalBERT, dmis-lab/biobert-v1.1, "
              "cardiffnlp/twitter-roberta-base, allenai/scibert_scivocab_uncased) on four of the "
              "paper's classification tasks (AGNews, MNLI, QNLI, RTE), with [CLS] and mean "
              "pooling. Features from scripts/task4_extract.py at its defaults: the last hidden "
              "layer, maximum length 256, a seeded sample of 10,000 training examples that does "
              "not depend on the encoder (RTE's whole training split of 2,490, so its three seeds "
              "see the same examples); 3 sample seeds; 504 scores. Code 6dd0f0c; extraction "
              "Slurm 10673386 to 10673392 (one gpu-partition job per encoder), scoring 10673657 "
              "(short partition). Gold: the paper's Table 2 "
              "(configs/task4/bassignana2022_table2.tsv, checked "
              "against the paper's own correlations by analysis/task4_gold_check.py), frozen and "
              "fully fine-tuned means over 5 seeds. A choice ties with the best within 0.1 points "
              "or inside a two-sided 95% interval from the published SDs, a rule fixed before any "
              "score. The paper's Airline and SciERC tasks were not run; the extractor covers the "
              "four "
              "above. Published-table issues: the RTE mean-pooled rows "
              "reproduce the paper's Pearson only to 0.605 and 0.581 against the printed 0.616 "
              "and 0.597, and the MNLI mean-pooled tuned SDs of the last four models repeat "
              "SciERC's.")
    M += [""]
    M += wrap("Files (md5): " + ", ".join(f"{k[1]} {PINNED[k]}" for k in PINNED if k[0] == "t4")
              + ".")
    M += [""] + wrap("LogME reproduction (Spearman between our LogME, mean over seeds, and the "
                     "published LogME over the 7 encoders):")
    M += ["", "| task | pooling | Spearman |", "|---|---|---|"]
    for r in repro:
        M.append(f"| {r['task']} | {r['pooling']} | {q(r['spearman_vs_published'])} |")
    M += ["", "Estimators against the published gold (4 tasks x 2 poolings x 3 seeds; regret in "
          "points):", "",
          "| estimator | gold | rho | tau-b | pairwise accuracy | top-1 | tied with best | "
          "regret mean / max | seed stability |", "|---|---|---|---|---|---|---|---|---|"]
    for (e, g), r in ts.items():
        M.append(f"| {e} | {g} | {q(r['rho'])} | {q(r['tau'])} | {q(r['pair_acc'], sign=False)} | "
                 f"{r['top1']}/24 | {r['tied_with_best']}/24 | "
                 f"{q(r['mean_regret'], '0.01', sign=False)} / "
                 f"{q(r['max_regret'], '0.01', sign=False)} | "
                 f"{q(r['seed_stability'], sign=False)} |")
    M += [""] + wrap("Reference rankings from the published table alone "
                     "(analysis/task4_ceiling.py, commit af372a3; one ranking per task and "
                     "pooling): the frozen oracle ranks by the published frozen means; best "
                     "average ranks by mean tuned rank on the other published tasks, same "
                     "pooling.")
    M += ["", "| ranking | gold | rho | tau-b | pairwise accuracy | top-1 | tied with best | "
          "regret mean / max |", "|---|---|---|---|---|---|---|---|"]
    for r in ceil:
        M.append(f"| {r['estimator']} | {r['regime']} | {signed(r['rho'])} | {signed(r['tau'])} | "
                 f"{r['pair_acc']} | {r['top1']} | {r['tied_with_best']} | "
                 f"{r['mean_regret']} / {r['max_regret']} |")
    M += [""]
    M += wrap(f"Tuned winners: RoBERTa on all six NLI settings, DistilBERT on both AGNews "
              f"settings. BioBERT, never a tuned winner, is H-score's pick in {bio['hscore']} of "
              f"24 rankings and LogME's in {bio['logme']}.")
    hf, ht = ts[("hscore", "frozen")], ts[("hscore", "tuned")]
    lf, ltu = ts[("logme", "frozen")], ts[("logme", "tuned")]
    M += [""]
    M += wrap(f"_Read: against frozen performance H-score ranks the encoders best (rho "
              f"{q(hf['rho'])}, top-1 {hf['top1']}/24), narrowly ahead of LogME ({q(lf['rho'])}), "
              f"as on StudentBERT. Against full fine-tuning both fall ({q(ht['rho'])} and "
              f"{q(ltu['rho'])}) and pick the tuned winner in {ht['top1']} and {ltu['top1']} of "
              f"24 rankings, with seed stability {q(ht['seed_stability'], sign=False)} and "
              f"{q(ltu['seed_stability'], sign=False)}, so seed spread does not flag the failure. "
              f"Exact knowledge of frozen performance picks the tuned winner in "
              f"{K[('frozen_oracle', 'tuned')]['top1'].replace('/', ' of ')} settings (rho "
              f"{signed(K[('frozen_oracle', 'tuned')]['rho'])}), so frozen performance is itself "
              f"a weak guide to fine-tuned performance here. The best-average rule has the lowest "
              f"mean regret against tuned gold ({lt['mean_regret']} points) but never picks the "
              f"winner. Our LogME reproduces the published ranking strongly on AGNews and partly "
              f"on the NLI tasks, and ranks tuned performance worse than the published LogME "
              f"({signed(K[('published_logme', 'tuned')]['rho'])} on the same settings)._")
    sec13 = "\n".join(M)
    return new_123, sec127 + "\n\n" + sec13 + "\n"


def apply(text: str, old: str, new: str, why: str) -> str:
    # Insertion-style edits keep their anchor, so "old in text" stays true after patching;
    # deciding by containment direction keeps re-runs as no-ops.
    if new in text and (old not in text or old in new):
        print(f"skip (already applied): {why}")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1; nothing written")
    print(f"ok: {why}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task4-dir", default="/projects/algl/dai.hany/task4")
    a = ap.parse_args()
    src = TARGET.read_text(encoding="utf-8")
    done = "### 12.7 Confidence signals" in src and "## 13. Cross-domain check" in src
    if done and OLD_123 not in src and OLD_124 not in src:
        print("skip (already applied): 12.3, 12.4, 12.7 and 13")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if "### 12.7" in src or "## 13." in src:
        sys.exit("ABORT: a 12.7 or 13 heading exists but not as this patcher writes it; "
                 "nothing written")
    new_123, sections = build(Inputs(Path(a.task4_dir)), src)
    out = apply(src, OLD_123, new_123, "12.3 shrunk H-score and learner draw")
    out = apply(out, OLD_124, "\n".join(wrap(NEW_124_TEXT)),
                "12.4 right, tied and wrong own-encoder picks")
    new_end = ANCHOR.replace(END, sections + "\n" + END)
    out = apply(out, ANCHOR, new_end, "sections 12.7 and 13")
    if out != src:
        TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
