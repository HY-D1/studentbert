from __future__ import annotations

import csv
import hashlib
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# Why this exists: RESULTS.md 12.5 ends on the open mechanism (estimators sample the train split,
# the in-domain encoder's pretraining data). The exposure test ran on 2026-09-24 (code 6ed04fe,
# decision rule fixed in analysis/exposure_report.py before any validation score existed;
# walltimes 6f8c756). This adds section 12.6 from exposure_v1/. Every number is read from those
# files at run time and rounded once, half away from zero; nothing is transcribed. The prose
# states only what CHECKS verify on the same files, and the input md5s are pinned to that run,
# so a different or edited file aborts the patch before anything is written. Harry, 2026-09-24.

TARGET = Path("RESULTS.md")
OUT = Path("exposure_v1")
EXPECTED_MD5 = {"exposure_summary.tsv": "adfd97ebdd98d70c618e37ef385bd0f9",
                "estimators_summary.tsv": "79715b91069b33d6bdbd7d88b5a0bdef",
                "estimators_cells.tsv": "0801251bb479829d7a461fb1446923cf",
                "exposure_report.md": "30c2dfd03a9c59cea6d5f6cf104e659d"}
ESTS = ("hscore_kt_causal", "logme_kt_causal")
SAMPLES = (("", "train draw"), ("@trainmatch", "matched train draw"),
           ("@val", "validation learners"))
TARGETS = ("assist2017", "ednet", "junyi", "algebra2005", "bridge2006", "assist2009",
           "algebra2006")
SMALL = ("algebra2005", "bridge2006", "assist2009", "assist2017", "algebra2006")
DETAIL = ("assist2017", "algebra2006", "bridge2006")
SEEDS = ("42", "1", "2")
ANCHOR = ("Scoring the estimators on validation learners would test it._\n\n"
          "_End of consolidated results._")


def q(x: str | Decimal, places: str = "0.0001", sign: bool = True) -> str:
    d = Decimal(x).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def read_tsv(name: str) -> list[dict]:
    p = OUT / name
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != EXPECTED_MD5[name]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {EXPECTED_MD5[name]} (the 2026-09-24 run); "
                 "nothing written")
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def pinned_text(name: str) -> str:
    p = OUT / name
    got = hashlib.md5(p.read_bytes()).hexdigest()
    if got != EXPECTED_MD5[name]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {EXPECTED_MD5[name]}; nothing written")
    return p.read_text(encoding="utf-8")


def build(summ: list[dict], esum: list[dict], cells: list[dict], results: str,
          report: str) -> str:
    fails: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    S = {(r["estimator"], r["target"]): r for r in summ}
    E = {r["estimator"]: r for r in esum if r["track"] == "B" and r["view"] == "pretrained"}
    C: dict = {}
    for r in cells:
        if r["track"] == "B" and r["budget"] == "n3000" and r["view"] == "pretrained":
            C[(r["estimator"], r["target"], r["seed"])] = r
    for est in ESTS:
        for t in TARGETS:
            need((est, t) in S, f"exposure summary lacks {est} {t}")
        for suf, _ in SAMPLES:
            need(est + suf in E, f"estimator summary lacks {est}{suf}")
            for t in DETAIL:
                for s in SEEDS:
                    need((est + suf, t, s) in C, f"cells lack {est}{suf} {t} seed {s}")
    if fails:
        sys.exit("ABORT, nothing written:\n  " + "\n  ".join(fails))

    def picks(est: str, t: str) -> list[str]:
        return [C[(est, t, s)]["choice"] for s in SEEDS]

    for est in ESTS:
        a, g = S[(est, "assist2017")], S[(est, "algebra2006")]
        need(a["verdict"] == "exposure", f"{est} assist2017 verdict {a['verdict']}")
        need((a["preferred_trainmatch"], a["preferred_val"]) == ("3/3", "1/3"),
             f"{est} assist2017 preferred {a['preferred_trainmatch']} / {a['preferred_val']}")
        need(picks(est, "assist2017") == ["src:assist2017"] * 3
             and picks(est + "@trainmatch", "assist2017") == ["src:assist2017"] * 3,
             f"{est} assist2017 train or matched picks are not in-domain on every seed")
        vp = picks(est + "@val", "assist2017")
        need(sorted(vp) == ["src:assist2017", "src:ednet", "src:ednet"],
             f"{est}@val assist2017 picks {vp}")
        need(g["verdict"] == "representation", f"{est} algebra2006 verdict {g['verdict']}")
        need(all(g[f"preferred_{p}"] == "3/3" for p in ("train", "trainmatch", "val")),
             f"{est} algebra2006 is not preferred 3/3 on every sample")
        need(Decimal(g["margin_val"]) > Decimal(g["margin_trainmatch"]),
             f"{est} algebra2006 margin does not grow on validation learners")
        for suf, _ in SAMPLES:
            need(all(C[(est + suf, "algebra2006", s)]["choice"] == "src:algebra2006"
                     and C[(est + suf, "algebra2006", s)]["equivalent"] == "False"
                     for s in SEEDS), f"{est}{suf} algebra2006 picks are not the untied own")
            need(all(C[(est + suf, "bridge2006", s)]["equivalent"] == "True" for s in SEEDS),
                 f"{est}{suf} bridge2006 has a pick not tied with the best")
        tr, tm, va = (E[est + suf] for suf, _ in SAMPLES)
        need(all(E[est + suf]["rankings"] == "21" and E[est + suf]["targets"] == "7"
                 for suf, _ in SAMPLES), f"{est} rows are not 21 rankings over 7 targets")
        need(Decimal(va["rho"]) < Decimal(tr["rho"])
             and Decimal(va["mean_regret"]) > Decimal(tr["mean_regret"]),
             f"{est} validation learners do not lose rho and gain regret against the train draw")
        need(Decimal(va["rho"]) > Decimal(tm["rho"]) and int(va["top1"]) > int(tm["top1"]),
             f"{est} validation learners do not beat the matched draw on rho and top-1")
        row = f"| {est} | 21 | {q(tr['rho'])} | {tr['top1']}/21 | {tr['equivalent']}/21 | " \
              f"{q(tr['mean_regret'], sign=False)} / {q(tr['max_regret'], sign=False)} |"
        need(row in results, f"train-draw row does not reproduce RESULTS.md 12.3: {row}")
        for t in SMALL:
            ratio = Decimal(S[(est, t)]["n_val"]) / Decimal(S[(est, t)]["n_train"])
            need(Decimal("0.12") <= ratio <= Decimal("0.13"), f"{t} val/train ratio {ratio}")
        for t in ("ednet", "junyi"):
            need(S[(est, t)]["n_train"] == S[(est, t)]["n_trainmatch"] == S[(est, t)]["n_val"]
                 == "3000", f"{est} {t} samples are not all 3,000 learners")
    need(S[("hscore_kt_causal", "bridge2006")]["verdict"] == "exposure"
         and S[("logme_kt_causal", "bridge2006")]["verdict"] == "representation",
         "bridge2006 verdicts are not exposure (H-score) and representation (LogME)")
    need("| algebra2006 | junyi +0.0087 |" in results,
         "RESULTS.md 12.1 no longer lists Junyi as the only top-equivalent into algebra2006")
    diffs = [ln for ln in report.splitlines() if "largest score difference" in ln]
    need(len(diffs) == 12 and all(ln.endswith("difference 0.000e+00") for ln in diffs)
         and all((" ednet " in ln) or (" junyi " in ln) for ln in diffs),
         "the determinism check is not 12 exact reproductions on EdNet and Junyi")
    for est in ESTS:
        for t in ("ednet", "junyi"):
            need(S[(est, t)]["margin_train"] == S[(est, t)]["margin_trainmatch"],
                 f"{est} {t} matched-draw margin differs from the train draw")
    if fails:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(fails))

    L = ["### 12.6 Exposure test for that failure (validation learners against a size-matched "
         "train draw; 3 estimator seeds; exposure_v1/)", "",
         "Both frozen-feature scorers re-run on the 7 x 7 grid with two more target samples: the",
         "validation learners (--split val), which no encoder saw (pretraining reads the train "
         "split and",
         "keeps its checkpoint by train loss), and a train draw with as many learners "
         "(--match_split val).",
         "Same candidates, seeds and 50,000-position cap. Code 6ed04fe, with the decision rule "
         "fixed in",
         "analysis/exposure_report.py before any validation score existed; walltimes 6f8c756. "
         "28 jobs,",
         "Slurm 10590371 to 10594649 (the Junyi validation Task 2 job 10591596 stalled on node "
         "d1020, was",
         "cancelled after 1:03:23 with no output, and ran again as 10594649). Margin: in-domain "
         "score minus",
         "the best other pretrained score, in each estimator's own units. Preferred: strictly top "
         "on at",
         "least 2 of 3 seeds. Verdict: exposure if preferred on the matched train draw and not on",
         "validation learners, representation if preferred on both, no preference if not "
         "preferred on the",
         "train draw. Where the matched draw is the train draw (EdNet, Junyi) it reproduces the "
         "train-draw",
         "scores exactly.", "",
         f"Files (md5): exposure_summary.tsv {EXPECTED_MD5['exposure_summary.tsv']},",
         f"estimators_summary.tsv {EXPECTED_MD5['estimators_summary.tsv']},",
         f"estimators_cells.tsv {EXPECTED_MD5['estimators_cells.tsv']},",
         f"exposure_report.md {EXPECTED_MD5['exposure_report.md']}.", "",
         "| estimator | target | learners train / matched / val | preferred train / matched / "
         "val | margin train / matched / val | verdict |",
         "|---|---|---|---|---|---|"]
    for est in ESTS:
        for t in TARGETS:
            r = S[(est, t)]
            L.append(f"| {est} | {t} | {r['n_train']} / {r['n_trainmatch']} / {r['n_val']} | "
                     f"{r['preferred_train']} / {r['preferred_trainmatch']} / "
                     f"{r['preferred_val']} | {q(r['margin_train'], '0.000001')} / "
                     f"{q(r['margin_trainmatch'], '0.000001')} / "
                     f"{q(r['margin_val'], '0.000001')} | {r['verdict']} |")
    L += ["", "Estimator quality by target sample (the seven N=3000 cells, pretrained view, "
          "21 rankings each):", "",
          "| estimator | sample | rho | top-1 | tied with best | regret mean / max |",
          "|---|---|---|---|---|---|"]
    for est in ESTS:
        for suf, name in SAMPLES:
            r = E[est + suf]
            L.append(f"| {est} | {name} | {q(r['rho'])} | {r['top1']}/21 | "
                     f"{r['equivalent']}/21 | {q(r['mean_regret'], sign=False)} / "
                     f"{q(r['max_regret'], sign=False)} |")
    L += ["", "Picks on the two failure targets and on Bridge 2006 (seeds 42, 1, 2; regret "
          "against the best source):", "",
          "| target | estimator | sample | picks | mean regret |", "|---|---|---|---|---|"]
    for t in DETAIL:
        for est in ESTS:
            for suf, name in SAMPLES:
                rs = [C[(est + suf, t, s)] for s in SEEDS]
                mean = sum(Decimal(r["regret"]) for r in rs) / Decimal(len(rs))
                pk = ", ".join(r["choice"].removeprefix("src:") for r in rs)
                L.append(f"| {t} | {est} | {name} | {pk} | {q(mean, sign=False)} |")
    L += ["",
          "_Read: pretraining exposure explains the in-domain preference on ASSISTments 2017 and "
          "not on",
          "Algebra 2006. On ASSISTments 2017 both estimators prefer the in-domain encoder on 3 of "
          "3 seeds of",
          "the matched train draw and on 1 of 3 with validation learners, where the other two "
          "seeds pick",
          "EdNet, the best source. On Algebra 2006 the preference holds on every seed of every "
          "sample, its",
          "margin grows on unseen learners, and Junyi is the only top-equivalent source (12.1), "
          "so that",
          "failure is neither exposure nor gold noise. On Bridge 2006 the estimators disagree "
          "(exposure for",
          "H-score, representation for LogME), but every pick there is tied with the best. "
          "Scoring",
          "validation learners is not a fix: on the five smaller targets a validation split holds "
          "an eighth",
          "as many learners as the train split, and in aggregate both estimators lose rank "
          "correlation and",
          "gain regret; at matched size, unseen learners give the higher rank correlation and "
          "top-1. On",
          "those five targets the train draw and the validation sample hold every learner of "
          "their split, so",
          "there the three seeds differ only in the position subsample and the random skill "
          "tables; the",
          "matched train draw is a different eighth of the train split at each seed._", ""]
    return "\n".join(L)


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
    src = TARGET.read_text(encoding="utf-8")
    if "### 12.6" in src:
        if "### 12.6 Exposure test" in src and ANCHOR not in src:
            print("skip (already applied): section 12.6")
            print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
            return
        sys.exit("ABORT: a 12.6 heading exists but not as this patcher writes it; nothing written")
    section = build(read_tsv("exposure_summary.tsv"), read_tsv("estimators_summary.tsv"),
                    read_tsv("estimators_cells.tsv"), src, pinned_text("exposure_report.md"))
    new = ("Scoring the estimators on validation learners would test it._\n\n" + section
           + "\n_End of consolidated results._")
    out = apply(src, ANCHOR, new, "section 12.6 exposure test")
    if out != src:
        TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
