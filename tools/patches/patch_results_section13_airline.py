from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
import textwrap
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# Why this exists: RESULTS.md section 13 says the paper's Airline and SciERC tasks were not run. On
# 2026-09-30 Airline was run (data split by the authors' own converter, extractor patched at
# 7b1cfb7, scoring Slurm 10716202, evaluation 10716582) and SciERC was excluded. This adds 13.1
# with the Airline results, the 10-setting summaries, the proof that the 4-task results above
# reproduce byte for byte, and the SciERC exclusion; it rewords the intro sentence that is now
# stale. Every number is read from pinned files at run time, the prose states only what CHECKS
# verify on them, and a different or edited file aborts before anything is written. Values are
# rounded once, half away from zero. Run from the code root with PYTHONPATH=. (it reads the gold
# through analysis/task4_evaluate.py). Harry, 2026-09-30.

TARGET = Path("RESULTS.md")
PINNED = {
    "task4_scores.jsonl": "9dfe5d930681ec04d79cb52c149bbc4e",
    "task4_eval_airline_summary.tsv": "753d37b09376ed5b573a0ccfaa34d8e2",
    "task4_eval_airline_repro.tsv": "c525a004ba2d7777c3f9480feeb523f7",
    "task4_eval_10_summary.tsv": "b5544568e4b1db985c0932156f8647fd",
    "task4_ceiling_10_cells.tsv": "04675b03fdfb32317e3502589c06f1fe",
    "task4_ceiling_10_report.md": "74e28a41113bddbbca88b0376824124f",
    "task4_eval_rerun4_summary.tsv": "17d4dfd7116bd9d2a0a4642719210c8d",
    "task4_ceiling_rerun_report.md": "dc37d1cc158b39359a646340948a52ba",
}
GOLD = Path("configs/task4/bassignana2022_table2.tsv")
GOLD_MD5 = "62dee49cdd471e280430885e28d6dd9d"
OLD_SCORES_MD5 = "52ed7805e369eb479f5ca45afcc1cb34"
SPLIT_MD5 = "b0fa4865e8d442b5e2146715e7a84c10"
ANCHOR = "\n\n_End of consolidated results._"
HEAD = "### 13.1 Airline added, SciERC excluded"
TWITTER, ROBERTA = "cardiffnlp/twitter-roberta-base", "roberta-base"
SCIBERT, BIOBERT = "allenai/scibert_scivocab_uncased", "dmis-lab/biobert-v1.1"
ESTS = ("hscore", "hscore_shrunk", "logme")
REFS = (("published_logme", "frozen"), ("loto_best_average", "frozen"), ("frozen_oracle", "tuned"),
        ("published_logme", "tuned"), ("loto_best_average", "tuned"))

EDITS = [
    ("before any score. The paper's Airline and SciERC tasks were not run; the extractor covers "
     "the four\nabove. Published-table issues: the RTE mean-pooled rows reproduce the paper's "
     "Pearson only to 0.605\n",
     "before any score. The paper's Airline and SciERC tasks were not run in this pass; on "
     "2026-09-30\nAirline was added and SciERC excluded (13.1). Published-table issues: the RTE "
     "mean-pooled rows\nreproduce the paper's Pearson only to 0.605\n",
     "13: the intro no longer says Airline was not run"),
]


def q(x, places: str = "0.0001", sign: bool = False) -> str:
    d = Decimal(str(x)).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=100, break_long_words=False, break_on_hyphens=False)


def raw(t4: Path, name: str) -> bytes:
    p = t4 / name
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing; nothing written")
    b = p.read_bytes()
    got = hashlib.md5(b).hexdigest()
    if got != PINNED[name]:
        sys.exit(f"ABORT: {p} md5 {got}, expected {PINNED[name]}; nothing written")
    return b


def tsv(t4: Path, name: str) -> list[dict]:
    return list(csv.DictReader(raw(t4, name).decode().splitlines(), delimiter="\t"))


def row(r: dict) -> str:
    n = r["rankings"]
    reg = f"{q(r['mean_regret'], '0.01')} / {q(r['max_regret'], '0.01')}"
    return (f"| {r['estimator']} | {r['regime']} | {q(r['rho'], sign=True)} | "
            f"{q(r['tau'], sign=True)} | {q(r['pair_acc'])} | {r['top1']}/{n} | "
            f"{r['tied_with_best']}/{n} | {reg} | {q(r['seed_stability'])} |")


def ceiling_section(report: str) -> dict:
    parts = report.split("## Tasks run in Task 4 plus airline", 1)
    if len(parts) != 2:
        return {}
    body = parts[1].split("\n## ", 1)[0].splitlines()
    rows = [ln for ln in body if ln.startswith("| ") and "---" not in ln][1:]
    cols = ([x.strip() for x in ln.split("|")[1:-1]] for ln in rows)
    return {(c[0], c[1]): c for c in cols}


def build(results: str, t4: Path) -> str:
    fails: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    lines = raw(t4, "task4_scores.jsonl").decode().splitlines(keepends=True)
    air = [r for r in map(json.loads, lines) if r["task"] == "airline"]
    need(len(lines) == 630 and len(air) == 126, f"{len(lines)} scores, {len(air)} for Airline")
    need(hashlib.md5("".join(lines[:504]).encode()).hexdigest() == OLD_SCORES_MD5,
         "the first 504 scores are not the 4-task file pinned in 13")
    need(all(r["converged"] for r in air), "an Airline score did not converge")
    raw(t4, "task4_ceiling_rerun_report.md")
    old = {(r["estimator"], r["regime"]): r for r in tsv(t4, "task4_eval_rerun4_summary.tsv")}
    need(f"task4_eval_summary.tsv {PINNED['task4_eval_rerun4_summary.tsv']}" in results
         and f"task4_ceiling_report.md\n{PINNED['task4_ceiling_rerun_report.md']}" in results,
         "13 no longer pins the 4-task summary and ceiling report that the reruns reproduce")

    metas = [json.loads(Path(f).read_text())
             for f in sorted(glob.glob(str(t4 / "features/airline/*.json")))]
    need(len(metas) == 21 and all(
        m["data_md5"] == SPLIT_MD5 and m["n"] == 10000 and m["n_train"] == 10248
        and m["max_length"] == 256 and m["hub"] == "local" for m in metas),
        "the 21 Airline feature files do not all record the pinned split, 10,000 of 10,248, 256")

    A = tsv(t4, "task4_eval_airline_summary.tsv")
    need(len(A) == 6 and all(
        r["rankings"] == r["top1"] == r["tied_with_best"] == "6" and float(r["max_regret"]) == 0
        and float(r["seed_stability"]) == 1 for r in A),
        "not every estimator picks the best in all 6 Airline rankings with zero regret")
    rep = {r["pooling"]: r for r in tsv(t4, "task4_eval_airline_repro.tsv")}
    need(set(rep) == {"cls", "mean"}
         and all(float(r["spearman_vs_published"]) == 1 for r in rep.values()),
         "our LogME does not match the published LogME ranking exactly on Airline")
    T = {(r["estimator"], r["regime"]): r for r in tsv(t4, "task4_eval_10_summary.tsv")}
    need(len(T) == 6 and all(r["rankings"] == "30" for r in T.values()),
         "the 10-setting summary is not 6 rows of 30 rankings")
    need(set(old) == set(T) and all(
        float(T[k]["rho"]) > float(old[k]["rho"])
        and float(T[k]["mean_regret"]) < float(old[k]["mean_regret"]) for k in T),
        "adding Airline does not raise every estimator's rho and lower its mean regret")

    R = ceiling_section(raw(t4, "task4_ceiling_10_report.md").decode())
    need(set(R) == set(REFS) and all(v[2] == "10" for v in R.values()),
         "the ceiling's Airline section is not the 5 reference rankings on 10 settings")
    cells = tsv(t4, "task4_ceiling_10_cells.tsv")
    oracle = [r for r in cells if r["task"] == "airline" and r["estimator"] == "frozen_oracle"]
    need(len(oracle) == 2 and len({r["rho"] for r in oracle}) == 1 and all(
        r["choice"] == r["best"] == TWITTER and float(r["regret"]) == 0 for r in oracle),
        "the frozen oracle does not pick Twitter-RoBERTa on Airline at one rho in both poolings")
    need(all(r["choice"] == TWITTER for r in cells if r["task"] == "airline"),
         "a published-table ranking does not pick Twitter-RoBERTa on Airline")
    wins = {r["task"] for r in cells if r["estimator"] == "loto_best_average"
            and r["regime"] == "tuned" and r["top1"] == "True"}
    nwin = sum(1 for r in cells if r["estimator"] == "loto_best_average"
               and r["regime"] == "tuned" and r["top1"] == "True" and r["task"] == "airline")
    need(wins == {"airline"} and nwin == 2,
         "the best-average rule's tuned winners are not exactly the two Airline settings")
    bavg = Decimal(R.get(("loto_best_average", "tuned"), ["0"] * 10)[8])
    need(all(bavg < Decimal(q(T[(e, "tuned")]["mean_regret"], "0.01")) for e in ESTS if T),
         "the best-average rule no longer has the lowest tuned regret on 10 settings")

    if hashlib.md5(GOLD.read_bytes()).hexdigest() != GOLD_MD5:
        fails.append(f"{GOLD} is not the gold table pinned here")
    else:
        from analysis.task4_evaluate import load_gold, tied
        g = load_gold(GOLD)

        def top(task: str, pool: str, reg: str) -> tuple[str, set]:
            d = g[(task, pool)]
            best = max(d, key=lambda m: d[m][reg][0])
            return best, {m for m in d if tied(d[m][reg], d[best][reg])}

        for pool in ("cls", "mean"):
            need(top("airline", pool, "frozen") == (TWITTER, {TWITTER}),
                 f"Twitter-RoBERTa is not the sole frozen best on Airline {pool}")
            need(top("airline", pool, "tuned") == (TWITTER, {TWITTER, ROBERTA}),
                 f"Airline {pool}: Twitter-RoBERTa tuned best, tied with RoBERTa only, fails")
            b, te = top("scierc", pool, "tuned")
            need(b == BIOBERT and SCIBERT in te,
                 f"SciERC {pool}: SciBERT is not tied with the tuned best, BioBERT")
    if fails:
        sys.exit("ABORT, the files do not support the prose; nothing written:\n  "
                 + "\n  ".join(fails))

    L = [f"{HEAD} (2026-09-30; scripts/task4_extract.py at 7b1cfb7; "
         "/projects/algl/dai.hany/task4/)", ""]
    L += wrap(
        "Data: Kaggle's Tweets.csv (crowdflower/twitter-airline-sentiment, md5 "
        "2fa808ea99b32814ccd64d7097d935f2, 14,640 tweets), split by the authors' own converter "
        "(mainlp/logme-nlp, commit 0046c725, project/src/tasks/sentiment/convert.py -rs 4012, "
        "run in a separate clone because it is GPL-3.0; Slurm 10709670) into train 10,248 (md5 "
        f"{SPLIT_MD5}), dev 1,464 (3d396e482e100a1274ccaecbc0d14d07) and test 2,928 "
        "(88c93304b94179117fbadfc72fa5d3a9). The paper scores LogME on the train split and "
        "reports frozen and tuned performance on dev; its gold rows are already in the table "
        "above.")
    L += [""] + wrap(
        "Setting: the same seven hub encoders, none pretrained on the target data; budget 10,000 "
        "of the 10,248 training tweets, a seeded sample that does not depend on the encoder; "
        "sample seeds 42, 1 and 2; no training (frozen features; the tuned gold is the paper's "
        "fine-tuning); input the tweet text as the converter leaves it, last hidden layer, [CLS] "
        "and the mean over non-padding subwords including [CLS] and [SEP] (the authors average "
        "word vectors and skip special tokens), maximum length 256. Extraction: 7 gpu-partition "
        "jobs (logs t4x_*.log in the folder above), every feature file recording the split's "
        "md5; scoring Slurm 10716202 (126 new scores, all converged); evaluation 10716582.")
    L += [""] + wrap(
        "Reproduction of the results above: the scores file grew from 504 to 630 lines and its "
        f"first 504 lines hash to {OLD_SCORES_MD5}, the 4-task file; rerun with --tasks agnews "
        "mnli qnli rte the evaluation reproduces all four of its outputs byte for byte (summary "
        f"{PINNED['task4_eval_rerun4_summary.tsv']}), and the ceiling's default report reproduces "
        f"too ({PINNED['task4_ceiling_rerun_report.md']}).")
    L += [""] + wrap("Files (md5): " + ", ".join(
        f"{n} {m}" for n, m in PINNED.items() if "rerun" not in n) + ".")
    hdr = ["| estimator | gold | rho | tau-b | pairwise accuracy | top-1 | tied with best | "
           "regret mean / max | seed stability |", "|---|---|---|---|---|---|---|---|---|"]
    L += [""] + wrap(
        "Airline, 2 poolings x 3 seeds (LogME reproduction: Spearman "
        f"{q(rep['cls']['spearman_vs_published'], sign=True)} for [CLS] and "
        f"{q(rep['mean']['spearman_vs_published'], sign=True)} for mean pooling):") + [""] + hdr
    L += [row(r) for r in A]
    L += [""] + wrap("All five tasks run, 10 settings x 3 seeds (the 8-setting rows are the "
                     "table above):") + [""] + hdr
    L += [row(T[(e, reg)]) for e in ESTS for reg in ("frozen", "tuned")]
    L += [""] + wrap("Reference rankings from the published table on the same 10 settings "
                     "(analysis/task4_ceiling.py --also_run airline):")
    L += ["", "| ranking | gold | rho | tau-b | pairwise accuracy | top-1 | tied with best | "
          "regret mean / max |", "|---|---|---|---|---|---|---|---|"]
    for k in REFS:
        v = R[k]
        L.append(f"| {v[0]} | {v[1]} | {q(v[3], sign=True)} | {q(v[4], sign=True)} | {v[5]} | "
                 f"{v[6]} | {v[7]} | {v[8]} / {v[9]} |")
    ht, lt, fo = T[("hscore", "tuned")], T[("logme", "tuned")], R[("frozen_oracle", "tuned")]
    L += [""] + wrap(
        "SciERC is excluded. The authors' repository has no SciERC converter "
        "(project/src/tasks/relclass holds only run scripts), so its entity-marked splits are not "
        "available, and the paper's LogME reads the markers as plain text while its classifiers "
        "add them as special tokens. Under the tie rule above, SciBERT, the domain-matched "
        "encoder, is also top-equivalent to the tuned best (BioBERT) in both poolings, so SciERC "
        "could not show a domain-matched failure.")
    L += [""] + wrap(
        "_Read: on Airline the domain-matched encoder is the right pick in both regimes and every "
        "estimator makes it: Twitter-RoBERTa is the sole frozen best and the tuned best, tied with "
        "RoBERTa, and H-score, its shrunk form and LogME pick it in all 6 rankings with zero "
        "regret and seed stability 1. Frozen and tuned rankings agree there (frozen oracle rho "
        f"{q(oracle[0]['rho'], sign=True)} against tuned gold in both poolings), so adding Airline "
        "raises every estimator's rank correlation and lowers its mean regret: H-score against "
        f"tuned gold reaches rho {q(ht['rho'], sign=True)} and top-1 {ht['top1']}/30 (LogME "
        f"{q(lt['rho'], sign=True)}, {lt['top1']}/30), and the frozen oracle picks the tuned "
        f"winner in {fo[6]} settings (2/8 above). The best-average rule keeps the lowest mean "
        f"tuned regret ({bavg} points) and picks the winner only on the two Airline settings. "
        "The failure review's fixed targets are defined on the 8 settings above._")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--t4", default="/projects/algl/dai.hany/task4")
    a = ap.parse_args()
    src = TARGET.read_text(encoding="utf-8")
    if HEAD in src and all(old not in src for old, _n, _w in EDITS):
        print("skip (already applied): 13 intro edit and 13.1")
        print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)
        return
    if HEAD in src or "### 13.1" in src:
        sys.exit("ABORT: a 13.1 heading exists but not as this patcher writes it; nothing written")
    section = build(src, Path(a.t4))
    out = src
    for old, new, why in EDITS:
        out = apply(out, old, new, why)
    out = apply(out, ANCHOR, section + ANCHOR, "section 13.1")
    TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest(), TARGET)


if __name__ == "__main__":
    main()
