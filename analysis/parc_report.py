from __future__ import annotations

"""PARC as an additional baseline (docs/parc_design.md): one report from the existing evaluators.

StudentBERT: the rows analysis/evaluate_estimators.py wrote for parc_kt_causal, the own-encoder
picks and each target's picks computed from the PARC score files, and the check that every other
row of that run equals benchmark_final/estimators_summary.tsv (RESULTS.md 12.7), so the gold, tie
rule and evaluator are the ones of 12.3. NLP: the rows analysis/task4_evaluate.py wrote for parc,
and the check that the same evaluator, run on the 13.1 score file, reproduces the six estimator
rows of RESULTS.md 13.1. Any failed check stops the script before anything is written.

  PYTHONPATH=. python analysis/parc_report.py --results RESULTS.md \
      --kt-summary benchmark_final/parc_estimators_summary.tsv \
      --old-summary benchmark_final/estimators_summary.tsv --kt-scores tg1_parc_kt_*.jsonl \
      --nlp-summary benchmark_final/task4_parc_summary.tsv \
      --nlp-check benchmark_final/task4_check10_summary.tsv --out benchmark_final/parc_report.md
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

OLD_MD5 = "a82f6da1cd995a4d13fd4c6a825969a6"
SKIP = ("extract_s", "score_s")


def tsv(path: str) -> list[dict]:
    return list(csv.DictReader(Path(path).read_text().splitlines(), delimiter="\t"))


def kt_rows(summary: str, old: str) -> tuple[list[dict], int]:
    if hashlib.md5(Path(old).read_bytes()).hexdigest() != OLD_MD5:
        sys.exit(f"ABORT: {old} is not the summary of RESULTS.md 12.7")
    new = {(r["estimator"], r["track"], r["view"]): r for r in tsv(summary)}
    ref = {(r["estimator"], r["track"], r["view"]): r for r in tsv(old)}
    shared = [k for k in new if k in ref]
    bad = [k for k in shared
           if {c: v for c, v in new[k].items() if c not in SKIP}
           != {c: v for c, v in ref[k].items() if c not in SKIP}]
    extra = [k for k in new if k not in ref]
    if bad or not shared:
        sys.exit(f"ABORT: rows differ from the 12.7 summary: {bad}")
    if sorted(k[0] for k in extra) != ["parc_kt_causal", "parc_kt_causal"]:
        sys.exit(f"ABORT: unexpected new rows {extra}")
    return [new[k] for k in sorted(extra, key=lambda k: k[2] != "pretrained")], len(shared)


def picks(paths: list[str]) -> tuple[dict, int, int]:
    by = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            r = json.loads(ln)
            if r["estimator"] != "parc_kt_causal":
                sys.exit(f"ABORT: {p} holds {r['estimator']}")
            by[r["target"]][r["seed"]][r["candidate"]] = r["score"]
    out, own, n = {}, 0, 0
    for t in sorted(by):
        if sorted(by[t]) != [1, 2, 42] or any(len(c) != 8 for c in by[t].values()):
            sys.exit(f"ABORT: {t} does not hold 8 candidates at seeds 42, 1 and 2")
        row = []
        for s in (42, 1, 2):
            pre = {k: v for k, v in by[t][s].items() if k != "scratch"}
            top = max(pre, key=lambda k: (pre[k], k))
            src = top.removeprefix("edubert_").removesuffix("_pretrain_full_encoder.pt")
            row.append("own" if src == t else src)
            own += src == t
            n += 1
        out[t] = row
    if len(out) != 7:
        sys.exit(f"ABORT: {len(out)} targets, expected 7")
    return out, own, n


def nlp_check(check: str, results: str) -> int:
    text = Path(results).read_text()
    i = text.find("### 13.1 Airline added")
    sec = text[i:text.find("\n_End of consolidated results._", i)]
    want = {}
    for ln in sec.splitlines():
        c = [x.strip() for x in ln.strip("|").split("|")]
        if ln.startswith("| ") and len(c) == 9 and c[0] in ("hscore", "hscore_shrunk", "logme") \
                and c[1] in ("frozen", "tuned") and c[5].endswith("/30"):
            want[(c[0], c[1])] = c
    if len(want) != 6:
        sys.exit(f"ABORT: found {len(want)} of the six 13.1 estimator rows in {results}")
    got = {(r["estimator"], r["regime"]): r for r in tsv(check)}
    for k, c in want.items():
        r = got.get(k)
        if r is None:
            sys.exit(f"ABORT: {check} lacks {k}")
        mine = [k[0], k[1], f"{float(r['rho']):+.4f}", f"{float(r['tau']):+.4f}",
                f"{float(r['pair_acc']):.4f}", f"{r['top1']}/{r['rankings']}",
                f"{r['tied_with_best']}/{r['rankings']}",
                f"{float(r['mean_regret']):.2f} / {float(r['max_regret']):.2f}",
                f"{float(r['seed_stability']):.4f}"]
        if mine != c:
            sys.exit(f"ABORT: {k} does not reproduce RESULTS.md 13.1: {mine} against {c}")
    return len(want)


def f4(x: str, sign: bool = True) -> str:
    return f"{float(x):+.4f}" if sign else f"{float(x):.4f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--kt-summary", required=True)
    ap.add_argument("--old-summary", required=True)
    ap.add_argument("--kt-scores", nargs="+", required=True)
    ap.add_argument("--nlp-summary", required=True)
    ap.add_argument("--nlp-check", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows, shared = kt_rows(a.kt_summary, a.old_summary)
    per, own, n = picks(a.kt_scores)
    nlp = {r["regime"]: r for r in tsv(a.nlp_summary) if r["estimator"] == "parc"}
    if sorted(nlp) != ["frozen", "tuned"]:
        sys.exit("ABORT: the NLP summary lacks the frozen and tuned PARC rows")
    checked = nlp_check(a.nlp_check, a.results)
    L = ["# PARC as an additional baseline", "",
         "Settings fixed in docs/parc_design.md before any score: probe of 500 with at least two "
         "per class, PCA to 32, no depth heuristic. Evaluated with the existing evaluators.", "",
         "## StudentBERT (7 targets x 3 estimator seeds)", "",
         "| view | rankings | rho | tau | decisive pairs | top-1 | tied with best | mean regret | "
         "max regret | negative picks | seed stability |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    def frac(v: str, n: str) -> str:
        return v if "/" in v else f"{v}/{n}"

    for r in rows:
        L.append(f"| {r['view']} | {r['rankings']} | {f4(r['rho'])} | {f4(r['tau'])} | "
                 f"{f4(r['pair_acc_decisive'], False)} | {frac(r['top1'], r['rankings'])} | "
                 f"{frac(r['equivalent'], r['rankings'])} | "
                 f"{float(r['mean_regret']):.4f} | {float(r['max_regret']):.4f} | "
                 f"{r['negative_choices']} | {f4(r['seed_stability_tau'], False)} |")
    L += ["", f"Own-encoder picks: {own}/{n}. PARC's top pretrained candidate per target at seeds "
          "42, 1 and 2: " + "; ".join(f"{t} {', '.join(v)}" for t, v in per.items()) + ".", "",
          f"Check: the other {shared} rows of the same evaluation equal "
          "benchmark_final/estimators_summary.tsv (RESULTS.md 12.7) exactly.", "",
          "## NLP benchmark (10 settings x 3 seeds)", "",
          "| gold | rankings | rho | tau | pairwise | top-1 | tied with best | mean regret | "
          "max regret | seed stability |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in ("frozen", "tuned"):
        r = nlp[g]
        L.append(f"| {g} | {r['rankings']} | {f4(r['rho'])} | {f4(r['tau'])} | "
                 f"{f4(r['pair_acc'], False)} | {r['top1']}/{r['rankings']} | "
                 f"{r['tied_with_best']}/{r['rankings']} | {float(r['mean_regret']):.2f} | "
                 f"{float(r['max_regret']):.2f} | {f4(r['seed_stability'], False)} |")
    L += ["", f"Check: the same evaluator on the 13.1 score file reproduces all {checked} "
          "estimator rows of RESULTS.md 13.1."]
    out = Path(a.out)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out.name} md5 {hashlib.md5(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
