from __future__ import annotations

# MRAP failure review, diagnostic D4 report: does a layer, or the in-domain skill table, explain
# the own-encoder pick on a target? Reads scripts/diagnose_logme.py records (LogME and, since
# tools/patches/patch_diagnose_hscore.py, H-score at every layer L0 to L6, plus the in-domain
# encoder re-scored with its skill table at the fine-tune's random start, "_skillrand") and judges
# each (estimator, layer, seed) ranking of the pretrained sources against the fine-tuned gold with
# the Task 3 annotate() and judge() (pretrained view, 0.001 margin). The skill-table variant
# replaces only the in-domain score. Seeds missing any candidate or the skill-table variant are
# listed and left out. The final layer is compared with the 7 x 7 scores of the same target,
# candidate and seed (the diagnostic runs on CPU, the 7 x 7 scores on GPU, so the difference is
# reported, not required to be zero).
#
#   PYTHONPATH=. python analysis/layer_diag_report.py --diag tg1_logmediag_kt_algebra2006.jsonl \
#       --executions benchmark_final/executions.tsv --task2 benchmark_final/tg1_task2_kt_*.jsonl \
#       --logme benchmark_final/tg1_logme_kt_7x7_*.jsonl \
#       --out benchmark_final/layer_diag_algebra2006

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

try:
    from analysis.evaluate_estimators import annotate, judge, load_gold
    from analysis.frozen_gold_report import cand, load_scores
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.evaluate_estimators import annotate, judge, load_gold
    from analysis.frozen_gold_report import cand, load_scores

BASES = ("hscore_kt_causal", "logme_kt_causal")
LAYERS = range(7)


def q(x, places: str = "0.0001", sign: bool = False) -> str:
    d = Decimal(repr(float(x))).quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def parse(path: str) -> tuple[str, dict, dict]:
    """(target, plain[(base, layer, seed)][cand] = score, rand[(base, layer, seed)] = score)."""
    plain: dict = defaultdict(dict)
    rand: dict = {}
    targets = set()
    for ln in open(path):
        r = json.loads(ln)
        e = r["estimator"]
        base, _, rest = e.partition("_L")
        if base not in BASES:
            continue
        layer = int(rest.split("_")[0])
        if layer != r["metadata"]["layer"]:
            sys.exit(f"ABORT: {e} carries layer {r['metadata']['layer']}")
        targets.add(r["target"])
        key = (base, layer, r["seed"])
        if e.endswith("_skillrand"):
            rand[key] = r["score"]
        else:
            plain[key][cand(r["candidate"])] = r["score"]
    if len(targets) != 1:
        sys.exit(f"ABORT: expected one target, found {sorted(targets)}")
    return targets.pop(), plain, rand


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--executions", required=True)
    ap.add_argument("--task2", nargs="+", required=True)
    ap.add_argument("--logme", nargs="+", required=True)
    ap.add_argument("--margin", type=float, default=0.001)
    ap.add_argument("--boots", type=int, default=20000)
    ap.add_argument("--out", required=True, help="prefix for _cells.tsv and _report.md")
    a = ap.parse_args(argv)
    t, plain, rand = parse(a.diag)
    cells, _ = load_gold(a.executions)
    cell = ("B", t, "n3000")
    srcs = sorted(c for c in cells[cell] if c.startswith("src:"))
    own = f"src:{t}"
    full = annotate(cells[cell], a.margin, a.boots)
    view = annotate({c: cells[cell][c] for c in srcs}, a.margin, a.boots)
    seeds_all = sorted({k[2] for k in plain})
    ok = [s for s in seeds_all
          if all(set(srcs) <= set(plain.get((b, L, s), {})) and (b, L, s) in rand
                 for b in BASES for L in LAYERS)]
    missing = [s for s in seeds_all if s not in ok]
    if not ok:
        sys.exit(f"ABORT: no complete seed for {t}; incomplete: {missing}")
    ref = {"hscore_kt_causal": load_scores(a.task2, "hscore_kt_causal"),
           "logme_kt_causal": load_scores(a.logme, "logme_kt_causal")}
    final_diff = {b: max(abs(plain[(b, 6, s)][c] - ref[b][(t, c, s)]) for s in ok for c in srcs)
                  for b in BASES}
    rows = []
    for b in BASES:
        for L in LAYERS:
            for s in ok:
                sc = {c: plain[(b, L, s)][c] for c in srcs}
                for variant, scores in (("trained skill table", sc),
                                        ("skill table reset", {**sc, own: rand[(b, L, s)]})):
                    j = judge(scores, view, full, a.margin)
                    rows.append({"estimator": b, "layer": L, "seed": s, "variant": variant,
                                 "pick": j["choice"], "top1": j["top1"],
                                 "tied": j["equivalent"], "regret": j["regret"], "rho": j["rho"]})
    with open(f"{a.out}_cells.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    best = max(srcs, key=lambda c: view[c]["gold"])
    L_ = [f"# Layer and skill-table diagnostic on {t} (failure review D4)", "",
          f"Complete seeds: {', '.join(map(str, ok))}; "
          f"incomplete: {', '.join(map(str, missing)) or 'none'}. "
          f"Fine-tuned best source: {best}. Final layer against the 7 x 7 scores, largest absolute "
          f"difference: H-score {final_diff['hscore_kt_causal']:.2e}, LogME "
          f"{final_diff['logme_kt_causal']:.2e}.", ""]
    for b in BASES:
        L_ += [f"## {b}", "", "| layer | variant | picks (seeds " + " ".join(map(str, ok)) + ") | "
               "top-1 | tied with best | mean regret | mean rho |", "|---|---|---|---|---|---|---|"]
        for L in LAYERS:
            for variant in ("trained skill table", "skill table reset"):
                rs = [r for r in rows if r["estimator"] == b and r["layer"] == L
                      and r["variant"] == variant]
                picks = ", ".join(r["pick"].removeprefix("src:") for r in rs)
                L_.append(f"| L{L} | {variant} | {picks} | "
                          f"{sum(r['top1'] for r in rs)}/{len(rs)} | "
                          f"{sum(r['tied'] for r in rs)}/{len(rs)} | "
                          f"{q(st.mean(r['regret'] for r in rs))} | "
                          f"{q(st.mean(r['rho'] for r in rs), sign=True)} |")
        L_ += [""]
    Path(f"{a.out}_report.md").write_text("\n".join(L_) + "\n")
    print(f"{t}: complete seeds {ok}, incomplete {missing}; final-layer diff H-score "
          f"{final_diff['hscore_kt_causal']:.2e}, LogME {final_diff['logme_kt_causal']:.2e}; "
          f"wrote {a.out}_cells.tsv and {a.out}_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
