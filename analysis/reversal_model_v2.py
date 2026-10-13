from __future__ import annotations

"""Reversal-aware pipeline, version 2: predict whether each alternative overtakes H-score's pick.

Prof. Hazra's feedback on version 1 (RESULTS.md 12.13): the pair-level model did not improve the
recommendation, so the next version must target the final decision. Her seven points, and how this
script meets each:

  1. Rows: only the pairs headed by H-score's top-ranked candidate (frozen_rank_a == 1), one per
     alternative: 7 per recommendation, 147 in all.
  2. Target: does the alternative overtake the pick after full fine-tuning (label reversal) against
     stable; near-ties are predicted but never fitted, as in version 1.
  3. Recommendation risk: the pairwise risks are combined by noisy-OR, 1 - prod(1 - p), the
     probability that at least one alternative overtakes the pick if the pairs were independent.
     The largest pairwise risk (version 1's rule) is reported beside it.
  4. Baselines: the all-pairs model of version 1 exactly as reported in 12.13, and the same
     all-pairs model with version 2's features, so that the effect of the new target is separated
     from the effect of the new features; plus always following H-score and the frozen margin.
  5. Features: frozen, early fine-tuning and history; the layer group is dropped. Early
     fine-tuning alone is the second variant.
  6. Grouped leave-one-target-out; the penalty and the threshold are chosen inside each fold from
     the training targets only. The threshold maximizes Youden's J (reversals caught minus false
     alarms, as rates) over the training targets' out-of-fold recommendation risks; ties go to the
     higher threshold. A fixed threshold of 0.5 is reported as well.
  7. Evaluation on recommendations: reversals caught, false alarms, selective regret, coverage,
     AURC, and calibration (Brier score and bins with counts).

Fixed before any version-2 result was seen (design freeze: the commit that adds this file, made
before its first run; the only real-data run before it reproduced 12.13's all-pairs baseline,
followed 16, caught 1, false alarms 4, selective regret 0.0015): the primary comparison is
the head-pair model with frozen, early and history features, noisy-OR risk and the fold-chosen
threshold, against always following H-score and against the all-pairs model of 12.13. Every other
row of the report is secondary. With 7 targets and 21 recommendations every estimate is coarse.

  PYTHONPATH=. python analysis/reversal_model_v2.py --pairs benchmark_final/reversal_pairs.tsv \
      --out benchmark_final/reversal_model_v2
"""

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import reversal_model as rm

FROZEN_V2 = ["margin_hscore_z", "logme_agrees", "frozen_rank_b", "a_in_domain", "b_in_domain",
             "b_scratch"]
FEATURES = {
    "head: frozen + early + history": FROZEN_V2 + rm.GROUPS["early"] + rm.GROUPS["history"],
    "head: early only": rm.GROUPS["early"],
}
ALLPAIRS = {
    "all pairs: version 1 (12.13)": [c for g in rm.GROUPS.values() for c in g],
    "all pairs: version 2 features": (rm.GROUPS["frozen"] + rm.GROUPS["early"]
                                      + rm.GROUPS["history"]),
}
PRIMARY = "head: frozen + early + history"
FIXED = 0.5
BINS = (0.0, 0.25, 0.5, 0.75, 1.0)


def is_head(r: dict) -> bool:
    return r["frozen_rank_a"] == "1"


def noisy_or(ps: list[float]) -> float:
    return 1.0 - math.prod(1.0 - p for p in ps)


def fit_predict(train: list[dict], test: list[dict], cols: list[str], label: str,
                lam: float) -> dict:
    tr = rm.fit_rows(train, label)
    y = np.array([r[label] == "reversal" for r in tr], float)
    if len(set(y)) < 2:
        return {rm.key(r): float(y.mean()) if len(y) else 0.0 for r in test}
    m = rm.fit(rm.matrix(tr, cols), y, lam)
    return {rm.key(r): float(p) for r, p in zip(test, rm.predict(m, rm.matrix(test, cols)))}


def rec_rows(head: list[dict], pred: dict, label: str) -> list[dict]:
    by = defaultdict(list)
    for r in head:
        by[(r["target"], r["est_seed"])].append(r)
    out = []
    for (t, s), rs in sorted(by.items()):
        ps = [pred[rm.key(r)] for r in rs]
        rev = [r["cand_b"] for r in rs if r[label] == "reversal"]
        amb = any(r[label] == "ambiguous" for r in rs) and not rev
        runner = [r for r in rs if r["frozen_rank_b"] == "2"][0]
        out.append({"target": t, "est_seed": s, "selected": rs[0]["cand_a"],
                    "risk_noisy_or": noisy_or(ps), "risk_max": max(ps),
                    "reversed": int(bool(rev)), "reversed_by": ",".join(rev),
                    "ambiguous": int(amb), "regret": float(rs[0]["target_regret_practical"]),
                    "margin_to_runner_up_z": float(runner["margin_hscore_z"])})
    return out


def youden(recs: list[dict], risk: str) -> float:
    """Threshold that maximizes TPR - FPR on these recommendations; ties go to the higher one."""
    use = [r for r in recs if not r["ambiguous"]]
    pos = [r for r in use if r["reversed"]]
    neg = [r for r in use if not r["reversed"]]
    if not pos or not neg:
        return FIXED
    best, best_j = FIXED, -math.inf
    for tau in sorted({r[risk] for r in use} | {FIXED, 1.0}):
        tpr = sum(r[risk] > tau for r in pos) / len(pos)
        fpr = sum(r[risk] > tau for r in neg) / len(neg)
        j = tpr - fpr
        if j > best_j or (j == best_j and tau > best):
            best, best_j = tau, j
    return best


def loto(rows: list[dict], cols: list[str], label: str, head_only: bool) -> tuple[dict, dict]:
    """Out-of-fold head-pair predictions, and per held-out target the penalty and thresholds."""
    pool = [r for r in rows if is_head(r)] if head_only else rows
    pred, fold = {}, {}
    targets = sorted({r["target"] for r in rows})
    for t in targets:
        train = [r for r in pool if r["target"] != t]
        lam = rm.choose_lambda(train, cols, label)
        pred.update(fit_predict(train, [r for r in rows if r["target"] == t and is_head(r)],
                                cols, label, lam))
        inner = {}
        for u in sorted({r["target"] for r in train}):
            inner.update(fit_predict([r for r in train if r["target"] != u],
                                     [r for r in rows if r["target"] == u and is_head(r)],
                                     cols, label, lam))
        irecs = rec_rows([r for r in rows if r["target"] in {k[0] for k in inner} and is_head(r)],
                         inner, label)
        fold[t] = {"lam": lam, "tau_noisy_or": youden(irecs, "risk_noisy_or"),
                   "tau_max": youden(irecs, "risk_max")}
    return pred, fold


def evaluate(recs: list[dict], risk: str, fold: dict | None) -> dict:
    use = [r for r in recs if not r["ambiguous"]]
    out = {"n": len(use), "reversed": sum(r["reversed"] for r in use)}
    rules = {"fixed 0.5": lambda r: FIXED}
    if fold is not None:
        rules["fold-chosen"] = lambda r: fold[r["target"]]["tau_" + risk.removeprefix("risk_")]
    for name, tau in rules.items():
        ab = [r for r in use if r[risk] > tau(r)]
        fol = [r for r in use if r[risk] <= tau(r)]
        out[name] = {"followed": len(fol), "caught": sum(r["reversed"] for r in ab),
                     "false_alarms": sum(not r["reversed"] for r in ab),
                     "selective_regret": float(np.mean([r["regret"] for r in fol]))
                     if fol else float("nan")}
    y = np.array([r["reversed"] for r in use], float)
    p = np.array([r[risk] for r in use])
    out["aurc"] = rm.aurc(use, lambda r: r[risk])
    out["brier"] = float(np.mean((p - y) ** 2))
    out["bins"] = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & ((p < hi) | (hi == 1.0))
        out["bins"].append((lo, hi, int(m.sum()), float(p[m].mean()) if m.any() else float("nan"),
                            float(y[m].mean()) if m.any() else float("nan")))
    return out


def f4(x) -> str:
    return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.4f}"


def run_all(rows: list[dict], label: str) -> dict:
    res = {}
    for name, cols in {**FEATURES, **ALLPAIRS}.items():
        pred, fold = loto(rows, cols, label, head_only=name.startswith("head"))
        recs = rec_rows([r for r in rows if is_head(r)], pred, label)
        res[name] = {"pred": pred, "fold": fold, "recs": recs,
                     "noisy_or": evaluate(recs, "risk_noisy_or", fold),
                     "max": evaluate(recs, "risk_max", fold)}
    return res


def report(main: dict, robust: dict, files: dict) -> str:
    L = ["# Reversal-risk model, version 2: the H-score recommendation as the target", "",
         f"Primary (fixed before results): {PRIMARY}, noisy-OR risk, fold-chosen threshold.", ""]
    for title, res in (("Main label", main), ("Robustness label (Algebra 2006 pair ambiguous)",
                                             robust)):
        recs0 = next(iter(res.values()))["recs"]
        use0 = [r for r in recs0 if not r["ambiguous"]]
        n, nrev = len(use0), sum(r["reversed"] for r in use0)
        mean_reg = float(np.mean([r["regret"] for r in use0]))
        marg = rm.aurc(use0, lambda r: -r["margin_to_runner_up_z"])
        L += [f"## {title}", "", f"Recommendations judged: {n}, of which {nrev} reversed. "
              f"Always following H-score: mean regret {f4(mean_reg)} (also the AURC of a random "
              f"order); frozen margin AURC {f4(marg)}.", "",
              "| model | risk | threshold | followed | caught | false alarms | selective regret |"
              " AURC | Brier |", "|---|---|---|---|---|---|---|---|---|"]
        for name, r in res.items():
            for agg in ("noisy_or", "max"):
                e = r[agg]
                for rule in ("fold-chosen", "fixed 0.5"):
                    s = e[rule]
                    L.append(f"| {name} | {agg.replace('_', '-')} | {rule} | {s['followed']}/{n} | "
                             f"{s['caught']}/{nrev} | {s['false_alarms']} | "
                             f"{f4(s['selective_regret'])} | {f4(e['aurc'])} | {f4(e['brier'])} |")
        p = res[PRIMARY]
        L += ["", f"Primary model, fold-chosen thresholds (noisy-OR): "
              + ", ".join(f"{t} {f4(v['tau_noisy_or'])}" for t, v in sorted(p["fold"].items()))
              + "; penalties: " + ", ".join(f"{t} {v['lam']}" for t, v in sorted(p["fold"].items()))
              + ".", "", "| primary, noisy-OR risk | recommendations | mean predicted | "
              "observed reversal rate |", "|---|---|---|---|"]
        for lo, hi, k, mp, ob in p["noisy_or"]["bins"]:
            L.append(f"| {lo:.2f} to {hi:.2f} | {k} | {f4(mp)} | {f4(ob)} |")
        L += ["", "| target | seed | selected | noisy-OR risk | fold threshold | abstain | "
              "reversed |", "|---|---|---|---|---|---|---|"]
        for r in p["recs"]:
            tau = p["fold"][r["target"]]["tau_noisy_or"]
            L.append(f"| {r['target']} | {r['est_seed']} | {r['selected']} | "
                     f"{f4(r['risk_noisy_or'])} | {f4(tau)} | {int(r['risk_noisy_or'] > tau)} | "
                     f"{r['reversed']}{' (ambiguous)' if r['ambiguous'] else ''} |")
        L.append("")
    L += ["With 7 targets and 21 recommendations every estimate here is coarse.", "",
          "Files: " + ", ".join(f"{k} {v}" for k, v in files.items())]
    return "\n".join(L) + "\n"


def write_tsv(path: Path, rows: list[dict]) -> str:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    return hashlib.md5(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True, help="prefix for the _recs, _pairs, _report files")
    a = ap.parse_args(argv)
    rows = rm.load(a.pairs)
    main_res = run_all(rows, "label_main")
    robust = run_all(rows, "label_robust")
    out = Path(a.out)
    recs, prs = [], []
    for lab, res in (("main", main_res), ("robust", robust)):
        for name, r in res.items():
            for rec in r["recs"]:
                tau = r["fold"][rec["target"]]["tau_noisy_or"]
                recs.append({"label": lab, "model": name, **rec, "tau_fold_noisy_or": tau})
            for k, p in sorted(r["pred"].items()):
                prs.append({"label": lab, "model": name, "target": k[0], "est_seed": k[1],
                            "cand_a": k[2], "cand_b": k[3], "p_overtake": p})
    files = {out.name + "_recs.tsv": write_tsv(out.with_name(out.name + "_recs.tsv"), recs),
             out.name + "_pairs.tsv": write_tsv(out.with_name(out.name + "_pairs.tsv"), prs)}
    rp = out.with_name(out.name + "_report.md")
    rp.write_text(report(main_res, robust, files))
    e = main_res[PRIMARY]["noisy_or"]["fold-chosen"]
    print(f"primary: followed {e['followed']}, caught {e['caught']}, false alarms "
          f"{e['false_alarms']}, selective regret {f4(e['selective_regret'])}; wrote "
          f"{', '.join(files)} and {rp.name} md5 {hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
