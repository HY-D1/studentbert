from __future__ import annotations

"""Reversal-aware pipeline, steps 6 to 9: reversal risk, held-out evaluation, reliability, abstain.

Input: benchmark_final/reversal_pairs.tsv from analysis/reversal_table.py (one row per target,
estimator seed and frozen-ordered candidate pair). Prof. Hazra's 2026-10-08 decisions apply: pairs
as rows, every pair of a held-out target held out together, near-ties their own class, the
benchmark label for the main analysis and Algebra 2006 ambiguous in the robustness analysis.

Step 6, model. L2-regularized logistic regression for P(reversal), fitted on the stable and reversal
pairs of the training targets (near-ties are neither, so they are predicted but never fitted).
Features are standardized with training-fold statistics only. The penalty is chosen from a fixed
grid by an inner leave-one-target-out over the training targets (mean log-loss). numpy and scipy
only.

Step 7, evaluation. Grouped leave-one-target-out: for each of the 7 targets, fit on the other 6 and
predict every pair of the held-out target. Pair-level: AUC and Brier on stable against reversal
pairs, calibration bins with counts, and the mean predicted risk of the near-tie pairs.

Step 8, reliability outputs, one per target and estimator seed: the frozen recommendation (H-score's
top pick), its reversal probability (the largest predicted risk over the pairs it heads), the
confidence (1 minus that), and an abstain flag at a threshold of 0.5, fixed before any result.

Step 9, selective rule. Always follow H-score, against follow only when the risk is at most 0.5
(abstain otherwise): coverage, mean regret of the followed recommendations, reversals caught and
false alarms. The risk-coverage curve and its area (AURC, lower is better) compare the model's risk
with the frozen margin to the runner-up (the existing confidence signal) and with an oracle.

Also: the robustness run (Algebra 2006's ambiguous pair left out of fitting and its recommendations
left out of the selective counts) and a feature-group note (each group alone, each group dropped).
With 7 targets every calibration estimate is coarse; the report says so and prints the counts.

  PYTHONPATH=. python analysis/reversal_model.py --pairs benchmark_final/reversal_pairs.tsv \
      --out benchmark_final/reversal_model
"""

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

GROUPS = {
    "frozen": ["margin_hscore_z", "logme_agrees", "frozen_rank_a", "frozen_rank_b",
               "pair_has_frozen_pick", "a_in_domain", "b_in_domain", "a_scratch", "b_scratch"],
    "early": ["val_diff_e1", "val_diff_e2", "val_diff_e3", "slope_diff", "rank_gain_a",
              "rank_gain_b", "gap_leader_e3_a", "gap_leader_e3_b", "b_closing"],
    "history": ["hist_avg_rank_diff", "hist_mean_gain_diff", "hist_pos_freq_diff",
                "hist_top1_freq_diff", "hist_mean_regret_diff", "hist_worst_regret_diff"],
    "layers": ["layer_agreement", "mid_order_agrees", "mid_rank_shift_a", "mid_rank_shift_b",
               "reset_delta_a", "reset_delta_b", "order_after_reset_agrees"],
}
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)
THRESHOLD = 0.5
BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def load(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    need = {c for g in GROUPS.values() for c in g} | {"target", "est_seed", "cand_a", "cand_b",
                                                       "label_main", "label_robust",
                                                       "target_regret_practical",
                                                       "frozen_rank_a", "frozen_rank_b"}
    missing = need - set(rows[0])
    if missing:
        sys.exit(f"ABORT: {path} lacks columns {sorted(missing)}")
    return rows


def matrix(rows: list[dict], cols: list[str]) -> np.ndarray:
    return np.array([[float(r[c]) for c in cols] for r in rows], dtype=float)


def fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple:
    """Standardize on X, then fit L2 logistic (intercept unpenalized); returns w, b, mu, sd."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    n, p = Z.shape

    def f(theta):
        w, b = theta[:p], theta[p]
        z = Z @ w + b
        loss = np.mean(np.logaddexp(0.0, z) - y * z) + lam * (w @ w) / (2 * n)
        s = 1.0 / (1.0 + np.exp(-z))
        g = np.empty(p + 1)
        g[:p] = Z.T @ (s - y) / n + lam * w / n
        g[p] = np.mean(s - y)
        return loss, g

    res = minimize(f, np.zeros(p + 1), jac=True, method="L-BFGS-B")
    return res.x[:p], float(res.x[p]), mu, sd


def predict(model: tuple, X: np.ndarray) -> np.ndarray:
    w, b, mu, sd = model
    z = ((X - mu) / sd) @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = sum((pp > neg).sum() + 0.5 * (pp == neg).sum() for pp in pos)
    return float(gt / (len(pos) * len(neg)))


def fit_rows(rows: list[dict], label: str) -> list[dict]:
    """Rows that enter fitting: stable or reversal under the given label."""
    return [r for r in rows if r[label] in ("stable", "reversal")]


def choose_lambda(rows: list[dict], cols: list[str], label: str) -> float:
    targets = sorted({r["target"] for r in rows})
    best, best_loss = LAMBDAS[0], math.inf
    for lam in LAMBDAS:
        losses = []
        for t in targets:
            tr = fit_rows([r for r in rows if r["target"] != t], label)
            te = fit_rows([r for r in rows if r["target"] == t], label)
            if not te or len({r[label] for r in tr}) < 2:
                continue
            m = fit(matrix(tr, cols), np.array([r[label] == "reversal" for r in tr], float), lam)
            y = np.array([r[label] == "reversal" for r in te], float)
            losses.append(logloss(y, predict(m, matrix(te, cols))))
        if losses and np.mean(losses) < best_loss:
            best, best_loss = lam, float(np.mean(losses))
    return best


def loto(rows: list[dict], cols: list[str], label: str) -> tuple[dict, dict]:
    """Out-of-fold P(reversal) for every row, and the penalty chosen in each fold."""
    pred, lams = {}, {}
    for t in sorted({r["target"] for r in rows}):
        train = [r for r in rows if r["target"] != t]
        lam = choose_lambda(train, cols, label)
        tr = fit_rows(train, label)
        m = fit(matrix(tr, cols), np.array([r[label] == "reversal" for r in tr], float), lam)
        test = [r for r in rows if r["target"] == t]
        for r, p in zip(test, predict(m, matrix(test, cols))):
            pred[(r["target"], r["est_seed"], r["cand_a"], r["cand_b"])] = float(p)
        lams[t] = lam
    return pred, lams


def key(r: dict) -> tuple:
    return (r["target"], r["est_seed"], r["cand_a"], r["cand_b"])


def recommendations(rows: list[dict], pred: dict, label: str) -> list[dict]:
    recs = []
    by = defaultdict(list)
    for r in rows:
        by[(r["target"], r["est_seed"])].append(r)
    for (t, s), rs in sorted(by.items()):
        head = [r for r in rs if r["frozen_rank_a"] == "1"]
        pick = head[0]["cand_a"]
        risk = max(pred[key(r)] for r in head)
        runner = [r for r in head if r["frozen_rank_b"] == "2"][0]
        regret = float(head[0]["target_regret_practical"])
        reversed_by = [r["cand_b"] for r in head if r[label] == "reversal"]
        ambiguous = any(r[label] == "ambiguous" for r in head) and not reversed_by
        recs.append({"target": t, "est_seed": s, "selected": pick, "reversal_probability": risk,
                     "confidence": 1.0 - risk, "abstain": int(risk > THRESHOLD),
                     "margin_to_runner_up_z": float(runner["margin_hscore_z"]), "regret": regret,
                     "reversed": int(bool(reversed_by)), "reversed_by": ",".join(reversed_by),
                     "ambiguous": int(ambiguous)})
    return recs


def aurc(recs: list[dict], score) -> float:
    order = sorted(recs, key=score)
    total, out = 0.0, 0.0
    for k, r in enumerate(order, 1):
        total += r["regret"]
        out += total / k
    return out / len(order)


def selective(recs: list[dict]) -> dict:
    use = [r for r in recs if not r["ambiguous"]]
    fol = [r for r in use if not r["abstain"]]
    # A recommendation counts as reversed only if a pair it heads is a reversal; a pick tied with
    # the best (a near-tie) is not a miss even when its regret is a little above zero.
    caught = sum(1 for r in use if r["abstain"] and r["reversed"])
    false = sum(1 for r in use if r["abstain"] and not r["reversed"])
    missed = sum(1 for r in fol if r["reversed"])
    return {"n": len(use), "always_regret": float(np.mean([r["regret"] for r in use])),
            "followed": len(fol),
            "followed_regret": float(np.mean([r["regret"] for r in fol])) if fol else float("nan"),
            "caught": caught, "false_alarms": false, "missed": missed,
            "reversed": sum(r["reversed"] for r in use),
            "aurc_model": aurc(use, lambda r: r["reversal_probability"]),
            "aurc_margin": aurc(use, lambda r: -r["margin_to_runner_up_z"]),
            "aurc_oracle": aurc(use, lambda r: r["regret"]),
            "aurc_random": float(np.mean([r["regret"] for r in use]))}


def pair_metrics(rows: list[dict], pred: dict, label: str) -> dict:
    fr = fit_rows(rows, label)
    y = np.array([r[label] == "reversal" for r in fr], float)
    p = np.array([pred[key(r)] for r in fr])
    ties = [pred[key(r)] for r in rows if r[label] == "near_tie"]
    per_t = {}
    for t in sorted({r["target"] for r in fr}):
        idx = [i for i, r in enumerate(fr) if r["target"] == t]
        per_t[t] = auc(y[idx], p[idx])
    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & ((p < hi) | (hi == 1.0))
        bins.append((lo, hi, int(m.sum()), float(p[m].mean()) if m.any() else float("nan"),
                     float(y[m].mean()) if m.any() else float("nan")))
    return {"n_stable": int((y == 0).sum()), "n_reversal": int((y == 1).sum()),
            "auc": auc(y, p), "brier": float(np.mean((p - y) ** 2)), "per_target_auc": per_t,
            "bins": bins, "near_tie_mean_risk": float(np.mean(ties)) if ties else float("nan"),
            "n_near_tie": len(ties)}


def run(rows: list[dict], cols: list[str], label: str) -> dict:
    pred, lams = loto(rows, cols, label)
    recs = recommendations(rows, pred, label)
    return {"pred": pred, "lams": lams, "pairs": pair_metrics(rows, pred, label), "recs": recs,
            "sel": selective(recs)}


def f4(x: float) -> str:
    return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.4f}"


def report(main: dict, robust: dict, groups: dict, files: dict) -> str:
    L = ["# Reversal-risk model: leave-one-target-out results", ""]
    for name, res in (("Main analysis (benchmark label)", main),
                      ("Robustness (Algebra 2006 pair ambiguous)", robust)):
        pm, sel = res["pairs"], res["sel"]
        L += [f"## {name}", "",
              f"Pairs fitted: {pm['n_stable']} stable, {pm['n_reversal']} reversal; near-tie pairs "
              f"predicted only ({pm['n_near_tie']}, mean predicted risk "
              f"{f4(pm['near_tie_mean_risk'])}). "
              f"Penalty per held-out target: "
              + ", ".join(f"{t} {lam}" for t, lam in sorted(res["lams"].items())) + ".", "",
              f"Pair-level AUC {f4(pm['auc'])}, Brier {f4(pm['brier'])}; per held-out target AUC: "
              + ", ".join(f"{t} {f4(a)}" for t, a in pm["per_target_auc"].items()) + ".", "",
              "| predicted risk | pairs | mean predicted | observed reversal rate |",
              "|---|---|---|---|"]
        for lo, hi, n, mp, ob in pm["bins"]:
            L.append(f"| {lo:.1f} to {hi:.1f} | {n} | {f4(mp)} | {f4(ob)} |")
        L += ["", f"Recommendations judged: {sel['n']} (target x estimator seed).", "",
              "| rule | followed | mean regret of followed | reversals caught | false alarms | "
              "missed |", "|---|---|---|---|---|---|",
              f"| always follow H-score | {sel['n']} | {f4(sel['always_regret'])} | 0 | 0 | "
              f"{sel['reversed']} |",
              f"| follow when risk <= {THRESHOLD} | {sel['followed']} | "
              f"{f4(sel['followed_regret'])} | {sel['caught']} | {sel['false_alarms']} | "
              f"{sel['missed']} |", "",
              f"AURC (lower is better): model {f4(sel['aurc_model'])}, frozen margin "
              f"{f4(sel['aurc_margin'])}, random {f4(sel['aurc_random'])}, oracle "
              f"{f4(sel['aurc_oracle'])}.", ""]
    L += ["## Feature groups (main label)", "",
          "| features | pair AUC | Brier | AURC |", "|---|---|---|---|"]
    for name, res in groups.items():
        L.append(f"| {name} | {f4(res['pairs']['auc'])} | {f4(res['pairs']['brier'])} | "
                 f"{f4(res['sel']['aurc_model'])} |")
    L += ["", "With 7 targets and 21 recommendations every calibration and selective estimate is "
          "coarse; read the counts, not only the rates.", "",
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
    ap.add_argument("--out", required=True, help="prefix for _pairs.tsv, _recs.tsv, _report.md")
    a = ap.parse_args(argv)
    rows = load(a.pairs)
    allc = [c for g in GROUPS.values() for c in g]
    main_res = run(rows, allc, "label_main")
    robust = run(rows, allc, "label_robust")
    groups = {"all": main_res}
    for g, cols in GROUPS.items():
        groups[f"{g} only"] = run(rows, cols, "label_main")
        groups[f"all but {g}"] = run(rows, [c for c in allc if c not in cols], "label_main")
    out = Path(a.out)
    prow = [{**{k: r[k] for k in ("target", "est_seed", "cand_a", "cand_b", "label_main",
                                  "label_robust")},
             "p_reversal_main": main_res["pred"][key(r)],
             "p_reversal_robust": robust["pred"][key(r)]} for r in rows]
    files = {out.name + "_pairs.tsv": write_tsv(out.with_name(out.name + "_pairs.tsv"), prow),
             out.name + "_recs.tsv": write_tsv(out.with_name(out.name + "_recs.tsv"),
                                               main_res["recs"])}
    rp = out.with_name(out.name + "_report.md")
    rp.write_text(report(main_res, robust, groups, files))
    s = main_res["sel"]
    print(f"pair AUC {f4(main_res['pairs']['auc'])}; followed {s['followed']}/{s['n']}, caught "
          f"{s['caught']}, false alarms {s['false_alarms']}; AURC model {f4(s['aurc_model'])} vs "
          f"margin {f4(s['aurc_margin'])}; wrote {', '.join(files)} and {rp.name} md5 "
          f"{hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
