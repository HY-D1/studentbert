from __future__ import annotations

"""Reversal-aware pipeline, steps 1 to 5: one row per frozen-ordered candidate pair.

Prof. Hazra's 2026-10-06 task, with her 2026-10-08 decisions: candidate pairs as rows, all pairs of
a target held out together; near-ties kept as their own class; the benchmark tie rule for the main
label, with Algebra 2006 treated as ambiguous in the robustness label; the 7 StudentBERT KT targets.

Rows. For every target, estimator seed (42, 1, 2) and pair of the 8 candidates (7 encoders plus
scratch), a is the candidate H-score ranks higher on that seed and b the lower one, so each row asks
whether full fine-tuning keeps or reverses the frozen order: 7 x 3 x 28 = 588 rows.

Label (step 1). d = gold(a) - gold(b), the mean over the fine-tuning seeds both have of the Track B
test AUC (executions.tsv, W&B full precision). near_tie when the paired bootstrap interval of the
per-seed differences includes zero or |d| <= 0.001, the benchmark's own rule and margin
(analysis/build_transfer_benchmark.py boot_mean_ci, 20,000 resamples, random.Random(0)); otherwise
stable when d > 0, reversal when d < 0. label_robust equals label_main except for the pairs in
AMBIGUOUS, whose order the evaluation learners do not separate (RESULTS.md 12.11). pair_regret is
what following the frozen order of the pair costs, max(0, -d); target_regret_* is what following
H-score's top pick costs on that target and seed, in the pretrained and practical views.

Features. Nothing below reads the held-out target's gold or any test split:
  frozen (step 5)   H-score and LogME at the final layer, their margins, the margin over the
                    spread of the 8 candidates' H-scores on that target and seed, ranks
  early (step 2)    validation AUC at epochs 1 to 3 of the fine-tune with the same seed as the
                    estimator, slopes, ranks among the 8, rank change, gap to the epoch-3 leader
  history (step 3)  each candidate's behaviour as a foreign source on the other 6 targets (the
                    held-out target and the candidate's own dataset excluded): mean rank, mean
                    gain over scratch, share of positive transfer, share of top-1, mean and worst
                    regret
  layers (step 4)   H-score at layer 3, how often layers 1 to 6 keep the pair's order, and for the
                    in-domain encoder the H-score change when its skill table is reset

  PYTHONPATH=. python analysis/reversal_table.py --executions benchmark_final/executions.tsv \
      --gold-cells benchmark_final/gold_cells.tsv --early benchmark_final/early_ft_epochs.tsv \
      --layers tg1_layerdiag7_kt_*.jsonl --out benchmark_final/reversal_pairs.tsv
"""

import argparse
import csv
import hashlib
import json
import random
import statistics as st
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from analysis.build_transfer_benchmark import boot_mean_ci

TARGETS = ("assist2017", "ednet", "junyi", "algebra2005", "bridge2006", "assist2009", "algebra2006")
EST_SEEDS = (42, 1, 2)
N_CAND = 8
MARGIN = 0.001
BOOTS = 20000
EARLY = (1, 2, 3)
MID, FINAL = 3, 6
AMBIGUOUS = {("algebra2006", frozenset({"src:algebra2006", "src:junyi"}))}


def cand_of(path_or_name: str) -> str:
    """Candidate id as executions.tsv writes it."""
    n = Path(path_or_name).name
    if n == "scratch":
        return "scratch"
    return "src:" + n.removeprefix("edubert_").removesuffix("_pretrain_full_encoder.pt")


def load_gold(path: str) -> dict:
    """gold[target][candidate][finetune_seed] = Track B N=3000 test AUC."""
    gold: dict = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r["track"], r["budget"], r["metric"], r["valid_for_primary_analysis"]) != \
                    ("B", "n3000", "test_auc", "yes") or r["target_dataset"] not in TARGETS:
                continue
            s = int(r["finetune_seed"])
            if s in gold[r["target_dataset"]][r["candidate"]]:
                sys.exit(f"ABORT: two primary rows for {r['target_dataset']} {r['candidate']} {s}")
            gold[r["target_dataset"]][r["candidate"]][s] = float(r["value"])
    for t in TARGETS:
        if len(gold[t]) != N_CAND or any(len(v) != 6 for v in gold[t].values()):
            sys.exit(f"ABORT: {t} gold is not {N_CAND} candidates x 6 seeds")
    return gold


def load_signs(path: str) -> dict:
    """signs[target][candidate] = transfer sign of the Track B N=3000 cell (gold_cells.tsv)."""
    out: dict = defaultdict(dict)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["track"] == "B" and r["budget"] == "n3000" and r["target"] in TARGETS:
                out[r["target"]][r["candidate"]] = r["transfer_sign"] or "none"
    return out


def load_layers(paths: list[str]) -> tuple[dict, dict, dict]:
    """hs[t][seed][cand][layer], lm[t][seed][cand] at the final layer, reset[t][seed][cand]."""
    hs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    lm: dict = defaultdict(lambda: defaultdict(dict))
    reset: dict = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            r = json.loads(ln)
            t, s, c, est = r["target"], int(r["seed"]), cand_of(r["candidate"]), r["estimator"]
            rand = r["metadata"]["skill_table_randomized"]
            if est.startswith("hscore_kt_causal_L"):
                layer = int(r["metadata"]["layer"])
                if rand:
                    if layer == FINAL:
                        reset[t][s][c] = float(r["score"])
                else:
                    hs[t][s][c][layer] = float(r["score"])
            elif est == f"logme_kt_causal_L{FINAL}" and not rand:
                lm[t][s][c] = float(r["score"])
    for t in TARGETS:
        for s in EST_SEEDS:
            if len(hs[t][s]) != N_CAND or any(len(v) != FINAL + 1 for v in hs[t][s].values()):
                sys.exit(f"ABORT: layer scores for {t} seed {s} are not {N_CAND} x 7 layers")
            if len(lm[t][s]) != N_CAND or set(reset[t][s]) != {f"src:{t}"}:
                sys.exit(f"ABORT: LogME or skill-reset scores missing for {t} seed {s}")
    return hs, lm, reset


def load_early(path: str) -> dict:
    """val[t][cand][seed][epoch] for the early epochs only; the test column is never read."""
    val: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            e = int(r["epoch"])
            if e in EARLY:
                val[r["target"]][r["candidate"]][int(r["finetune_seed"])][e] = float(r["val_auc"])
    for t in TARGETS:
        for s in EST_SEEDS:
            if len([c for c in val[t] if len(val[t][c].get(s, {})) == len(EARLY)]) != N_CAND:
                sys.exit(f"ABORT: early epochs for {t} seed {s} are not complete")
    return val


def ranks(score: dict) -> dict:
    """1 = highest; ties broken by candidate id so every run gives the same order."""
    order = sorted(score, key=lambda c: (-score[c], c))
    return {c: i + 1 for i, c in enumerate(order)}


def gold_means(gold: dict) -> dict:
    return {t: {c: st.mean(v.values()) for c, v in gold[t].items()} for t in gold}


def history(gold: dict, signs: dict, held_out: str) -> dict:
    """Each candidate as a foreign source on the other targets; held_out never enters."""
    gm = gold_means(gold)
    out = {}
    cands = sorted(gm[held_out])
    for c in cands:
        own = c.removeprefix("src:") if c != "scratch" else None
        ts = [t for t in TARGETS if t not in (held_out, own)]
        rk, gain, pos, top, reg = [], [], [], [], []
        for t in ts:
            g = gm[t]
            r = ranks(g)
            best = max(g.values())
            rk.append(r[c])
            gain.append(g[c] - g["scratch"])
            pos.append(1.0 if signs[t].get(c) == "positive" else 0.0)
            top.append(1.0 if r[c] == 1 else 0.0)
            reg.append(best - g[c])
        out[c] = {"avg_rank": st.mean(rk), "mean_gain": st.mean(gain), "pos_freq": st.mean(pos),
                  "top1_freq": st.mean(top), "mean_regret": st.mean(reg), "worst_regret": max(reg),
                  "n_targets": len(ts)}
    return out


def label(gold: dict, t: str, a: str, b: str, idx: list) -> dict:
    seeds = sorted(set(gold[t][a]) & set(gold[t][b]))
    d = [gold[t][a][s] - gold[t][b][s] for s in seeds]
    mean_d = st.mean(d)
    lo, hi = boot_mean_ci(d, idx)
    if (lo <= 0 <= hi) or abs(mean_d) <= MARGIN:
        main = "near_tie"
    else:
        main = "stable" if mean_d > 0 else "reversal"
    robust = "ambiguous" if (t, frozenset({a, b})) in AMBIGUOUS else main
    return {"gold_diff": mean_d, "gold_ci_lo": lo, "gold_ci_hi": hi, "label_main": main,
            "label_robust": robust, "pair_regret": max(0.0, -mean_d)}


def build(gold: dict, signs: dict, hs: dict, lm: dict, reset: dict, val: dict) -> tuple[list, list]:
    rng = random.Random(0)
    idx = [[rng.randrange(6) for _ in range(6)] for _ in range(BOOTS)]
    gm = gold_means(gold)
    rows, targets = [], []
    for t in TARGETS:
        hist = history(gold, signs, t)
        cands = sorted(gm[t])
        src = [c for c in cands if c != "scratch"]
        for s in EST_SEEDS:
            h6 = {c: hs[t][s][c][FINAL] for c in cands}
            h3 = {c: hs[t][s][c][MID] for c in cands}
            r6, r3 = ranks(h6), ranks(h3)
            sd6 = st.pstdev(h6.values())
            e = {c: val[t][c][s] for c in cands}
            re1 = ranks({c: e[c][1] for c in cands})
            re3 = ranks({c: e[c][3] for c in cands})
            lead3 = max(e[c][3] for c in cands)
            pick_pre = max(src, key=lambda c: (h6[c], c))
            pick_all = max(cands, key=lambda c: (h6[c], c))
            reg_pre = max(gm[t][c] for c in src) - gm[t][pick_pre]
            reg_all = max(gm[t].values()) - gm[t][pick_all]
            targets.append({"target": t, "est_seed": s, "frozen_pick_pretrained": pick_pre,
                            "frozen_pick_practical": pick_all,
                            "target_regret_pretrained": reg_pre,
                            "target_regret_practical": reg_all})
            own = f"src:{t}"
            for x, y in combinations(cands, 2):
                a, b = (x, y) if r6[x] < r6[y] else (y, x)
                row = {"target": t, "est_seed": s, "cand_a": a, "cand_b": b,
                       "frozen_rank_a": r6[a], "frozen_rank_b": r6[b],
                       "pair_has_frozen_pick": int(a == pick_all),
                       "a_in_domain": int(a == own), "b_in_domain": int(b == own),
                       "a_scratch": int(a == "scratch"), "b_scratch": int(b == "scratch"),
                       "hscore_a": h6[a], "hscore_b": h6[b], "margin_hscore": h6[a] - h6[b],
                       "margin_hscore_z": (h6[a] - h6[b]) / sd6 if sd6 else 0.0,
                       "logme_a": lm[t][s][a], "logme_b": lm[t][s][b],
                       "margin_logme": lm[t][s][a] - lm[t][s][b],
                       "logme_agrees": int(lm[t][s][a] > lm[t][s][b])}
                for ep in EARLY:
                    row[f"val_e{ep}_a"], row[f"val_e{ep}_b"] = e[a][ep], e[b][ep]
                    row[f"val_diff_e{ep}"] = e[a][ep] - e[b][ep]
                row.update({"slope_a": e[a][3] - e[a][1], "slope_b": e[b][3] - e[b][1],
                            "slope_diff": (e[a][3] - e[a][1]) - (e[b][3] - e[b][1]),
                            "rank_e1_a": re1[a], "rank_e1_b": re1[b],
                            "rank_e3_a": re3[a], "rank_e3_b": re3[b],
                            "rank_gain_a": re1[a] - re3[a], "rank_gain_b": re1[b] - re3[b],
                            "gap_leader_e3_a": lead3 - e[a][3], "gap_leader_e3_b": lead3 - e[b][3],
                            "b_closing": (e[a][3] - e[b][3]) - (e[a][1] - e[b][1])})
                for k in ("avg_rank", "mean_gain", "pos_freq", "top1_freq", "mean_regret",
                          "worst_regret"):
                    row[f"hist_{k}_a"], row[f"hist_{k}_b"] = hist[a][k], hist[b][k]
                    row[f"hist_{k}_diff"] = hist[a][k] - hist[b][k]
                agree = sum(hs[t][s][a][L] > hs[t][s][b][L] for L in range(1, FINAL + 1))
                ra = {a: reset[t][s].get(a, h6[a]), b: reset[t][s].get(b, h6[b])}
                row.update({"hscore_mid_a": h3[a], "hscore_mid_b": h3[b],
                            "mid_order_agrees": int(h3[a] > h3[b]),
                            "layer_agreement": agree / FINAL,
                            "mid_rank_shift_a": r3[a] - r6[a], "mid_rank_shift_b": r3[b] - r6[b],
                            "reset_delta_a": ra[a] - h6[a], "reset_delta_b": ra[b] - h6[b],
                            "order_after_reset_agrees": int(ra[a] > ra[b])})
                row.update(label(gold, t, a, b, idx))
                row["gold_a"], row["gold_b"] = gm[t][a], gm[t][b]
                row["target_regret_pretrained"], row["target_regret_practical"] = reg_pre, reg_all
                rows.append(row)
    return rows, targets


def write(path: Path, rows: list) -> str:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    return hashlib.md5(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executions", required=True)
    ap.add_argument("--gold-cells", required=True)
    ap.add_argument("--early", required=True)
    ap.add_argument("--layers", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="pairs TSV; <stem>_targets.tsv goes beside it")
    a = ap.parse_args(argv)
    gold = load_gold(a.executions)
    hs, lm, reset = load_layers(a.layers)
    rows, targets = build(gold, load_signs(a.gold_cells), hs, lm, reset, load_early(a.early))
    out = Path(a.out)
    m1 = write(out, rows)
    tout = out.with_name(out.stem + "_targets.tsv")
    m2 = write(tout, targets)
    n = {k: sum(r["label_main"] == k for r in rows) for k in ("stable", "near_tie", "reversal")}
    reg = st.mean(r["target_regret_pretrained"] for r in targets)
    print(f"{len(rows)} pair rows {n}; H-score pretrained regret {reg:.4f} over {len(targets)} "
          f"rankings; wrote {out} md5 {m1} and {tout} md5 {m2}")


if __name__ == "__main__":
    main()
