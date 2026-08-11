#!/usr/bin/env python3
"""Leave-one-dataset-out (LODO) test: do dataset statistics predict the regime?

Replaces the recorded "n_students 5/6, pps 3/6" note in collect_all_results.py,
which has no script and no log behind it (confirmed: a grep for lodo/LODO across
all 645 logs returns nothing). This re-derives the result over all 7 datasets so
the number in the paper is reproducible.

WHAT IT DOES
  For each candidate feature, and for each held-out dataset in turn:
    1. fit a single-feature threshold rule on the other 6 datasets,
    2. predict the held-out dataset's regime,
    3. record whether the prediction was right.
  Reports x/7 per feature, plus which datasets each feature missed.

THRESHOLD FITTING (deterministic, no randomness, no tuning on the held-out set)
  Candidates are the midpoints between consecutive distinct training values.
  Both directions (higher = skill-driven, higher = correctness-driven) are tried.
  The winner maximises TRAINING accuracy; ties break on the larger margin to the
  nearest training point, then on the lower threshold. Every chosen threshold is
  printed so the fit is auditable.

WHY THE RESULT IS EXPECTED TO BE WEAK, AND WHY THAT IS THE POINT
  Only 2 of 7 datasets are correctness-driven. Holding either one out leaves a
  single correctness-driven example in training. The script reports this fold
  fragility explicitly rather than burying it in an average.

INPUTS
  No files and no GPU. The 7-row table below is transcribed from RESULTS.md
  section 0.1 (itself parsed from processed/<ds>/vocab_stats.md); the regime
  labels are from RESULTS.md section 4. num_skills was independently re-verified
  against the ablation logs (grep -o "num_skills=[0-9]*"). Both practice-per-skill
  variants are DERIVED here rather than transcribed, so the definitions are
  explicit and the 512-step model cap is applied where it bites.

USAGE (from the repo root)
    python3 analysis/lodo_regime.py
    python3 analysis/lodo_regime.py --cap 512 --out lodo_report.md
"""
import argparse

# name, students, skills, interactions, median_len, mean_len, correct_rate, regime
# Source: RESULTS.md section 0.1 (statistics) and section 4 (regime labels).
DATASETS = [
    ("assist2017",   1708,   102,     942814,  441.0,  552.0, 0.3727, "skill"),
    ("ednet",      441997,   142,   93373359,   30.0,  211.3, 0.6575, "correct"),
    ("junyi",       61442,  1326,   16164318,   87.0,  263.1, 0.7037, "correct"),
    ("algebra2005",   567,   109,     606983,  581.0, 1070.5, 0.7553, "skill"),
    ("bridge2006",   1130,   492,    1817393, 1373.0, 1608.3, 0.8322, "skill"),
    ("assist2009",   3119,   123,     454232,   40.0,  145.6, 0.6910, "skill"),
    ("algebra2006",  1310,   484,    1808472, 1168.5, 1380.5, 0.7788, "skill"),
]

FEATURE_ORDER = [
    "pps_effective",
    "pps_raw",
    "n_students",
    "n_interactions",
    "n_skills",
    "median_len",
    "mean_len",
    "correct_rate",
]


def build(cap):
    """Return [(name, {feature: value}, regime), ...]."""
    rows = []
    for name, students, skills, inter, med, mean, corr, regime in DATASETS:
        feats = {
            "n_students": float(students),
            "n_skills": float(skills),
            "n_interactions": float(inter),
            "median_len": float(med),
            "mean_len": float(mean),
            "correct_rate": float(corr),
            "pps_raw": med / skills,
            "pps_effective": min(med, cap) / skills,
        }
        rows.append((name, feats, regime))
    return rows


def fit_threshold(values, labels):
    """Best (threshold, direction, train_accuracy, n_ties) by training accuracy.

    direction +1 means "value above threshold implies skill-driven".
    """
    uniq = sorted(set(values))
    if len(uniq) == 1:
        cands = [uniq[0]]
    else:
        cands = [(a + b) / 2.0 for a, b in zip(uniq, uniq[1:])]

    best_key = None
    best = None
    ties = 0
    for t in cands:
        for d in (1, -1):
            preds = ["skill" if d * (v - t) > 0 else "correct" for v in values]
            hits = sum(1 for p, l in zip(preds, labels) if p == l)
            acc = hits / len(labels)
            margin = min(abs(v - t) for v in values)
            key = (acc, margin, -t)
            if best_key is None or key > best_key:
                best_key = key
                best = (t, d, acc)
                ties = 0
            elif key == best_key:
                ties += 1
    return best[0], best[1], best[2], ties


def predict(value, threshold, direction):
    return "skill" if direction * (value - threshold) > 0 else "correct"


def run(rows, feature):
    """LODO over every dataset for one feature."""
    folds = []
    for i, (held_name, held_feats, held_regime) in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != i]
        values = [r[1][feature] for r in train]
        labels = [r[2] for r in train]
        t, d, train_acc, ties = fit_threshold(values, labels)
        pred = predict(held_feats[feature], t, d)
        n_corr_train = sum(1 for l in labels if l == "correct")
        folds.append({
            "held": held_name,
            "truth": held_regime,
            "pred": pred,
            "ok": pred == held_regime,
            "threshold": t,
            "direction": d,
            "train_acc": train_acc,
            "ties": ties,
            "n_correct_in_train": n_corr_train,
            "value": held_feats[feature],
        })
    return folds


def in_sample_accuracy(rows, feature):
    values = [r[1][feature] for r in rows]
    labels = [r[2] for r in rows]
    t, d, acc, _ = fit_threshold(values, labels)
    return acc, t, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=512,
                    help="model max_seq_len used for the effective pps definition")
    ap.add_argument("--out", default=None,
                    help="write the markdown block to this file as well as stdout")
    args = ap.parse_args()

    rows = build(args.cap)
    n = len(rows)
    lines = []
    W = lines.append

    W("### Leave-one-dataset-out regime prediction "
      "(analysis/lodo_regime.py, no logs required, deterministic)")
    W("")
    W(f"Single-feature threshold rules, {n} folds, one per held-out dataset. "
      f"Threshold fitted on the other {n - 1} by maximising training accuracy; "
      f"effective practice-per-skill uses min(median length, {args.cap}) / skills.")
    W("")

    W("| Feature | LODO correct | In-sample | Missed |")
    W("|---|---|---|---|")
    results = {}
    for feat in FEATURE_ORDER:
        folds = run(rows, feat)
        results[feat] = folds
        hits = sum(1 for f in folds if f["ok"])
        ins_acc, ins_t, ins_d = in_sample_accuracy(rows, feat)
        missed = ", ".join(f["held"] for f in folds if not f["ok"]) or "none"
        W(f"| {feat} | {hits}/{n} | {ins_acc * n:.0f}/{n} | {missed} |")
    W("")

    perfect = [f for f in FEATURE_ORDER if in_sample_accuracy(rows, f)[0] == 1.0]
    W(f"Features that separate all {n} datasets in-sample: "
      + (", ".join(perfect) if perfect else "none") + ".")
    W("")

    W("**Per-fold detail.** `dir +1` means a higher value predicts skill-driven.")
    W("")
    for feat in FEATURE_ORDER:
        W(f"_{feat}_")
        W("")
        W("| Held out | Value | Threshold | dir | Train acc | Correctness-driven "
          "in train | Predicted | Truth | OK |")
        W("|---|---|---|---|---|---|---|---|---|")
        for f in results[feat]:
            W(f"| {f['held']} | {f['value']:.4g} | {f['threshold']:.4g} | "
              f"{f['direction']:+d} | {f['train_acc']:.3f} | "
              f"{f['n_correct_in_train']} | {f['pred']} | {f['truth']} | "
              f"{'yes' if f['ok'] else 'NO'} |")
        W("")

    frag = [f["held"] for f in results[FEATURE_ORDER[0]]
            if f["n_correct_in_train"] < 2]
    W(f"_Read: {n} datasets, {sum(1 for r in rows if r[2] == 'correct')} of them "
      f"correctness-driven. On the {len(frag)} folds that hold out a "
      f"correctness-driven dataset ({', '.join(frag)}), the training set contains "
      f"a single correctness-driven example, so the fitted threshold is set by one "
      f"point. Any feature scoring highly here is fitting {n - 1} points with one "
      f"free parameter and one direction bit; in-sample separation is therefore "
      f"uninformative and is reported only to show how many features achieve it._")

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
