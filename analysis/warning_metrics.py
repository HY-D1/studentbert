from __future__ import annotations

"""Formal evaluation of the early-fine-tuning warning signal (Prof. Hazra, after RESULTS.md 12.19).

Added after every warning result was seen, at her request; it changes no rule and no decision. For
each recommendation the graded warning score at epoch k is

    s_k = (best validation score of an alternative after k epochs) - (that of H-score's pick),

validation accuracy on the NLP benchmark and validation AUC on StudentBERT. The fixed rule flags a
recommendation exactly when s_k > 0 (ties go to H-score's pick), so s_k is the rule's continuous
version, defined here before any of the numbers below were computed. Reported per data set and
epoch:

  precision, recall and the false-alarm rate at the rule's own threshold (s_k > 0), which must
  reproduce the counts of RESULTS.md 12.15 to 12.19 (the script stops otherwise). Ties: the
  replications (analysis/phase2_eval.py) and StudentBERT break an exact tie at the top in
  H-score's favour, which is s_k > 0; the original NLP test (analysis/nlp_option1_eval.py, seeds
  42, 1 and 2) ranked by validation accuracy and then by encoder name, so an alternative tied with
  H-score's pick and named earlier was counted as the leader. Each set is checked against its own
  implementation, and every exact tie is listed with both counts;
  AUROC and AUPRC (average precision) of s_k for detecting the recommendations whose pick is
  overtaken after full fine-tuning, with a cluster bootstrap over tasks or targets (20,000
  resamples, random.Random(0)) and the AUPRC of a random score (the share overtaken);
  the same two for the frozen signal, the negated margin between H-score's pick and its runner-up
  (standardized within the ranking), as the comparison without any fine-tuning.

Data: StudentBERT (21 recommendations, retrospective) from benchmark_final/reversal_pairs.tsv; the
NLP benchmark on three seed sets (prospective; two fresh-seed replications), 30 recommendations
each, from the head-pair files and the early fine-tunes of each set.

  PYTHONPATH=. python analysis/warning_metrics.py --pairs benchmark_final/reversal_pairs.tsv \
      --nlp benchmark_final/nlp_head:/projects/algl/dai.hany/task4/early_ft:42,1,2 \
      --nlp benchmark_final/p2_head:/projects/algl/dai.hany/task4/early_ft_fresh:3,4,5 \
      --nlp benchmark_final/p3_head:/projects/algl/dai.hany/task4/early_ft_p3:6,7,8 \
      --out benchmark_final/warning_metrics
"""

import argparse
import csv
import hashlib
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

PAIRS_MD5 = "8f57558d142869b3e34a35ad18d095f7"
EPOCHS = (1, 2, 3)
BOOTS = 20000
# (caught, overtaken, false alarms, not overtaken) per epoch, from RESULTS.md 12.15 to 12.19
EXPECTED = {
    "StudentBERT": {1: (1, 7, 3, 14), 2: (6, 7, 3, 14), 3: (6, 7, 3, 14)},
    "42,1,2": {1: (19, 20, 4, 10), 2: (18, 20, 7, 10), 3: (18, 20, 4, 10)},
    "3,4,5": {1: (14, 18, 6, 12), 2: (17, 18, 5, 12), 3: (16, 18, 8, 12)},
    "6,7,8": {1: (17, 20, 5, 10), 2: (19, 20, 4, 10), 3: (18, 20, 3, 10)},
}
NAMES = {"StudentBERT": "StudentBERT (retrospective)",
         "42,1,2": "NLP, seeds 42, 1, 2 (prospective)",
         "3,4,5": "NLP, seeds 3, 4, 5", "6,7,8": "NLP, seeds 6, 7, 8"}


def name_flag(acc: dict, pick: str) -> bool:
    """The original NLP rule: H-score's pick is not first by (-accuracy, encoder name)."""
    return sorted(acc, key=lambda m: (-acc[m], m))[0] != pick


def auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def average_precision(scores: list[float], labels: list[int]) -> float:
    """Step-wise average precision; tied scores enter together."""
    total = sum(labels)
    if total == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    ap, tp, fp, i, prev_recall = 0.0, 0, 0, 0, 0.0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        recall = tp / total
        ap += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall, i = recall, j
    return ap


def cluster_ci(recs: list[dict], key: str, metric) -> tuple[float, float]:
    by = defaultdict(list)
    for r in recs:
        by[r["group"]].append(r)
    groups = sorted(by)
    rng = random.Random(0)
    vals = []
    for _ in range(BOOTS):
        sample = [r for g in (rng.choice(groups) for _ in groups) for r in by[g]]
        v = metric(sample, key)
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def m_auroc(recs: list[dict], key: str) -> float:
    return auroc([r[key] for r in recs if r["y"]], [r[key] for r in recs if not r["y"]])


def m_ap(recs: list[dict], key: str) -> float:
    return average_precision([r[key] for r in recs], [r["y"] for r in recs])


def studentbert(path: str) -> list[dict]:
    p = Path(path)
    if hashlib.md5(p.read_bytes()).hexdigest() != PAIRS_MD5:
        sys.exit(f"ABORT: {p} is not the pair table of RESULTS.md 12.13")
    by = defaultdict(list)
    for r in csv.DictReader(p.read_text().splitlines(), delimiter="\t"):
        if r["frozen_rank_a"] == "1":
            by[(r["target"], r["est_seed"])].append(r)
    out = []
    for (t, s), rs in sorted(by.items()):
        rec = {"group": t, "id": f"{t} seed {s}",
               "y": int(any(r["label_main"] == "reversal" for r in rs)),
               "frozen": -float([r for r in rs if r["frozen_rank_b"] == "2"][0]["margin_hscore_z"])}
        for k in EPOCHS:
            rec[f"s{k}"] = max(float(r[f"val_e{k}_b"]) for r in rs) - float(rs[0][f"val_e{k}_a"])
            rec[f"n{k}"] = rec[f"s{k}"] > 0
        out.append(rec)
    if len(out) != 21:
        sys.exit(f"ABORT: {len(out)} StudentBERT recommendations, expected 21")
    return out


def nlp(prefix: str, root: str, seeds: tuple[int, ...]) -> list[dict]:
    report = Path(prefix + "_report.md").read_text()
    listed = {}
    for name in ("pairs", "recs"):
        p = Path(f"{prefix}_{name}.tsv")
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        if f"{p.name} {md5}" not in report:
            sys.exit(f"ABORT: {p} does not match its report")
        listed[name] = list(csv.DictReader(p.read_text().splitlines(), delimiter="\t"))
    rev = {(r["task"], r["pooling"], r["seed"]): r["reversed"] == "1" for r in listed["recs"]}
    by = defaultdict(list)
    for r in listed["pairs"]:
        by[(r["task"], r["pooling"], r["seed"])].append(r)
    out = []
    for (t, pool, s), rs in sorted(by.items()):
        if int(s) not in seeds:
            sys.exit(f"ABORT: {prefix} holds seed {s}, expected {seeds}")
        pick = rs[0]["pick"]
        hs = {pick: float(rs[0]["hscore_pick"]), **{r["alternative"]: float(r["hscore_alt"])
                                                     for r in rs}}
        runner = max((m for m in hs if m != pick), key=lambda m: hs[m])
        sd = st.pstdev(hs.values()) or 1.0
        rec = {"group": t, "id": f"{t} {pool} seed {s}", "y": int(rev[(t, pool, s)]),
               "frozen": -(hs[pick] - hs[runner]) / sd}
        acc = {}
        for m in hs:
            f = Path(root) / t / f"{m.replace('/', '__')}_{pool}_s{s}.json"
            run = json.loads(f.read_text())
            if [e["epoch"] for e in run["epochs"]] != list(EPOCHS):
                sys.exit(f"ABORT: {f} does not hold 3 epochs")
            acc[m] = [e["val_acc"] for e in run["epochs"]]
        for k in EPOCHS:
            rec[f"s{k}"] = max(acc[m][k - 1] for m in hs if m != pick) - acc[pick][k - 1]
            rec[f"n{k}"] = name_flag({m: acc[m][k - 1] for m in hs}, pick)
        out.append(rec)
    if len(out) != 30:
        sys.exit(f"ABORT: {prefix} gives {len(out)} recommendations, expected 30")
    return out


def evaluate(name: str, recs: list[dict], ties: list[str] | None = None
             ) -> tuple[list[str], list[dict]]:
    pos = sum(r["y"] for r in recs)
    rows, lines = [], []
    for k in EPOCHS:
        key = f"s{k}"
        tp = sum(1 for r in recs if r[key] > 0 and r["y"])
        fp = sum(1 for r in recs if r[key] > 0 and not r["y"])
        own = f"n{k}" if name == "42,1,2" else key
        otp = sum(1 for r in recs if (r[own] if own != key else r[key] > 0) and r["y"])
        ofp = sum(1 for r in recs if (r[own] if own != key else r[key] > 0) and not r["y"])
        if name in EXPECTED and EXPECTED[name][k] != (otp, pos, ofp, len(recs) - pos):
            sys.exit(f"ABORT: {name} epoch {k}: caught {otp}/{pos}, false alarms "
                     f"{ofp}/{len(recs) - pos} do not reproduce RESULTS.md "
                     f"{EXPECTED[name][k]}")
        if ties is not None:
            for r in recs:
                if r[key] == 0:
                    ties.append(f"{NAMES.get(name, name)}, epoch {k}: {r['id']} "
                                f"({'overtaken' if r['y'] else 'not overtaken'}; flagged by "
                                f"{'the original rule only' if r[f'n{k}'] else 'neither rule'})")
        a, ap = m_auroc(recs, key), m_ap(recs, key)
        alo, ahi = cluster_ci(recs, key, m_auroc)
        plo, phi = cluster_ci(recs, key, m_ap)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rows.append({"data": name, "signal": f"epoch {k}", "n": len(recs), "overtaken": pos,
                     "precision": prec, "recall": tp / pos, "false_alarm_rate":
                     fp / (len(recs) - pos), "auroc": a, "auroc_lo": alo, "auroc_hi": ahi,
                     "auprc": ap, "auprc_lo": plo, "auprc_hi": phi, "prevalence": pos / len(recs)})
        lines.append(f"| {NAMES.get(name, name)} | epoch {k} | {tp}/{tp + fp} ({prec:.2f}) | "
                     f"{tp}/{pos} ({tp / pos:.2f}) | {a:.2f} [{alo:.2f}, {ahi:.2f}] | "
                     f"{ap:.2f} [{plo:.2f}, {phi:.2f}] |")
    a, ap = m_auroc(recs, "frozen"), m_ap(recs, "frozen")
    alo, ahi = cluster_ci(recs, "frozen", m_auroc)
    plo, phi = cluster_ci(recs, "frozen", m_ap)
    rows.append({"data": name, "signal": "frozen margin", "n": len(recs), "overtaken": pos,
                 "precision": "", "recall": "", "false_alarm_rate": "", "auroc": a,
                 "auroc_lo": alo, "auroc_hi": ahi, "auprc": ap, "auprc_lo": plo, "auprc_hi": phi,
                 "prevalence": pos / len(recs)})
    lines.append(f"| {NAMES.get(name, name)} | frozen margin | -- | -- | {a:.2f} [{alo:.2f}, "
                 f"{ahi:.2f}] | {ap:.2f} [{plo:.2f}, {phi:.2f}] |")
    return lines, rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--nlp", action="append", default=[], help="head prefix:early-ft dir:seeds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sets = {"StudentBERT": studentbert(a.pairs)}
    for spec in a.nlp:
        prefix, root, seeds = spec.split(":")
        sets[seeds] = nlp(prefix, root, tuple(int(x) for x in seeds.split(",")))
    nlp_keys = [k for k in sets if k != "StudentBERT"]
    if len(nlp_keys) == 3:
        sets["NLP, all three seed sets"] = [r for k in nlp_keys for r in sets[k]]
    L = ["# The warning signal as a detector, epochs 1 to 3", "",
         "Added after the warning results were seen; no rule or decision changes. Score: the best "
         "alternative's validation score after k epochs minus that of H-score's pick; the rule "
         "flags when it is above zero. Precision and recall at that threshold reproduce RESULTS.md "
         "12.15 to 12.19. AUROC and AUPRC: threshold-free, with a cluster bootstrap over tasks "
         "(NLP) or targets (StudentBERT). Frozen margin: H-score's standardized margin over its "
         "runner-up, negated, which needs no fine-tuning.", "",
         "| data | signal | precision (flagged) | recall (overtaken) | AUROC [95% CI] | "
         "AUPRC [95% CI] |", "|---|---|---|---|---|---|"]
    rows, ties = [], []
    for name, recs in sets.items():
        lines, r = evaluate(name, recs, ties if not name.startswith("NLP, all") else None)
        L += lines
        rows += r
    L += ["", "AUPRC of a random score: the share overtaken, "
          + ", ".join(f"{NAMES.get(n, n)} {sum(r['y'] for r in s) / len(s):.2f}"
                      for n, s in sets.items()) + ".", "",
          "With 7 targets and 5 tasks the intervals are wide; the pooled NLP row reuses the same "
          "5 tasks three times, so its interval resamples tasks, not seed sets.", "",
          "Precision and recall above break ties in H-score's favour (s_k > 0) for every set. "
          "Exact ties at the top, where an alternative's validation score equals that of "
          "H-score's pick:"] + ([f"- {x}" for x in ties] or ["- none"])
    out = Path(a.out)
    tp = out.with_name(out.name + ".tsv")
    with tp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    L += ["", f"Files: {tp.name} {hashlib.md5(tp.read_bytes()).hexdigest()}"]
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print(f"wrote {tp.name} and {rp.name} md5 {hashlib.md5(rp.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
