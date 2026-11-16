from __future__ import annotations

"""Regret against compute for static selection, the early leader, the override rule, the
warning rule with abstention and successive halving (docs/regret_compute_design.md), on
StudentBERT, the NLP benchmark and Quick-Tune micro. Reads recorded trajectories, outcomes, scores
and timings only; no model is run.

  PYTHONPATH=. python analysis/regret_compute.py --out benchmark_final/regret_compute \
      --trajectories benchmark_final/trajectories.tsv --pairs benchmark_final/reversal_pairs.tsv \
      --task2 tg1_task2_kt_*.jsonl \
      --nlp-heads benchmark_final/nlp_head_pairs.tsv benchmark_final/p2_head_pairs.tsv \
                  benchmark_final/p3_head_pairs.tsv \
      --nlp-early /projects/algl/dai.hany/task4/early_ft \
                  /projects/algl/dai.hany/task4/early_ft_fresh \
                  /projects/algl/dai.hany/task4/early_ft_p3 \
      --nlp-scores /projects/algl/dai.hany/task4/task4_scores.jsonl \
                   /projects/algl/dai.hany/task4/task4_scores_fresh.jsonl \
                   /projects/algl/dai.hany/task4/task4_scores_p3.jsonl \
      --quicktune /projects/algl/dai.hany/quicktune \
      --qt-scores /projects/algl/dai.hany/quicktune/qt_scores.jsonl \
      --qt-extract-log /projects/algl/dai.hany/quicktune/features/extract_log.jsonl
"""

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

PINS = {"trajectories.tsv": "e93329cbc69ae8a3af843c544d43793b",
        "reversal_pairs.tsv": "8f57558d142869b3e34a35ad18d095f7",
        "nlp_head_pairs.tsv": "d023dd9346eb1020afa11d1dacbaf579",
        "p2_head_pairs.tsv": "9123ad27c17496af7f5f51cdb1987bc1",
        "p3_head_pairs.tsv": "9bad0afce43c426c8105473a6b8f918c",
        "qt_scores.jsonl": "b7f48f2f5c6ee9c45ce17201a6d2f9ea",
        "extract_log.jsonl": "50ea8a9ba8fd20713c25211fd30c6f8e"}
BUDGETS = {"studentbert": (1, 2, 3, 5, 10, 20), "nlp": (1, 2, 3),
           "quicktune": (1, 2, 3, 5, 10, 20, 30, 40, 50)}
SH_START = {"studentbert": (1, 2, 5), "quicktune": (1, 2, 3)}
QT_EPOCHS, QT_MIN_RUNS, QT_MARGIN, SB_MARGIN = 50, 3, 0.5, 0.001
BOOT = 10000
# StudentBERT values recorded in RESULTS.md 12.17, 12.18 (part B) and 12.24; the run stops if the
# common accounting does not reproduce them: (method, budget) -> (chosen, mean regret).
SB_EXPECTED = {("static H-score", ""): (21, "0.0012"), ("early leader", "epoch 1"): (21, "0.0031"),
               ("early leader", "epoch 3"): (21, "0.0006"),
               ("early leader", "epoch 20"): (21, "0.0017"),
               ("override rule", "epoch 3"): (21, "0.0006"),
               ("warning, abstain", "epoch 1"): (17, "0.0009"),
               ("warning, abstain", "epoch 3"): (12, "0.0009")}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def pin(paths: list[Path], skip: bool) -> list[str]:
    out = []
    for p in paths:
        h = md5(p)
        want = PINS.get(p.name)
        if want and h != want and not skip:
            sys.exit(f"ABORT: {p} md5 {h}, expected {want}")
        out.append(f"{p.name} {h}")
    return out


class Decision:
    """One selection problem: candidates, outcome, H-score order, curves and costs."""

    def __init__(self, bench: str, cluster: str, key: str, outcome: dict, hscore: dict,
                 curves: dict, costs: dict, static_s: float, margin, pick: str | None = None,
                 static_on_axis: bool = True) -> None:
        self.bench, self.cluster, self.key = bench, cluster, key
        self.cands = sorted(outcome)
        self.outcome, self.curves, self.costs = outcome, curves, costs
        order = sorted(self.cands, key=lambda c: (-hscore[c], c))
        if pick is not None:
            if hscore[pick] != max(hscore.values()):
                sys.exit(f"ABORT: {bench} {key}: the recorded pick is not H-score's top")
            order = [pick] + [c for c in order if c != pick]
        self.order = order
        self.pick = order[0]
        self.static_s, self.margin = static_s, margin
        # On Quick-Tune, frozen scoring ran on other hardware than the fine-tuning curves, so it
        # is reported but not added to the fine-tuning axis.
        self.static_axis = static_s if static_on_axis else 0.0
        self.best = max(outcome.values())
        self.epochs = min(len(curves[c]) for c in self.cands)

    def regret(self, c: str) -> float:
        return self.best - self.outcome[c]

    def val(self, c: str, k: int) -> float:
        return self.curves[c][k - 1]

    def leader(self, cands: list[str], k: int) -> str:
        top = max(self.val(c, k) for c in cands)
        return next(c for c in self.order if c in cands and self.val(c, k) == top)

    def ft_cost(self, epochs: dict) -> float:
        return sum(self.costs[c][e - 1] for c, e in epochs.items() if e > 0)


def static(d: Decision) -> tuple[str, float]:
    return d.pick, d.static_s


def early(d: Decision, k: int) -> tuple[str, float]:
    return d.leader(d.cands, k), d.ft_cost({c: k for c in d.cands})


def override(d: Decision, k: int) -> tuple[str, float]:
    lead = d.leader(d.cands, k)
    gap = d.val(lead, k) - d.val(d.pick, k)
    choice = lead if lead != d.pick and gap > d.margin(d, lead, d.pick, k) else d.pick
    return choice, d.static_axis + d.ft_cost({c: k for c in d.cands})


def abstain(d: Decision, k: int) -> tuple[str | None, float]:
    flagged = max(d.val(c, k) for c in d.cands if c != d.pick) > d.val(d.pick, k)
    return (None if flagged else d.pick), d.static_axis + d.ft_cost({c: k for c in d.cands})


def halving(d: Decision, rungs: list[int], eta: int) -> tuple[str, float]:
    alive, trained = list(d.order), {c: 0 for c in d.cands}
    for e in rungs:
        e = min(e, d.epochs)
        for c in alive:
            trained[c] = max(trained[c], e)
        if len(alive) == 1:
            break
        ranked = sorted(alive, key=lambda c: (-d.val(c, e), d.order.index(c)))
        alive = ranked[:math.ceil(len(alive) / eta)]
        if len(alive) == 1:
            break
    if len(alive) > 1:
        alive = [d.leader(alive, max(trained[c] for c in alive))]
    return alive[0], d.ft_cost(trained)


def geometric(start: int, eta: int, n: int, cap: int) -> list[int]:
    rungs, e, alive = [], start, n
    while True:
        rungs.append(min(e, cap))
        if alive <= 1 or e >= cap:
            break
        alive = math.ceil(alive / eta)
        if alive == 1:
            break
        e *= eta
    return rungs


def full_best_val(d: Decision) -> tuple[str, float]:
    top = max(max(d.curves[c]) for c in d.cands)
    choice = next(c for c in d.order if max(d.curves[c]) == top)
    return choice, d.ft_cost({c: d.epochs for c in d.cands})


def studentbert(traj: Path, pairs: Path, task2: list[Path]) -> list[Decision]:
    curves = collections.defaultdict(dict)
    for r in csv.DictReader(traj.read_text().splitlines(), delimiter="\t"):
        if r["benchmark"] == "studentbert":
            curves[(r["target"], r["seed"])].setdefault(r["candidate"], {})[int(r["epoch"])] = (
                float(r["score"]), float(r["cumulative_s"]))
    cost = collections.defaultdict(list)
    for p in task2:
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["estimator"] == "hscore_kt_causal":
                cost[(r["target"], str(r["seed"]))].append(
                    r["feature_extraction_time"] + r["scoring_time"])
    heads = collections.defaultdict(list)
    for r in csv.DictReader(pairs.read_text().splitlines(), delimiter="\t"):
        if r["frozen_rank_a"] == "1":
            heads[(r["target"], r["est_seed"])].append(r)
    out = []
    for (t, s), rs in sorted(heads.items()):
        pick = rs[0]["cand_a"]
        gold = {pick: float(rs[0]["gold_a"]), **{r["cand_b"]: float(r["gold_b"]) for r in rs}}
        hs = {pick: float(rs[0]["hscore_a"]), **{r["cand_b"]: float(r["hscore_b"]) for r in rs}}
        tr = curves[(t, s)]
        if set(tr) != set(gold) or any(sorted(tr[c]) != list(range(1, 21)) for c in gold):
            sys.exit(f"ABORT: StudentBERT {t} seed {s}: trajectories do not cover the candidates")
        if len(cost[(t, s)]) != 8:
            sys.exit(f"ABORT: StudentBERT {t} seed {s}: {len(cost[(t, s)])} scoring records, "
                     "expected 8")
        out.append(Decision(
            "studentbert", t, f"{t}/{s}", gold, hs,
            {c: [tr[c][e][0] for e in range(1, 21)] for c in gold},
            {c: [tr[c][e][1] for e in range(1, 21)] for c in gold},
            sum(cost[(t, s)]), lambda d, a, b, k: SB_MARGIN, pick=pick))
    if len(out) != 21:
        sys.exit(f"ABORT: {len(out)} StudentBERT decisions, expected 21")
    return out


def nlp(heads: list[Path], early: list[Path], scores: list[Path]) -> list[Decision]:
    out = []
    for hp, ed, sp in zip(heads, early, scores):
        sc = {}
        for line in sp.read_text().splitlines():
            r = json.loads(line)
            if r["estimator"] == "hscore":
                sc[(r["task"], r["pooling"], int(r["seed"]), r["model"])] = (
                    r["extract_s"] + r["scoring_s"])
        by = collections.defaultdict(list)
        for r in csv.DictReader(hp.read_text().splitlines(), delimiter="\t"):
            by[(r["task"], r["pooling"], int(r["seed"]))].append(r)
        for (t, p, s), rs in sorted(by.items()):
            pick = rs[0]["pick"]
            perf = {pick: float(rs[0]["tuned_pick"]),
                    **{r["alternative"]: float(r["tuned_alt"]) for r in rs}}
            hs = {pick: float(rs[0]["hscore_pick"]),
                  **{r["alternative"]: float(r["hscore_alt"]) for r in rs}}
            curves, costs, n_val = {}, {}, set()
            for m in perf:
                f = ed / t / f"{m.replace('/', '__')}_{p}_s{s}.json"
                rec = json.loads(f.read_text())
                ep = sorted(rec["epochs"], key=lambda e: e["epoch"])
                if [e["epoch"] for e in ep] != [1, 2, 3]:
                    sys.exit(f"ABORT: {f}: epochs {[e['epoch'] for e in ep]}")
                curves[m] = [e["val_acc"] for e in ep]
                costs[m] = [sum(e["epoch_s"] for e in ep[:i + 1]) for i in range(3)]
                n_val.add(rec["n_val"])
            if len(n_val) != 1:
                sys.exit(f"ABORT: NLP {t} {p} {s}: encoders differ in validation size")
            missing = [m for m in perf if (t, p, s, m) not in sc]
            if missing:
                sys.exit(f"ABORT: NLP {t} {p} {s}: no scoring record for {missing}")
            n = n_val.pop()

            def margin(d: Decision, a: str, b: str, k: int, n: int = n) -> float:
                q = (d.val(a, k) + d.val(b, k)) / 2
                return 1.96 * math.sqrt(2 * q * (1 - q) / n)
            out.append(Decision("nlp", t, f"{t}/{p}/{s}", perf, hs, curves, costs,
                                sum(sc[(t, p, s, m)] for m in perf), margin, pick=pick))
    if len(out) != 90:
        sys.exit(f"ABORT: {len(out)} NLP decisions, expected 90")
    return out


def quicktune(root: Path, qt_scores: Path, extract_log: Path) -> tuple[list[Decision], list]:
    d = root / "micro" / "micro"
    table = list(csv.DictReader((d / "unnormalized_args_table.csv").open()))
    idc = list(table[0])[0]
    top1 = json.load((d / "eval_top1.json").open())
    tr_t = json.load((d / "train_time.json").open())
    ev_t = json.load((d / "eval_time.json").open())
    task_of = {rid: t for t, runs in top1.items() for rid in runs}
    runs = collections.defaultdict(list)
    for r in table:
        m = [k[len("cat__model_"):] for k, v in r.items()
             if k.startswith("cat__model_") and float(v or 0)]
        t = task_of[r[idc]]
        if len(top1[t][r[idc]]) >= QT_EPOCHS:
            runs[(t.rsplit("/", 1)[1], m[0])].append((t, r[idc]))
    curves, costs = {}, {}
    for (code, m), rs in runs.items():
        if len(rs) < QT_MIN_RUNS:
            continue
        curves[(code, m)] = [st.median(top1[t][rid][e] for t, rid in rs)
                             for e in range(QT_EPOCHS)]
        per = [st.median(tr_t[t][rid][e] + ev_t[t][rid][e] for t, rid in rs)
               for e in range(QT_EPOCHS)]
        costs[(code, m)] = [sum(per[:e + 1]) for e in range(QT_EPOCHS)]
    ext = {}
    for line in extract_log.read_text().splitlines():
        r = json.loads(line)
        ext[(r["task"], r["model"])] = r["seconds"]
    sc = collections.defaultdict(dict)
    for line in qt_scores.read_text().splitlines():
        r = json.loads(line)
        if r["score"] is not None:
            sc[(r["estimator"], r["task"], r["seed"])][r["model"]] = (r["score"], r["seconds"])
    cands = collections.defaultdict(list)
    for (code, m) in curves:
        cands[code].append(m)
    out, plain = [], []
    for code in sorted(cands):
        for seed in (42, 1, 2):
            shr = sc[("hscore_shrunk", code, seed)]
            ms = sorted(cands[code])
            if any(m not in shr for m in ms):
                sys.exit(f"ABORT: Quick-Tune {code} seed {seed}: missing shrunk H-score")
            static_s = sum(ext[(code, m)] + shr[m][1] for m in ms)
            outcome = {m: curves[(code, m)][-1] for m in ms}
            cv = {m: curves[(code, m)] for m in ms}
            co = {m: costs[(code, m)] for m in ms}
            out.append(Decision("quicktune", code, f"{code}/{seed}", outcome,
                                {m: shr[m][0] for m in ms}, cv, co, static_s,
                                lambda d, a, b, k: QT_MARGIN, static_on_axis=False))
            pl = sc[("hscore", code, seed)]
            plain.append(Decision("quicktune", code, f"{code}/{seed}", outcome,
                                  {m: pl[m][0] for m in ms}, cv, co,
                                  sum(ext[(code, m)] + pl[m][1] for m in ms),
                                  lambda d, a, b, k: QT_MARGIN, static_on_axis=False))
    if len(out) != 90:
        sys.exit(f"ABORT: {len(out)} Quick-Tune decisions, expected 90")
    return out, plain


def boot_ci(ds: list[Decision], vals: list[float]) -> tuple[float, float]:
    by = collections.defaultdict(list)
    for d, v in zip(ds, vals):
        by[d.cluster].append(v)
    keys = sorted(by)
    rng = random.Random(0)
    means = []
    for _ in range(BOOT):
        xs = [v for k in (rng.choice(keys) for _ in keys) for v in by[k]]
        means.append(st.mean(xs))
    means.sort()
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def evaluate(bench: str, ds: list[Decision], plain: list[Decision] | None = None) -> list[dict]:
    rows = []

    def add(method: str, param: str, fn, pool: list[Decision] = ds) -> None:
        res = [fn(d) for d in pool]
        followed = [(d, c) for d, (c, _) in zip(pool, res) if c is not None]
        regs = [d.regret(c) for d, c in followed]
        lo, hi = boot_ci([d for d, _ in followed], regs) if followed else (math.nan, math.nan)
        rows.append({"benchmark": bench, "method": method, "param": param,
                     "decisions": len(pool), "chosen": len(followed),
                     "compute_gpu_h": st.mean(s for _, s in res) / 3600,
                     "regret": st.mean(regs) if regs else math.nan, "ci_low": lo, "ci_high": hi,
                     "zero_regret": sum(r == 0 for r in regs)})
    add("static H-score", "", static)
    if plain:
        add("static plain H-score", "", static, plain)
    for k in BUDGETS[bench]:
        add("early leader", f"epoch {k}", lambda d, k=k: early(d, k))
        add("override rule", f"epoch {k}", lambda d, k=k: override(d, k))
        add("warning, abstain", f"epoch {k}", lambda d, k=k: abstain(d, k))
    for eta in (2, 3):
        name = "successive halving" + ("" if eta == 2 else " (factor 3)")
        if bench == "nlp":
            add(name, "rungs 1,2,3", lambda d, eta=eta: halving(d, [1, 2, 3], eta))
            continue
        for r in SH_START[bench]:
            add(name, f"start {r}", lambda d, r=r, eta=eta:
                halving(d, geometric(r, eta, len(d.cands), d.epochs), eta))
    if bench == "studentbert":
        add("full fine-tuning", "best validation", full_best_val)
    return rows


def frontier(rows: list[dict], exclude_static_compute: bool) -> set[int]:
    pts = [(i, (0.0 if exclude_static_compute and r["method"].startswith("static") else
                r["compute_gpu_h"]), r["regret"]) for i, r in enumerate(rows)
           if r["chosen"] == r["decisions"]]
    keep = set()
    for i, c, g in pts:
        if not any((c2 <= c and g2 <= g) and (c2 < c or g2 < g) for j, c2, g2 in pts if j != i):
            keep.add(i)
    return keep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--task2", nargs="+", required=True)
    ap.add_argument("--nlp-heads", nargs=3, required=True)
    ap.add_argument("--nlp-early", nargs=3, required=True)
    ap.add_argument("--nlp-scores", nargs=3, required=True)
    ap.add_argument("--quicktune", required=True)
    ap.add_argument("--qt-scores", required=True)
    ap.add_argument("--qt-extract-log", required=True)
    ap.add_argument("--no-pin", action="store_true", help="tests only: skip the md5 pins")
    a = ap.parse_args(argv)
    files = [Path(a.trajectories), Path(a.pairs), *map(Path, a.task2), *map(Path, a.nlp_heads),
             *map(Path, a.nlp_scores), Path(a.qt_scores), Path(a.qt_extract_log)]
    inputs = pin(files, a.no_pin)
    sb = studentbert(Path(a.trajectories), Path(a.pairs), [Path(p) for p in a.task2])
    nl = nlp([Path(p) for p in a.nlp_heads], [Path(p) for p in a.nlp_early],
             [Path(p) for p in a.nlp_scores])
    qt, qt_plain = quicktune(Path(a.quicktune), Path(a.qt_scores), Path(a.qt_extract_log))
    L = ["# Regret against compute: static, early leader, override, abstention, successive "
         "halving", "", "Pre-specified in docs/regret_compute_design.md (H-score, early leader and "
         "rule results were known before; successive halving, the common compute accounting and "
         "the frontier are new; the script was test-run on the StudentBERT scores with stand-in "
         "costs while it was written, and no definition changed after that). StudentBERT "
         "reproduces RESULTS.md 12.17, 12.18 and 12.24. Inputs: " + ", ".join(inputs) + ".", ""]
    all_rows = []
    units = {"studentbert": "test AUC", "nlp": "points", "quicktune": "top-1 points"}
    notes = {"studentbert": "21 decisions, 8 candidates; compute is cluster GPU time (fine-tunes "
                            "on V100-SXM2; the scoring jobs' GPU is not in their records).",
             "nlp": "90 decisions, 7 encoders; curves stop at epoch 3, so every method stops "
                    "there; compute is V100-SXM2 time.",
             "quicktune": "90 decisions, models with at least 3 complete runs, median curves; "
                          "fine-tuning compute is Quick-Tune's recorded time on its own hardware; "
                          "the static methods' compute is H200 scoring time, shown in its column "
                          "but treated as zero for the frontier, since it is not on the same "
                          "axis."}
    for bench, ds, extra in (("studentbert", sb, None), ("nlp", nl, None),
                             ("quicktune", qt, qt_plain)):
        rows = evaluate(bench, ds, extra)
        if bench == "studentbert":
            got = {(r["method"], r["param"]): (r["chosen"], f"{r['regret']:.4f}") for r in rows}
            bad = {k: (got.get(k), v) for k, v in SB_EXPECTED.items() if got.get(k) != v}
            if bad:
                sys.exit(f"ABORT: StudentBERT does not reproduce RESULTS.md: {bad}")
        front = frontier(rows, exclude_static_compute=bench == "quicktune")
        L += [f"## {bench} ({units[bench]})", "", notes[bench], "",
              "| method | setting | compute per decision (GPU-h) | chosen | mean regret [95% "
              "cluster CI] | zero regret | on the frontier |", "|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rows):
            r["frontier"] = i in front
            ci = (f"{r['regret']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
                  if bench == "studentbert" else
                  f"{r['regret']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
            L.append(f"| {r['method']} | {r['param']} | {r['compute_gpu_h']:.3f} | "
                     f"{r['chosen']}/{r['decisions']} | {ci} | {r['zero_regret']} | "
                     f"{'yes' if r['frontier'] else ''} |")
        L.append("")
        all_rows += rows
    out = Path(a.out)
    tsv = out.with_name(out.name + ".tsv")
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    L.append(f"Files: {tsv.name} {md5(tsv)}")
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {tsv.name} and {rp.name} md5 {md5(rp)}")


if __name__ == "__main__":
    main()
