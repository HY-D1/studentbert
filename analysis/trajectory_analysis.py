from __future__ import annotations

"""Trajectory and compute analysis (docs/trajectory_design.md, docs/trajectory_amendment1.md).

StudentBERT: for each of the 21 H-score recommendations, when the warning decision settles on the
final outcome, how much compute that took, and the regret of the early leader and of the
recommendations the rule follows at each epoch budget. Quick-Tune: for each task, when the early
leader settles on a choice tied with the best final score, the compute to that point, and the
regret of the early leader at each budget. Reads only the pinned trajectory and pair tables.

  python analysis/trajectory_analysis.py --trajectories benchmark_final/trajectories.tsv \
      --pairs benchmark_final/reversal_pairs.tsv --out benchmark_final/trajectory_analysis
"""

import argparse
import collections
import csv
import hashlib
import statistics as st
import sys
from pathlib import Path

TRAJ_MD5 = "e93329cbc69ae8a3af843c544d43793b"
PAIRS_MD5 = "8f57558d142869b3e34a35ad18d095f7"
SB_BUDGETS = (1, 2, 3, 5, 10, 20)
QT_BUDGETS = (1, 2, 3, 5, 10, 20, 30, 40, 50)
QT_TIE = 0.5
# Reproduction of recorded results: followed by the warning rule and its regret (RESULTS.md 12.17
# and 12.20) and the early leader's regret (12.18, part B) at epochs 1 to 3.
SB_EXPECTED = {1: (17, "0.0009", "0.0031"), 2: (12, "0.0003", None), 3: (12, "0.0009", "0.0006")}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def load_traj(path: Path) -> dict:
    """(benchmark, target, seed) -> candidate -> epoch -> (score, cumulative seconds)."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in csv.DictReader(path.read_text().splitlines(), delimiter="\t"):
        if r["benchmark"] == "studentbert" or r["benchmark"].startswith("quicktune_"):
            out[(r["benchmark"], r["target"], r["seed"])][r["candidate"]][int(r["epoch"])] = (
                float(r["score"]), float(r["cumulative_s"]))
    return out


def settle(flags: list[bool], truth: bool) -> int | None:
    """First 1-based epoch from which every flag equals truth; None if it never settles."""
    k = None
    for i in range(len(flags) - 1, -1, -1):
        if flags[i] != truth:
            break
        k = i + 1
    return k


def studentbert(traj: dict, pairs: Path) -> tuple[list[str], list[dict]]:
    heads = collections.defaultdict(list)
    for r in csv.DictReader(pairs.read_text().splitlines(), delimiter="\t"):
        if r["frozen_rank_a"] == "1":
            heads[(r["target"], r["est_seed"])].append(r)
    if len(heads) != 21:
        sys.exit(f"ABORT: {len(heads)} StudentBERT recommendations, expected 21")
    recs = []
    for (t, s), rs in sorted(heads.items()):
        pick = rs[0]["cand_a"]
        gold = {pick: float(rs[0]["gold_a"]), **{r["cand_b"]: float(r["gold_b"]) for r in rs}}
        tr = traj.get(("studentbert", t, s))
        if tr is None or set(tr) != set(gold) or any(len(tr[c]) != 20 for c in gold):
            sys.exit(f"ABORT: {t} seed {s} lacks 20 epochs for its 8 candidates")
        best = max(gold.values())
        over = any(r["label_main"] == "reversal" for r in rs)
        flags, leaders, comp = [], [], []
        for k in range(1, 21):
            sc = {c: tr[c][k][0] for c in gold}
            top = max(sc.values())
            flags.append(max(v for c, v in sc.items() if c != pick) > sc[pick])
            leaders.append(pick if sc[pick] == top else min(c for c in sc if sc[c] == top))
            comp.append(sum(tr[c][k][1] for c in gold))
        recs.append({"target": t, "seed": s, "pick": pick, "overtaken": over,
                     "pick_regret": best - gold[pick], "flags": flags,
                     "leader_regret": [best - gold[c] for c in leaders], "compute": comp,
                     "settle": settle(flags, over)})
    L = ["## StudentBERT (21 recommendations, retrospective)", "",
         "| budget (epochs) | compute per decision (GPU-h) | early leader regret | followed by "
         "the rule | regret of followed | overtaken among followed |", "|---|---|---|---|---|---|"]
    rows = []
    L.append(f"| 0 (H-score alone) | 0 | {st.mean(r['pick_regret'] for r in recs):.4f} | 21/21 | "
             f"{st.mean(r['pick_regret'] for r in recs):.4f} | "
             f"{sum(r['overtaken'] for r in recs)} |")
    for k in SB_BUDGETS:
        fol = [r for r in recs if not r["flags"][k - 1]]
        comp = st.mean(r["compute"][k - 1] for r in recs) / 3600
        lr = st.mean(r["leader_regret"][k - 1] for r in recs)
        fr = st.mean(r["pick_regret"] for r in fol) if fol else float("nan")
        L.append(f"| {k} | {comp:.3f} | {lr:.4f} | {len(fol)}/21 | {fr:.4f} | "
                 f"{sum(r['overtaken'] for r in fol)} |")
        if k in SB_EXPECTED:
            n, fr_want, lr_want = SB_EXPECTED[k]
            if len(fol) != n or f"{fr:.4f}" != fr_want or (lr_want and f"{lr:.4f}" != lr_want):
                sys.exit(f"ABORT: epoch {k} gives {len(fol)} followed, regret {fr:.4f}, leader "
                         f"{lr:.4f}; RESULTS.md records {SB_EXPECTED[k]}")
        rows.append({"benchmark": "studentbert", "budget": k, "compute_gpu_h": comp,
                     "leader_regret": lr, "followed": len(fol), "followed_regret": fr})
    L.append("")
    for label, want in (("overtaken", True), ("not overtaken", False)):
        sub = [r for r in recs if r["overtaken"] == want]
        det = [r["settle"] for r in sub]
        shares = [r["compute"][r["settle"] - 1] / r["compute"][-1] for r in sub if r["settle"]]
        L.append(f"- {label} ({len(sub)}): the decision settles at epochs "
                 f"{sorted(d for d in det if d)}; never within 20 epochs: "
                 f"{sum(d is None for d in det)}. Compute at settling, as a share of fine-tuning "
                 f"all 8 candidates for 20 epochs: "
                 + (f"median {st.median(shares):.2f}, range {min(shares):.2f} to {max(shares):.2f}."
                    if shares else "none."))
    L += ["", "Settling per recommendation (target, seed, overtaken, epoch): "
          + "; ".join(f"{r['target']} {r['seed']} {'yes' if r['overtaken'] else 'no'} "
                      f"{r['settle'] or 'never'}" for r in recs) + "."]
    return L, rows


def quicktune(traj: dict) -> tuple[list[str], list[dict]]:
    L, rows = [], []
    for v in ("micro", "mini", "extended"):
        tasks = {k[1]: c for k, c in traj.items() if k[0] == f"quicktune_{v}"}
        if not tasks:
            sys.exit(f"ABORT: no Quick-Tune {v} rows")
        res = []
        for t, cands in sorted(tasks.items()):
            if any(len(e) != 50 for e in cands.values()):
                sys.exit(f"ABORT: {v} {t} has a candidate without 50 epochs")
            final = {c: e[50][0] for c, e in cands.items()}
            best = max(final.values())
            hits, regs, comp = [], [], []
            for k in range(1, 51):
                sc = {c: e[k][0] for c, e in cands.items()}
                top = max(sc.values())
                lead = min(c for c in sc if sc[c] == top)
                regs.append(best - final[lead])
                hits.append(best - final[lead] <= QT_TIE)
                comp.append(sum(e[k][1] for e in cands.values()))
            res.append({"task": t, "n": len(cands), "hits": hits, "regs": regs, "comp": comp,
                        "settle": settle(hits, True)})
        L += [f"## Quick-Tune {v} ({len(res)} tasks, {min(r['n'] for r in res)} to "
              f"{max(r['n'] for r in res)} models per task; rule A)", "",
              "| budget (epochs) | compute per decision (GPU-h) | early leader regret (points) | "
              "early leader tied with the best |", "|---|---|---|---|"]
        for k in QT_BUDGETS:
            comp = st.mean(r["comp"][k - 1] for r in res) / 3600
            reg = st.mean(r["regs"][k - 1] for r in res)
            hit = sum(r["hits"][k - 1] for r in res)
            L.append(f"| {k} | {comp:.2f} | {reg:.2f} | {hit}/{len(res)} |")
            rows.append({"benchmark": f"quicktune_{v}", "budget": k, "compute_gpu_h": comp,
                         "leader_regret": reg, "followed": hit, "followed_regret": ""})
        det = [r["settle"] for r in res]
        shares = [r["comp"][r["settle"] - 1] / r["comp"][-1] for r in res if r["settle"]]
        by = {e: sum(1 for d in det if d and d <= e) for e in (1, 3, 5, 10, 20)}
        L += ["", f"- Winner detection: settles by epoch 1 on {by[1]}, by 3 on {by[3]}, by 5 on "
              f"{by[5]}, by 10 on {by[10]} and by 20 on {by[20]} of {len(res)} tasks; median "
              f"settling epoch {st.median(d for d in det if d):.0f}; compute at settling as a "
              f"share of running all candidates for 50 epochs: median {st.median(shares):.2f}.",
              ""]
    return L, rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pin", action="store_true", help="tests only: skip the md5 pins")
    a = ap.parse_args(argv)
    tp, pp = Path(a.trajectories), Path(a.pairs)
    if not a.no_pin:
        for p, want in ((tp, TRAJ_MD5), (pp, PAIRS_MD5)):
            if md5(p) != want:
                sys.exit(f"ABORT: {p} md5 {md5(p)}, expected {want}")
    traj = load_traj(tp)
    L = ["# Trajectory and compute analysis", "",
         "Pre-specified in docs/trajectory_design.md and docs/trajectory_amendment1.md; "
         "descriptive (StudentBERT results were seen before). "
         f"Inputs: {tp.name} {md5(tp)}, {pp.name} "
         f"{md5(pp)}. Compute: StudentBERT per-epoch costs are measured run times spread evenly "
         "over epochs; Quick-Tune costs are the recorded per-epoch times. Quick-Tune models are "
         "not compared under identical hyperparameters (rule A).", ""]
    l1, r1 = studentbert(traj, pp)
    l2, r2 = quicktune(traj)
    L += l1 + [""] + l2
    out = Path(a.out)
    tsv = out.with_name(out.name + ".tsv")
    with tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(r1[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in r1 + r2:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    L += [f"Files: {tsv.name} {md5(tsv)}"]
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {tsv.name} and {rp.name} md5 {md5(rp)}")


if __name__ == "__main__":
    main()
