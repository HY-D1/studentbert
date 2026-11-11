from __future__ import annotations

"""H-score's choice on Quick-Tune micro against the fine-tuning trajectories
(docs/trajectory_amendment2.md): overtaken recommendations, the warning flag and its detection
epoch, regret against compute, and each estimator's agreement with the final ranking. Reads the
pinned trajectory table, the scores of analysis/qt_score.py and the extraction log.

  PYTHONPATH=. python analysis/qt_reversal.py --trajectories benchmark_final/trajectories.tsv \
      --scores /projects/algl/dai.hany/quicktune/qt_scores.jsonl \
      --extract-log /projects/algl/dai.hany/quicktune/features/extract_log.jsonl \
      --out benchmark_final/qt_reversal
"""

import argparse
import collections
import csv
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

TRAJ_MD5 = "e93329cbc69ae8a3af843c544d43793b"
TIE = 0.5
BUDGETS = (1, 2, 3, 5, 10, 20, 30, 40, 50)
PRIMARY = "hscore_shrunk"
ESTIMATORS = ("hscore_shrunk", "hscore", "logme")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def settle(flags: list[bool], truth: bool) -> int | None:
    k = None
    for i in range(len(flags) - 1, -1, -1):
        if flags[i] != truth:
            break
        k = i + 1
    return k


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--extract-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pin", action="store_true", help="tests only: skip the md5 pin")
    a = ap.parse_args(argv)
    tp, sp, lp = Path(a.trajectories), Path(a.scores), Path(a.extract_log)
    if not a.no_pin and md5(tp) != TRAJ_MD5:
        sys.exit(f"ABORT: {tp} md5 {md5(tp)}, expected {TRAJ_MD5}")
    traj = collections.defaultdict(dict)
    for r in csv.DictReader(tp.read_text().splitlines(), delimiter="\t"):
        if r["benchmark"] == "quicktune_micro":
            traj[(r["target"].rsplit("/", 1)[1], r["candidate"])][int(r["epoch"])] = (
                float(r["score"]), float(r["cumulative_s"]))
    cands = collections.defaultdict(list)
    for (t, c) in traj:
        cands[t].append(c)
    scores = collections.defaultdict(dict)
    undefined = collections.Counter()
    for line in sp.read_text().splitlines():
        r = json.loads(line)
        if r["score"] is None:
            undefined[r["estimator"]] += 1
            continue
        scores[(r["estimator"], r["task"], r["seed"])][r["model"]] = r["score"]
    seeds = sorted({k[2] for k in scores})
    L = ["# H-score on Quick-Tune micro: reversals, detection and compute", "",
         "Pre-specified in docs/qt_hscore_design.md and docs/trajectory_amendment2.md. Inputs: "
         f"{tp.name} {md5(tp)}, {sp.name} {md5(sp)}, {lp.name} {md5(lp)}. Limitations: rule A "
         "(models not compared under identical hyperparameters) and the image split (the "
         "estimators may see images on which Quick-Tune's curves were validated).", ""]
    recs, est_rows = [], []
    for name in ESTIMATORS:
        rho, top, reg, covered, total = [], 0, [], 0, 0
        for t in sorted(cands):
            final = {c: traj[(t, c)][50][0] for c in cands[t]}
            best = max(final.values())
            for s in seeds:
                sc = scores.get((name, t, s), {})
                total += len(cands[t])
                covered += len(sc)
                if name == PRIMARY and set(sc) != set(cands[t]):
                    sys.exit(f"ABORT: {name} {t} seed {s}: scores do not cover the candidates")
                if len(sc) < 2:
                    continue
                order = sorted(sc)
                rho.append(spearman([sc[c] for c in order], [final[c] for c in order]))
                hi = max(sc.values())
                pick = min(c for c in sc if sc[c] == hi)
                top += best - final[pick] <= TIE
                reg.append(best - final[pick])
                if name == PRIMARY:
                    flags = []
                    for k in range(1, 51):
                        ek = {c: traj[(t, c)][k][0] for c in cands[t]}
                        flags.append(max(v for c, v in ek.items() if c != pick) > ek[pick])
                    over = best - final[pick] > TIE
                    recs.append({"task": t, "seed": s, "pick": pick, "regret": best - final[pick],
                                 "overtaken": over, "flags": flags,
                                 "compute": [sum(traj[(t, c)][k][1] for c in cands[t])
                                             for k in range(1, 51)],
                                 "settle": settle(flags, over)})
        n = len(rho)
        est_rows.append((name, st.mean(rho) if rho else float("nan"), top, n,
                         st.mean(reg) if reg else float("nan"), covered, total))
    L += ["## Estimators against the epoch-50 ranking (30 tasks x 3 seeds)", "",
          "| estimator | candidates scored | rankings | mean Spearman | top-1 tied with the best "
          "| mean regret (points) |", "|---|---|---|---|---|---|"]
    for name, rho, top, n, reg, covered, total in est_rows:
        L.append(f"| {name} | {covered}/{total} | {n} | {rho:+.2f} | {top}/{n} | {reg:.2f} |")
    L += ["", "Undefined scores (features interpolate the labels exactly), by estimator: "
          + (", ".join(f"{k} {v}" for k, v in sorted(undefined.items())) or "none")
          + ". Where an estimator is undefined for some candidates, its ranking covers the rest."]
    over = [r for r in recs if r["overtaken"]]
    keep = [r for r in recs if not r["overtaken"]]
    L += ["", f"## Warning rule on shrunk H-score's choice ({len(recs)} recommendations, "
          f"{len(over)} overtaken)", "",
          "| budget (epochs) | compute per decision (GPU-h) | caught | false alarms | followed | "
          "regret of followed (points) |", "|---|---|---|---|---|---|"]
    L.append(f"| 0 (H-score alone) | 0 | 0/{len(over)} | 0/{len(keep)} | {len(recs)}/{len(recs)} "
             f"| {st.mean(r['regret'] for r in recs):.2f} |")
    for k in BUDGETS:
        fol = [r for r in recs if not r["flags"][k - 1]]
        comp = st.mean(r["compute"][k - 1] for r in recs) / 3600
        fr = f"{st.mean(r['regret'] for r in fol):.2f}" if fol else "none followed"
        L.append(f"| {k} | {comp:.2f} | {sum(r['flags'][k - 1] for r in over)}/{len(over)} | "
                 f"{sum(r['flags'][k - 1] for r in keep)}/{len(keep)} | {len(fol)}/{len(recs)} | "
                 f"{fr} |")
    for label, sub in (("overtaken", over), ("not overtaken", keep)):
        det = [r["settle"] for r in sub]
        hit = [d for d in det if d]
        shares = [r["compute"][r["settle"] - 1] / r["compute"][-1] for r in sub if r["settle"]]
        L.append("")
        L.append(f"- {label} ({len(sub)}): the decision settles by epoch 1 on "
                 f"{sum(1 for d in hit if d <= 1)}, by 3 on {sum(1 for d in hit if d <= 3)}, by "
                 f"10 on {sum(1 for d in hit if d <= 10)}, by 50 on {len(hit)}; median settling "
                 f"epoch {st.median(hit) if hit else 'none'}; compute at settling as a share of "
                 "50 epochs for all candidates: "
                 + (f"median {st.median(shares):.2f}." if shares else "none."))
    ext = collections.defaultdict(float)
    for line in lp.read_text().splitlines():
        r = json.loads(line)
        if r["model"] in cands.get(r["task"], []):
            ext[r["task"]] += r["seconds"]
    gpus = sorted({json.loads(x)["gpu"] for x in lp.read_text().splitlines()})
    L += ["", f"Frozen-feature extraction for a task's candidates (all 800 or 760 images): mean "
          f"{st.mean(ext.values()) / 60:.1f} GPU-minutes on {', '.join(gpus)}; the fine-tuning "
          "compute above is Quick-Tune's recorded time on its own, unrecorded hardware, so the two "
          "are reported side by side and not added.", ""]
    out = Path(a.out)
    tsv = out.with_name(out.name + ".tsv")
    with tsv.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["task", "seed", "pick", "regret", "overtaken", "settle"]
                   + [f"flag_e{k}" for k in BUDGETS])
        for r in recs:
            w.writerow([r["task"], r["seed"], r["pick"], repr(r["regret"]), r["overtaken"],
                        r["settle"] or ""] + [r["flags"][k - 1] for k in BUDGETS])
    L.append(f"Files: {tsv.name} {md5(tsv)}")
    rp = out.with_name(out.name + "_report.md")
    rp.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"wrote {tsv.name} and {rp.name} md5 {md5(rp)}")


if __name__ == "__main__":
    main()
