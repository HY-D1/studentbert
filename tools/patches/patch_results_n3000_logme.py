#!/usr/bin/env python3
"""Add sections 2.1 (verified N=3000 KT source table) and 6.1 (LogME comparison)
to RESULTS.md. Run from the repo root: python3 patch_results_n3000_logme.py

Idempotent: refuses to run twice. All numbers are recomputed here from the
per-seed values parsed out of the logs (poster_gaps_evidence.txt, July 30 2026),
so nothing is hand-copied. Asserts the headline gains before writing.
"""
from statistics import mean, pstdev
import sys

PATH = "RESULTS.md"

# ---- per-seed test AUC, seeds 1/2/42, from the parsed logs ------------------
# runs: edubert_<target>_kt_<t>_<cond>_n3000_seed{1,2,42}
KT = {
 "assist2017": {"scratch":[0.6691,0.6704,0.6710], "indomain":[0.6930,0.6914,0.6958],
                "fromednet":[0.6954,0.6957,0.7000], "fromjunyi":[0.6891,0.6887,0.6893]},
 "ednet":      {"scratch":[0.6643,0.6652,0.6661], "indomain":[0.6718,0.6738,0.6729],
                "fromjunyi":[0.6689,0.6697,0.6693], "fromassist":[0.6639,0.6664,0.6638]},
 "junyi":      {"scratch":[0.7340,0.7365,0.7350], "indomain":[0.7387,0.7391,0.7386],
                "fromednet":[0.7415,0.7415,0.7411], "fromassist":[0.7339,0.7340,0.7347]},
}
# LOGME_RESULT lines from w7_logme_8263344.log (scripts/compute_logme.py)
LOGME = {
 "assist2017": {"indomain":-0.563884, "fromednet":-0.565629, "fromjunyi":-0.624513, "scratch":-0.669368},
 "ednet":      {"indomain":-0.548155, "fromassist":-0.569707, "fromjunyi":-0.611066, "scratch":-0.650869},
 "junyi":      {"fromednet":-0.439082, "fromassist":-0.506391, "indomain":-0.551480, "scratch":-0.574950},
}
# W6 probe2 mean accuracies (already log-verified in section 6)
PROBE = {
 "assist2017": {"indomain":0.1458, "fromednet":0.1417, "fromjunyi":0.1414},
 "ednet":      {"indomain":0.1437, "fromjunyi":0.1354, "fromassist":0.1337},
 "junyi":      {"fromednet":0.0209, "indomain":0.0201, "fromassist":0.0187},
}

gains = {(t, c): mean(v) - mean(KT[t]["scratch"])
         for t, conds in KT.items() for c, v in conds.items() if c != "scratch"}
assert round(gains[("assist2017","fromednet")], 4) == 0.0269
assert round(gains[("assist2017","indomain")], 4)  == 0.0232
assert round(gains[("assist2017","fromjunyi")], 4) == 0.0189

def spearman3(xs, ys):
    rx = {k: i for i, k in enumerate(sorted(xs, key=xs.get, reverse=True), 1)}
    ry = {k: i for i, k in enumerate(sorted(ys, key=ys.get, reverse=True), 1)}
    return 1 - 6 * sum((rx[k] - ry[k]) ** 2 for k in xs) / 24

rho = {}
for name, score in (("logme", LOGME), ("probe", PROBE)):
    rho[name] = [spearman3({c: v for c, v in score[t].items() if c != "scratch"},
                           {c: gains[(t, c)] for c in score[t] if c != "scratch"})
                 for t in KT]
assert round(mean(rho["logme"]), 2) == 0.50 and round(mean(rho["probe"]), 2) == 0.83

def cell(t, c):
    if c not in KT[t]:
        return "(in-domain)" if c.replace("from","") in t or t.startswith(c.replace("from","")) else "-"
    v = KT[t][c]
    g = f" ({gains[(t,c)]:+.4f})" if c != "scratch" else ""
    return f"{mean(v):.4f} ±{pstdev(v):.4f}{g}"

block21 = """
### 2.1 Verified source table at N=3000 (KT test AUC, parsed from logs) [EDM; poster R2]

Runs `edubert_<target>_kt_<t>_{scratch|indomain|fromednet|fromjunyi|fromassist}_n3000_seed{1,2,42}`, 3 seeds, mean ±pstdev, gain vs scratch in parentheses. Log-verified July 30 2026 (poster_gaps_evidence.txt).

| Target | scratch | indomain | fromednet | fromjunyi | fromassist |
|---|---|---|---|---|---|
| assist2017 | %s | %s | %s | %s | (=indomain) |
| ednet | %s | %s | (=indomain) | %s | %s |
| junyi | %s | %s | %s | (=indomain) | %s |

Per-seed paired gains vs scratch, assist2017 target: fromednet +0.0263/+0.0253/+0.0290 (3/3 positive), indomain +0.0239/+0.0210/+0.0248, fromjunyi +0.0200/+0.0183/+0.0183.

_Read: the biggest source (EdNet 442K) beats in-domain on both cross-domain targets (assist2017 +0.0269 vs +0.0232; junyi +0.0062 vs +0.0036); on EdNet's own target in-domain leads (+0.0076) and the granularity-closest source (ASSIST) transfers WORST, below scratch (-0.0005). Quantitative backing for the scale>granularity claim and the poster R2 card._
""" % (cell("assist2017","scratch"), cell("assist2017","indomain"), cell("assist2017","fromednet"), cell("assist2017","fromjunyi"),
       cell("ednet","scratch"), cell("ednet","indomain"), cell("ednet","fromjunyi"), cell("ednet","fromassist"),
       cell("junyi","scratch"), cell("junyi","indomain"), cell("junyi","fromednet"), cell("junyi","fromassist"))

lg = lambda t, c: f"{LOGME[t][c]:.6f}" if c in LOGME[t] else "-"
block61 = f"""
### 6.1 LogME vs the domain probe (w7_logme_8263344.log, scripts/compute_logme.py) [placement: ICLR or NeurIPS, advisor decision pending]

LogME on frozen encoders (full-objective + scratch), 3 targets, higher = better:

| Target | scratch | indomain | fromednet | fromjunyi | fromassist |
|---|---|---|---|---|---|
| assist2017 | {lg('assist2017','scratch')} | {lg('assist2017','indomain')} | {lg('assist2017','fromednet')} | {lg('assist2017','fromjunyi')} | (=indomain) |
| ednet | {lg('ednet','scratch')} | {lg('ednet','indomain')} | (=indomain) | {lg('ednet','fromjunyi')} | {lg('ednet','fromassist')} |
| junyi | {lg('junyi','scratch')} | {lg('junyi','indomain')} | {lg('junyi','fromednet')} | (=indomain) | {lg('junyi','fromassist')} |

Per-target Spearman rho against the same N=3000 KT gains (section 2.1), 3 pretrained sources per target:
LogME {[round(r,1) for r in rho['logme']]} mean {mean(rho['logme']):.2f}; probe {[round(r,1) for r in rho['probe']]} mean {mean(rho['probe']):.2f}.

_Read: identical 9 source-target pairs, identical gains. LogME does rank scratch last on all 3 targets (the coarse pretrain-vs-not signal is right) but mis-orders the pretrained sources on every target; the domain-specific masked-skill probe (section 6) tracks the source ordering better (0.83 vs 0.50)._
"""

txt = open(PATH).read()
for marker in ("### 2.1 Verified source table", "### 6.1 LogME"):
    if marker in txt:
        sys.exit(f"ALREADY PATCHED ({marker!r} present). Nothing done.")
for anchor in ("\n## 3. ", "\n## 7. "):
    assert txt.count(anchor) == 1, f"anchor {anchor!r} not unique in {PATH}"

txt = txt.replace("\n## 3. ", block21 + "\n## 3. ")
txt = txt.replace("\n## 7. ", block61 + "\n## 7. ")
open(PATH, "w").write(txt)
print("PATCHED. Inserted 2.1 before section 3 and 6.1 before section 7.")
print("Verify: grep -n '### 2.1\\|### 6.1' RESULTS.md")
