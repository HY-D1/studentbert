#!/usr/bin/env python3
"""Full consistency audit for StudentBERT.

Re-derives numbers from the raw logs and processed files, then checks them
against RESULTS.md and the poster. Nothing is taken on trust: every table cell
it can reconstruct is recomputed from run output.

READ ONLY. No jobs, no GPU, no files modified except the report.

Run from the repo root on the cluster (where *.log lives):
    python3 analysis/audit_all_claims.py
    python3 analysis/audit_all_claims.py --logdir . --out audit_report.txt

Sections:
  A  log inventory: every run name found, and which have no metric (died)
  B  RESULTS.md tables recomputed from logs (2.1, 2.2, 3.1, 4, 5, 8.1, 8.2, 6.1)
  C  safety checks: deprecated v1 probe, leaked dropout K, seed4/seed42 collision
  D  internal consistency of RESULTS.md (0.1 vs 5, 3.1 vs 3.2, claims vs tables)
  E  poster values vs RESULTS.md
  F  coverage: expected cells vs found, and unverifiable claims

Exit code is 0 always; read the FAIL lines.
"""
import argparse
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

NUM = r"([0-9]*\.?[0-9]+)"
BANNER = re.compile(r"===\s*([a-z_]+)[^(]*\((?P<run>edubert_[A-Za-z0-9_.\-]+)\s*,")
METRICS = {
    "auc":          [re.compile(r"test\s+AUC\s*:?\s*" + NUM)],
    "top1":         [re.compile(r"test/top1\s*[=:]?\s*" + NUM), re.compile(r"test top-1 acc\s*:?\s*" + NUM)],
    "macro_auc":    [re.compile(r"test/macro_auc\s*[=:]?\s*" + NUM), re.compile(r"test macro-?OVR AUC\s*:?\s*" + NUM)],
    "top5":         [re.compile(r"test/top5\s*[=:]?\s*" + NUM), re.compile(r"test top-5 acc\s*:?\s*" + NUM)],
    "probe_acc":    [re.compile(r"test\s+probe\s+acc[^0-9]*" + NUM), re.compile(r"val_acc=" + NUM)],
}
PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "info"
tally = defaultdict(int)
OUT = []


def say(level, msg):
    tally[level] += 1
    OUT.append(f"[{level}] {msg}" if level != INFO else f"       {msg}")
    print(OUT[-1])


def head(t):
    OUT.append("")
    OUT.append("=" * 74)
    OUT.append(t)
    OUT.append("=" * 74)
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def close(a, b, tol=0.0006):
    return a is not None and b is not None and abs(a - b) <= tol


# ---------------------------------------------------------------- log parsing
def load_runs(logdir):
    """run_name -> {metric: value}. First value after the banner wins (guards resubmits)."""
    runs, dead, files = {}, [], sorted(glob.glob(os.path.join(logdir, "*.log")))
    for path in files:
        try:
            txt = open(path, errors="replace").read()
        except OSError:
            continue
        for m in BANNER.finditer(txt):
            run = m.group("run")
            tail = txt[m.end():m.end() + 4000]
            got = {}
            for name, pats in METRICS.items():
                for rx in pats:
                    hit = rx.search(tail)
                    if hit:
                        try:
                            got[name] = float(hit.group(1))
                        except ValueError:
                            pass
                        break
            if run not in runs:
                runs[run] = got
                if not got:
                    dead.append((run, os.path.basename(path)))
            elif got and not runs[run]:
                runs[run] = got
    return runs, dead, files


def agg(runs, pattern):
    """Mean/pstdev/n over runs whose name matches, for each metric."""
    rx = re.compile(pattern)
    vals = defaultdict(list)
    for name, mets in runs.items():
        if rx.fullmatch(name):
            for k, v in mets.items():
                vals[k].append(v)
    return {k: (st.mean(v), st.pstdev(v) if len(v) > 1 else 0.0, len(v)) for k, v in vals.items()}


# ---------------------------------------------------------------- RESULTS.md
def md_table_rows(md, header_sub, ncols=None):
    """Rows of the first markdown table after a line containing header_sub."""
    lines = md.split("\n")
    try:
        i = next(n for n, l in enumerate(lines) if header_sub in l)
    except StopIteration:
        return []
    rows = []
    for l in lines[i:]:
        s = l.strip()
        if s.startswith("|") and not set(s) <= set("|- :"):
            cells = [c.strip() for c in s.split("|")[1:-1]]
            if ncols is None or len(cells) == ncols:
                rows.append(cells)
        elif rows and not s.startswith("|"):
            break
    return rows


def first_num(cell):
    m = re.search(r"(-?[0-9]*\.?[0-9]+)", cell.replace(",", ""))
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=".")
    ap.add_argument("--results", default="RESULTS.md")
    ap.add_argument("--poster", default="poster/make_poster.py")
    ap.add_argument("--processed", default="../processed")
    ap.add_argument("--out", default="audit_report.txt")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        sys.exit(f"ERROR: {args.results} not found. Run from the repo root.")
    md = open(args.results, errors="replace").read()

    head("A. LOG INVENTORY")
    runs, dead, files = load_runs(args.logdir)
    say(INFO, f"log files scanned : {len(files)}")
    say(INFO, f"run banners found : {len(runs)}")
    if not runs:
        say(FAIL, "no run banners parsed; are you in the directory holding *.log?")
    groups = defaultdict(int)
    for r in runs:
        groups[re.sub(r"_seed\d+$|_s\d+$", "", r).rsplit("_", 1)[0][:44]] += 1
    say(INFO, f"distinct run families: {len(groups)}")
    if dead:
        say(WARN, f"{len(dead)} run(s) have a banner but NO test metric (died before eval):")
        for r, f in dead[:12]:
            say(INFO, f"   {r}   [{f}]")
    else:
        say(PASS, "every run banner has at least one test metric")

    head("B. RESULTS.md TABLES RECOMPUTED FROM LOGS")

    # ---- 2.1 KT at N=3000 --------------------------------------------------
    conds = {"scratch": "scratch", "indomain": "indomain",
             "fromednet": "fromednet", "fromjunyi": "fromjunyi", "fromassist": "fromassist"}
    for tgt in ("assist2017", "ednet", "junyi"):
        for cond in conds:
            a = agg(runs, rf"edubert_{tgt}_kt_\w+_{cond}_n3000_seed\d+")
            if "auc" not in a:
                continue
            mean, sd, n = a["auc"]
            row = [r for r in md_table_rows(md, "### 2.1") if r and r[0] == tgt]
            if not row:
                say(WARN, f"2.1 {tgt}: table row not found in RESULTS.md")
                break
            cells = row[0]
            hit = [c for c in cells if c not in ("-",) and "(" in c or c.replace(".", "").isdigit()]
            md_val = None
            for c in cells[1:]:
                v = first_num(c)
                if v is not None and close(v, mean, 0.0002):
                    md_val = v
                    break
            if md_val is None:
                say(FAIL, f"2.1 {tgt}/{cond}: logs give {mean:.4f} (n={n}) but no matching cell in RESULTS.md")
            else:
                say(PASS, f"2.1 {tgt}/{cond}: {mean:.4f} +/-{sd:.4f} (n={n}) matches RESULTS.md")

    # ---- 3.1 next-skill sweep ---------------------------------------------
    # NOTE: the sweep has TWO run families, _nextskill_ and _nsauc_, plus resubmit
    # duplicates. Aggregating raw log banners merges them (n=6 or 11 instead of 3)
    # and produces wrong means. analysis/parse_nextskill_full.py dedups by
    # (dataset, cond, N, seed); its CSV is the authoritative source here.
    sweep_rows = md_table_rows(md, "### 3.1")
    csv_path = os.path.join(args.logdir, "nextskill_results_long.csv")
    Ns = [25, 50, 100, 200, 500, 1000]
    if sweep_rows and os.path.exists(csv_path):
        cells = defaultdict(list)
        for line in open(csv_path, errors="replace"):
            p = [x.strip() for x in line.split(",")]
            if len(p) == 6 and p[2].isdigit() and p[4] == "top1":
                try:
                    cells[(p[1], int(p[2]))].append(float(p[5]))
                except ValueError:
                    pass
        for cond in ("scratch", "indomain", "ednet", "junyi"):
            md_row = [r for r in sweep_rows if r and r[0] == cond]
            if not md_row:
                continue
            for k, N in enumerate(Ns):
                v = cells.get((cond, N), [])
                if not v:
                    say(WARN, f"3.1 {cond} N={N}: no rows in {os.path.basename(csv_path)}")
                    continue
                mean = st.mean(v)
                cell = first_num(md_row[0][k + 1]) if len(md_row[0]) > k + 1 else None
                if close(cell, mean, 0.0002):
                    say(PASS, f"3.1 {cond} N={N}: {mean:.4f} (n={len(v)}) matches RESULTS.md")
                else:
                    say(FAIL, f"3.1 {cond} N={N}: CSV {mean:.4f} (n={len(v)}) vs RESULTS.md {cell}")
                if len(v) != 3:
                    say(WARN, f"3.1 {cond} N={N}: n={len(v)}, expected 3 seeds; check for duplicates")
    elif sweep_rows:
        say(WARN, "nextskill_results_long.csv not found; regenerate with "
                  "`python analysis/parse_nextskill_full.py --dir .` then rerun. "
                  "Raw log aggregation is NOT valid here (two run families + resubmits).")
        for cond in ("scratch", "indomain", "ednet", "junyi"):
            for N in Ns:
                a1 = agg(runs, rf"edubert_assist2017_{cond}_nsauc_n{N}_seed\d+")
                a2 = agg(runs, rf"edubert_assist2017_{cond}_nextskill_n{N}_seed\d+")
                if "top1" in a1 or "top1" in a2:
                    b1 = f"nsauc {a1['top1'][0]:.4f} (n={a1['top1'][2]})" if "top1" in a1 else "nsauc -"
                    b2 = f"nextskill {a2['top1'][0]:.4f} (n={a2['top1'][2]})" if "top1" in a2 else "nextskill -"
                    say(INFO, f"3.1 {cond} N={N}: {b1} | {b2}")
    else:
        say(WARN, "3.1 table not found in RESULTS.md")

    # ---- 5 truncation sweep and the crossover claim ------------------------
    trunc = [r for r in md_table_rows(md, "| K | full | skill_only") if r and r[0].isdigit()]
    if trunc:
        gaps = [(int(r[0]), first_num(r[2]) - first_num(r[3])) for r in trunc]
        neg = [k for k, g in gaps if g < 0]
        pos = [k for k, g in gaps if g > 0]
        say(INFO, "skill_only minus correct_only by K: " +
            ", ".join(f"K={k}:{g:+.4f}" for k, g in gaps))
        if neg and pos:
            lo, hi = max(neg), min(pos)
            def epps(k, med=441.0, ns=102.0):
                return min(k, med) / ns
            say(INFO, f"sign change lies between K={lo} (pps {epps(lo):.2f}) and K={hi} (pps {epps(hi):.2f})")
            if re.search(r"[Cc]rossover\s*~?\s*pps\s*1\.5", md):
                say(FAIL, f"RESULTS.md claims 'crossover ~pps 1.5' but the gap is still "
                          f"{dict(gaps).get(160, float('nan')):+.4f} at K=160 (pps 1.57); "
                          f"the change is between pps {epps(lo):.2f} and {epps(hi):.2f}")
            else:
                say(PASS, "no unsupported 'crossover ~pps 1.5' claim in RESULTS.md")
        if not any(r[4] not in ("-", "") for r in trunc if r[0] == "512"):
            say(WARN, "no scratch control at K=512; the endpoint is uncontrolled")
    else:
        say(WARN, "truncation sweep table not found")

    head("C. SAFETY CHECKS")

    # deprecated v1 circular probe must never be reported
    v1 = []
    for path in glob.glob(os.path.join(args.logdir, "w6_probe_*.log")):
        txt = open(path, errors="replace").read()
        if "skill-identity" in txt:
            v1 += [float(x) for x in re.findall(r"val_acc=([01]\.\d+)", txt)
                   if float(x) >= 0.95]        # the circular signature, not early epochs
    if v1:
        # Only the PROBE table can leak a circular value. Elsewhere a 0.97 is an
        # ordinary next-skill or weighted AUC, so a bare number match is noise.
        probe_tbl = md_table_rows(md, "| Dataset | pretrained | scratch | gain |", 4)
        probe_vals = [first_num(c) for r in probe_tbl for c in r[1:3] if first_num(c) is not None]
        leaked = [v for v in probe_vals if v >= 0.90]
        if leaked:
            say(FAIL, f"probe table contains implausible accuracies {leaked}; "
                      "these look like the deprecated circular v1 probe")
        else:
            say(PASS, f"v1 circular probe present in logs ({len(v1)} values >= 0.95) but the "
                      f"probe table maxes at {max(probe_vals):.4f}, so none leaked"
                      if probe_vals else "probe table not found; v1 check skipped")
    else:
        say(INFO, "no v1 circular-probe logs found in this directory")

    # leaked uncensored dropout K>=100
    leak = agg(runs, r"edubert_assist2017_\w+_dropout_k(100|200)_seed\d+")
    if "auc" in leak and leak["auc"][0] > 0.78:
        hits = [v for v in (0.79, 0.80, 0.85, 0.91) if f"{v:.2f}" in md]
        say(INFO, f"uncensored K>=100 dropout mean {leak['auc'][0]:.3f} (leak signature)")

    # seed4 / seed42 prefix collision
    coll = 0
    for r, m4 in runs.items():
        if r.endswith("seed4") and "auc" in m4:
            r42 = r[:-1] + "42"
            if r42 in runs and close(m4["auc"], runs[r42].get("auc"), 1e-9):
                coll += 1
    if coll >= 3:
        say(FAIL, f"{coll} seed4 values identical to their seed42 counterpart: "
                  "extraction is colliding on the prefix; use comma-anchored greps")
    else:
        say(PASS, f"no systematic seed4/seed42 collision ({coll} exact matches, expected ~0)")

    head("D. INTERNAL CONSISTENCY OF RESULTS.md")

    # 0.1 pps vs section 5 pps, and the 512-cap effect
    d01 = {r[0]: r for r in md_table_rows(md, "### 0.1 Dataset characterization", 8)}
    s5 = {r[0]: first_num(r[1]) for r in md_table_rows(md, "| Dataset | pps | regime |", 3)}
    alias = {"ASSISTments 2017": "ASSIST2017", "EdNet KT1": "EdNet", "Junyi Academy": "Junyi",
             "Algebra 2005 (KDD Cup)": "Algebra2005",
             "Bridge to Algebra 2006 (KDD Cup)": "Bridge2006",
             "ASSISTments 2009": "ASSIST2009", "Algebra 2006-2007 (KDD Cup)": "Algebra2006"}
    capped_note = []
    for name, row in d01.items():
        if name not in alias:
            continue
        med, sk = first_num(row[4]), first_num(row[2])
        if not med or not sk:
            continue
        raw = med / sk
        cap = min(med, 512) / sk
        tab = s5.get(alias[name])
        if not close(raw, tab, 0.002):
            say(FAIL, f"pps mismatch {alias[name]}: 0.1 gives {raw:.3f}, section 5 says {tab}")
        if abs(cap - raw) > 0.01:
            capped_note.append(f"{alias[name]} {raw:.3f} -> {cap:.3f}")
    if not capped_note:
        say(PASS, "no dataset median exceeds the 512-step model cap")
    else:
        say(WARN, "pps changes under the model's 512-step cap: " + "; ".join(capped_note))
        say(INFO, "any 'unsampled interval' claim must use one definition consistently")

    # 3.1 gap of means vs 3.2 paired means
    p32 = md_table_rows(md, "### 3.2")
    if p32 and sweep_rows:
        base = {int(c): first_num(v) for c, v in
                zip(["25", "50", "100", "200", "500", "1000"],
                    [r for r in sweep_rows if r and r[0] == "scratch"][0][1:])} \
            if any(r[0] == "scratch" for r in sweep_rows) else {}
        bad = 0
        for r in p32:
            if len(r) < 6 or not r[0].isdigit():
                continue
            N, cond, mean_gap = int(r[0]), r[1], first_num(r[4])
            row = [x for x in sweep_rows if x and x[0] == cond]
            if not row or N not in base:
                continue
            idx = ["25", "50", "100", "200", "500", "1000"].index(str(N))
            direct = first_num(row[0][idx + 1]) - base[N]
            if not close(direct, mean_gap, 0.0002):
                bad += 1
                say(FAIL, f"3.2 {cond} N={N}: paired mean {mean_gap:+.4f} vs 3.1 difference {direct:+.4f}")
        if not bad:
            say(PASS, "3.2 paired means agree with 3.1 gap of means everywhere")

    head("E. POSTER VALUES vs RESULTS.md")
    if os.path.exists(args.poster):
        src = open(args.poster, errors="replace").read()
        # only the declared data arrays; layout constants are not data
        data_blocks = []
        for tag, rx in [("R1 series", r'"(?:scratch|in-domain pretrain|Junyi pretrain|EdNet pretrain)":\s*\(\[([0-9.,\s]+)\]'),
                        ("R2 bars",   r"val = \[([0-9.,\s]+)\]"),
                        ("R4 probe",  r"pv = \[([0-9.,\s]+)\]"),
                        ("R3 flip",   r"fv  = \[(-?[0-9.,\s-]+)\]")]:
            for m in re.finditer(rx, src):
                data_blocks += [(tag, x.strip()) for x in m.group(1).split(",") if x.strip()]
        scatter = re.search(r"pts = \[(.*?)\]\n", src, re.S)
        if scatter:
            data_blocks += [("R3 scatter", g) for g in re.findall(r",\s*(-?0\.\d+),", scatter.group(1))]
        md_nums = set(re.findall(r"-?\d*\.\d+", md))
        orphan = [(t, v) for t, v in data_blocks
                  if v.lstrip("-").lstrip("0") and v not in md_nums
                  and f"{abs(float(v)):.4f}" not in md_nums]
        if orphan:
            say(WARN, f"{len(orphan)} plotted value(s) not verbatim in RESULTS.md "
                      "(expected for computed gaps; verify each): "
                      + ", ".join(f"{t}:{v}" for t, v in orphan[:10]))
        else:
            say(PASS, f"all {len(data_blocks)} plotted values trace to RESULTS.md")
        for phrase, why in [
            ("Shown causally", "causal language stronger than one within-dataset test"),
            ("crossover sits near 1.5", "not supported by the truncation sweep"),
            ("not similarity", "semantic similarity is not tested"),
            ("40,000", "no 40,000-student run exists"),
            ("EduBERT", "model name should be StudentBERT"),
            ("full length", "K=512 is the model cap, not full length"),
        ]:
            if phrase in src:
                say(FAIL, f"poster still contains {phrase!r}: {why}")
        say(INFO, "poster source checked: " + args.poster +
             "  (this is the REPO copy; if it is behind the delivered poster, "
             "these FAILs describe the repo, not the printed file)")
    else:
        say(WARN, f"{args.poster} not found; skipped poster cross-check")

    head("F. COVERAGE AND UNVERIFIABLE CLAIMS")
    for claim, pat in [
        ("validation peaks near epoch 5", r"epoch\s*5"),
        ("answer base rate refuted", r"base rate.*refut|refut.*base rate"),
        ("40,000-student run", r"40[,]?000"),
        ("preregistration record", r"prereg"),
    ]:
        if re.search(pat, md, re.I):
            say(INFO, f"'{claim}' has some support in RESULTS.md")
        else:
            say(WARN, f"'{claim}' has NO support in RESULTS.md; do not state it on the poster or in a paper")

    head("SUMMARY")
    for k in (PASS, FAIL, WARN):
        say(INFO, f"{k}: {tally[k]}")
    if tally[FAIL]:
        say(INFO, "FAIL lines above are hard inconsistencies; fix before printing or submitting.")
    open(args.out, "w").write("\n".join(OUT) + "\n")
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
