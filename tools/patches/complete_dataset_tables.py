#!/usr/bin/env python3
"""Complete the dataset characterization from the processed vocab_stats.md files.

Does four things, all from real files, nothing hardcoded:
  1. inserts RESULTS.md section 0.1, a full 7-dataset characterization table
  2. fills the two empty README rows (Algebra 2005, Bridge to Algebra 2006)
  3. fixes the hardcoded "# Algebra2005 vocab stats" title in
     scripts/preprocess_algebra2005.py, which is reused for the other KDD sets
  4. repairs the wrong title already written into bridge2006/vocab_stats.md and
     algebra2006/vocab_stats.md, without touching their numbers

Run from the repo root:  python3 tools/patches/complete_dataset_tables.py
Idempotent. Needs ../processed/<ds>/vocab_stats.md to exist (cluster only).
"""
import os
import re
import sys

RESULTS = "RESULTS.md"
README = "README.md"
PREP = "scripts/preprocess_algebra2005.py"
PROC = "../processed"

ORDER = ["assist2017", "ednet", "junyi", "algebra2005", "bridge2006", "assist2009", "algebra2006"]
PRETTY = {"assist2017": "ASSISTments 2017", "ednet": "EdNet KT1", "junyi": "Junyi Academy",
          "algebra2005": "Algebra 2005 (KDD Cup)", "bridge2006": "Bridge to Algebra 2006 (KDD Cup)",
          "assist2009": "ASSISTments 2009", "algebra2006": "Algebra 2006-2007 (KDD Cup)"}
FIELDS = [("students", r"students \(>= \d+ interactions\): ([\d.]+)"),
          ("skills", r"skills \([^)]*\): ([\d.]+)"),
          ("interactions", r"total interactions: ([\d.]+)"),
          ("median_len", r"median seq len: ([\d.]+)"),
          ("mean_len", r"mean seq len: ([\d.]+)"),
          ("base_rate", r"PROCESSED correct base rate: ([\d.]+)")]


def parse(ds):
    p = os.path.join(PROC, ds, "vocab_stats.md")
    if not os.path.exists(p):
        return None
    txt = open(p, errors="ignore").read()
    rec = {}
    for name, pat in FIELDS:
        m = re.search(pat, txt)
        rec[name] = float(m.group(1)) if m else None
    return rec


def main():
    for f in (RESULTS, README):
        if not os.path.exists(f):
            sys.exit(f"ERROR: {f} not found. Run from the repo root.")

    stats = {ds: parse(ds) for ds in ORDER}
    missing = [d for d, v in stats.items() if v is None]
    if missing:
        sys.exit(f"ERROR: no vocab_stats.md for {missing}. Expected under {PROC}/<ds>/. "
                 "This patch must run on the cluster.")

    # ---- 1. RESULTS.md section 0.1 -------------------------------------------
    res = open(RESULTS).read()
    if "### 0.1 Dataset characterization" in res:
        print("RESULTS.md 0.1: already present, skipped")
    else:
        rows = []
        for ds in ORDER:
            s = stats[ds]
            pps = s["median_len"] / s["skills"] if s["skills"] else float("nan")
            rows.append(f"| {PRETTY[ds]} | {int(s['students']):,} | {int(s['skills']):,} | "
                        f"{int(s['interactions']):,} | {s['median_len']:.0f} | {s['mean_len']:.0f} | "
                        f"{s['base_rate']:.4f} | {pps:.3f} |")
        block = ("\n### 0.1 Dataset characterization (parsed from processed/<ds>/vocab_stats.md)\n\n"
                 "| Dataset | Students | Skills | Interactions | Median len | Mean len | Correct rate | pps |\n"
                 "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
                 "_Read: pps here is recomputed as median length / skills and reproduces the section 5 "
                 "ordering. Correct rate is the PROCESSED base rate, i.e. after the min-interaction "
                 "filter, so it can differ from a raw-file rate quoted elsewhere._\n")
        anchor = "\n---\n## 1. "
        if res.count(anchor) != 1:
            sys.exit(f"ERROR: anchor {anchor!r} not unique in {RESULTS}")
        open(RESULTS, "w").write(res.replace(anchor, block + "\n---\n## 1. "))
        print("RESULTS.md: inserted 0.1")

    # ---- 2. README rows -------------------------------------------------------
    rd = open(README).read()
    filled = 0
    for ds in ("algebra2005", "bridge2006"):
        s = stats[ds]
        pat = re.compile(r"(\| " + re.escape(PRETTY[ds]) + r" \| )[^|]*(\| )[^|]*(\|)")
        m = pat.search(rd)
        if not m:
            print(f"README: row for {PRETTY[ds]!r} not found, skipped")
            continue
        if m.group(0).count("—") == 0:
            print(f"README: {ds} row already filled, skipped")
            continue
        rd = pat.sub(lambda mm, s=s: f"{mm.group(1)}{int(s['students']):,} {mm.group(2)}"
                                     f"{int(s['skills']):,} {mm.group(3)}", rd, count=1)
        filled += 1
    if filled:
        open(README, "w").write(rd)
    print(f"README: filled {filled} row(s)")

    # ---- 3. preprocessing script title ---------------------------------------
    if os.path.exists(PREP):
        src = open(PREP).read()
        old = 'f"# Algebra2005 vocab stats\\n\\n"'
        new = 'f"# {Path(args.out_dir).name} vocab stats\\n\\n"'
        if new in src:
            print(f"{PREP}: already fixed, skipped")
        elif old in src:
            if "from pathlib import Path" not in src and "import pathlib" not in src:
                print(f"WARNING: {PREP} has no pathlib import; using os.path instead")
                new = 'f"# {os.path.basename(str(args.out_dir).rstrip(chr(47)))} vocab stats\\n\\n"'
            open(PREP, "w").write(src.replace(old, new, 1))
            print(f"{PREP}: title now derives from --out_dir")
        else:
            print(f"{PREP}: title line not found verbatim, skipped")

    # ---- 4. repair titles already written to disk -----------------------------
    for ds in ORDER:
        p = os.path.join(PROC, ds, "vocab_stats.md")
        txt = open(p, errors="ignore").read()
        first = txt.split("\n", 1)[0]
        want = f"# {ds} vocab stats"
        if first.startswith("# ") and first != want and "vocab stats" in first:
            open(p, "w").write(want + txt[len(first):])
            print(f"{p}: title {first!r} -> {want!r} (numbers untouched)")

    # ---- 5. self-check: recomputed pps must agree with the section 5 table ----
    res2 = open(RESULTS).read()
    sec5 = dict(re.findall(r"\| (Junyi|EdNet|ASSIST2009|Algebra2006|Bridge2006|ASSIST2017|Algebra2005)"
                           r" \| ([\d.]+) \|", res2))
    alias = {"assist2017": "ASSIST2017", "ednet": "EdNet", "junyi": "Junyi",
             "algebra2005": "Algebra2005", "bridge2006": "Bridge2006",
             "assist2009": "ASSIST2009", "algebra2006": "Algebra2006"}
    print("\npps cross-check, recomputed vs section 5:")
    bad = 0
    for ds in ORDER:
        s = stats[ds]
        got = s["median_len"] / s["skills"] if s["skills"] else float("nan")
        want = float(sec5.get(alias[ds], "nan"))
        flag = "" if abs(got - want) < 0.011 else "   <-- MISMATCH, check rounding or a stale table"
        bad += bool(flag)
        print(f"  {alias[ds]:12s} {got:7.3f} vs {want:7.3f}{flag}")
    if bad:
        print(f"  {bad} mismatch(es). The 0.1 table is computed from vocab_stats.md and is the "
              "primary source; reconcile section 5 by hand if the difference is real.")

    print("\nVerify: grep -n '### 0.1' RESULTS.md ; sed -n '10,22p' README.md")


if __name__ == "__main__":
    main()
