#!/usr/bin/env python3
"""Complete the dataset characterization from the processed vocab_stats.md files.

The seven datasets were written by different preprocessing scripts, so the
vocab_stats.md field wording is not uniform. This version tries several phrasings
per field, never crashes on a field it cannot find, and prints the contents of any
file it could not fully parse so the missing pattern can be added.

Does four things, all from real files:
  1. RESULTS.md section 0.1, a 7-dataset characterization table
  2. fills the two empty README rows (Algebra 2005, Bridge to Algebra 2006)
  3. fixes the hardcoded title in scripts/preprocess_algebra2005.py
  4. repairs wrong titles already written into processed/<ds>/vocab_stats.md

Each step runs independently: one failing does not block the others.

Run from the repo root:  python3 tools/patches/complete_dataset_tables.py
Idempotent.
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

PATTERNS = {
 "students": [r"students \(>=[^)]*\):\s*([\d,]+)", r"\bn_students\b\s*[:=]\s*([\d,]+)",
              r"\bnum_students\b\s*[:=]\s*([\d,]+)", r"students(?:[^:\n]*)[:=]\s*([\d,]+)"],
 "skills":   [r"skills \([^)]*\):\s*([\d,]+)", r"\bn_skills\b\s*[:=]\s*([\d,]+)",
              r"\bnum_skills\b\s*[:=]\s*([\d,]+)", r"vocab size\s*[:=]\s*([\d,]+)",
              r"skills(?:[^:\n]*)[:=]\s*([\d,]+)"],
 # NOTE: no loose "interactions" fallback here. "Min interactions filter : 10"
 # matched it and put 10 in the table for three datasets.
 "interactions": [r"total interactions\s*[:=]\s*([\d,]+)",
                  r"Rows after all filters\s*[:=]\s*([\d,]+)",
                  r"Rows kept \(after meta join\)\s*[:=]\s*([\d,]+)",
                  r"\bn_interactions\b\s*[:=]\s*([\d,]+)"],
 "median_len": [r"median seq len\s*[:=]\s*([\d.]+)",
                r"median (?:sequence )?len(?:gth)?\s*[:=]\s*([\d.]+)",
                r"^\s*median\s*[:=]\s*([\d.]+)",
                r"\bmedian\s+([\d.]+)"],
 "mean_len":   [r"mean seq len\s*[:=]\s*([\d.]+)",
                r"mean (?:sequence )?len(?:gth)?\s*[:=]\s*([\d.]+)",
                r"^\s*mean\s*[:=]\s*([\d.]+)",
                r"\bmean\s+([\d.]+)"],
 "base_rate":  [r"PROCESSED correct base rate\s*[:=]\s*([\d.]+)",
                r"Fraction correct\s*[:=]\s*([\d.]+)",
                r"correct(?: base)? rate\s*[:=]\s*([\d.]+)",
                r"\bfrac_correct\b\s*[:=]\s*([\d.]+)", r"base rate\s*[:=]\s*([\d.]+)"],
}


def num(x):
    try:
        return float(x.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def parse(ds):
    p = os.path.join(PROC, ds, "vocab_stats.md")
    if not os.path.exists(p):
        return None, None
    txt = open(p, errors="ignore").read()
    rec = {}
    for field, pats in PATTERNS.items():
        rec[field] = None
        for pat in pats:
            m = re.search(pat, txt, re.I | re.M)
            if m:
                rec[field] = num(m.group(1))
                break
    # structural guard: the min-interaction filter is 10, so a dataset cannot have
    # fewer than 10*students interactions. This catches a stray match such as
    # "Min interactions filter : 10".
    if rec["interactions"] is not None and rec["students"]:
        if rec["interactions"] < 10 * rec["students"]:
            rec["_bad_interactions"] = rec["interactions"]
            rec["interactions"] = None
    return rec, txt


def cell(v, kind="int"):
    if v is None:
        return "-"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "len":
        return f"{v:.1f}"
    return f"{v:.4f}"


def fix_section5(text, stats, alias):
    """Rewrite the section 5 pps column from the processed files.

    Scoped to the pps table only. An unscoped regex matched the section 1
    baseline table first and rewrote an AUC as a pps value, so the block is
    located by its header and edits are confined to it.
    """
    changed = []
    hdr = "| Dataset | pps | regime |"
    i = text.find(hdr)
    if i < 0:
        return text, changed, "section 5 pps table header not found; nothing changed"
    j = text.find("\n\n", i)
    if j < 0:
        return text, changed, "could not find the end of the pps table; nothing changed"
    block = text[i:j]

    for ds, key in alias.items():
        s = stats.get(ds)
        if not s or s["median_len"] is None or not s["skills"]:
            continue
        val = s["median_len"] / s["skills"]
        pat = re.compile(r"(\| " + re.escape(key) + r" \| )([\d.]+)( \|)")
        m = pat.search(block)
        if m and abs(float(m.group(2)) - val) >= 0.0015:
            changed.append((key, m.group(2), f"{val:.3f}"))
            block = pat.sub(lambda mm: f"{mm.group(1)}{val:.3f}{mm.group(3)}", block, count=1)
    text = text[:i] + block + text[j:]

    # prose, only on lines that already talk about pps
    lines = text.split("\n")
    for n, line in enumerate(lines):
        if "pps" not in line.lower():
            continue
        for ds, key in alias.items():
            s = stats.get(ds)
            if not s or s["median_len"] is None or not s["skills"]:
                continue
            val = s["median_len"] / s["skills"]
            pat = re.compile(r"(" + re.escape(key.lower()) + r"(?:-07)? at (?:pps )?)([\d.]+)", re.I)
            m = pat.search(line)
            if m and abs(float(m.group(2)) - val) >= 0.0015:
                changed.append((key + " (prose)", m.group(2), f"{val:.3f}"))
                lines[n] = pat.sub(lambda x: f"{x.group(1)}{val:.3f}", line, count=1)
                line = lines[n]
    return "\n".join(lines), changed, None


def main():
    refresh = "--refresh" in sys.argv
    fix5 = "--fix-section5" in sys.argv
    for f in (RESULTS, README):
        if not os.path.exists(f):
            sys.exit(f"ERROR: {f} not found. Run from the repo root.")

    stats, raw = {}, {}
    for ds in ORDER:
        stats[ds], raw[ds] = parse(ds)

    absent = [d for d in ORDER if stats[d] is None]
    if absent:
        print(f"WARNING: no vocab_stats.md for {absent} under {PROC}/<ds>/")

    gaps = {d: [f for f, v in stats[d].items() if v is None and not f.startswith("_")]
            for d in ORDER if stats[d]}
    gaps = {d: g for d, g in gaps.items() if g}
    for d in ORDER:
        if stats[d] and "_bad_interactions" in stats[d]:
            print(f"REJECTED for {d}: interactions={stats[d]['_bad_interactions']:,} is below "
                  f"10 x {int(stats[d]['students']):,} students; a filter line was matched by mistake")
    if gaps:
        print("\nFIELDS NOT PARSED:")
        for d, g in gaps.items():
            print(f"  {d}: {', '.join(g)}")
        print("\nContents of the files with gaps, so the patterns can be extended:")
        for d in gaps:
            print(f"--- {PROC}/{d}/vocab_stats.md")
            for line in raw[d].strip().split("\n")[:16]:
                print(f"    {line}")
        print()

    res = open(RESULTS).read()
    if "### 0.1 Dataset characterization" in res and refresh:
        res = re.sub(r"\n### 0\.1 Dataset characterization.*?(?=\n---\n## 1\. )", "", res, flags=re.S)
        open(RESULTS, "w").write(res)
        print("RESULTS.md 0.1: removed old block (--refresh)")
    if "### 0.1 Dataset characterization" in res:
        print("RESULTS.md 0.1: already present, skipped (use --refresh to rebuild)")
    else:
        rows = []
        for ds in ORDER:
            s = stats[ds]
            if s is None:
                rows.append(f"| {PRETTY[ds]} | - | - | - | - | - | - | - |")
                continue
            pps = (s["median_len"] / s["skills"]
                   if s["median_len"] is not None and s["skills"] else None)
            rows.append(
                f"| {PRETTY[ds]} | {cell(s['students'])} | {cell(s['skills'])} | "
                f"{cell(s['interactions'])} | {cell(s['median_len'], 'len')} | "
                f"{cell(s['mean_len'], 'len')} | {cell(s['base_rate'], 'rate')} | "
                + ("-" if pps is None else f"{pps:.3f}") + " |")
        block = ("\n### 0.1 Dataset characterization (parsed from processed/<ds>/vocab_stats.md)\n\n"
                 "| Dataset | Students | Skills | Interactions | Median len | Mean len | Correct rate | pps |\n"
                 "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
                 "_Read: pps here is recomputed as median length / skills and reproduces the section 5 "
                 "ordering. Correct rate is the PROCESSED base rate, after the min-interaction filter, "
                 "so it can differ from a raw-file rate quoted elsewhere. A dash means the field is not "
                 "recorded in that dataset's vocab_stats.md._\n")
        anchor = "\n---\n## 1. "
        if res.count(anchor) != 1:
            print(f"ERROR: anchor {anchor!r} not unique in {RESULTS}; 0.1 NOT inserted")
        else:
            open(RESULTS, "w").write(res.replace(anchor, block + "\n---\n## 1. "))
            print("RESULTS.md: inserted 0.1")

    rd = open(README).read()
    filled = 0
    for ds in ("algebra2005", "bridge2006"):
        s = stats.get(ds)
        if not s or s["students"] is None or s["skills"] is None:
            print(f"README: {ds} students/skills unparsed, row left alone")
            continue
        pat = re.compile(r"(\| " + re.escape(PRETTY[ds]) + r" \| )[^|]*(\| )[^|]*(\|)")
        m = pat.search(rd)
        if not m:
            print(f"README: row for {PRETTY[ds]!r} not found, skipped")
            continue
        if not re.search(r"\|\s*[—-]\s*\|", m.group(0)):
            print(f"README: {ds} row already filled, skipped")
            continue
        rd = pat.sub(lambda mm, s=s: f"{mm.group(1)}{int(s['students']):,} {mm.group(2)}"
                                     f"{int(s['skills']):,} {mm.group(3)}", rd, count=1)
        filled += 1
    if filled:
        open(README, "w").write(rd)
    print(f"README: filled {filled} row(s)")

    if os.path.exists(PREP):
        src = open(PREP).read()
        old = 'f"# Algebra2005 vocab stats\\n\\n"'
        new = 'f"# {Path(args.out_dir).name} vocab stats\\n\\n"'
        if new in src:
            print(f"{PREP}: already fixed, skipped")
        elif old in src:
            if "from pathlib import Path" not in src and "import pathlib" not in src:
                new = 'f"# {os.path.basename(str(args.out_dir).rstrip(chr(47)))} vocab stats\\n\\n"'
            open(PREP, "w").write(src.replace(old, new, 1))
            print(f"{PREP}: title now derives from --out_dir")
        else:
            print(f"{PREP}: title line not found verbatim, skipped")

    for ds in ORDER:
        p = os.path.join(PROC, ds, "vocab_stats.md")
        if not os.path.exists(p):
            continue
        txt = open(p, errors="ignore").read()
        first = txt.split("\n", 1)[0]
        want = f"# {ds} vocab stats"
        if first.startswith("# ") and first != want and "vocab stats" in first.lower():
            open(p, "w").write(want + txt[len(first):])
            print(f"{p}: title {first!r} -> {want!r} (numbers untouched)")

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
        if not s or s["median_len"] is None or not s["skills"]:
            print(f"  {alias[ds]:12s} not computable (missing median_len or skills)")
            continue
        got = s["median_len"] / s["skills"]
        want = float(sec5.get(alias[ds], "nan"))
        flag = "" if abs(got - want) < 0.011 else "   <-- MISMATCH, check rounding or a stale table"
        bad += bool(flag)
        print(f"  {alias[ds]:12s} {got:7.3f} vs {want:7.3f}   delta {got-want:+.4f}{flag}")
    if bad:
        print(f"  {bad} mismatch(es). The 0.1 table comes from vocab_stats.md and is the primary "
              "source; reconcile section 5 by hand if the difference is real.")

    if fix5:
        res3 = open(RESULTS).read()
        res3, changed, err = fix_section5(res3, stats, alias)
        if err:
            print(f"\nsection 5: {err}")
        elif changed:
            open(RESULTS, "w").write(res3)
            print("\nsection 5 corrected from the processed files:")
            for k, was, now in changed:
                print(f"  {k}: {was} -> {now}")
        else:
            print("\nsection 5: nothing to correct")
    elif bad:
        print("  rerun with --fix-section5 to correct these in place")

    print("\nVerify: grep -n '### 0.1' RESULTS.md ; sed -n '11,20p' README.md")


if __name__ == "__main__":
    main()
