#!/usr/bin/env python3
"""Add section 2.2 (next-skill at N=3000, all 3 targets) to RESULTS.md.

Those runs exist in the logs but were never collected into RESULTS.md. This
parses them fresh; nothing is transcribed by hand. Section 2.1 covers the same
budget for knowledge tracing, so 2.2 completes the source comparison on the
second task.

Run from the repo root (on the cluster, where the logs live):
    python3 tools/patches/add_nextskill_n3000_section.py
Idempotent.
"""
import glob
import os
import re
import statistics as st
import sys

PATH = "RESULTS.md"
LOGDIR = "."
TARGETS = ["assist2017", "ednet", "junyi"]
CONDS = ["scratch", "indomain", "fromednet", "fromjunyi", "fromassist"]
# banner: "=== next_skill (edubert_<t>_ns_<t>_<cond>_n3000_seed<S>, ...) ==="
# metric: "test macro-OVR AUC: 0.9813  (over 92 classes)"
NAME = re.compile(r"edubert_(\w+?)_ns_\w+?_(scratch|indomain|fromednet|fromjunyi|fromassist)"
                  r"_n3000_seed(\d+)")
METRIC = re.compile(r"test macro-OVR AUC:\s*([\d.]+)")


def collect():
    vals, classes = {}, {}
    for path in sorted(glob.glob(os.path.join(LOGDIR, "*.log"))):
        try:
            txt = open(path, errors="ignore").read()
        except OSError:
            continue
        for m in NAME.finditer(txt):
            tgt, cond, seed = m.group(1), m.group(2), int(m.group(3))
            if tgt not in TARGETS:
                continue
            # take the first macro-OVR line AFTER this run name, and only if it
            # is close enough to be part of the same block
            tail = txt[m.end():m.end() + 900]
            mm = METRIC.search(tail)
            if not mm:
                continue
            key = (tgt, cond, seed)
            if key not in vals:                       # first hit wins: guards resubmits
                vals[key] = float(mm.group(1))
                cm = re.search(r"\(over (\d+) classes\)", tail)
                if cm:
                    classes[tgt] = int(cm.group(1))
    return vals, classes


def main():
    if not os.path.exists(PATH):
        sys.exit("ERROR: RESULTS.md not found. Run from the repo root.")
    txt = open(PATH).read()
    if "### 2.2 Source comparison at N=3000" in txt:
        sys.exit("ALREADY PATCHED. Nothing done.")

    vals, classes = collect()
    if not vals:
        sys.exit("ERROR: no N=3000 next-skill runs parsed. Run this on the cluster, "
                 "in the directory holding the *.log files.")

    rows, reads = [], []
    for tgt in TARGETS:
        cells, gains = [], {}
        base = [vals[k] for k in vals if k[0] == tgt and k[1] == "scratch"]
        for cond in CONDS:
            v = [vals[k] for k in sorted(vals) if k[0] == tgt and k[1] == cond]
            if not v:
                cells.append("-")
                continue
            m = st.mean(v)
            sd = st.pstdev(v) if len(v) > 1 else 0.0
            if cond == "scratch":
                cells.append(f"{m:.4f} ±{sd:.4f} (n={len(v)})")
            else:
                g = m - st.mean(base) if base else float("nan")
                gains[cond] = g
                cells.append(f"{m:.4f} ±{sd:.4f} ({g:+.4f})")
        rows.append(f"| {tgt} ({classes.get(tgt, '?')} classes) | " + " | ".join(cells) + " |")
        if gains:
            best = max(gains, key=gains.get)
            reads.append(f"{tgt}: best is {best} at {gains[best]:+.4f}")

    block = ("\n### 2.2 Source comparison at N=3000, next-skill macro-OVR AUC (parsed from logs)\n\n"
             "Runs `edubert_<target>_ns_<t>_<cond>_n3000_seed{1,2,42}`, mean ±pstdev, gain vs "
             "scratch in parentheses. Same budget and same sources as 2.1, second task.\n\n"
             "| Target | scratch | indomain | fromednet | fromjunyi | fromassist |\n"
             "|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
             "_Read: " + "; ".join(reads) + ". Next-skill macro-OVR sits far above chance on all "
             "three targets, so the absolute values are compressed and the gaps are correspondingly "
             "small; read the ordering, not the magnitude, and compare against the knowledge-tracing "
             "gaps in 2.1 before making any source claim._\n")

    anchor = "\n## 3. "
    if txt.count(anchor) != 1:
        sys.exit(f"ERROR: anchor {anchor!r} not unique; check: grep -n '^## ' {PATH}")
    open(PATH, "w").write(txt.replace(anchor, block + "\n## 3. "))
    print(f"PATCHED. Inserted 2.2 before section 3, from {len(vals)} parsed runs.")
    for tgt in TARGETS:
        n = len([k for k in vals if k[0] == tgt])
        print(f"  {tgt}: {n} runs")
    print("Verify: grep -n '### 2.2' RESULTS.md")


if __name__ == "__main__":
    main()
