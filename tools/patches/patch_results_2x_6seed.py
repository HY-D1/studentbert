#!/usr/bin/env python3
# Rebuild RESULTS.md sections 2.1, 2.2 and 3 from the 6-seed campaign.
#
# Companion to patch_results_base2.py, which corrected sections 1, 2, 8 and
# 8.1. Sections 2.1 and 2.2 were left at 3 seeds and now contradict the
# NeurIPS paper's Tables 3 and 4. They also feed analysis/make_neurips_figure.py,
# which parses 2.1 directly, so the figure would print 3-seed values beside
# 6-seed tables until this runs.
#
# Sections 2.1 and 2.2 are replaced whole, delimited by their headings rather
# than by exact body text, because the campaign changed every cell. The old
# block is printed before writing so nothing is discarded silently. The two
# section 3 bullets use exact anchors.
#
# Idempotent and atomic: if a replacement is already present it is skipped,
# and if any anchor is ambiguous nothing is written.
#
#   python3 tools/patches/patch_results_2x_6seed.py --file RESULTS.md --dry-run
#   python3 tools/patches/patch_results_2x_6seed.py --file RESULTS.md

from __future__ import annotations

import argparse
import hashlib
import sys

H21 = "### 2.1 "
H22 = "### 2.2 "
H3 = "## 3. "

NEW_21 = """### 2.1 Source comparison at N=3000, all 3 targets (KT test AUC, parsed from logs, 6 seeds)

Runs `edubert_<target>_kt_<t>_{scratch|indomain|fromednet|fromjunyi|fromassist}_n3000_seed{42,1,2,3,4,5}`, mean ±pstdev, then the gain vs scratch paired by seed with a 20,000-resample bootstrap CI and the count of seeds in the direction of the mean. SUPERSEDES the earlier 3-seed table.

| Target | scratch | indomain | fromednet | fromjunyi | fromassist |
|---|---|---|---|---|---|
| assist2017 | 0.6697 ±0.0008 | 0.6927 ±0.0018 (+0.0231 [+0.0220,+0.0239] 6/6) | 0.6957 ±0.0021 (+0.0260 [+0.0250,+0.0274] 6/6) | 0.6910 ±0.0016 (+0.0214 [+0.0196,+0.0231] 6/6) | (=indomain) |
| ednet | 0.6644 ±0.0014 | 0.6732 ±0.0007 (+0.0088 [+0.0076,+0.0100] 6/6) | (=indomain) | 0.6687 ±0.0010 (+0.0043 [+0.0033,+0.0053] 6/6) | 0.6640 ±0.0017 (-0.0004 [-0.0016,+0.0009] 2/6) |
| junyi | 0.7352 ±0.0007 | 0.7394 ±0.0007 (+0.0042 [+0.0032,+0.0051] 6/6) | 0.7417 ±0.0004 (+0.0064 [+0.0057,+0.0072] 6/6) | (=indomain) | 0.7340 ±0.0009 (-0.0012 [-0.0020,-0.0001] 1/6) |

Per-seed gains are in the inventory; recover them with `analysis/paired_bootstrap_pair.py --tsv run_inventory.tsv --metric test_auc --a <cond stem> --b <scratch stem>`.

_Read: ranking cross-dataset sources by pretraining corpus size (training-split students: ASSIST 1,366, Junyi 49,153, EdNet 353,597) reproduces the observed ordering on all 3 targets. EdNet as a foreign source beats in-domain on both cross-domain targets with NON-OVERLAPPING intervals (assist2017 +0.0260 [+0.0250,+0.0274] vs +0.0231 [+0.0220,+0.0239]; junyi +0.0064 [+0.0057,+0.0072] vs +0.0042 [+0.0032,+0.0051]). On EdNet's own target in-domain leads (+0.0088). The granularity-closest source (ASSIST, 102 skills vs EdNet's 142) does NOT transfer on the EdNet target: -0.0004 with an interval spanning zero, 2/6 seeds, so write "fails to transfer", not "worse than scratch". On the Junyi target ASSIST is -0.0012 with an interval EXCLUDING zero, 1/6 seeds, so there it does cost accuracy. CAVEAT: for the assist2017 target, N=3000 is its FULL training split (1,366 < 3000), so that row is not a reduced-data condition._

"""

NEW_22 = """### 2.2 Source comparison at N=3000, next-skill macro-OVR AUC (parsed from logs, 6 seeds)

Runs `edubert_<target>_ns_<t>_<cond>_n3000_seed{42,1,2,3,4,5}`, same budget and same sources as 2.1, second task. Same reporting format.

| Target | scratch | indomain | fromednet | fromjunyi | fromassist |
|---|---|---|---|---|---|
| assist2017 (92 classes) | 0.9796 ±0.0002 | 0.9820 ±0.0002 (+0.0024 [+0.0021,+0.0026] 6/6) | 0.9815 ±0.0003 (+0.0019 [+0.0016,+0.0022] 6/6) | 0.9822 ±0.0002 (+0.0026 [+0.0023,+0.0029] 6/6) | - |
| ednet (142 classes) | 0.8704 ±0.0007 | 0.8849 ±0.0007 (+0.0145 [+0.0143,+0.0147] 6/6) | - | 0.8752 ±0.0006 (+0.0047 [+0.0043,+0.0053] 6/6) | 0.8732 ±0.0005 (+0.0028 [+0.0024,+0.0034] 6/6) |
| junyi (1326 classes) | 0.9883 ±0.0013 | 0.9912 ±0.0004 (+0.0029 [+0.0020,+0.0043] 6/6) | 0.9897 ±0.0002 (+0.0015 [+0.0004,+0.0028] 4/6) | - | 0.9900 ±0.0001 (+0.0018 [+0.0008,+0.0030] 6/6) |

_Read: THE SCALE ORDERING OF 2.1 DOES NOT CARRY TO THIS TASK. It reproduces on the ednet target only (Junyi +0.0047 > ASSIST +0.0028). On assist2017 the smaller Junyi source beats the larger EdNet source (+0.0026 vs +0.0019), and on junyi the smallest source beats the largest (+0.0018 vs +0.0015, the latter 4/6 seeds). In-domain is also best on 2 of 3 targets here, so the foreign-beats-in-domain reversal of 2.1 does not reproduce either. Absolute values sit between 0.87 and 0.99, so read the ordering and not the magnitude. SCOPE both the scale claim and the foreign-beats-in-domain claim to KNOWLEDGE TRACING._

"""

EXACT = [
    ("section-3 scale boundary",
     "- **Scale boundary (recorded):** cross-dataset transfer gains are largest when the target is data-poor; at full target scale they fade toward ~0.",
     "- **Scale boundary (parsed from logs, 6 seeds):** cross-dataset transfer gains are largest when the target is data-poor; at full target scale every cross-dataset gain falls to within noise of zero."),
    ("section-3 only survivor",
     "- **Only survivor at full scale:** in-domain EdNet KT gain +0.0069, CI [+0.0065,+0.0075], 6/6 seeds. (in-domain EdNet KT gain at full target scale; only cross-dataset gain that survives (recorded))",
     "- **Only survivor at full scale (parsed from logs):** in-domain EdNet KT at 20,000 target students, 0.6846 ±0.0004 against scratch 0.6777 ±0.0004, gain +0.0069 CI [+0.0065,+0.0075], 6/6 seeds, both n=6. This reproduced exactly when recomputed from per-seed logs; it is no longer a recorded value."),
]


def span(text, start_marker, end_marker, name):
    if text.count(start_marker) != 1:
        sys.exit("ABORT, no write. %r start marker matched %d times."
                 % (name, text.count(start_marker)))
    if text.count(end_marker) != 1:
        sys.exit("ABORT, no write. %r end marker matched %d times."
                 % (name, text.count(end_marker)))
    a = text.index(start_marker)
    b = text.index(end_marker)
    if b <= a:
        sys.exit("ABORT, no write. %r end marker precedes start." % name)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="RESULTS.md")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t = open(args.file, encoding="utf-8").read()
    print("before md5 %s" % hashlib.md5(t.encode("utf-8")).hexdigest())
    applied, skipped = [], []

    if NEW_22.strip() in t:
        skipped.append("section-2.2 block")
    else:
        a, b = span(t, H22, H3, "section-2.2")
        print("--- replacing %d bytes of section 2.2 ---" % (b - a))
        print(t[a:b])
        t = t[:a] + NEW_22 + t[b:]
        applied.append("section-2.2 block")

    if NEW_21.strip() in t:
        skipped.append("section-2.1 block")
    else:
        a, b = span(t, H21, H22, "section-2.1")
        print("--- replacing %d bytes of section 2.1 ---" % (b - a))
        print(t[a:b])
        t = t[:a] + NEW_21 + t[b:]
        applied.append("section-2.1 block")

    for name, old, new in EXACT:
        if new in t:
            skipped.append(name)
            continue
        if t.count(old) != 1:
            sys.exit("ABORT, no write. Anchor %r matched %d times, expected 1."
                     % (name, t.count(old)))
        t = t.replace(old, new)
        applied.append(name)

    for n in applied:
        print("  apply   %s" % n)
    for n in skipped:
        print("  already %s" % n)

    if args.dry_run:
        print("dry run, nothing written")
        return
    if not applied:
        print("nothing to do")
        return
    open(args.file, "w", encoding="utf-8").write(t)
    print("after  md5 %s" % hashlib.md5(t.encode("utf-8")).hexdigest())
    print("wrote %s, %d edits applied" % (args.file, len(applied)))


if __name__ == "__main__":
    main()
