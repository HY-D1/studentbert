#!/usr/bin/env python3
# Two behaviour-PRESERVING repairs to scripts/downstream_edubert.py.
#
# Neither changes any number produced on the seven processed datasets, so no
# result in RESULTS.md or in the paper moves. Both turn a silent defect into a
# loud one, and both make the paper's stated invariants true by construction
# rather than by accident.
#
# 1. ELIGIBILITY. build_dropout_labels() ends with
#        for sid, row in sid_to_row.items(): labels.setdefault(row, 0)
#    which labels every NPZ student, so the eligibility filter at line ~92
#    ("keep only students present in the label dict") can never remove anyone.
#    On these datasets nothing is added, because preprocessing enforces
#    MIN_INTERACTIONS = 10 while the label builder's --min_interactions
#    defaults to 5, so every NPZ student is already labelled. Verified from
#    the run records: W&B reports data/pos_rate exactly 0.25, and dilution by
#    label-0 students would push it strictly below the quantile.
#    The fix raises instead of defaulting. On these datasets the check passes
#    and behaviour is identical. If anyone later builds labels with a higher
#    --min_interactions, the run stops rather than silently training on
#    ineligible students at label 0.
#    This is the invariant the paper states in Datasets and Tasks: students
#    below the eligibility minimum take no part in training or evaluation.
#
# 2. BUDGET. The dropout branch never applies --n_students; only the
#    next-skill branch does. Jobs passed --n_students 3000 and run names carry
#    n3000, so the names misdescribe the cohort. The paper makes no dropout
#    budget claim, so nothing printed is wrong. The fix does NOT start
#    applying the flag, because that would change future results and make them
#    incomparable to every dropout number already reported. It warns loudly,
#    records the actual training cohort size, and logs both to W&B.
#
# NOT TOUCHED, deliberately: the whole-dataset quantile threshold (the paper
# discloses it and changing it would invalidate every dropout label), the MLM
# masking scheme, the bidirectional dropout encoder, and SAINT+ timing. Each
# is described in the paper as it currently behaves; repairing any of them
# requires a rerun and a matching paper edit in the same pass.
#
# Run from the repo root:  python3 tools/patches/patch_downstream_guards.py
from __future__ import annotations

import ast
import hashlib
import io
import os
import sys

PATH = "scripts/downstream_edubert.py"

OLD_ELIG = (
    "    for sid, row in sid_to_row.items():\n"
    "        labels.setdefault(row, 0)\n"
)

NEW_ELIG = (
    "    missing = [row for row in sid_to_row.values() if row not in labels]\n"
    "    if missing:\n"
    "        raise SystemExit(\n"
    "            \"%d of %d students in sequences.npz have no dropout label. \"\n"
    "            \"They are below the eligibility minimum used when \"\n"
    "            \"dropout_labels.json was built. Earlier revisions defaulted \"\n"
    "            \"them to label 0, which defeated the eligibility filter in \"\n"
    "            \"StudentSeqDataset. Rebuild the label file with a \"\n"
    "            \"--min_interactions at or below the preprocessing floor, or \"\n"
    "            \"drop those students from splits.json.\"\n"
    "            % (len(missing), len(sid_to_row)))\n"
)

OLD_BUDGET = (
    "    if args.task == \"dropout\":\n"
    "        labels, pos_rate, mean_total = build_dropout_labels(args.processed_dir, args.max_seq_len)\n"
    "        print(f\"[dropout] label=bottom-quartile disengagement; positive rate={pos_rate:.4f}\")\n"
    "        if wb: wb.log({\"data/pos_rate\": pos_rate})\n"
)

NEW_BUDGET = (
    "    if args.task == \"dropout\":\n"
    "        labels, pos_rate, mean_total = build_dropout_labels(args.processed_dir, args.max_seq_len)\n"
    "        print(f\"[dropout] label=bottom-quartile disengagement; positive rate={pos_rate:.4f}\")\n"
    "        if wb: wb.log({\"data/pos_rate\": pos_rate})\n"
    "        if args.n_students:\n"
    "            print(\"[dropout] WARNING: --n_students=%d is IGNORED by the \"\n"
    "                  \"dropout task and always has been. The flag applies to \"\n"
    "                  \"next-skill only. Training uses the full eligible \"\n"
    "                  \"training split. A run name containing nNNNN does not \"\n"
    "                  \"describe this cohort.\" % args.n_students)\n"
)

EDITS = [
    ("eligibility_raise", OLD_ELIG, NEW_ELIG),
    ("budget_warning", OLD_BUDGET, NEW_BUDGET),
]


def main() -> int:
    if not os.path.exists(PATH):
        print("FAIL: %s not found. Run this from the repo root." % PATH)
        return 1
    src = io.open(PATH, encoding="utf-8").read()
    print("before md5 %s  bytes %d  lines %d" % (
        hashlib.md5(src.encode("utf-8")).hexdigest(), len(src),
        src.count("\n")))

    bad = False
    for tag, old, _ in EDITS:
        n = src.count(old)
        print("  %-20s matches=%d" % (tag, n))
        if n != 1:
            bad = True
    if bad:
        print("\nFAIL: an anchor did not match exactly once. NOTHING WRITTEN.")
        return 1

    out = src
    for _, old, new in EDITS:
        out = out.replace(old, new)

    try:
        ast.parse(out)
    except SyntaxError as exc:
        print("\nFAIL: patched source does not parse: %s" % exc)
        print("NOTHING WRITTEN.")
        return 1

    first = out.split("\n", 1)[0].strip()
    if first != "from __future__ import annotations":
        print("\nFAIL: line 1 is %r, expected the __future__ import." % first)
        print("NOTHING WRITTEN.")
        return 1

    io.open(PATH, "w", encoding="utf-8").write(out)
    print("after  md5 %s  bytes %d  lines %d" % (
        hashlib.md5(out.encode("utf-8")).hexdigest(), len(out),
        out.count("\n")))
    print("wrote %s, %d edits applied, ast.parse OK" % (PATH, len(EDITS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
