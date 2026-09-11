#!/usr/bin/env python3
# Updates RESULTS.md Section 8.1 (ASSIST dropout K-sweep) to the 5 dp W&B
# values. Read-only until every anchor is verified to appear exactly once.
# Run from the repo root:  python3 tools/patch_results_81.py
from __future__ import annotations

import hashlib
import io
import os
import sys

PATH = "RESULTS.md"

EDITS = [
    (
        "k5_read",
        "| 0.6991 +/-0.0122 | scratch best (+0.0128) |",
        "| 0.6991 +/-0.0122 | scratch best (+0.0127) |",
    ),
    (
        "k20_scratch",
        "| 20 | **0.7023 +/-0.0080** |",
        "| 20 | **0.7024 +/-0.0080** |",
    ),
    (
        "k50_scratch_indomain",
        "| 50 | 0.7654 +/-0.0054 | 0.7282 +/-0.0253 |",
        "| 50 | 0.7654 +/-0.0053 | 0.7281 +/-0.0253 |",
    ),
    (
        "read_line",
        "_Read: pretraining is WORSE than scratch on ASSIST dropout at every "
        "clean K, with in-domain intervals excluding zero at K=5/10/20/50 and "
        "0/3 seeds positive each time.",
        "_Read: in-domain pretraining is WORSE than scratch on ASSIST dropout "
        "at every clean K. In-domain intervals exclude zero at K=5/10/20 with "
        "0/3 seeds positive; at K=50 seed42 reaches +0.0007, so that interval "
        "spans zero and 1/3 seeds are positive. From-junyi is +0.0045 at K=50 "
        "on 3/3 seeds, which is inside the reproduction spread recorded below, "
        "so no direction is read from it.",
    ),
    (
        "provenance_note",
        "| 200 (LEAKED) | 0.8865 +/-0.0454 | 0.8971 +/-0.0121 | "
        "0.9029 +/-0.0163 | 0.9085 +/-0.0143 | inflated, do not report |\n",
        "| 200 (LEAKED) | 0.8865 +/-0.0454 | 0.8971 +/-0.0121 | "
        "0.9029 +/-0.0163 | 0.9085 +/-0.0143 | inflated, do not report |\n"
        "\n"
        "Source for K=5/10/20/50: w6_dropoutK_s42 (7987745), w6_dropoutK_s1 "
        "(7987751), w6_dropoutK_s2 (7987752). Per-seed AUC read from the W&B "
        "run summaries at 5 dp, not from the 4 dp log banners; means and "
        "standard deviations formed before rounding, ties rounded away from "
        "zero. K=100/200 rows are unchanged 4 dp values and are not reported "
        "anywhere. w5_dropout_k50 (7853281) covers K=50 only and is "
        "superseded; it is kept below as a reproduction check, not as a "
        "data source.\n"
        "\n"
        "Reproduction check, same command, same encoder checkpoints, same "
        "seed, unchanged dropout code path, all nodes gpu:v100-sxm2: "
        "7853281 ran on d1002, 7987745 on d1019, 7987751 and 7987752 on "
        "d1017. Same-seed absolute differences across the 12 K=50 cells are "
        "min 0.0012, median 0.0134, max 0.0435. The 8 cells comparing d1002 "
        "with d1017 span 0.0035 to 0.0435, so node pairing does not predict "
        "the size. src/utils.py set_seed seeds random, numpy and torch and "
        "sets no determinism flags. KT n3000 runs duplicated the same way "
        "differ by at most 0.0024.\n",
    ),
]


def main() -> int:
    if not os.path.exists(PATH):
        print("FAIL: %s not found. Run this from the repo root." % PATH)
        return 1
    src = io.open(PATH, encoding="utf-8").read()
    print("before md5 %s  bytes %d" % (
        hashlib.md5(src.encode("utf-8")).hexdigest(), len(src)))

    bad = False
    for tag, old, _ in EDITS:
        n = src.count(old)
        print("  %-22s matches=%d" % (tag, n))
        if n != 1:
            bad = True
    if bad:
        print("\nFAIL: at least one anchor did not match exactly once.")
        print("NOTHING WAS WRITTEN. Section 8.1 as it stands:")
        for i, line in enumerate(src.split("\n"), 1):
            if "dropout" in line.lower() and "| K |" in line:
                pass
        start = src.find("| K | scratch | indomain | ednet | junyi | read |")
        if start >= 0:
            print(src[start:start + 1400])
        return 1

    out = src
    for _, old, new in EDITS:
        out = out.replace(old, new)
    io.open(PATH, "w", encoding="utf-8").write(out)
    print("after  md5 %s  bytes %d" % (
        hashlib.md5(out.encode("utf-8")).hexdigest(), len(out)))
    print("wrote %s, %d edits applied" % (PATH, len(EDITS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
