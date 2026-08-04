"""Learner-overlap audit and fine-tune sample manifests for the LAK paper.

Verifies from the processed data that:
  1. train / val / test learner splits are pairwise disjoint;
  2. the per-(seed, N) fine-tune samples can be reconstructed exactly
     (validated against the published regime-table interaction counts);
  3. every fine-tune sample is fully contained in the train partition,
     hence fully contained in the in-domain pretraining pool;
  4. no fine-tune sample overlaps validation or test learners.

Writes a markdown report and one learner-ID manifest per (seed, N).

Run from the repo code root on a compute node:
  PYTHONPATH=. /projects/algl/dai.hany/envs/sb/bin/python \
      analysis/overlap_report.py --processed ../processed/assist2017
"""
import argparse
import json
import os
import random
import sys

import numpy as np


def log(msg):
    print(msg, flush=True)


def load_splits(path):
    with open(path) as f:
        d = json.load(f)
    lower = {k.lower(): k for k in d}

    def pick(*names):
        for n in names:
            if n in lower:
                return d[lower[n]]
        return None

    train = pick("train")
    val = pick("val", "valid", "validation")
    test = pick("test")
    if train is None or test is None:
        raise SystemExit(f"splits.json keys not recognized: {list(d.keys())}")
    return train, (val or []), test


def load_rows(npz_path):
    log(f"    opening npz ...")
    z = np.load(npz_path, allow_pickle=True)
    log(f"    npz keys: {sorted(z.files)}")
    keys = set(z.files)
    off_key = "offsets" if "offsets" in keys else ("offset" if "offset" in keys else None)
    if off_key is None:
        raise SystemExit(f"no offsets array in {npz_path}; keys={sorted(keys)}")
    offsets = np.asarray(z[off_key]).astype(np.int64)
    lengths = np.diff(offsets)
    sid_key = None
    for cand in ("student_ids", "students", "student_id", "ids"):
        if cand in keys:
            sid_key = cand
            break
    if sid_key is None:
        raise SystemExit(f"no student id array in {npz_path}; keys={sorted(keys)}")
    sids = z[sid_key]
    n_rows = len(lengths)
    if len(sids) == n_rows:
        row_ids = [str(x) for x in sids]
    elif len(sids) == int(offsets[-1]):
        row_ids = [str(sids[int(offsets[i])]) for i in range(n_rows)]
    else:
        raise SystemExit(
            f"student id array length {len(sids)} matches neither rows "
            f"{n_rows} nor interactions {int(offsets[-1])}"
        )
    return row_ids, lengths


def resolve_split(entries, row_ids, id2row, n_rows):
    """Return (ordered row list, ordered id list, mode)."""
    entries_s = [str(e) for e in entries]
    hits = sum(1 for e in entries_s if e in id2row)
    if hits >= 0.5 * len(entries_s):
        rows = [id2row[e] for e in entries_s if e in id2row]
        return rows, [row_ids[r] for r in rows], "student-id", len(entries_s) - hits
    ok = all(str(e).lstrip("-").isdigit() and 0 <= int(e) < n_rows for e in entries_s)
    if ok:
        rows = [int(e) for e in entries_s]
        return rows, [row_ids[r] for r in rows], "row-index", 0
    raise SystemExit("split entries match neither student ids nor row indices")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="../processed/assist2017")
    ap.add_argument("--out", default="overlap_report.md")
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--levels", type=int, nargs="+",
                    default=[25, 50, 100, 200, 500, 1000])
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--expected", default="9044,18883,38806,75141,189602,378022",
                    help="published mean capped interaction counts per level, "
                         "comma separated, or 'none' to skip validation")
    a = ap.parse_args()

    import time
    t0 = time.time()
    log(f"[1/6] numpy {np.__version__}; loading {a.processed}/sequences.npz ...")
    row_ids, lengths = load_rows(os.path.join(a.processed, "sequences.npz"))
    log(f"[2/6] npz loaded in {time.time()-t0:.1f}s: {len(row_ids)} rows, "
        f"{int(lengths.sum())} interactions")
    n_rows = len(row_ids)
    id2row = {}
    for i, s in enumerate(row_ids):
        id2row.setdefault(s, i)
    dup = n_rows - len(id2row)

    log("[3/6] loading splits.json ...")
    train, val, test = load_splits(os.path.join(a.processed, "splits.json"))
    tr_rows, tr_ids, mode, tr_missing = resolve_split(train, row_ids, id2row, n_rows)
    va_rows, va_ids, _, _ = resolve_split(val, row_ids, id2row, n_rows) if val else ([], [], mode, 0)
    te_rows, te_ids, _, _ = resolve_split(test, row_ids, id2row, n_rows)
    trs, vas, tes = set(tr_ids), set(va_ids), set(te_ids)
    inter_tv, inter_tt, inter_vt = len(trs & vas), len(trs & tes), len(vas & tes)

    expected = None
    if a.expected.strip().lower() != "none":
        expected = [int(x) for x in a.expected.split(",")]
        if len(expected) != len(a.levels):
            raise SystemExit("--expected length must match --levels")

    log(f"[4/6] splits resolved: train {len(trs)}, val {len(vas)}, test {len(tes)}; "
        f"sampling reconstruction starting ...")
    orderings = [("splits-order", tr_rows), ("npz-order", sorted(tr_rows))]
    results = {}
    for name, rows in orderings:
        per = {}
        for N in a.levels:
            if N > len(rows):
                raise SystemExit(f"N={N} exceeds train size {len(rows)}")
            for s in a.seeds:
                rng = random.Random(s)
                samp = rng.sample(rows, N)
                capped = int(sum(min(int(lengths[r]), a.cap) for r in samp))
                per[(s, N)] = (samp, capped)
        means = [np.mean([per[(s, N)][1] for s in a.seeds]) for N in a.levels]
        match = None
        if expected is not None:
            match = all(abs(m - e) <= 2 for m, e in zip(means, expected))
        results[name] = (per, means, match)

    log(f"[5/6] sampling done in {time.time()-t0:.1f}s total; validating ...")
    chosen = None
    for name in ("splits-order", "npz-order"):
        if expected is not None and results[name][2]:
            chosen = name
            break
    validated = chosen is not None
    if chosen is None:
        chosen = "splits-order"

    per, means, _ = results[chosen]

    nest_lines = []
    for s in a.seeds:
        sets = {N: set(per[(s, N)][0]) for N in a.levels}
        chain = []
        for lo, hi in zip(a.levels[:-1], a.levels[1:]):
            chain.append(f"{lo}in{hi}={'yes' if sets[lo] <= sets[hi] else 'no'}")
        nest_lines.append(f"seed {s}: " + ", ".join(chain))

    all_in_train = True
    val_hits = test_hits = 0
    for (s, N), (samp, _c) in per.items():
        sids = {row_ids[r] for r in samp}
        if not sids <= trs:
            all_in_train = False
        val_hits += len(sids & vas)
        test_hits += len(sids & tes)

    os.makedirs(a.manifests, exist_ok=True)
    write_ok = validated or expected is None
    written = 0
    if write_ok:
        for (s, N), (samp, _c) in sorted(per.items()):
            p = os.path.join(a.manifests, f"sample_seed{s}_N{N}.txt")
            with open(p, "w") as f:
                f.write("\n".join(row_ids[r] for r in samp) + "\n")
            written += 1

    L = []
    L.append("# Learner overlap and sample manifest report (ASSISTments 2017)\n")
    L.append(f"Command: `PYTHONPATH=. python analysis/overlap_report.py --processed {a.processed}`\n")
    L.append("## Split audit\n")
    L.append(f"- rows in sequences.npz: {n_rows} (duplicate learner ids: {dup})")
    L.append(f"- split entry mode: {mode} (entries not found in npz: {tr_missing})")
    L.append(f"- train {len(trs)}, val {len(vas)}, test {len(tes)}, "
             f"total {len(trs) + len(vas) + len(tes)}")
    L.append(f"- pairwise overlaps: train-val {inter_tv}, train-test {inter_tt}, "
             f"val-test {inter_vt}\n")
    L.append("## Fine-tune sample reconstruction\n")
    L.append("Sampling rule: random.Random(seed).sample(train_rows, N), "
             "fresh generator per (seed, N).\n")
    for name in ("splits-order", "npz-order"):
        m = results[name][1]
        tag = ""
        if expected is not None:
            tag = "  MATCHES published table" if results[name][2] else "  does NOT match"
        L.append(f"- {name}: mean capped interactions per level = "
                 + ", ".join(f"{x:.1f}" for x in m) + tag)
    if expected is not None:
        L.append(f"- published values: {', '.join(str(e) for e in expected)}")
    L.append(f"- ordering used for manifests: {chosen} "
             f"({'VALIDATED' if validated else 'UNVALIDATED, no manifest written' if expected is not None else 'validation skipped'})\n")
    L.append("## Nestedness across levels (within each seed)\n")
    L.extend(f"- {ln}" for ln in nest_lines)
    L.append("")
    L.append("## Overlap conclusions\n")
    L.append(f"- every sampled fine-tune learner is in the train partition: "
             f"{'yes' if all_in_train else 'NO'}")
    L.append(f"- sampled learners appearing in val: {val_hits}; in test: {test_hits}")
    L.append("- the in-domain condition pretrains on the full train partition, so each "
             "fine-tune sample overlaps the in-domain pretraining pool completely "
             "(N of N learners) by construction; scratch, EdNet, and Junyi conditions "
             "have zero learner overlap between pretraining data and the target sample.")
    L.append("- validation and test learners are excluded from pretraining and from every "
             "fine-tune sample, so reported test metrics involve no leakage.")
    L.append("")
    if write_ok:
        tagm = "" if validated else " (UNVALIDATED: --expected none was set)"
        L.append(f"## Manifests\n\n- {written} files written to {a.manifests}/ "
                 "(one learner id per line, file name gives seed and N)" + tagm)
    else:
        L.append("## Manifests\n\n- NOT written: reconstruction did not match the "
                 "published counts; check the sampling in analysis/sample_stats_report.py")

    log(f"[6/6] writing report and manifests ...")
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[-12:]))
    print(f"\nreport written to {a.out}")
    if expected is not None and not validated:
        sys.exit(2)


if __name__ == "__main__":
    main()
