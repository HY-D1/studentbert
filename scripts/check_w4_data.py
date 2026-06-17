"""W4 prep check: verify EdNet + Junyi processed data are loadable and schema-matched
to ASSISTments BEFORE spending GPU hours on cross-dataset pretraining.

Confirms for each dataset:
  - npz loads; arrays present with expected dtypes
  - offsets are CSR-consistent (len == n_students+1, last == n_interactions)
  - time_bin values are within the shared 0..5 scheme (0=PAD) -> time_emb transfers
  - skill ids are within 1..K (0=PAD); reports K (=skill_emb size needed)
  - splits.json sums to n_students

Run on cluster in (sb):
    PYTHONPATH=. python scripts/check_w4_data.py
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path("../processed")
DATASETS = ["assist2017", "ednet", "junyi"]
NUM_TIME_BINS = 5  # shared scheme: bins 1..5, 0=PAD


def check(name: str) -> dict:
    d = ROOT / name
    npz = np.load(d / "sequences.npz")
    skill = npz["skill"]; correct = npz["correct"]
    tb = npz["time_bin"]; offs = npz["offsets"]; sid = npz["student_ids"]
    vocab = json.loads((d / "skill_vocab.json").read_text())
    splits = json.loads((d / "splits.json").read_text())
    K = max(vocab.values())

    n_students = len(sid)
    n_int = len(skill)
    issues = []
    if len(offs) != n_students + 1:
        issues.append(f"offsets len {len(offs)} != n_students+1 {n_students+1}")
    if int(offs[-1]) != n_int:
        issues.append(f"offsets[-1] {int(offs[-1])} != n_interactions {n_int}")
    tb_min, tb_max = int(tb.min()), int(tb.max())
    if tb_min < 0 or tb_max > NUM_TIME_BINS:
        issues.append(f"time_bin range [{tb_min},{tb_max}] outside 0..{NUM_TIME_BINS}")
    sk_min, sk_max = int(skill.min()), int(skill.max())
    if sk_min < 0 or sk_max > K:
        issues.append(f"skill range [{sk_min},{sk_max}] outside 0..K({K})")
    if not set(np.unique(correct)).issubset({0, 1}):
        issues.append(f"correct has values beyond 0/1: {np.unique(correct)[:8]}")
    split_total = sum(len(splits[k]) for k in ("train", "val", "test"))
    if split_total != n_students:
        issues.append(f"splits sum {split_total} != n_students {n_students}")

    return {"name": name, "students": n_students, "interactions": n_int,
            "K": K, "time_bin_range": (tb_min, tb_max),
            "train": len(splits["train"]), "issues": issues}


def main():
    rows = [check(n) for n in DATASETS]
    print(f"{'dataset':12} {'students':>9} {'interactions':>13} {'K':>6} "
          f"{'time_bins':>10} {'train':>8}")
    print("-" * 64)
    for r in rows:
        print(f"{r['name']:12} {r['students']:>9,} {r['interactions']:>13,} "
              f"{r['K']:>6} {str(r['time_bin_range']):>10} {r['train']:>8,}")
    print()
    # cross-dataset compatibility verdict
    tb_ok = all(r["time_bin_range"][1] <= NUM_TIME_BINS and r["time_bin_range"][0] >= 0
                for r in rows)
    print("CROSS-DATASET CHECK")
    print(f"  time_bin scheme shared across all 3 (0..{NUM_TIME_BINS})? "
          f"{'YES -> time_emb transfers' if tb_ok else 'NO -> time_emb will NOT transfer cleanly'}")
    print("  skill vocab sizes (K): " +
          ", ".join(f"{r['name']}={r['K']}" for r in rows) +
          "  -> skill_emb/skill_head re-init per target (expected; disjoint vocabs)")
    any_issue = False
    for r in rows:
        if r["issues"]:
            any_issue = True
            print(f"  !! {r['name']} ISSUES: {r['issues']}")
    print("\nVERDICT:", "ALL CLEAR \u2014 safe to pretrain" if (tb_ok and not any_issue)
          else "FIX ISSUES BEFORE PRETRAINING")


if __name__ == "__main__":
    main()
