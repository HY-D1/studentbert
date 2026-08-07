#!/usr/bin/env python3
"""Add evaluation-only label exclusion and prediction saving to
scripts/downstream_edubert.py.

Run from the repo root:
    /projects/algl/dai.hany/envs/sb/bin/python tools/patches/add_noskill_exclusion.py

What it adds (nothing existing is changed or removed):
  --exclude_label NAME   resolve NAME in skill_vocab.json and additionally
                         report next-skill metrics with that class excluded
  --save_preds           write test-set argmax + top-5 + targets to an npz so
                         future metric variants never need another GPU run
  --preds_dir DIR        where those npz files go (default ../preds)

Two exclusion variants are computed and logged, so the reporting choice can be
made after the runs rather than before:
  *_excl         rows whose TRUE label is the excluded class are dropped; the
                 model may still predict that class, which counts as an error
  *_excl_masked  same rows dropped AND the excluded class removed from the
                 prediction space, so the model cannot emit it at all

Adds no imports: json, Path, np and torch are already imported by the target,
which keeps the `from __future__ import annotations` line untouched at line 1.

Idempotent: re-running detects the already-patched text and exits without
duplicating anything.
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("scripts/downstream_edubert.py")

ARGS_OLD = '    ap.add_argument("--wandb", action="store_true")'

ARGS_NEW = '''    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--exclude_label", default=None,
                    help="skill_vocab.json key to exclude from next-skill metrics "
                         "(evaluation only; training is unchanged)")
    ap.add_argument("--save_preds", action="store_true",
                    help="write test-set predictions to --preds_dir as npz")
    ap.add_argument("--preds_dir", default="../preds")'''

EVAL_OLD = '''        if wb: wb.log({"test/top1": t1, "test/top5": t5, "test/macro_auc": m_auc, "test/weighted_auc": w_auc, "test/macro_top1": mt1})'''

EVAL_NEW = '''        if wb: wb.log({"test/top1": t1, "test/top5": t5, "test/macro_auc": m_auc, "test/weighted_auc": w_auc, "test/macro_top1": mt1})

        excl_idx = None
        if args.exclude_label:
            _vocab = json.loads((Path(args.processed_dir) / "skill_vocab.json").read_text())
            excl_idx = _vocab.get(args.exclude_label)
            if excl_idx is None:
                raise SystemExit(f"--exclude_label {args.exclude_label!r} not found in skill_vocab.json")

        if excl_idx is not None:
            keep = vd & (tg != excl_idx)
            pres_x = present[present != excl_idx]
            n_drop = int((vd & (tg == excl_idx)).sum())
            t1_x = topk_acc(lg, tg, keep, 1)
            t5_x = topk_acc(lg, tg, keep, 5)
            mt1_x, n_mt1_x = macro_top1(lg, tg, keep, pres_x.tolist())
            m_auc_x, w_auc_x, n_cls_x = macro_ovr_auc(probs, tg, keep, pres_x.tolist())
            lg_m = lg.clone()
            lg_m[:, excl_idx] = float("-inf")
            t1_xm = topk_acc(lg_m, tg, keep, 1)
            mt1_xm, n_mt1_xm = macro_top1(lg_m, tg, keep, pres_x.tolist())
            print(f"--- excluding {args.exclude_label!r} (class {excl_idx}); dropped {n_drop} of {int(vd.sum())} scored interactions ---")
            print(f"test top-1 acc  (excl): {t1_x:.4f}")
            print(f"test top-5 acc  (excl): {t5_x:.4f}")
            print(f"test macro-top1 (excl): {mt1_x:.4f}  (over {n_mt1_x} classes)")
            print(f"test macro-OVR AUC (excl): {m_auc_x:.4f}  (over {n_cls_x} classes)")
            print(f"test weighted-OVR AUC (excl): {w_auc_x:.4f}  (over {n_cls_x} classes)")
            print(f"test top-1 acc  (excl, class masked): {t1_xm:.4f}")
            print(f"test macro-top1 (excl, class masked): {mt1_xm:.4f}  (over {n_mt1_xm} classes)")
            if wb: wb.log({"test/top1_excl": t1_x, "test/top5_excl": t5_x,
                           "test/macro_top1_excl": mt1_x, "test/macro_auc_excl": m_auc_x,
                           "test/weighted_auc_excl": w_auc_x,
                           "test/top1_excl_masked": t1_xm, "test/macro_top1_excl_masked": mt1_xm,
                           "test/n_excluded": n_drop, "test/excluded_class": excl_idx})

        if args.save_preds:
            _pdir = Path(args.preds_dir)
            _pdir.mkdir(parents=True, exist_ok=True)
            _out = _pdir / f"{run_name}_testpreds.npz"
            np.savez_compressed(
                _out,
                target=tg.cpu().numpy().astype(np.int16),
                valid=vd.cpu().numpy(),
                pred=lg.argmax(dim=-1).cpu().numpy().astype(np.int16),
                top5=lg.topk(5, dim=-1).indices.cpu().numpy().astype(np.int16),
            )
            print(f"saved test predictions to {_out}")'''

EDITS = [("argparse block", ARGS_OLD, ARGS_NEW), ("next-skill eval block", EVAL_OLD, EVAL_NEW)]


def main() -> int:
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found. Run this from the repo root.")
        return 1

    src = TARGET.read_text()
    original = src
    applied, skipped = [], []

    for name, old, new in EDITS:
        if new in src:
            skipped.append(name)
            continue
        n = src.count(old)
        if n != 1:
            print(f"FAIL: anchor for {name} found {n} times, expected exactly 1. No changes written.")
            return 1
        src = src.replace(old, new, 1)
        applied.append(name)

    if not applied:
        print("Already patched, nothing to do. " + ", ".join(skipped))
        return 0

    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"FAIL: patched source does not parse ({e}). No changes written.")
        return 1

    first_line = src.splitlines()[0].strip()
    if first_line != "from __future__ import annotations":
        print(f"FAIL: line 1 is now {first_line!r}. No changes written.")
        return 1

    backup = TARGET.with_suffix(".py.bak_noskill")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
    TARGET.write_text(src)

    print(f"patched {TARGET}")
    for name in applied:
        print(f"  applied: {name}")
    for name in skipped:
        print(f"  already present: {name}")
    print(f"  backup: {backup}")
    print(f"  bytes {len(original)} -> {len(src)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
