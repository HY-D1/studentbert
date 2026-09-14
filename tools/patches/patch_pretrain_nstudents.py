"""Add --n_students to the pretraining path so a SOURCE corpus can be
subsampled by student count.

WHY: every transfer claim in the papers rests on three sources whose size is
perfectly confounded with their identity (platform, population, item pool,
skill taxonomy). Holding identity fixed and varying only the student count is
the manipulation that separates the two. Until now --n_students existed only
on the fine-tune path, where it subsamples the TARGET; the pretraining path
had no student cap at all, only --max_interactions.

BEHAVIOUR-PRESERVING: the new argument defaults to None and the new code is
guarded by `if n_students is not None`, so every existing checkpoint and every
number already reported is reproduced bit-for-bit by the patched code.

SAFETY: this script refuses to write anything unless all four anchors match
exactly once. It never inserts imports above line 1, which is the failure mode
that broke four scripts in this repo by displacing `from __future__ import
annotations`. After writing it ast.parses every file under scripts/ and src/
and reports the result.

Run from the repo root:
    PYTHONPATH=. python tools/patches/patch_pretrain_nstudents.py
"""

from __future__ import annotations

import ast
import pathlib
import sys

DATASET = pathlib.Path("src/data/dataset.py")
PRETRAIN = pathlib.Path("scripts/pretrain_edubert.py")

EDITS = [
    (
        DATASET,
        "signature: accept an optional student cap",
        '    def __init__(self, processed_dir: str, split: str, max_seq_len: int = 512):\n'
        '        """processed_dir: folder containing sequences.npz + splits.json.\n'
        "        split: 'train' | 'val' | 'test'.\n"
        '        """\n',
        '    def __init__(self, processed_dir: str, split: str, max_seq_len: int = 512,\n'
        '                 n_students: int | None = None, subsample_seed: int = 42):\n'
        '        """processed_dir: folder containing sequences.npz + splits.json.\n'
        "        split: 'train' | 'val' | 'test'.\n"
        '        n_students: if set, keep a seeded random subsample of this many\n'
        '        students from the split. Default None reproduces the previous\n'
        '        behaviour exactly, so existing results are unaffected.\n'
        '        subsample_seed: seed for that draw. Passing a different seed at the\n'
        '        same size gives an independent draw, which is how source-sampling\n'
        '        variance can be measured later without changing this code.\n'
        '        """\n',
    ),
    (
        DATASET,
        "draw the subsample after the split rows are resolved",
        "        self.rows = [id_to_row[s] for s in wanted if s in id_to_row]\n",
        "        self.rows = [id_to_row[s] for s in wanted if s in id_to_row]\n"
        "        if n_students is not None and n_students < len(self.rows):\n"
        "            # sort first: `wanted` is a set, so an unsorted draw would depend\n"
        "            # on set iteration order and would not be reproducible.\n"
        "            ordered = sorted(self.rows)\n"
        "            rng = np.random.default_rng(subsample_seed)\n"
        "            pick = rng.choice(len(ordered), size=n_students, replace=False)\n"
        "            self.rows = [ordered[i] for i in sorted(pick.tolist())]\n"
        "        self.n_students_requested = n_students\n"
        "        self.subsample_seed = subsample_seed\n",
    ),
    (
        PRETRAIN,
        "argparse: add --n_students",
        '    ap.add_argument("--max_interactions", type=int, default=None,\n'
        '                    help="if set, pretrain on a subset of ~this many interactions (tiny exp)")\n',
        '    ap.add_argument("--max_interactions", type=int, default=None,\n'
        '                    help="if set, pretrain on a subset of ~this many interactions (tiny exp)")\n'
        '    ap.add_argument("--n_students", type=int, default=None,\n'
        '                    help="if set, pretrain on a seeded random subsample of this "\n'
        '                         "many TRAINING students (source-scale ablation). "\n'
        '                         "Default None uses the whole training split.")\n',
    ),
    (
        PRETRAIN,
        "pass the cap through and print the realized size",
        '    train_ds = InteractionDataset(args.processed_dir, "train", args.max_seq_len)\n',
        '    train_ds = InteractionDataset(args.processed_dir, "train", args.max_seq_len,\n'
        "                                  n_students=args.n_students,\n"
        "                                  subsample_seed=args.seed)\n"
        "    if args.n_students is not None:\n"
        '        print(f"source subsample: {len(train_ds)} training students '
        '(requested {args.n_students}, draw seed {args.seed})", flush=True)\n',
    ),
]


def main() -> None:
    for path, _, _, _ in EDITS:
        if not path.exists():
            sys.exit("ABORT: %s not found. Run from the repo root." % path)

    # Decide applied vs pending for every edit BEFORE writing anything.
    # NOTE: three of these four anchors are INSERTION style, meaning `old` is a
    # substring of `new`. Counting `old` after application therefore still
    # returns 1, and a naive re-run would duplicate the inserted block. This
    # repo has been bitten by exactly that. Decide on `new` alone: if the
    # replacement text is present, the edit is applied, full stop.
    plan = []
    for path, label, old, new in EDITS:
        src = path.read_text()
        if new in src:
            plan.append((path, label, None, None, "already applied"))
            continue
        n_old = src.count(old)
        if n_old != 1:
            sys.exit(
                "ABORT before writing: anchor %r in %s matched %d times, expected 1. "
                "Nothing has been modified." % (label, path, n_old)
            )
        plan.append((path, label, old, new, "pending"))

    if all(p[4] == "already applied" for p in plan):
        print("all four edits are already applied; nothing to do")
    else:
        buf = {}
        for path, label, old, new, state in plan:
            if state == "already applied":
                print("skip (already applied): %s" % label)
                continue
            src = buf.get(path, path.read_text())
            buf[path] = src.replace(old, new, 1)
            print("apply: %s  [%s]" % (label, path))
        for path, text in buf.items():
            path.write_text(text)
            print("wrote %s" % path)

    files = sorted(pathlib.Path("scripts").rglob("*.py")) + sorted(
        pathlib.Path("src").rglob("*.py")
    )
    bad = []
    for f in files:
        try:
            ast.parse(f.read_text())
        except SyntaxError as exc:
            bad.append("%s:%s %s" % (f, exc.lineno, exc.msg))
    print("\nast.parse over %d files under scripts/ and src/" % len(files))
    if bad:
        print("SYNTAX ERRORS:")
        for b in bad:
            print("   " + b)
        sys.exit(1)
    print("syntax errors: 0")


if __name__ == "__main__":
    main()
