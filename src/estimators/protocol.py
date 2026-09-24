"""Target-sample protocol shared by the scoring CLIs (MRAP Section I, exposure test 2026-09-24).

The default protocol is the one every earlier score used: the fine-tune's learner draw from the
train split. The exposure test adds two. --split val scores validation learners, which no encoder
saw during pretraining; --match_split val scores a train draw with as many learners as that
validation score, so a difference between the two is not a sample-size effect. A non-default
protocol must carry --score_tag, which evaluate_estimators.py appends to the estimator name;
without it the new scores would silently replace the train-draw scores of the same estimator,
target, seed and candidate. This module imports no torch, so its argument rules test anywhere.
"""

from __future__ import annotations

import argparse
import re

# The splits an estimator may read. The test split holds the gold, so no scorer may touch it.
SCORABLE_SPLITS = ("train", "val")
TAG = re.compile(r"^[a-z0-9_]+$")


def check_split(split: str) -> None:
    if split not in SCORABLE_SPLITS:
        raise ValueError(f"split must be one of {SCORABLE_SPLITS}, got {split!r}; the test "
                         "split holds the gold and no estimator may read it")


def add_protocol_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--split", choices=SCORABLE_SPLITS, default="train",
                    help="learners to score (default train, the fine-tune's draw)")
    ap.add_argument("--match_split", choices=("val",), default=None,
                    help="score a train draw as large as this split, capped at --n_students")
    ap.add_argument("--score_tag", default=None,
                    help="required for a non-default protocol; appended to estimator names")


def resolve_protocol(a: argparse.Namespace) -> tuple[str, int | None, dict]:
    """(split, learners to draw, metadata to record) for parsed arguments.

    The default protocol returns empty metadata, so its score lines keep their old keys and names.
    target_budget records the gold cell the score mirrors, because a matched draw's own learner
    count (170 on ASSISTments 2017) does not name a benchmark cell.
    """
    if a.split == "train" and a.match_split is None:
        if a.score_tag:
            raise SystemExit("--score_tag marks a non-default protocol; the train draw keeps "
                             "its plain estimator names")
        return a.split, a.n_students, {}
    if not a.score_tag or not TAG.match(a.score_tag):
        raise SystemExit(f"a non-default protocol needs --score_tag matching {TAG.pattern}, "
                         f"got {a.score_tag!r}")
    if a.match_split is not None and a.split != "train":
        raise SystemExit(f"--match_split sizes a train draw; it cannot be combined with "
                         f"--split {a.split}")
    extra = {"score_tag": a.score_tag,
             "target_budget": f"n{a.n_students}" if a.n_students else "full_split"}
    n = a.n_students
    if a.match_split is not None:
        from src.estimators.features import matched_n

        n = matched_n(a.target_dir, a.n_students, a.match_split)
        extra["matched_to_split"] = a.match_split
    return a.split, n, extra
