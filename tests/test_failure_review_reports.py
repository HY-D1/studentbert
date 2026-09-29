"""Tests for analysis/frozen_gold_report.py (D6) and analysis/layer_diag_report.py (D4).

  PYTHONPATH=. python tests/test_failure_review_reports.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from analysis import frozen_gold_report as fgr  # noqa: E402
from analysis import layer_diag_report as ldr  # noqa: E402
from tests.test_similarity_baseline import SEVEN, write_exec  # noqa: E402

T = "algebra2006"
CANDS = ["scratch"] + [f"edubert_{d}_pretrain_full_encoder.pt" for d in SEVEN]


def score_of(name: str) -> float:
    order = {"scratch": 0.1, f"edubert_{T}_pretrain_full_encoder.pt": 0.9,
             "edubert_junyi_pretrain_full_encoder.pt": 0.8}
    return order.get(name, 0.2 + 0.01 * CANDS.index(name))


def write_scores(path: Path, estimator: str) -> None:
    recs = [{"estimator": estimator, "candidate": c, "target": T, "seed": s, "score": score_of(c),
             "metadata": {"n_students_requested": 3000}} for s in fgr.SEEDS for c in CANDS]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def write_frozen(path: Path, skip_one: bool = False, hs_shift: float = 0.0) -> None:
    recs = []
    for s in fgr.SEEDS:
        for c in CANDS:
            for ro, lam in fgr.READOUTS:
                if skip_one and (s, c, ro, lam) == (2, CANDS[3], *fgr.PRIMARY):
                    continue
                recs.append({"target": T, "candidate": c, "seed": s, "readout": ro, "lam": lam,
                             "frozen_test_auc": 0.6 + score_of(c) / 10,
                             "details": {"hscore_on_train_features": score_of(c) + hs_shift}})
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def d6_args(tmp: Path) -> list[str]:
    return ["--executions", str(tmp / "e.tsv"), "--frozen", str(tmp / "f.jsonl"),
            "--task2", str(tmp / "h.jsonl"), "--logme", str(tmp / "l.jsonl"), "--boots", "200",
            "--out", str(tmp / "o")]


def setup_d6(tmp: Path, **kw) -> None:
    write_exec(tmp / "e.tsv", (T,))
    write_scores(tmp / "h.jsonl", "hscore_kt_causal")
    write_scores(tmp / "l.jsonl", "logme_kt_causal")
    write_frozen(tmp / "f.jsonl", **kw)


def test_d6_judges_a_complete_target():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        setup_d6(tmp)
        assert fgr.main(d6_args(tmp)) == 0
        rep = (tmp / "o_report.md").read_text()
        assert f"| {T} | src:{T} | 1 | src:{T} |" in rep, rep


def test_d6_refuses_features_that_differ_from_the_estimators():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        setup_d6(tmp, hs_shift=1e-3)
        try:
            fgr.main(d6_args(tmp))
        except SystemExit as exc:
            assert "differ from the estimators" in str(exc)
        else:
            raise AssertionError("misaligned frozen-gold features were accepted")


def test_d6_skips_an_incomplete_target():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        setup_d6(tmp, skip_one=True)
        try:
            fgr.main(d6_args(tmp))
        except SystemExit as exc:
            assert "no complete target" in str(exc)
        else:
            raise AssertionError("an incomplete target was judged")


def write_diag(path: Path, seeds=(42, 1), own_reset_score=0.0) -> None:
    recs = []
    for s in seeds:
        for base in ldr.BASES:
            for L in ldr.LAYERS:
                for c in CANDS:
                    recs.append({"estimator": f"{base}_L{L}", "target": T, "seed": s,
                                 "candidate": c, "score": score_of(c), "metadata": {"layer": L}})
                recs.append({"estimator": f"{base}_L{L}_skillrand", "target": T, "seed": s,
                             "candidate": CANDS[-1], "score": own_reset_score,
                             "metadata": {"layer": L}})
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_d4_replaces_only_the_in_domain_score():
    assert CANDS[-1] == f"edubert_{T}_pretrain_full_encoder.pt"
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        write_exec(tmp / "e.tsv", (T,))
        write_scores(tmp / "h.jsonl", "hscore_kt_causal")
        write_scores(tmp / "l.jsonl", "logme_kt_causal")
        write_diag(tmp / "d.jsonl")
        assert ldr.main(["--diag", str(tmp / "d.jsonl"), "--executions", str(tmp / "e.tsv"),
                         "--task2", str(tmp / "h.jsonl"), "--logme", str(tmp / "l.jsonl"),
                         "--boots", "200", "--out", str(tmp / "o")]) == 0
        rep = (tmp / "o_report.md").read_text()
        assert f"| L6 | trained skill table | {T}, {T} |" in rep
        assert "| L6 | skill table reset | junyi, junyi |" in rep
        assert "Complete seeds: 1, 42; incomplete: none" in rep


def main() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except (Exception, SystemExit):
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
