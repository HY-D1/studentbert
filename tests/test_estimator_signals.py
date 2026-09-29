"""Tests for analysis/estimator_signals.py on synthetic score files.

  PYTHONPATH=. python tests/test_estimator_signals.py
"""

from __future__ import annotations

import contextlib
import io
import json
import statistics as st
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import estimator_signals as sg  # noqa: E402

CK = "edubert_{}_pretrain_full_encoder.pt"


def rec(est, tgt, seed, cand, score, conv=True, split="train", tag=None, direction=None):
    md = {"n_students_requested": 3000, "score_split": split}
    if tag:
        md["score_tag"] = tag
    return {"estimator": est, "access_regime": "F", "candidate": cand, "target": tgt,
            "seed": seed, "score": score, "score_direction": direction or "higher_is_better",
            "n_target_examples": 10, "n_target_labels": 10, "feature_extraction_time": 1.0,
            "scoring_time": 0.5, "peak_memory_mb": 100.0,
            "uncertainty_signals": {"converged": conv}, "metadata": md}


def write(path, recs):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in recs))


def fixture(tmp):
    recs = []
    vals = {42: (0.30, 0.20, 0.90), 1: (0.31, 0.22, 0.95), 2: (0.24, 0.26, 0.99)}
    for seed, (a, b, s) in vals.items():
        recs += [rec("hs", "assist2017", seed, CK.format("ednet"), a),
                 rec("hs", "assist2017", seed, CK.format("junyi"), b),
                 rec("hs", "assist2017", seed, "scratch", s, conv=seed != 2)]
        recs += [rec("hs", "assist2017", seed, CK.format("ednet"), a + 0.5, split="val",
                     tag="val"),
                 rec("hs", "assist2017", seed, CK.format("junyi"), b, split="val", tag="val")]
    write(f"{tmp}/s.jsonl", recs)
    return f"{tmp}/s.jsonl"


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = sg.main(argv)
    return rc, buf.getvalue()


def rows_of(path):
    lines = Path(path).read_text().splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def test_pretrained_view_excludes_scratch_and_counts_agreement():
    with tempfile.TemporaryDirectory() as t:
        rc, _ = run(["--scores", fixture(t), "--out", f"{t}/o"])
        rows = {r["estimator"]: r for r in rows_of(f"{t}/o.tsv")}
        r = rows["hs"]
        assert rc == 0 and r["modal_pick"] == "src:ednet" and r["pick_agreement"] == "2/3", r
        want = st.mean([0.30 - 0.20, 0.31 - 0.22, 0.26 - 0.24])
        assert abs(float(r["margin"]) - want) < 5e-7, r
        assert r["not_converged"] == "0", r


def test_practical_view_keeps_scratch_and_its_convergence_flag():
    with tempfile.TemporaryDirectory() as t:
        run(["--scores", fixture(t), "--view", "practical", "--out", f"{t}/o"])
        r = {x["estimator"]: x for x in rows_of(f"{t}/o.tsv")}["hs"]
        assert r["modal_pick"] == "scratch" and r["pick_agreement"] == "3/3", r
        assert r["not_converged"] == "1", r


def test_protocol_tag_is_a_separate_group():
    with tempfile.TemporaryDirectory() as t:
        run(["--scores", fixture(t), "--out", f"{t}/o"])
        ests = {r["estimator"] for r in rows_of(f"{t}/o.tsv")}
        assert ests == {"hs", "hs@val"}, ests


def test_margin_over_sd_uses_seed_noise():
    with tempfile.TemporaryDirectory() as t:
        run(["--scores", fixture(t), "--out", f"{t}/o"])
        r = {x["estimator"]: x for x in rows_of(f"{t}/o.tsv")}["hs"]
        sd = st.mean([st.pstdev([0.30, 0.31, 0.24]), st.pstdev([0.20, 0.22, 0.26])])
        assert abs(float(r["seed_sd"]) - sd) < 5e-7
        assert abs(float(r["margin_over_sd"]) - float(r["margin"]) / sd) < 1e-6, r


def test_lower_is_better_scores_are_oriented():
    with tempfile.TemporaryDirectory() as t:
        recs = [rec("loss", "junyi", s, CK.format(c), v, direction="lower_is_better")
                for s in (42, 1) for c, v in (("ednet", 0.1), ("junyi", 0.5))]
        write(f"{t}/s.jsonl", recs)
        run(["--scores", f"{t}/s.jsonl", "--out", f"{t}/o"])
        r = rows_of(f"{t}/o.tsv")[0]
        assert r["modal_pick"] == "src:ednet" and abs(float(r["margin"]) - 0.4) < 1e-9, r


def test_regret_join_by_group_and_view():
    with tempfile.TemporaryDirectory() as t:
        cells = f"{t}/c.tsv"
        Path(cells).write_text("estimator\ttarget\tbudget\tseed\tview\tregret\n"
                               "hs\tassist2017\tn3000\t42\tpretrained\t0.003\n"
                               "hs\tassist2017\tn3000\t1\tpretrained\t0.001\n"
                               "hs\tassist2017\tn3000\t1\tpractical\t0.900\n")
        run(["--scores", fixture(t), "--cells", cells, "--out", f"{t}/o"])
        r = {x["estimator"]: x for x in rows_of(f"{t}/o.tsv")}["hs"]
        assert abs(float(r["mean_regret"]) - 0.002) < 1e-12, r


def test_empty_input_aborts():
    with tempfile.TemporaryDirectory() as t:
        write(f"{t}/s.jsonl", [rec("hs", "x", 1, "not_a_checkpoint.pt", 0.1)])
        try:
            run(["--scores", f"{t}/s.jsonl", "--out", f"{t}/o"])
        except SystemExit as ex:
            assert "no scored groups" in str(ex)
        else:
            raise AssertionError("empty input did not abort")


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
