from __future__ import annotations

"""Tests for analysis/nlp_option1_eval.py on synthetic early fine-tunes (no cluster files needed).

  PYTHONPATH=. python tests/test_nlp_option1_eval.py
"""

import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import nlp_option1_eval as ev  # noqa: E402
from analysis import task4_evaluate as te  # noqa: E402

GOLD = te.load_gold()
MODELS = sorted(GOLD[("rte", "cls")])


def runs(acc=lambda i, e: 0.6 + 0.01 * i + 0.001 * e) -> dict:
    out = {}
    for t in ev.TASKS:
        for p in ev.POOLINGS:
            for s in ev.SEEDS:
                for i, m in enumerate(MODELS):
                    out[(t, p, s, m)] = {"epochs": [{"epoch": e, "val_acc": acc(i, e),
                                                     "epoch_s": 60.0} for e in ev.EPOCHS],
                                         "train_idx_md5": "x", "val_idx_md5": "y", "n_train": 1,
                                         "n_val": 1, "gpu": "Tesla V100-SXM2-32GB",
                                         "total_s": 180.0}
    return out


def pairs_recs(pick: str = MODELS[0]) -> tuple[list[dict], list[dict]]:
    pairs = []
    for j, b in enumerate(m for m in MODELS if m != pick):
        pairs.append({"task": "rte", "pooling": "cls", "seed": "42", "pick": pick,
                      "alternative": b, "hscore_pick": "1.0", "hscore_alt": str(0.9 - 0.1 * j),
                      "logme_pick": "0.5", "logme_alt": "0.4",
                      "label": "reversal" if j == 0 else "stable"})
    recs = [{"task": "rte", "pooling": "cls", "seed": "42", "regret_points": "1.5"}]
    return pairs, recs


def test_a_short_budget_never_sees_later_epochs():
    pairs, recs = pairs_recs()
    r1 = ev.build_rows(pairs, recs, runs(), GOLD, 1)
    late = runs(lambda i, e: 0.6 + 0.01 * i + 0.001 * e + (0.3 if e > 1 else 0.0) * (i % 2))
    r1b = ev.build_rows(pairs, recs, late, GOLD, 1)
    assert r1 == r1b
    assert not any(k.startswith(("val_e2", "val_e3")) for k in ev.feature_cols(1))
    assert all(c in r1[0] for c in ev.feature_cols(1))


def test_the_rule_abstains_iff_the_pick_is_not_the_leader():
    pairs, recs = pairs_recs(pick=MODELS[-1])
    rows = ev.build_rows(pairs, recs, runs(), GOLD, 3)
    assert ev.rule_recs(rows)[0]["risk_noisy_or"] == 0.0
    pairs, recs = pairs_recs(pick=MODELS[0])
    rows = ev.build_rows(pairs, recs, runs(), GOLD, 3)
    assert ev.rule_recs(rows)[0]["risk_noisy_or"] == 1.0


def test_history_comes_only_from_other_tasks():
    pairs, recs = pairs_recs()
    base = ev.build_rows(pairs, recs, runs(), GOLD, 3)
    g2 = copy.deepcopy(GOLD)
    for m in g2[("rte", "cls")]:
        g2[("rte", "cls")][m]["tuned"] = (0.0, 1.0)
    assert ev.build_rows(pairs, recs, runs(), g2, 3) == base


def test_collector_aborts_on_a_missing_run_or_different_samples():
    with tempfile.TemporaryDirectory() as d:
        rec = runs()
        for (t, p, s, m), r in rec.items():
            f = Path(d) / t / f"{ev.slug(m)}_{p}_s{s}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(r))
        assert len(ev.load_early(d, MODELS)) == 210
        f = Path(d) / "mnli" / f"{ev.slug(MODELS[2])}_mean_s1.json"
        r = json.loads(f.read_text())
        r["val_idx_md5"] = "other"
        f.write_text(json.dumps(r))
        for broken in (lambda: f.write_text(json.dumps(r)), lambda: f.unlink()):
            broken()
            try:
                ev.load_early(d, MODELS)
            except SystemExit as exc:
                assert "ABORT" in str(exc)
            else:
                raise AssertionError("an inconsistent or missing run was accepted")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 (report every failure, then exit non-zero)
            failed += 1
            print(f"FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
