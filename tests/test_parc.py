from __future__ import annotations

"""Tests for src/estimators/parc.py. Run from the code root:

  PYTHONPATH=. python tests/test_parc.py
"""

import sys

import numpy as np

from src.estimators.parc import parc, probe_indices


def separable(n: int = 2000, d: int = 64, seed: int = 0, gap: float = 3.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    F = rng.normal(size=(n, d))
    F[:, :8] += gap * y[:, None]
    return F, y


def test_orientation() -> None:
    F, y = separable()
    good, info = parc(F, y, seed=1)
    noise = np.random.default_rng(5).normal(size=F.shape)
    bad, _ = parc(noise, y, seed=1)
    assert good > 0.3 and abs(bad) < 0.05 and good > bad
    assert info["n_probe"] == 500 and info["dim"] == 32 and info["classes"] == 2


def test_deterministic_and_shared_probe() -> None:
    F, y = separable()
    a, _ = parc(F, y, seed=7)
    b, _ = parc(F, y, seed=7)
    assert a == b
    assert np.array_equal(probe_indices(y, 500, 7), probe_indices(y, 500, 7))
    assert not np.array_equal(probe_indices(y, 500, 7), probe_indices(y, 500, 8))


def test_rare_class_gets_two() -> None:
    y = np.zeros(10000, dtype=int)
    y[[17, 4242, 9001]] = 1
    for s in range(20):
        idx = probe_indices(y, 500, s)
        assert idx.size == 500 and np.sum(y[idx] == 1) >= 2 and np.unique(idx).size == 500


def test_small_and_degenerate_inputs() -> None:
    F, y = separable(n=300, d=8)
    _, info = parc(F, y, seed=0)
    assert info["n_probe"] == 300 and info["dim"] == 8
    try:
        parc(F, np.zeros(300, dtype=int))
    except ValueError:
        pass
    else:
        raise AssertionError("a single class was accepted")


def test_multiclass() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, size=3000)
    F = rng.normal(size=(3000, 48))
    F[np.arange(3000), y] += 4.0
    s, info = parc(F, y, seed=2)
    assert s > 0.2 and info["classes"] == 4


def main() -> None:
    tests = [test_orientation, test_deterministic_and_shared_probe, test_rare_class_gets_two,
             test_small_and_degenerate_inputs, test_multiclass]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
