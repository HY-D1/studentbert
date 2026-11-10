from __future__ import annotations

"""Smoke test for scripts/task4_score_parc.py on a synthetic feature file. Run from the code root:

  PYTHONPATH=. python tests/test_task4_score_parc.py
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t4p", REPO / "scripts/task4_score_parc.py")
t4p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t4p)


def test_records_and_resume() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "feat" / "rte"
        root.mkdir(parents=True)
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=800)
        cls = rng.normal(size=(800, 40))
        cls[:, 0] += 3 * y
        np.savez(root / "org__m_s42.npz", labels=y, cls=cls, mean=rng.normal(size=(800, 40)))
        (root / "org__m_s42.json").write_text(json.dumps(
            {"task": "rte", "model": "org/m", "seed": 42, "extract_s": 1.0, "peak_gpu_mb": 0}))
        out = Path(d) / "scores.jsonl"
        t4p.main(["--features", str(Path(d) / "feat"), "--out", str(out)])
        recs = [json.loads(x) for x in out.read_text().splitlines()]
        assert len(recs) == 2 and {r["pooling"] for r in recs} == {"cls", "mean"}
        by = {r["pooling"]: r for r in recs}
        assert by["cls"]["score"] > by["mean"]["score"] and by["cls"]["n_probe"] == 500
        t4p.main(["--features", str(Path(d) / "feat"), "--out", str(out)])
        assert len(out.read_text().splitlines()) == 2


def main() -> None:
    try:
        test_records_and_resume()
        print("ok   test_records_and_resume\n1 passed, 0 failed")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL test_records_and_resume: {e!r}\n0 passed, 1 failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
