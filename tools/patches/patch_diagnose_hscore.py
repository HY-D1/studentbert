from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: failure-review diagnostic D4. The layer and skill-table diagnostics of
# scripts/diagnose_logme.py ran for LogME on the three original targets only, while the
# documented failure that exposure does not explain is on Algebra 2006, and H-score is the
# strongest estimator there too. This adds plain H-score at every layer, on the same features,
# positions and labels as the LogME rows, and a test that each H-score row equals hscore() on the
# layer's features. LogME rows, their names and their uncertainty signals are unchanged. Harry,
# 2026-09-29.

SCRIPT = Path("scripts/diagnose_logme.py")
TESTS = Path("tests/test_estimators.py")

EDITS = {
    SCRIPT: [
        ('"""Diagnose where the leakage-safe KT LogME loses the transfer signal (MRAP Section T).\n',
         '"""Diagnose where the leakage-safe KT LogME loses the transfer signal (MRAP Section T).\n'
         "\n"
         "Plain H-score is scored on the same features at every layer as well (estimator names\n"
         "hscore_kt_causal_L<k>), for the Algebra 2006 diagnostic of the failure review.\n",
         "docstring names the H-score rows"),
        ("from src.estimators.logme import logme, logme_per_group\n",
         "from src.estimators.hscore import hscore\n"
         "from src.estimators.logme import logme, logme_per_group\n",
         "import hscore"),
        ("def n_positions(subset) -> int:\n",
         "# Uncertainty signals kept per estimator family; H-score is closed form, so it has no\n"
         "# convergence flag to report.\n"
         'SIGNALS = {"logme": ("alpha", "beta", "gamma", "converged"), "hscore": ("rank", "n")}\n'
         "\n"
         "\n"
         "def n_positions(subset) -> int:\n",
         "signal keys per family"),
        ('                        (f"logme_kt_causal_per_skill_L{L}{suffix}",\n'
         "                         lambda F=F: logme_per_group(F, y, nxt, min_group=min_group))):\n",
         '                        (f"logme_kt_causal_per_skill_L{L}{suffix}",\n'
         "                         lambda F=F: logme_per_group(F, y, nxt, min_group=min_group)),\n"
         '                        (f"hscore_kt_causal_L{L}{suffix}", lambda F=F: hscore(F, y))):\n',
         "H-score at every layer"),
        ('                uncertainty_signals={k: info[k] for k in ("alpha", "beta", "gamma", '
         '"converged")},\n',
         '                uncertainty_signals={k: info[k] for k in SIGNALS[est.split("_kt_")[0]]},\n',
         "signals by family"),
        ('                    plain = [r for r in rs if "per_skill" not in r.estimator]\n'
         '                    print(f"DIAG target={target} seed={seed} cand={Path(c).name} '
         'skillrand={rnd} "\n'
         '                          f"maxdiff={diff} "\n'
         '                          + " ".join(f"L{r.metadata[\'layer\']}={r.score:.5f}" for r in '
         "plain),\n"
         "                          flush=True)\n",
         '                    plain = [r for r in rs if r.estimator.startswith("logme_kt_causal_L")]\n'
         '                    hs = [r for r in rs if r.estimator.startswith("hscore_kt_causal_L")]\n'
         '                    print(f"DIAG target={target} seed={seed} cand={Path(c).name} '
         'skillrand={rnd} "\n'
         '                          f"maxdiff={diff} "\n'
         '                          + " ".join(f"L{r.metadata[\'layer\']}={r.score:.5f}" for r in '
         "plain)\n"
         '                          + " | hscore "\n'
         '                          + " ".join(f"L{r.metadata[\'layer\']}={r.score:.5f}" for r in hs),\n'
         "                          flush=True)\n",
         "DIAG line keeps the LogME part and adds H-score"),
    ],
    TESTS: [
        ("\n\ndef test_O6_score_orientation():\n",
         "\n\ndef test_diagnostic_scores_hscore_at_every_layer():\n"
         "    _torch()\n"
         "    import importlib.util as _u\n"
         "\n"
         "    from src.estimators.features import (build_backbone, cap_positions, sample_target,\n"
         "                                         target_num_skills)\n"
         "    from src.estimators.hscore import hscore\n"
         "\n"
         '    spec = _u.spec_from_file_location("diagnose_logme", REPO / "scripts" / '
         '"diagnose_logme.py")\n'
         "    diag = _u.module_from_spec(spec)\n"
         "    spec.loader.exec_module(diag)\n"
         "    with tempfile.TemporaryDirectory() as tmp:\n"
         '        tgt = _make_processed(Path(tmp), "tgt")\n'
         "        out, _ = diag.score_layers(None, tgt, seed=7, n_students=12, max_positions=150,\n"
         '                                   min_group=20, device="cpu", d_model=16, n_layers=2,\n'
         "                                   max_seq_len=64)\n"
         "        bb = build_backbone(target_num_skills(tgt), seed=7, d_model=16, n_layers=2, "
         "max_len=64)\n"
         "        sub, _, _ = sample_target(tgt, 12, 7, 64)\n"
         "        sel = cap_positions(diag.n_positions(sub), 150, 7)\n"
         '        feats, y, _ = diag.layer_features(bb, sub, sel, "cpu")\n'
         '        hs = {r.estimator: r for r in out if r.estimator.startswith("hscore_kt_causal_L")}\n'
         '        assert sorted(hs) == [f"hscore_kt_causal_L{i}" for i in range(3)], sorted(hs)\n'
         "        for i, F in enumerate(feats):\n"
         '            r = hs[f"hscore_kt_causal_L{i}"]\n'
         "            assert abs(r.score - hscore(F, y)[0]) <= 1e-9 * max(1.0, abs(r.score))\n"
         '            assert set(r.uncertainty_signals) == {"rank", "n"}\n'
         '        lm = [r for r in out if r.estimator.startswith("logme_kt_causal_L")]\n'
         '        assert len(lm) == 3 and all(set(r.uncertainty_signals) == {"alpha", "beta", '
         '"gamma",\n'
         '                                                                   "converged"} for r '
         "in lm)\n"
         "\n\ndef test_O6_score_orientation():\n",
         "test: H-score rows equal hscore() per layer, LogME signals unchanged"),
    ],
}


def apply(text: str, old: str, new: str, why: str) -> str:
    # Insertion-style edits keep their anchor, so "old in text" stays true after patching;
    # deciding by containment direction keeps re-runs as no-ops.
    if new in text and (old not in text or old in new):
        print(f"skip (already applied): {why}")
        return text
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1; nothing written")
    print(f"ok: {why}")
    return text.replace(old, new, 1)


def main() -> None:
    staged = {}
    for path, edits in EDITS.items():
        src = path.read_text(encoding="utf-8")
        out = src
        for old, new, why in edits:
            out = apply(out, old, new, f"{path}: {why}")
        staged[path] = (src, out)
    for path, (src, out) in staged.items():
        if out != src:
            path.write_text(out, encoding="utf-8")
        print("md5", hashlib.md5(path.read_bytes()).hexdigest(), path)


if __name__ == "__main__":
    main()
