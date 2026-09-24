"""MRAP Section O tests for the transferability estimators and the benchmark builder.

pytest is declared in environment.yml but missing from the sb env (checked 2026-09-22), so this
file runs either way:
  PYTHONPATH=. python tests/test_estimators.py --require-torch    (the cluster invocation)
  PYTHONPATH=. python -m pytest tests/test_estimators.py
Torch tests skip without torch; --require-torch turns every skip into a failure.
"""

from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import io
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.estimators.base import EstimatorResult, rank_candidates  # noqa: E402
from src.estimators.logme import logme, logme_per_group  # noqa: E402

_RUNNER = False


class _Skip(Exception):
    pass


def _torch():
    try:
        import torch
    except ModuleNotFoundError:
        if _RUNNER:
            raise _Skip("torch not installed") from None
        import pytest

        pytest.skip("torch not installed")
    return torch


def _explicit_evidence(F, y, a, b):
    F = F - F.mean(0, keepdims=True)
    y = y.astype(float) - y.mean()
    N, D = F.shape
    A = a * np.eye(D) + b * F.T @ F
    m = b * np.linalg.solve(A, F.T @ y)
    return (0.5 * D * math.log(a) + 0.5 * N * math.log(b) - 0.5 * np.linalg.slogdet(A)[1]
            - 0.5 * b * float(((y - F @ m) ** 2).sum()) - 0.5 * a * float(m @ m)
            - 0.5 * N * math.log(2 * math.pi)) / N


def _make_processed(root: Path, name: str, *, n_students: int = 24, num_skills: int = 6,
                    seed: int = 0) -> Path:
    """A tiny dataset in the shared schema (sequences.npz, splits.json, skill_vocab.json)."""
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    lengths = rng.integers(20, 40, size=n_students)
    total = int(lengths.sum())
    ids = np.arange(100, 100 + n_students)
    np.savez(d / "sequences.npz", student_ids=ids,
             skill=rng.integers(1, num_skills + 1, size=total),
             correct=rng.integers(0, 2, size=total), time_bin=rng.integers(1, 6, size=total),
             offsets=np.concatenate([[0], np.cumsum(lengths)]))
    n_train = n_students - 4
    (d / "splits.json").write_text(json.dumps({"train": ids[:n_train].tolist(),
                                               "val": ids[n_train:n_train + 2].tolist(),
                                               "test": ids[n_train + 2:].tolist()}))
    (d / "skill_vocab.json").write_text(json.dumps({f"k{i}": i for i in range(1, num_skills + 1)}))
    return d


def _fake_encoder(path: Path, source_dir: str, num_skills: int, seed: int, d_model: int = 16,
                  n_layers: int = 2, max_len: int = 64) -> Path:
    torch = _torch()
    from src.estimators.features import build_backbone

    bb = build_backbone(num_skills, seed=seed, d_model=d_model, n_layers=n_layers, max_len=max_len)
    torch.save({"model_state": bb.state_dict(), "epoch": 3, "mlm_loss": 1.25,
                "config": {"processed_dir": source_dir, "objective": "full"}}, path)
    return path


# --------------------------------------------------------------------------- LogME, numpy only
def test_logme_matches_explicit_evidence():
    rng = np.random.default_rng(1)
    for N, D in ((4000, 64), (300, 256)):
        F = rng.normal(size=(N, D))
        y = (rng.random(N) < 1 / (1 + np.exp(-F[:, :3].sum(1)))).astype(int)
        e, info = logme(F, y)
        assert info["converged"], info
        assert abs(e - _explicit_evidence(F, y, info["alpha"], info["beta"])) < 1e-10


def test_logme_refuses_unbounded_evidence():
    rng = np.random.default_rng(2)
    F = rng.normal(size=(40, 256))
    y = (rng.random(40) < 0.5).astype(int)
    try:
        logme(F, y)
    except ValueError:
        return
    raise AssertionError("N < rank + 2 can be interpolated exactly and must be refused")


def test_logme_label_flip_invariant():
    rng = np.random.default_rng(3)
    F = rng.normal(size=(2000, 32))
    y = (rng.random(2000) < 0.3).astype(int)
    assert abs(logme(F, y)[0] - logme(F, 1 - y)[0]) < 1e-12


def test_logme_per_group_is_the_shared_evidence_optimum():
    rng = np.random.default_rng(4)
    F = rng.normal(size=(3000, 16))
    g = rng.integers(0, 6, size=3000)
    y = (rng.random(3000) < 1 / (1 + np.exp(-(0.3 * g - 0.8 + F[:, 0] * (g % 2))))).astype(int)
    score, info = logme_per_group(F, y, g, min_group=20)
    total = sum(_explicit_evidence(F[g == k], y[g == k], info["alpha"], info["beta"])
                * (g == k).sum() for k in range(6)) / 3000
    assert abs(score - total) < 1e-10
    for fa, fb in ((1.5, 1.0), (1 / 1.5, 1.0), (1.0, 1.5), (1.0, 1 / 1.5)):
        other = sum(_explicit_evidence(F[g == k], y[g == k], info["alpha"] * fa,
                                       info["beta"] * fb) * (g == k).sum() for k in range(6))
        assert other / 3000 <= score + 1e-12, "fixed point is not the evidence maximum"
    assert abs(logme_per_group(F, y, np.zeros(3000, int))[0] - logme(F, y)[0]) < 1e-12


def test_logme_per_group_exact_when_groups_are_smaller_than_the_feature_dim():
    # Junyi has 1,326 skills over 50,000 positions, so many readouts see fewer samples than the
    # 256 feature dimensions; their log-determinant then carries (D - rank) prior-only terms.
    rng = np.random.default_rng(6)
    D = 16
    sizes = [400, 400, 400] + [12] * 5
    g = np.repeat(np.arange(len(sizes)), sizes)
    F = rng.normal(size=(g.size, D))
    y = (rng.random(g.size) < 1 / (1 + np.exp(-F[:, 0] - 0.2 * g))).astype(int)
    score, info = logme_per_group(F, y, g, min_group=1)
    assert info["groups_own_readout"] == len(sizes)
    total = sum(_explicit_evidence(F[g == k], y[g == k], info["alpha"], info["beta"])
                * (g == k).sum() for k in range(len(sizes))) / g.size
    assert abs(score - total) < 1e-10, (score, total)


def test_old_compute_logme_double_counts_the_prior():
    src = (REPO / "scripts" / "compute_logme.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "logme")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "old_logme", "exec"), ns)
    rng = np.random.default_rng(5)
    F = rng.normal(size=(20000, 64)).astype(np.float32)
    y = (rng.random(20000) < 1 / (1 + np.exp(-0.5 * F[:, :4].sum(1)))).astype(np.int64)
    old = ns["logme"](F.copy(), y.copy())
    new, info = logme(F, y)
    gap = old - new
    assert gap < 0
    assert abs(gap + info["gamma"] / (2 * 20000)) < 1e-4 * info["gamma"] / 20000 + 1e-7, gap


# --------------------------------------------------------------------------- Section O
def test_O1_causal_features_ignore_the_future():
    torch = _torch()
    from src.estimators.features import build_backbone, encode_causal

    bb = build_backbone(6, seed=0, d_model=32, n_layers=2, max_len=64).eval()
    g = torch.Generator().manual_seed(0)
    B, L, cut = 2, 12, 7
    skill = torch.randint(1, 7, (B, L), generator=g)
    correct = torch.randint(0, 2, (B, L), generator=g)
    tb = torch.randint(1, 6, (B, L), generator=g)
    pad = torch.zeros(B, L, dtype=torch.bool)
    s2, c2, t2 = skill.clone(), correct.clone(), tb.clone()
    c2[:, cut:] = 1 - c2[:, cut:]
    s2[:, cut:] = s2[:, cut:] % 6 + 1
    t2[:, cut:] = t2[:, cut:] % 5 + 1
    with torch.no_grad():
        causal_gap = (encode_causal(bb, skill, correct, tb, pad)[:, :cut]
                      - encode_causal(bb, s2, c2, t2, pad)[:, :cut]).abs().max().item()
        bidir_gap = (bb.encode(skill, correct, tb, key_padding_mask=pad)[:, :cut]
                     - bb.encode(s2, c2, t2, key_padding_mask=pad)[:, :cut]).abs().max().item()
    assert causal_gap <= 1e-6, f"causal features at t < {cut} moved by {causal_gap}"
    assert bidir_gap > 1e-4, f"negative control failed: bidirectional gap {bidir_gap}"


def test_O1_extractor_reproduces_the_finetune_encoder():
    torch = _torch()
    from src.estimators.features import build_backbone, encode_causal
    from src.utils import set_seed

    spec = importlib.util.spec_from_file_location("finetune_edubert",
                                                  REPO / "scripts" / "finetune_edubert.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    set_seed(11)
    kt = mod.EduBERTForKT(num_skills=6, d_model=32, n_layers=2, dropout=0.1, max_len=64).eval()
    bb = build_backbone(6, seed=11, d_model=32, n_layers=2, max_len=64).eval()
    assert torch.equal(kt.backbone.skill_emb.weight, bb.skill_emb.weight), \
        "random start differs from the fine-tune's at the same seed"
    g = torch.Generator().manual_seed(1)
    skill = torch.randint(1, 7, (3, 15), generator=g)
    correct = torch.randint(0, 2, (3, 15), generator=g)
    tb = torch.randint(1, 6, (3, 15), generator=g)
    pad = torch.zeros(3, 15, dtype=torch.bool)
    pad[0, 10:] = True
    with torch.no_grad():
        ref = kt.encode_causal(skill, correct, tb, pad)
        ours = encode_causal(bb, skill, correct, tb, pad)
    real = ~pad
    assert (ref[real] - ours[real]).abs().max().item() == 0.0


def test_O2_estimators_never_read_results():
    banned_modules = ("analysis", "wandb")
    banned_text = ("RESULTS", "inventory", "perseed", "sacct", "wandb", ".tsv")
    files = sorted((REPO / "src" / "estimators").glob("*.py"))
    files.append(REPO / "scripts" / "score_transferability.py")
    files += [REPO / "scripts" / "score_task2.py", REPO / "scripts" / "fewshot_proxy.py"]
    for path in files:
        tree = ast.parse(path.read_text())
        body = tree.body[1:] if tree.body and isinstance(getattr(tree.body[0], "value", None),
                                                         ast.Constant) else tree.body
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) \
                    else [node.module or ""]
                for n in names:
                    assert not n.startswith(banned_modules), f"{path.name} imports {n}"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for t in banned_text:
                    assert t not in node.value, f"{path.name} mentions {t!r} outside its docstring"


def test_O3_vocabulary_loaded_by_intent_not_shape():
    torch = _torch()
    from src.estimators.features import build_backbone, load_candidate

    with tempfile.TemporaryDirectory() as tmp:
        ck = _fake_encoder(Path(tmp) / "src.pt", "/x/processed/srcA", 6, seed=5)
        tgt = build_backbone(6, seed=9, d_model=16, n_layers=2, max_len=64)
        own_before = tgt.skill_emb.weight.detach().clone()
        rep = load_candidate(tgt, ck, "tgtB")
        assert not rep["in_domain"]
        assert rep["skipped"] == ["skill_emb.weight", "skill_head.bias", "skill_head.weight"]
        assert rep["loaded"] == rep["total"] - 3
        assert torch.equal(tgt.skill_emb.weight, own_before), "same-K source skill table loaded"
        same = build_backbone(6, seed=9, d_model=16, n_layers=2, max_len=64)
        rep2 = load_candidate(same, ck, "srcA")
        src_state = torch.load(ck, weights_only=False)["model_state"]
        assert rep2["in_domain"] and rep2["loaded"] == rep2["total"] and not rep2["skipped"]
        assert torch.equal(same.skill_emb.weight, src_state["skill_emb.weight"])
        bare = Path(tmp) / "bare.pt"
        torch.save({"model_state": src_state}, bare)
        try:
            load_candidate(build_backbone(6, seed=9, d_model=16, n_layers=2, max_len=64), bare,
                           "tgtB")
        except ValueError:
            return
        raise AssertionError("a checkpoint without config.processed_dir must be refused")


def test_O4_seed_reproducibility():
    _torch()
    from src.estimators.kt_logme import score_kt_logme

    with tempfile.TemporaryDirectory() as tmp:
        tgt = _make_processed(Path(tmp), "tgt")
        kw = {"n_students": 12, "max_positions": None, "device": "cpu", "d_model": 16,
              "n_layers": 2, "max_seq_len": 64, "min_group": 20}
        a = score_kt_logme(None, tgt, seed=3, **kw)
        b = score_kt_logme(None, tgt, seed=3, **kw)
        c = score_kt_logme(None, tgt, seed=4, **kw)
        for x, y in zip(a, b):
            assert x.score == y.score and x.metadata["sample_fingerprint"] == \
                y.metadata["sample_fingerprint"]
        assert a[0].metadata["sample_fingerprint"] != c[0].metadata["sample_fingerprint"]


def test_O5_candidates_share_the_target_sample():
    _torch()
    from src.estimators.kt_logme import score_kt_logme

    with tempfile.TemporaryDirectory() as tmp:
        tgt = _make_processed(Path(tmp), "tgt")
        ck = _fake_encoder(Path(tmp) / "a.pt", "/x/processed/srcA", 6, seed=21)
        kw = {"n_students": 12, "max_positions": 300, "device": "cpu", "d_model": 16,
              "n_layers": 2, "max_seq_len": 64}
        s = score_kt_logme(None, tgt, seed=42, **kw)
        p = score_kt_logme(str(ck), tgt, seed=42, **kw)
        for x, y in zip(s, p):
            assert x.metadata["sample_fingerprint"] == y.metadata["sample_fingerprint"]
            assert x.n_target_examples == y.n_target_examples
            assert x.metadata["positions_total"] == y.metadata["positions_total"]
        assert p[0].metadata["load"]["skipped"], "cross-domain vocabulary should be skipped"
        assert s[0].score != p[0].score


def test_diagnostic_layers_reproduce_the_estimator():
    _torch()
    import importlib.util as _u

    from src.estimators.features import (build_backbone, cap_positions, kt_features,
                                         sample_target)

    spec = _u.spec_from_file_location("diagnose_logme", REPO / "scripts" / "diagnose_logme.py")
    diag = _u.module_from_spec(spec)
    spec.loader.exec_module(diag)
    with tempfile.TemporaryDirectory() as tmp:
        tgt = _make_processed(Path(tmp), "tgt")
        bb = build_backbone(6, seed=7, d_model=16, n_layers=2, max_len=64)
        sub, _, _ = sample_target(tgt, 12, 7, 64)
        F, y, _ = kt_features(bb, sub, "cpu")
        sel = cap_positions(len(y), 150, 7)
        assert diag.n_positions(sub) == len(y)
        feats, y2, _ = diag.layer_features(bb, sub, sel, "cpu")
        assert len(feats) == 3, "embedding output plus one entry per encoder layer"
        assert np.allclose(feats[-1], F[sel], rtol=0, atol=1e-6) and np.array_equal(y2, y[sel])
        assert not np.allclose(feats[0], feats[-1], atol=1e-3), "layers should differ"


def test_O6_score_orientation():
    def mk(c, score, direction="higher_is_better", est="e"):
        return EstimatorResult(est, "F", c, "t", 1, score, direction, 1, 1, 0.0, 0.0, None)

    assert [r.candidate for r in rank_candidates([mk("a", 1.0), mk("b", 2.0)])] == ["b", "a"]
    lo = [mk("a", 1.0, "lower_is_better"), mk("b", 2.0, "lower_is_better")]
    assert [r.candidate for r in rank_candidates(lo)] == ["a", "b"]
    mixed = [mk("a", 1), mk("b", 1, est="x")]
    for bad in (lambda: mk("a", 1.0, "up"), lambda: rank_candidates(mixed)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("invalid direction or mixed estimators must raise")


# --------------------------------------------------------------------------- builder
def _kt_log(run, init, value, *, students=1000, seed=42, encoder=None, wb_first=True, rid="r1"):
    wb = [f"wandb: Syncing run {run}", f"wandb: View run at https://wandb.ai/dhy666666o-n/"
          f"StudentBERT/runs/{rid}", f"wandb: View run {run} at: https://wandb.ai/dhy666666o-n/"
          f"StudentBERT/runs/{rid}"]
    out = [f"device=cuda  num_skills=123  run={run}  init={init}  train_students={students}"
           f"  seed={seed}"]
    if encoder:
        out.append(f"loaded 79/82 tensors from ../checkpoints/{encoder} (epoch 10, mlm 2.8784)")
    out += [f"epoch {i:3d}  train_loss=0.6000  val_AUC=0.7000" for i in range(1, 31)]
    out += ["", f"=== EduBERT-KT ({run}, init={init}) ===", "train students : 1000",
            "best val AUC   : 0.7100", f"test AUC       : {value:.4f}", "test ECE       : 0.0100"]
    return (wb + out) if wb_first else (out + wb)


def _write(tmp: Path, name: str, lines: list[str]) -> None:
    (tmp / name).write_text("\n".join(lines) + "\n")


def _build(tmp: Path, **extra):
    from analysis import build_transfer_benchmark as btb

    execs = []
    for p in sorted(tmp.glob("*.log")):
        execs.extend(btb.scan_file(p)[0])
    btb.annotate(execs, extra.get("sacct", {}), extra.get("gres", {}), extra.get("wandb", {}))
    dups = btb.resolve_duplicates(execs, 0.0005)
    btb.pair_scratch(execs)
    return {(e["file"], e["run"]): e for e in execs}, dups


def test_O7_quarantine_and_leak_protection():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _write(tmp, "w6_probe_7988582.log", [
            "device=cuda  num_skills=102  run=edubert_assist2017_probe_scratch_seed42  "
            "init=scratch  seed=42",
            "=== probe (skill-identity) (edubert_assist2017_probe_scratch_seed42, "
            "init=scratch) ===",
            "test probe acc: 0.9990"])
        _write(tmp, "w6_dropoutK_s42_7987745.log", [
            "device=cuda  task=dropout  num_skills=102  run=edubert_assist2017_scratch_dropout_"
            "k100_seed42  init=scratch  seed=42",
            "=== dropout (edubert_assist2017_scratch_dropout_k100_seed42, init=scratch) ===",
            "test AUC          : 0.9000"])
        _write(tmp, "w8_a09abl_full_s42_9000001.log", _kt_log(
            "edubert_assist2009_a09abl_full_seed42", "pretrained", 0.8701,
            encoder="edubert_ednet_pretrain_full_encoder.pt"))
        rows, _ = _build(tmp)
        v1 = rows[("w6_probe_7988582.log", "edubert_assist2017_probe_scratch_seed42")]
        leak = rows[("w6_dropoutK_s42_7987745.log",
                     "edubert_assist2017_scratch_dropout_k100_seed42")]
        good = rows[("w8_a09abl_full_s42_9000001.log", "edubert_assist2009_a09abl_full_seed42")]
        assert v1["leakage_status"] == "quarantined" and v1["exclusion"]
        assert leak["leakage_status"] == "leaked_K" and leak["exclusion"]
        assert not good["exclusion"] and good["fam"]["track"] == "A"
        assert good["encoder_status"] == "ok" and good["wandb_id"] == "r1"


def test_builder_duplicate_rules_and_joins():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        run_b = "edubert_assist2017_kt_assist_fromednet_n3000_seed42"
        _write(tmp, f"kt_assist_fromednet_n3000_seed42_10000001.log",
               _kt_log(run_b, "pretrained", 0.6961, students=1366,
                       encoder="edubert_ednet_pretrain_full_encoder.pt", rid="pin1"))
        w6 = (_kt_log("edubert_assist2017_kt_assist_scratch_n3000_seed42", "scratch", 0.6690,
                      students=1366, rid="w6a", wb_first=False)
              + _kt_log(run_b, "pretrained", 0.6950, students=1366,
                        encoder="edubert_ednet_pretrain_full_encoder.pt", rid="w6b"))
        _write(tmp, "w6_kt_assist_n3000_7900000.log", w6)
        run_d = "edubert_ednet_drop_ednet_fromassist_k10_n3000_seed3"
        for jid, v in (("8128209", 0.5043), ("8130112", 0.7152)):
            _write(tmp, f"w7_de_fromassist_k10_s3_{jid}.log", [
                f"device=cuda  task=dropout  num_skills=142  run={run_d}  init=pretrained  seed=3",
                "loaded 79/82 encoder tensors from ../checkpoints/edubert_assist2017_pretrain_full_"
                "encoder.pt", f"=== dropout ({run_d}, init=pretrained) ===",
                f"test AUC          : {v:.4f}"])
        run_s = "edubert_junyi_kt_junyi_fromednet_n150000d2_src_n3000_seed5"
        for jid, v in (("10418300", 0.7388), ("10418301", 0.7391)):
            _write(tmp, f"kt_junyi_fromednet_n150000d2_src_n3000_seed5_{jid}.log",
                   _kt_log(run_s, "pretrained", v, students=3000, seed=5, rid=f"s{jid}",
                           encoder="edubert_ednet_pretrain_ednet_n150000d2_encoder.pt"))
        run_x = "edubert_assist2009_a09abl_skill_only_seed1"
        for jid, v in (("9100001", 0.8690), ("9100002", 0.8697)):
            _write(tmp, f"w8_a09abl_skill_only_s1_{jid}.log",
                   _kt_log(run_x, "pretrained", v, seed=1, rid=f"x{jid}",
                           encoder="edubert_ednet_pretrain_ednet_skill_only_encoder.pt"))
        _write(tmp, "w8_a09abl_correct_only_s1_9200001.log",
               _kt_log("edubert_assist2009_a09abl_correct_only_seed1", "pretrained", 0.8660,
                       seed=1, rid="bad", encoder="edubert_ednet_pretrain_full_encoder.pt"))
        sacct = {"9100001": {"State": "COMPLETED", "NodeList": "d1001", "Elapsed": "00:30:00"},
                 "9100002": {"State": "TIMEOUT", "NodeList": "c2204", "Elapsed": "06:00:00"},
                 "10000001": {"State": "COMPLETED", "NodeList": "d1001", "Elapsed": "00:20:00"}}
        gres = {"d1001": "v100-sxm2", "c2204": "v100-pcie"}
        wandb = {"pin1": {"id": "pin1", "name": run_b, "summary": {"test/auc": 0.69612345}},
                 "s10418300": {"id": "s10418300", "name": run_s, "summary": {"test/auc": 0.7399}}}
        rows, dups = _build(tmp, sacct=sacct, gres=gres, wandb=wandb)

        pin = rows[("kt_assist_fromednet_n3000_seed42_10000001.log", run_b)]
        old = rows[("w6_kt_assist_n3000_7900000.log", run_b)]
        assert not pin["exclusion"] and "superseded campaign W6" in old["exclusion"]
        assert pin["values"]["test_auc"] == 0.69612345
        assert pin["prov"]["test_auc"] == "wandb_verified"
        assert pin["gpu_type"] == "v100-sxm2" and pin["wandb_id"] == "pin1"
        assert old["wandb_id"] == "w6b", "W&B id must come from the copy's own log, by name"
        scr = rows[("w6_kt_assist_n3000_7900000.log",
                    "edubert_assist2017_kt_assist_scratch_n3000_seed42")]
        assert scr["wandb_id"] == "w6a" and not scr["exclusion"]
        assert abs(pin["gain"] - (0.69612345 - 0.6690)) < 1e-12

        drops = [rows[(f"w7_de_fromassist_k10_s3_{j}.log", run_d)] for j in ("8128209", "8130112")]
        assert all("unresolved duplicate" in d["exclusion"] for d in drops)
        s11 = [rows[(f"kt_junyi_fromednet_n150000d2_src_n3000_seed5_{j}.log", run_s)]
               for j in ("10418300", "10418301")]
        assert "W&B" in s11[0]["exclusion"] and "conflict" not in s11[1]["exclusion"]
        assert s11[1]["exclusion"] == "" and s11[1]["fam"]["track"] == "S11"
        x = [rows[(f"w8_a09abl_skill_only_s1_{j}.log", run_x)] for j in ("9100001", "9100002")]
        assert not x[0]["exclusion"] and "job ended TIMEOUT" in x[1]["exclusion"]
        bad = rows[("w8_a09abl_correct_only_s1_9200001.log",
                    "edubert_assist2009_a09abl_correct_only_seed1")]
        assert bad["exclusion"].startswith("mismatch"), bad["exclusion"]
        assert {d["run"] for d in dups} == {run_b, run_d, run_s, run_x}


def test_builder_classifies_every_known_family():
    from analysis.build_transfer_benchmark import classify

    cases = {
        "edubert_assist2017_objabl_full_n1000_seed42": ("A", "assist2017", "full"),
        "edubert_assist2017_tg_objabl_scratch_n1000_seed3": ("A", "assist2017", "scratch"),
        "edubert_ednet_regime_ednet_correct_only_n1000_seed5": ("A", "ednet", "correct_only"),
        "edubert_bridge2006_scratch_bridge2006_seed2": ("A", "bridge2006", "scratch"),
        "edubert_junyi_kt_junyi_indomain_n3000_seed1": ("B", "junyi", "src:junyi"),
        "edubert_ednet_ns_ednet_fromassist_n3000_seed4": ("C-ns", "ednet", "src:assist2017"),
        "edubert_assist2017_kt_assist_fromednet_n1366d1_src_n3000_seed1":
            ("S11", "assist2017", "ednet_n1366_d1"),
        "edubert_assist2017_junyi_dropout_k50_seed2": ("C-drop", "assist2017", "src:junyi"),
        "edubert_junyi_drop_junyi_fromednet_k5_n3000_seed1": ("C-drop", "junyi", "src:ednet"),
        "edubert_algebra2006_probe7_algebra2006_scratch_s2": ("probe7", "algebra2006", "scratch"),
        "edubert_ednet_tg_probe7_ednet_skill_only_s42": ("probe7", "ednet", "skill_only"),
        "edubert_assist2017_probe2_ednet_seed1": ("probe2", "assist2017", "src:ednet"),
        "edubert_junyi_probe2_junyi_fromassist_seed2": ("probe2", "junyi", "src:assist2017"),
        "edubert_assist2017_trunc_skill_only_k160_seed3": ("A", "assist2017", "skill_only"),
        "edubert_ednet_ktfull_ednet_fromjunyi_n20000_seed1": ("B", "ednet", "src:junyi"),
        "edubert_assist2017_ednet_cold_pre_n100_s42": ("B", "assist2017", "src:ednet"),
        "edubert_assist2017_cold_scr_n25_s1": ("B", "assist2017", "scratch"),
        "edubert_assist2017_drop_assist_fromjunyi_cens_k200_seed2": ("C-drop", "assist2017",
                                                                     "src:junyi"),
        "edubert_bridge2006_tgb_bridge2006_fromassist2009_n3000_seed4": ("B", "bridge2006",
                                                                         "src:assist2009"),
        "edubert_algebra2005_tga_correct_only_r128d1_full_seed5": ("A", "algebra2005",
                                                                   "correct_only@r128d1"),
    }
    for run, want in cases.items():
        f = classify(run, "kt")
        assert f and (f["track"], f["target"], f["candidate"]) == want, (run, f)
    assert classify("edubert_assist2017_kt_ednet_fromednet_n3000_seed1", "kt") is None
    f = classify("edubert_ednet_regime_ednet_full_n1000_seed2", "kt")
    assert f["source"] == "junyi" and f["budget"] == "n1000", f
    assert classify("edubert_assist2017_trunc_full_k512_seed1", "kt")["budget"] == "trunc_K512"
    assert classify("edubert_assist2017_drop_assist_scratch_cens_k100_seed1", "dropout")["censored"]


def test_builder_keeps_the_confirmatory_holdout_closed():
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        _write(tmp, "tg1_x_1.log", _kt_log("edubert_xes3g5m_kt_xes_scratch_n3000_seed1", "scratch",
                                             0.7, rid="h1"))
        rows, _ = _build(tmp)
        (e,) = rows.values()
        assert e["exclusion"].startswith("confirmatory holdout"), e["exclusion"]


def test_evaluator_views_and_truncation_policies():
    from analysis import evaluate_estimators as ev

    vals = {"scratch": {1: 0.80, 2: 0.81, 3: 0.80, 4: 0.81},
            "src:a": {1: 0.70, 2: 0.71, 3: 0.70, 4: 0.71},
            "src:b": {1: 0.60, 2: 0.61, 3: 0.60, 4: 0.61}}
    full = ev.annotate(vals, 0.001, 500)
    pre = ev.annotate({c: v for c, v in vals.items() if c != "scratch"}, 0.001, 500)
    r = ev.judge({"src:a": 2.0, "src:b": 1.0}, pre, full, 0.001)
    assert r["top1"] and r["equivalent"], "best pretrained pick must be tied with the view's best"
    assert r["negative_choice"], "negative transfer is still judged against scratch"
    pol = ev.lodo_policies()
    assert pol["LODO pps_effective"]("assist2017", "trunc_K10") == "correct_only"
    assert pol["LODO pps_effective"]("assist2017", "trunc_K512") == "full"
    ks = {pol["LODO n_skills"]("assist2017", f"trunc_K{k}") for k in (10, 40, 512)}
    assert len(ks) == 1, "a cap-free feature must not change with K"
    # End to end through main(): scratch wins the cell outright, the size rule picks the best
    # pretrained source, and the pretrained view must count that pick as tied with its best.
    cols = ["track", "metric", "target_dataset", "budget", "candidate", "finetune_seed", "value",
            "valid_for_primary_analysis", "run_id", "log_file"]
    lines = ["\t".join(cols)]
    for cand, base in (("scratch", 0.80), ("src:big", 0.70), ("src:small", 0.60),
                       ("src:unscored", 0.75)):
        for seed, eps in ((1, 0.0), (2, 0.01), (3, 0.0), (4, 0.01)):
            lines.append("\t".join(["B", "test_auc", "tgt", "n3000", cand, str(seed),
                                    str(base + eps), "yes", f"r_{cand}_{seed}", "x.log"]))
    with tempfile.TemporaryDirectory() as tmp:
        ex = Path(tmp) / "executions.tsv"
        ex.write_text("\n".join(lines) + "\n")
        for ds, n in (("big", 50), ("small", 5), ("unscored", 1)):
            (Path(tmp) / ds).mkdir()
            (Path(tmp) / ds / "splits.json").write_text(json.dumps({"train": list(range(n))}))
        argv = sys.argv
        sys.argv = ["evaluate_estimators.py", "--executions", str(ex), "--boots", "500",
                    "--processed-root", tmp, "--out-prefix", str(Path(tmp) / "e")]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ev.main()
        finally:
            sys.argv = argv
        got = {r["view"]: r for r in csv.DictReader(open(Path(tmp) / "e_cells.tsv"),
                                                    delimiter="\t")
               if r["estimator"] == "largest source"}
        assert got["pretrained"]["top1"] == "False", "src:unscored (0.75) beats src:big (0.70)"
        assert got["pretrained"]["coverage"] == "3/3", "sizes must cover every source"
        assert got["practical"]["top1"] == "False" and got["practical"]["negative_choice"] == "True"
        assert abs(float(got["practical"]["regret"]) - 0.10) < 1e-9
        (Path(tmp) / "junyi").mkdir()
        (Path(tmp) / "junyi" / "splits.json").write_text(json.dumps({"train": [1, 2]}))
        try:
            ev.train_split_sizes(tmp, {"src:junyi"})
        except SystemExit as err:
            assert "49153" in str(err)
        else:
            raise AssertionError("a size that disagrees with RESULTS.md must stop the run")
    with tempfile.TemporaryDirectory() as tmp:
        j = Path(tmp) / "p.jsonl"
        j.write_text(json.dumps({"estimator": "fewshot_ft_fit50_e5", "candidate": "scratch",
                                 "target": "tgt", "seed": 1, "score": 0.6,
                                 "metadata": {"target_budget": "n3000"}}) + "\n")
        assert ("fewshot_ft_fit50_e5", "tgt", "n3000") in ev.logme_scores([str(j)])
    # Coverage: an unscored candidate that is best still counts against the choice.
    vals3 = {"src:a": {1: 0.60, 2: 0.61}, "src:b": {1: 0.70, 2: 0.71},
             "src:c": {1: 0.50, 2: 0.51}}
    st3 = ev.annotate(vals3, 0.001, 200)
    r = ev.judge({"src:a": 1.0, "src:c": 0.0}, st3, st3, 0.001)
    assert not r["top1"] and r["coverage"] == "2/3" and abs(r["regret"] - 0.10) < 1e-9


def test_draws_are_replicates_not_candidates():
    from analysis.build_transfer_benchmark import split_draw_cells

    seeds = {1: 0.0, 2: 0.002}
    cell = {"scratch": {s: 0.60 + e for s, e in seeds.items()},
            "full": {s: 0.70 + e for s, e in seeds.items()},
            "skill_only": {s: 0.69 + e for s, e in seeds.items()},
            "correct_only": {s: 0.66 + e for s, e in seeds.items()}}
    for obj, base in (("full", 0.70), ("skill_only", 0.68), ("correct_only", 0.67)):
        for d, off in ((42, 0.0), (1, 0.003), (2, -0.003)):
            if obj == "full" and d == 42:
                continue
            cell[f"{obj}@r128d{d}"] = {s: base + off + e for s, e in seeds.items()}
    out, dm = split_draw_cells({("A", "tgt", "n1000"): cell})
    assert set(out[("A", "tgt", "n1000")]) == {"scratch", "full", "skill_only", "correct_only"}
    r128 = out[("A-r128", "tgt", "n1000")]
    assert set(r128) == {"scratch", "full", "skill_only", "correct_only"}
    assert abs(r128["skill_only"][1] - 0.68) < 1e-12 and abs(r128["full"][1] - 0.70) < 1e-12
    assert abs(dm[("tgt", "n1000", "correct_only")][1] - 0.674) < 1e-12, "mean over both seeds"
    del cell["correct_only@r128d2"]
    out, dm = split_draw_cells({("A", "tgt", "n1000"): cell})
    assert ("A-r128", "tgt", "n1000") not in out, "no cell until every objective has 3 draws"
    assert ("tgt", "n1000", "full") in dm and ("tgt", "n1000", "correct_only") not in dm


def test_builder_main_runs_end_to_end():
    # The helper above stops before main(); this runs the whole build, report included, so a
    # defect in the writing stage (a NameError in write_report on 2026-09-23) cannot pass.
    from analysis import build_transfer_benchmark as btb

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        for seed, v in ((1, 0.70), (2, 0.71)):
            for cond, off in (("scratch", 0.0), ("fromjunyi", 0.01)):
                init = "scratch" if cond == "scratch" else "pretrained"
                enc = None if cond == "scratch" else "edubert_junyi_pretrain_full_encoder.pt"
                _write(tmp, f"kt_{cond}_{seed}_1{seed}.log",
                       _kt_log(f"edubert_assist2017_kt_assist_{cond}_n3000_seed{seed}", init,
                               v + off, seed=seed, encoder=enc, rid=f"id{cond}{seed}"))
        out = tmp / "bench"
        argv = sys.argv
        sys.argv = ["build_transfer_benchmark.py", "--logdir", str(tmp), "--outdir", str(out),
                    "--boots", "200"]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                btb.main()
        finally:
            sys.argv = argv
        report = (out / "report.md").read_text()
        assert "## Track A encoder draws" in report and "## RESULTS.md cross-check" in report
        assert (out / "gold_cells.tsv").exists() and (out / "missing_cells.tsv").exists()


def test_task2_hscore_identities():
    from src.estimators.hscore import hscore, ledoit_wolf

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    f = rng.normal(size=(2000, 12)) + 0.8 * y[:, None] * rng.normal(size=12)
    h, info = hscore(f, y)
    p1 = y.mean()
    p0 = 1 - p1
    d = f[y == 1].mean(0) - f[y == 0].mean(0)
    cov = np.cov(f.T, bias=True)
    assert abs(h - p0 * p1 * d @ np.linalg.pinv(cov) @ d) < 1e-9, "binary closed form"
    a = rng.normal(size=(12, 12)) + 3 * np.eye(12)
    assert abs(hscore(f @ a, y)[0] - h) < 1e-8, "invariant to invertible linear maps"
    assert hscore(rng.normal(size=(2000, 12)), y)[0] < 0.1 * h, "random features score low"
    shrunk_cov, s = ledoit_wolf(f - f.mean(0))
    assert 0.0 <= s <= 1.0 and np.linalg.eigvalsh(shrunk_cov).min() > 0
    assert hscore(f, y, shrink=True)[1]["shrinkage"] == s
    try:
        hscore(f, np.zeros(2000))
    except ValueError:
        pass
    else:
        raise AssertionError("one class must be refused")


def test_task2_nleep_properties():
    from src.estimators.nleep import gmm_diag, nleep, pca

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1500)
    f = rng.normal(size=(1500, 16)) + 1.5 * y[:, None] * rng.normal(size=16)
    p1 = y.mean()
    entropy = -(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1))
    assert abs(nleep(f, y, k=1)[0] + entropy) < 1e-12, "one component gives minus H(y)"
    s8 = nleep(f, y, k=8, dim=8, seed=3)[0]
    assert s8 <= 0 and s8 == nleep(f, y, k=8, dim=8, seed=3)[0]
    assert s8 > nleep(rng.normal(size=(1500, 16)), y, k=8, dim=8, seed=3)[0] + 0.05
    from src.estimators.nleep import leep_from_resp

    hard = np.eye(2)[y]
    assert abs(leep_from_resp(hard, y)) < 1e-12, "clusters equal to the labels give LEEP 0"
    ll = gmm_diag(pca(f, 8), 4, seed=2)["loglik"]
    assert all(b >= a - 1e-9 for a, b in zip(ll, ll[1:])), "EM never lowers the likelihood"


def test_fewshot_proxy_protocol():
    torch = _torch()
    import importlib.util as _u

    import src.data.dataset as dsmod
    from src.estimators.features import build_backbone, cap_positions, kt_features, sample_target

    spec = _u.spec_from_file_location("fewshot_proxy", REPO / "scripts" / "fewshot_proxy.py")
    fp = _u.module_from_spec(spec)
    spec.loader.exec_module(fp)
    a1, b1 = fp.draw_learners(40, 10, 12, 5)
    a2, b2 = fp.draw_learners(40, 10, 12, 5)
    assert not set(a1) & set(b1) and list(a1) == list(a2) and list(b1) == list(b2)
    try:
        fp.draw_learners(20, 10, 12, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("an oversized draw must be refused")
    with tempfile.TemporaryDirectory() as tmp:
        tgt = _make_processed(Path(tmp), "tgt", n_students=60)
        ck = Path(tmp) / "enc.pt"
        torch.save({"model_state": build_backbone(6, seed=0).state_dict()}, ck)
        splits_read = []
        real = dsmod.InteractionDataset.__init__

        def spy(self, d, split, *args, **kw):
            splits_read.append(split)
            return real(self, d, split, *args, **kw)

        fp.InteractionDataset.__init__ = spy
        try:
            score, info = fp.run_proxy(str(ck), str(tgt), seed=1, n_fit=16, n_eval=16, epochs=2)
        finally:
            fp.InteractionDataset.__init__ = real
        assert set(splits_read) == {"train"}, f"proxy read {splits_read}"
        assert 0.0 <= score <= 1.0 and len(info["eval_auc_by_epoch"]) == 2
        assert info["load"]["loaded"] == info["load"]["total"] > 0, "in-domain loads every tensor"
        spec2 = _u.spec_from_file_location("score_task2", REPO / "scripts" / "score_task2.py")
        st2 = _u.module_from_spec(spec2)
        spec2.loader.exec_module(st2)
        f, y, _, _ = st2.features_like_logme(None, tgt, seed=3, n_students=20,
                                             max_positions=200, device="cpu")
        sub, _, _ = sample_target(tgt, 20, 3)
        f2, y2, _ = kt_features(build_backbone(6, seed=3), sub, "cpu")
        sel = cap_positions(len(y2), 200, 3)
        assert np.allclose(f, f2[sel], atol=1e-6) and np.array_equal(y, y2[sel])


def test_builder_ragged_cell_uses_pairwise_seeds():
    from analysis.build_transfer_benchmark import cell_stats

    import random as _random

    vals = {"scratch": {1: 0.50, 2: 0.52, 3: 0.51, 4: 0.49},
            "src:a": {1: 0.60, 2: 0.61, 3: 0.62, 4: 0.59},
            "src:b": {1: 0.55, 2: 0.56}}
    rows = {r["candidate"]: r for r in cell_stats(vals, 2000, _random.Random(0))}
    assert rows["src:a"]["gain_k_of_n"] == "4/4", rows["src:a"]["gain_k_of_n"]
    assert rows["src:b"]["gain_k_of_n"] == "2/2", rows["src:b"]["gain_k_of_n"]


def test_builder_ladder_guard_and_results_md_check():
    from analysis import build_transfer_benchmark as btb

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        rs = tmp / "RESULTS.md"
        rs.write_text("Per-draw gains at 353,597, junyi: +0.0043 / +0.0061 / +0.0060.\n")
        cells = btb.results_md_cells(str(rs))
        assert cells[("S11", "junyi", "n353597_d42")] == 0.0043
        assert cells[("S11", "junyi", "n353597_d2")] == 0.0060
        cellvals = {("A", "assist2017", "n1000"): {"full": {1: 0.69, 2: 0.69}},
                    ("A", "assist2017", "trunc_K10"): {"full": {1: 0.60, 2: 0.60}},
                    ("B", "ednet", "n3000"): {"scratch": {1: 0.66}},
                    ("B", "ednet", "n20000"): {"scratch": {1: 0.68}}}
        ours = btb.cross_check_means(cellvals, {}, {})
        assert ours[("A", "assist2017", "full")][0] == 0.69, "truncation cell leaked in"
        assert ours[("B", "ednet", "scratch")][0] == 0.66, "full-scale cell leaked in"
        design = btb.expected_cells()
        keys = {(t_, g, b, c, s) for t_, g, b, c, s in design}
        assert len(keys) == len(design), "expected design has colliding keys"
        assert ("B", "assist2017", "n100", "src:ednet", 42) in keys
        assert ("B", "assist2017", "n3000", "src:ednet", 42) in keys


# --------------------------------------------------------------------------- runner
def main() -> int:
    global _RUNNER
    _RUNNER = True
    require_torch = "--require-torch" in sys.argv
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except _Skip as ex:
            if require_torch:
                failed += 1
                print(f"FAIL  {name}  (skipped under --require-torch: {ex})")
            else:
                skipped += 1
                print(f"SKIP  {name}  ({ex})")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
