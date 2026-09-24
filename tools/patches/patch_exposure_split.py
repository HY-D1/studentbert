from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: RESULTS.md 12.4 documents that every estimator prefers the target's own encoder,
# and 12.5 leaves one mechanism open: the scorers draw target learners from the train split, the
# in-domain encoder's pretraining data, while the gold is test AUC on unseen learners. This patch
# lets the scorers read the validation split (--split val) or a train draw of the same size
# (--match_split val), tags those scores so the evaluator keeps them apart (estimator@tag), adds
# the exposure queue to gen_tg1_jobs.sh, adds five tests and ignores the new outputs. Default
# behaviour is unchanged: the train draw, the same sample fingerprint and the same scores.
# Harry, 2026-09-24 hand-off item 2. Needs src/estimators/protocol.py and
# analysis/exposure_report.py in place first.

REQUIRED = ("src/estimators/protocol.py", "analysis/exposure_report.py")

FEATURES_OLD_SAMPLE = '''def sample_target(processed_dir: str | Path, n_students: int | None, seed: int,
                  max_seq_len: int = 512):
    """The fine-tune's learner draw (first_n_students in finetune_edubert.py) for this seed."""
    ds = InteractionDataset(str(processed_dir), "train", max_seq_len)
'''
FEATURES_NEW_SAMPLE = '''def split_size(processed_dir: str | Path, split: str, max_seq_len: int = 512) -> int:
    """Learners the scorer can draw from in this split, counted by the dataset class itself."""
    check_split(split)
    return len(InteractionDataset(str(processed_dir), split, max_seq_len))


def matched_n(processed_dir: str | Path, n_students: int | None, match_split: str,
              max_seq_len: int = 512) -> int:
    """Size of a train draw that matches a score on match_split: min(n_students, its learners).

    A validation split holds about a tenth of the learners, so comparing its score with the full
    train draw would confound exposure with sample size.
    """
    n = split_size(processed_dir, match_split, max_seq_len)
    return n if n_students is None else min(n_students, n)


def sample_target(processed_dir: str | Path, n_students: int | None, seed: int,
                  max_seq_len: int = 512, split: str = "train"):
    """The fine-tune's learner draw (first_n_students in finetune_edubert.py) for this seed.

    split="val" draws the same way from the validation learners, which no encoder saw during
    pretraining. The default is the draw every earlier score used, unchanged.
    """
    check_split(split)
    ds = InteractionDataset(str(processed_dir), split, max_seq_len)
'''

TRANSFER_OLD_CALL = r'''                rs = score_kt_logme(None if c == "scratch" else c, a.target_dir, seed=seed,
                                    n_students=a.n_students, max_positions=a.max_positions,
                                    min_group=a.min_group, device=a.device)
                for r in rs:
                    fh.write(json.dumps(r.to_json()) + "\n")
'''
TRANSFER_NEW_CALL = r'''                rs = score_kt_logme(None if c == "scratch" else c, a.target_dir, seed=seed,
                                    n_students=n_draw, max_positions=a.max_positions,
                                    min_group=a.min_group, device=a.device, split=split)
                for r in rs:
                    r.metadata.update(extra)
                    fh.write(json.dumps(r.to_json()) + "\n")
'''

ARGS_OLD = '''    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
'''
ARGS_NEW = '''    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    add_protocol_args(ap)
    a = ap.parse_args()
    split, n_draw, extra = resolve_protocol(a)
'''

EVAL_OLD = '''            n = r["metadata"].get("n_students_requested")
            budget = r["metadata"].get("target_budget") or (f"n{n}" if n else "full_split")
            out[(r["estimator"], r["target"], budget)][r["seed"]][cand] = (r["score"], r)
'''
EVAL_NEW = '''            md = r["metadata"]
            n = md.get("n_students_requested")
            budget = md.get("target_budget") or (f"n{n}" if n else "full_split")
            # A non-default sample protocol keeps its own name (estimator@tag), so a validation
            # score can never overwrite the train-draw score of the same estimator and seed.
            tag = md.get("score_tag") or (md["score_split"]
                                          if md.get("score_split", "train") != "train" else None)
            est = f"{r['estimator']}@{tag}" if tag else r["estimator"]
            out[(est, r["target"], budget)][r["seed"]][cand] = (r["score"], r)
'''

GEN_HEADER_OLD = "#   logmediag 3 jobs: scripts/diagnose_logme.py on the same draw (every layer, and in-domain\n"
GEN_HEADER_NEW = (
    "#   exposure 28 jobs: the pretraining-exposure test (RESULTS.md 12.5). The LogME and Task 2\n"
    "#            scorers on the 7 x 7 grid, re-run on the validation learners (--split val) and\n"
    "#            on a train draw of the same size (--match_split val). Output\n"
    "#            tg1_exposure_*.jsonl, read by analysis/exposure_report.py.\n"
    + GEN_HEADER_OLD)
GEN_HELP_OLD = ('  echo "set QUEUES to one or more of: scratch probe logme logme7 task2feat '
                'task2fewshot logmediag trackb7 objdraws"\n')
GEN_HELP_NEW = ('  echo "set QUEUES to one or more of: scratch probe logme logme7 task2feat '
                'task2fewshot exposure logmediag trackb7 objdraws"\n')
GEN_BLOCK_OLD = "for Q2 in task2feat task2fewshot; do\n"
GEN_BLOCK_NEW = '''if wants exposure; then
  : "${LOGME_GPU:?set LOGME_GPU (any is fine for scoring)}"
  GRES="$(gres_line "$LOGME_GPU")" || exit 1
  CKS=""
  for SRC in $DATASETS7; do
    CK="../checkpoints/edubert_${SRC}_pretrain_full_encoder.pt"
    if [ ! -f "$CK" ]; then
      echo "MISSING ENCODER: $CK"
      exit 1
    fi
    CKS="$CKS $CK"
  done
  Q="$CODE/queue_tg1_exposure"
  mkdir -p "$Q"
  rm -f "$Q"/*.sbatch
  for DS in $DATASETS7; do
    case "$DS" in
      ednet|junyi) WALL=03:00:00 ;;
      *) WALL=01:00:00 ;;
    esac
    for P in val trainmatch; do
      if [ "$P" = "val" ]; then
        PARGS="--split val --score_tag val"
      else
        PARGS="--match_split val --score_tag trainmatch"
      fi
      emit "$Q" "tg1_exposure_${P}_logme_${DS}" "tg1_exposure_${P}_logme_${DS}" "$GRES" "$WALL" 32G \\
        "PYTHONPATH=. $PY scripts/score_transferability.py --target_dir ../processed/$DS --candidates scratch$CKS --n_students 3000 --seeds $(seeds "42 1 2") $PARGS --out tg1_exposure_${P}_logme_kt_${DS}.jsonl"
      emit "$Q" "tg1_exposure_${P}_task2_${DS}" "tg1_exposure_${P}_task2_${DS}" "$GRES" "$WALL" 48G \\
        "PYTHONPATH=. $PY scripts/score_task2.py --target_dir ../processed/$DS --candidates scratch$CKS --n_students 3000 --seeds $(seeds "42 1 2") $PARGS --out tg1_exposure_${P}_task2_kt_${DS}.jsonl"
    done
  done
  echo "exposure queue: $Q"
fi

''' + GEN_BLOCK_OLD

TESTS_ANCHOR = "\n\n# --------------------------------------------------------------------------- runner\n"
TESTS_NEW = r'''

# --------------------------------------------------------------------------- exposure test
def test_exposure_protocol_arguments():
    import argparse

    from src.estimators.protocol import add_protocol_args, check_split, resolve_protocol

    def parse(*argv):
        ap = argparse.ArgumentParser()
        ap.add_argument("--target_dir", default="unused")
        ap.add_argument("--n_students", type=int, default=3000)
        add_protocol_args(ap)
        with contextlib.redirect_stderr(io.StringIO()):
            return ap.parse_args(list(argv))

    assert resolve_protocol(parse()) == ("train", 3000, {}), "the default protocol must not change"
    assert resolve_protocol(parse("--split", "val", "--score_tag", "val")) == \
        ("val", 3000, {"score_tag": "val", "target_budget": "n3000"})
    refused = [("--split", "val"), ("--score_tag", "val"), ("--split", "val", "--score_tag", "V-1"),
               ("--split", "val", "--match_split", "val", "--score_tag", "x"),
               ("--split", "test", "--score_tag", "x")]
    for argv in refused:
        try:
            resolve_protocol(parse(*argv))
        except SystemExit:
            continue
        raise AssertionError(f"protocol {argv} must be refused")
    try:
        check_split("test")
    except ValueError:
        return
    raise AssertionError("the test split holds the gold and must never be scorable")


def test_exposure_validation_draw_is_disjoint_and_test_is_refused():
    _torch()
    from src.estimators.features import matched_n, sample_target, split_size

    with tempfile.TemporaryDirectory() as tmp:
        tgt = _make_processed(Path(tmp), "tgt")
        ids = np.load(tgt / "sequences.npz")["student_ids"]
        splits = json.loads((tgt / "splits.json").read_text())
        _, rows_default, fp_default = sample_target(tgt, 12, 7)
        _, rows_train, fp_train = sample_target(tgt, 12, 7, split="train")
        assert rows_default == rows_train and fp_default == fp_train, \
            "the default draw must stay the train draw every earlier score used"
        _, all_train, _ = sample_target(tgt, None, 7)
        _, rows_val, fp_val = sample_target(tgt, 12, 7, split="val")
        assert sorted(int(ids[r]) for r in rows_val) == sorted(splits["val"])
        assert not set(rows_val) & set(all_train) and fp_val != fp_train
        assert split_size(tgt, "val") == len(splits["val"])
        assert matched_n(tgt, 3000, "val") == len(splits["val"])
        assert matched_n(tgt, 1, "val") == 1 and matched_n(tgt, None, "val") == len(splits["val"])
        for call in (lambda: sample_target(tgt, 12, 7, split="test"),
                     lambda: split_size(tgt, "test")):
            try:
                call()
            except ValueError:
                continue
            raise AssertionError("the test split holds the gold and must never be scorable")


def test_exposure_scores_carry_their_protocol():
    _torch()
    import importlib.util as _u

    from src.estimators.kt_logme import score_kt_logme

    with tempfile.TemporaryDirectory() as tmp:
        tgt = _make_processed(Path(tmp), "tgt")
        sp = json.loads((tgt / "splits.json").read_text())
        allids = sp["train"] + sp["val"] + sp["test"]
        (tgt / "splits.json").write_text(json.dumps({"train": allids[:14], "val": allids[14:20],
                                                     "test": allids[20:]}))
        kw = {"n_students": 12, "max_positions": None, "device": "cpu", "d_model": 16,
              "n_layers": 2, "max_seq_len": 64}
        a = score_kt_logme(None, tgt, seed=3, **kw)
        b = score_kt_logme(None, tgt, seed=3, split="train", **kw)
        v = score_kt_logme(None, tgt, seed=3, split="val", **kw)
        assert [x.score for x in a] == [x.score for x in b]
        assert {x.metadata["score_split"] for x in a} == {"train"}
        assert {x.metadata["score_split"] for x in v} == {"val"}
        assert v[0].metadata["n_students"] == 6 and v[0].metadata["n_students_requested"] == 12
        assert v[0].metadata["sample_fingerprint"] != a[0].metadata["sample_fingerprint"]
        spec = _u.spec_from_file_location("score_task2", REPO / "scripts" / "score_task2.py")
        t2 = _u.module_from_spec(spec)
        spec.loader.exec_module(t2)
        _, _, meta, _ = t2.features_like_logme(None, tgt, seed=3, n_students=12,
                                               max_positions=None, device="cpu", split="val")
        assert meta["score_split"] == "val" and meta["n_students"] == 6
        assert meta["sample_fingerprint"] == v[0].metadata["sample_fingerprint"]


def test_evaluator_keeps_sample_protocols_apart():
    from analysis import evaluate_estimators as ev

    lines = [("scratch", 0.1, {"n_students_requested": 3000}),
             ("scratch", 0.2, {"n_students_requested": 3000, "score_split": "val",
                               "score_tag": "val", "target_budget": "n3000"}),
             ("scratch", 0.3, {"n_students_requested": 170, "score_split": "train",
                               "score_tag": "trainmatch", "target_budget": "n3000"}),
             ("edubert_a_pretrain_full_encoder.pt", 0.4,
              {"n_students_requested": 3000, "score_split": "val"})]
    with tempfile.TemporaryDirectory() as tmp:
        j = Path(tmp) / "s.jsonl"
        j.write_text("".join(json.dumps({"estimator": "logme_kt_causal", "candidate": c,
                                         "target": "tgt", "seed": 1, "score": s,
                                         "metadata": md}) + "\n" for c, s, md in lines))
        got = ev.logme_scores([str(j)])
    base, val, tm = (("logme_kt_causal" + t, "tgt", "n3000") for t in ("", "@val", "@trainmatch"))
    assert set(got) == {base, val, tm}, sorted(got)
    assert got[base][1]["scratch"][0] == 0.1 and got[tm][1]["scratch"][0] == 0.3
    assert got[val][1]["scratch"][0] == 0.2 and got[val][1]["src:a"][0] == 0.4


def test_exposure_report_rule_and_completeness():
    from analysis import exposure_report as xr

    # In-domain margin per protocol (train, trainmatch, val), the same at all three seeds.
    # "e" ties on val: preferred means strictly top, so a zero margin is not a preference.
    # "f" is preferred on val at seed 42 only: 1 of 3 seeds is not a majority.
    # Scratch scores highest everywhere: the margin is among pretrained candidates only.
    plans = {"a": (0.2, 0.2, -0.1), "b": (0.2, 0.2, 0.2), "c": (0.2, -0.1, -0.1),
             "d": (-0.1, 0.2, 0.2), "e": (0.2, 0.2, 0.0), "f": (0.2, 0.2, (0.2, -0.1, -0.1))}
    recs = []
    for tgt, margins in plans.items():
        for prot, m in zip(xr.PROTOCOLS, margins):
            same = tgt == "b" and prot == "trainmatch"
            md = {"n_students_requested": 3000, "n_students": 100, "positions_used": 900,
                  "sample_fingerprint": tgt + ("train" if same else prot)}
            if prot != "train":
                md.update(score_tag=prot, target_budget="n3000")
            for i, seed in enumerate((42, 1, 2)):
                own = 1.0 if same else 1.0 + (m[i] if isinstance(m, tuple) else m)
                scores = {"scratch": 1.5, f"edubert_{tgt}_pretrain_full_encoder.pt": own,
                          "edubert_x_pretrain_full_encoder.pt": 1.0 if not same else 0.8,
                          "edubert_y_pretrain_full_encoder.pt": 0.5}
                recs += [{"estimator": "hscore_kt_causal", "candidate": c, "target": tgt,
                          "seed": seed, "score": s, "metadata": md} for c, s in scores.items()]
    with tempfile.TemporaryDirectory() as tmp:
        j = Path(tmp) / "s.jsonl"
        j.write_text("".join(json.dumps(r) + "\n" for r in recs))
        data = xr.load([str(j)], "n3000")
        assert not xr.check_complete(data, ["hscore_kt_causal"], 4)
        _, summary, determinism = xr.analyse(data, ["hscore_kt_causal"])
        got = {s["target"]: s["verdict"] for s in summary}
        assert got == {"a": "exposure", "b": "representation", "c": "sample size",
                       "d": "no preference", "e": "exposure", "f": "exposure"}, got
        assert xr.protocol_of({"score_split": "val"}) == "val" and xr.protocol_of({}) == "train"
        assert len(determinism) == 3 and all(" b " in d for d in determinism), determinism
        assert all("difference 2.000e-01" in d for d in determinism), \
            "a changed score on an identical sample must show in the determinism check"
        argv = sys.argv
        sys.argv = ["exposure_report.py", "--scores", str(j), "--n-candidates", "4",
                    "--estimators", "hscore_kt_causal", "--out-prefix", str(Path(tmp) / "x" / "e")]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                xr.main()
        finally:
            sys.argv = argv
        report = (Path(tmp) / "x" / "e_report.md").read_text()
        assert "| hscore_kt_causal | a |" in report and "exposure |" in report
        drop = [r for r in recs if not (r["target"] == "a" and r["seed"] == 2
                                        and r["metadata"].get("score_tag") == "val")]
        j.write_text("".join(json.dumps(r) + "\n" for r in drop))
        bad = xr.check_complete(xr.load([str(j)], "n3000"), ["hscore_kt_causal"], 4)
        assert bad == ["hscore_kt_causal a val: seeds [1, 42], expected [1, 2, 42]"], bad
        short = [r for r in recs if not (r["target"] == "b" and r["seed"] == 1 and r["candidate"]
                                         == "scratch" and r["metadata"].get("score_tag") == "val")]
        j.write_text("".join(json.dumps(r) + "\n" for r in short))
        bad = xr.check_complete(xr.load([str(j)], "n3000"), ["hscore_kt_causal"], 4)
        assert bad == ["hscore_kt_causal b val seed 1: 3 of 4 candidates"], bad
        clash = dict(recs[0], score=recs[0]["score"] + 1.0)
        j.write_text("".join(json.dumps(r) + "\n" for r in recs + [recs[0], clash]))
        try:
            xr.load([str(j)], "n3000")
        except SystemExit as err:
            assert "disagreeing duplicate" in str(err)
        else:
            raise AssertionError("a disagreeing duplicate must stop the report")
''' + TESTS_ANCHOR

EDITS = [
    ("src/estimators/features.py",
     "from src.data.dataset import InteractionDataset, collate_fn\n"
     "from src.models.edubert import EduBERT\n",
     "from src.data.dataset import InteractionDataset, collate_fn\n"
     "from src.estimators.protocol import check_split\n"
     "from src.models.edubert import EduBERT\n",
     "features: import check_split"),
    ("src/estimators/features.py", FEATURES_OLD_SAMPLE, FEATURES_NEW_SAMPLE,
     "features: split, split_size, matched_n"),
    ("src/estimators/kt_logme.py",
     "                   batch_size: int = 64) -> list[EstimatorResult]:\n",
     '                   batch_size: int = 64, split: str = "train") -> list[EstimatorResult]:\n',
     "kt_logme: split argument"),
    ("src/estimators/kt_logme.py",
     "    subset, rows, fingerprint = sample_target(target_dir, n_students, seed, max_seq_len)\n",
     "    subset, rows, fingerprint = sample_target(target_dir, n_students, seed, max_seq_len,\n"
     "                                              split=split)\n",
     "kt_logme: pass split to the draw"),
    ("src/estimators/kt_logme.py",
     '              "candidate_path": candidate or "scratch"}\n',
     '              "candidate_path": candidate or "scratch", "score_split": split}\n',
     "kt_logme: record score_split"),
    ("scripts/score_task2.py",
     "from src.estimators.nleep import nleep\n",
     "from src.estimators.nleep import nleep\n"
     "from src.estimators.protocol import add_protocol_args, resolve_protocol\n",
     "score_task2: import protocol"),
    ("scripts/score_task2.py",
     "def features_like_logme(candidate, target_dir, *, seed, n_students, max_positions, device):\n",
     "def features_like_logme(candidate, target_dir, *, seed, n_students, max_positions, device,\n"
     '                        split="train"):\n',
     "score_task2: split argument"),
    ("scripts/score_task2.py",
     "    subset, rows, fingerprint = sample_target(target_dir, n_students, seed)\n",
     "    subset, rows, fingerprint = sample_target(target_dir, n_students, seed, split=split)\n",
     "score_task2: pass split to the draw"),
    ("scripts/score_task2.py",
     '            "candidate_path": candidate or "scratch"}\n',
     '            "candidate_path": candidate or "scratch", "score_split": split}\n',
     "score_task2: record score_split"),
    ("scripts/score_task2.py", ARGS_OLD, ARGS_NEW, "score_task2: protocol arguments"),
    ("scripts/score_task2.py",
     "                    ck, a.target_dir, seed=seed, n_students=a.n_students,\n"
     "                    max_positions=a.max_positions, device=device)\n",
     "                    ck, a.target_dir, seed=seed, n_students=n_draw,\n"
     "                    max_positions=a.max_positions, device=device, split=split)\n",
     "score_task2: draw size and split"),
    ("scripts/score_task2.py",
     '                        metadata={**meta, "feature_dim": int(feats.shape[1])})\n',
     '                        metadata={**meta, **extra, "feature_dim": int(feats.shape[1])})\n',
     "score_task2: record protocol metadata"),
    ("scripts/score_transferability.py",
     "from src.estimators.kt_logme import ranks_by_seed, score_kt_logme\n",
     "from src.estimators.kt_logme import ranks_by_seed, score_kt_logme\n"
     "from src.estimators.protocol import add_protocol_args, resolve_protocol\n",
     "score_transferability: import protocol"),
    ("scripts/score_transferability.py", ARGS_OLD, ARGS_NEW,
     "score_transferability: protocol arguments"),
    ("scripts/score_transferability.py", TRANSFER_OLD_CALL, TRANSFER_NEW_CALL,
     "score_transferability: draw size, split, protocol metadata"),
    ("analysis/evaluate_estimators.py",
     "  LogME and any other scorer   JSON lines from scripts/score_transferability.py (--scores)\n",
     "  LogME and any other scorer   JSON lines from scripts/score_transferability.py (--scores);\n"
     "                               a --score_tag protocol is reported as estimator@tag\n",
     "evaluator: docstring"),
    ("analysis/evaluate_estimators.py", EVAL_OLD, EVAL_NEW, "evaluator: estimator@tag"),
    ("slurm/generators/gen_tg1_jobs.sh", GEN_HEADER_OLD, GEN_HEADER_NEW, "generator: header"),
    ("slurm/generators/gen_tg1_jobs.sh", GEN_HELP_OLD, GEN_HELP_NEW, "generator: help line"),
    ("slurm/generators/gen_tg1_jobs.sh", GEN_BLOCK_OLD, GEN_BLOCK_NEW,
     "generator: exposure queue"),
    ("tests/test_estimators.py", TESTS_ANCHOR, TESTS_NEW, "tests: exposure tests"),
    (".gitignore", "tg1_fewshot_kt_*.jsonl\n",
     "tg1_fewshot_kt_*.jsonl\ntg1_exposure_*.jsonl\nexposure_*/\n",
     "gitignore: exposure scores and report folder"),
]


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
    missing = [p for p in REQUIRED if not Path(p).is_file()]
    if missing:
        sys.exit(f"ABORT: copy these into the repo first: {missing}; nothing written")
    files = sorted({f for f, *_ in EDITS})
    src = {f: Path(f).read_text(encoding="utf-8") for f in files}
    out = dict(src)
    for f, old, new, why in EDITS:
        out[f] = apply(out[f], old, new, why)
    for f in files:
        if out[f] != src[f]:
            Path(f).write_text(out[f], encoding="utf-8")
    for f in files + list(REQUIRED):
        print("md5", hashlib.md5(Path(f).read_bytes()).hexdigest(), f)


if __name__ == "__main__":
    main()
