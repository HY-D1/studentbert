from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: the Task 4 cross-domain check ran 4 of the 6 classification tasks of Bassignana
# et al. (2022). On 2026-09-30 Airline became reproducible from the authors' own converter
# (mainlp/logme-nlp, commit 0046c725, sentiment/convert.py -rs 4012, run in a separate clone
# because it is GPL-3.0), and SciERC was excluded (its entity-marked splits were never released,
# and under the Task 4 tie rule SciBERT is top-equivalent after fine-tuning anyway). This adds
# Airline as a local task to the extractor, lets the generator build jobs for chosen tasks only,
# lets the evaluator reproduce the earlier 4-task evaluation exactly (--tasks), and lets the
# ceiling report an extra section (--also_run) while its default report stays byte-identical.
# Every hub-task code path is unchanged. Each file is checked against its pre-patch md5, every
# anchor must match exactly once, and nothing is written unless every edit passes. Harry,
# 2026-09-30.

PRE = {
    "scripts/task4_extract.py": "f6685d8e61157b1be71005921bc81e1f",
    "slurm/generators/gen_task4_jobs.sh": "e4316008887ff4a89aee46670eed4bf8",
    "analysis/task4_evaluate.py": "3da461de61e1553a7164c45188d7601b",
    "analysis/task4_ceiling.py": "f345e82b03304bd183dbf2aef563232f",
    "tests/test_task4.py": "a60d602de78c69255090e36fb0fa4532",
}

READ_LOCAL = '''def read_local(path: str, md5: str, col: str,
               allowed: tuple[int, ...]) -> tuple[list[str], np.ndarray]:
    """Texts and integer labels of a local CSV split; aborts on a different or malformed file."""
    p = Path(path)
    if not p.is_file():
        sys.exit(f"ABORT: {p} is missing")
    raw = p.read_bytes()
    got = hashlib.md5(raw).hexdigest()
    if got != md5:
        sys.exit(f"ABORT: {p} md5 {got}, expected {md5}")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    texts = [r[col] for r in rows]
    labels = np.asarray([int(r["label"]) for r in rows])
    if not rows or any(not t for t in texts) or not set(labels.tolist()) <= set(allowed):
        sys.exit(f"ABORT: {p} has no rows, an empty text or a label outside {allowed}")
    return texts, labels


'''

SELECT_TASKS = '''def select_tasks(scores: dict, tasks: list[str] | None) -> dict:
    """Only the named tasks, so a later run can reproduce an earlier, smaller evaluation exactly."""
    if not tasks:
        return scores
    missing = set(tasks) - {t for _e, t, _p in scores}
    if missing:
        sys.exit(f"ABORT: no scores for task(s) {sorted(missing)}")
    return {k: v for k, v in scores.items() if k[1] in tasks}


'''

NEW_TESTS = '''def test_local_split_reads_quoted_text_and_checks_md5_and_labels():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "x-train.csv"
        with open(p, "w", encoding="utf8", newline="") as fh:
            w = csv.writer(fh, quoting=csv.QUOTE_ALL)
            w.writerow(["text", "label"])
            w.writerows([['a, "quoted"\\nline', "0"], ["@united thanks", "2"], ["ok", "1"]])
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        texts, labels = ex.read_local(str(p), md5, "text", (0, 1, 2))
        assert texts == ['a, "quoted"\\nline', "@united thanks", "ok"], texts
        assert labels.tolist() == [0, 2, 1]
        for bad_md5, allowed in (("0" * 32, (0, 1, 2)), (md5, (0, 1))):
            try:
                ex.read_local(str(p), bad_md5, "text", allowed)
            except SystemExit as exc:
                assert "ABORT" in str(exc)
            else:
                raise AssertionError("a wrong md5 or a label outside the set did not abort")


def test_airline_is_a_local_task_with_a_pinned_split_and_scierc_is_absent():
    assert ex.TASKS["airline"][0] == "local" and set(ex.LOCAL) == {"airline"}
    path, md5, col, allowed = ex.LOCAL["airline"]
    assert path.endswith("/airline-train.csv") and len(md5) == 32 and allowed == (0, 1, 2)
    assert col == ex.TASKS["airline"][2][0] and "scierc" not in ex.TASKS


def test_task_filter_keeps_only_named_tasks_and_aborts_on_unscored():
    s = scores_from([1, 2, 3, 4, 5, 6, 7])
    s[("hscore", "airline", "cls")] = s[("hscore", "rte", "cls")]
    assert {k[1] for k in ev.select_tasks(s, ["rte"])} == {"rte"}
    assert ev.select_tasks(s, None) is s and ev.select_tasks(s, []) is s
    try:
        ev.select_tasks(s, ["qnli"])
    except SystemExit as exc:
        assert "qnli" in str(exc)
    else:
        raise AssertionError("an unscored task did not abort")


def test_ceiling_extra_section_leaves_the_default_report_unchanged():
    ce = _load("t4_ceiling", "analysis/task4_ceiling.py")
    with tempfile.TemporaryDirectory() as t:
        with contextlib.redirect_stdout(io.StringIO()):
            ce.main(["--out", f"{t}/a"])
            ce.main(["--out", f"{t}/b", "--also_run", "airline"])
        a = Path(f"{t}/a_report.md").read_text()
        b = Path(f"{t}/b_report.md").read_text()
        head, sep, tail = b.partition("## Tasks run in Task 4 plus airline")
        assert sep and "plus airline" not in a
        rest = tail.split("## All six published tasks", 1)[1]
        assert head + "## All six published tasks" + rest == a
        for bad in (["rte"], ["nosuchtask"]):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ce.main(["--out", f"{t}/c", "--also_run", *bad])
            except SystemExit as exc:
                assert "ABORT" in str(exc)
            else:
                raise AssertionError(f"--also_run {bad} did not abort")


'''

EDITS = {
    "scripts/task4_extract.py": [
        ("import argparse\nimport json\nimport sys\nimport time\n",
         "import argparse\nimport csv\nimport hashlib\nimport io\nimport json\nimport sys\nimport time\n",
         "extract: imports"),
        ("# task -> (hub id, config, (first text column, second text column or None))\n",
         "# task -> (hub id, config, (first text column, second text column or None)); hub \"local\"\n"
         "# reads the split pinned in LOCAL instead of the Hugging Face hub\n",
         "extract: TASKS comment"),
        ('    "rte": ("nyu-mll/glue", "rte", ("sentence1", "sentence2")),\n}\n',
         '    "rte": ("nyu-mll/glue", "rte", ("sentence1", "sentence2")),\n'
         '    "airline": ("local", None, ("text", None)),\n}\n'
         "# Local tasks read a train split made outside this repo: task -> (file, md5, text column,\n"
         "# labels). airline: the authors' own converter (mainlp/logme-nlp, commit 0046c725,\n"
         "# sentiment/convert.py -rs 4012) on Kaggle's Tweets.csv (md5\n"
         "# 2fa808ea99b32814ccd64d7097d935f2), run in a separate clone because that code is GPL-3.0;\n"
         "# only its output is read here. SciERC is excluded: its entity-marked splits were never\n"
         "# released (mrap_task4_crossdomain.md).\n"
         "LOCAL = {\n"
         '    "airline": ("/projects/algl/dai.hany/task4/data/airline/airline-train.csv",\n'
         '                "b0fa4865e8d442b5e2146715e7a84c10", "text", (0, 1, 2)),\n'
         "}\n",
         "extract: airline task and LOCAL"),
        ("def slug(model: str) -> str:\n", READ_LOCAL + "def slug(model: str) -> str:\n",
         "extract: read_local"),
        ("    from datasets import load_dataset\n", "    from datasets import Dataset, load_dataset\n",
         "extract: Dataset import"),
        ('    hub, cfg, (col_a, col_b) = TASKS[a.task]\n'
         '    ds = load_dataset(hub, cfg, split="train")\n'
         '    labels = np.asarray(ds["label"])\n',
         '    hub, cfg, (col_a, col_b) = TASKS[a.task]\n'
         '    source = {"hub": hub, "config": cfg}\n'
         '    if hub == "local":\n'
         '        path, md5, col, allowed = LOCAL[a.task]\n'
         '        texts, labels = read_local(path, md5, col, allowed)\n'
         '        ds = Dataset.from_dict({col: texts, "label": labels.tolist()})\n'
         '        source.update(data_file=path, data_md5=md5)\n'
         '    else:\n'
         '        ds = load_dataset(hub, cfg, split="train")\n'
         '        labels = np.asarray(ds["label"])\n',
         "extract: local loading branch"),
        ('            "torch": torch.__version__, "hub": hub, "config": cfg}\n',
         '            "torch": torch.__version__, **source}\n',
         "extract: meta keeps hub and config, adds the data file for local tasks"),
    ],
    "slurm/generators/gen_task4_jobs.sh": [
        ("#   bash slurm/generators/gen_task4_jobs.sh\n#   ls queue_task4_extract/ queue_task4_score/\n",
         "#   bash slurm/generators/gen_task4_jobs.sh\n#   ls queue_task4_extract/ queue_task4_score/\n"
         "# The Airline extension (2026-09-30; SciERC excluded) builds its jobs with\n"
         "#   TASKS=airline QX=queue_task4_airline_x QS=queue_task4_airline_s bash \\\n"
         "#       slurm/generators/gen_task4_jobs.sh\n",
         "generator: usage"),
        ('TASKS="agnews mnli qnli rte"\n', 'TASKS=${TASKS:-"agnews mnli qnli rte"}\n',
         "generator: TASKS overridable"),
    ],
    "analysis/task4_evaluate.py": [
        ("#       --out /projects/algl/dai.hany/task4/task4_eval\n",
         "#       --out /projects/algl/dai.hany/task4/task4_eval\n"
         "# With --tasks agnews mnli qnli rte it reproduces the 2026-09-28 evaluation exactly after\n"
         "# Airline is scored into the same file.\n",
         "evaluate: usage"),
        ("def evaluate(scores: dict, gold: dict) -> tuple[list[dict], list[dict]]:\n",
         SELECT_TASKS + "def evaluate(scores: dict, gold: dict) -> tuple[list[dict], list[dict]]:\n",
         "evaluate: select_tasks"),
        ('    ap.add_argument("--out", required=True, help="prefix for _cells.tsv, _summary.tsv, _report.md")\n',
         '    ap.add_argument("--out", required=True, help="prefix for _cells.tsv, _summary.tsv, _report.md")\n'
         '    ap.add_argument("--tasks", nargs="*", help="evaluate only these tasks (default: all scored)")\n',
         "evaluate: --tasks"),
        ("    rows, repro = evaluate(load_scores(a.scores), load_gold(Path(a.gold)))\n",
         "    rows, repro = evaluate(select_tasks(load_scores(a.scores), a.tasks), load_gold(Path(a.gold)))\n",
         "evaluate: filter before judging"),
    ],
    "analysis/task4_ceiling.py": [
        ('#   PYTHONPATH=. python analysis/task4_ceiling.py --out /projects/algl/dai.hany/task4/task4_ceiling\n',
         '#   PYTHONPATH=. python analysis/task4_ceiling.py --out /projects/algl/dai.hany/task4/task4_ceiling\n'
         "# --also_run airline adds a section for the tasks run so far plus Airline (2026-09-30); without\n"
         "# it the report is byte-identical to the 2026-09-29 one, so review 7.5's targets stay checkable.\n",
         "ceiling: usage"),
        ('    ap.add_argument("--out", required=True, help="prefix for _cells.tsv and _report.md")\n',
         '    ap.add_argument("--out", required=True, help="prefix for _cells.tsv and _report.md")\n'
         '    ap.add_argument("--also_run", nargs="*", default=[], help="later tasks, extra section")\n',
         "ceiling: --also_run"),
        ("    rows = cells(load_gold(Path(a.gold)))\n",
         "    gold = load_gold(Path(a.gold))\n"
         "    extra = tuple(a.also_run)\n"
         "    if set(extra) - {t for t, _p in gold} or set(extra) & set(RUN_TASKS):\n"
         '        sys.exit(f"ABORT: --also_run {list(extra)} must name published tasks not in {RUN_TASKS}")\n'
         "    rows = cells(gold)\n",
         "ceiling: validate --also_run"),
        ('    for title, tasks in (("Tasks run in Task 4 (AGNews, MNLI, QNLI, RTE)", RUN_TASKS),\n'
         '                         ("All six published tasks", None)):\n',
         '    sections = [("Tasks run in Task 4 (AGNews, MNLI, QNLI, RTE)", RUN_TASKS)]\n'
         "    if extra:\n"
         '        sections.append((f"Tasks run in Task 4 plus {\', \'.join(extra)}", RUN_TASKS + extra))\n'
         '    sections.append(("All six published tasks", None))\n'
         "    for title, tasks in sections:\n",
         "ceiling: extra section"),
    ],
    "tests/test_task4.py": [
        ("import contextlib\nimport importlib.util\n",
         "import contextlib\nimport csv\nimport hashlib\nimport importlib.util\n",
         "tests: imports"),
        ("def main() -> int:\n", NEW_TESTS + "def main() -> int:\n", "tests: Airline, filter, ceiling"),
    ],
}


def apply(text: str, old: str, new: str, why: str) -> tuple[str, bool]:
    if new in text and (old not in text or old in new):
        print(f"skip (already applied): {why}")
        return text, False
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT ({why}): anchor matched {n} times, expected 1; nothing written")
    print(f"ok: {why}")
    return text.replace(old, new, 1), True


def main() -> None:
    out = {}
    for rel, edits in EDITS.items():
        p = Path(rel)
        if not p.is_file():
            sys.exit(f"ABORT: {rel} is missing (run from the code root); nothing written")
        src = p.read_text(encoding="utf-8")
        text, changed = src, []
        for old, new, why in edits:
            text, did = apply(text, old, new, why)
            changed.append(did)
        if any(changed) and not all(changed):
            sys.exit(f"ABORT: {rel} is partly patched; nothing written")
        if all(changed) and hashlib.md5(src.encode()).hexdigest() != PRE[rel]:
            sys.exit(f"ABORT: {rel} differs from the pre-patch file {PRE[rel]}; nothing written")
        out[rel] = (src, text)
    for rel, (src, text) in out.items():
        if text != src:
            Path(rel).write_text(text, encoding="utf-8")
        print("md5", hashlib.md5(Path(rel).read_bytes()).hexdigest(), rel)


if __name__ == "__main__":
    main()
