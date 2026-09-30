from __future__ import annotations

"""Record the 2026-09-30 log checks in RESULTS.md.

Adds, with provenance: the SAINT+ per-seed table and settings (new 1.1), the SAINT+
response-time caveat and the per-column epoch budget (section 1), the truncation provenance
(section 5), the objective-conditioned probe on assist2017 and junyi (new 6.2), and the Junyi
release (section 0). Numbers are computed here from the values the checks returned, rounded
once, half up; nothing is typed as a rounded result. The two new headings are added to the
collector's MANUAL list so it keeps refusing to overwrite RESULTS.md.

Run from the code directory:  python tools/patches/patch_results_20260930.py
All-or-nothing: every anchor must match exactly once or nothing is written. A second run
reports "already applied" and leaves both files byte-identical.
"""

import argparse
import ast
import hashlib
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SEEDS = [42, 1, 2, 3, 4, 5]
# W&B test/auc, full precision, matched by run id (analysis/saint_base2_table.py, exit 0).
SAINT = {
    "assist2017": [0.6285888872766102, 0.6289931766601495, 0.6275449347115293,
                   0.6321792246593663, 0.6273729835074804, 0.629321074367951],
    "ednet": [0.6863394225486057, 0.6866870332849244, 0.685048659316821,
              0.6857325651310046, 0.6863097715490013, 0.6858878143796964],
    "junyi": [0.7984680758485486, 0.7982597285906815, 0.7994167448402392,
              0.7983373020525548, 0.7983289907435711, 0.7979041828315142],
    "algebra2005": [0.8420082638725566, 0.8369170854878044, 0.8390814987660288,
                    0.8269529829883701, 0.7800306673374817, 0.8340236782924368],
    "bridge2006": [0.8520941501339202, 0.8552399756850023, 0.8530819724278474,
                   0.8544441412871662, 0.8546588037356977, 0.850983665666481],
    "assist2009": [0.836340371634008, 0.8422930763687081, 0.841309013176771,
                   0.8415931854292358, 0.8408785042893335, 0.8410538602857177],
    "algebra2006": [0.876182991571212, 0.8740892316300455, 0.8759978197046578,
                    0.8748160116639301, 0.8740707163616566, 0.8734226196243835],
}
# Section 1 row labels, used to check each computed mean against the SAINT+ column there.
SEC1_LABEL = {"assist2017": "ASSIST2017", "ednet": "EdNet", "junyi": "Junyi",
              "algebra2005": "Algebra2005", "bridge2006": "Bridge2006",
              "assist2009": "ASSIST2009", "algebra2006": "Algebra2006"}
# Probe accuracy, 4 dp log banners (seeds 42, 1, 2); both campaigns returned identical values.
PROBE = {("assist2017", "skill_only"): [0.1451, 0.1464, 0.1444],
         ("assist2017", "correct_only"): [0.1114, 0.1118, 0.1133],
         ("junyi", "skill_only"): [0.0209, 0.0210, 0.0204],
         ("junyi", "correct_only"): [0.0137, 0.0139, 0.0136]}
JUNYI_STATED_ROWS = 16217311  # "Log_Problem.csv recorded 16,217,311 problem attempts" (Kaggle page)


def D(x) -> Decimal:
    return Decimal(repr(x))


def q(x: Decimal, places: str) -> str:
    return str(x.quantize(Decimal(places), rounding=ROUND_HALF_UP))


def mean_sd(vals) -> tuple[Decimal, Decimal]:
    d = [D(v) for v in vals]
    m = sum(d) / len(d)
    return m, (sum((x - m) ** 2 for x in d) / len(d)).sqrt()


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def count_rows(path: Path) -> tuple[int, str]:
    n, last = 0, b"\n"
    with path.open("rb") as fh:
        header = fh.readline()
        for buf in iter(lambda: fh.read(1 << 20), b""):
            n += buf.count(b"\n")
            last = buf[-1:]
    if last != b"\n":
        n += 1
    return n, header.decode(errors="replace").strip()


def saint_block() -> str:
    rows = []
    for ds, vals in SAINT.items():
        m, sd = mean_sd(vals)
        rows.append(f"| {ds} | " + " | ".join(q(D(v), "0.0001") for v in vals)
                    + f" | {q(m, '0.0001')} ±{q(sd, '0.0001')} |")
    a05 = SAINT["algebra2005"]
    m5 = sum(D(v) for i, v in enumerate(a05) if i != 4) / 5
    others = [D(v) for i, v in enumerate(a05) if i != 4]
    return (
        "\n\n### 1.1 SAINT+ per-seed KT test AUC, base2 campaign (logs + W&B full precision)\n\n"
        "Runs `base2_saint_<ds>_seed{42,1,2,3,4,5}`: 42 of 42 completed, one completed log per "
        "seed (43 logs found; the extra one is an attempt that ended before logging). Every run "
        "trained 30 epochs (30 epoch lines in each log; W&B config epochs=30, dropout 0.1, "
        "batch_size 64). Values are W&B `test/auc` matched by run id, never by name, rounded once "
        "to 4 dp; each agrees with its log banner within 5e-5. Mean and population SD from the "
        "unrounded values. Produced by `analysis/saint_base2_table.py` (exit 0, 2026-09-30).\n\n"
        "| Dataset | s42 | s1 | s2 | s3 | s4 | s5 | mean ±pstdev |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "Model and training (`scripts/train_baseline.py` line 206, `src/models/saint_plus.py`): "
        "encoder-decoder transformer, 2 encoder and 2 decoder layers, d_model 256, 8 heads, "
        "d_ff 1,024, dropout 0.1, max_len 512; Adam lr 1e-3, batch 64, linear warm-up over the "
        "first 10% of steps then linear decay (`--warmup_frac` default 0.1, not overridden by "
        "`slurm/generators/gen_baselines_uniform.sh`, so it applies to DKT, AKT and SAINT+ "
        "alike); best checkpoint by validation AUC.\n\n"
        "_Read: every mean reproduces the SAINT+ column of section 1. Algebra2005 is the one "
        f"high-variance cell (pstdev {q(mean_sd(a05)[1], '0.0001')}): seed 4 at "
        f"{q(D(a05[4]), '0.0001')} against {q(min(others), '0.0001')} to "
        f"{q(max(others), '0.0001')} for the other five; excluding it gives "
        f"{q(m5, '0.0001')}. Either value is above DKT on that row (0.7985). "
        "`slurm/run_saint_100.sbatch` (100 epochs, assist2017 only, run_type baseline) is a "
        "superseded campaign and feeds no cell._"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="RESULTS.md")
    ap.add_argument("--collector", default="analysis/collect_all_results.py")
    ap.add_argument("--raw_junyi", default="../raw/Log_Problem.csv")
    ap.add_argument("--expect_rows", type=int, default=JUNYI_STATED_ROWS)
    a = ap.parse_args()

    res, col = Path(a.results), Path(a.collector)
    text, ctext = res.read_text(), col.read_text()
    md5_res, md5_col = md5(res), md5(col)
    print(f"before: {res} {md5_res}   {col} {md5_col}")

    # Guard: the computed SAINT+ means must equal the section 1 column before anything is written.
    sec1 = text.split("## 2. Source comparison")[0]
    for ds, vals in SAINT.items():
        want = q(mean_sd(vals)[0], "0.0001")
        line = next((ln for ln in sec1.splitlines() if ln.startswith(f"| {SEC1_LABEL[ds]} |")), None)
        cell = line.split("|")[4].strip().strip("*") if line else None
        if cell != want:
            sys.exit(f"ABORT: {ds} SAINT+ mean {want} != section 1 cell {cell!r}; nothing written")

    probe_rows = []
    for (ds, obj), vals in PROBE.items():
        m, sd = mean_sd(vals)
        probe_rows.append(f"| {ds} | {obj} | " + " | ".join(f"{v:.4f}" for v in vals)
                          + f" | {q(m, '0.0001')} ±{q(sd, '0.0001')} |")

    edits = []  # (name, anchor, insert_after_anchor, marker proving it is already applied)
    edits.append(("section 1 caveat, epochs and 1.1",
        "correct[t-1] at position t, with causal masks on encoder, decoder target and memory, "
        "so no position sees its own answer.",
        "\n\n**SAINT+ response-time caveat (read from `src/models/saint_plus.py`, 2026-09-30):** "
        "the leakage check above covers the answer only. The response-time bin is added to the "
        "encoder input unshifted (line 74) and the causal mask admits the diagonal, so the timing "
        "of the response being predicted is in scope for its own prediction. DKT and AKT do not "
        "read the time bin, and EduBERT predicts step t+1 from position t, so none of them sees "
        "the target step's timing. The size of the effect on the SAINT+ cells has not been "
        "measured.\n\n"
        "**Epoch budget per column (log check 2026-09-30, 6 completed logs per condition):** "
        "DKT, AKT and SAINT+ `base2_<model>_<ds>` 30 epochs on all 7. scratch and EduBERT-pt: "
        "assist2017 `kt_assist_{scratch,indomain}_n3000` 20 epochs at 1,366 train students (the "
        "3,000 request exceeds the split); ednet and junyi `w7_ktfull_<ds>_{scratch,indomain}` "
        "20 epochs; the 4 newer datasets `w8_scratch_<ds>` and "
        "`w8_{algabl,bridgeabl,a09abl,alg06abl}_full` 30 epochs. `w7_ktfull_ednet_scratch` seed 42 "
        "also has an empty log (job 8176342, 0 epochs) superseded by job 8176348. Superseded "
        "campaigns that feed no cell: `w6_akt_baseline` (AKT, assist2017, 100 epochs, seeds 42, 1 "
        "and 2, three runs in one log) and `dktclean` (DKT, assist2017, 20 epochs, one run, no "
        "--seed)." + saint_block(),
        "### 1.1 SAINT+ per-seed KT test AUC"))
    edits.append(("section 5 truncation provenance",
        "| 512 | 0.6941 | 0.6920 | 0.6655 | 0.6675 |",
        "\n\n**Truncation provenance (log check 2026-09-30):** `w8_trunc_{full,skill_only,"
        "correct_only,scratch}_k{10,20,40,80,160,320,512}`, 28 cells, 6 logs each. The pretrained "
        "cells load `edubert_ednet_pretrain_full_encoder.pt`, "
        "`edubert_ednet_pretrain_ednet_skill_only_encoder.pt` and "
        "`edubert_ednet_pretrain_ednet_correct_only_encoder.pt` (EdNet source); every run trained "
        "30 epochs on all 1,366 assist2017 train students (no --n_students).",
        "**Truncation provenance (log check 2026-09-30):**"))
    edits.append(("new 6.2 objective-conditioned probe",
        "\n## 7. Embedding geometry (honest negative)",
        None,
        "### 6.2 Objective-conditioned probe"))
    block62 = (
        "### 6.2 Objective-conditioned probe, assist2017 and junyi (EdNet-source skill_only and "
        "correct_only encoders; runs `probe2_<ds>_obj_<obj>_seed*` and `tg_probe7_<ds>_<obj>_s*`)\n\n"
        "Every run loaded 79/82 tensors from `edubert_ednet_pretrain_ednet_{skill_only,"
        "correct_only}_encoder.pt`, so these are cross-dataset EdNet-source encoders, not "
        "in-domain ones (log check 2026-09-30). The two campaigns return identical 4 dp values. "
        "Values are 4 dp log banners; mean and population SD computed from them.\n\n"
        "| Target | objective | s42 | s1 | s2 | mean ±pstdev |\n|---|---|---|---|---|---|\n"
        + "\n".join(probe_rows) + "\n\n"
        "Comparators from section 6, same source: EdNet full-objective encoder 0.1416 ±0.0011 "
        "(assist2017) and 0.0209 ±0.0002 (junyi), re-read at full precision from "
        "`wandb_by_id.jsonl` on 2026-09-30 as 0.141640 ±0.001067 and 0.020918 ±0.000214; scratch "
        "0.1366 ±0.0011 and 0.0165 ±0.0004.\n\n"
        "_Read: skill_only is above correct_only on both targets with no overlap across seeds, "
        "and correct_only is below scratch on both. The in-domain full-objective probes are the "
        "separate source-sweep cells above (assist2017 0.1458, junyi 0.0202)._\n"
    )

    new, applied = text, []
    for name, anchor, ins, marker in edits:
        if marker in new:
            print(f"already applied: {name}")
            continue
        n = new.count(anchor)
        if n != 1:
            sys.exit(f"ABORT: anchor for '{name}' found {n} times, expected 1; nothing written")
        if ins is None:
            new = new.replace(anchor, "\n" + block62 + anchor)
        else:
            new = new.replace(anchor, anchor + ins)
        applied.append(name)

    junyi_marker = "- **Junyi release:**"
    exit_code = 0
    if junyi_marker in new:
        print("already applied: section 0 Junyi release")
    else:
        raw = Path(a.raw_junyi)
        if not raw.exists():
            print(f"WARNING: {raw} not found; Junyi release line not written")
            exit_code = 2
        else:
            rows, header = count_rows(raw)
            print(f"{raw}: {rows} data rows; header: {header[:120]}")
            if rows != a.expect_rows:
                print(f"WARNING: {rows} != stated {a.expect_rows}; Junyi release line not written")
                exit_code = 2
            else:
                anchor = "- **Parsers:** `collect_all_results.py` (this file);"
                if new.count(anchor) != 1:
                    sys.exit("ABORT: section 0 Parsers anchor not unique; nothing written")
                i = new.index(anchor)
                j = new.index("\n", i)
                line = (f"\n- **Junyi release:** Kaggle 2020 \"Junyi Academy Online Learning "
                        f"Activity Dataset\" (`raw/Log_Problem.csv`, {rows:,} data rows counted "
                        f"2026-09-30, equal to the problem-attempt count the release states; "
                        f"license CC BY-NC-SA 4.0; "
                        f"https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-"
                        f"dataset-by-junyi-academy). Not the 2015 EduData problem log "
                        f"(`junyi_ProblemLog_original.csv`).")
                new = new[:j] + line + new[j:]
                applied.append("section 0 Junyi release")

    old_manual = 'MANUAL = ["### 2.1", "### 2.2", "### 3.1", "### 3.2", "### 6.1", "### 8.1",'
    new_manual = 'MANUAL = ["### 1.1", "### 2.1", "### 2.2", "### 3.1", "### 3.2", "### 6.1", "### 6.2", "### 8.1",'
    cnew = ctext
    if new_manual in ctext:
        print("already applied: collector MANUAL list")
    elif ctext.count(old_manual) != 1:
        sys.exit("ABORT: collector MANUAL anchor not found exactly once; nothing written")
    else:
        cnew = ctext.replace(old_manual, new_manual)
        ast.parse(cnew)
        applied.append("collector MANUAL list")

    for marker in ("### 1.1 SAINT+ per-seed KT test AUC", "**Truncation provenance (log check",
                   "### 6.2 Objective-conditioned probe", "**SAINT+ response-time caveat"):
        if new.count(marker) != 1:
            sys.exit(f"ABORT: marker {marker!r} would appear {new.count(marker)} times; nothing written")

    if new != text:
        res.write_text(new)
    if cnew != ctext:
        col.write_text(cnew)
    print("applied: " + (", ".join(applied) if applied else "nothing (already up to date)"))
    print(f"after:  {res} {md5(res)}   {col} {md5(col)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
