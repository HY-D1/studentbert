"""Write the matched-scale source ablation into RESULTS.md and protect it.

Two edits, one commit. Section 11 is hand-built, so `## 11.` MUST go into the
collector's MANUAL array or the next `collect_all_results.py` run overwrites it.
That is the whole reason these two changes belong together.

Run from the repo root:
    PYTHONPATH=. python tools/patches/patch_results_srcscale.py
"""

from __future__ import annotations

import pathlib
import sys

RESULTS = pathlib.Path("RESULTS.md")
COLLECTOR = pathlib.Path("analysis/collect_all_results.py")

SECTION = r"""
## 11. Matched-scale source ablation: does corpus size cause transfer quality?

Source identity held FIXED at EdNet KT1, only the number of pretraining
students varied. This is the manipulation that separates source SIZE from
source IDENTITY, which are perfectly confounded in section 2.1 (three sources,
three sizes, three platforms).

Encoders `edubert_ednet_pretrain_ednet_n{SIZE}{,d1,d2}_encoder.pt`, built by
`slurm/generators/gen_source_scale.sh` with `scripts/pretrain_edubert.py
--n_students SIZE --seed {42,1,2} --epochs 10 --batch_size 128 --lr 1e-3
--warmup_frac 0.05`. THREE INDEPENDENT ENCODERS PER SIZE; the draw seed is the
pretrain `--seed`, so each draw differs in both the sampled students and the
initialization, and the spread across draws bounds total encoder-build variance
at that size, not sampling variance alone.

Transfer runs `edubert_<target>_kt_<t>_fromednet_n{SIZE}{,d1,d2}_src_n3000_seed{42,1,2,3,4,5}`,
targets assist2017 and junyi only (the two where EdNet is foreign), KT only,
N=3000 target budget, 6 seeds, all pinned to gpu:v100-sxm2. Scratch controls
are REUSED from the n3000 grid (`kt_{assist,junyi}_scratch_n3000_seed*`), same
budget, same seeds, same GPU model, so the pairing is valid and no scratch jobs
were re-run. The 353,597 row is the pre-existing full-corpus encoder
`edubert_ednet_pretrain_full_encoder.pt` and its existing n3000 transfer runs.

| Source students | mlm loss (3 draws) | assist2017 gain | junyi gain | draw spread (assist) |
|---|---|---|---|---|
| 1,366 | 4.5344 / 5.2432 / 5.2274 / 4.8507 | UNSTABLE, see below | UNSTABLE | 0.0533 |
| 5,000 | 4.0644 / 4.0596 / 4.0566 | -0.0018 | -0.0037 | 0.0013 |
| 15,000 | 3.6959 / 3.6867 / 3.6952 | +0.0002 | -0.0023 | 0.0010 |
| 49,153 | 3.1054 / 3.0965 / 3.1116 | +0.0177 | +0.0014 | 0.0018 |
| 150,000 | 2.9565 / 2.9613 / 2.9585 | +0.0240 | +0.0043 | 0.0012 |
| 353,597 | 2.8784 (one draw only) | +0.0260 | +0.0064 | n/a |

Gains are means over the three draws, each draw itself a 6-seed mean paired
against the same scratch controls (assist2017 scratch 0.6697, junyi 0.7352).
1,366 and 49,153 are the ASSISTments and Junyi TRAINING-SPLIT sizes, so those
two rows are exact size matches to the other two sources in section 2.1.

Per-draw gains at 150,000, assist2017: +0.0233 / +0.0245 / +0.0243.
Per-draw gains at 150,000, junyi: +0.0043 / +0.0042 / +0.0044.

THE 1,366 ROW IS A STABILITY BOUNDARY, NOT A MEASUREMENT. Four encoders at that
size span 0.7088 in final mlm loss (4.5344 to 5.2432) against a chance value of
ln(143) + ln(2) = 5.656, so they range from barely-trained to nearly-untrained.
Downstream the fine-tune becomes bimodal: 24 runs were executed TWICE with an
identical command, identical encoder and identical seed, and differ by up to
0.0700 AUC, median 0.0198 to 0.0423 depending on the cell, with junyi values
piling up at either ~0.6835 or ~0.732 and almost nothing between. For scale,
the largest effect anywhere in this file is +0.0260. Report this row as a
characterized failure mode with the duplicate-execution evidence attached;
never as a point estimate.

PROVENANCE NOTE ON THE 1,366 d42 ENCODER: it was built interactively (job
10349907, node d1013, 2026-09-14 17:58:52 to 18:00:12) without `--wandb`, so it
has no job log and no W&B run, and it is the only encoder in the ladder on
unpinned hardware. Its loss of 4.5344 is recorded from the terminal session.

DISCARDED RUNS: a first batch of 24 transfer runs at 150,000 (jobs 10362252 to
10362281, submitted 12:07) loaded encoders that had not finished training; the
pretraining script writes a checkpoint on every loss improvement, and jobs
10361662 and 10361663 did not end until 13:17. Those logs are quarantined in
`stale_n150000d/` and MUST NOT be aggregated. They produced a spurious draw
spread of 0.0175 which disappeared to 0.0012 on the valid rerun. One run,
`kt_junyi_fromednet_n150000d2_src_n3000_seed5`, executed twice in the valid
batch (0.7388 and 0.7391); either is usable.

_Read: holding source identity fixed and varying only student count reproduces
the entire range of behaviour seen across the three real sources. Below ~15,000
pretraining students EdNet-as-source does not help and on junyi actively hurts;
above ~49,153 it helps on both targets; the sign flip sits between 15,000 and
49,153 and replicates on both targets across three independently built encoders
per size, with a draw spread never above 0.0018. Corpus SIZE is therefore doing
the work, not corpus identity. Two further observations. (a) EdNet cut to
Junyi's 49,153 students gives +0.0177 [+0.0162,+0.0194] on assist2017 against
real Junyi's +0.0214 [+0.0196,+0.0231], so at matched size EdNet is if anything
the weaker corpus per student and its full-scale advantage is entirely a size
effect. (b) Within a fixed vocabulary the pretraining losses ARE comparable, so
this is the first clean loss-versus-transfer test: loss falls smoothly and
monotonically while transfer is a threshold, with 50.6% of the total loss
improvement (4.5344 to 3.6959) buying a transfer change of -0.0001 to -0.0004.
Loss tracks transfer ordinally (Spearman -0.83 on both targets) but not
proportionally. CAVEAT: the 353,597 row rests on ONE encoder and cannot get
further draws without re-pretraining the whole corpus._

DIGIT COLLISION: this Spearman -0.83 is a THIRD quantity at that magnitude,
alongside the probe-vs-transfer rho +0.83 (section 6) and the scale/pps
confound r -0.83 (section 5). Never mix them.

_End of consolidated results._
"""


def main() -> None:
    for p in (RESULTS, COLLECTOR):
        if not p.exists():
            sys.exit("ABORT: %s not found. Run from the repo root." % p)

    md = RESULTS.read_text()
    col = COLLECTOR.read_text()

    if "## 11. Matched-scale source ablation" in md and '"## 11."' in col:
        print("already applied; nothing to do")
        return

    tail = "_End of consolidated results._"
    if md.count(tail) != 1:
        sys.exit("ABORT: expected exactly one %r in RESULTS.md, found %d"
                 % (tail, md.count(tail)))
    if "## 11." in md:
        sys.exit("ABORT: RESULTS.md already has a section 11; not appending.")

    old_manual = '    MANUAL = ["### 2.1", "### 2.2", "### 3.1", "### 3.2", "### 6.1", "### 8.1",\n              "### 8.2", "## 10."]'
    new_manual = '    MANUAL = ["### 2.1", "### 2.2", "### 3.1", "### 3.2", "### 6.1", "### 8.1",\n              "### 8.2", "## 10.", "## 11."]'
    if col.count(old_manual) != 1:
        if '"## 11."' in col:
            print("collector already protects '## 11.'")
        else:
            sys.exit("ABORT: MANUAL array anchor did not match exactly once. "
                     "RESULTS.md has NOT been modified.")
    else:
        COLLECTOR.write_text(col.replace(old_manual, new_manual, 1))
        print("ok: '## 11.' added to the collector MANUAL array")

    RESULTS.write_text(md.replace(tail, SECTION.strip() + "\n", 1))
    print("ok: section 11 appended to RESULTS.md")

    md2 = RESULTS.read_text()
    print()
    print("RESULTS.md sections now: %d" % md2.count("\n## "))
    print("'_End of consolidated results._' occurrences: %d (must be 1)"
          % md2.count("_End of consolidated results._"))
    print("collector protects section 11: %s"
          % ('"## 11."' in COLLECTOR.read_text()))


if __name__ == "__main__":
    main()
