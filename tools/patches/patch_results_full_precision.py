from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Why this exists: the MRAP transfer benchmark (2026-09-23) recomputed RESULTS.md's values from
# the per-seed W&B summaries at full precision instead of the 4 dp log banners, and rechecked
# provenance. It found a copied value in section 11 (the 353,597 draw-42 entry was the 150,000
# one), the EdNet row of section 4 run with Junyi-source encoders, disagreeing duplicate runs in
# section 8.2, and cells that 4 dp averaging moved by 0.0001. The NeurIPS and C&E drafts were
# corrected the same day; this brings RESULTS.md in line. Intervals are rerun with the script
# that made them (paired_bootstrap_pair.py, rng seed 42; paired_bootstrap_objective.py, rng
# seed 0). Values are averaged in Decimal and rounded once, half away from zero.

TARGET = Path("RESULTS.md")
TAG = "CORRECTED 2026-09-23 (tools/patches/patch_results_full_precision.py)"

E = [
    # section 1
    ("| EdNet | 0.6799 | 0.6720 | **0.6860** | 0.6777 | 0.6846 | +0.0069 [+0.0065,+0.0075] 6/6 |",
     "| EdNet | 0.6799 | 0.6721 | **0.6860** | 0.6777 | 0.6846 | +0.0070 [+0.0066,+0.0075] 6/6 |", "s1 EdNet"),
    ("| Junyi | 0.7586 | 0.7534 | **0.7984** |", "| Junyi | 0.7586 | 0.7534 | **0.7985** |", "s1 Junyi SAINT+"),
    ("| Bridge2006 | 0.7947 |", "| Bridge2006 | 0.7946 |", "s1 Bridge DKT"),
    ("| ASSIST2009 | **0.8761** |", "| ASSIST2009 | **0.8762** |", "s1 ASSIST2009 DKT"),
    ("| Algebra2006 | 0.8028 | 0.7741 | **0.8748** | 0.7868 |",
     "| Algebra2006 | 0.8028 | 0.7741 | **0.8748** | 0.7867 |", "s1 Algebra2006 scratch"),
    ("AKT is lowest on 6 of 7, SAINT+ is lowest on ASSIST2017.",
     "AKT is lowest on 5 of 7, and SAINT+ is lowest on ASSIST2017 and ASSIST2009. " + TAG
     + ": five cells recomputed from per-seed W&B values at full precision (EdNet AKT and gain, "
     "Junyi SAINT+, Bridge2006 DKT, ASSIST2009 DKT, Algebra2006 scratch), and the AKT count "
     "corrected from 6 to 5.", "s1 read"),
    # section 2.1
    ("and the count of seeds in the direction of the mean. SUPERSEDES the earlier 3-seed table.",
     "and the count of seeds with a positive gain. SUPERSEDES the earlier 3-seed table. " + TAG
     + ": means, gains and intervals recomputed at full precision with the same bootstrap; "
     "four cells moved by 0.0001, and the count column holds positive seeds, as the two "
     "negative rows always did.", "s2.1 header"),
    ("0.6957 ±0.0021 (+0.0260 [+0.0250,+0.0274] 6/6)", "0.6957 ±0.0021 (+0.0261 [+0.0251,+0.0274] 6/6)", "s2.1 ednet->assist"),
    ("0.6732 ±0.0007 (+0.0088 [+0.0076,+0.0100] 6/6)", "0.6732 ±0.0007 (+0.0088 [+0.0076,+0.0099] 6/6)", "s2.1 ednet in-domain"),
    ("0.6687 ±0.0010 (+0.0043 [+0.0033,+0.0053] 6/6)", "0.6687 ±0.0010 (+0.0043 [+0.0032,+0.0053] 6/6)", "s2.1 junyi->ednet"),
    ("0.7417 ±0.0004 (+0.0064 [+0.0057,+0.0072] 6/6)", "0.7416 ±0.0004 (+0.0065 [+0.0057,+0.0072] 6/6)", "s2.1 ednet->junyi"),
    ("(assist2017 +0.0260 [+0.0250,+0.0274] vs +0.0231 [+0.0220,+0.0239]; junyi +0.0064 [+0.0057,+0.0072] vs",
     "(assist2017 +0.0261 [+0.0251,+0.0274] vs +0.0231 [+0.0220,+0.0239]; junyi +0.0065 [+0.0057,+0.0072] vs", "s2.1 read"),
    # section 2.2
    ("same budget and same sources as 2.1, second task. Same reporting format.",
     "same budget and same sources as 2.1, second task. Same reporting format. " + TAG
     + ": two interval ends recomputed at full precision.", "s2.2 header"),
    ("0.9822 ±0.0002 (+0.0026 [+0.0023,+0.0029] 6/6)", "0.9822 ±0.0002 (+0.0026 [+0.0023,+0.0028] 6/6)", "s2.2 junyi->assist"),
    ("0.9912 ±0.0004 (+0.0029 [+0.0020,+0.0043] 6/6)", "0.9912 ±0.0004 (+0.0029 [+0.0019,+0.0043] 6/6)", "s2.2 junyi in-domain"),
    # section 3
    ("gain +0.0069 CI [+0.0065,+0.0075], 6/6 seeds, both n=6. This reproduced exactly when "
     "recomputed from per-seed logs; it is no longer a recorded value.",
     "gain +0.0070 CI [+0.0066,+0.0075], 6/6 seeds, both n=6. " + TAG + ": recomputed from "
     "per-seed W&B values at full precision (0.0069654); the 4 dp log values average to exactly "
     "0.00695, which had been rounded to +0.0069.", "s3 full-scale gain"),
    # section 4
    ("Per-dataset objective ablation (EdNet-source encoders: full / skill_only / correct_only -> target), from logs:",
     "Per-dataset objective ablation (EdNet-source encoders: full / skill_only / correct_only -> "
     "target, EXCEPT the ednet row, whose encoders are Junyi-source: every w8_regime_ednet log "
     "loads edubert_junyi_pretrain_{full,junyi_skill_only,junyi_correct_only}_encoder.pt), from "
     "logs. " + TAG + ": the ednet-row source, and means and pstdev recomputed from per-seed W&B "
     "values at full precision.", "s4 header"),
    ("| assist2017 | 0.6896 ±0.0021 (n=6) | 0.6860 ±0.0021 (n=6) | 0.6620 ±0.0008 (n=6) |",
     "| assist2017 | 0.6896 ±0.0022 (n=6) | 0.6860 ±0.0021 (n=6) | 0.6619 ±0.0008 (n=6) |", "s4 assist2017"),
    ("| ednet | 0.6599 ±0.0009 (n=6) |", "| ednet | 0.6599 ±0.0008 (n=6) |", "s4 ednet"),
    ("| algebra2005 | 0.7873 ±0.0037 (n=6) | 0.7871 ±0.0023 (n=6) | 0.7643 ±0.0057 (n=6) |",
     "| algebra2005 | 0.7873 ±0.0038 (n=6) | 0.7871 ±0.0023 (n=6) | 0.7644 ±0.0057 (n=6) |", "s4 algebra2005"),
    ("| assist2009 | 0.8699 ±0.0017 (n=6) | 0.8692 ±0.0016 (n=6) | 0.8662 ±0.0007 (n=6) |",
     "| assist2009 | 0.8699 ±0.0018 (n=6) | 0.8692 ±0.0016 (n=6) | 0.8662 ±0.0006 (n=6) |", "s4 assist2009"),
    ("skill-correct +0.0186 CI [+0.0148,+0.0228] 6/6", "skill-correct +0.0186 CI [+0.0148,+0.0227] 6/6", "s4 algebra2006 CI"),
    # section 5
    ("- K=10 (pps~0.10): correct-skill +0.0121 CI [+0.0033,+0.0212] (6/6)",
     "- K=10 (pps~0.10): correct-skill +0.0121 CI [+0.0033,+0.0212] (4/6; " + TAG
     + ": the seed count read 6/6, but 4 of the 6 per-seed differences are positive at 4 dp "
     "and at full precision)", "s5 K=10 count"),
    ("but scratch gains only +0.0079 while full gains", "but scratch gains only +0.0078 while full gains", "s5 scratch step"),
    ("**Truncation sweep (from logs, KT AUC means):**",
     "**Truncation sweep (from logs, KT AUC means; " + TAG + ": four cells recomputed at full precision):**", "s5 table header"),
    ("| 20 | 0.6410 | 0.6399 | 0.6426 | 0.6476 |", "| 20 | 0.6410 | 0.6399 | 0.6427 | 0.6476 |", "s5 K=20"),
    ("| 40 | 0.6438 | 0.6415 | 0.6504 | 0.6423 |", "| 40 | 0.6437 | 0.6415 | 0.6503 | 0.6423 |", "s5 K=40"),
    ("| 80 | 0.6496 | 0.6450 |", "| 80 | 0.6497 | 0.6450 |", "s5 K=80"),
    # section 6
    ("| assist2017 | 0.1417 ±0.0010 (n=3) | 0.1366 ±0.0011 (n=3) | +0.0051 |",
     "| assist2017 | 0.1416 ±0.0011 (n=3) | 0.1366 ±0.0011 (n=3) | +0.0050 |", "s6 probe7 assist2017"),
    ("_Pretrained beats scratch on all 7 (recorded gains +0.004 to +0.039, all positive).",
     TAG + ": assist2017 pretrained 0.1417 -> 0.1416 (pstdev 0.0011) and its gain +0.0051 -> "
     "+0.0050, full precision; the same runs appear in the probe2 table below.\n\n"
     "_Pretrained beats scratch on all 7 (recorded gains +0.004 to +0.039, all positive).", "s6 note"),
    ("| assist2017 | **0.1458 +/-0.0002** | 0.1417 +/-0.0010 |",
     "| assist2017 | **0.1458 +/-0.0002** | 0.1416 +/-0.0011 |", "s6 probe2 assist2017"),
    ("| junyi | 0.0201 +/-0.0001 |", "| junyi | 0.0202 +/-0.0001 |", "s6 probe2 junyi in-domain"),
    ("(assist2017 probe7-full == probe2 EdNet-source 0.1417;", "(assist2017 probe7-full == probe2 EdNet-source 0.1416;", "s6 consistency"),
    # section 8.2
    ("Paired-by-seed effects vs scratch on this corrected grid, matching the recorded 8-seed "
     "paired-bootstrap results exactly (so the original analysis used correct extraction):",
     TAG + ". Seven seed-condition pairs in this grid were executed twice and the copies "
     "disagree: scratch k5 s4 (0.4995 / 0.5285) and s5 (0.5000 / 0.5275); fromassist k5 s3 "
     "(0.5000 / 0.5139), s6 (0.5364 / 0.7159) and s7 (0.5130 / 0.4995); fromassist k10 s3 "
     "(0.5043 / 0.7152) and s6 (0.6456 / 0.5194). The table shows the first copy. Those seeds "
     "are now dropped from every comparison they enter instead of taking a copy; where one copy "
     "came from a job that later timed out (scratch and indomain k5 s42), the completed job's "
     "copy is kept, as in the table. The fromjunyi k10 and indomain k10 cells touch no "
     "disagreeing pair and are unchanged.\n\n"
     "Paired-by-seed effects vs scratch on this corrected grid, matching the recorded 8-seed "
     "paired-bootstrap results exactly (so the original analysis used correct extraction):", "s8.2 note"),
    ("- indomain k5: mean +0.0968 (6/8 seeds positive) - recorded +0.097, CI [+0.029, +0.164].",
     "- indomain k5: mean +0.0902 [+0.0153, +0.1601], 4/6 seeds positive, seeds 4 and 5 dropped "
     "(paired_bootstrap_pair.py, rng seed 42, W&B values). It read +0.0968 [+0.0295, +0.1637] "
     "6/8 on the first copies.", "s8.2 indomain k5"),
    # section 11
    ("| 150,000 | 2.9565 / 2.9613 / 2.9585 | +0.0240 | +0.0043 | 0.0012 |",
     "| 150,000 | 2.9565 / 2.9613 / 2.9585 | +0.0241 | +0.0043 | 0.0012 |", "s11 150,000"),
    ("| 353,597 | 2.8784 / 2.8872 / 2.8740 | +0.0250 | +0.0055 | 0.0034 |",
     "| 353,597 | 2.8784 / 2.8872 / 2.8740 | +0.0259 | +0.0062 | 0.0017 |", "s11 353,597"),
    ("Per-draw gains at 150,000, assist2017: +0.0233 / +0.0245 / +0.0243.",
     "Per-draw gains at 150,000, assist2017: +0.0234 / +0.0245 / +0.0243.", "s11 per-draw 150,000"),
    ("Per-draw gains at 353,597, assist2017: +0.0233 / +0.0250 / +0.0267.",
     "Per-draw gains at 353,597, assist2017: +0.0261 / +0.0250 / +0.0267.", "s11 per-draw 353,597 assist"),
    ("Per-draw gains at 353,597, junyi: +0.0043 / +0.0061 / +0.0060.",
     "Per-draw gains at 353,597, junyi: +0.0065 / +0.0062 / +0.0060.", "s11 per-draw 353,597 junyi"),
    ("THE 353,597 ROW WAS REVISED DOWNWARD ON 2026-09-18.",
     "THE 353,597 ROW WAS CORRECTED ON 2026-09-23 (" + TAG.split(" (")[1].rstrip(")")
     + "). The 2026-09-18 revision below entered the 150,000 draw-42 values (+0.0233 / +0.0043) "
     "in place of the 353,597 draw-42 values. Draw 42 at 353,597 is the pre-existing encoder and "
     "its n3000 transfer runs (section 2.1), +0.0261 / +0.0065 at full precision, so the row is "
     "+0.0259 / +0.0062 with draw spreads 0.0017 / 0.0004, and the pre-existing encoder is the "
     "middle draw on assist2017 and the strongest on junyi, not the weakest. The superseded "
     "2026-09-18 note follows for provenance. THE 353,597 ROW WAS REVISED DOWNWARD ON 2026-09-18.",
     "s11 correction note"),
    ("The 0.0034 and 0.0019 spreads here therefore bound initialization variance",
     "The 0.0017 and 0.0004 spreads here therefore bound initialization variance", "s11 spreads"),
    ("For scale,\nthe largest effect anywhere in this file is +0.0260.",
     "For scale,\nthe largest effect anywhere in this file is +0.0261.", "s11 largest effect"),
    ("(c) The curve SATURATES at the top. Going from\n150,000 to the full 353,597 students, a 2.4-fold "
     "increase in corpus, buys\n+0.0010 on assist2017 (+0.0240 to +0.0250) and +0.0012 on junyi "
     "(+0.0043 to\n+0.0055). Both are smaller than the draw spread at the larger size, 0.0034 and\n"
     "0.0019, so the final step of the curve is NOT resolvable at three draws. The\ndefensible "
     "statement is that transfer rises steeply from 5,000 to 49,153,\ncontinues to 150,000, and "
     "then flattens; do not read the top of the curve as\nstill climbing. This is the source-side "
     "analogue of the target-side saturation\nin section 4._",
     "(c) The curve keeps rising at the top, with diminishing returns (" + TAG + "; it read "
     "\"saturates\" on the copied 353,597 value). Going from\n150,000 to the full 353,597 students, "
     "a 2.4-fold increase in corpus, buys\n+0.0018 on assist2017 (+0.0241 to +0.0259) and +0.0019 "
     "on junyi (+0.0043 to\n+0.0062), and every 353,597 draw exceeds every 150,000 draw on both "
     "targets\n(exact one-sided rank test p = 0.05, the floor at three draws per size). The step\n"
     "is smaller than the one below it (+0.0063 and +0.0029 from 49,153 to 150,000).\nThe "
     "defensible statement is that transfer rises steeply from 5,000 to 49,153,\ncontinues to "
     "150,000, and keeps rising more slowly to the full corpus; no plateau\nappears in the measured "
     "range. The target side (section 3) differs: there the\nbenefit shrinks as target data grows._",
     "s11 read (c)"),
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
    src = TARGET.read_text(encoding="utf-8")
    out = src
    for old, new, why in E:
        out = apply(out, old, new, why)
    if out != src:
        TARGET.write_text(out, encoding="utf-8")
    print("md5", hashlib.md5(TARGET.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
