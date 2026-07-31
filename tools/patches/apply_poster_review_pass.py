#!/usr/bin/env python3
"""Apply the poster readability review pass (16pt body) to poster/make_poster.py.

Idempotent: a pair whose NEW text is already present is skipped; a pair whose
OLD text is absent (and NEW also absent) is reported as a CONFLICT, which is
what happens if a local manual edit touched that block. Run:

    python3 apply_poster_review_pass.py [path/to/make_poster.py]
"""
import ast
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "poster/make_poster.py"

PAIRS = [
# ---------------------------------------------------------------- heights
("stack(0, [0.148, 0.168, 0.404])",
 "stack(0, [0.160, 0.152, 0.408])"),
("stack(1, [0.268, 0.197, 0.086, 0.164])",
 "stack(1, [0.264, 0.194, 0.094, 0.163])"),
("stack(2, [0.428, 0.304])",
 "stack(2, [0.404, 0.328])"),
("stack(3, [0.416, 0.142, 0.162])",
 "stack(3, [0.388, 0.146, 0.194])"),
# ---------------------------------------------------------------- card 1
("""mot = [
    "Adaptive learning platforms personalize by modeling what each student knows from their interaction history (knowledge tracing).",
    "Those models are data-hungry, yet every new course, school, or tool starts with only dozens of students.",
    "In NLP and vision, pretrain-then-fine-tune is the standard fix. For student data there is no recipe: pretrain on what, for what, and when is it worth it?",
]""",
 """mot = [
    "Adaptive platforms model what each student knows from their interaction history (knowledge tracing).",
    "Those models are data-hungry, yet every new course or tool starts with only dozens of students.",
    "Cold-start learners get the least personalized support, and small classrooms may never reach big-data scale.",
    "In NLP and vision, pretrain-then-fine-tune is the standard fix. For student data there is no recipe.",
]"""),
# ---------------------------------------------------------------- card 3
("""T(cx, ct, W("StudentBERT is a BERT-style model for student learning histories: every step is a "
            "skill practiced, whether the answer was right, and a response-time bin.", 68),
  14.5, ls=1.24, color=SLATE)""",
 """T(cx, ct, W("StudentBERT is a BERT-style model for student learning histories: every step is a "
            "skill practiced, whether the answer was right, and a response-time bin.", 58),
  16, ls=1.24, color=SLATE)"""),
("ax = mkax([cx, cy + 0.010, cw, ct - cy - 0.042])",
 "ax = mkax([cx, cy + 0.010, cw, ct - cy - 0.052])"),
("""ax.text(0.5, 0.155, W("Scale of the study: 9 pretrained encoders, 7 target datasets, hundreds of controlled "
                      "fine-tuning runs. Sources always pretrain at full scale; targets are subsampled to fixed "
                      "budgets N. Every run is logged to Weights & Biases; RESULTS.md consolidates all numbers.", 66),
        fontsize=13.5, ha="center", va="top", color=SLATE, linespacing=1.3)""",
 """ax.text(0.5, 0.155, W("Scale: 9 pretrained encoders, 7 target datasets, hundreds of controlled "
                      "fine-tuning runs. Sources pretrain at full scale; targets are "
                      "subsampled to fixed budgets N.", 54),
        fontsize=16, ha="center", va="top", color=SLATE, linespacing=1.3)"""),
# diagram label bumps (+1pt on everything under 15)
('dbox(x0, 0.865, tw, th, ["MASKED"], fc=INK, ec=INK, sizes=[14.5], colors=["white"], weights=["bold"])',
 'dbox(x0, 0.865, tw, th, ["MASKED"], fc=INK, ec=INK, sizes=[15.5], colors=["white"], weights=["bold"])'),
('dbox(x0, 0.865, tw, th, [a, b], sizes=[14.5, 12.5], colors=[INK, SLATE], weights=["bold", "normal"])',
 'dbox(x0, 0.865, tw, th, [a, b], sizes=[15.5, 13.5], colors=[INK, SLATE], weights=["bold", "normal"])'),
('        ax.text(max(x1, x2) + lx, (y1 + y2) / 2, lbl, fontsize=14.5, color=SLATE, va="center")',
 '        ax.text(max(x1, x2) + lx, (y1 + y2) / 2, lbl, fontsize=15.5, color=SLATE, va="center")'),
('dbox(0.15, 0.685, 0.70, 0.112, ["StudentBERT", "transformer encoder"], fc="#FDECEC", ec=RED, lw=2.2,\n     sizes=[19, 13.5], colors=[RED, SLATE], weights=["bold", "normal"])',
 'dbox(0.15, 0.685, 0.70, 0.112, ["StudentBERT", "transformer encoder"], fc="#FDECEC", ec=RED, lw=2.2,\n     sizes=[19, 14.5], colors=[RED, SLATE], weights=["bold", "normal"])'),
('dbox(0.06, 0.535, 0.88, 0.090, ["Pretraining: predict the masked skill AND correctness"],\n     sizes=[14.5], weights=["bold"])',
 'dbox(0.06, 0.535, 0.88, 0.090, ["Pretraining: predict the masked skill AND correctness"],\n     sizes=[15.5], weights=["bold"])'),
('ax.text(0.83, 0.479, "objective\\nablation", fontsize=12.5, color=SLATE, va="center")',
 'ax.text(0.83, 0.479, "objective\\nablation", fontsize=13.5, color=SLATE, va="center")'),
('    dbox(x0, 0.235, 0.305, 0.115, [a, b], sizes=[14, 12.5], weights=["bold", "normal"], colors=[INK, SLATE])',
 '    dbox(x0, 0.235, 0.305, 0.115, [a, b], sizes=[15, 13.5], weights=["bold", "normal"], colors=[INK, SLATE])'),
('    dbox(0.10 + i * 0.24, 0.448, 0.21, 0.062, [ob], fc="#F1F0EC", ec=SLATE, lw=1.3, sizes=[13])',
 '    dbox(0.10 + i * 0.24, 0.448, 0.21, 0.062, [ob], fc="#F1F0EC", ec=SLATE, lw=1.3, sizes=[14])'),
# ---------------------------------------------------------------- card 4 captions
("""T(cx, yy - 0.002, W("PPS proxy = median raw sequence length / skills, measured before the model's 512-step cap. Regime = "
                    "whether skill-only or correct-only transfers better (result R3); the full "
                    "objective has the highest mean on 5 of 7.", 74),
  13.5, color=SLATE, ls=1.25)""",
 """T(cx, yy - 0.002, W("PPS proxy = median raw sequence length / skills, before the 512-step cap. "
                    "Regime = which objective transfers better (R3).", 62),
  16, color=SLATE, ls=1.25)"""),
("""T(cx, yy - 0.036, W("One schema for all 7: sequences of (skill, correct, response-time bin); split 80/10/10 by "
                    "student, seed 42; min 10 interactions. Total 511,273 retained learner sequences, 115.2M interactions.", 74),
  13.5, color=SLATE, ls=1.25)""",
 """T(cx, yy - 0.030, W("One schema for all 7; split 80/10/10 by student; min 10 interactions; "
                    "511,273 retained sequences, 115.2M interactions.", 62),
  16, color=SLATE, ls=1.25)"""),
# ---------------------------------------------------------------- card 5
("""ev = [
    "Metrics: knowledge-tracing AUC (0.5 = chance, 1.0 = perfect), next-skill top-1, and early-disengagement AUC. Disengagement = bottom quartile of total interactions in the eligible cohort; prefix censoring keeps length out of the input.",
    "Budget-matched: every condition fine-tunes on the same number of target students (N).",
    "Seeds vary fine-tuning only: one fixed pretrained checkpoint per source and objective, so intervals cover downstream variation, not the whole pipeline. 3 seeds descriptive, 6 to 8 paired-bootstrap for confirmatory claims.",
    "Predict-before-test: predictions were written into the experiment scripts before the runs. Algebra 2006-07 is the one that held, at +0.0186.",
    "Fine-tuning picks the best checkpoint by validation; pretraining picks the lowest training MLM loss.",
]""",
 """ev = [
    "Metrics: knowledge-tracing AUC (0.5 = chance, 1.0 = perfect), next-skill top-1, early-disengagement AUC (bottom quartile of activity; prefix censoring keeps length out of the input).",
    "Budget-matched: every condition fine-tunes on the same number of target students (N).",
    "Seeds vary fine-tuning only, on one fixed checkpoint per source and objective; fine-tuning picks the best checkpoint by validation, pretraining the lowest training MLM loss. 3 seeds descriptive, 6 to 8 with paired bootstrap for confirmatory claims.",
    "Predict-before-test: predictions were written into the scripts before the runs. Algebra 2006-07 held, at +0.0186.",
]"""),
("""    T(cx + 0.0085, yy, W(m, 72), 14.5, ls=1.26)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0050), 0.0032, 0.0032,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0106 * len(textwrap.wrap(m, 72)) + 0.0050""",
 """    T(cx + 0.0085, yy, W(m, 64), 16, ls=1.26)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0055), 0.0034, 0.0034,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0117 * len(textwrap.wrap(m, 64)) + 0.0052"""),
# ---------------------------------------------------------------- card 6
("""    T(cx + 0.0085, yy, W(m, 76), 13, ls=1.22)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0046), 0.003, 0.003,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0088 * len(textwrap.wrap(m, 76)) + 0.0033""",
 """    T(cx + 0.0085, yy, W(m, 62), 16, ls=1.22)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0055), 0.0034, 0.0034,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0117 * len(textwrap.wrap(m, 62)) + 0.0040"""),
# ---------------------------------------------------------------- R1
("""T(cx, ct, W("Next-skill recommendation on ASSISTments 2017: given a student's history, which skill "
            "do they practice next? Top-1 accuracy means the model's first guess is right. Fine-tuned "
            "on N = 25 to 1,000 students (4 conditions x 6 budgets x 3 seeds). In-domain means "
            "pretrained on more data from the same platform; the other two sources come from "
            "different platforms entirely.", 78),
  15.5, ls=1.26, color=SLATE)""",
 """T(cx, ct, W("Next-skill prediction on ASSISTments 2017: which skill does a student practice next? "
            "Top-1 = the first guess is right. Fine-tuned on N = 25 to 1,000 students "
            "(4 conditions x 6 budgets x 3 seeds). In-domain = pretrained on more data "
            "from the same platform.", 62),
  16, ls=1.26, color=SLATE)"""),
('ax1.set_xticks(N); ax1.set_xticklabels([str(n) for n in N], fontsize=14)',
 'ax1.set_xticks(N); ax1.set_xticklabels([str(n) for n in N], fontsize=15)'),
('ax1.tick_params(axis="y", labelsize=14)',
 'ax1.tick_params(axis="y", labelsize=15)'),
('ax1.set_ylabel("top-1 accuracy (%), axis truncated", fontsize=14)',
 'ax1.set_ylabel("top-1 accuracy (%), axis truncated", fontsize=15)'),
('ax1.set_xlabel("number of target students (log scale)", fontsize=15)',
 'ax1.set_xlabel("number of target students (log scale)", fontsize=16)'),
('ax1.text(34, 81.3, "largest gains", fontsize=13.5, color=RED, ha="center", weight="bold")',
 'ax1.text(34, 81.3, "largest gains", fontsize=14.5, color=RED, ha="center", weight="bold")'),
('ax1.text(100, 81.3, "modest", fontsize=13.5, color=SLATE, ha="center")',
 'ax1.text(100, 81.3, "modest", fontsize=14.5, color=SLATE, ha="center")'),
('ax1.text(470, 81.3, "gap under 1.5 pts", fontsize=13.5, color=SLATE, ha="center")',
 'ax1.text(470, 81.3, "gap under 1.5 pts", fontsize=14.5, color=SLATE, ha="center")'),
('ax1.text(28, 70.3, "+9.2 pts", fontsize=15, weight="bold", color=INK)',
 'ax1.text(28, 70.3, "+9.2 pts", fontsize=16, weight="bold", color=INK)'),
('ax1.text(25, 64.0, "EdNet at N=25 is high\\nvariance (std 4.6 pts)", fontsize=11.5, color=SLATE)',
 'ax1.text(25, 64.0, "EdNet at N=25 is high\\nvariance (std 4.6 pts)", fontsize=12.5, color=SLATE)'),
('ax1.legend(fontsize=13, loc="lower right", frameon=False)',
 'ax1.legend(fontsize=14, loc="lower right", frameon=False)'),
("""    box(bx, cy + 0.004, bw3, 0.070, fc="#F4F2EE", ec=EDGE, r=0.005)
    T(bx + 0.004, cy + 0.0665, h, 14.5, weight="bold", color=RED)
    T(bx + 0.004, cy + 0.0525, W(b, 27), 12, ls=1.18)""",
 """    box(bx, cy + 0.004, bw3, 0.076, fc="#F4F2EE", ec=EDGE, r=0.005)
    T(bx + 0.004, cy + 0.0725, h, 15.5, weight="bold", color=RED)
    T(bx + 0.004, cy + 0.0570, W(b, 24), 13.5, ls=1.18)"""),
# ---------------------------------------------------------------- R2
("""T(cx, ct, W("Knowledge-tracing AUC gain over scratch on the ASSISTments 2017 target. The budget cap "
            "was 3,000 students, and this target has 1,366 training students, so every condition "
            "uses its full split.", 78), 15.5, ls=1.26, color=SLATE)""",
 """T(cx, ct, W("A different question after R1: the task switches to knowledge tracing, the budget is "
            "fixed, and the pretraining source is what varies. AUC gain over scratch on the "
            "ASSISTments 2017 target; budget cap 3,000, this target has 1,366 training students, "
            "so every condition uses its full split.", 60),
  16, ls=1.26, color=SLATE)"""),
('ax2 = mkax([cx + 0.048, cy + 0.136, cw - 0.070, 0.088])',
 'ax2 = mkax([cx + 0.054, cy + 0.146, cw - 0.076, 0.080])'),
('    ax2.text(v + 0.0006, y0, f"+{v:.4f}", fontsize=15, va="center", weight="bold")',
 '    ax2.text(v + 0.0006, y0, f"+{v:.4f}", fontsize=15.5, va="center", weight="bold")'),
('ax2.set_yticks(yp); ax2.set_yticklabels(src, fontsize=14)',
 'ax2.set_yticks(yp); ax2.set_yticklabels(src, fontsize=15)'),
('ax2.set_xticks([0, 0.01, 0.02, 0.03]); ax2.tick_params(axis="x", labelsize=13)',
 'ax2.set_xticks([0, 0.01, 0.02, 0.03]); ax2.tick_params(axis="x", labelsize=14)'),
('ax2.set_xlabel("AUC gain vs scratch", fontsize=14)',
 'ax2.set_xlabel("AUC gain vs scratch", fontsize=15)'),
("""notes = [
    "The EdNet source was best on all 3 targets, including where a smaller in-domain source was available. On the EdNet target the closest match in skill-vocabulary size (ASSISTments) transfers worst, below scratch.",
    "Cross-dataset loading drops the skill embeddings, so this compares corpus scale and granularity, not semantic similarity. Source size and pretraining compute were not independently controlled.",
    "On the large EdNet and Junyi targets gains shrink toward 0 as target data grows; only in-domain EdNet survives (+0.0069 AUC, 6/6 seeds). ASSISTments 2017 is small enough that even its full split is low-resource.",
]
yy = cy + 0.100
for m in notes:
    T(cx + 0.0085, yy, W(m, 76), 14.5, ls=1.24)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.005), 0.0032, 0.0032,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0105 * len(textwrap.wrap(m, 76)) + 0.0045""",
 """notes = [
    "The 442K-student EdNet source was best in all 3 source tests. On the EdNet target the closest skill-vocabulary match (ASSISTments) transfers worst, below scratch.",
    "Cross-dataset loading drops the skill embeddings: this compares corpus scale and granularity, not semantic similarity.",
    "Gains shrink toward 0 as target data grows; only in-domain EdNet survives (+0.0069 AUC, 6/6 seeds). ASSISTments 2017's full split is still low-resource.",
]
yy = cy + 0.108
for m in notes:
    T(cx + 0.0085, yy, W(m, 60), 16, ls=1.24)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0055), 0.0034, 0.0034,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0117 * len(textwrap.wrap(m, 60)) + 0.0048"""),
# ---------------------------------------------------------------- R3
("""T(cx, ct, W("Using EdNet as the fixed pretraining source, three objectives were fine-tuned into every "
            "target, 6 fine-tuning seeds per cell on one fixed checkpoint per objective. Above 0: "
            "skill-only beats correct-only.", 78),
  14, ls=1.26, color=SLATE)""",
 """T(cx, ct, W("EdNet is the fixed pretraining source; three objectives fine-tuned into every target, "
            "6 seeds per cell. Above 0: skill-only beats correct-only. The full objective has the "
            "highest mean on 5 of 7.", 60),
  16, ls=1.26, color=SLATE)"""),
('axs = mkax([cx + 0.036, cy + 0.162, cw - 0.052, 0.170])',
 'axs = mkax([cx + 0.036, cy + 0.154, cw - 0.052, 0.140])'),
('axs.text(8.9, 0.0022, "above 0: skill-only better", fontsize=10.5, color=SLATE, ha="right")',
 'axs.text(8.9, 0.0022, "above 0: skill-only better", fontsize=11.5, color=SLATE, ha="right")'),
('axs.text(8.9, -0.0060, "below 0: correct-only better", fontsize=10.5, color=SLATE, ha="right")',
 'axs.text(8.9, -0.0060, "below 0: correct-only better", fontsize=11.5, color=SLATE, ha="right")'),
('                 fontsize=12.5, color=INK, weight="bold")',
 '                 fontsize=13.5, color=INK, weight="bold")'),
('axs.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"], fontsize=13)',
 'axs.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"], fontsize=13.5)'),
('axs.tick_params(axis="y", labelsize=13)',
 'axs.tick_params(axis="y", labelsize=13.5)'),
('axs.set_xlabel("practice per skill (log scale)", fontsize=14.5)',
 'axs.set_xlabel("practice per skill (log scale)", fontsize=15)'),
('axs.set_ylabel("skill-only minus correct-only\\n(transfer AUC gap)", fontsize=13.5)',
 'axs.set_ylabel("skill-only minus correct-only\\n(transfer AUC gap)", fontsize=14)'),
('axs.text(0.26, 0.0285, "observed regime gap;\\nno dataset sampled here", fontsize=11.5,\n         color=SLATE, ha="center", gid="band_caption")',
 'axs.text(0.26, 0.0285, "observed regime gap;\\nno dataset sampled here", fontsize=12.5,\n         color=SLATE, ha="center", gid="band_caption")'),
('axs.text(0.046, 0.024, "skill-driven", fontsize=12, color=RED, weight="bold")',
 'axs.text(0.046, 0.024, "skill-driven", fontsize=13, color=RED, weight="bold")'),
('axs.text(0.046, -0.018, "correctness-driven", fontsize=12, color=BLUE, weight="bold")',
 'axs.text(0.046, -0.018, "correctness-driven", fontsize=13, color=BLUE, weight="bold")'),
('axf.set_xticklabels(["K = 10\\n(short)", "K = 512\\n(model cap)"], fontsize=12)',
 'axf.set_xticklabels(["K = 10\\n(short)", "K = 512\\n(model cap)"], fontsize=13)'),
('axf.tick_params(axis="y", labelsize=11.5)',
 'axf.tick_params(axis="y", labelsize=12.5)'),
('axf.set_ylabel("skill minus correct", fontsize=11)',
 'axf.set_ylabel("skill minus correct", fontsize=12)'),
('T(cx + 0.120, cy + 0.128, "Within-dataset truncation test", 14, weight="bold", color=RED)',
 'T(cx + 0.120, cy + 0.128, "Within-dataset truncation test", 15.5, weight="bold", color=RED)'),
("""caus = [
    "Retain each learner's most recent K interactions in ASSISTments 2017, holding skills (102) and students (1,708) fixed: the preference reverses, skill-only at K=512, correct-only at K=10.",
    "CIs exclude 0 at both ends. A scratch control run to K=320 changes with K without showing the endpoint reversal, but does not isolate density from the other effects of truncation.",
    "Prospective test passed: Algebra 2006-07, predicted skill-driven from pps 2.41, confirmed at +0.0186, CI [+0.0149, +0.0228], 6/6 seeds.",
]
yy = cy + 0.119
for m in caus:
    T(cx + 0.120, yy, W(m, 40), 12.5, ls=1.24)
    yy -= 0.0089 * len(textwrap.wrap(m, 40)) + 0.0034""",
 """caus = [
    "Keep each learner's most recent K interactions, skills (102) and students (1,708) held fixed: the preference reverses, skill-only at K=512, correct-only at K=10; CIs exclude 0 at both ends. A scratch control run to K=320 changes with K without the reversal.",
    "Prospective test passed: Algebra 2006-07, predicted skill-driven from pps 2.41, confirmed at +0.0186, CI [+0.0149, +0.0228], 6/6 seeds.",
]
yy = cy + 0.116
for m in caus:
    T(cx + 0.120, yy, W(m, 36), 13.5, ls=1.24)
    yy -= 0.0096 * len(textwrap.wrap(m, 36)) + 0.0036"""),
# ---------------------------------------------------------------- R4
('    axp.text(v + 0.0007, y0, f"+{v:.4f}", fontsize=11.5, va="center")',
 '    axp.text(v + 0.0007, y0, f"+{v:.4f}", fontsize=12.5, va="center")'),
('axp.set_yticks(yp); axp.set_yticklabels(pl, fontsize=11.5)',
 'axp.set_yticks(yp); axp.set_yticklabels(pl, fontsize=12.5)'),
('axp.tick_params(axis="x", labelsize=11)',
 'axp.tick_params(axis="x", labelsize=12)'),
('axp.set_xlabel("probe gain: pretrained minus scratch", fontsize=11.5)',
 'axp.set_xlabel("probe gain: pretrained minus scratch", fontsize=12.5)'),
("""T(cx + 0.146, ct, W("Blank out every skill ID, then ask the frozen model to name them. With the "
                    "EdNet-full encoder the probe beats scratch on all 7 targets. On the original "
                    "3 targets, probe scores rank sources the way transfer does (mean within-target "
                    "rank correlation 0.83 vs 0.50 for LogME). Descriptive, and out of distribution.", 30), 12.5, ls=1.26)""",
 """T(cx + 0.146, ct, W("Blank out every skill ID, then ask the frozen model to name them. The probe "
                    "beats scratch on all 7 targets (EdNet-full encoder). On the original 3, probe "
                    "scores rank sources the way transfer does (mean within-target rank correlation "
                    "0.83 vs 0.50 for LogME). Descriptive, and out of distribution.", 28), 13.5, ls=1.26)"""),
# ---------------------------------------------------------------- card 7
("""lim = [
    "Practice per skill orders all 7 datasets, but leave-one-dataset-out prediction holds for only 3 of 6 held-out sets: associated, not established.",
    "PPS is a proxy: skills are defined differently per dataset, and it is measured before the model's 512-step cap, which lowers Bridge 2006 to 1.04 and Algebra 2006-07 to 1.06, so the unsampled interval is 0.33 to 1.04.",
    "The truncation test keeps each learner's most recent K interactions, changing amount, horizon, skill mix and density together; the crossover is not localized. Size, compute and density stay confounded at n = 7.",
    "Scarce-data effect; embedding geometry failed a vocabulary control; disengagement results mixed. 10.1% of next-skill labels are an unlabeled placeholder, but averaging per skill roughly doubles the N=25 gain (+0.18 macro top-1), so it is not a frequent-skill artifact.",
]""",
 """lim = [
    "Practice per skill orders all 7 datasets, but leave-one-dataset-out prediction holds for only 3 of 6 held-out sets: associated, not established. Next step: prospective tests on new mid-density datasets.",
    "PPS is a proxy: skills are defined differently per dataset, and it is measured before the 512-step cap, which lowers Bridge 2006 to 1.04 and Algebra 2006-07 to 1.06; the unsampled interval is 0.33 to 1.04.",
    "The truncation test keeps each learner's most recent K interactions, changing amount, horizon, skill mix and density together; the crossover is not localized. Size, compute and density stay confounded at n = 7.",
    "Scarce-data effect; embedding geometry failed a vocabulary control; disengagement mixed. 10.1% of next-skill labels are a placeholder, but per-skill averaging roughly doubles the N=25 gain (+0.18 macro top-1): not a frequent-skill artifact.",
]"""),
("""yy = ct
for m in lim:
    T(cx + 0.0085, yy, W(m, 74), 13.5, ls=1.20)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0048), 0.003, 0.003,
                                 transform=fig.transFigure, fc=SLATE, ec="none"))
    yy -= 0.0093 * len(textwrap.wrap(m, 74)) + 0.0032
for m in nxt:
    T(cx + 0.0085, yy, W(m, 74), 13.5, ls=1.20)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0048), 0.003, 0.003,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0093 * len(textwrap.wrap(m, 74)) + 0.0032""",
 """yy = ct
for m in lim:
    T(cx + 0.0085, yy, W(m, 70), 14, ls=1.20)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0052), 0.0032, 0.0032,
                                 transform=fig.transFigure, fc=SLATE, ec="none"))
    yy -= 0.0096 * len(textwrap.wrap(m, 70)) + 0.0034
for m in nxt:
    T(cx + 0.0085, yy, W(m, 70), 14, ls=1.20)
    fig.patches.append(Rectangle((cx + 0.0012, yy - 0.0052), 0.0032, 0.0032,
                                 transform=fig.transFigure, fc=RED, ec="none"))
    yy -= 0.0096 * len(textwrap.wrap(m, 70)) + 0.0034"""),
("rh = 0.0220",
 "rh = 0.0214"),
("axf = mkax([cx + 0.032, cy + 0.026, 0.082, 0.094])",
 "axf = mkax([cx + 0.032, cy + 0.024, 0.082, 0.090])"),
("""nxt = [
    "Prospectively test practice-per-skill on new mid-density datasets.",
]""",
 "nxt = []"),
('darrow(0.5, 0.862, 0.5, 0.800, "mask skill and correctness at 15% of steps")',
 'darrow(0.5, 0.862, 0.5, 0.800, "mask skill and correctness\\nat 15% of steps")'),
('    (5.330,  0.0228, "Algebra 2005","s",  1.10,  -0.0018),',
 '    (5.330,  0.0228, "Algebra 2005","s",  0.40,  -0.0052),'),
('    (2.414,  0.0186, "Algebra 06-07","s", -0.90,  0.0038),',
 '    (2.414,  0.0186, "Algebra 06-07","s", -1.35,  0.0042),'),
("if olap(ab, tb, tol=0.0020):",
 "if olap(ab, tb, tol=0.0008):"),
("""# QR placeholder (replace with a real QR code before printing)
qx, qy, qw, qh = 0.892, 0.9040, 0.0430, 0.0645
box(qx, qy, qw, qh, fc="#FBFAF8", ec=FAINT, lw=1.8, r=0.004)
T(qx + qw / 2, qy + qh * 0.62, "QR CODE", 15, color=FAINT, weight="bold", ha="center", va="center")
T(qx + qw / 2, qy + qh * 0.36, "(add before print)", 11.5, color=FAINT, ha="center", va="center")""",
 """# QR code area: image axes created near the end so fig.axes[:6] stays the charts
qx, qy, qw, qh = 0.892, 0.9040, 0.0430, 0.0645
box(qx, qy, qw, qh, fc="white", ec=FAINT, lw=1.8, r=0.004)
import os as _os
HAVE_QR = _os.path.exists("qr_code.png")
if not HAVE_QR:
    T(qx + qw / 2, qy + qh * 0.62, "QR CODE", 15, color=FAINT, weight="bold", ha="center", va="center")
    T(qx + qw / 2, qy + qh * 0.36, "(add before print)", 11.5, color=FAINT, ha="center", va="center")
    print("WARNING: poster/qr_code.png missing, rendering placeholder")"""),
("""# ---------------------------------------------------------------- QA pass
# Catch text or chart artists that spill outside the card they belong to.""",
 """# ---------------------------------------------------------------- QR image
# Created after all chart axes so name-to-axes pairings stay intact.
QR_AX = None
if HAVE_QR:
    _qr_w = 0.037
    _qr_h = _qr_w * FW / FH
    QR_AX = fig.add_axes([qx + (qw - _qr_w) / 2, qy + (qh - _qr_h) / 2, _qr_w, _qr_h], zorder=6)
    QR_AX.imshow(plt.imread("qr_code.png"), cmap="gray", interpolation="nearest", aspect="auto")
    QR_AX.axis("off")

# ---------------------------------------------------------------- QA pass
# Catch text or chart artists that spill outside the card they belong to."""),
("""for a in fig.axes:
    b = fbox(a, tight=True)
    if not any(inside(b, r, pad=0.003) for r in CARDS):""",
 """for a in fig.axes:
    b = fbox(a, tight=True)
    if b[1] > 0.802:
        continue
    if not any(inside(b, r, pad=0.003) for r in CARDS):"""),
("""    json.dump(dict(fig=[FW, FH], paper=hx(PAPER), shapes=shapes, texts=texts,
                   charts=charts), open("poster_layout.json", "w"), indent=1)""",
 """    if QR_AX is not None:
        qp = QR_AX.get_position()
        charts.append(dict(name="qr_code", x=qp.x0, y=qp.y1, w=qp.x1 - qp.x0, h=qp.y1 - qp.y0))

    json.dump(dict(fig=[FW, FH], paper=hx(PAPER), shapes=shapes, texts=texts,
                   charts=charts), open("poster_layout.json", "w"), indent=1)"""),
# ---------------------------------------------------------------- mirror of the
# manual Keynote edits, so a rebuild from source reproduces the printed file.
("""T(0.0475, 0.9445, "Northeastern University", 20.5, weight="bold", va="center")
T(0.0475, 0.9165, "Khoury College of Computer Sciences", 16.5, color=SLATE, va="center")""",
 """T(0.0475, 0.9300, "Northeastern University", 20.5, weight="bold", va="center")"""),
("""T(0.500, 0.9030, "Hanyu Dai   \u00b7   Supervised by Prof. Hazra Imran", 21, ha="center")
T(0.500, 0.8830, "CS7980 Research Capstone   \u00b7   Khoury College of Computer Sciences, Northeastern University Vancouver   \u00b7   Summer 2026",
  17.5, color=SLATE, ha="center")""",
 """# Byline and affiliation share one row. AUTHOR_IN_HEADER=False reproduces the
# printed Keynote header exactly; True is the default, the presenter's name
# belongs near the title.
AUTHOR_IN_HEADER = True
HDRY, HDRGAP = 0.8992, 0.0155
_byline = ("Hanyu Dai   \u00b7   Supervised by Prof. Hazra Imran" if AUTHOR_IN_HEADER
           else "Supervised by Prof. Hazra Imran")
_affil = "Khoury College of Computer Sciences, Northeastern University Vancouver   \u00b7   Summer 2026"
_t1 = fig.text(0.500 - HDRGAP, HDRY, _byline, fontsize=21, color=INK,
               ha="right", va="baseline", zorder=5)
_t2 = fig.text(0.500 + HDRGAP, HDRY, _affil, fontsize=17.5, color=SLATE,
               ha="left", va="baseline", zorder=5)
# centre the pair as a unit so the row aligns with the title above it
_r0 = fig.canvas.get_renderer()
_i0 = fig.transFigure.inverted()
_bb1, _bb2 = _t1.get_window_extent(_r0), _t2.get_window_extent(_r0)
_sh = 0.500 - (_i0.transform([[_bb1.x0, 0]])[0][0]
               + _i0.transform([[_bb2.x1, 0]])[0][0]) / 2
_t1.set_x(0.500 - HDRGAP + _sh)
_t2.set_x(0.500 + HDRGAP + _sh)"""),
('T(qx + qw / 2, qy - 0.0215, "github.com/HY-D1/studentbert", 13, color=SLATE, ha="center")',
 '# printed link under the QR removed: the code itself carries the URL'),
("""box(MARG, 0.010, 1 - 2 * MARG, 0.033, fc="#EFEDE8", ec=EDGE, r=0.005)
T(MARG + 0.006, 0.0265, "Hanyu Dai  \u00b7  dai.hany@northeastern.edu  \u00b7  github.com/HY-D1/studentbert",
  15, va="center", weight="bold")
T(0.5, 0.0265, "Khoury College of Computer Sciences, Northeastern University Vancouver  \u00b7  August 2026 Showcase",
  14.5, va="center", ha="center", color=SLATE)
T(1 - MARG - 0.006, 0.0265, "Computations: Northeastern Explorer HPC  \u00b7  Tracking: Weights & Biases",
  14, va="center", ha="right", color=SLATE)""",
 """# Footer band and the centre/right credits removed; the contact line sits
# directly above the bottom accent rule.
T(MARG + 0.006, 0.0265, "Hanyu Dai  \u00b7  dai.hany@northeastern.edu  \u00b7  github.com/HY-D1/studentbert",
  15, va="center", weight="bold")"""),
]

OPTIONAL_PAIRS = [
("axs = mkax([cx + 0.036, cy + 0.148, cw - 0.052, 0.146])",
 "axs = mkax([cx + 0.036, cy + 0.154, cw - 0.052, 0.140])"),
("""    "Those models are data-hungry, yet every new course or tool starts with only dozens of students.",
    "In NLP and vision, pretrain-then-fine-tune is the standard fix. For student data there is no recipe.",""",
 """    "Those models are data-hungry, yet every new course or tool starts with only dozens of students.",
    "Cold-start learners get the least personalized support, and small classrooms may never reach big-data scale.",
    "In NLP and vision, pretrain-then-fine-tune is the standard fix. For student data there is no recipe.","""),
("stack(0, [0.128, 0.152, 0.440])", "stack(0, [0.160, 0.152, 0.408])"),
("stack(2, [0.424, 0.308])", "stack(2, [0.404, 0.328])"),
("ax2 = mkax([cx + 0.054, cy + 0.150, cw - 0.076, 0.082])",
 "ax2 = mkax([cx + 0.054, cy + 0.146, cw - 0.076, 0.080])"),
("yy = cy + 0.112", "yy = cy + 0.108"),
("""T(cx, ct, W("Knowledge-tracing AUC gain over scratch on the ASSISTments 2017 target. Budget cap "
            "3,000 students; this target has 1,366, so every condition uses its full split.", 62),
  16, ls=1.26, color=SLATE)""",
 """T(cx, ct, W("A different question after R1: the task switches to knowledge tracing, the budget is "
            "fixed, and the pretraining source is what varies. AUC gain over scratch on the "
            "ASSISTments 2017 target; budget cap 3,000, this target has 1,366 training students, "
            "so every condition uses its full split.", 60),
  16, ls=1.26, color=SLATE)"""),
]

REQUIRED_AFTER = [
    "PPS proxy", "model cap", "442K-student EdNet source was best in all 3 source tests",
    "Prospective test passed", "+0.0186, CI [+0.0149, +0.0228], 6/6 seeds",
    "0.83 vs 0.50 for LogME", "+0.0069 AUC, 6/6 seeds", "+0.092 top-1",
    "(+0.18 macro top-1)", "leave-one-dataset-out prediction holds for only 3 of 6",
    "scratch control run to K=320", "0.693 vs 0.670",
    "A 2015 DKT still wins the 4 datasets added later",
    "Size, compute and density stay confounded at n = 7",
    "Supervised by Prof. Hazra Imran",
    "Northeastern University Vancouver   \u00b7   Summer 2026",
    "dai.hany@northeastern.edu",
]
FORBIDDEN_AFTER = ["\u2014", "Shown causally", "full length", "Four papers",
                   "crossover sits near", "40,000",
                   "CS7980 Research Capstone", "August 2026 Showcase",
                   "Computations: Northeastern Explorer HPC"]


def decide(src, old, new):
    """'apply', 'skip', or a conflict string.

    Insertion pairs keep the anchor, so old is a substring of new and a naive
    'old in src' test fires again after the edit and duplicates the block.
    When both strings are present, the containment direction settles it.
    """
    has_old, has_new = old in src, new in src
    if has_old and has_new:
        if old in new:
            return "skip"          # anchor kept: new present means already done
        if new in old:
            return "apply"         # deletion: old present means still pending
        return f"AMBIGUOUS (old and new both present): {old[:60]!r}"
    if has_new:
        return "skip"
    if not has_old:
        return f"NOT FOUND (local edit here?): {old[:70]!r}"
    return "apply"


def main():
    src = open(PATH, encoding="utf-8").read()
    applied = skipped = optional = 0
    conflicts = []
    for old, new in OPTIONAL_PAIRS:
        if decide(src, old, new) == "apply" and src.count(old) == 1:
            src = src.replace(old, new, 1)
            optional += 1
    for old, new in PAIRS:
        if new.startswith("PLACEHOLDER"):
            continue
        verdict = decide(src, old, new)
        if verdict == "skip":
            skipped += 1
        elif verdict == "apply":
            if src.count(old) != 1:
                conflicts.append(f"NON-UNIQUE ({src.count(old)}x): {old[:70]!r}")
                continue
            src = src.replace(old, new, 1)
            applied += 1
        else:
            conflicts.append(verdict)
    open(PATH, "w", encoding="utf-8").write(src)
    ast.parse(src)
    missing = [r for r in REQUIRED_AFTER if r not in src]
    present = [f for f in FORBIDDEN_AFTER if f in src]
    print(f"applied {applied}, already-applied {skipped}, optional {optional}, conflicts {len(conflicts)}")
    for c in conflicts:
        print("  CONFLICT:", c)
    if missing:
        print("  MISSING REQUIRED:", missing)
    if present:
        print("  FORBIDDEN PRESENT:", present)
    if conflicts or missing or present:
        sys.exit(1)
    print("ast.parse OK; all required statements present; no forbidden phrasings")


if __name__ == "__main__":
    main()
