# Paper C: trajectories, compute, Quick-Tune, multi-fidelity and selective trust

Generated from RESULTS.md (md5 99ed91735c97c5ead1002b774d9c7a0c) by tools/split_results_by_paper.py. Do not edit: RESULTS.md stays the record, and every section below is copied from it unchanged. Sections, with the md5 of their text:

- 12.22: 6392724f81af44ab39099997ca3282d3
- 12.23: 93b91e8062ad824a3ccfde33ef9b5980
- 12.24: 79b884997aa661cc928c29d23e7e2f60
- 12.25: 8dc9e8ad9a1fc7c97351adfed5a7e6e4
- 12.26: c693e05224dcea69825a0c916412b632
- 12.27: e3a818fe2163f1a47cd79f5995138cb2

---

### 12.22 Measured cost of fully fine-tuning every StudentBERT candidate (analysis/kt_finetune_cost.py, commit 4c8e75c, job 11631444)

Task: Prof. Hazra (Nov 2026) asked for the measured cost of fully fine-tuning all candidates on one
KT target, toward a regret-versus-compute analysis. Source: the runs behind the Track B gold in
executions.tsv (task kt, valid for the primary analysis, 20 epochs, 3,000 target learners or the
whole training split). Every target, candidate and seed was run twice (campaigns TG1 and N3000P); a
rerun is not part of a method's cost, so each counts once, at the median elapsed time of its copies.
Elapsed is the scheduler's wall-clock time for a one-GPU job, including start-up, data loading and
evaluation, so it is an upper bound on training time.

Files (md5): kt_finetune_cost.py f0470c7cd0b60990e8955bb9439d846d, executions.tsv
92a3e5250b50727f911fe6f09d4b225b, kt_finetune_cost_report.md 19400b9033fa169bb7f78d1269d624f2,
kt_finetune_cost.tsv 8d48cdbf826faab22a1cccce4fd81d89.

Report as written by analysis/kt_finetune_cost.py (heading lowered two levels):


Source: executions.tsv (md5 92a3e5250b50727f911fe6f09d4b225b). One run per target, candidate and seed (20 epochs, 3,000 target learners or the whole training split), at the median elapsed time of its copies; wall-clock of a one-GPU job.

| target | fine-tunes | all candidates x 6 seeds (GPU-h) | one seed, all 8 candidates (GPU-h) | minutes per fine-tune | GPU | copies per run |
|---|---|---|---|---|---|---|
| algebra2005 | 48 | 0.71 | 0.12 | 0.8 to 1.0 | v100-sxm2 | 2 |
| algebra2006 | 48 | 1.34 | 0.22 | 1.5 to 1.8 | v100-sxm2 | 2 |
| assist2009 | 48 | 2.70 | 0.45 | 3.2 to 3.7 | v100-sxm2 | 2 |
| assist2017 | 48 | 1.53 | 0.26 | 1.8 to 2.3 | v100-sxm2 | 2 |
| bridge2006 | 48 | 1.13 | 0.19 | 1.3 to 1.6 | v100-sxm2 | 2 |
| ednet | 48 | 19.83 | 3.30 | 23.3 to 35.9 | v100-sxm2 | 2 |
| junyi | 48 | 5.56 | 0.93 | 6.5 to 9.5 | v100-sxm2 | 2 |

Whole grid (7 targets x 8 candidates x 6 seeds): 32.8 GPU-hours.

Files: kt_finetune_cost.tsv 8d48cdbf826faab22a1cccce4fd81d89

_Read: fully fine-tuning all 8 candidates once, at one seed, costs 0.12 to 0.45 GPU-hours on the
five smaller targets, 0.93 on Junyi Academy and 3.30 on EdNet, all on V100-SXM2; the whole 7 by 8 by
6 grid took 32.8 GPU-hours. Candidates cost about the same within a target, so the target's size
sets the cost. With 20 epochs of similar length, one epoch is roughly a twentieth of a full
fine-tune, an approximation until per-epoch times are recorded._
### 12.23 Per-epoch trajectories and compute for every candidate (analysis/trajectory_table.py, commit 9302502, job 11634489)

Task: Prof. Hazra (Nov 2026) asked to keep, for every check, each candidate's per-epoch ranking
trajectory and compute cost. One table holds, per candidate per epoch, the validation score, the
rank among the candidates of its ranking and the epoch and cumulative seconds: StudentBERT (7
targets x 8 candidates x 6 seeds, epochs 1 to 20; cost the measured run time spread evenly over its
epochs), the NLP short fine-tunes (three seed sets, epochs 1 to 3; cost measured per epoch) and
Quick-Tune (Pineda Arango et al., ICLR 2024; epochs 1 to 50; cost the recorded training and
evaluation time per epoch). The design was committed before the table was built.

Quick-Tune rule A (Harry's choice, approved by Prof. Hazra): every Quick-Tune run uses its own
random hyperparameters, and no configuration is shared across models on any task
(analysis/qt_inspect.py), so each model on each task is represented by its first complete 50-epoch
run in file order, a rule that uses no result. Limitation: the models are therefore not compared
under identical hyperparameters, and a model can rank low because its run drew poor settings rather
than because it transfers worse. This limitation is stated wherever Quick-Tune results are used.

Files (md5): trajectory_design.md 9da80165f872ca8e572a0782afe474eb, trajectory_table.py
5040a7fe5ea64f355ac875ed391cd88e, qt_inspect.py c811fac46ece2136aace4cff051c6ea8,
trajectories_report.md 5b0d4c16558db76ddc6f32d0e35f2b52, trajectories.tsv
e93329cbc69ae8a3af843c544d43793b, early_ft_epochs.tsv 602d8f8e997384835453edcc45e8e6da,
executions.tsv 92a3e5250b50727f911fe6f09d4b225b, micro.zip ed189adccce30d8b7f4a6e52b8f0caae,
mini.zip 0560bb028c2ea3c03cfb11165e9a6d87, extended.zip 7619c2a3c6d5e13b410037037bfe577a.

Report as written by analysis/trajectory_table.py:


Definitions in docs/trajectory_design.md. Rows per benchmark: nlp_early 1890, quicktune_extended 24250, quicktune_micro 33700, quicktune_mini 31900, studentbert 6720.

- studentbert: 336 runs from early_ft_epochs.tsv (md5 602d8f8e997384835453edcc45e8e6da), cost from executions.tsv (md5 92a3e5250b50727f911fe6f09d4b225b)
- nlp_early: 210 runs from /projects/algl/dai.hany/task4/early_ft
- nlp_early: 210 runs from /projects/algl/dai.hany/task4/early_ft_fresh
- nlp_early: 210 runs from /projects/algl/dai.hany/task4/early_ft_p3
- quicktune_micro: rule A chose 674 runs on 30 tasks (models per task min 20, max 24; all 24 models on 4 tasks); runs shorter than 50 epochs not used: 2563; micro.zip md5 ed189adccce30d8b7f4a6e52b8f0caae
- quicktune_mini: rule A chose 638 runs on 30 tasks (models per task min 12, max 24; all 24 models on 11 tasks); runs shorter than 50 epochs not used: 2035; mini.zip md5 0560bb028c2ea3c03cfb11165e9a6d87
- quicktune_extended: rule A chose 485 runs on 26 tasks (models per task min 6, max 23; all 24 models on 0 tasks); runs shorter than 50 epochs not used: 2146; extended.zip md5 7619c2a3c6d5e13b410037037bfe577a

Files: trajectories.tsv e93329cbc69ae8a3af843c544d43793b

_Read: rule A keeps 20 to 24 models on every micro task, 12 to 24 on mini and 6 to 23 on extended,
where many runs end before 50 epochs; all 24 models have a complete run on 4 micro and 11 mini tasks
and on no extended task. Candidate sets therefore differ by task, and every Quick-Tune regret is
judged within its task's own set._
### 12.24 Trajectory and compute analysis: StudentBERT and Quick-Tune (analysis/trajectory_analysis.py, commit 1790a07, job 11762159)

Task: Prof. Hazra (Nov 2026) asked for the pre-specified trajectory and compute analysis on
StudentBERT and Quick-Tune. Definitions: docs/trajectory_design.md and
docs/trajectory_amendment1.md, committed with the script before the run; the script reproduced the
warning-rule and early-leader results of 12.17, 12.18 and 12.20 at epochs 1 to 3 before reporting.
The analysis is descriptive: the StudentBERT outcomes were seen before, and the script was test-run
on the StudentBERT scores with stand-in costs while it was written. Quick-Tune uses rule A (12.23),
so its models are not compared under identical hyperparameters.

Files (md5): trajectory_design.md 9da80165f872ca8e572a0782afe474eb, trajectory_amendment1.md
9fb6a874c0d025e0678735381a26f5ce, trajectory_analysis.py 4e524cb49c5da4fead0aea6ffa0417ac,
trajectory_analysis_report.md 0e04cb3dccee6bf136ac3ee8a376c763, trajectory_analysis.tsv
b6fb4d2fb7d63ce1b2aa8ee7f57e5a85, trajectories.tsv e93329cbc69ae8a3af843c544d43793b,
reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7.

Report as written by analysis/trajectory_analysis.py (headings lowered two levels):


Pre-specified in docs/trajectory_design.md and docs/trajectory_amendment1.md; descriptive (StudentBERT results were seen before). Inputs: trajectories.tsv e93329cbc69ae8a3af843c544d43793b, reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7. Compute: StudentBERT per-epoch costs are measured run times spread evenly over epochs; Quick-Tune costs are the recorded per-epoch times. Quick-Tune models are not compared under identical hyperparameters (rule A).

#### StudentBERT (21 recommendations, retrospective)

| budget (epochs) | compute per decision (GPU-h) | early leader regret | followed by the rule | regret of followed | overtaken among followed |
|---|---|---|---|---|---|
| 0 (H-score alone) | 0 | 0.0012 | 21/21 | 0.0012 | 7 |
| 1 | 0.039 | 0.0031 | 17/21 | 0.0009 | 6 |
| 2 | 0.078 | 0.0010 | 12/21 | 0.0003 | 1 |
| 3 | 0.118 | 0.0006 | 12/21 | 0.0009 | 1 |
| 5 | 0.196 | 0.0006 | 10/21 | 0.0010 | 1 |
| 10 | 0.392 | 0.0021 | 5/21 | 0.0007 | 2 |
| 20 | 0.784 | 0.0017 | 10/21 | 0.0009 | 4 |

- overtaken (7): the decision settles at epochs [7, 14, 20]; never within 20 epochs: 4. Compute at settling, as a share of fine-tuning all 8 candidates for 20 epochs: median 0.70, range 0.35 to 1.00.
- not overtaken (14): the decision settles at epochs [3, 7, 10, 11, 12, 14]; never within 20 epochs: 8. Compute at settling, as a share of fine-tuning all 8 candidates for 20 epochs: median 0.52, range 0.15 to 0.70.

Settling per recommendation (target, seed, overtaken, epoch): algebra2005 1 no 3; algebra2005 2 no 7; algebra2005 42 yes 7; algebra2006 1 yes never; algebra2006 2 yes never; algebra2006 42 yes never; assist2009 1 no never; assist2009 2 no never; assist2009 42 no never; assist2017 1 yes never; assist2017 2 yes 20; assist2017 42 yes 14; bridge2006 1 no 14; bridge2006 2 no never; bridge2006 42 no never; ednet 1 no 10; ednet 2 no 11; ednet 42 no 12; junyi 1 no never; junyi 2 no never; junyi 42 no never.

#### Quick-Tune micro (30 tasks, 20 to 24 models per task; rule A)

| budget (epochs) | compute per decision (GPU-h) | early leader regret (points) | early leader tied with the best |
|---|---|---|---|
| 1 | 0.13 | 20.69 | 4/30 |
| 2 | 0.26 | 17.11 | 10/30 |
| 3 | 0.38 | 9.74 | 11/30 |
| 5 | 0.64 | 5.58 | 17/30 |
| 10 | 1.28 | 4.22 | 18/30 |
| 20 | 2.57 | 1.57 | 21/30 |
| 30 | 3.90 | 0.46 | 22/30 |
| 40 | 5.26 | 0.02 | 29/30 |
| 50 | 6.62 | 0.00 | 30/30 |

- Winner detection: settles by epoch 1 on 1, by 3 on 4, by 5 on 5, by 10 on 10 and by 20 on 13 of 30 tasks; median settling epoch 26; compute at settling as a share of running all candidates for 50 epochs: median 0.49.

#### Quick-Tune mini (30 tasks, 12 to 24 models per task; rule A)

| budget (epochs) | compute per decision (GPU-h) | early leader regret (points) | early leader tied with the best |
|---|---|---|---|
| 1 | 0.39 | 18.06 | 5/30 |
| 2 | 0.78 | 14.80 | 8/30 |
| 3 | 1.18 | 8.25 | 13/30 |
| 5 | 1.96 | 6.96 | 14/30 |
| 10 | 3.93 | 1.69 | 19/30 |
| 20 | 7.86 | 1.31 | 24/30 |
| 30 | 11.92 | 0.25 | 27/30 |
| 40 | 16.13 | 0.07 | 29/30 |
| 50 | 20.33 | 0.00 | 30/30 |

- Winner detection: settles by epoch 1 on 2, by 3 on 6, by 5 on 8, by 10 on 15 and by 20 on 18 of 30 tasks; median settling epoch 12; compute at settling as a share of running all candidates for 50 epochs: median 0.25.

#### Quick-Tune extended (26 tasks, 6 to 23 models per task; rule A)

| budget (epochs) | compute per decision (GPU-h) | early leader regret (points) | early leader tied with the best |
|---|---|---|---|
| 1 | 1.57 | 9.56 | 7/26 |
| 2 | 3.14 | 13.61 | 11/26 |
| 3 | 4.71 | 13.89 | 11/26 |
| 5 | 7.85 | 5.53 | 11/26 |
| 10 | 15.69 | 1.59 | 18/26 |
| 20 | 31.38 | 0.56 | 20/26 |
| 30 | 47.06 | 0.27 | 22/26 |
| 40 | 62.73 | 0.20 | 24/26 |
| 50 | 78.39 | 0.00 | 26/26 |

- Winner detection: settles by epoch 1 on 4, by 3 on 9, by 5 on 9, by 10 on 14 and by 20 on 19 of 26 tasks; median settling epoch 10; compute at settling as a share of running all candidates for 50 epochs: median 0.20.

Files: trajectory_analysis.tsv b6fb4d2fb7d63ce1b2aa8ee7f57e5a85

_Read: on StudentBERT (retrospective; costs are run times spread evenly over epochs), fine-tuning
all 8 candidates at one seed costs 0.039 GPU-hours per epoch and 0.78 for all 20. After 2 or 3
epochs, 10 to 15% of that, the warning rule follows 12 of 21 recommendations with mean regret 0.0003
and 0.0009 AUC, and after 3 to 5 epochs the early leader halves H-score's regret (0.0006 against
0.0012). A longer budget does not help: the early leader's regret rises to 0.0021 after 10 epochs
and 0.0017 after 20, worse than H-score alone, which suggests that one seed's validation ranking
late in training is a noisy guide to the six-seed test gold. The strict detection rule accordingly
settles on only 3 of the 7 overtaken recommendations within 20 epochs; on Algebra 2006 and one
ASSISTments 2017 seed the own encoder regains the validation lead late. On Quick-Tune, early leaders
are poor: their mean regret is 21, 18 and 10 points after one epoch on micro, mini and extended, 5.5
to 7.0 after five, and 0.6 to 1.6 after twenty. The eventual winner is the stable early leader by
epoch 10 on 10 of 30 micro, 15 of 30 mini and 14 of 26 extended tasks, at a median of 49%, 25% and
20% of the compute of running every candidate for 50 epochs. The epoch-50 score defines the final
ranking, so regret there is zero by construction. H-score's choice does not yet exist on Quick-Tune,
so these numbers describe how fast the eventual winner emerges, not when an H-score reversal becomes
detectable._
### 12.25 H-score on Quick-Tune micro (analysis/qt_reversal.py, commit c930810, job 11808894; post-hoc diagnostic analysis/qt_diagnostics.py, commit e748869, job 11809464)

Task: Prof. Hazra (Nov 2026) asked whether H-score can be computed on the Quick-Tune tasks and
whether their curves can show when a reversal becomes detectable. Design: docs/qt_hscore_design.md
and docs/trajectory_amendment2.md, both committed before any feature or score existed. The 30
Meta-Album micro tasks (40 images per class, 20 classes, 19 on DOG and PNU) were fetched from
OpenML; frozen features of the 24 models were extracted on an H200 with each model's own
preprocessing; H-score, shrunk H-score and LogME (the NLP estimator code) scored each task's rule A
candidates on stratified halves at seeds 42, 1 and 2. The recommendation, fixed in advance, is the
shrunk H-score's choice; plain H-score and LogME are secondary rows. Limitations: rule A (12.23;
models are not compared under identical hyperparameters) and the image split (the estimators may see
images on which Quick-Tune's curves were validated).

Files (md5): qt_hscore_design.md 9131a8e9221faf387cb4ff3c9d87fcf1, trajectory_amendment2.md
8859f77870d589845fc5d373a6cae773, qt_fetch_metaalbum.py 0850a7b55419bc5acd752956d136287f,
qt_extract_features.py 49f0d1fcdba7442465f3a306fb2b3f2b, qt_score.py
3ace18aa5aee44106002d00f7a78defb, qt_reversal.py 1877ad66cf8d54f28fddf358aa9c2791, qt_diagnostics.py
8bb9857c7a93b2825c300085d902a585, trajectories.tsv e93329cbc69ae8a3af843c544d43793b,
qt_reversal_report.md 5b6c386718ae4818c81be9eb4410bf48, qt_reversal.tsv
9fc1b8f63f4978e6b9728edc2d02a1cf, qt_diagnostics_11809464.log 2507c1726127b85679ed7d9729214224,
metaalbum_micro_report.json 1700e621ccce6d64692db92068378006, extract_log.jsonl
50ea8a9ba8fd20713c25211fd30c6f8e, qt_scores.jsonl b7f48f2f5c6ee9c45ce17201a6d2f9ea.

Pre-specified result, as written by analysis/qt_reversal.py (headings lowered two levels):


Pre-specified in docs/qt_hscore_design.md and docs/trajectory_amendment2.md. Inputs: trajectories.tsv e93329cbc69ae8a3af843c544d43793b, qt_scores.jsonl b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl 50ea8a9ba8fd20713c25211fd30c6f8e. Limitations: rule A (models not compared under identical hyperparameters) and the image split (the estimators may see images on which Quick-Tune's curves were validated).

#### Estimators against the epoch-50 ranking (30 tasks x 3 seeds)

| estimator | candidates scored | rankings | mean Spearman | top-1 tied with the best | mean regret (points) |
|---|---|---|---|---|---|
| hscore_shrunk | 2022/2022 | 90 | +0.11 | 5/90 | 56.54 |
| hscore | 2022/2022 | 90 | +0.17 | 4/90 | 34.46 |
| logme | 1146/2022 | 90 | +0.05 | 0/90 | 60.09 |

Undefined scores (features interpolate the labels exactly), by estimator: logme 876. Where an estimator is undefined for some candidates, its ranking covers the rest.

#### Warning rule on shrunk H-score's choice (90 recommendations, 85 overtaken)

| budget (epochs) | compute per decision (GPU-h) | caught | false alarms | followed | regret of followed (points) |
|---|---|---|---|---|---|
| 0 (H-score alone) | 0 | 0/85 | 0/5 | 90/90 | 56.54 |
| 1 | 0.13 | 82/85 | 5/5 | 3/90 | 25.62 |
| 2 | 0.26 | 82/85 | 5/5 | 3/90 | 25.62 |
| 3 | 0.38 | 82/85 | 5/5 | 3/90 | 25.62 |
| 5 | 0.64 | 82/85 | 1/5 | 7/90 | 1.34 |
| 10 | 1.28 | 81/85 | 1/5 | 8/90 | 1.83 |
| 20 | 2.57 | 85/85 | 1/5 | 4/90 | 0.00 |
| 30 | 3.90 | 82/85 | 1/5 | 7/90 | 1.34 |
| 40 | 5.26 | 85/85 | 0/5 | 5/90 | 0.00 |
| 50 | 6.62 | 85/85 | 0/5 | 5/90 | 0.00 |

- overtaken (85): the decision settles by epoch 1 on 75, by 3 on 75, by 10 on 78, by 50 on 85; median settling epoch 1; compute at settling as a share of 50 epochs for all candidates: median 0.02.

- not overtaken (5): the decision settles by epoch 1 on 0, by 3 on 0, by 10 on 4, by 50 on 5; median settling epoch 5; compute at settling as a share of 50 epochs for all candidates: median 0.10.

Frozen-feature extraction for a task's candidates (all 800 or 760 images): mean 2.9 GPU-minutes on NVIDIA H200; the fine-tuning compute above is Quick-Tune's recorded time on its own, unrecorded hardware, so the two are reported side by side and not added.

Files: qt_reversal.tsv 9fc1b8f63f4978e6b9728edc2d02a1cf

Post-hoc diagnostic, added after the result above to explain it; it changes no rule and no result (output of analysis/qt_diagnostics.py, job 11809464):

```
Rule A runs: 674; ending near chance (top-1 <= 10 at epoch 50): 257
Near-chance share and mean final top-1 by model (dimension):
  xcit_nano_12_p8_384_dist (128): 23/30 near chance, mean final 16.3
  edgenext_xx_small (168): 9/30 near chance, mean final 32.9
  edgenext_x_small (192): 13/30 near chance, mean final 33.5
  xcit_tiny_12_p8_384_dist (192): 16/30 near chance, mean final 22.2
  xcit_tiny_24_p8_384_dist (192): 13/30 near chance, mean final 29.9
  dla46x_c (256): 9/30 near chance, mean final 25.7
  edgenext_small (304): 11/30 near chance, mean final 35.1
  mobilevit_xxs (320): 13/30 near chance, mean final 22.6
  deit3_small_patch16_384_in21ft1k (384): 16/30 near chance, mean final 27.5
  mobilevit_xs (384): 11/30 near chance, mean final 21.6
  mobilevitv2_075 (384): 13/30 near chance, mean final 24.3
  volo_d1_384 (384): 3/30 near chance, mean final 46.9
  xcit_small_12_p8_384_dist (384): 13/30 near chance, mean final 33.1
  volo_d3_448 (512): 8/30 near chance, mean final 47.5
  beit_base_patch16_384 (768): 18/30 near chance, mean final 21.7
  convnext_small_384_in22ft1k (768): 4/30 near chance, mean final 55.8
  volo_d4_448 (768): 4/26 near chance, mean final 51.1
  volo_d5_448 (768): 1/23 near chance, mean final 56.3
  volo_d5_512 (768): 1/5 near chance, mean final 43.1
  beit_large_patch16_512 (1024): 16/20 near chance, mean final 7.8
  swinv2_base_window12to24_192to384_22kft1k (1024): 7/30 near chance, mean final 52.7
  tf_efficientnet_b4_ns (1792): 15/30 near chance, mean final 36.0
  tf_efficientnet_b6_ns (2304): 14/30 near chance, mean final 25.4
  tf_efficientnet_b7_ns (2560): 6/30 near chance, mean final 39.2
Mean Spearman of final top-1 with feature dimension, per task: +0.12
Mean regret of a random candidate: 50.88 points
hscore_shrunk: picks [('tf_efficientnet_b6_ns', 27), ('beit_large_patch16_512', 25), ('tf_efficientnet_b7_ns', 16), ('swinv2_base_window12to24_192to384_22kft1k', 14), ('beit_base_patch16_384', 6), ('tf_efficientnet_b4_ns', 2)]; mean final of picks 27.7; picks near chance 39/90; mean Spearman of score with dimension +0.93
hscore: picks [('volo_d3_448', 47), ('volo_d4_448', 15), ('volo_d5_448', 10), ('volo_d5_512', 5), ('convnext_small_384_in22ft1k', 3), ('volo_d1_384', 2)]; mean final of picks 49.7; picks near chance 15/90; mean Spearman of score with dimension +0.88
logme: picks [('deit3_small_patch16_384_in21ft1k', 49), ('edgenext_small', 20), ('xcit_small_12_p8_384_dist', 13), ('xcit_tiny_24_p8_384_dist', 5), ('volo_d1_384', 3)]; mean final of picks 24.1; picks near chance 51/90; mean Spearman of score with dimension +0.44
Mean best final top-1 per task: 84.2
```

_Read: under the pre-specified rule, shrunk H-score's choice is overtaken on 85 of 90
recommendations, with a mean regret of 56.5 points, more than the 50.9 of a random candidate. The
post-hoc diagnostic shows two causes. First, 257 of the 674 rule A runs (38%) end near chance (top-1
at most 10 points at epoch 50), among them 16 of the 20 BEiT-large runs: under rule A a model's
final accuracy mostly reflects whether its one sampled configuration worked, and it barely relates
to feature dimension (mean Spearman +0.12). Second, with 400 images (380 on DOG and PNU) H-score
largely tracks feature dimension (mean Spearman of score with dimension +0.93 for shrunk and +0.88
for plain H-score): shrunk H-score picks EfficientNet-B6, BEiT-large, EfficientNet-B7 or SwinV2 on
82 of 90 recommendations. Plain H-score picks a VOLO model on at least 79 of 90 and does better
(mean regret 34.5); the pre-specified primary is not changed after the fact. LogME is undefined for
876 of the 2,022 candidate scores, where the feature dimension is at least the number of images.
Quick-Tune under rule A therefore cannot test whether an estimator's choice is overtaken after
fine-tuning, and the warning and detection rows above are not interpreted. Another representation of
each model, such as its best final accuracy over all its complete runs, would need its own
pre-specified analysis._
### 12.26 Quick-Tune micro: model-level outcome and dimension control (analysis/qt_model_level.py, commit ef899da, job 11851877)

Task: Prof. Hazra's reply to 12.25 (Nov 2026) asked for a model-level outcome that aggregates each
model's complete runs, with best-run accuracy only as a sensitivity analysis, since the best single
run rewards lucky hyperparameters and drifts toward selecting a model together with its tuning; and
for a check of the conclusions after controlling for representation dimensionality.
docs/trajectory_amendment3.md fixed both before the run. Outcome: each model's median (primary),
mean or best final top-1 over all its complete 50-epoch runs on a task, for models with at least 3
such runs. Scores: the 12.25 scores (raw), and the same estimators on each model's features
projected on their top 128 principal components, computed from the scored images without labels
(pca128). The recommendation is again the shrunk H-score's choice. This is a follow-up to 12.25, not
a replacement; rule A is not used here, and the outcome instead averages over Quick-Tune's sampled
hyperparameters. Remaining limitation: the estimators may see images on which Quick-Tune's curves
were validated.

Files (md5): trajectory_amendment3.md 68997af8cd801ebca198776b2626c181, qt_model_level.py
498ccccb5d9ed6de7a71417d1a1c40b2, qt_score.py 3ace18aa5aee44106002d00f7a78defb, qt_reversal.py
1877ad66cf8d54f28fddf358aa9c2791, qt_model_level_report.md dfdeb25cd579b43651df73b1e7239f0f,
qt_model_level.tsv a83777a3aebb5a13c9a2659f5acf6223, qt_scores.jsonl
b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl 50ea8a9ba8fd20713c25211fd30c6f8e, micro.zip
ed189adccce30d8b7f4a6e52b8f0caae.

Report as written by analysis/qt_model_level.py (headings lowered two levels):


Pre-specified in docs/trajectory_amendment3.md. Inputs: qt_scores.jsonl b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl 50ea8a9ba8fd20713c25211fd30c6f8e. Model-task pairs with complete runs: 674; included (at least 3 complete runs): 614 on 30 tasks; excluded: 60. Complete runs per included pair: median 10, range 3 to 23.

#### Outcome: median final top-1 over complete runs (primary)

Random model: mean regret 43.76 points. Outcome against log-dimension: mean Spearman +0.22.

| features | estimator | rankings | Spearman with outcome | partial, controlling log-dimension | score with log-dimension | top-1 tied with the best | mean regret (points) |
|---|---|---|---|---|---|---|---|
| raw | hscore_shrunk | 90 | +0.21 | +0.04 | +0.93 | 6/90 | 40.56 |
| raw | hscore | 90 | +0.27 | +0.18 | +0.90 | 7/90 | 26.23 |
| raw | logme | 90 | +0.12 | +0.10 | +0.44 | 6/90 | 45.57 |
| pca128 | hscore_shrunk | 90 | +0.20 |  | +0.67 | 4/90 | 47.89 |
| pca128 | hscore | 90 | +0.22 |  | +0.70 | 9/90 | 40.91 |
| pca128 | logme | 90 | +0.20 |  | +0.66 | 7/90 | 36.21 |

#### Outcome: mean final top-1 over complete runs

Random model: mean regret 31.30 points. Outcome against log-dimension: mean Spearman +0.24.

| features | estimator | rankings | Spearman with outcome | partial, controlling log-dimension | score with log-dimension | top-1 tied with the best | mean regret (points) |
|---|---|---|---|---|---|---|---|
| raw | hscore_shrunk | 90 | +0.27 | +0.16 | +0.93 | 6/90 | 28.02 |
| raw | hscore | 90 | +0.32 | +0.24 | +0.90 | 7/90 | 18.88 |
| raw | logme | 90 | +0.19 | +0.21 | +0.44 | 3/90 | 31.11 |
| pca128 | hscore_shrunk | 90 | +0.29 |  | +0.67 | 4/90 | 31.74 |
| pca128 | hscore | 90 | +0.29 |  | +0.70 | 14/90 | 26.95 |
| pca128 | logme | 90 | +0.28 |  | +0.66 | 13/90 | 24.21 |

#### Outcome: best final top-1 over complete runs

Random model: mean regret 20.30 points. Outcome against log-dimension: mean Spearman +0.24.

| features | estimator | rankings | Spearman with outcome | partial, controlling log-dimension | score with log-dimension | top-1 tied with the best | mean regret (points) |
|---|---|---|---|---|---|---|---|
| raw | hscore_shrunk | 90 | +0.34 | +0.32 | +0.93 | 22/90 | 14.42 |
| raw | hscore | 90 | +0.31 | +0.24 | +0.90 | 13/90 | 13.52 |
| raw | logme | 90 | +0.28 | +0.36 | +0.44 | 9/90 | 11.94 |
| pca128 | hscore_shrunk | 90 | +0.41 |  | +0.67 | 32/90 | 11.00 |
| pca128 | hscore | 90 | +0.40 |  | +0.70 | 26/90 | 11.67 |
| pca128 | logme | 90 | +0.38 |  | +0.66 | 28/90 | 9.97 |

Files: qt_model_level.tsv a83777a3aebb5a13c9a2659f5acf6223

_Read: against a model's typical fine-tuned accuracy (the median over 614 model-task pairs with 3 to
23 complete runs), every estimator agrees only weakly, with mean Spearman +0.12 to +0.27.
Controlling for log feature dimension leaves almost nothing of the shrunk H-score's correlation
(partial +0.04), and its choice does about as well as a random model (mean regret 40.56 against
43.76 points): its apparent signal was mostly feature size. Plain H-score keeps a modest signal
(partial +0.18) and a clearly lower regret (26.23). Projecting every model on 128 principal
components does not remove the link to size, since the scores still correlate +0.66 to +0.70 with
the original dimension, and the outcome itself rises slightly with dimension (+0.22); size and
transferability cannot be fully separated here, and the partial correlation is the cleaner control.
With best-run accuracy, the sensitivity outcome, all correlations are higher (+0.28 to +0.41) and
regrets lower, as expected when lucky settings are rewarded. The conclusion of 12.25 therefore holds
with a model-level outcome and with dimension controlled: on Quick-Tune micro, H-score predicts a
model's typical fine-tuned accuracy only weakly, and its shrunk form hardly at all once feature size
is accounted for._
### 12.27 Regret against compute: static, early leader, override, abstention, halving (analysis/regret_compute.py, commit 935ce90, job 11919022)

Task: Prof. Hazra (Nov 2026) asked, as the key experiment for the methods question, for a
regret-versus-compute comparison of H-score (static selection), the early leader, our warning and
override rule, and successive halving, under the same pre-specified criteria, with each method's
place on the frontier. docs/regret_compute_design.md was committed before the script existed.
H-score's, the early leader's and the rule's results were known before (12.13 to 12.26); successive
halving, the common compute accounting and the frontier are new, and the script was test-run on the
StudentBERT scores with stand-in costs while it was written. Decisions: 21 on StudentBERT (8
candidates, six-seed test gold), 90 on NLP (7 encoders, published fine-tuned gold; every method
stops at epoch 3, where the curves end) and 90 on Quick-Tune micro (models with at least 3 complete
runs, median curves and the model-level median outcome of 12.26). Compute is GPU-hours per decision:
StudentBERT's per-epoch costs are measured run times spread evenly over epochs and its scoring jobs'
GPU is not recorded; NLP is V100-SXM2 time; Quick-Tune's fine-tuning time is its own on unrecorded
hardware, so the static points' H200 scoring time is shown but counted as zero for the frontier. The
frontier is the set of points that no other point beats on both mean compute and mean regret, among
the methods that choose on every decision; abstention is reported beside it.

Files (md5): regret_compute_design.md 4dc6393deea651bf59aa8144e1ce1608, regret_compute.py
fd4288700ffd9af74fe92911371935ae, regret_compute_figure.py 1ef0df36f93e72b3a2c9f09627f4bb20,
regret_compute_report.md fe8155d94f0d79572f9df5af382f7908, regret_compute.tsv
d77242c6243450571be358a5661c5ca6, trajectories.tsv e93329cbc69ae8a3af843c544d43793b,
reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7, nlp_head_pairs.tsv
d023dd9346eb1020afa11d1dacbaf579, p2_head_pairs.tsv 9123ad27c17496af7f5f51cdb1987bc1,
p3_head_pairs.tsv 9bad0afce43c426c8105473a6b8f918c, qt_scores.jsonl
b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl 50ea8a9ba8fd20713c25211fd30c6f8e. The frontier
figure, benchmark_final/regret_compute_frontier.pdf, is drawn from regret_compute.tsv by
analysis/regret_compute_figure.py.

Report as written by analysis/regret_compute.py (headings lowered two levels):


Pre-specified in docs/regret_compute_design.md (H-score, early leader and rule results were known before; successive halving, the common compute accounting and the frontier are new; the script was test-run on the StudentBERT scores with stand-in costs while it was written, and no definition changed after that). StudentBERT reproduces RESULTS.md 12.17, 12.18 and 12.24. Inputs: trajectories.tsv e93329cbc69ae8a3af843c544d43793b, reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7, tg1_task2_kt_algebra2005.jsonl 268ff75c4c59a23b2fa4d9bcc85b60aa, tg1_task2_kt_algebra2006.jsonl d0f758940ee42bfdd8275e00ff12fe49, tg1_task2_kt_assist2009.jsonl 5823dd2999f96921409b9cce5272d924, tg1_task2_kt_assist2017.jsonl f8a29e8c85e022e3306350ee159e31ff, tg1_task2_kt_bridge2006.jsonl 38235d16b5d18d72d97bb9d8b9a9a0e0, tg1_task2_kt_ednet.jsonl 59096f8a8941896c18ca60515ffdd9f2, tg1_task2_kt_junyi.jsonl bc1af53d5d042f25d57057952795dad0, nlp_head_pairs.tsv d023dd9346eb1020afa11d1dacbaf579, p2_head_pairs.tsv 9123ad27c17496af7f5f51cdb1987bc1, p3_head_pairs.tsv 9bad0afce43c426c8105473a6b8f918c, task4_scores.jsonl 9dfe5d930681ec04d79cb52c149bbc4e, task4_scores_fresh.jsonl 20c403eb43ade91bac987af41538299e, task4_scores_p3.jsonl 34ef47d06f1e7d155c585547a51f9220, qt_scores.jsonl b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl 50ea8a9ba8fd20713c25211fd30c6f8e.

#### studentbert (test AUC)

21 decisions, 8 candidates; compute is cluster GPU time (fine-tunes on V100-SXM2; the scoring jobs' GPU is not in their records).

| method | setting | compute per decision (GPU-h) | chosen | mean regret [95% cluster CI] | zero regret | on the frontier |
|---|---|---|---|---|---|---|
| static H-score |  | 0.008 | 21/21 | 0.0012 [0.0002, 0.0023] | 11 | yes |
| early leader | epoch 1 | 0.039 | 21/21 | 0.0031 [0.0007, 0.0070] | 8 |  |
| override rule | epoch 1 | 0.047 | 21/21 | 0.0031 [0.0007, 0.0070] | 8 |  |
| warning, abstain | epoch 1 | 0.047 | 17/21 | 0.0009 [0.0001, 0.0018] | 8 |  |
| early leader | epoch 2 | 0.078 | 21/21 | 0.0010 [0.0002, 0.0025] | 13 |  |
| override rule | epoch 2 | 0.086 | 21/21 | 0.0011 [0.0002, 0.0025] | 13 |  |
| warning, abstain | epoch 2 | 0.086 | 12/21 | 0.0003 [0.0000, 0.0008] | 8 |  |
| early leader | epoch 3 | 0.118 | 21/21 | 0.0006 [0.0000, 0.0016] | 18 | yes |
| override rule | epoch 3 | 0.125 | 21/21 | 0.0006 [0.0001, 0.0016] | 17 |  |
| warning, abstain | epoch 3 | 0.125 | 12/21 | 0.0009 [0.0000, 0.0024] | 10 |  |
| early leader | epoch 5 | 0.196 | 21/21 | 0.0006 [0.0000, 0.0016] | 18 | yes |
| override rule | epoch 5 | 0.204 | 21/21 | 0.0005 [0.0000, 0.0015] | 19 | yes |
| warning, abstain | epoch 5 | 0.204 | 10/21 | 0.0010 [0.0000, 0.0026] | 9 |  |
| early leader | epoch 10 | 0.392 | 21/21 | 0.0021 [0.0004, 0.0041] | 11 |  |
| override rule | epoch 10 | 0.400 | 21/21 | 0.0017 [0.0003, 0.0034] | 13 |  |
| warning, abstain | epoch 10 | 0.400 | 5/21 | 0.0007 [0.0000, 0.0017] | 3 |  |
| early leader | epoch 20 | 0.784 | 21/21 | 0.0017 [0.0005, 0.0034] | 8 |  |
| override rule | epoch 20 | 0.792 | 21/21 | 0.0017 [0.0004, 0.0034] | 7 |  |
| warning, abstain | epoch 20 | 0.792 | 10/21 | 0.0009 [0.0000, 0.0019] | 5 |  |
| successive halving | start 1 | 0.078 | 21/21 | 0.0007 [0.0002, 0.0013] | 13 | yes |
| successive halving | start 2 | 0.157 | 21/21 | 0.0007 [0.0001, 0.0015] | 15 |  |
| successive halving | start 5 | 0.394 | 21/21 | 0.0013 [0.0004, 0.0024] | 9 |  |
| successive halving (factor 3) | start 1 | 0.068 | 21/21 | 0.0008 [0.0000, 0.0018] | 15 | yes |
| successive halving (factor 3) | start 2 | 0.138 | 21/21 | 0.0014 [0.0002, 0.0034] | 11 |  |
| successive halving (factor 3) | start 5 | 0.344 | 21/21 | 0.0009 [0.0004, 0.0014] | 11 |  |
| full fine-tuning | best validation | 0.784 | 21/21 | 0.0003 [0.0000, 0.0006] | 15 | yes |

#### nlp (points)

90 decisions, 7 encoders; curves stop at epoch 3, so every method stops there; compute is V100-SXM2 time.

| method | setting | compute per decision (GPU-h) | chosen | mean regret [95% cluster CI] | zero regret | on the frontier |
|---|---|---|---|---|---|---|
| static H-score |  | 0.029 | 90/90 | 2.59 [1.00, 3.93] | 32 | yes |
| early leader | epoch 1 | 0.085 | 90/90 | 1.64 [0.46, 3.33] | 40 | yes |
| override rule | epoch 1 | 0.115 | 90/90 | 2.16 [0.60, 4.05] | 34 |  |
| warning, abstain | epoch 1 | 0.115 | 25/90 | 0.70 [0.00, 2.08] | 17 |  |
| early leader | epoch 2 | 0.169 | 90/90 | 1.50 [0.43, 3.28] | 38 |  |
| override rule | epoch 2 | 0.199 | 90/90 | 2.06 [0.47, 3.95] | 35 |  |
| warning, abstain | epoch 2 | 0.199 | 20/90 | 0.90 [0.00, 3.45] | 16 |  |
| early leader | epoch 3 | 0.253 | 90/90 | 0.97 [0.27, 2.03] | 48 | yes |
| override rule | epoch 3 | 0.283 | 90/90 | 1.61 [0.39, 3.09] | 45 |  |
| warning, abstain | epoch 3 | 0.283 | 24/90 | 0.50 [0.00, 1.56] | 17 |  |
| successive halving | rungs 1,2,3 | 0.158 | 90/90 | 1.26 [0.43, 2.69] | 41 | yes |
| successive halving (factor 3) | rungs 1,2,3 | 0.121 | 90/90 | 1.62 [0.42, 3.64] | 36 | yes |

#### quicktune (top-1 points)

90 decisions, models with at least 3 complete runs, median curves; fine-tuning compute is Quick-Tune's recorded time on its own hardware; the static methods' compute is H200 scoring time, shown in its column but treated as zero for the frontier, since it is not on the same axis.

| method | setting | compute per decision (GPU-h) | chosen | mean regret [95% cluster CI] | zero regret | on the frontier |
|---|---|---|---|---|---|---|
| static H-score |  | 0.040 | 90/90 | 40.56 [30.41, 51.01] | 6 |  |
| static plain H-score |  | 0.039 | 90/90 | 26.23 [19.82, 33.08] | 6 | yes |
| early leader | epoch 1 | 0.094 | 90/90 | 33.58 [24.30, 43.13] | 12 |  |
| override rule | epoch 1 | 0.094 | 90/90 | 32.44 [23.62, 41.64] | 12 |  |
| warning, abstain | epoch 1 | 0.094 | 3/90 | 16.56 [16.56, 16.56] | 0 |  |
| early leader | epoch 2 | 0.189 | 90/90 | 15.18 [9.28, 21.94] | 21 | yes |
| override rule | epoch 2 | 0.189 | 90/90 | 15.18 [9.28, 21.94] | 21 | yes |
| warning, abstain | epoch 2 | 0.189 | 0/90 | nan [nan, nan] | 0 |  |
| early leader | epoch 3 | 0.283 | 90/90 | 13.20 [7.82, 19.70] | 27 | yes |
| override rule | epoch 3 | 0.283 | 90/90 | 13.20 [7.82, 19.70] | 27 | yes |
| warning, abstain | epoch 3 | 0.283 | 0/90 | nan [nan, nan] | 0 |  |
| early leader | epoch 5 | 0.474 | 90/90 | 9.79 [5.01, 15.74] | 36 |  |
| override rule | epoch 5 | 0.474 | 90/90 | 9.79 [5.01, 15.74] | 36 |  |
| warning, abstain | epoch 5 | 0.474 | 3/90 | 0.00 [0.00, 0.00] | 3 |  |
| early leader | epoch 10 | 0.953 | 90/90 | 1.06 [0.33, 1.97] | 72 | yes |
| override rule | epoch 10 | 0.953 | 90/90 | 1.06 [0.33, 1.97] | 72 | yes |
| warning, abstain | epoch 10 | 0.953 | 6/90 | 0.00 [0.00, 0.00] | 6 |  |
| early leader | epoch 20 | 1.911 | 90/90 | 1.08 [0.24, 2.18] | 72 |  |
| override rule | epoch 20 | 1.911 | 90/90 | 1.08 [0.24, 2.18] | 72 |  |
| warning, abstain | epoch 20 | 1.911 | 6/90 | 0.00 [0.00, 0.00] | 6 |  |
| early leader | epoch 30 | 2.901 | 90/90 | 0.25 [0.00, 0.62] | 81 | yes |
| override rule | epoch 30 | 2.901 | 90/90 | 0.25 [0.00, 0.62] | 81 | yes |
| warning, abstain | epoch 30 | 2.901 | 6/90 | 0.00 [0.00, 0.00] | 6 |  |
| early leader | epoch 40 | 3.923 | 90/90 | 0.02 [0.00, 0.06] | 87 | yes |
| override rule | epoch 40 | 3.923 | 90/90 | 0.02 [0.00, 0.06] | 87 | yes |
| warning, abstain | epoch 40 | 3.923 | 6/90 | 0.00 [0.00, 0.00] | 6 |  |
| early leader | epoch 50 | 4.946 | 90/90 | 0.00 [0.00, 0.00] | 90 | yes |
| override rule | epoch 50 | 4.946 | 90/90 | 0.00 [0.00, 0.00] | 90 | yes |
| warning, abstain | epoch 50 | 4.946 | 6/90 | 0.00 [0.00, 0.00] | 6 |  |
| successive halving | start 1 | 0.417 | 90/90 | 9.30 [5.42, 13.56] | 36 | yes |
| successive halving | start 2 | 0.932 | 90/90 | 2.51 [0.72, 4.72] | 69 | yes |
| successive halving | start 3 | 1.459 | 90/90 | 2.67 [0.08, 7.31] | 78 |  |
| successive halving (factor 3) | start 1 | 0.287 | 90/90 | 11.65 [6.68, 17.76] | 27 | yes |
| successive halving (factor 3) | start 2 | 0.657 | 90/90 | 4.28 [2.13, 6.77] | 48 | yes |
| successive halving (factor 3) | start 3 | 0.987 | 90/90 | 3.43 [0.26, 8.53] | 72 |  |

Files: regret_compute.tsv d77242c6243450571be358a5661c5ca6

_Read: static selection is always the cheapest point and never good enough once a little fine-tuning
is allowed. A simple multi-fidelity method matches or beats the override rule on every benchmark. On
StudentBERT, successive halving started at one epoch reaches 0.0007 AUC at 0.078 GPU-hours per
decision, a tenth of fully fine-tuning every candidate (0.784 GPU-hours, 0.0003 when chosen by best
validation), against 0.0012 for H-score alone; the early leader reaches 0.0006 at 3 epochs, and the
override's best point, 0.0005 at 5 epochs, lies within the others' intervals. On NLP the override
rule is never on the frontier: it keeps H-score's choice when the margin is not cleared, and its
regret stays above the early leader's at every budget (1.61 against 0.97 points at 3 epochs); the
frontier is H-score, the early leader at 1 and 3 epochs and successive halving. On Quick-Tune the
override equals the early leader from 2 epochs on, because H-score's choice is always far behind,
and the early leader and successive halving share the frontier (1.06 points at 10 epochs, 2.51 for
halving at similar compute). What the rule adds is the warning: with abstention, the decisions it
follows have the lowest regret (0.0003 AUC on 12 of 21 at 2 epochs on StudentBERT, 0.50 points on 24
of 90 at 3 epochs on NLP), at partial coverage, and on Quick-Tune it follows almost none (0 to 6 of
90; regret prints as nan where none is followed). On StudentBERT every method gets worse beyond
about 5 epochs (the early leader 0.0021 at 10 epochs, halving 0.0013 when started at 5), the
non-monotonicity of 12.24. StudentBERT is retrospective, NLP is truncated at 3 epochs, and
Quick-Tune's outcome shares its curves with the methods' trajectories, so its regret at 50 epochs is
zero by construction._

