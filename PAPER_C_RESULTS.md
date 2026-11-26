# Paper C: trajectories, compute, Quick-Tune, multi-fidelity and selective trust

Generated from RESULTS.md (md5 3700313c02db18d4488df1ae282c023e) by tools/split_results_by_paper.py. Do not edit: RESULTS.md stays the record, and every section below is copied from it unchanged. Sections, with the md5 of their text:

- 12.22: 6392724f81af44ab39099997ca3282d3
- 12.23: 93b91e8062ad824a3ccfde33ef9b5980
- 12.24: 79b884997aa661cc928c29d23e7e2f60
- 12.25: 8dc9e8ad9a1fc7c97351adfed5a7e6e4
- 12.26: c693e05224dcea69825a0c916412b632
- 12.27: df90cccbdf4689f68030cc59be843a95
- 12.28: d4a3b48cb1b9f972f25ab0c227cec2b4
- 12.29: 1941de4498107feb47c7ffbb55f2069f
- 12.30: 31c2e841fff3d5946d105ce993a5925d
- 12.31: 4f712713ef0b12879146f0ce0dd81cd7

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
### 12.28 Selective trust in the static recommendation: coverage, compute and transfer (analysis/selective_trust.py, commit b90163f, job 11951748)

Task: Prof. Hazra (Nov 2026), for Paper C: the promising part of the rule is the warning, or
selective-trust, signal. She asked when the static (H-score) recommendation can be trusted, when to
abstain or keep evaluating, how much compute that decision needs, and whether one common threshold
transfers across domains or each domain needs its own budget, with the early leader and successive
halving (12.27) as the multi-fidelity comparisons. docs/selective_trust_design.md (commit c8ca7dd)
and its amendments 1 (882de48: area, ties, pooling, budget rule, intervals, references) and 2
(992bd44: an equal-domain-weight sensitivity analysis of the transfer test) were committed before
analysis/selective_trust.py (b90163f), which was not run on real data before this job; its tests use
synthetic decisions. Evidence classes: the rule at t = 0 was known before (12.17, 12.20, 12.27) and
is reproduced exactly, decision by decision and in every 12.27 warning row (the run stops
otherwise); the threshold family, the normalized units, the area, the transfer test and its
sensitivity analysis, the budget question and the reference points are pre-specified. Decisions and
costs are those of 12.27, loaded by its code: 21 on StudentBERT, 90 on NLP (curves to epoch 3) and
90 on Quick-Tune micro. Compute per decision is frozen scoring plus k epochs of every candidate;
Quick-Tune's scoring stays off the fine-tuning axis, so its trust-all row shows 0.000 GPU-hours
where 12.27's static row shows 0.040, and the share of full compute divides by fine-tuning alone, so
StudentBERT's share at 20 epochs is 1.010.

Files (md5): selective_trust_design.md f63bfdc6bc0df97a97fb2fc97ef0eddd,
selective_trust_amendment_1.md de8a55b63b442dd2d1f28bb7bb2c1c9e, selective_trust_amendment_2.md
e8c9ae36be77cfb3fd887b1a6f0a82f2, selective_trust.py 561f9d471f9df4686c35bf6f5c7129a0,
test_selective_trust.py 38cbc5781baa722f080d80fe7610d98a, regret_compute.py
fd4288700ffd9af74fe92911371935ae, selective_trust_report.md 6d8abb21cfb3fea5ea23b1c9bd3a7ee3,
selective_trust.tsv afde26b12e7cf0a30472b35febc94e65, trajectories.tsv
e93329cbc69ae8a3af843c544d43793b, reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7,
tg1_task2_kt_algebra2005.jsonl 268ff75c4c59a23b2fa4d9bcc85b60aa, tg1_task2_kt_algebra2006.jsonl
d0f758940ee42bfdd8275e00ff12fe49, tg1_task2_kt_assist2009.jsonl 5823dd2999f96921409b9cce5272d924,
tg1_task2_kt_assist2017.jsonl f8a29e8c85e022e3306350ee159e31ff, tg1_task2_kt_bridge2006.jsonl
38235d16b5d18d72d97bb9d8b9a9a0e0, tg1_task2_kt_ednet.jsonl 59096f8a8941896c18ca60515ffdd9f2,
tg1_task2_kt_junyi.jsonl bc1af53d5d042f25d57057952795dad0, nlp_head_pairs.tsv
d023dd9346eb1020afa11d1dacbaf579, p2_head_pairs.tsv 9123ad27c17496af7f5f51cdb1987bc1,
p3_head_pairs.tsv 9bad0afce43c426c8105473a6b8f918c, task4_scores.jsonl
9dfe5d930681ec04d79cb52c149bbc4e, task4_scores_fresh.jsonl 20c403eb43ade91bac987af41538299e,
task4_scores_p3.jsonl 34ef47d06f1e7d155c585547a51f9220, qt_scores.jsonl
b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl 50ea8a9ba8fd20713c25211fd30c6f8e.
selective_trust.tsv holds the full grid (domain, budget and threshold: coverage, selective regret
raw and normalized, compute), 741 rows.

Report as written by analysis/selective_trust.py (headings lowered two levels):


Evidence classes: the rule at t = 0 (trust the static choice unless another candidate is strictly ahead) was known before (RESULTS.md 12.17, 12.20 and 12.27) and is reproduced here exactly, decision by decision and in every 12.27 "warning, abstain" row. The threshold family, the normalized units, the area, the leave-one-domain-out transfer and its equal-domain-weight sensitivity analysis, the budget question and the reference points are pre-specified in docs/selective_trust_design.md (commit c8ca7dd), amendment 1 (commit 882de48) and amendment 2, committed before this script. The script was not run on real data before this run (the inputs are on the cluster only); its tests use synthetic decisions. Static choice: H-score (shrunk H-score on Quick-Tune). Signal: the normalized lead L (the static choice's validation score minus the best other candidate's, over the spread of all candidates at epoch k; 0 when the spread is 0); trust when L >= t, t from -1.00 to 1.00 in steps of 0.05. Normalized regret: regret over the domain's mean regret of a random candidate. Targets: normalized selective regret at most 0.10, coverage at least 0.50. Intervals: 95% cluster bootstrap over targets or tasks, 10,000 resamples, random.Random(0).

Inputs: trajectories.tsv e93329cbc69ae8a3af843c544d43793b, reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7, tg1_task2_kt_algebra2005.jsonl 268ff75c4c59a23b2fa4d9bcc85b60aa, tg1_task2_kt_algebra2006.jsonl d0f758940ee42bfdd8275e00ff12fe49, tg1_task2_kt_assist2009.jsonl 5823dd2999f96921409b9cce5272d924, tg1_task2_kt_assist2017.jsonl f8a29e8c85e022e3306350ee159e31ff, tg1_task2_kt_bridge2006.jsonl 38235d16b5d18d72d97bb9d8b9a9a0e0, tg1_task2_kt_ednet.jsonl 59096f8a8941896c18ca60515ffdd9f2, tg1_task2_kt_junyi.jsonl bc1af53d5d042f25d57057952795dad0, nlp_head_pairs.tsv d023dd9346eb1020afa11d1dacbaf579, p2_head_pairs.tsv 9123ad27c17496af7f5f51cdb1987bc1, p3_head_pairs.tsv 9bad0afce43c426c8105473a6b8f918c, task4_scores.jsonl 9dfe5d930681ec04d79cb52c149bbc4e, task4_scores_fresh.jsonl 20c403eb43ade91bac987af41538299e, task4_scores_p3.jsonl 34ef47d06f1e7d155c585547a51f9220, qt_scores.jsonl b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl 50ea8a9ba8fd20713c25211fd30c6f8e. Design files: selective_trust_design.md f63bfdc6bc0df97a97fb2fc97ef0eddd, selective_trust_amendment_1.md de8a55b63b442dd2d1f28bb7bb2c1c9e, selective_trust_amendment_2.md e8c9ae36be77cfb3fd887b1a6f0a82f2. regret_compute.py fd4288700ffd9af74fe92911371935ae.

#### studentbert (test AUC)

21 decisions. Mean regret of a random candidate (the normalizer): 0.007540 test AUC. Fine-tuning every candidate fully: 0.784 GPU-h per decision.

At t = 0 (the common fixed rule) at each budget, with the area under the normalized selective-regret curve over all thresholds and the multi-fidelity references of 12.27 at the nearest compute (method setting: mean regret at GPU-h).

| budget | compute per decision (GPU-h) | share of full compute | trusted, coverage [95% CI] | selective regret [95% CI] | normalized [95% CI] | area (from coverage) | early leader | successive halving | successive halving (factor 3) |
|---|---|---|---|---|---|---|---|---|---|
| 0 (trust all) | 0.008 | 0.010 | 21/21, 1.000 [1.000, 1.000] | 0.0012 [0.0002, 0.0023] | 0.164 [0.033, 0.301] | n/a | epoch 1: 0.0031 at 0.039 | start 1: 0.0007 at 0.078 | start 1: 0.0008 at 0.068 |
| epoch 1 | 0.047 | 0.060 | 17/21, 0.810 [0.619, 0.952] | 0.0009 [0.0001, 0.0018] | 0.122 [0.012, 0.244] | 0.080 (from 0.048) | epoch 1: 0.0031 at 0.039 | start 1: 0.0007 at 0.078 | start 1: 0.0008 at 0.068 |
| epoch 2 | 0.086 | 0.110 | 12/21, 0.571 [0.238, 0.857] | 0.0003 [0.0000, 0.0008] | 0.037 [0.000, 0.100] | 0.056 (from 0.048) | epoch 2: 0.0010 at 0.078 | start 1: 0.0007 at 0.078 | start 1: 0.0008 at 0.068 |
| epoch 3 | 0.125 | 0.160 | 12/21, 0.571 [0.238, 0.857] | 0.0009 [0.0000, 0.0024] | 0.120 [0.000, 0.322] | 0.098 (from 0.048) | epoch 3: 0.0006 at 0.118 | start 2: 0.0007 at 0.157 | start 2: 0.0014 at 0.138 |
| epoch 5 | 0.204 | 0.260 | 10/21, 0.476 [0.143, 0.857] | 0.0010 [0.0000, 0.0026] | 0.137 [0.000, 0.343] | 0.099 (from 0.048) | epoch 5: 0.0006 at 0.196 | start 2: 0.0007 at 0.157 | start 2: 0.0014 at 0.138 |
| epoch 10 | 0.400 | 0.510 | 5/21, 0.238 [0.048, 0.476] | 0.0007 [0.0000, 0.0017] | 0.092 [0.000, 0.229] | 0.181 (from 0.048) | epoch 10: 0.0021 at 0.392 | start 5: 0.0013 at 0.394 | start 5: 0.0009 at 0.344 |
| epoch 20 | 0.792 | 1.010 | 10/21, 0.476 [0.190, 0.762] | 0.0009 [0.0000, 0.0019] | 0.115 [0.006, 0.248] | 0.112 (from 0.048) | epoch 20: 0.0017 at 0.784 | start 5: 0.0013 at 0.394 | start 5: 0.0009 at 0.344 |

#### nlp (points)

90 decisions. Mean regret of a random candidate (the normalizer): 3.3633 points. NLP curves stop at epoch 3, so there is no full-compute share.

At t = 0 (the common fixed rule) at each budget, with the area under the normalized selective-regret curve over all thresholds and the multi-fidelity references of 12.27 at the nearest compute (method setting: mean regret at GPU-h).

| budget | compute per decision (GPU-h) | share of full compute | trusted, coverage [95% CI] | selective regret [95% CI] | normalized [95% CI] | area (from coverage) | early leader | successive halving | successive halving (factor 3) |
|---|---|---|---|---|---|---|---|---|---|
| 0 (trust all) | 0.029 | n/a | 90/90, 1.000 [1.000, 1.000] | 2.59 [1.00, 3.93] | 0.770 [0.296, 1.170] | n/a | epoch 1: 1.64 at 0.085 | rungs 1,2,3: 1.26 at 0.158 | rungs 1,2,3: 1.62 at 0.121 |
| epoch 1 | 0.115 | n/a | 25/90, 0.278 [0.078, 0.533] | 0.70 [0.00, 2.08] | 0.208 [0.000, 0.620] | 0.472 (from 0.011) | epoch 1: 1.64 at 0.085 | rungs 1,2,3: 1.26 at 0.158 | rungs 1,2,3: 1.62 at 0.121 |
| epoch 2 | 0.199 | n/a | 20/90, 0.222 [0.089, 0.444] | 0.90 [0.00, 3.45] | 0.269 [0.000, 1.026] | 0.526 (from 0.011) | epoch 2: 1.50 at 0.169 | rungs 1,2,3: 1.26 at 0.158 | rungs 1,2,3: 1.62 at 0.121 |
| epoch 3 | 0.283 | n/a | 24/90, 0.267 [0.067, 0.522] | 0.50 [0.00, 1.56] | 0.149 [0.000, 0.465] | 0.424 (from 0.022) | epoch 3: 0.97 at 0.253 | rungs 1,2,3: 1.26 at 0.158 | rungs 1,2,3: 1.62 at 0.121 |

#### quicktune (top-1 points)

90 decisions. Mean regret of a random candidate (the normalizer): 43.7596 top-1 points. Fine-tuning every candidate fully: 4.946 GPU-h per decision.

At t = 0 (the common fixed rule) at each budget, with the area under the normalized selective-regret curve over all thresholds and the multi-fidelity references of 12.27 at the nearest compute (method setting: mean regret at GPU-h).

| budget | compute per decision (GPU-h) | share of full compute | trusted, coverage [95% CI] | selective regret [95% CI] | normalized [95% CI] | area (from coverage) | early leader | successive halving | successive halving (factor 3) |
|---|---|---|---|---|---|---|---|---|---|
| 0 (trust all) | 0.000 | 0.000 | 90/90, 1.000 [1.000, 1.000] | 40.56 [30.41, 51.01] | 0.927 [0.695, 1.166] | n/a | epoch 1: 33.58 at 0.094 | start 1: 9.30 at 0.417 | start 1: 11.65 at 0.287 |
| epoch 1 | 0.094 | 0.019 | 3/90, 0.033 [0.000, 0.100] | 16.56 [16.56, 16.56] | 0.378 [0.378, 0.378] | 0.815 (from 0.033) | epoch 1: 33.58 at 0.094 | start 1: 9.30 at 0.417 | start 1: 11.65 at 0.287 |
| epoch 2 | 0.189 | 0.038 | 0/90, 0.000 [0.000, 0.000] | n/a | n/a | 0.710 (from 0.033) | epoch 2: 15.18 at 0.189 | start 1: 9.30 at 0.417 | start 1: 11.65 at 0.287 |
| epoch 3 | 0.283 | 0.057 | 0/90, 0.000 [0.000, 0.000] | n/a | n/a | 0.571 (from 0.033) | epoch 3: 13.20 at 0.283 | start 1: 9.30 at 0.417 | start 1: 11.65 at 0.287 |
| epoch 5 | 0.474 | 0.096 | 3/90, 0.033 [0.000, 0.100] | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.520 (from 0.033) | epoch 5: 9.79 at 0.474 | start 1: 9.30 at 0.417 | start 2: 4.28 at 0.657 |
| epoch 10 | 0.953 | 0.193 | 6/90, 0.067 [0.000, 0.167] | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.438 (from 0.033) | epoch 10: 1.06 at 0.953 | start 2: 2.51 at 0.932 | start 3: 3.43 at 0.987 |
| epoch 20 | 1.911 | 0.386 | 6/90, 0.067 [0.000, 0.167] | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.423 (from 0.067) | epoch 20: 1.08 at 1.911 | start 3: 2.67 at 1.459 | start 3: 3.43 at 0.987 |
| epoch 30 | 2.901 | 0.587 | 6/90, 0.067 [0.000, 0.167] | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.428 (from 0.033) | epoch 30: 0.25 at 2.901 | start 3: 2.67 at 1.459 | start 3: 3.43 at 0.987 |
| epoch 40 | 3.923 | 0.793 | 6/90, 0.067 [0.000, 0.167] | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.416 (from 0.033) | epoch 40: 0.02 at 3.923 | start 3: 2.67 at 1.459 | start 3: 3.43 at 0.987 |
| epoch 50 | 4.946 | 1.000 | 6/90, 0.067 [0.000, 0.167] | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.433 (from 0.033) | epoch 50: 0.00 at 4.946 | start 3: 2.67 at 1.459 | start 3: 3.43 at 0.987 |

#### Leave-one-domain-out threshold transfer (k = 1, 2, 3)

The threshold is chosen on the two other domains (the highest pooled coverage with pooled normalized selective regret at most 0.10; ties to the largest t of the band) and applied to the held-out domain, beside the held-out domain's own best threshold under the same target. Primary rule: the source domains pooled by decision (amendment 1). Sensitivity analysis (amendment 2): each source domain weighted equally, coverage above zero required in both. "none": no threshold qualifies.

| budget | held out | rule | source threshold [band] | source coverage, normalized regret | held-out trusted, coverage [95% CI] | held-out normalized regret [95% CI] | at the band's smallest t: coverage, normalized regret | own best threshold [band] | own trusted, coverage [95% CI] | own normalized regret [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| epoch 1 | studentbert | primary (by decision) | 0.30 [0.30, 0.30] | 0.011, 0.000 | 6/21, 0.286 [0.095, 0.476] | 0.050 [0.000, 0.149] | 6/21, 0.286 [0.095, 0.476]; 0.050 [0.000, 0.149] | 0.20 [0.20, 0.20] | 9/21, 0.429 [0.190, 0.667] | 0.085 [0.012, 0.235] |
| epoch 1 | studentbert | sensitivity analysis (equal domain weight) | none | n/a | n/a | n/a | n/a | 0.20 [0.20, 0.20] | 9/21, 0.429 [0.190, 0.667] | 0.085 [0.012, 0.235] |
| epoch 1 | nlp | primary (by decision) | 0.20 [0.20, 0.20] | 0.081, 0.085 | 7/90, 0.078 [0.022, 0.133] | 0.352 [0.000, 0.639] | 7/90, 0.078 [0.022, 0.133]; 0.352 [0.000, 0.639] | 0.30 [0.30, 0.30] | 2/90, 0.022 [0.000, 0.067] | 0.000 [0.000, 0.000] |
| epoch 1 | nlp | sensitivity analysis (equal domain weight) | none | n/a | n/a | n/a | n/a | 0.30 [0.30, 0.30] | 2/90, 0.022 [0.000, 0.067] | 0.000 [0.000, 0.000] |
| epoch 1 | quicktune | primary (by decision) | 0.30 [0.30, 0.30] | 0.072, 0.037 | 0/90, 0.000 [0.000, 0.000] | n/a | 0/90, 0.000 [0.000, 0.000]; n/a | none | n/a | n/a |
| epoch 1 | quicktune | sensitivity analysis (equal domain weight) | 0.30 [0.30, 0.30] | 0.154, 0.025 | 0/90, 0.000 [0.000, 0.000] | n/a | 0/90, 0.000 [0.000, 0.000]; n/a | none | n/a | n/a |
| epoch 2 | studentbert | primary (by decision) | 0.30 [0.25, 0.30] | 0.017, 0.000 | 4/21, 0.190 [0.000, 0.476] | 0.017 [0.000, 0.070] | 5/21, 0.238 [0.000, 0.524]; 0.028 [0.000, 0.070] | -0.10 [-0.25, -0.10] | 17/21, 0.810 [0.524, 1.000] | 0.099 [0.012, 0.217] |
| epoch 2 | studentbert | sensitivity analysis (equal domain weight) | none | n/a | n/a | n/a | n/a | -0.10 [-0.25, -0.10] | 17/21, 0.810 [0.524, 1.000] | 0.099 [0.012, 0.217] |
| epoch 2 | nlp | primary (by decision) | -0.10 [-0.25, -0.10] | 0.153, 0.099 | 29/90, 0.322 [0.200, 0.522] | 0.473 [0.084, 1.215] | 46/90, 0.511 [0.356, 0.733]; 0.534 [0.165, 1.089] | 0.30 [0.25, 0.30] | 3/90, 0.033 [0.000, 0.100] | 0.000 [0.000, 0.000] |
| epoch 2 | nlp | sensitivity analysis (equal domain weight) | none | n/a | n/a | n/a | n/a | 0.30 [0.25, 0.30] | 3/90, 0.033 [0.000, 0.100] | 0.000 [0.000, 0.000] |
| epoch 2 | quicktune | primary (by decision) | 0.25 [0.25, 0.25] | 0.072, 0.017 | 0/90, 0.000 [0.000, 0.000] | n/a | 0/90, 0.000 [0.000, 0.000]; n/a | -0.40 [-0.45, -0.40] | 3/90, 0.033 [0.000, 0.100] | 0.057 [0.057, 0.057] |
| epoch 2 | quicktune | sensitivity analysis (equal domain weight) | 0.25 [0.25, 0.25] | 0.136, 0.014 | 0/90, 0.000 [0.000, 0.000] | n/a | 0/90, 0.000 [0.000, 0.000]; n/a | -0.40 [-0.45, -0.40] | 3/90, 0.033 [0.000, 0.100] | 0.057 [0.057, 0.057] |
| epoch 3 | studentbert | primary (by decision) | 0.10 [0.10, 0.10] | 0.094, 0.075 | 8/21, 0.381 [0.095, 0.714] | 0.171 [0.000, 0.685] | 8/21, 0.381 [0.095, 0.714]; 0.171 [0.000, 0.685] | 0.15 [0.15, 0.15] | 5/21, 0.238 [0.000, 0.571] | 0.000 [0.000, 0.000] |
| epoch 3 | studentbert | sensitivity analysis (equal domain weight) | none | n/a | n/a | n/a | n/a | 0.15 [0.15, 0.15] | 5/21, 0.238 [0.000, 0.571] | 0.000 [0.000, 0.000] |
| epoch 3 | nlp | primary (by decision) | 0.15 [0.15, 0.15] | 0.045, 0.000 | 13/90, 0.144 [0.044, 0.300] | 0.098 [0.000, 0.479] | 13/90, 0.144 [0.044, 0.300]; 0.098 [0.000, 0.479] | 0.10 [0.10, 0.10] | 17/90, 0.189 [0.044, 0.411] | 0.075 [0.000, 0.479] |
| epoch 3 | nlp | sensitivity analysis (equal domain weight) | -0.35 [-0.35, -0.35] | 0.493, 0.076 | 54/90, 0.600 [0.378, 0.822] | 0.570 [0.227, 1.104] | 54/90, 0.600 [0.378, 0.822]; 0.570 [0.227, 1.104] | 0.10 [0.10, 0.10] | 17/90, 0.189 [0.044, 0.411] | 0.075 [0.000, 0.479] |
| epoch 3 | quicktune | primary (by decision) | 0.15 [0.15, 0.15] | 0.162, 0.071 | 0/90, 0.000 [0.000, 0.000] | n/a | 0/90, 0.000 [0.000, 0.000]; n/a | -0.30 [-0.35, -0.30] | 3/90, 0.033 [0.000, 0.100] | 0.000 [0.000, 0.000] |
| epoch 3 | quicktune | sensitivity analysis (equal domain weight) | 0.15 [0.15, 0.15] | 0.191, 0.049 | 0/90, 0.000 [0.000, 0.000] | n/a | 0/90, 0.000 [0.000, 0.000]; n/a | -0.30 [-0.35, -0.30] | 3/90, 0.033 [0.000, 0.100] | 0.000 [0.000, 0.000] |

#### The domain-specific budget

Per domain, the smallest budget (k = 0, trusting every decision, checked first) at which some threshold reaches normalized selective regret at most 0.10 with coverage at least 0.50, with the highest-coverage such threshold (ties to the largest t).

| domain | smallest budget | threshold [band] | trusted, coverage [95% CI] | normalized selective regret [95% CI] | compute per decision (GPU-h) | share of full compute |
|---|---|---|---|---|---|---|
| studentbert | epoch 2 | -0.10 [-0.25, -0.10] | 17/21, 0.810 [0.524, 1.000] | 0.099 [0.012, 0.217] | 0.086 | 0.110 |
| nlp | not reached | n/a | n/a | n/a | n/a | n/a |
| quicktune | not reached | n/a | n/a | n/a | n/a | n/a |

Limitations: StudentBERT is retrospective; NLP stops at epoch 3; Quick-Tune's outcome shares its curves with the signal, so its selective regret at 50 epochs is zero by construction; the t = 0 results were known before. Quick-Tune's frozen scoring (H200) is off the fine-tuning compute axis, as in 12.27.

Files: selective_trust.tsv afde26b12e7cf0a30472b35febc94e65

_Read: in these data no single threshold carries across domains, and only one domain reaches the
targets within its budgets. At t = 0, StudentBERT trusts 17/21 decisions at 1 epoch (normalized
selective regret 0.122) and 12/21 at 2 epochs (0.037), NLP trusts 20/90 to 25/90 (0.149 to 0.269),
and Quick-Tune at most 6/90. Only StudentBERT meets the pre-specified targets (normalized selective
regret at most 0.10 with coverage at least 0.50): at 2 epochs with t = -0.10 it trusts 17/21 at
0.099 [0.012, 0.217], for 0.086 GPU-hours per decision, 0.110 of fully fine-tuning every candidate;
the point sits at the target and its interval spans it. NLP, whose curves stop at 3 epochs, and
Quick-Tune, up to 50, do not reach the targets. A threshold chosen on the two other domains misses
on the held-out one: held-out Quick-Tune trusts 0/90 at every shared budget; held-out NLP has
normalized selective regret 0.352 and 0.473 at 1 and 2 epochs and 0.098 on 13/90 at 3; held-out
StudentBERT stays under the target at 1 and 2 epochs (0.050 and 0.017) but trusts fewer decisions
than its own best threshold (6/21 against 9/21, 4/21 against 17/21), and has 0.171 at 3 epochs. The
equal-domain-weight sensitivity analysis finds no qualifying threshold in 5 of the 9 cases; where it
differs from the primary rule (NLP held out at 3 epochs, t = -0.35), the held-out regret is 0.570.
At StudentBERT's operating point the multi-fidelity references choose on every decision at similar
compute (the early leader 0.0010 AUC at 0.078 GPU-hours, successive halving 0.0007 at 0.078, against
a normalizer of 0.007540), so at that compute selective trust is not shown to beat successive
halving on regret; what it adds is knowing which decisions to evaluate further. Intervals are wide
(7 StudentBERT targets, 5 NLP tasks). StudentBERT is retrospective, NLP stops at 3 epochs,
Quick-Tune's outcome shares its curves with the signal, and the t = 0 results were known before._
### 12.29 NLP selective trust at 1, 2, 3, 5 and 10 epochs: coverage, regret and compute (analysis/nlp_budget_extension.py, commit 6f269bf, job 12136024)

Task: Prof. Hazra (2026-11-20 and 11-23), for Paper C: extend the NLP early fine-tuning runs to 5
and 10 epochs with the same models, tasks, seeds, decision rule, threshold and compute accounting,
without new tuning, and report coverage, regret and compute at 1, 2, 3, 5 and 10 epochs, whether NLP
ever reaches the selective-trust target, where the curve flattens or gets worse, the best budget
against successive halving and the early leader at similar compute, and whether the useful budget is
still domain-dependent. docs/nlp_budget_extension_design.md (commit 29652a2) fixed the runs, rules
and read-outs before any run; the 630 runs (7 encoders x 5 tasks x 2 poolings x 9 seeds) were rerun
from the pretrained encoders to 10 epochs with scripts/task4_early_ft.py unchanged
(slurm/generators/gen_nlp_ft10_jobs.sh, commit 998efa3; 210 jobs, all COMPLETED, 76.02 GPU-hours on
V100-SXM2), after a 5-run protocol check. analysis/nlp_budget_extension.py (commit 3b542c0) was not
run on real data before its first run (job 12135671); a report label (the factor-3 row of item 4)
and the limitation wording on epochs 1 to 3 were then corrected (commit 6f269bf) and the analysis
rerun (job 12136024) with identical tables (the two table files have the same md5 in both runs).
Evidence classes: the results up to epoch 3 were known before (12.27, 12.28); epochs 4 to 10, the
budgets 5 and 10 and the read-outs are pre-specified and descriptive. Every NLP budget, including 1
to 3, comes from the new runs, so 12.28 stays as recorded.

Files (md5): nlp_budget_extension_design.md d82d64f47e3d23f35a94316446a0041b,
nlp_budget_extension.py 39d08a0959e4f99f5c01510912b2b840, test_nlp_budget_extension.py
94a8f7bbd4fab16aff25b83450b3ffc5, selective_trust.py 561f9d471f9df4686c35bf6f5c7129a0,
regret_compute.py fd4288700ffd9af74fe92911371935ae, compare_ft_epochs.py
16fbc8a6b27de3bec529729e1dec032e, gen_nlp_ft10_jobs.sh 8038b2ea85aa01443c800a6a891e604b,
task4_early_ft.py 9c87a401c955fc90aca772e5c17cbf08, nlp_budget_extension_report.md
6737d85651b7bedb453967c3674e1333, nlp_budget_extension.tsv adb591aa86fbaed5a606da575b2c288f,
nlp_budget_extension_references.tsv 2e655e0112e3d896d70a75f36579b0e0, trajectories.tsv
e93329cbc69ae8a3af843c544d43793b, reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7,
tg1_task2_kt_algebra2005.jsonl 268ff75c4c59a23b2fa4d9bcc85b60aa, tg1_task2_kt_algebra2006.jsonl
d0f758940ee42bfdd8275e00ff12fe49, tg1_task2_kt_assist2009.jsonl 5823dd2999f96921409b9cce5272d924,
tg1_task2_kt_assist2017.jsonl f8a29e8c85e022e3306350ee159e31ff, tg1_task2_kt_bridge2006.jsonl
38235d16b5d18d72d97bb9d8b9a9a0e0, tg1_task2_kt_ednet.jsonl 59096f8a8941896c18ca60515ffdd9f2,
tg1_task2_kt_junyi.jsonl bc1af53d5d042f25d57057952795dad0, nlp_head_pairs.tsv
d023dd9346eb1020afa11d1dacbaf579, p2_head_pairs.tsv 9123ad27c17496af7f5f51cdb1987bc1,
p3_head_pairs.tsv 9bad0afce43c426c8105473a6b8f918c, task4_scores.jsonl
9dfe5d930681ec04d79cb52c149bbc4e, task4_scores_fresh.jsonl 20c403eb43ade91bac987af41538299e,
task4_scores_p3.jsonl 34ef47d06f1e7d155c585547a51f9220, qt_scores.jsonl
b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl 50ea8a9ba8fd20713c25211fd30c6f8e. The 630 new
runs are pinned by the manifest md5 in the report; the tables hold the full threshold grid and the
reference rows.

Report as written by analysis/nlp_budget_extension.py (headings lowered two levels):


Evidence classes: the runs, rules and read-outs are fixed in docs/nlp_budget_extension_design.md (commit 29652a2), committed before any run; the results up to epoch 3 were known before (RESULTS.md 12.27 and 12.28); epochs 4 to 10, the budgets 5 and 10, the flattening and worsening read-outs and the transfer test at 5 and 10 are new; all read-outs are descriptive. Every NLP budget comes from the new 10-epoch runs (one campaign). Rules as in RESULTS.md 12.28 (analysis/selective_trust.py, imported): trust when the normalized lead L >= t, t from -1.00 to 1.00 in steps of 0.05, t = 0 the common fixed rule, normalized regret over a random candidate's mean regret, targets 0.10 and 0.50, 95% cluster bootstrap over the 5 tasks (10,000 resamples, random.Random(0)). The script was not run on real data before this run; its tests use synthetic decisions.

Inputs: trajectories.tsv e93329cbc69ae8a3af843c544d43793b, reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7, tg1_task2_kt_algebra2005.jsonl 268ff75c4c59a23b2fa4d9bcc85b60aa, tg1_task2_kt_algebra2006.jsonl d0f758940ee42bfdd8275e00ff12fe49, tg1_task2_kt_assist2009.jsonl 5823dd2999f96921409b9cce5272d924, tg1_task2_kt_assist2017.jsonl f8a29e8c85e022e3306350ee159e31ff, tg1_task2_kt_bridge2006.jsonl 38235d16b5d18d72d97bb9d8b9a9a0e0, tg1_task2_kt_ednet.jsonl 59096f8a8941896c18ca60515ffdd9f2, tg1_task2_kt_junyi.jsonl bc1af53d5d042f25d57057952795dad0, nlp_head_pairs.tsv d023dd9346eb1020afa11d1dacbaf579, p2_head_pairs.tsv 9123ad27c17496af7f5f51cdb1987bc1, p3_head_pairs.tsv 9bad0afce43c426c8105473a6b8f918c, task4_scores.jsonl 9dfe5d930681ec04d79cb52c149bbc4e, task4_scores_fresh.jsonl 20c403eb43ade91bac987af41538299e, task4_scores_p3.jsonl 34ef47d06f1e7d155c585547a51f9220, qt_scores.jsonl b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl 50ea8a9ba8fd20713c25211fd30c6f8e. Design files: selective_trust_design.md f63bfdc6bc0df97a97fb2fc97ef0eddd, selective_trust_amendment_1.md de8a55b63b442dd2d1f28bb7bb2c1c9e, selective_trust_amendment_2.md e8c9ae36be77cfb3fd887b1a6f0a82f2, nlp_budget_extension_design.md d82d64f47e3d23f35a94316446a0041b. Code: selective_trust.py 561f9d471f9df4686c35bf6f5c7129a0, compare_ft_epochs.py 16fbc8a6b27de3bec529729e1dec032e, regret_compute.py fd4288700ffd9af74fe92911371935ae. New runs: 630 (manifest md5 b9301dfc082b1e045459546c26521f56).

Checks passed: the recorded NLP, StudentBERT and Quick-Tune decisions reproduce every RESULTS.md 12.27 "warning, abstain" row; the reference rows reproduce the 12.27 early leader and successive-halving rows on the recorded runs; every new run has epochs 1 to 10 and the data and settings of its recorded run; the normalizer is unchanged.

#### The new runs against the recorded 3-epoch runs

| seed group | runs | epochs 1 to 3 identical | largest validation-accuracy difference | GPU-hours (10 epochs) |
|---|---|---|---|---|
| 42, 1, 2 | 210 | 44 | 0.4730 | 25.33 |
| 3, 4, 5 | 210 | 40 | 0.4475 | 25.38 |
| 6, 7, 8 | 210 | 42 | 0.2094 | 25.31 |

t = 0 at 1, 2 and 3 epochs, new runs beside RESULTS.md 12.28 (trusted decisions, mean selective regret in points):

| budget | new runs | RESULTS.md 12.28 |
|---|---|---|
| epoch 1 | 26, 1.27 | 25, 0.70 |
| epoch 2 | 20, 0.73 | 20, 0.90 |
| epoch 3 | 21, 0.47 | 24, 0.50 |

#### 1. Per budget

90 decisions. Mean regret of a random candidate (the normalizer): 3.3633 points. At t = 0, with the area under the normalized selective-regret curve over all thresholds, the highest-coverage threshold that meets normalized selective regret <= 0.10 at that budget, and the references at the nearest compute (method setting: mean regret in points at GPU-h).

| budget | compute per decision (GPU-h) | trusted, coverage [95% CI] | selective regret [95% CI] | normalized [95% CI] | area (from coverage) | best threshold [band]: trusted, coverage [95% CI]; normalized [95% CI] | early leader | successive halving | successive halving (factor 3) |
|---|---|---|---|---|---|---|---|---|---|
| 0 (trust all) | 0.029 | 90/90, 1.000 [1.000, 1.000] | 2.59 [1.00, 3.93] | 0.770 [0.296, 1.170] | n/a | n/a | epoch 1: 1.64 at 0.086 | rungs 1,2,3: 1.35 at 0.158 | rungs 1,2,3: 1.56 at 0.122 |
| epoch 1 | 0.115 | 26/90, 0.289 [0.111, 0.522] | 1.27 [0.13, 3.80] | 0.377 [0.039, 1.129] | 0.514 (from 0.011) | 0.40 [0.35, 0.40]: 1/90, 0.011 [0.000, 0.033]; 0.000 [0.000, 0.000] | epoch 1: 1.64 at 0.086 | rungs 1,2,3: 1.35 at 0.158 | rungs 1,2,3: 1.56 at 0.122 |
| epoch 2 | 0.199 | 20/90, 0.222 [0.056, 0.433] | 0.73 [0.00, 3.45] | 0.217 [0.000, 1.026] | 0.479 (from 0.044) | 0.15 [0.15, 0.15]: 7/90, 0.078 [0.000, 0.211]; 0.091 [0.000, 0.639] | epoch 2: 1.21 at 0.170 | start 1: 1.43 at 0.183 | start 1: 1.28 at 0.158 |
| epoch 3 | 0.283 | 21/90, 0.233 [0.044, 0.467] | 0.47 [0.00, 1.48] | 0.139 [0.000, 0.439] | 0.394 (from 0.011) | 0.15 [0.10, 0.15]: 11/90, 0.122 [0.000, 0.322]; 0.075 [0.000, 0.412] | epoch 3: 1.00 at 0.254 | start 2: 1.34 at 0.363 | start 2: 1.32 at 0.315 |
| epoch 5 | 0.452 | 21/90, 0.233 [0.100, 0.422] | 0.74 [0.00, 2.07] | 0.219 [0.000, 0.614] | 0.432 (from 0.033) | 0.10 [0.10, 0.10]: 14/90, 0.156 [0.011, 0.389]; 0.046 [0.000, 0.639] | epoch 5: 1.00 at 0.423 | start 3: 1.10 at 0.498 | start 3: 0.95 at 0.472 |
| epoch 10 | 0.874 | 29/90, 0.322 [0.144, 0.556] | 0.75 [0.00, 1.81] | 0.223 [0.000, 0.540] | 0.410 (from 0.011) | 0.60 [0.45, 0.60]: 1/90, 0.011 [0.000, 0.033]; 0.000 [0.000, 0.000] | epoch 10: 1.02 at 0.845 | start 3: 1.10 at 0.498 | start 3: 0.95 at 0.472 |

#### 2. The domain-specific budget for NLP (k = 0, 1, 2, 3, 5, 10)

The smallest budget (k = 0, trusting every decision, checked first) at which some threshold reaches normalized selective regret at most 0.10 with coverage at least 0.50, with the highest-coverage such threshold (ties to the largest t).

Not reached at any budget up to 10 epochs.

#### 3. Does more budget help

k* is the budget with the lowest mean normalized regret (ties to the lower compute). Flattening is the smallest budget k after which no larger budget lowers the mean normalized regret by more than 0.02. Worsening lists every budget larger than k* with a higher mean normalized regret, with a 95% paired cluster-bootstrap interval of R(k) minus R(k*) on the same resampled tasks. For successive halving the configurations are ordered by compute per decision. Coverage at t = 0: 0 (trust all) 90/90, epoch 1 26/90, epoch 2 20/90, epoch 3 21/90, epoch 5 21/90, epoch 10 29/90.

- selective trust at t = 0 (normalized selective regret over the trusted decisions): lowest mean normalized regret (k*) at epoch 3, 0.139 at 0.283 GPU-h; flattening at epoch 3 (0.139); larger budgets with a higher mean normalized regret than k*:
  - epoch 5: 0.219, difference 0.080 [-0.020, 0.410], 2 resamples without a trusted decision left out
  - epoch 10: 0.223, difference 0.084 [-0.010, 0.349], 2 resamples without a trusted decision left out
- early leader (normalized regret over all 90 decisions): lowest mean normalized regret (k*) at epoch 3, 0.296 at 0.254 GPU-h; flattening at epoch 3 (0.296); larger budgets with a higher mean normalized regret than k*:
  - epoch 5: 0.297, difference 0.000 [-0.065, 0.058]
  - epoch 10: 0.302, difference 0.006 [-0.067, 0.065]
- successive halving (normalized regret over all 90 decisions): lowest mean normalized regret (k*) at start 3, 0.327 at 0.498 GPU-h; flattening at start 3 (0.327); no larger budget has a higher mean normalized regret than k*.
- successive halving (factor 3) (normalized regret over all 90 decisions): lowest mean normalized regret (k*) at start 3, 0.282 at 0.472 GPU-h; flattening at start 3 (0.282); no larger budget has a higher mean normalized regret than k*.

#### 4. The best budget against the references at similar compute

| method | k* | compute (GPU-h) | normalized regret | early leader | successive halving | successive halving (factor 3) |
|---|---|---|---|---|---|---|
| selective trust at t = 0 | epoch 3 | 0.283 | 0.139 | epoch 3: 0.296 [0.097, 0.602] at 0.254 | start 2: 0.399 [0.104, 0.883] at 0.363 | start 2: 0.392 [0.130, 0.798] at 0.315 |
| early leader | epoch 3 | 0.254 | 0.296 | epoch 3: 0.296 [0.097, 0.602] at 0.254 | start 1: 0.426 [0.116, 0.892] at 0.183 | start 2: 0.392 [0.130, 0.798] at 0.315 |
| successive halving | start 3 | 0.498 | 0.327 | epoch 5: 0.297 [0.118, 0.548] at 0.423 | start 3: 0.327 [0.132, 0.608] at 0.498 | start 3: 0.282 [0.082, 0.587] at 0.472 |
| successive halving (factor 3) | start 3 | 0.472 | 0.282 | epoch 5: 0.297 [0.118, 0.548] at 0.423 | start 3: 0.327 [0.132, 0.608] at 0.498 | start 3: 0.282 [0.082, 0.587] at 0.472 |

Selective regret is over the trusted decisions only; the references are over all 90 decisions.

#### 5. Secondary: leave-one-domain-out threshold transfer at k = 5 and 10

As in RESULTS.md 12.28: the threshold is chosen on the two other domains and applied to the held-out one, beside its own best threshold. StudentBERT and Quick-Tune as in 12.28, NLP from the 10-epoch runs. "none": no threshold qualifies.

| budget | held out | rule | source threshold [band] | source coverage, normalized regret | held-out trusted, coverage [95% CI] | held-out normalized regret [95% CI] | own best threshold [band] | own trusted, coverage [95% CI] | own normalized regret [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| epoch 5 | studentbert | primary (by decision) | 0.10 [0.10, 0.10] | 0.094, 0.038 | 8/21, 0.381 [0.095, 0.714] | 0.171 [0.000, 0.685] | 0.15 [0.15, 0.15] | 7/21, 0.333 [0.048, 0.667] | 0.000 [0.000, 0.000] |
| epoch 5 | studentbert | sensitivity analysis (equal domain weight) | 0.05 [0.05, 0.05] | 0.122, 0.067 | 10/21, 0.476 [0.143, 0.857] | 0.137 [0.000, 0.343] | 0.15 [0.15, 0.15] | 7/21, 0.333 [0.048, 0.667] | 0.000 [0.000, 0.000] |
| epoch 5 | nlp | primary (by decision) | 0.15 [0.15, 0.15] | 0.090, 0.000 | 12/90, 0.133 [0.011, 0.322] | 0.053 [0.000, 0.639] | 0.10 [0.10, 0.10] | 14/90, 0.156 [0.011, 0.389] | 0.046 [0.000, 0.639] |
| epoch 5 | nlp | sensitivity analysis (equal domain weight) | -0.30 [-0.30, -0.30] | 0.493, 0.086 | 49/90, 0.544 [0.278, 0.811] | 0.442 [0.132, 0.982] | 0.10 [0.10, 0.10] | 14/90, 0.156 [0.011, 0.389] | 0.046 [0.000, 0.639] |
| epoch 5 | quicktune | primary (by decision) | 0.10 [0.10, 0.10] | 0.198, 0.091 | 3/90, 0.033 [0.000, 0.100] | 0.000 [0.000, 0.000] | 0.35 [-0.30, 0.35] | 3/90, 0.033 [0.000, 0.100] | 0.000 [0.000, 0.000] |
| epoch 5 | quicktune | sensitivity analysis (equal domain weight) | 0.15 [0.15, 0.15] | 0.233, 0.027 | 3/90, 0.033 [0.000, 0.100] | 0.000 [0.000, 0.000] | 0.35 [-0.30, 0.35] | 3/90, 0.033 [0.000, 0.100] | 0.000 [0.000, 0.000] |
| epoch 10 | studentbert | primary (by decision) | 0.05 [0.05, 0.05] | 0.150, 0.094 | 4/21, 0.190 [0.048, 0.381] | 0.057 [0.000, 0.229] | 0.00 [0.00, 0.00] | 5/21, 0.238 [0.048, 0.476] | 0.092 [0.000, 0.229] |
| epoch 10 | studentbert | sensitivity analysis (equal domain weight) | -0.10 [-0.10, -0.10] | 0.217, 0.098 | 10/21, 0.476 [0.190, 0.714] | 0.269 [0.075, 0.412] | 0.00 [0.00, 0.00] | 5/21, 0.238 [0.048, 0.476] | 0.092 [0.000, 0.229] |
| epoch 10 | nlp | primary (by decision) | 0.00 [0.00, 0.00] | 0.099, 0.042 | 29/90, 0.322 [0.144, 0.556] | 0.223 [0.000, 0.540] | 0.60 [0.45, 0.60] | 1/90, 0.011 [0.000, 0.033] | 0.000 [0.000, 0.000] |
| epoch 10 | nlp | sensitivity analysis (equal domain weight) | 0.00 [0.00, 0.00] | 0.152, 0.046 | 29/90, 0.322 [0.144, 0.556] | 0.223 [0.000, 0.540] | 0.60 [0.45, 0.60] | 1/90, 0.011 [0.000, 0.033] | 0.000 [0.000, 0.000] |
| epoch 10 | quicktune | primary (by decision) | 0.60 [0.45, 0.60] | 0.009, 0.000 | 0/90, 0.000 [0.000, 0.000] | n/a | -0.30 [-0.30, -0.30] | 12/90, 0.133 [0.033, 0.267] | 0.050 [0.000, 0.107] |
| epoch 10 | quicktune | sensitivity analysis (equal domain weight) | 0.05 [0.05, 0.05] | 0.212, 0.089 | 6/90, 0.067 [0.000, 0.167] | 0.000 [0.000, 0.000] | -0.30 [-0.30, -0.30] | 12/90, 0.133 [0.033, 0.267] | 0.050 [0.000, 0.107] |

Limitations: five NLP tasks make the intervals wide; the outcome is the published fully fine-tuned performance, not epoch 10 of these 10,000-example runs; epochs 1 to 3 differ from the recorded runs in most runs, by up to the largest difference in the table above (GPU nondeterminism), so every budget comes from the new runs; the NLP result up to epoch 3 was known before this extension was designed.

Files: nlp_budget_extension.tsv adb591aa86fbaed5a606da575b2c288f, nlp_budget_extension_references.tsv 2e655e0112e3d896d70a75f36579b0e0

_Read: more early fine-tuning does not bring NLP to the selective-trust target. At t = 0 the trusted
decisions are 26/90, 20/90, 21/90, 21/90 and 29/90 at 1, 2, 3, 5 and 10 epochs, with normalized
selective regret 0.377, 0.217, 0.139, 0.219 and 0.223, for 0.115 to 0.874 GPU-hours per decision. No
budget up to 10 epochs reaches normalized selective regret at most 0.10 with coverage at least 0.50;
the highest coverage under the regret target is 14/90 (0.156) at 5 epochs. Selective trust and the
early leader are lowest at 3 epochs (0.139 and 0.296) and flatten there; at 5 and 10 epochs
selective trust is higher by 0.080 [-0.020, 0.410] and 0.084 [-0.010, 0.349], and the early leader
by 0.000 and 0.006, intervals that span zero, so a worsening is not shown. Successive halving is
lowest at its largest configuration (start 3: 0.327 at 0.498 GPU-hours; factor 3: 0.282 at 0.472).
Near 0.25 to 0.28 GPU-hours, selective trust at 3 epochs has 0.139 over its 21 trusted decisions,
the early leader at 3 epochs 0.296 over all 90, and successive halving 0.392 to 0.399; selective
regret counts only the trusted decisions, so these are not like for like. With 12.28 (StudentBERT
reaches the target at 2 epochs, Quick-Tune does not), NLP's answer stays not reached up to 10
epochs; at 5 and 10 epochs a threshold chosen on the two other domains still misses on some held-out
domains (held-out StudentBERT 0.171 at 5 epochs, held-out NLP 0.442 under equal domain weight at 5,
held-out Quick-Tune 0/90 at 10). The reruns did not replay the recorded first epochs exactly (44, 40
and 42 of 210 runs per seed group repeat epochs 1 to 3; the largest difference is 0.4730), and the t
= 0 rows at 1 to 3 epochs moved with them (1.27 points at 1 epoch against 0.70 in 12.28). Five tasks
make the intervals wide, and the outcome is the published fully fine-tuned performance._
### 12.30 Noise floor per domain, and every method relative to it (analysis/noise_floor.py, commit 6f269bf, job 12136043)

Task: Prof. Hazra (2026-11-24, before drafting Paper C): a noise-floor analysis per domain, to show
whether the regret differences between methods are meaningfully above the variability from using one
validation seed instead of the multi-seed final outcome, with methods reported relative to the floor
and paired bootstrap uncertainty. The floor and read-outs follow a design written on 2026-11-24
before analysis/noise_floor.py; at Harry's request the design is kept outside the repository, so no
commit documents its date, and the analysis is described as designed before the run, not as
pre-specified. The script (commit 4d49821) was not run on real data before its first run (job
12135833); after the audit its Quick-Tune wording was corrected, as marked in the report, and it was
rerun (commit 6f269bf, job 12136043) with the same table (noise_floor.tsv has the same md5 in both
runs). Floor: the regret of choosing by one seed's (or one run's) complete fine-tuning, the best
validation score over its epochs; StudentBERT and Quick-Tune decisions as in 12.27, NLP from the
10-epoch campaign of 12.29. The StudentBERT floor at the estimator seeds is the 12.27 row full
fine-tuning, best validation, known before. Descriptive.

Files (md5): noise_floor.py fcc5c16b485fc061a04ff43599851224, test_noise_floor.py
5c20425bfe2eb6cee8d399aaa6f66d4f, nlp_budget_extension.py 39d08a0959e4f99f5c01510912b2b840,
reversal_table.py 6f56ff64bd2bc6b86637c4a8eb464b51, regret_compute.py
fd4288700ffd9af74fe92911371935ae, noise_floor_report.md c0b276810b4dfd6c3fc7ca05e9c996fc,
noise_floor.tsv 39a48176dd549371d5b56cd89a781834, executions.tsv 92a3e5250b50727f911fe6f09d4b225b,
trajectories.tsv e93329cbc69ae8a3af843c544d43793b, reversal_pairs.tsv
8f57558d142869b3e34a35ad18d095f7, tg1_task2_kt_algebra2005.jsonl 268ff75c4c59a23b2fa4d9bcc85b60aa,
tg1_task2_kt_algebra2006.jsonl d0f758940ee42bfdd8275e00ff12fe49, tg1_task2_kt_assist2009.jsonl
5823dd2999f96921409b9cce5272d924, tg1_task2_kt_assist2017.jsonl f8a29e8c85e022e3306350ee159e31ff,
tg1_task2_kt_bridge2006.jsonl 38235d16b5d18d72d97bb9d8b9a9a0e0, tg1_task2_kt_ednet.jsonl
59096f8a8941896c18ca60515ffdd9f2, tg1_task2_kt_junyi.jsonl bc1af53d5d042f25d57057952795dad0,
nlp_head_pairs.tsv d023dd9346eb1020afa11d1dacbaf579, p2_head_pairs.tsv
9123ad27c17496af7f5f51cdb1987bc1, p3_head_pairs.tsv 9bad0afce43c426c8105473a6b8f918c,
task4_scores.jsonl 9dfe5d930681ec04d79cb52c149bbc4e, task4_scores_fresh.jsonl
20c403eb43ade91bac987af41538299e, task4_scores_p3.jsonl 34ef47d06f1e7d155c585547a51f9220,
qt_scores.jsonl b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl
50ea8a9ba8fd20713c25211fd30c6f8e. The design file is outside the repository and is not pinned.

Report as written by analysis/noise_floor.py (headings lowered two levels):


Evidence classes: the floor and the read-outs follow a design written on 2026-11-24 before this script, kept outside the repository at Harry's request, so no commit documents its date; the read-outs are descriptive. The StudentBERT floor at the estimator seeds is the RESULTS.md 12.27 row "full fine-tuning, best validation", known before. NLP uses the 10-epoch campaign of analysis/nlp_budget_extension.py. Floor: the regret, against the domain's outcome, of choosing by one seed's (or one run's) complete fine-tuning, the best validation score over its epochs. Excess: the mean over the decisions a method follows of its regret minus the decision's floor, with a 95% paired cluster bootstrap (targets or tasks, 10,000 resamples, random.Random(0)). The script was not run on real data before this run; its tests use synthetic data.

Inputs: trajectories.tsv e93329cbc69ae8a3af843c544d43793b, reversal_pairs.tsv 8f57558d142869b3e34a35ad18d095f7, tg1_task2_kt_algebra2005.jsonl 268ff75c4c59a23b2fa4d9bcc85b60aa, tg1_task2_kt_algebra2006.jsonl d0f758940ee42bfdd8275e00ff12fe49, tg1_task2_kt_assist2009.jsonl 5823dd2999f96921409b9cce5272d924, tg1_task2_kt_assist2017.jsonl f8a29e8c85e022e3306350ee159e31ff, tg1_task2_kt_bridge2006.jsonl 38235d16b5d18d72d97bb9d8b9a9a0e0, tg1_task2_kt_ednet.jsonl 59096f8a8941896c18ca60515ffdd9f2, tg1_task2_kt_junyi.jsonl bc1af53d5d042f25d57057952795dad0, nlp_head_pairs.tsv d023dd9346eb1020afa11d1dacbaf579, p2_head_pairs.tsv 9123ad27c17496af7f5f51cdb1987bc1, p3_head_pairs.tsv 9bad0afce43c426c8105473a6b8f918c, task4_scores.jsonl 9dfe5d930681ec04d79cb52c149bbc4e, task4_scores_fresh.jsonl 20c403eb43ade91bac987af41538299e, task4_scores_p3.jsonl 34ef47d06f1e7d155c585547a51f9220, qt_scores.jsonl b7f48f2f5c6ee9c45ce17201a6d2f9ea, extract_log.jsonl 50ea8a9ba8fd20713c25211fd30c6f8e, executions.tsv 92a3e5250b50727f911fe6f09d4b225b. Design files: selective_trust_design.md f63bfdc6bc0df97a97fb2fc97ef0eddd, selective_trust_amendment_1.md de8a55b63b442dd2d1f28bb7bb2c1c9e, selective_trust_amendment_2.md e8c9ae36be77cfb3fd887b1a6f0a82f2, nlp_budget_extension_design.md d82d64f47e3d23f35a94316446a0041b. Code: nlp_budget_extension.py 39d08a0959e4f99f5c01510912b2b840, reversal_table.py 6f56ff64bd2bc6b86637c4a8eb464b51, regret_compute.py fd4288700ffd9af74fe92911371935ae. NLP 10-epoch runs: 630 (manifest md5 b9301dfc082b1e045459546c26521f56).

Checks passed: the recorded decisions of all three domains reproduce every RESULTS.md 12.27 "warning, abstain" row; every StudentBERT and Quick-Tune method row reproduces its 12.27 row; the NLP reference rows reproduce 12.27 on the recorded runs; the 10-epoch runs match their recorded runs in data and settings; the Quick-Tune run medians reproduce the 12.27 curves; the executions.tsv 6-seed means reproduce the StudentBERT gold.

#### studentbert (test AUC)

Primary floor: 0.0003 [0.0000, 0.0006] test AUC, normalized 0.033; zero regret on 15 of 21 decisions.

Spread over the 6 fine-tuning seeds (each seed's pick, mean over the 7 targets; ties to the candidate name): seed 1 0.0004, seed 2 0.0009, seed 3 0.0001, seed 4 0.0001, seed 5 0.0004, seed 42 0.0003; range 0.0001 to 0.0009.

Secondary, the seed oracle (the best test AUC of one seed against the 6-seed mean): seed 1 0.0002, seed 2 0.0002, seed 3 0.0000, seed 4 0.0001, seed 5 0.0000, seed 42 0.0001; mean 0.0001.

| method | setting | decisions followed | compute (GPU-h) | regret | floor on those decisions | excess over the floor [95% CI] | normalized excess [95% CI] | reading | difference from the static choice [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| static H-score |  | 21/21 | 0.008 | 0.0012 | 0.0003 | 0.0010 [0.0000, 0.0020] | 0.131 [0.003, 0.269] | above the floor | n/a |
| early leader | epoch 1 | 21/21 | 0.039 | 0.0031 | 0.0003 | 0.0029 [0.0004, 0.0068] | 0.383 [0.050, 0.908] | above the floor | 0.0019 [0.0000, 0.0049] |
| override rule | epoch 1 | 21/21 | 0.047 | 0.0031 | 0.0003 | 0.0029 [0.0004, 0.0068] | 0.383 [0.050, 0.908] | above the floor | 0.0019 [0.0000, 0.0049] |
| warning, abstain | epoch 1 | 17/21 | 0.047 | 0.0009 | 0.0003 | 0.0006 [-0.0000, 0.0016] | 0.084 [-0.002, 0.210] | not distinguishable from the floor | 0.0000 [0.0000, 0.0000] |
| early leader | epoch 2 | 21/21 | 0.078 | 0.0010 | 0.0003 | 0.0008 [-0.0002, 0.0023] | 0.101 [-0.022, 0.309] | not distinguishable from the floor | -0.0002 [-0.0013, 0.0007] |
| override rule | epoch 2 | 21/21 | 0.086 | 0.0011 | 0.0003 | 0.0008 [-0.0002, 0.0024] | 0.107 [-0.026, 0.315] | not distinguishable from the floor | -0.0002 [-0.0010, 0.0006] |
| warning, abstain | epoch 2 | 12/21 | 0.086 | 0.0003 | 0.0003 | 0.0000 [-0.0001, 0.0001] | 0.001 [-0.013, 0.016] | not distinguishable from the floor | 0.0000 [0.0000, 0.0000] |
| early leader | epoch 3 | 21/21 | 0.118 | 0.0006 | 0.0003 | 0.0004 [-0.0004, 0.0015] | 0.048 [-0.058, 0.200] | not distinguishable from the floor | -0.0006 [-0.0016, 0.0001] |
| override rule | epoch 3 | 21/21 | 0.125 | 0.0006 | 0.0003 | 0.0004 [-0.0004, 0.0015] | 0.051 [-0.058, 0.204] | not distinguishable from the floor | -0.0006 [-0.0015, 0.0002] |
| warning, abstain | epoch 3 | 12/21 | 0.125 | 0.0009 | 0.0001 | 0.0008 [-0.0001, 0.0024] | 0.110 [-0.015, 0.316] | not distinguishable from the floor | 0.0000 [0.0000, 0.0000] |
| early leader | epoch 5 | 21/21 | 0.196 | 0.0006 | 0.0003 | 0.0003 [-0.0005, 0.0015] | 0.046 [-0.063, 0.198] | not distinguishable from the floor | -0.0006 [-0.0016, 0.0001] |
| override rule | epoch 5 | 21/21 | 0.204 | 0.0005 | 0.0003 | 0.0003 [-0.0005, 0.0014] | 0.034 [-0.070, 0.189] | not distinguishable from the floor | -0.0007 [-0.0016, -0.0000] |
| warning, abstain | epoch 5 | 10/21 | 0.204 | 0.0010 | 0.0000 | 0.0010 [-0.0002, 0.0026] | 0.132 [-0.026, 0.343] | not distinguishable from the floor | 0.0000 [0.0000, 0.0000] |
| early leader | epoch 10 | 21/21 | 0.392 | 0.0021 | 0.0003 | 0.0019 [0.0000, 0.0039] | 0.249 [0.002, 0.522] | above the floor | 0.0009 [-0.0017, 0.0037] |
| override rule | epoch 10 | 21/21 | 0.400 | 0.0017 | 0.0003 | 0.0014 [0.0000, 0.0032] | 0.187 [0.000, 0.427] | not distinguishable from the floor | 0.0004 [-0.0018, 0.0029] |
| warning, abstain | epoch 10 | 5/21 | 0.400 | 0.0007 | 0.0007 | 0.0000 [0.0000, 0.0000] | 0.000 [0.000, 0.000] | not distinguishable from the floor | 0.0000 [0.0000, 0.0000] |
| early leader | epoch 20 | 21/21 | 0.784 | 0.0017 | 0.0003 | 0.0015 [0.0003, 0.0032] | 0.197 [0.045, 0.427] | above the floor | 0.0005 [-0.0014, 0.0027] |
| override rule | epoch 20 | 21/21 | 0.792 | 0.0017 | 0.0003 | 0.0014 [0.0002, 0.0032] | 0.188 [0.028, 0.422] | above the floor | 0.0004 [-0.0013, 0.0026] |
| warning, abstain | epoch 20 | 10/21 | 0.792 | 0.0009 | 0.0004 | 0.0005 [0.0000, 0.0014] | 0.062 [0.000, 0.182] | not distinguishable from the floor | 0.0000 [0.0000, 0.0000] |
| successive halving | start 1 | 21/21 | 0.078 | 0.0007 | 0.0003 | 0.0004 [-0.0002, 0.0011] | 0.058 [-0.022, 0.144] | not distinguishable from the floor | -0.0006 [-0.0011, -0.0001] |
| successive halving | start 2 | 21/21 | 0.157 | 0.0007 | 0.0003 | 0.0005 [-0.0002, 0.0013] | 0.064 [-0.024, 0.167] | not distinguishable from the floor | -0.0005 [-0.0016, 0.0007] |
| successive halving | start 5 | 21/21 | 0.394 | 0.0013 | 0.0003 | 0.0011 [0.0002, 0.0022] | 0.140 [0.032, 0.292] | above the floor | 0.0001 [-0.0016, 0.0018] |
| successive halving (factor 3) | start 1 | 21/21 | 0.068 | 0.0008 | 0.0003 | 0.0006 [-0.0004, 0.0017] | 0.075 [-0.052, 0.228] | not distinguishable from the floor | -0.0004 [-0.0009, -0.0000] |
| successive halving (factor 3) | start 2 | 21/21 | 0.138 | 0.0014 | 0.0003 | 0.0012 [-0.0001, 0.0032] | 0.153 [-0.018, 0.428] | not distinguishable from the floor | 0.0002 [-0.0012, 0.0016] |
| successive halving (factor 3) | start 5 | 21/21 | 0.344 | 0.0009 | 0.0003 | 0.0006 [0.0001, 0.0013] | 0.086 [0.019, 0.166] | above the floor | -0.0003 [-0.0017, 0.0009] |
| full fine-tuning | best validation | 21/21 | 0.784 | 0.0003 | 0.0003 | 0.0000 [0.0000, 0.0000] | 0.000 [0.000, 0.000] | not distinguishable from the floor | -0.0010 [-0.0020, -0.0000] |

#### nlp (points)

Primary floor: 0.93 [0.16, 1.97] points, normalized 0.278; zero regret on 54 of 90 decisions.

Spread over the 9 estimator seeds (mean over 10 decisions each): seed 1 1.12, seed 2 1.22, seed 3 1.82, seed 4 0.72, seed 5 0.34, seed 6 0.33, seed 7 0.91, seed 8 0.93, seed 42 1.03; range 0.33 to 1.82. The NLP outcome is the published fully fine-tuned performance, so this floor includes the difference between these 10,000-example runs and the published protocol.

Secondary, within the protocol (one seed's epoch-10 pick against the mean epoch-10 validation accuracy of the other 8 seeds, validation points, not subtracted from the methods): 1.33.

| method | setting | decisions followed | compute (GPU-h) | regret | floor on those decisions | excess over the floor [95% CI] | normalized excess [95% CI] | reading | difference from the static choice [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| static H-score |  | 90/90 | 0.029 | 2.59 | 0.93 | 1.65 [0.16, 3.21] | 0.492 [0.047, 0.953] | above the floor | n/a |
| early leader | epoch 1 | 90/90 | 0.086 | 1.64 | 0.93 | 0.71 [-0.04, 1.45] | 0.210 [-0.012, 0.431] | not distinguishable from the floor | -0.95 [-2.53, 0.47] |
| override rule | epoch 1 | 90/90 | 0.115 | 2.02 | 0.93 | 1.09 [0.13, 2.17] | 0.324 [0.039, 0.646] | above the floor | -0.57 [-2.13, 0.59] |
| warning, abstain | epoch 1 | 26/90 | 0.115 | 1.27 | 0.78 | 0.49 [-0.13, 2.18] | 0.145 [-0.037, 0.647] | not distinguishable from the floor | 0.00 [0.00, 0.00] |
| early leader | epoch 2 | 90/90 | 0.170 | 1.21 | 0.93 | 0.28 [-0.15, 0.68] | 0.083 [-0.046, 0.202] | not distinguishable from the floor | -1.37 [-2.93, 0.16] |
| override rule | epoch 2 | 90/90 | 0.199 | 2.13 | 0.93 | 1.19 [0.08, 2.59] | 0.354 [0.023, 0.770] | above the floor | -0.46 [-2.40, 1.52] |
| warning, abstain | epoch 2 | 20/90 | 0.199 | 0.73 | 0.48 | 0.24 [-0.20, 2.56] | 0.073 [-0.061, 0.761] | not distinguishable from the floor | 0.00 [0.00, 0.00] |
| early leader | epoch 3 | 90/90 | 0.254 | 1.00 | 0.93 | 0.06 [-0.13, 0.21] | 0.018 [-0.039, 0.062] | not distinguishable from the floor | -1.59 [-3.21, -0.17] |
| override rule | epoch 3 | 90/90 | 0.283 | 1.59 | 0.93 | 0.66 [-0.02, 1.34] | 0.196 [-0.005, 0.398] | not distinguishable from the floor | -0.99 [-2.71, 0.38] |
| warning, abstain | epoch 3 | 21/90 | 0.283 | 0.47 | 0.55 | -0.08 [-0.17, 0.00] | -0.025 [-0.052, 0.000] | not distinguishable from the floor | 0.00 [0.00, 0.00] |
| early leader | epoch 5 | 90/90 | 0.423 | 1.00 | 0.93 | 0.06 [-0.19, 0.34] | 0.019 [-0.057, 0.102] | not distinguishable from the floor | -1.59 [-3.09, -0.26] |
| override rule | epoch 5 | 90/90 | 0.452 | 1.42 | 0.93 | 0.48 [0.01, 1.17] | 0.143 [0.002, 0.348] | above the floor | -1.17 [-2.81, -0.02] |
| warning, abstain | epoch 5 | 21/90 | 0.452 | 0.74 | 0.39 | 0.35 [-0.30, 1.53] | 0.103 [-0.090, 0.456] | not distinguishable from the floor | 0.00 [0.00, 0.00] |
| early leader | epoch 10 | 90/90 | 0.845 | 1.02 | 0.93 | 0.08 [-0.15, 0.33] | 0.024 [-0.044, 0.098] | not distinguishable from the floor | -1.57 [-3.02, -0.21] |
| override rule | epoch 10 | 90/90 | 0.874 | 1.62 | 0.93 | 0.68 [0.04, 1.41] | 0.203 [0.011, 0.419] | above the floor | -0.97 [-2.60, 0.31] |
| warning, abstain | epoch 10 | 29/90 | 0.874 | 0.75 | 0.60 | 0.15 [-0.48, 1.03] | 0.044 [-0.143, 0.307] | not distinguishable from the floor | 0.00 [0.00, 0.00] |
| successive halving | rungs 1,2,3 | 90/90 | 0.158 | 1.35 | 0.93 | 0.42 [-0.23, 1.10] | 0.124 [-0.069, 0.326] | not distinguishable from the floor | -1.24 [-2.96, 0.43] |
| successive halving | start 1 | 90/90 | 0.183 | 1.43 | 0.93 | 0.50 [-0.19, 1.19] | 0.148 [-0.057, 0.353] | not distinguishable from the floor | -1.16 [-2.87, 0.36] |
| successive halving | start 2 | 90/90 | 0.363 | 1.34 | 0.93 | 0.41 [-0.09, 1.05] | 0.121 [-0.027, 0.313] | not distinguishable from the floor | -1.25 [-3.07, 0.45] |
| successive halving | start 3 | 90/90 | 0.498 | 1.10 | 0.93 | 0.16 [0.04, 0.29] | 0.049 [0.011, 0.087] | above the floor | -1.49 [-3.05, -0.05] |
| successive halving (factor 3) | rungs 1,2,3 | 90/90 | 0.122 | 1.56 | 0.93 | 0.63 [-0.10, 1.49] | 0.187 [-0.031, 0.444] | not distinguishable from the floor | -1.03 [-2.84, 0.70] |
| successive halving (factor 3) | start 1 | 90/90 | 0.158 | 1.28 | 0.93 | 0.35 [-0.15, 0.91] | 0.103 [-0.043, 0.269] | not distinguishable from the floor | -1.31 [-3.01, 0.29] |
| successive halving (factor 3) | start 2 | 90/90 | 0.315 | 1.32 | 0.93 | 0.38 [-0.10, 0.87] | 0.114 [-0.030, 0.258] | not distinguishable from the floor | -1.27 [-2.97, 0.27] |
| successive halving (factor 3) | start 3 | 90/90 | 0.472 | 0.95 | 0.93 | 0.02 [-0.31, 0.30] | 0.005 [-0.092, 0.091] | not distinguishable from the floor | -1.64 [-3.25, -0.26] |

#### quicktune (top-1 points)

Primary floor: 23.62 [19.58, 27.62] top-1 points, normalized 0.540; zero regret on 0 of 90 decisions.

Spread over the 10000 draws of one run per model (2.5th and 97.5th percentiles of the mean floor): 15.02 to 32.73. Ties go to the model name. Added after the first run (2026-11-25): every Quick-Tune run has its own random hyperparameters (docs/trajectory_design.md, rule A), so one run per model measures the variability of a hyperparameter draw, not of a seed; the methods read curves that are medians over the same runs that define the outcome, so this floor is not comparable with them and the readings below are not a seed-noise comparison.

| method | setting | decisions followed | compute (GPU-h) | regret | floor on those decisions | excess over the floor [95% CI] | normalized excess [95% CI] | reading | difference from the static choice [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| static H-score |  | 90/90 | 0.040 | 40.56 | 23.62 | 16.95 [8.59, 25.53] | 0.387 [0.196, 0.584] | above the floor | n/a |
| early leader | epoch 1 | 90/90 | 0.094 | 33.58 | 23.62 | 9.96 [-0.27, 20.44] | 0.228 [-0.006, 0.467] | not distinguishable from the floor | -6.98 [-20.84, 7.47] |
| override rule | epoch 1 | 90/90 | 0.094 | 32.44 | 23.62 | 8.83 [-1.15, 19.27] | 0.202 [-0.026, 0.440] | not distinguishable from the floor | -8.12 [-21.72, 6.17] |
| warning, abstain | epoch 1 | 3/90 | 0.094 | 16.56 | 9.72 | 6.84 [6.84, 6.84] | 0.156 [0.156, 0.156] | above the floor | 0.00 [0.00, 0.00] |
| early leader | epoch 2 | 90/90 | 0.189 | 15.18 | 23.62 | -8.44 [-14.96, -1.60] | -0.193 [-0.342, -0.036] | below the floor | -25.39 [-36.86, -14.01] |
| override rule | epoch 2 | 90/90 | 0.189 | 15.18 | 23.62 | -8.44 [-14.96, -1.60] | -0.193 [-0.342, -0.036] | below the floor | -25.39 [-36.86, -14.01] |
| warning, abstain | epoch 2 | 0 | 0.189 | n/a | n/a | n/a | n/a | n/a | n/a |
| early leader | epoch 3 | 90/90 | 0.283 | 13.20 | 23.62 | -10.42 [-16.87, -3.49] | -0.238 [-0.386, -0.080] | below the floor | -27.37 [-39.26, -15.42] |
| override rule | epoch 3 | 90/90 | 0.283 | 13.20 | 23.62 | -10.42 [-16.87, -3.49] | -0.238 [-0.386, -0.080] | below the floor | -27.37 [-39.26, -15.42] |
| warning, abstain | epoch 3 | 0 | 0.283 | n/a | n/a | n/a | n/a | n/a | n/a |
| early leader | epoch 5 | 90/90 | 0.474 | 9.79 | 23.62 | -13.83 [-19.90, -7.50] | -0.316 [-0.455, -0.171] | below the floor | -30.77 [-41.70, -19.85] |
| override rule | epoch 5 | 90/90 | 0.474 | 9.79 | 23.62 | -13.83 [-19.90, -7.50] | -0.316 [-0.455, -0.171] | below the floor | -30.77 [-41.70, -19.85] |
| warning, abstain | epoch 5 | 3/90 | 0.474 | 0.00 | 8.69 | -8.69 [-8.69, -8.69] | -0.199 [-0.199, -0.199] | below the floor | 0.00 [0.00, 0.00] |
| early leader | epoch 10 | 90/90 | 0.953 | 1.06 | 23.62 | -22.55 [-26.96, -18.06] | -0.515 [-0.616, -0.413] | below the floor | -39.50 [-50.09, -29.24] |
| override rule | epoch 10 | 90/90 | 0.953 | 1.06 | 23.62 | -22.55 [-26.96, -18.06] | -0.515 [-0.616, -0.413] | below the floor | -39.50 [-50.09, -29.24] |
| warning, abstain | epoch 10 | 6/90 | 0.953 | 0.00 | 13.78 | -13.78 [-18.87, -8.69] | -0.315 [-0.431, -0.199] | below the floor | 0.00 [0.00, 0.00] |
| early leader | epoch 20 | 90/90 | 1.911 | 1.08 | 23.62 | -22.53 [-26.83, -18.11] | -0.515 [-0.613, -0.414] | below the floor | -39.48 [-50.23, -29.06] |
| override rule | epoch 20 | 90/90 | 1.911 | 1.08 | 23.62 | -22.53 [-26.83, -18.11] | -0.515 [-0.613, -0.414] | below the floor | -39.48 [-50.23, -29.06] |
| warning, abstain | epoch 20 | 6/90 | 1.911 | 0.00 | 13.78 | -13.78 [-18.87, -8.69] | -0.315 [-0.431, -0.199] | below the floor | 0.00 [0.00, 0.00] |
| early leader | epoch 30 | 90/90 | 2.901 | 0.25 | 23.62 | -23.37 [-27.27, -19.43] | -0.534 [-0.623, -0.444] | below the floor | -40.31 [-50.68, -30.18] |
| override rule | epoch 30 | 90/90 | 2.901 | 0.25 | 23.62 | -23.37 [-27.27, -19.43] | -0.534 [-0.623, -0.444] | below the floor | -40.31 [-50.68, -30.18] |
| warning, abstain | epoch 30 | 6/90 | 2.901 | 0.00 | 13.78 | -13.78 [-18.87, -8.69] | -0.315 [-0.431, -0.199] | below the floor | 0.00 [0.00, 0.00] |
| early leader | epoch 40 | 90/90 | 3.923 | 0.02 | 23.62 | -23.59 [-27.60, -19.57] | -0.539 [-0.631, -0.447] | below the floor | -40.54 [-50.95, -30.39] |
| override rule | epoch 40 | 90/90 | 3.923 | 0.02 | 23.62 | -23.59 [-27.60, -19.57] | -0.539 [-0.631, -0.447] | below the floor | -40.54 [-50.95, -30.39] |
| warning, abstain | epoch 40 | 6/90 | 3.923 | 0.00 | 13.78 | -13.78 [-18.87, -8.69] | -0.315 [-0.431, -0.199] | below the floor | 0.00 [0.00, 0.00] |
| early leader | epoch 50 | 90/90 | 4.946 | 0.00 | 23.62 | -23.62 [-27.62, -19.58] | -0.540 [-0.631, -0.447] | below the floor | -40.56 [-51.01, -30.41] |
| override rule | epoch 50 | 90/90 | 4.946 | 0.00 | 23.62 | -23.62 [-27.62, -19.58] | -0.540 [-0.631, -0.447] | below the floor | -40.56 [-51.01, -30.41] |
| warning, abstain | epoch 50 | 6/90 | 4.946 | 0.00 | 13.78 | -13.78 [-18.87, -8.69] | -0.315 [-0.431, -0.199] | below the floor | 0.00 [0.00, 0.00] |
| successive halving | start 1 | 90/90 | 0.417 | 9.30 | 23.62 | -14.31 [-19.41, -9.15] | -0.327 [-0.444, -0.209] | below the floor | -31.26 [-42.64, -19.98] |
| successive halving | start 2 | 90/90 | 0.932 | 2.51 | 23.62 | -21.11 [-25.91, -16.21] | -0.482 [-0.592, -0.371] | below the floor | -38.05 [-49.30, -26.93] |
| successive halving | start 3 | 90/90 | 1.459 | 2.67 | 23.62 | -20.95 [-25.95, -15.55] | -0.479 [-0.593, -0.355] | below the floor | -37.90 [-48.09, -28.08] |
| successive halving (factor 3) | start 1 | 90/90 | 0.287 | 11.65 | 23.62 | -11.96 [-17.68, -5.90] | -0.273 [-0.404, -0.135] | below the floor | -28.91 [-39.63, -18.20] |
| successive halving (factor 3) | start 2 | 90/90 | 0.657 | 4.28 | 23.62 | -19.34 [-24.50, -14.11] | -0.442 [-0.560, -0.322] | below the floor | -36.29 [-47.14, -25.40] |
| successive halving (factor 3) | start 3 | 90/90 | 0.987 | 3.43 | 23.62 | -20.19 [-25.46, -14.55] | -0.461 [-0.582, -0.332] | below the floor | -37.14 [-47.58, -26.83] |

Limitations: StudentBERT has 7 targets and NLP 5 tasks, so intervals are wide; the NLP floor includes the protocol difference to the published outcome; the Quick-Tune floor measures hyperparameter-draw variability and is not comparable with the methods; the StudentBERT floor at the estimator seeds was known before (12.27).

Files: noise_floor.tsv 39a48176dd549371d5b56cd89a781834

_Read: on StudentBERT and NLP the static choice's regret is above the seed floor, and most
multi-fidelity rows are not distinguishable from it. StudentBERT's floor is 0.0003 [0.0000, 0.0006]
test AUC (15 of 21 decisions at zero; 0.0001 to 0.0009 over the 6 seeds; the seed oracle, which no
selection rule can remove, 0.0001); the static H-score choice is above it by 0.0010 [0.0000,
0.0020], the early leader at 2, 3 and 5 epochs and successive halving started at 1 or 2 epochs are
not distinguishable from it, and the early leader at 1, 10 and 20 epochs is above it. NLP's floor is
0.93 [0.16, 1.97] points (0.33 to 1.82 over the 9 seeds; 1.33 validation points within the
protocol); the static choice is above it by 1.65 [0.16, 3.21], the early leader is not
distinguishable from it at any budget (0.71 [-0.04, 1.45] at 1 epoch, 0.06 [-0.13, 0.21] at 3), the
override rule is above it at 1, 2, 5 and 10 epochs, and successive halving is not distinguishable
from it except at start 3 (0.16 [0.04, 0.29]). The static choice's excess over the floor (0.0010
AUC, 1.65 points) is larger than the floor itself (0.0003, 0.93). On Quick-Tune the floor (23.62
top-1 points) measures a hyperparameter draw, not a seed, and is not comparable with the methods,
which read median curves over the same runs that define the outcome; the Quick-Tune rows are not a
seed-noise comparison. StudentBERT has 7 targets and NLP 5 tasks, and the NLP floor includes the
protocol difference to the published outcome._
### 12.31 Corrections and added limitations for 12.24 to 12.30 (audit of 2026-11-25)

Task: an audit of the record (Harry, 2026-11-25) checked the readings of 12.24 to 12.30 against
their tables, design files, code and commit history, and found eight places where a reading says
more than the evidence or leaves out a limitation. This section corrects them. It changes no number
and runs no analysis (descriptive). The earlier sections stay as recorded, because the record is
append-only and the split files list each section's md5; read each of them with the item below that
names it. Every number below is printed in the section it cites, except the accuracy points of item
1, which convert a printed fraction.

Files (md5): scripts/task4_early_ft.py 9c87a401c955fc90aca772e5c17cbf08, tools/compare_ft_epochs.py
16fbc8a6b27de3bec529729e1dec032e, docs/nlp_budget_extension_design.md
d82d64f47e3d23f35a94316446a0041b, docs/qt_hscore_design.md 9131a8e9221faf387cb4ff3c9d87fcf1,
docs/trajectory_amendment2.md 8859f77870d589845fc5d373a6cae773, docs/selective_trust_design.md
f63bfdc6bc0df97a97fb2fc97ef0eddd, analysis/regret_compute.py fd4288700ffd9af74fe92911371935ae,
analysis/noise_floor.py fcc5c16b485fc061a04ff43599851224, docs/regret_compute_design.md
4dc6393deea651bf59aa8144e1ce1608.

1. 12.29, the reruns. The table "The new runs against the recorded 3-epoch runs" gives the largest
   validation-accuracy differences 0.4730, 0.4475 and 0.2094 without a unit, and the _Read_ repeats
   0.4730. They are fractions of validation accuracy, that is 47.30, 44.75 and 20.94 accuracy points
   (val_acc is correct / len(va), scripts/task4_early_ft.py line 174, checked to lie in [0, 1] at
   line 61; the difference is abs(a["val_acc"] - b["val_acc"]), tools/compare_ft_epochs.py line 35).
   The limitations name the cause "(GPU nondeterminism)". That is the cause the design expected
   ("Epochs 1 to 3 may differ slightly from the recorded runs (GPU nondeterminism)",
   docs/nlp_budget_extension_design.md), not a tested one: the check confirms only that each new run
   has the data and settings of its recorded run, and differences of up to 47.30 points are not
   slight. The decision of 12.29 to take every budget from the new runs is unaffected.

2. 12.25, the design order. "Design: docs/qt_hscore_design.md and docs/trajectory_amendment2.md,
   both committed before any feature or score existed" holds for the design, committed in 404c1f7
   (2026-11-10 17:06) as fixed before extraction, together with the feature extraction script. It
   does not hold for amendment 2, committed in c930810 (2026-11-10 20:38), which opens: "Follows
   `docs/qt_hscore_design.md`, whose features now exist for all 30 micro tasks and 24 models (job
   11805425; 720 files)." Amendment 2 was committed after the features existed and before any score:
   it is committed with analysis/qt_score.py and analysis/qt_reversal.py before either runs on the
   real features, as it states.

3. 12.24, the early leader on StudentBERT. "after 3 to 5 epochs the early leader halves H-score's
   regret (0.0006 against 0.0012)" and "A longer budget does not help: the early leader's regret
   rises to 0.0021 after 10 epochs and 0.0017 after 20, worse than H-score alone" compare point
   estimates. The paired differences from the static choice in 12.30 span zero at each of these
   budgets: -0.0006 [-0.0016, 0.0001] at 3 and at 5 epochs, 0.0009 [-0.0017, 0.0037] at 10 and
   0.0005 [-0.0014, 0.0027] at 20. Neither the halving nor the later worsening is shown.

4. 12.28 and 12.29, the StudentBERT operating point. The budget and threshold at which "Only
   StudentBERT meets the pre-specified targets" (2 epochs, t = -0.10: 17/21 at 0.099 [0.012, 0.217])
   are chosen on the same 21 decisions on which they are scored (docs/selective_trust_design.md,
   Reported, item 4: the smallest budget at which some threshold reaches the targets), so this point
   is in-sample and optimistic. The design's out-of-sample test is the leave-one-domain-out transfer
   (Reported, item 3): there a threshold chosen on the two other domains gives held-out StudentBERT
   0.171 at 5 epochs (12.29). 12.28 adds that "the point sits at the target and its interval spans
   it"; the phrase "StudentBERT reaches the target at 2 epochs" in 12.29 carries the same caveat.

5. 12.27, the _Read_. (a) "On StudentBERT every method gets worse beyond about 5 epochs": in the
   12.27 table the early leader (0.0006 at 5 epochs, 0.0021 at 10), the override rule (0.0005,
   0.0017) and successive halving (0.0007 started at 2, 0.0013 started at 5) are higher later, but
   warning, abstain is not (0.0010 at 5, 0.0007 at 10), nor is successive halving with factor 3
   (0.0014 started at 2, 0.0009 started at 5); all of these are point estimates (item 3). (b)
   "static selection is always the cheapest point and never good enough once a little fine-tuning is
   allowed": by the paired differences from the static choice in 12.30, on NLP the early leader at
   3, 5 and 10 epochs (-1.59 [-3.21, -0.17], -1.59 [-3.09, -0.26], -1.57 [-3.02, -0.21]), the
   override rule at 5 epochs (-1.17 [-2.81, -0.02]) and successive halving started at 3 (-1.49
   [-3.05, -0.05]; with factor 3, -1.64 [-3.25, -0.26]) have lower regret than static, and every
   other NLP row, including all at 1 and 2 epochs, has an interval that includes zero. On
   StudentBERT only successive halving started at 1 epoch is clearly lower (-0.0006 [-0.0011,
   -0.0001]); three intervals end at a printed -0.0000 (successive halving with factor 3 started at
   1, the override rule at 5 epochs, full fine-tuning), and every early-leader interval includes or
   touches zero. On Quick-Tune the early leader and the override rule are lower from 2 epochs on but
   not at 1 epoch (-6.98 [-20.84, 7.47] and -8.12 [-21.72, 6.17]), and there the methods read the
   same runs that define the outcome (12.27, 12.30).

6. Quick-Tune compute in 12.27 to 12.30. Each model's curve is the median over its complete runs (at
   least 3; median 10 per included pair, 12.26), while its compute is the median of one run's time
   per epoch (analysis/regret_compute.py lines 267 to 271). The Quick-Tune methods are therefore
   charged for one run per model while their curves summarize several, which understates their
   compute relative to the information they use. No number changes; this is a limitation of the
   Quick-Tune fine-tuning compute in 12.27 to 12.30.

7. 12.30, the StudentBERT floor. The floor chooses by one seed's best validation score
   (analysis/noise_floor.py lines 80 to 81), and that seed's test AUC is one of the six averaged
   into the gold (lines 122 to 123 check each decision's outcome against the 6-seed mean; 12.30
   lists the same six seeds, 1, 2, 3, 4, 5 and 42, for the spread and for the seed oracle). The
   choosing seed thus also enters the outcome, which biases the StudentBERT floor downward
   (optimistic); the NLP floor is measured against the published outcome, which our runs do not
   enter. Separately, "The static choice's excess over the floor (0.0010 AUC, 1.65 points) is larger
   than the floor itself (0.0003, 0.93)" compares point estimates: the excess intervals [0.0000,
   0.0020] and [0.16, 3.21] contain 0.0003 and 0.93.

8. 12.27, the design order. "docs/regret_compute_design.md was committed before the script existed"
   should read "before the script was committed": the design is commit ef198fe (2026-11-16 18:16:56)
   and the script, 441 lines with 67 lines of tests, commit 935ce90 seven minutes later (18:24:04).
   12.27 already states that "the script was test-run on the StudentBERT scores with stand-in costs
   while it was written".

_Read: no number in 12.24 to 12.30 changes. Items 3, 5 and 7 narrow three readings to what the
intervals show: on StudentBERT no early-leader difference from the static choice excludes zero as
printed, longer budgets are not shown to hurt, and the static choice's excess over the seed floor is
not shown to exceed the floor. Item 4 marks StudentBERT's operating point as in-sample; items 6 and
7 add two limitations (Quick-Tune compute is charged for one run, the StudentBERT floor is
optimistic); items 1, 2 and 8 correct statements of fact. What stands: on NLP the early leader at 3,
5 and 10 epochs, the override rule at 5 and successive halving at its largest setting have lower
regret than the static choice with intervals that exclude zero, and on StudentBERT successive
halving started at 1 epoch does._
