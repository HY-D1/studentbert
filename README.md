# StudentBERT

BERT-style pretrained foundation model for student learning sequences.

A transformer is pretrained on student interaction sequences, `(skill, correctness, response_time_bin)` tuples, with masked-interaction modeling, then fine-tuned for downstream educational tasks: knowledge tracing (next-step correctness), early-dropout prediction, and next-skill prediction. The research contribution is a systematic transferability study: what governs cross-dataset educational pretraining transfer, where the gains live (low-resource targets), where they vanish (full target scale), and a mechanistic account via masked-skill probing and objective ablation.

**All headline numbers, per experiment, live in [`RESULTS.md`](RESULTS.md)**. Its base sections came from `analysis/collect_all_results.py`; every later section was added by an audited patch in `tools/patches/` (`patch_results_*.py`), and the collector refuses to overwrite those. Experiment index (week 6 only): [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Datasets (7)

| Dataset | Students | Skills | Domain |
|---|---|---|---|
| ASSISTments 2017 | 1,708 | 102 | US math tutoring |
| EdNet KT1 | 441,997 | 142 | Korean TOEIC English prep |
| Junyi Academy | 61,442 | 1,326 | Taiwanese K-12 math |
| Algebra 2005 (KDD Cup) | 567 | 109 | Cognitive Tutor algebra |
| Bridge to Algebra 2006 (KDD Cup) | 1,130 | 492 | Cognitive Tutor algebra |
| ASSISTments 2009 | 3,119 | 123 | US math tutoring |
| Algebra 2006-2007 (KDD Cup) | 1,310 | 484 | Cognitive Tutor algebra |

Raw data is **never committed**. All data, checkpoints, and logs live on the Northeastern Explorer HPC under `/projects/algl/<user>/studentbert/{raw,processed,checkpoints,logs,code}`. Every dataset is preprocessed to an identical schema: `sequences.npz` (CSR-packed `student_ids/skill/correct/time_bin/offsets`), `skill_vocab.json`, `splits.json` (80/10/10 by student, seed 42), `vocab_stats.md`.

## Repository layout

```
studentbert/
├── src/
│   ├── data/               # dataset classes
│   ├── models/             # edubert, dkt, akt, saint_plus
│   ├── training/           # masking
│   ├── eval/               # AUC, ECE, F1, top-k metrics
│   ├── estimators/         # transferability estimators: LogME, H-score, NLEEP, PARC,
│   │                       #   causal KT features, target-sample protocol
│   └── utils.py            # device and seed helpers
├── scripts/                # preprocess_*.py, pretrain/finetune/downstream entry points,
│                           #   probe_edubert_v2.py, estimator scoring, Task 4 (NLP), checks
├── analysis/               # result collection, benchmark build, estimator evaluation,
│                           #   paper assets (make_paperA_assets.py, make_paperB_assets.py)
├── slurm/                  # archived sbatch job scripts
│   └── generators/         # gen_*.sh that emit one-condition-x-seed sbatch batches
├── tools/                  # cluster helpers: audit_consistency, cleanup_sb (dry-run by default),
│   │                       #   drip_submit / drip_multi, tg1_final_build, reversal_pipeline,
│   │                       #   collect_poster_gaps, probe_rerun_info
│   └── patches/            # audited one-shot patches to scripts, generators, .gitignore and
│                           #   RESULTS.md sections (kept for provenance)
├── configs/                # Paper B encoder recipes log, Task 4 published tables
├── docs/                   # design files committed before each pre-specified analysis
├── poster/                 # poster build scripts
├── tests/
├── RESULTS.md              # canonical consolidated results
└── EXPERIMENTS.md          # experiment index
```

## Setup

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate studentbert
wandb login
pre-commit install
```

GPU runs require CUDA 12.1 builds of PyTorch (`torch 2.5.1+cu121`); cu130 wheels fall back to CPU on the cluster.

On the cluster, jobs do not use the `studentbert` env name: they call the absolute env `/projects/algl/<user>/envs/sb`, and the Task 4 NLP scripts (`scripts/task4_*.py`, which need `transformers` and `datasets`) run in a separate env, `/projects/algl/<user>/envs/t4`.

## Reproducing the pipeline

1. **Preprocess** a dataset to the shared schema: `scripts/preprocess_<dataset>.py` (EdNet reads directly from `EdNet-KT1.zip`, no extraction).
2. **Pretrain** the encoder: `scripts/pretrain_edubert.py` (masked-interaction MLM; predicts masked skill and correctness; `--objective {full,skill_only,correct_only}` for the ablation).
3. **Fine-tune / evaluate**: `scripts/finetune_edubert.py` (KT, causal attention mask at fine-tune) and `scripts/downstream_edubert.py` (`--task dropout|next_skill`, with `--window_censor` and `macro_top1` for leakage-safe dropout and frequency-normalized next-skill).
4. **Probe** representations: `scripts/probe_edubert_v2.py` (masked-skill decodability; v1 is deprecated as circular).
5. **Collect**: `analysis/collect_all_results.py` regenerates the base sections of `RESULTS.md` from logs (write to `RESULTS_generated.md` and diff; it refuses to overwrite the hand-added sections).

Cluster jobs are submitted via sbatch (see `slurm/` for the exact scripts behind every reported number); large sweeps are split one condition x seed per job. All runs log to W&B project `StudentBERT`.

## Evaluation conventions

3-seed minimum (8 seeds + paired bootstrap for high-variance comparisons); paired bootstrap 95% CIs for all key gaps; best-checkpoint-by-validation reported; leakage diagnostics (`scripts/check_dropout_leak.py`) required before any length-derived label is reported; macro and weighted OVR AUC plus macro-top1 for skewed multiclass targets.

## Baselines

DKT (Piech et al., 2015), AKT (Ghosh et al., 2020), SAINT+ (Choi et al., 2020), implemented in `src/models/` and run through `scripts/train_baseline.py` on identical splits.
