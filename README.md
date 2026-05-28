# StudentBERT

BERT-style pretrained foundation model for student learning sequences.

Pretrain a transformer on student interaction sequences — `(skill, correctness, response_time_bin)` tuples — to learn the general structure of how students learn, then fine-tune cheaply for downstream educational tasks (knowledge tracing, dropout prediction, next-activity prediction).

## Datasets
- **ASSISTments 2017** — math tutoring logs
- **EdNet (KT1)** — Korean TOEIC English prep
- **Junyi Academy** — Taiwanese K-12 math

All data lives on the cluster at `/work/algl` — **never committed to git**. Local dev uses a tiny slice under `data/sample/`.

## Setup (M1 MacBook)
```bash
# 1. Create environment
conda env create -f environment.yml
conda activate studentbert

# 2. Verify
python scripts/check_setup.py

# 3. Enable formatting hooks
pre-commit install

# 4. Log in to experiment tracking
wandb login
```

## Project structure
```
studentbert/
├── configs/         # hydra YAML configs for runs
├── data/sample/     # tiny local slice for dev (committed); real data on cluster
├── notebooks/       # exploration
├── scripts/         # preprocess_*.py, train_*.py, check_setup.py
├── src/
│   ├── data/        # dataset classes, tokenization
│   ├── models/      # backbone + task heads
│   ├── training/    # trainer, masking, callbacks
│   └── eval/        # AUC, ECE, F1 metrics
└── tests/
```

## Workflow
1. Pipeline development on M1 with a small data slice (MPS backend).
2. Real pretraining/fine-tuning runs on Northeastern Explorer HPC (CUDA), data + checkpoints in `/work/algl`.
3. All runs tracked in W&B project `StudentBERT` (public; link shared with advisor).

## Metrics
- **AUC-ROC** — headline, knowledge tracing
- **ECE** — calibration / confidence trustworthiness
- **F1 (minority class)** — dropout prediction

## Baselines
DKT, DKVMN, AKT, SAINT (closest comparable). Reproduced via `pykt-toolkit` preprocessing to avoid KC-level label leakage.
