#!/usr/bin/env python3
"""Record the seed2 rerun outcome in RESULTS.md section 8.2 and close the cell at n=7.

Run from the repo root:  python3 tools/patches/close_ednet_dropout_seed2.py
Idempotent: exits cleanly if already applied.

Use this INSTEAD of finish_ednet_dropout_seed2.py when the rerun does not produce
a value. If a value is ever obtained, apply the finisher instead.
"""
import sys

PATH = "RESULTS.md"

OLD = ("indomain k10 seed2 died before eval (no banner locally, not on W&B): "
       "cell is n=7; a resubmit is optional and changes no claim.")

NEW = ("indomain k10 seed2 has no value and the cell stands at n=7. Every completed run of this "
       "condition logs exactly 165766 optimizer steps. Throughput is node-dependent by an order of "
       "magnitude: 59-92 steps/s on the fast nodes (d4052, h200, cascadelake; wall-clock 0.50-0.78 h) "
       "against 6.3 steps/s on a V100 node (c2207, v100-pcie, zen; 1 CPU allocated, GPU utilisation 2%, "
       "so the loader and not the GPU is the bottleneck), which needs 7.3 h. Against the 6 h walltime in "
       "the sbatch, a slow-node draw cannot finish: the original attempt reached 137073 steps (82.7%) at "
       "5.98 h and died at the wall, and the July 30 2026 rerun (job 8839160) was on the same trajectory, "
       "40.0% at 2.93 h projecting to 81.9% at the wall, and was cancelled. An earlier rerun that day "
       "(job 8838715) failed in 19 s at wandb.init after /home/dai.hany/.bashrc was lost, taking the "
       "WANDB_API_KEY with it. Completing this cell needs --time=08:00:00 and more than one CPU, not a "
       "plain resubmit; the seed42 value in this table came from a slow-node run that took 7.29 h. No "
       "claim depends on it either way.")

txt = open(PATH).read()
if NEW[:60] in txt:
    sys.exit("ALREADY APPLIED. Nothing done.")
if OLD not in txt:
    sys.exit(f"ERROR: anchor sentence not found in {PATH}. Section 8.2 may already have been "
             "edited, or the finisher was applied instead. Check by hand: grep -n 'seed2' RESULTS.md")

open(PATH, "w").write(txt.replace(OLD, NEW))
print("APPLIED. Section 8.2 now records the seed2 rerun outcome; cell stays n=7 (63/64 cells).")
print("Verify: grep -n 'seed2 has no value' RESULTS.md")
