# Task 1a: measure dataset-level structural properties that might explain
# whether a target is skill-driven or correctness-driven in transfer.
# READ-ONLY: reads sequences.npz + skill_vocab.json, writes nothing. Quota-safe.
# Usage: PYTHONPATH=. python measure_dataset_properties.py ../processed/assist2017 ../processed/junyi ../processed/ednet
import sys, json
import numpy as np
from pathlib import Path

def load(processed_dir):
    d = Path(processed_dir)
    z = np.load(d / "sequences.npz")
    vocab = json.loads((d / "skill_vocab.json").read_text())
    return z, vocab

def seq_iter(z):
    # CSR-style: offsets delimit each student's slice
    off = z["offsets"]
    skill = z["skill"]; correct = z["correct"]
    for i in range(len(off) - 1):
        a, b = off[i], off[i+1]
        if b > a:
            yield skill[a:b], correct[a:b]

def properties(processed_dir):
    z, vocab = load(processed_dir)
    name = Path(processed_dir).name
    skill = z["skill"]; correct = z["correct"]; off = z["offsets"]
    n_students = len(off) - 1
    n_skills = max(int(v) for v in vocab.values())
    total_interactions = len(skill)

    # sequence lengths
    lens = np.diff(off)
    lens = lens[lens > 0]

    # correctness base rate + how balanced (entropy of the binary label)
    base_rate = float(correct.mean())
    p = base_rate
    label_entropy = 0.0 if p in (0.0, 1.0) else -(p*np.log2(p) + (1-p)*np.log2(1-p))

    # skill frequency distribution -> normalized entropy (concept concentration)
    real = skill[skill > 0]
    counts = np.bincount(real, minlength=n_skills + 1)[1:]  # drop PAD idx 0
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    skill_entropy = float(-(nz * np.log2(nz)).sum())
    max_ent = np.log2(n_skills) if n_skills > 1 else 1.0
    skill_entropy_norm = float(skill_entropy / max_ent)  # 0=all one skill, 1=uniform

    # skills per student (distinct), and repetition (interactions per distinct skill within a student)
    distinct_per_student = []
    rep_ratio = []  # interactions / distinct skills within a student (higher=more repetition)
    for s, c in seq_iter(z):
        sr = s[s > 0]
        if len(sr) == 0: continue
        distinct = len(np.unique(sr))
        distinct_per_student.append(distinct)
        rep_ratio.append(len(sr) / max(distinct, 1))
    distinct_per_student = np.array(distinct_per_student)
    rep_ratio = np.array(rep_ratio)

    interactions_per_skill = total_interactions / n_skills

    return {
        "dataset": name,
        "n_students": n_students,
        "n_skills": n_skills,
        "total_interactions": total_interactions,
        "median_seq_len": float(np.median(lens)),
        "mean_seq_len": float(lens.mean()),
        "correct_base_rate": round(base_rate, 4),
        "label_entropy_bits": round(label_entropy, 4),          # 1.0 = perfectly balanced 50/50
        "skill_entropy_norm": round(skill_entropy_norm, 4),     # 1.0 = skills used uniformly, low = concentrated
        "mean_distinct_skills_per_student": round(float(distinct_per_student.mean()), 2),
        "median_distinct_skills_per_student": float(np.median(distinct_per_student)),
        "mean_repetition_ratio": round(float(rep_ratio.mean()), 3),  # interactions per distinct skill (higher=drill/repeat)
        "interactions_per_skill": round(interactions_per_skill, 1),
    }

if __name__ == "__main__":
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: python measure_dataset_properties.py <processed_dir> [<processed_dir> ...]")
        sys.exit(1)
    rows = []
    for d in dirs:
        try:
            rows.append(properties(d))
        except Exception as e:
            print(f"ERROR on {d}: {e}")
    # print as an aligned table
    if rows:
        keys = list(rows[0].keys())
        print("\n=== DATASET PROPERTIES ===")
        for k in keys:
            line = f"{k:36s}"
            for r in rows:
                v = r[k]
                line += f"  {str(v):>14s}"
            print(line)
        print("\n(regime known: ASSIST=skill-driven, Junyi=correctness-driven; EdNet=TBD via 1b)")
