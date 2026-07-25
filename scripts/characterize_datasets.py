# TASK 2: structural characterization of all 6 datasets to check if ASSISTments2009
# is a genuine outlier on some property beyond practice-per-skill.
# READ-ONLY, reads sequences.npz for each. Quota-safe.
# Usage: PYTHONPATH=. python characterize_datasets.py \
#   ../processed/assist2017 ../processed/algebra2005 ../processed/bridge2006 \
#   ../processed/assist2009 ../processed/ednet ../processed/junyi
from __future__ import annotations
import sys, json
import numpy as np
from pathlib import Path

def gini(x):
    x=np.sort(np.asarray(x,dtype=float)); n=len(x)
    if n==0 or x.sum()==0: return 0.0
    cum=np.cumsum(x)
    return (n+1-2*np.sum(cum)/cum[-1])/n

def characterize(pdir):
    d=Path(pdir); z=np.load(d/"sequences.npz")
    vocab=json.loads((d/"skill_vocab.json").read_text())
    off=z["offsets"]; skill=z["skill"]; correct=z["correct"]
    n_students=len(off)-1
    n_skills=max(int(v) for v in vocab.values())
    total=len(skill)

    # per-student stats. For huge datasets, subsample students (distribution
    # summaries are stable under sampling; exact values not needed). Cap at 5000.
    rng=np.random.default_rng(42)
    if n_students>5000:
        stu_idx=rng.choice(n_students,5000,replace=False)
    else:
        stu_idx=np.arange(n_students)
    distinct_per_stu=[]; rep_per_stu=[]
    revisit_rate=[]
    correct_run_lens=[]
    for i in stu_idx:
        a,b=off[i],off[i+1]
        s=skill[a:b]; c=correct[a:b]
        sr=s[s>0]
        if len(sr)==0: continue
        distinct=len(np.unique(sr))
        distinct_per_stu.append(distinct)
        rep_per_stu.append(len(sr)/max(distinct,1))
        seen=set(); rev=0
        for x in sr:
            if x in seen: rev+=1
            seen.add(x)
        revisit_rate.append(rev/len(sr))
        # correct-run lengths overall (consecutive correct answers)
        run=0
        for ci in c:
            if ci==1: run+=1
            else:
                if run>0: correct_run_lens.append(run)
                run=0
        if run>0: correct_run_lens.append(run)

    # skill usage concentration
    real=skill[skill>0]
    counts=np.bincount(real,minlength=n_skills+1)[1:]
    skill_gini=gini(counts)

    distinct_per_stu=np.array(distinct_per_stu)
    rep_per_stu=np.array(rep_per_stu)
    revisit_rate=np.array(revisit_rate)
    crl=np.array(correct_run_lens) if correct_run_lens else np.array([0])

    lens=np.diff(off); lens=lens[lens>0]
    return {
        "dataset": d.name,
        "_perstudent_sampled": int(len(stu_idx)),
        "n_students": n_students,
        "n_skills": n_skills,
        "base_rate": round(float(correct.mean()),4),
        "median_seq_len": float(np.median(lens)),
        "practice_per_skill": round(float(np.median(lens))/n_skills,3),
        "mean_repetition": round(float(rep_per_stu.mean()),3),
        "mean_distinct_skills": round(float(distinct_per_stu.mean()),2),
        "skill_revisit_rate": round(float(revisit_rate.mean()),3),
        "skill_gini": round(skill_gini,3),
        "mean_correct_run": round(float(crl.mean()),3),
        "median_correct_run": float(np.median(crl)),
        "frac_correct_runs_ge3": round(float(np.mean(crl>=3)),3),   # mastery-learning signature
    }

if __name__=="__main__":
    dirs=sys.argv[1:]
    rows=[characterize(d) for d in dirs]
    keys=list(rows[0].keys())
    print("\n=== STRUCTURAL CHARACTERIZATION (all 6, regime in header) ===")
    # print transposed for readability
    regime={"assist2017":"skill","algebra2005":"skill","bridge2006":"skill",
            "assist2009":"skill","ednet":"corr","junyi":"corr"}
    hdr=f"{'property':22s}"
    for r in rows: hdr+=f"{r['dataset'][:10]:>11s}"
    print(hdr)
    reg=f"{'REGIME':22s}"
    for r in rows: reg+=f"{regime.get(r['dataset'],'?'):>11s}"
    print(reg)
    print("-"*len(hdr))
    for k in keys:
        if k=="dataset": continue
        line=f"{k:22s}"
        for r in rows: line+=f"{str(r[k]):>11s}"
        print(line)
    print("\nOUTLIER CHECK: compare ASSISTments2009 vs the other 3 skill-driven")
    print("(assist2017, algebra2005, bridge2006) on each property. Does any cleanly")
    print("separate it? Focus on mastery signature (frac_correct_runs_ge3, mean_correct_run).")
