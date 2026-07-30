#!/usr/bin/env python3
"""
collect_all_results.py - Consolidate ALL StudentBERT experiment logs into one
canonical, consistent results file for paper writing.

Outputs:
  RESULTS.md    - human-readable, organized by paper angle, with provenance
  results.json  - machine-readable (every number + its source log + n_seeds)

Run on the cluster from the code dir:
  cd /projects/algl/dai.hany/studentbert/code
  python collect_all_results.py --logdir . --out_md RESULTS.md --out_json results.json

Design:
  - Parses AUC / probe-acc / etc. from logs by known naming patterns.
  - Computes mean + std + n_seeds per condition (dedups by taking one value per seed).
  - Tags every result block with which paper(s) use it.
  - NEVER invents numbers: if logs for a block are missing, it writes
    "NOT FOUND (expected pattern: ...)" so you can see gaps, not fabricate.
  - Paired-bootstrap CIs that were computed earlier are NOT re-derived here
    (they need the per-seed arrays); instead the script emits the per-seed
    arrays so you can re-run a paired bootstrap if needed. It reports the CIs
    that are recorded in a small curated dict (from your verified analyses),
    clearly marked as "recorded from prior analysis".
"""
from __future__ import annotations
import argparse
import os
import sys, json, re, glob, os
from collections import defaultdict
import statistics as st

# ----------------------------------------------------------------------------
# 0. CONFIG: datasets, seeds, and the curated "recorded" values that came from
#    prior verified analyses (paired bootstrap CIs etc.) which cannot be
#    recomputed from a single-number-per-log parse. These are transcribed from
#    your verified results, and each is tagged with how it was obtained.
# ----------------------------------------------------------------------------
DATASETS = ["assist2017","ednet","junyi","algebra2005","bridge2006","assist2009","algebra2006"]
SEEDS = ["42","1","2","3","4","5"]

# Recorded CIs / values from prior verified analyses (paired bootstrap, etc.).
# Marked clearly so the paper cites them as computed, and you can re-derive from
# per-seed arrays this script also dumps.
RECORDED = {
  "objrev_ednet_to_assist_skill_minus_correct": {
     "value": None, "note": "skill 0.686 ~ full 0.690 >> correct 0.662; gap skill-correct large; 6/6 seeds, CI excludes 0 (recorded)"},
  "objrev_ednet_to_junyi_correct_minus_skill": {
     "value": None, "note": "correct 0.728 > full 0.723 > skill 0.715; 6/6 seeds, CI excludes 0 (recorded)"},
  "objrev_algebra2006_skill_minus_correct": {
     "value": "+0.0186", "ci": "[+0.0149,+0.0228]", "seeds": "6/6",
     "note": "first prospective predict-before-test to hold (recorded)"},
  "scale_indomain_ednet_fullscale": {
     "value": "+0.0069", "ci": "[+0.0065,+0.0075]", "seeds": "6/6",
     "note": "in-domain EdNet KT gain at full target scale; only cross-dataset gain that survives (recorded)"},
  "task1_truncation_K512": {
     "value": "-0.0265", "ci": "[-0.0280,-0.0253]", "seeds": "0/6",
     "note": "ASSIST2017 truncated K=512/pps5.0: skill-driven (correct-skill negative) (recorded)"},
  "task1_truncation_K10": {
     "value": "+0.0121", "ci": "[+0.0033,+0.0212]", "seeds": "6/6",
     "note": "ASSIST2017 truncated K=10/pps0.10: correctness-driven (correct-skill positive); regime FLIPPED holding skills+students fixed (recorded)"},
  "pps_ordering": {
     "note": "practice-per-skill orders all 7: skill-driven >=0.325, correctness-driven <=0.211 (recorded)",
     "values": {"Junyi":0.066,"EdNet":0.211,"ASSIST2009":0.325,"Algebra2006":2.41,
                "Bridge2006":2.79,"ASSIST2017":4.33,"Algebra2005":5.33}},
  "probe_gains_7": {
     "note": "masked-skill probe, pretrained(EdNet-full) minus scratch, 3 seeds each, all positive (recorded means)",
     "values": {"assist2017":0.0051,"ednet":0.0143,"junyi":0.0044,"algebra2005":0.0282,
                "bridge2006":0.0314,"assist2009":0.0389,"algebra2006":0.0110}},
  "embedding_coherence_negative": {
     "note": "vocab-invariant coherence separated regimes raw, but matched-skill (top-100) control: confound persisted r=+0.87 AND separation broke (Algebra2006 skill rose above EdNet corr). REPORT AS NEGATIVE."},
  "coherence_vs_probe_tiein": {
     "note": "raw rho=-0.68 is a vocab-size artifact (both track n_skills); clean coherence-vs-probe-gain rho=-0.32 n.s. DROPPED."},
  "lodo_negative": {
     "note": "leave-one-dataset-out: multivariate 5/6 but overparameterized at n=6, not trustworthy; single features n_students 5/6, pps 3/6. No combo generalizes. NEGATIVE."},
}

# Uniform 7-dataset baseline table (recorded, memory #10) - each row DKT/AKT/scratch/EduBERT-pretrained
BASELINE_TABLE = {
  "ASSIST2017":  [0.690,0.650,0.670,0.693],
  "EdNet":       [0.680,0.672,0.678,0.685],
  "Junyi":       [0.759,0.754,0.757,0.758],
  "Algebra2005": [0.798,0.776,0.781,0.787],
  "Bridge2006":  [0.795,0.773,0.773,0.775],
  "ASSIST2009":  [0.876,0.863,0.870,0.870],
  "Algebra2006": [0.803,0.775,0.787,0.790],
}

# ----------------------------------------------------------------------------
# 1. LOG PARSING HELPERS
# ----------------------------------------------------------------------------
def read(path):
    try:
        with open(path, errors="ignore") as f: return f.read()
    except Exception: return ""

def grab_after(text, marker, field_regex, window=6):
    """Find `marker`, then within the next `window` lines match field_regex, return first group."""
    lines = text.splitlines()
    for i,l in enumerate(lines):
        if marker in l:
            for j in range(i, min(i+window, len(lines))):
                m = re.search(field_regex, lines[j])
                if m: return m.group(1)
    return None

def last_auc(text):
    """Last 'test AUC ...' number in a log (best-ckpt eval prints late)."""
    vals = re.findall(r"test\s+AUC[^0-9]*([01]\.\d+)", text)
    return vals[-1] if vals else None

def kt_auc_block(text):
    """AUC from a === EduBERT-KT block or a plain 'test AUC'."""
    v = grab_after(text, "=== EduBERT-KT", r"test\s+AUC[^0-9]*([01]\.\d+)")
    if v: return v
    return last_auc(text)

def probe_acc(text):
    v = re.findall(r"test\s+probe\s+acc[^0-9]*([01]\.\d+)", text)
    return v[-1] if v else None

def baseline_auc(text):
    v = re.findall(r"test\s+AUC[^0-9]*([01]\.\d+)", text)
    return v[-1] if v else None

def agg(vals):
    """mean/std/n from a list of string floats (dedup already done by caller)."""
    fs = [float(x) for x in vals if x is not None]
    if not fs: return None
    return {"mean": round(st.mean(fs),4),
            "std": round(st.pstdev(fs),4) if len(fs)>1 else 0.0,
            "n": len(fs), "values": [round(f,4) for f in fs]}

def collect_by_seed(logdir, pattern_fn, parse_fn):
    """For each seed, find first matching log, parse one value. Returns list of values (dedup by seed)."""
    out = []
    for sd in SEEDS:
        found = None
        for pat in pattern_fn(sd):
            files = sorted(glob.glob(os.path.join(logdir, pat)))
            for fp in files:
                val = parse_fn(read(fp))
                if val is not None:
                    found = val; break
            if found: break
        if found is not None: out.append(found)
    return out

# ----------------------------------------------------------------------------
# 2. COLLECTORS for each experiment family (using your log-naming conventions)
# ----------------------------------------------------------------------------
def collect_objective_ablation(logdir):
    """Objective ablation per dataset. Log prefixes vary by dataset (memory)."""
    prefix = {"assist2017":"w7_objabl","junyi":"w7_objabl2","ednet":"w8_regime_ednet",
              "algebra2005":"w8_algabl","bridge2006":"w8_bridgeabl",
              "assist2009":"w8_a09abl","algebra2006":"w8_alg06abl"}
    res = {}
    for ds,pfx in prefix.items():
        res[ds] = {}
        for obj in ["full","skill_only","correct_only"]:
            def pf(sd, pfx=pfx, obj=obj, ds=ds):
                # ednet logs include the dataset token already in pfx
                return [f"{pfx}_{obj}_s{sd}_*.log", f"{pfx}_{obj}_s{sd}.log"]
            vals = collect_by_seed(logdir, pf, kt_auc_block)
            res[ds][obj] = agg(vals)
    return res

def collect_baselines_new(logdir):
    """DKT/AKT/scratch for the newer datasets, if logs present."""
    res = {}
    for ds in ["algebra2005","bridge2006","assist2009","algebra2006"]:
        res[ds] = {}
        for model in ["dkt","akt"]:
            def pf(sd, ds=ds, model=model):
                return [f"w8_base_{model}_{ds}_s{sd}_*.log", f"w8_base_{model}_{ds}_s{sd}.log"]
            res[ds][model] = agg(collect_by_seed(logdir, pf, baseline_auc))
        def pfs(sd, ds=ds):
            return [f"w8_scratch_{ds}_s{sd}_*.log"]
        res[ds]["scratch"] = agg(collect_by_seed(logdir, pfs, kt_auc_block))
    return res

def collect_probe7(logdir):
    """Masked-skill probe, full vs scratch, all 7."""
    res = {}
    for ds in DATASETS:
        res[ds] = {}
        for cond in ["full","scratch"]:
            def pf(sd, ds=ds, cond=cond):
                return [f"w8_probe7_{ds}_{cond}_s{sd}_*.log"]
            res[ds][cond] = agg(collect_by_seed(logdir, pf, probe_acc))
        # gain
        f = res[ds]["full"]; sc = res[ds]["scratch"]
        if f and sc:
            res[ds]["gain"] = round(f["mean"]-sc["mean"],4)
    return res

def collect_truncation(logdir):
    """Task 1 truncation flip: obj x K x seed. Report per-K per-obj means."""
    Ks = ["10","20","40","80","160","320","512"]
    res = {}
    for K in Ks:
        res[K] = {}
        for obj in ["full","skill_only","correct_only","scratch"]:
            def pf(sd, K=K, obj=obj):
                return [f"w8_trunc_{obj}_k{K}_s{sd}_*.log", f"w8_trunc_{obj}_k{K}_s{sd}.log"]
            res[K][obj] = agg(collect_by_seed(logdir, pf, kt_auc_block))
    return res

# ----------------------------------------------------------------------------
# 3. RENDER RESULTS.md
# ----------------------------------------------------------------------------
def fmt_agg(a):
    if a is None: return "NOT FOUND"
    return f"{a['mean']:.4f} \u00b1{a['std']:.4f} (n={a['n']})"

def fmt_vals(a):
    if a is None: return "-"
    return ", ".join(f"{v:.4f}" for v in a["values"])

def render_md(coll):
    L = []
    W = L.append
    W("# StudentBERT - Consolidated Results\n")
    W("_Auto-generated by `collect_all_results.py`. Every number below is parsed from a logged run "
      "or transcribed from a prior verified analysis (marked 'recorded'). Blocks that could not be "
      "found in the logs are flagged NOT FOUND rather than filled in._\n")
    W("\n**Paper tags:** `[EDM]` source comparison + seq-length ; `[LAK]` low-N advantage ; "
      "`[NeurIPS]` mechanism (objective reversal, causal flip, pps) ; `[ICLR]` representation/mechanism.\n")

    # 0. Methods / provenance
    W("\n---\n## 0. Methods & provenance  `[all]`\n")
    W(f"- **Datasets (7):** {', '.join(DATASETS)}. Uniform npz schema (student_ids, skill, correct, "
      "time_bin, CSR offsets); 80/10/10 split by student, seed 42; min 10 interactions.")
    W(f"- **Seeds:** up to {len(SEEDS)} ({', '.join(SEEDS)}). Means/std reported; paired-bootstrap CIs "
      "recorded where computed.")
    W("- **Task/metric:** knowledge tracing = next-step correctness, AUC (0.5 chance, 1.0 perfect). "
      "Best-checkpoint-by-val used for test.")
    W("- **Fine-tune correctness guard:** causal attention mask at KT/next-skill; whole-prefix pooling for dropout.")
    W("- **Provenance:** each number traces to a W&B run + committed config. Local log means below are a "
      "cross-check; W&B is the system of record.")

    # 1. Baselines
    W("\n---\n## 1. Baseline table (7 datasets)  `[EDM]` `[all]`\n")
    W("Knowledge-tracing test AUC. Columns: DKT / AKT / EduBERT-scratch / EduBERT-pretrained. "
      "(Recorded uniform table; EduBERT-pretrained = full-objective number for the 4 newer sets.)\n")
    W("| Dataset | DKT | AKT | scratch | EduBERT-pt |")
    W("|---|---|---|---|---|")
    for ds,row in BASELINE_TABLE.items():
        W(f"| {ds} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} | **{row[3]:.3f}** |")
    W("\n_Read: DKT is the strongest simple baseline and wins on the 4 newer datasets; the pretraining "
      "edge concentrates on the original 3 (esp in-domain EdNet); AKT weakest. Pretraining is regime/scale "
      "specific, not universal._")
    # cross-check vs collected new-dataset baselines
    nb = coll.get("baselines_new")
    if nb:
        W("\n**Log cross-check (newer datasets, parsed from logs):**\n")
        W("| Dataset | DKT (log) | AKT (log) | scratch (log) |")
        W("|---|---|---|---|")
        for ds in ["algebra2005","bridge2006","assist2009","algebra2006"]:
            r = nb.get(ds,{})
            W(f"| {ds} | {fmt_agg(r.get('dkt'))} | {fmt_agg(r.get('akt'))} | {fmt_agg(r.get('scratch'))} |")

    # 2. Multi-dataset transfer / source scale
    W("\n---\n## 2. Source comparison: scale > granularity  `[EDM]`\n")
    W("- **Central finding (recorded):** large sources (EdNet 442K, Junyi 61K) transfer well everywhere; "
      "small source (ASSIST 1.7K) transfers poorly everywhere. On the EdNet target, ASSIST (closest "
      "granularity) transfers WORST; Junyi (far granularity) helps more. Validated across 3 targets at "
      "budget-matched N=3000, 3 seeds.")
    W("- **Loss != transfer (recorded):** Junyi has lower pretraining loss but worse KT transfer than EdNet.")
    W("- **Moderator:** sequence density (practice-per-skill) - see section 5.")

    # 3. Scale boundary / low-N
    W("\n---\n## 3. Low-resource advantage & scale boundary  `[LAK]`\n")
    W("- **Scale boundary (recorded):** cross-dataset transfer gains are largest when the target is "
      "data-poor; at full target scale they fade toward ~0.")
    r = RECORDED["scale_indomain_ednet_fullscale"]
    W(f"- **Only survivor at full scale:** in-domain EdNet KT gain {r['value']}, CI {r['ci']}, {r['seeds']} seeds. "
      f"({r['note']})")
    W("- **Takeaway:** cross-dataset pretraining is a low-resource tool. State the boundary plainly.")

    # 4. Objective reversal
    W("\n---\n## 4. Objective reversal (headline)  `[NeurIPS]` `[ICLR]`\n")
    W("Best pretraining objective is target-dependent. Per-dataset objective ablation "
      "(EdNet-source encoders: full / skill_only / correct_only -> target), from logs:\n")
    oa = coll.get("objective_ablation",{})
    W("| Target | full | skill_only | correct_only | pattern |")
    W("|---|---|---|---|---|")
    for ds in DATASETS:
        r = oa.get(ds,{})
        f,s,c = r.get("full"),r.get("skill_only"),r.get("correct_only")
        pat = ""
        if f and s and c:
            if s["mean"]>=c["mean"]: pat="skill-driven (skill>=correct)"
            else: pat="correctness-driven (correct>skill)"
        W(f"| {ds} | {fmt_agg(f)} | {fmt_agg(s)} | {fmt_agg(c)} | {pat} |")
    W("\n**Recorded reversals (paired bootstrap):**")
    W(f"- EdNet->ASSIST: {RECORDED['objrev_ednet_to_assist_skill_minus_correct']['note']}")
    W(f"- EdNet->Junyi: {RECORDED['objrev_ednet_to_junyi_correct_minus_skill']['note']}")
    rr = RECORDED["objrev_algebra2006_skill_minus_correct"]
    W(f"- Algebra2006 (prospective): skill-correct {rr['value']} CI {rr['ci']} {rr['seeds']}. {rr['note']}")

    # 5. Regime characterization
    W("\n---\n## 5. What governs the regime: practice-per-skill  `[NeurIPS]`\n")
    pv = RECORDED["pps_ordering"]["values"]
    W("Practice-per-skill (pps = median seq length / n_skills) orders all 7 datasets:\n")
    W("| Dataset | pps | regime |")
    W("|---|---|---|")
    for ds,v in sorted(pv.items(), key=lambda kv: kv[1]):
        reg = "correctness-driven" if v<=0.211 else "skill-driven"
        W(f"| {ds} | {v:.3f} | {reg} |")
    W(f"\n_{RECORDED['pps_ordering']['note']}_")
    W("\n**Causal truncation flip (recorded):** truncating ASSIST2017 sequence length while holding "
      "skill count (102) and #students (1708) FIXED flips the regime:")
    t512 = RECORDED["task1_truncation_K512"]; t10 = RECORDED["task1_truncation_K10"]
    W(f"- K=512 (pps~5.0): correct-skill {t512['value']} CI {t512['ci']} ({t512['seeds']}) -> {t512['note']}")
    W(f"- K=10 (pps~0.10): correct-skill {t10['value']} CI {t10['ci']} ({t10['seeds']}) -> {t10['note']}")
    W("- Scratch control rises smoothly with K (rules out pure data-quantity). Crossover ~pps 1.5.")
    # truncation table from logs
    tr = coll.get("truncation")
    if tr:
        W("\n**Truncation sweep (from logs, KT AUC means):**\n")
        W("| K | full | skill_only | correct_only | scratch |")
        W("|---|---|---|---|---|")
        for K in ["10","20","40","80","160","320","512"]:
            r = tr.get(K,{})
            def m(x): 
                a=r.get(x); return f"{a['mean']:.4f}" if a else "-"
            W(f"| {K} | {m('full')} | {m('skill_only')} | {m('correct_only')} | {m('scratch')} |")
    W("\n**Honest limit:** precise threshold fuzzy (empty gap 0.33-2.79). LODO does not generalize at n=6 "
      f"({RECORDED['lodo_negative']['note']}).")

    # 6. Probe mechanism
    W("\n---\n## 6. Probe mechanism (7 datasets)  `[ICLR]` `[NeurIPS]`\n")
    W("Masked-skill probe on the frozen representation (EdNet-full encoder vs scratch). "
      "Higher = skill more decodable. From logs:\n")
    pr = coll.get("probe7",{})
    W("| Dataset | pretrained | scratch | gain |")
    W("|---|---|---|---|")
    for ds in DATASETS:
        r = pr.get(ds,{})
        g = r.get("gain")
        W(f"| {ds} | {fmt_agg(r.get('full'))} | {fmt_agg(r.get('scratch'))} | "
          f"{('+' + format(g,'.4f')) if isinstance(g,float) else '-'} |")
    W("\n_Pretrained beats scratch on all 7 (recorded gains +0.004 to +0.039, all positive). "
      "Mechanism: pretraining organizes the representation around skills._")

    # 7. Embedding analysis (negative)
    W("\n---\n## 7. Embedding geometry (honest negative)  `[ICLR]`\n")
    W(f"- {RECORDED['embedding_coherence_negative']['note']}")
    W(f"- Tie-in: {RECORDED['coherence_vs_probe_tiein']['note']}")
    W("- **Report as:** we checked whether embedding geometry differs by regime; it did not survive a "
      "vocabulary-size control. Probe (section 6) carries the representational evidence.")

    # 8. Downstream
    W("\n---\n## 8. Downstream tasks  `[LAK]`\n")
    W("- **Dropout (recorded):** Junyi clean, in-domain best; ASSIST pretraining does NOT help (scratch best); "
      "EdNet high-variance/inconclusive at 3 seeds, but 8-seed PAIRED bootstrap found two real effects "
      "(in-domain K=5 +0.097 CI[+0.029,+0.164] 6/8; fromJunyi K=10 +0.095 CI[+0.057,+0.132] 8/8). "
      "Clean K is dataset-dependent (ASSIST K<=50, EdNet/Junyi K<=10); use --window_censor for high K.")
    W("- **Next-skill (recorded):** pretraining helps most at low N; ordering identical under macro and "
      "weighted OVR AUC (metric-robust). Junyi frequency-saturated: plain top-1 flat, but macro-top1 "
      "in-domain +0.007 (best all seeds) and top-5 in-domain +0.055.")

    # 9. Negatives
    W("\n---\n## 9. Honest negatives & caveats  `[all]`\n")
    W("- Embedding coherence does not separate regimes under a vocab control (section 7).")
    W("- Coherence-vs-probe tie-in is a vocab-size artifact - dropped (section 7).")
    W("- LODO cross-dataset prediction does not generalize at n=6 (overparameterized).")
    W("- Regime threshold is fuzzy (empty pps gap 0.33-2.79); pps ordering holds, exact boundary not localized.")
    W("- Scale/pps observationally confounded at n=7; the causal claim rests on the single ASSIST2017 "
      "truncation manipulation.")
    W("- EdNet dropout high-variance; ASSIST dropout pretraining doesn't help.")

    W("\n---\n_End of consolidated results._\n")
    return "\n".join(L)

# ----------------------------------------------------------------------------
# 4. MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=".")
    ap.add_argument("--out_md", default="RESULTS.md")
    ap.add_argument("--out_json", default="results.json")
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out_md even if it contains hand-added sections")
    args = ap.parse_args()

    # Sections this script does NOT regenerate. Overwriting RESULTS.md while any
    # of them are present silently destroys them.
    MANUAL = ["### 2.1", "### 3.1", "### 3.2", "### 6.1", "### 8.1", "### 8.2", "## 10."]
    if os.path.exists(args.out_md) and not args.force:
        existing = open(args.out_md, errors="ignore").read()
        present = [m for m in MANUAL if m in existing]
        if present:
            sys.exit(
                f"REFUSING to overwrite {args.out_md}: it contains hand-added sections "
                f"that this script does not regenerate ({', '.join(present)}).\n"
                f"  Write elsewhere and diff:  --out_md RESULTS_generated.md\n"
                f"  Or overwrite deliberately: --force  (the sections above will be lost)")

    coll = {}
    print("collecting objective ablation ...")
    coll["objective_ablation"] = collect_objective_ablation(args.logdir)
    print("collecting probe (7 datasets) ...")
    coll["probe7"] = collect_probe7(args.logdir)
    print("collecting new-dataset baselines ...")
    coll["baselines_new"] = collect_baselines_new(args.logdir)
    print("collecting truncation sweep ...")
    coll["truncation"] = collect_truncation(args.logdir)
    coll["recorded"] = RECORDED
    coll["baseline_table"] = BASELINE_TABLE

    with open(args.out_json,"w") as f: json.dump(coll, f, indent=2)
    md = render_md(coll)
    with open(args.out_md,"w") as f: f.write(md)

    # quick gap report to stderr
    print("\n=== GAP CHECK (blocks NOT FOUND in logs) ===")
    gaps=0
    for ds,r in coll["objective_ablation"].items():
        for obj,a in r.items():
            if a is None: print(f"  objabl {ds}/{obj}: NOT FOUND"); gaps+=1
    for ds,r in coll["probe7"].items():
        for cond in ["full","scratch"]:
            if r.get(cond) is None: print(f"  probe7 {ds}/{cond}: NOT FOUND"); gaps+=1
    if gaps==0: print("  no gaps in objective-ablation or probe7 (all conditions parsed)")
    print(f"\nWrote {args.out_md} and {args.out_json}")

if __name__=="__main__":
    main()
