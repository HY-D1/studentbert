# SKILL EMBEDDING ANALYSIS for the NeurIPS mechanism story.
# For each dataset's pretrained encoder, analyze the learned skill embedding space:
#   1. nearest-neighbor coherence (do co-occurring skills embed close together?)
#   2. UMAP/t-SNE 2D visualization
#   3. coherence split by regime (skill-driven vs correctness-driven)
#   4. correlate embedding coherence with probe decodability (ties both analyses)
#
# COHERENCE METRIC: skills that co-occur in student sequences should be neighbors
# in embedding space IF the embedding captures skill structure. We measure:
#   for each skill, its top-k embedding neighbors; coherence = fraction of those
#   neighbors that are also top-k CO-OCCURRENCE neighbors (from the data).
#   High coherence = embedding geometry reflects real skill co-occurrence structure.
#
# READ-ONLY on data; loads the encoder checkpoint's skill_emb weights.
# Usage:
#   PYTHONPATH=. python analyze_skill_embeddings.py \
#     --encoder ../checkpoints/edubert_ednet_pretrain_full_encoder.pt \
#     --datasets assist2017:skill ednet:corr junyi:corr algebra2005:skill \
#                bridge2006:skill assist2009:skill algebra2006:skill \
#     --processed_root ../processed --out embedding_analysis
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np

def load_skill_emb(ckpt_path):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck)
    # find the skill embedding weight
    for k in sd:
        if "skill_emb" in k and "weight" in k:
            return sd[k].numpy()  # [vocab+1, d]
    raise KeyError(f"no skill_emb weight in {ckpt_path}; keys: {[k for k in sd][:10]}")

def cooccurrence_matrix(processed_dir, n_skills):
    # co-occurrence counts: skills co-occur if they appear in the same student seq.
    z = np.load(Path(processed_dir)/"sequences.npz")
    off = z["offsets"]; skill = z["skill"]
    co = np.zeros((n_skills, n_skills), dtype=np.float32)
    n_stu = len(off)-1
    rng = np.random.default_rng(42)
    idx = rng.choice(n_stu, min(n_stu,5000), replace=False) if n_stu>5000 else range(n_stu)
    for i in idx:
        a,b = off[i], off[i+1]
        sk = np.unique(skill[a:b]); sk = sk[(sk>0)&(sk<=n_skills)]
        sk0 = sk-1  # 0-indexed
        for x in sk0:
            co[x, sk0] += 1
    np.fill_diagonal(co, 0.0)
    return co

def embedding_similarity(emb, n_skills):
    # cosine similarity between skill embeddings (skip PAD=0)
    E = emb[1:n_skills+1].astype(np.float64)
    norm = np.linalg.norm(E, axis=1, keepdims=True); norm[norm==0]=1.0
    En = E/norm
    return En @ En.T  # [n_skills, n_skills]

def coherence_rankcorr(processed_dir, emb, n_skills, top_n=0):
    # VOCAB-INVARIANT coherence: Spearman rank-correlation between embedding
    # cosine-similarity and co-occurrence count, over all off-diagonal skill pairs.
    # If top_n>0, restrict to the top_n MOST FREQUENT skills first, so vocabulary
    # size is equalized across datasets (removes the n_skills confound by design).
    from scipy.stats import spearmanr
    co_full = cooccurrence_matrix(processed_dir, n_skills)
    sim_full = embedding_similarity(emb, n_skills)
    if top_n and top_n < n_skills:
        # rank skills by total co-occurrence mass (proxy for frequency), keep top_n
        freq = co_full.sum(axis=1)
        keep = np.argsort(-freq)[:top_n]
        co = co_full[np.ix_(keep, keep)]
        sim = sim_full[np.ix_(keep, keep)]
        nk = top_n
    else:
        co, sim, nk = co_full, sim_full, n_skills
    iu = np.triu_indices(nk, k=1)
    co_v = co[iu]; sim_v = sim[iu]
    # only consider pairs that ever co-occur (co>0) plus a sample of zeros, to
    # avoid the correlation being dominated by the huge mass of never-co-occur
    # zero pairs. Standard practice: keep all pairs (zeros are informative: they
    # SHOULD be far in embedding space). Report both all-pairs and co>0-only.
    rho_all, _ = spearmanr(co_v, sim_v)
    mask = co_v > 0
    if mask.sum() >= 10:
        rho_pos, _ = spearmanr(co_v[mask], sim_v[mask])
    else:
        rho_pos = float("nan")
    return float(rho_all), float(rho_pos), int(mask.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="pretrained encoder ckpt (skill_emb source)")
    ap.add_argument("--datasets", nargs="+", required=True, help="name:regime pairs")
    ap.add_argument("--processed_root", default="../processed")
    ap.add_argument("--out", default="embedding_analysis")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--make_plots", action="store_true")
    ap.add_argument("--top_n_skills", type=int, default=0, help="if >0, restrict to top-N most frequent skills (matched across datasets, removes n_skills confound)")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # NOTE: the EdNet encoder's skill_emb is EdNet's vocab. To analyze per-DATASET
    # skill embeddings, we need each dataset's OWN pretrained skill_emb. Since our
    # transfer setup re-inits skill_emb per target, the meaningful embedding to
    # analyze is the one learned DURING pretraining on that dataset's own encoder
    # (edubert_<ds>_pretrain_full_encoder.pt) where it exists, else note it.
    # For datasets without their own pretrain encoder, we analyze the EdNet source
    # embedding restricted to shared structure (documented as a limitation).

    results=[]
    for spec in args.datasets:
        name, regime = spec.split(":")
        pdir = Path(args.processed_root)/name
        vocab = json.loads((pdir/"skill_vocab.json").read_text())
        n_skills = max(int(v) for v in vocab.values())
        # prefer the dataset's OWN pretrain encoder if it exists
        own = Path(args.encoder).parent / f"edubert_{name}_pretrain_full_encoder.pt"
        use_ckpt = own if own.exists() else Path(args.encoder)
        src = "own" if own.exists() else "ednet_source"
        try:
            emb = load_skill_emb(use_ckpt)
        except Exception as e:
            print(f"  {name}: SKIP ({e})"); continue
        vocab_in_emb = emb.shape[0]-1
        nk = min(n_skills, vocab_in_emb)
        rho_all, rho_pos, n_pos = coherence_rankcorr(pdir, emb, nk, top_n=args.top_n_skills)
        results.append({"dataset":name,"regime":regime,"n_skills":nk,
                        "emb_source":src,"coherence_rho_all":round(rho_all,4),
                        "coherence_rho_cooccur":round(rho_pos,4),"n_cooccur_pairs":n_pos})
        print(f"  {name:12s} regime={regime:5s} n_skills={nk:4d} emb={src:12s} rho_all={rho_all:+.4f} rho_cooccur={rho_pos:+.4f} (n_pairs={n_pos})")
        if args.make_plots:
            try:
                from sklearn.manifold import TSNE
                E = emb[1:nk+1]
                ts = TSNE(n_components=2, random_state=42, perplexity=min(30,nk-1)).fit_transform(E)
                np.save(outdir/f"tsne_{name}.npy", ts)
            except Exception as e:
                print(f"    (tsne skipped: {e})")

    # summary: coherence by regime + correlation with probe (probe passed via file if available)
    (outdir/"coherence_results.json").write_text(json.dumps(results, indent=2))
    print("\n=== COHERENCE BY REGIME ===")
    for field in ["coherence_rho_all","coherence_rho_cooccur"]:
        sk=[r[field] for r in results if r["regime"]=="skill" and not (r[field]!=r[field])]
        co=[r[field] for r in results if r["regime"]=="corr" and not (r[field]!=r[field])]
        print(f"  [{field}]")
        if sk: print(f"    skill-driven: mean {np.mean(sk):+.4f}  vals {[round(x,4) for x in sk]}")
        if co: print(f"    corr-driven:  mean {np.mean(co):+.4f}  vals {[round(x,4) for x in co]}")
    print("\nSaved coherence_results.json. To correlate with probe decodability,")
    print("pass probe accuracies and compute Spearman (done in a follow-up once probe7 lands).")

if __name__=="__main__":
    main()
