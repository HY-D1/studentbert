"""Generate tiny synthetic packed data to smoke-test models WITHOUT real datasets.
Usage: python scripts/_make_test_data.py  -> writes data/test_assist/
"""
import numpy as np, json, os
rng = np.random.default_rng(0)
K = 50
os.makedirs("data/test_assist", exist_ok=True)
sids, skill, correct, tbin, offs = [], [], [], [], [0]
for u in range(1, 201):
    n = int(rng.integers(20, 300))
    sids.append(u)
    skill.extend(rng.integers(1, K+1, n).tolist())
    correct.extend(rng.integers(0, 2, n).tolist())
    tbin.extend(rng.integers(1, 6, n).tolist())
    offs.append(len(skill))
np.savez_compressed("data/test_assist/sequences.npz",
    student_ids=np.array(sids,dtype=np.int64), skill=np.array(skill,dtype=np.int32),
    correct=np.array(correct,dtype=np.int8), time_bin=np.array(tbin,dtype=np.int16),
    offsets=np.array(offs,dtype=np.int64))
json.dump({f"s{i}":i for i in range(1,K+1)}, open("data/test_assist/skill_vocab.json","w"))
perm = (rng.permutation(200)+1)
json.dump({"train":perm[:160].tolist(),"val":perm[160:180].tolist(),"test":perm[180:].tolist()},
          open("data/test_assist/splits.json","w"))
print("wrote data/test_assist (200 students, 50 skills)")
