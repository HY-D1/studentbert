import json, numpy as np, sys
from pathlib import Path
d = Path(sys.argv[1] if len(sys.argv)>1 else "../processed/assist2017")
npz = np.load(d/"sequences.npz"); off=npz["offsets"]; ids=npz["student_ids"]
lab = json.loads((d/"dropout_labels.json").read_text()); lab.pop("_meta",None)
row = {int(s):i for i,s in enumerate(ids)}
for K in [20,50,100,200]:
    ds=dt=0
    for sid,l in lab.items():
        r=row.get(int(sid))
        if r is None or l!=1: continue
        dt+=1
        if off[r+1]-off[r] < K: ds+=1
    print(f"K={K}: {ds}/{dt} dropout students <K = {100*ds/dt:.0f}% reveal")
