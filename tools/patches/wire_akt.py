p = "scripts/train_baseline.py"
s = open(p).read()
if "from src.models.akt import AKT" not in s:
    s = s.replace("from src.models.saint_plus import SAINTPlus",
                  "from src.models.saint_plus import SAINTPlus\nfrom src.models.akt import AKT")
s = s.replace('ap.add_argument("--model", choices=["dkt", "saint"], required=True)',
              'ap.add_argument("--model", choices=["dkt", "saint", "akt"], required=True)')
s = s.replace(
    '    else:\n        model = SAINTPlus(num_skills=num_skills, d_model=256, n_heads=8,\n                          n_layers=2, dropout=dropout, max_len=args.max_seq_len).to(device)',
    '    elif args.model == "akt":\n        model = AKT(num_skills=num_skills, d_model=256, n_heads=8,\n                    n_blocks=2, d_ff=1024, dropout=dropout, max_len=args.max_seq_len).to(device)\n    else:\n        model = SAINTPlus(num_skills=num_skills, d_model=256, n_heads=8,\n                          n_layers=2, dropout=dropout, max_len=args.max_seq_len).to(device)'
)
open(p, "w").write(s)
print("wired AKT")
