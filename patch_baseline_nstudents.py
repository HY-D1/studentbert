p = "scripts/train_baseline.py"
s = open(p).read()
if "--n_students" not in s:
    s = s.replace(
        '    ap.add_argument("--run_type", default="baseline",',
        '    ap.add_argument("--n_students", type=int, default=None,\n'
        '                    help="subsample TRAIN to first N students (seeded); None=all")\n'
        '    ap.add_argument("--run_type", default="baseline",'
    )
old = '    train_loader = make_loader("train", True)'
new = (
    '    train_ds = InteractionDataset(args.processed_dir, "train", args.max_seq_len)\n'
    '    if args.n_students is not None and args.n_students < len(train_ds.rows):\n'
    '        import numpy as _np\n'
    '        _rng = _np.random.default_rng(args.seed)\n'
    '        _order = _rng.permutation(len(train_ds.rows))[:args.n_students]\n'
    '        train_ds.rows = [train_ds.rows[i] for i in _order]\n'
    '        print(f"subsampled train to {len(train_ds.rows)} students (seed {args.seed})")\n'
    '    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)'
)
if old in s:
    s = s.replace(old, new)
else:
    print("WARN: train_loader line not found; check manually")
open(p, "w").write(s)
print("patched train_baseline.py")
