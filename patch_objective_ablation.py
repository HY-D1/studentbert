# Adds --objective {full,correct_only,skill_only} to pretrain_edubert.py for the
# pretraining-objective ablation. full = skill_loss + correct_loss (current default).
# correct_only = predict only masked correctness. skill_only = predict only masked skill.
# Cuts cleanly at the loss-combination line; masking + model heads unchanged, so any
# transfer difference is attributable purely to the training signal.
# Run from repo root: python patch_objective_ablation.py
p = "scripts/pretrain_edubert.py"
s = open(p).read()
if "--objective" in s:
    print(f"{p}: already patched"); raise SystemExit

# 1. add the CLI flag (after run_type arg)
old_arg = '    ap.add_argument("--run_type", default="pretrain_full")'
new_arg = ('    ap.add_argument("--objective", choices=["full", "correct_only", "skill_only"],\n'
           '                    default="full",\n'
           '                    help="ablation: which masked targets to predict during pretraining")\n'
           '    ap.add_argument("--run_type", default="pretrain_full")')
assert old_arg in s, "run_type arg not found"
s = s.replace(old_arg, new_arg, 1)

# 2. replace the loss-combination line with objective-gated version
old_loss = "            loss = loss_s + loss_c"
new_loss = ("            if args.objective == 'full':\n"
            "                loss = loss_s + loss_c\n"
            "            elif args.objective == 'correct_only':\n"
            "                loss = loss_c\n"
            "            elif args.objective == 'skill_only':\n"
            "                loss = loss_s")
assert old_loss in s, "loss combination line not found"
s = s.replace(old_loss, new_loss, 1)

open(p, "w").write(s)
print(f"{p}: patched with --objective flag")
