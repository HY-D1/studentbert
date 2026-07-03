# Adds source/target/condition config fields to wandb.init in all training scripts,
# so W&B panels group one-click (no run-name regex). Idempotent; run from repo root:
#   python patch_wandb_fields.py
# Derivation: target=dataset; source=parsed from encoder_ckpt filename; condition=
#   scratch | indomain (source==target) | <source name>.
import os, re

HELPER = (
    "def _wandb_fields(args, dataset):\n"
    "    # derive source/target/condition for clean W&B grouping\n"
    "    src = 'none'\n"
    "    ck = getattr(args, 'encoder_ckpt', None)\n"
    "    if ck:\n"
    "        b = os.path.basename(ck)\n"
    "        m = re.match(r'edubert_([a-zA-Z0-9]+)_pretrain', b)\n"
    "        if m: src = m.group(1)\n"
    "    init = getattr(args, 'init', 'scratch')\n"
    "    cond = 'scratch' if init == 'scratch' else ('indomain' if src == dataset else src)\n"
    "    return {'source': src, 'target': dataset, 'condition': cond}\n\n\n"
)

def ensure_imports(s):
    if "\nimport os" not in s and not s.startswith("import os"): s = "import os\n" + s
    if "\nimport re" not in s and not s.startswith("import re"): s = "import re\n" + s
    return s

def patch(path, anchor, repl):
    if not os.path.exists(path):
        print(f"{path}: MISSING, skipped"); return
    s = open(path).read()
    if "_wandb_fields" in s:
        print(f"{path}: already patched"); return
    s = ensure_imports(s)
    if "\ndef main(" in s:
        s = s.replace("\ndef main(", "\n" + HELPER + "def main(", 1)
    else:
        s = HELPER + s
    if anchor in s:
        s = s.replace(anchor, repl, 1); print(f"{path}: patched")
    else:
        print(f"{path}: WARN anchor not found - patch config manually")
    open(path, "w").write(s)

fd_anchor = '"encoder_ckpt": args.encoder_ckpt or "none"})'
fd_repl   = '"encoder_ckpt": args.encoder_ckpt or "none", **_wandb_fields(args, dataset)})'
patch("scripts/finetune_edubert.py", fd_anchor, fd_repl)
patch("scripts/downstream_edubert.py", fd_anchor, fd_repl)
patch("scripts/probe_edubert_v2.py", fd_anchor, fd_repl)

tb_anchor = '"seed": args.seed,\n            },'
tb_repl   = '"seed": args.seed,\n                **_wandb_fields(args, dataset),\n            },'
patch("scripts/train_baseline.py", tb_anchor, tb_repl)
print("done - verify: grep -c _wandb_fields scripts/*.py ; python -c 'import ast; ...'")
