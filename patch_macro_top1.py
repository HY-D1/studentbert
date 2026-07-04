# Adds macro (frequency-normalized) top-1 for next-skill: mean per-class top-1 recall,
# upweighting rare skills (advisor point 4, Junyi 1326-class saturation). Wires it into
# the test-eval block and logs test/macro_top1. Run from repo root: python patch_macro_top1.py
p = "scripts/downstream_edubert.py"
s = open(p).read()
if "macro_top1" in s:
    print(f"{p}: already patched"); raise SystemExit

# 1. define macro_top1 right after topk_acc
anchor = ("    hit = (topk == tt.unsqueeze(-1)).any(-1).float()\n"
          "    return hit.mean().item()\n")
assert anchor in s, "topk_acc return not found"
helper = anchor + (
    "\n\n"
    "def macro_top1(logits, target, valid, present_classes):\n"
    "    # frequency-normalized top-1: mean over classes of per-class top-1 recall.\n"
    "    lt = logits[valid]\n"
    "    tt = target[valid]\n"
    "    if len(tt) == 0:\n"
    "        return 0.0, 0\n"
    "    pred = lt.argmax(dim=-1).cpu().numpy()\n"
    "    tt_np = tt.cpu().numpy()\n"
    "    recalls = []\n"
    "    for c in present_classes:\n"
    "        idx = (tt_np == c)\n"
    "        n = int(idx.sum())\n"
    "        if n == 0:\n"
    "            continue\n"
    "        recalls.append(float((pred[idx] == c).sum()) / n)\n"
    "    if not recalls:\n"
    "        return 0.0, 0\n"
    "    return float(sum(recalls) / len(recalls)), len(recalls)\n"
)
s = s.replace(anchor, helper, 1)

# 2. wire the call into the test block (after macro_ovr_auc line)
call_anchor = "        m_auc, w_auc, n_cls = macro_ovr_auc(probs, tg, vd, present.tolist())\n"
assert call_anchor in s, "test macro_ovr_auc call not found"
call_add = call_anchor + "        mt1, n_mt1 = macro_top1(lg, tg, vd, present.tolist())\n"
s = s.replace(call_anchor, call_add, 1)

# 3. print it
print_anchor = '        print(f"test macro-OVR AUC: {m_auc:.4f}  (over {n_cls} classes)")\n'
assert print_anchor in s, "test print line not found"
print_add = print_anchor + '        print(f"test macro-top1   : {mt1:.4f}  (over {n_mt1} classes)")\n'
s = s.replace(print_anchor, print_add, 1)

# 4. log it to wandb
log_anchor = '        if wb: wb.log({"test/top1": t1, "test/top5": t5, "test/macro_auc": m_auc, "test/weighted_auc": w_auc})\n'
assert log_anchor in s, "test wandb log line not found"
log_new = '        if wb: wb.log({"test/top1": t1, "test/top5": t5, "test/macro_auc": m_auc, "test/weighted_auc": w_auc, "test/macro_top1": mt1})\n'
s = s.replace(log_anchor, log_new, 1)

open(p, "w").write(s)
print(f"{p}: macro_top1 added + wired into test eval + logged")
