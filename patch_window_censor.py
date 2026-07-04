# Windowing fix for dropout K-sweep: censor students whose TOTAL interaction count
# is <= k_prefix (their whole trajectory fits in the prefix -> label leaks).
# Adds --window_censor flag; when set, eligible rows require total_len > k_prefix.
# This recovers K=100/200 as clean eval points on the long-sequence subset.
# Run from repo root: python patch_window_censor.py
import re

p = "scripts/downstream_edubert.py"
s = open(p).read()
if "window_censor" in s:
    print(f"{p}: already patched"); raise SystemExit

# 1. add window_censor param to the dataset __init__ signature
old_sig = "def __init__(self, processed_dir, split, max_seq_len, dropout_labels=None, k_prefix=None):"
new_sig = "def __init__(self, processed_dir, split, max_seq_len, dropout_labels=None, k_prefix=None, window_censor=False):"
assert old_sig in s, "init signature not found"
s = s.replace(old_sig, new_sig, 1)

# 2. store the flag + apply the censor filter after the eligibility filter
old_filt = ("        if dropout_labels is not None:\n"
            "            # keep only eligible students (those present in the label dict)\n"
            "            self.rows = [r for r in self.rows if r in dropout_labels]")
new_filt = ("        self.window_censor = window_censor\n"
            "        if dropout_labels is not None:\n"
            "            # keep only eligible students (those present in the label dict)\n"
            "            self.rows = [r for r in self.rows if r in dropout_labels]\n"
            "            if window_censor and k_prefix is not None:\n"
            "                # censor students whose whole trajectory fits in the prefix\n"
            "                # (total_len <= K means the label leaks into the input window)\n"
            "                def _tot(r):\n"
            "                    return int(self.offsets[r + 1] - self.offsets[r])\n"
            "                before = len(self.rows)\n"
            "                self.rows = [r for r in self.rows if _tot(r) > k_prefix]\n"
            "                print(f'window_censor: kept {len(self.rows)}/{before} students with total_len > {k_prefix}')")
assert old_filt in s, "eligibility filter block not found"
s = s.replace(old_filt, new_filt, 1)

# 3. add the CLI flag (near k_prefix arg)
old_arg = '    ap.add_argument("--k_prefix", type=int, default=20,'
new_arg = ('    ap.add_argument("--window_censor", action="store_true",\n'
           '                    help="exclude students whose total_len <= k_prefix (recovers leaky high-K)")\n'
           '    ap.add_argument("--k_prefix", type=int, default=20,')
assert old_arg in s, "k_prefix arg not found"
s = s.replace(old_arg, new_arg, 1)

# 4. thread the flag into BOTH dataset constructions (train + eval use k_prefix=args.k_prefix)
old_ds = "k_prefix=args.k_prefix)"
new_ds = "k_prefix=args.k_prefix, window_censor=args.window_censor)"
n = s.count(old_ds)
s = s.replace(old_ds, new_ds)
print(f"threaded window_censor into {n} dataset construction(s)")

open(p, "w").write(s)
print(f"{p}: patched")
