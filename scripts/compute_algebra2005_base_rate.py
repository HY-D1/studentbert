# STEP 1 ONLY: compute Algebra2005 correctness base rate from raw KDD Cup file.
# Does NOT touch the pipeline, does NOT run any model, does NOT compute the regime.
# Just: mean of "Correct First Attempt" over all interactions = the base rate.
# Usage: python compute_algebra2005_base_rate.py <path_to_algebra_2005_2006_train.txt>
import sys, csv

def main(path):
    total = 0
    correct_sum = 0
    missing = 0
    # KDD Cup files are tab-delimited with a header row
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f, delimiter='\t')
        # find the correctness column (exact name in KDD Cup format)
        cols = reader.fieldnames
        cfa_col = None
        for c in cols:
            if c.strip().lower() == "correct first attempt":
                cfa_col = c; break
        if cfa_col is None:
            print("ERROR: 'Correct First Attempt' column not found. Columns are:")
            for c in cols: print("   ", repr(c))
            return
        print(f"using correctness column: {cfa_col!r}")
        for row in reader:
            v = row.get(cfa_col, "").strip()
            if v == "" :
                missing += 1
                continue
            try:
                iv = int(float(v))
            except ValueError:
                missing += 1
                continue
            total += 1
            correct_sum += iv
    if total == 0:
        print("no valid rows parsed"); return
    base_rate = correct_sum / total
    print(f"\n=== ALGEBRA2005 BASE RATE ===")
    print(f"total interactions (with correctness): {total}")
    print(f"missing/blank correctness rows       : {missing}")
    print(f"correct count                        : {correct_sum}")
    print(f"BASE RATE (mean Correct First Attempt): {base_rate:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python compute_algebra2005_base_rate.py <algebra_2005_2006_train.txt>")
        sys.exit(1)
    main(sys.argv[1])
