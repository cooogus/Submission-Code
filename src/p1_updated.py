"""
p1_updated.py
-------------
Takes the existing p1_results.csv (41 IRT fits from Squad A's cleaned data)
and partitions them into two reporting groups:

  1. All fits       -> all 41 rows, no filtering
  2. Clean subset   -> rows passing all three quality criteria:
       (a) n_questions >= 30
       (b) gamma NOT at a boundary  (gamma > 0.01  AND  gamma < 29.9)
       (c) |alpha - beta| > 0.02

Reports mean R² for each group and prints a full audit table.
"""

import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV  = "p1_results.csv"   # path to your existing results file
GAMMA_LO   = 0.01               # anything <= this is considered boundary (≈ 0.001)
GAMMA_HI   = 29.9               # anything >= this is considered boundary (≈ 30)
N_MIN      = 30                 # minimum number of questions
ALPHA_BETA_DELTA = 0.02         # minimum |alpha - beta| required

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from '{INPUT_CSV}'\n")

# ── Quality flag columns ──────────────────────────────────────────────────────

df["flag_n_ok"]       = df["n_questions"] >= N_MIN
df["flag_gamma_ok"]   = (df["gamma"] > GAMMA_LO) & (df["gamma"] < GAMMA_HI)
df["flag_ab_ok"]      = (df["alpha"] - df["beta"]).abs() > ALPHA_BETA_DELTA

df["is_clean"] = df["flag_n_ok"] & df["flag_gamma_ok"] & df["flag_ab_ok"]

# human-readable failure reason (for the audit table)
def failure_reason(row):
    reasons = []
    if not row["flag_n_ok"]:
        reasons.append(f"n={row['n_questions']}<30")
    if not row["flag_gamma_ok"]:
        g = row["gamma"]
        tag = "γ≈0" if g <= GAMMA_LO else "γ≈30"
        reasons.append(tag)
    if not row["flag_ab_ok"]:
        reasons.append(f"|α-β|={abs(row['alpha']-row['beta']):.4f}≤0.02")
    return ", ".join(reasons) if reasons else "PASS"

df["fail_reason"] = df.apply(failure_reason, axis=1)

# ── Split ─────────────────────────────────────────────────────────────────────

all_fits   = df.copy()
clean_fits = df[df["is_clean"]].copy()

# ── Statistics ────────────────────────────────────────────────────────────────

mean_r2_all   = all_fits["r2"].mean()
mean_r2_clean = clean_fits["r2"].mean()

n_all   = len(all_fits)
n_clean = len(clean_fits)
n_dirty = n_all - n_clean

# per-filter counts (of excluded rows)
n_fail_n     = (~df["flag_n_ok"]).sum()
n_fail_gamma = (~df["flag_gamma_ok"]).sum()
n_fail_ab    = (~df["flag_ab_ok"]).sum()

# ── Audit table ───────────────────────────────────────────────────────────────

print("=" * 80)
print("AUDIT TABLE  (all 41 fits, with quality flags)")
print("=" * 80)

audit_cols = [
    "model", "benchmark", "n_questions",
    "alpha", "beta", "gamma", "r2",
    "flag_n_ok", "flag_gamma_ok", "flag_ab_ok", "fail_reason"
]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)
print(df[audit_cols].to_string(index=False))

# ── Summary report ────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)

print(f"\n{'Group':<20} {'N fits':>7}  {'Mean R²':>9}  {'Median R²':>10}  {'Min R²':>8}  {'Max R²':>8}")
print("-" * 70)

for label, subset in [("All fits", all_fits), ("Clean subset", clean_fits)]:
    r2 = subset["r2"]
    print(
        f"{label:<20} {len(r2):>7}  {r2.mean():>9.4f}  {r2.median():>10.4f}"
        f"  {r2.min():>8.4f}  {r2.max():>8.4f}"
    )

print("\n" + "-" * 70)
print(f"Excluded fits       : {n_dirty} / {n_all}")
print(f"  └─ n < 30         : {n_fail_n}")
print(f"  └─ γ at boundary  : {n_fail_gamma}")
print(f"  └─ |α-β| ≤ 0.02   : {n_fail_ab}")
print("  (a single fit can fail multiple criteria)")

print(f"\nMean R² all fits    : {mean_r2_all:.4f}")
print(f"Mean R² clean only  : {mean_r2_clean:.4f}")
print(f"Difference          : {mean_r2_all - mean_r2_clean:+.4f}  "
      f"({'inflated' if mean_r2_all > mean_r2_clean else 'deflated'} by including dirty fits)")

# ── Export ────────────────────────────────────────────────────────────────────

out_all   = "p1_all_fits.csv"
out_clean = "p1_clean_fits.csv"

all_fits.to_csv(out_all, index=False)
clean_fits.to_csv(out_clean, index=False)

print(f"\nSaved: {out_all}  ({n_all} rows)")
print(f"Saved: {out_clean}  ({n_clean} rows)")
print("Done.")
