"""
check_figo_distribution.py
Compare FIGO 2018 Stage distribution between paired (CT+PET) and CT-only patients.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CSV     = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR  = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
PET_DIR = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
OUT_DIR = "./outputs/distribution_causes"
os.makedirs(OUT_DIR, exist_ok=True)

# FIGO severity order for sorting
FIGO_ORDER = [
    'IA1','IA2','IB1','IB2','IB3',
    'IIA1','IIA2','IIB',
    'IIIA','IIIB','IIIC1','IIIC2',
    'IVA','IVB',
]
FIGO_MAJOR = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}

def figo_major(stage):
    """Extract major stage number (I→1, II→2, III→3, IV→4)."""
    s = str(stage).strip()
    for k in ['IVA','IVB','IV']:
        if s.startswith(k): return 4
    for k in ['IIIA','IIIB','IIIC1','IIIC2','III']:
        if s.startswith(k): return 3
    for k in ['IIA1','IIA2','IIB','II']:
        if s.startswith(k): return 2
    for k in ['IA1','IA2','IB1','IB2','IB3','I']:
        if s.startswith(k): return 1
    return np.nan

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df['Patient_ID'] = df['Patient_ID'].astype(str)
df['has_ct']  = df['Patient_ID'].apply(lambda p: os.path.exists(os.path.join(CT_DIR,  f"{p}_CT.npy")))
df['has_pet'] = df['Patient_ID'].apply(lambda p: os.path.exists(os.path.join(PET_DIR, f"{p}_PET.npy")))
df = df[df['has_ct']].copy()

figo_col = 'FIGO 2018 Stage'
df[figo_col] = df[figo_col].astype(str).str.strip()
df['figo_major'] = df[figo_col].apply(figo_major)

df_paired = df[df['has_pet']].copy()
df_ctonly = df[~df['has_pet']].copy()

# ── Print raw stage counts ────────────────────────────────────────────────────
print("=" * 60)
print("  FIGO 2018 Stage distribution")
print("=" * 60)

all_stages = sorted(
    set(df_paired[figo_col].dropna()) | set(df_ctonly[figo_col].dropna()),
    key=lambda s: FIGO_ORDER.index(s) if s in FIGO_ORDER else 99
)

print(f"\n{'Stage':<12} {'Paired':>10} {'Paired%':>9} {'CT-only':>10} {'CTonly%':>9}")
print("-" * 55)
for stage in all_stages:
    n_p = (df_paired[figo_col] == stage).sum()
    n_c = (df_ctonly[figo_col] == stage).sum()
    pct_p = 100 * n_p / len(df_paired) if len(df_paired) > 0 else 0
    pct_c = 100 * n_c / len(df_ctonly) if len(df_ctonly) > 0 else 0
    print(f"  {stage:<10} {n_p:>10}  {pct_p:>7.1f}%  {n_c:>10}  {pct_c:>7.1f}%")

# Unknown/missing
unk_p = df_paired[figo_col].isna().sum() + (df_paired[figo_col] == 'nan').sum()
unk_c = df_ctonly[figo_col].isna().sum() + (df_ctonly[figo_col] == 'nan').sum()
print(f"  {'unknown':<10} {unk_p:>10}             {unk_c:>10}")

# ── Major stage summary ───────────────────────────────────────────────────────
print(f"\n{'Major Stage':<12} {'Paired':>10} {'Paired%':>9} {'CT-only':>10} {'CTonly%':>9}")
print("-" * 55)
for major in [1, 2, 3, 4]:
    n_p = (df_paired['figo_major'] == major).sum()
    n_c = (df_ctonly['figo_major'] == major).sum()
    pct_p = 100 * n_p / len(df_paired)
    pct_c = 100 * n_c / len(df_ctonly)
    label = {1:'I', 2:'II', 3:'III', 4:'IV'}[major]
    print(f"  {label:<10} {n_p:>10}  {pct_p:>7.1f}%  {n_c:>10}  {pct_c:>7.1f}%")

# ── Statistical tests ─────────────────────────────────────────────────────────
valid_p = df_paired['figo_major'].dropna()
valid_c = df_ctonly['figo_major'].dropna()

mw_stat, mw_p = stats.mannwhitneyu(valid_p, valid_c, alternative='two-sided')
t_stat,  t_p  = stats.ttest_ind(valid_p, valid_c)
print(f"\n  Paired   major stage: mean={valid_p.mean():.2f}  median={valid_p.median():.1f}")
print(f"  CT-only  major stage: mean={valid_c.mean():.2f}  median={valid_c.median():.1f}")
print(f"\n  Mann-Whitney U: p={mw_p:.4f}" + ("  ← significant" if mw_p < 0.05 else ""))
print(f"  t-test        : p={t_p:.4f}"  + ("  ← significant" if t_p  < 0.05 else ""))

# Chi-square on major stage (2×4 table)
categories = [1, 2, 3, 4]
counts_p = [int((valid_p == c).sum()) for c in categories]
counts_c = [int((valid_c == c).sum()) for c in categories]
table = np.array([counts_p, counts_c])
if table.min() >= 0 and table.shape[1] >= 2:
    chi2, chi_p, _, _ = stats.chi2_contingency(table)
    print(f"  Chi-square    : χ²={chi2:.2f}  p={chi_p:.4f}" +
          ("  ← significant" if chi_p < 0.05 else ""))

# ── Stacked bar chart ─────────────────────────────────────────────────────────
colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']   # I=green II=blue III=orange IV=red
labels = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']

pct_p_list = [100 * (valid_p == c).mean() for c in categories]
pct_c_list = [100 * (valid_c == c).mean() for c in categories]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Stacked bar
ax = axes[0]
x      = np.arange(2)
bottoms = np.zeros(2)
for i, (label, color) in enumerate(zip(labels, colors)):
    vals = [pct_p_list[i], pct_c_list[i]]
    ax.bar(x, vals, bottom=bottoms, color=color, alpha=0.85, label=label, width=0.5)
    for j, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 2:
            ax.text(x[j], b + v / 2, f"{v:.0f}%", ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
    bottoms += np.array(vals)
ax.set_xticks(x)
ax.set_xticklabels([f"Paired\n(n={len(valid_p)})", f"CT-only\n(n={len(valid_c)})"])
ax.set_ylabel("Percentage (%)")
ax.set_title(f"FIGO Stage distribution\nχ²p={chi_p:.4f}  MW p={mw_p:.4f}")
ax.legend(loc='upper right', fontsize=8)

# Box plot of numeric major stage
ax = axes[1]
ax.boxplot([valid_p, valid_c], tick_labels=["Paired", "CT-only"],
           patch_artist=True, boxprops=dict(facecolor="steelblue", alpha=0.6))
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(['I', 'II', 'III', 'IV'])
ax.set_ylabel("FIGO Major Stage")
ax.set_title(f"Major stage distribution\nMW p={mw_p:.4f}  t p={t_p:.4f}")

plt.tight_layout()
out = os.path.join(OUT_DIR, "figo_stage_distribution.png")
plt.savefig(out, dpi=150)
print(f"\n  Figure saved → {out}")
