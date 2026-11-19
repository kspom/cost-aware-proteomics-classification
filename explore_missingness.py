"""
Stage 1 — load Olink long-format data, characterise missingness, drop bad samples.

Pipeline:
  1. Load long-format file.
  2. Compute per-sample and per-protein missingness.
  3. Plot both (per-sample grouped by cancer, sorted desc within group;
     per-protein sorted desc).
  4. Filter samples with > SAMPLE_MISSING_THRESHOLD missing proteins.
  5. Pivot survivors to wide matrix and save for downstream stages.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("pancancer_olink_data_biostudies_v2.txt")
OUT_DIR = Path(".")
SAMPLE_MISSING_THRESHOLD = 110  # drop samples with strictly more than this many missing proteins


# ---------------------------------------------------------------- 1. load ---
df = pd.read_csv(DATA, sep=r"\s+", engine="c")
for col in ["Sample_ID", "Cancer", "Assay", "OlinkID", "UniProt", "Panel"]:
    df[col] = df[col].str.strip('"')

n_proteins = df["Assay"].nunique()
n_samples = df["Sample_ID"].nunique()
density = 100 * len(df) / (n_samples * n_proteins)
print(f"Loaded: {n_samples} samples × {n_proteins} proteins, "
      f"{len(df):,} rows ({density:.2f}% dense)")


# ----------------------------------------------------- 2. missingness stats ---
sample_meta = df.drop_duplicates("Sample_ID")[["Sample_ID", "Cancer"]]
sample_missing = (n_proteins - df.groupby("Sample_ID").size()).rename("missing")
sample_df = sample_meta.merge(sample_missing, on="Sample_ID")

protein_missing = (n_samples - df.groupby("Assay").size()).rename("missing")
protein_df = protein_missing.reset_index()


# ------------------------------------------------- 3a. per-sample bar plot ---
cancer_order = (
    sample_df.groupby("Cancer").size().sort_values(ascending=False).index.tolist()
)
sample_df["cancer_rank"] = sample_df["Cancer"].map(
    {c: i for i, c in enumerate(cancer_order)}
)
sample_df = sample_df.sort_values(
    ["cancer_rank", "missing"], ascending=[True, False]
).reset_index(drop=True)

cmap = plt.colormaps.get_cmap("tab20")
cancer_colors = {c: cmap(i % 20) for i, c in enumerate(cancer_order)}
colors = sample_df["Cancer"].map(cancer_colors).tolist()

fig, ax = plt.subplots(figsize=(20, 6))
x = np.arange(len(sample_df))
ax.bar(x, sample_df["missing"].values, color=colors, width=1.0, edgecolor="none")
ax.set_xlabel("Samples (grouped by cancer, sorted by missing count desc within group)")
ax.set_ylabel(f"Number of missing proteins (out of {n_proteins})")
ax.set_title("Per-sample missingness — grouped by cancer type")
ax.set_xlim(-0.5, len(sample_df) - 0.5)

boundaries = []
prev = None
for i, c in enumerate(sample_df["Cancer"]):
    if c != prev:
        boundaries.append((i, c))
        prev = c
for bx, _ in boundaries[1:]:
    ax.axvline(bx - 0.5, color="black", lw=0.4, alpha=0.4)

tick_pos, tick_label = [], []
for j, (bx, c) in enumerate(boundaries):
    end = boundaries[j + 1][0] if j + 1 < len(boundaries) else len(sample_df)
    tick_pos.append((bx + end) / 2)
    tick_label.append(f"{c}\n(n={end - bx})")
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_label, rotation=0, fontsize=9)

ax.axhline(SAMPLE_MISSING_THRESHOLD, color="red", lw=0.8, ls="--", alpha=0.8)
ax.text(len(sample_df) * 0.995, SAMPLE_MISSING_THRESHOLD + 8,
        f"threshold = {SAMPLE_MISSING_THRESHOLD}",
        color="red", fontsize=9, ha="right")

plt.tight_layout()
plt.savefig(OUT_DIR / "missingness_per_sample.png", dpi=140)
plt.close()


# ------------------------------------------------ 3b. per-protein bar plot ---
protein_df = protein_df.sort_values("missing", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 5))
ax.bar(np.arange(len(protein_df)), protein_df["missing"].values,
       color="steelblue", width=1.0, edgecolor="none")
ax.set_xlabel(f"Proteins (n={len(protein_df)}), sorted by missing count desc")
ax.set_ylabel("Number of samples missing this protein")
ax.set_title("Per-protein missingness")
ax.set_xlim(-0.5, len(protein_df) - 0.5)

ax.text(0.01, 0.95,
        f"median={int(protein_df['missing'].median())}, "
        f"max={int(protein_df['missing'].max())}, "
        f"proteins with 0 missing={int((protein_df['missing'] == 0).sum())}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey", alpha=0.8))

plt.tight_layout()
plt.savefig(OUT_DIR / "missingness_per_protein.png", dpi=140)
plt.close()


# ------------------------------------------------ 4. filter & report stats ---
keep_ids = sample_df.loc[
    sample_df["missing"] <= SAMPLE_MISSING_THRESHOLD, "Sample_ID"
]
n_dropped = n_samples - len(keep_ids)
print(f"\nSample filter (>{SAMPLE_MISSING_THRESHOLD} missing): "
      f"kept {len(keep_ids)}/{n_samples} ({n_dropped} dropped)")

print("\nClass balance after filter:")
kept = sample_df[sample_df["Sample_ID"].isin(keep_ids)]
total = sample_df.groupby("Cancer").size()
remaining = kept.groupby("Cancer").size()
balance = pd.DataFrame({"before": total, "after": remaining}).fillna(0).astype(int)
balance["dropped"] = balance["before"] - balance["after"]
print(balance.sort_values("before", ascending=False).to_string())


# ----------------------------- 5. pivot survivors to wide and save to disk ---
df_filt = df[df["Sample_ID"].isin(keep_ids)]
wide = df_filt.pivot(index="Sample_ID", columns="Assay", values="NPX")
labels = (
    df_filt.drop_duplicates("Sample_ID").set_index("Sample_ID")["Cancer"]
)
wide = wide.join(labels.rename("__Cancer__"))

out_csv = OUT_DIR / "filtered_wide.csv"
wide.to_csv(out_csv)
print(f"\nSaved wide matrix: {out_csv} "
      f"({wide.shape[0]} samples × {wide.shape[1] - 1} proteins + 1 label column)")

post = wide.drop(columns="__Cancer__")
post_missing = post.isna().sum()
print(f"Per-protein missingness AFTER filter: "
      f"max={int(post_missing.max())}, "
      f"median={int(post_missing.median())}, "
      f"proteins with 0 missing={int((post_missing == 0).sum())}")
