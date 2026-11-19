"""Deployable boxplot + formulas for the 12 PairSeed OvR detectors at
k = 8. Reads the per-cancer panels produced by `pairseed_ovr_k8.py`,
fits binary LDA per cancer on the full cohort, and writes:

  - a 12-panel grid of LDA-score boxplots (one panel per cancer,
    showing the score of that cancer's OvR detector across all 12
    disease groups);
  - a CSV with the deployable formula (intercept + per-protein weight)
    for each of the 12 detectors.

Outputs:
  paper_figures/pairseed_ovr_k8_boxplots.pdf
  paper_figures/pairseed_ovr_k8_formulas.csv
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import lda_fit  # noqa: E402

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
K_TARGET = 8

per_cancer = pd.read_csv(OUT / "pairseed_ovr_k8_panels.csv")
panels = {}
print(f"PairSeed OvR k={K_TARGET} per-cancer panels:")
for _, r in per_cancer.iterrows():
    panel = r["proteins"].split(",")
    panels[r["cancer"]] = panel
    print(f"  {r['cancer']:>6} (k={len(panel)}): {' + '.join(panel)}")

ALL_12 = sorted(panels.keys())
all_proteins = set().union(*panels.values())
print(f"Total unique: {len(all_proteins)}")

wide = pd.read_csv("filtered_wide.csv", index_col="Sample_ID")
labels = wide["__Cancer__"]
X_full = wide.drop(columns="__Cancer__")
if "XG" in X_full.columns:
    X_full = X_full.drop(columns="XG")
nan_frac = X_full.isna().mean(axis=0)
DENSE = nan_frac[nan_frac <= 0.05].index.tolist()
keep = X_full[DENSE].notna().all(axis=1)
X_use = X_full.loc[keep, DENSE]
labels_use = labels[keep].to_numpy()
print(f"Cohort: {len(X_use)} complete-case samples")


def fit_raw_lda(panel, target_class):
    Xp = X_use[panel].to_numpy()
    y = (labels_use == target_class).astype(int)
    w, b = lda_fit(Xp, y)
    return w, b


# Score every sample for every detector
score = np.zeros((len(X_use), len(ALL_12)))
formula_rows = []
all_protein_cols = sorted(all_proteins)
for j, c in enumerate(ALL_12):
    panel = panels[c]
    w, b = fit_raw_lda(panel, c)
    Xp = X_use[panel].to_numpy()
    score[:, j] = Xp @ w + b
    row = {"class": c, "panel_size": len(panel), "intercept": float(b)}
    for p in all_protein_cols:
        row[p] = 0.0
    for p, weight in zip(panel, w):
        row[p] = float(weight)
    formula_rows.append(row)

pd.DataFrame(formula_rows).to_csv(
    OUT / "pairseed_ovr_k8_formulas.csv", index=False)
print(f"Saved {OUT/'pairseed_ovr_k8_formulas.csv'}")

# Boxplot grid
fig, axes = plt.subplots(3, 4, figsize=(20, 13))
axes = axes.flatten()
for j, c in enumerate(ALL_12):
    ax = axes[j]
    by_disease = []
    for d in ALL_12:
        mask = (labels_use == d)
        by_disease.append(score[mask, j])
    bp = ax.boxplot(by_disease, labels=ALL_12, showmeans=False,
                    patch_artist=True, widths=0.65,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=3,
                                    markerfacecolor="grey",
                                    markeredgecolor="grey"))
    for k, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#1f7a3c" if ALL_12[k] == c else "#a8c0e0")
        patch.set_alpha(0.85 if ALL_12[k] == c else 0.55)
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    # break the panel list into multiple lines so titles don't overlap
    plist = panels[c]
    title_lines = [f"{c} (k={len(plist)})"]
    chunk = []
    for p in plist:
        chunk.append(p)
        if len(" + ".join(chunk)) > 32:
            title_lines.append(" + ".join(chunk))
            chunk = []
    if chunk:
        title_lines.append(" + ".join(chunk))
    ax.set_title("\n".join(title_lines),
                 fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", rotation=60, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)
fig.suptitle(
    f"PairSeed OvR at k = {K_TARGET} "
    f"({len(all_proteins)} unique proteins across the 12 detectors).\n"
    f"Each subplot: one cancer's binary LDA score across the 12 "
    f"disease groups (green = target).",
    fontsize=13, y=0.995)
plt.tight_layout(rect=(0, 0, 1, 0.965))
plt.savefig(OUT / "pairseed_ovr_k8_boxplots.pdf",
            bbox_inches="tight")
plt.close()
print(f"Saved {OUT/'pairseed_ovr_k8_boxplots.pdf'}")
