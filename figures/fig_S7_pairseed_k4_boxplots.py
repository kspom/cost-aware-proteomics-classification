"""Supplementary Figure S7 — LDA score boxplots for the 12 OvR
PairSeed-no-cascade detectors at the compact k = 4 operating point
(~44 unique proteins across the 12 detectors).

Per-cancer panels are read from the no-cascade family CSV at
k_target = 4. For each cancer a binary OvR LDA is fit on the cohort
using only that cancer's 4 proteins; raw NPX inputs (no
standardisation) keep the formulas directly deployable.

Outputs:
    paper_figures/pairseed_k4_formulas.csv    (one row per
        cancer: intercept + per-protein raw-NPX weights)
    paper_figures/pairseed_k4_proteins.csv    (per-cancer
        markers ranked by |weight|)
    paper_figures/fig_S7_pairseed_k4_boxplots.pdf

Layout and fonts as in Figure 1 (4 rows x 3 columns).
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import lda_fit, load_cohort, panel_mask, panel_scores  # noqa: E402
import fig_style  # noqa: E402

fig_style.apply()

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
K_TARGET = 4


# -------------------------- load panels ---------------------------------

def load_panels_at_k(k):
    new = OUT / "pairseed_no_cascade_per_cancer_panels.csv"
    legacy = OUT / "pairseed_per_cancer_panels_v2.csv"
    if new.exists():
        print(f"  loaded {new}")
        df = pd.read_csv(new)
    elif legacy.exists():
        print(f"  loaded {legacy}  (legacy fallback)")
        df = pd.read_csv(legacy)
    else:
        raise FileNotFoundError(
            "Neither pairseed_no_cascade_per_cancer_panels.csv nor "
            "pairseed_per_cancer_panels_v2.csv is in paper_figures/")
    df = df[df["k_target"] == k]
    return {r["cancer"]: r["proteins"].split(",") for _, r in df.iterrows()}


print(f"Per-cancer panels at k_target = {K_TARGET}:")
panels = load_panels_at_k(K_TARGET)
for c in sorted(panels):
    print(f"  {c:>6} (k={len(panels[c])}): {' + '.join(panels[c])}")

ALL_12 = sorted(panels.keys())
all_proteins = set().union(*panels.values())
print(f"Total unique proteins across 12 detectors: {len(all_proteins)}")


# -------------------------- load cohort ---------------------------------

# ALL patients, per-panel complete case (NaN kept)
X_use, labels_use, BLOCKS = load_cohort()
COMPLETE = BLOCKS[1] == 0   # boxplots show the 1094 complete patients
print(f"Cohort: {len(X_use)} samples (per-panel complete case)")


def fit_raw_lda(panel, target_class):
    """Binary LDA `target_class vs all-other` on raw NPX inputs."""
    m = panel_mask([X_use.columns.get_loc(p) for p in panel], BLOCKS)
    Xp = X_use.loc[m, panel].to_numpy(dtype=float)
    y = (labels_use[m] == target_class).astype(int)
    w, b = lda_fit(Xp, y)
    return w, b


# -------------------------- fit + collect formulas ----------------------

formula_rows = []
imp_rows = []
all_protein_cols = sorted(all_proteins)
score_per_sample = np.full((len(X_use), len(ALL_12)), np.nan)

for j, c in enumerate(ALL_12):
    panel = panels[c]
    w, b = fit_raw_lda(panel, c)
    row = {"class": c, "panel_size": len(panel), "intercept": float(b)}
    for p in all_protein_cols:
        row[p] = 0.0
    for p, weight in zip(panel, w):
        row[p] = float(weight)
    formula_rows.append(row)
    for rank, (p, weight) in enumerate(
            sorted(zip(panel, w), key=lambda x: -abs(x[1])), 1):
        imp_rows.append({"class": c, "rank": rank,
                         "protein": p, "weight_raw_NPX": float(weight)})
    score_per_sample[:, j] = panel_scores(
        X_use.to_numpy(dtype=float),
        [X_use.columns.get_loc(p) for p in panel], w, b)

formula_df = pd.DataFrame(formula_rows)
formula_df.to_csv(OUT / "pairseed_k4_formulas.csv", index=False)
imp_df = pd.DataFrame(imp_rows)
imp_df.to_csv(OUT / "pairseed_k4_proteins.csv", index=False)
print(f"Saved {OUT/'pairseed_k4_formulas.csv'}")
print(f"Saved {OUT/'pairseed_k4_proteins.csv'}")

print("\nPer-class formulas (raw NPX weights):")
for c in ALL_12:
    panel = panels[c]
    fr = formula_df[formula_df["class"] == c].iloc[0]
    parts = [f"{w:+.3f}*{p}" for p, w in
             sorted(zip(panel, [fr[p] for p in panel]),
                    key=lambda x: -abs(x[1]))]
    print(f"  I({c}) = {fr['intercept']:+.3f}  "
          f"{' '.join(parts)}")


# -------------------------- plot 4×3 grid (as Figure 1) ---------------

N_ROWS, N_COLS = 4, 3
SUBPLOT_W, SUBPLOT_H = 5.5, 3.6
TITLE_FONT = 16
TICK_FONT = 16
YLABEL_FONT = 20
TARGET_COLOR = fig_style.COLORS["pairseed_no_cascade"]
NEG_COLOR = "#a8c0e0"

fig, axes = plt.subplots(N_ROWS, N_COLS,
                         figsize=(N_COLS * SUBPLOT_W, N_ROWS * SUBPLOT_H))
axes = axes.flatten()
for j, c in enumerate(ALL_12):
    ax = axes[j]
    col = score_per_sample[:, j]
    by_disease = [col[(labels_use == d) & ~np.isnan(col) & COMPLETE]
                  for d in ALL_12]
    bp = ax.boxplot(
        by_disease, labels=ALL_12, showmeans=False,
        patch_artist=True, widths=0.65,
        medianprops=dict(color="black", linewidth=1.4),
        flierprops=dict(marker=".", markersize=2.5,
                        markerfacecolor="grey", markeredgecolor="grey"),
    )
    for k, patch in enumerate(bp["boxes"]):
        is_target = ALL_12[k] == c
        patch.set_facecolor(TARGET_COLOR if is_target else NEG_COLOR)
        patch.set_alpha(0.85 if is_target else 0.55)
    ax.axhline(0, color=fig_style.COLORS["grid"],
               linewidth=0.8, linestyle="--", zorder=0)
    panel = panels[c]
    title_lines = [f"{c} — panel size {len(panel)}"]
    chunk = []
    for p in panel:
        chunk.append(p)
        if len(" + ".join(chunk)) > 28:
            title_lines.append(" + ".join(chunk))
            chunk = []
    if chunk:
        title_lines.append(" + ".join(chunk))
    ax.set_title("\n".join(title_lines), fontsize=TITLE_FONT,
                 fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=TICK_FONT)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")
    ax.tick_params(axis="y", labelsize=TICK_FONT)
    ax.grid(True, axis="y", alpha=0.25)
    if j % N_COLS == 0:
        ax.set_ylabel("LDA score", fontsize=YLABEL_FONT)

plt.tight_layout()
out_pdf = OUT / "fig_S7_pairseed_k4_boxplots.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_pdf}")
