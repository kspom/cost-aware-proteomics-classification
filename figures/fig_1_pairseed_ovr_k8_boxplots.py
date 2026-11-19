"""Figure 1 — PairSeed-OvR boxplots at k = 8.

12-panel grid arranged as **4 rows × 3 columns**. Each subplot shows
the LDA score of one cancer's OvR detector across all 12 disease
groups (target class highlighted in green, non-target groups in
light blue). Reads the canonical k = 8 panels produced by
`pairseed_ovr_k8.py`; refits binary LDA per cancer on all patients
with complete data for that panel (of 1273); each box shows the
patients whose score is available.

Input:
    paper_figures/pairseed_ovr_k8_panels.csv
    filtered_wide.csv

Output:
    paper_figures/fig_1_pairseed_ovr_k8_boxplots.pdf
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
K_TARGET = 8

# Visual layout
N_ROWS, N_COLS = 4, 3
SUBPLOT_W, SUBPLOT_H = 5.5, 3.6  # inches per subplot (bigger labels)
FIG_W = N_COLS * SUBPLOT_W
FIG_H = N_ROWS * SUBPLOT_H

# Per-subplot font sizes (~1.7x the fig_style defaults — disease labels
# at 90° rotation need more room).
TITLE_FONT = 16
TICK_FONT = 16
YLABEL_FONT = 20

TARGET_COLOR = fig_style.COLORS["pairseed_no_cascade"]
NEG_COLOR = "#a8c0e0"


# -------------------------- load panels ----------------------------------

per_cancer = pd.read_csv(OUT / "pairseed_ovr_k8_panels.csv")
panels = {r["cancer"]: r["proteins"].split(",")
          for _, r in per_cancer.iterrows()}
ALL_12 = sorted(panels.keys())
all_proteins = sorted(set().union(*panels.values()))
print(f"PairSeed OvR k = {K_TARGET}: {len(panels)} detectors, "
      f"{len(all_proteins)} unique proteins")


# -------------------------- load cohort ----------------------------------

# ALL patients, per-panel complete case (NaN kept)
X_use, labels_use, BLOCKS = load_cohort()
COMPLETE = BLOCKS[1] == 0   # boxplots show the 1094 complete patients
print(f"Cohort: {len(X_use)} samples (per-panel complete case)")


# -------------------------- fit per cancer and score ---------------------

XNP = X_use.to_numpy(dtype=float)
P_INDEX = {p: i for i, p in enumerate(X_use.columns)}
score = np.full((len(X_use), len(ALL_12)), np.nan)
for j, c in enumerate(ALL_12):
    idx = [P_INDEX[p] for p in panels[c]]
    m = panel_mask(idx, BLOCKS)
    y = (labels_use[m] == c).astype(int)
    w, b = lda_fit(XNP[m][:, idx], y)
    score[:, j] = panel_scores(XNP, idx, w, b)
    print(f"  {c:>6}: {int(m.sum())} patients with complete panel")


# -------------------------- plot 4×3 grid --------------------------------

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_W, FIG_H))
axes = axes.flatten()

for j, c in enumerate(ALL_12):
    ax = axes[j]
    col = score[:, j]
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

    # Title: cancer name + panel size + panel composition broken into
    # short lines so subplot titles don't overlap.
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
    ax.set_title("\n".join(title_lines),
                 fontsize=TITLE_FONT,
                 fontweight="bold", pad=6)

    ax.tick_params(axis="x", labelsize=TICK_FONT)
    plt.setp(ax.get_xticklabels(), rotation=90,
             ha="center", va="top")
    ax.tick_params(axis="y", labelsize=TICK_FONT)
    ax.grid(True, axis="y", alpha=0.25)

    # Y-axis label only on left column
    if j % N_COLS == 0:
        ax.set_ylabel("LDA score", fontsize=YLABEL_FONT)

plt.tight_layout()
out_pdf = OUT / "fig_1_pairseed_ovr_k8_boxplots.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved {out_pdf}")
