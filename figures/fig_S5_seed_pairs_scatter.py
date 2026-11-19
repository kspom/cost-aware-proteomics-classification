"""Supplementary Figure S5 — best seed pair (top-1 PairSeed
2-protein classifier) for each of the 12 cancers, plotted as a
2-D scatter.

For each disease the figure shows the cohort projected onto the two
best-discriminating proteins identified by PairSeed's seed-pair
search (Stage 1 of the algorithm; the k = 2 panel from the
no-cascade family). The target class is rendered in its
manuscript-standard cancer-specific colour; the remaining 11
classes are shown as small light-grey dots in the background. Some
cancers (e.g. MYEL, GLIOM, CLL) show a sharp 2-D cluster that
already isolates the target — the seed-pair search succeeds even
without any further expansion. Others (e.g. BRC, ENDC) show heavy
overlap with the background, indicating that a 2-protein detector
is insufficient and more proteins are required.

Input:
    paper_figures/pairseed_no_cascade_per_cancer_panels.csv  (from analysis/pairseed_no_cascade_family.py;
        rows with k_target = 2 give the top seed pair per cancer)
    filtered_wide.csv                                        (from explore_missingness.py)

Output:
    paper_figures/fig_S5_seed_pairs_scatter.pdf
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import fig_style  # noqa: E402
from pairseed_lib import load_cohort  # noqa: E402

fig_style.apply()

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Per-figure fonts
AXIS_LABEL_FONT = 14
TICK_FONT = 11
TITLE_FONT = 14

# Cancer-specific palette (matches the Alvez 2023 Fig. 5b ordering /
# colours, kept consistent across the manuscript).
CANCER_COLORS = {
    "AML":   "#9ec9e0",   # light blue
    "CLL":   "#1f6fb4",   # mid blue
    "LYMPH": "#1f5e54",   # dark teal
    "MYEL":  "#4cc5a3",   # mint green
    "LUNGC": "#a8c43d",   # olive
    "CRC":   "#a08864",   # tan / brown
    "GLIOM": "#e9c836",   # yellow
    "PRC":   "#e0843f",   # orange
    "BRC":   "#e9aaaa",   # pink / salmon
    "CVX":   "#a91828",   # dark red
    "ENDC":  "#c79bd0",   # light purple
    "OVC":   "#5a2d6a",   # dark purple
}

# Subplot order: alphabetical to match the boxplot Fig. 1 layout.
ALL_12 = sorted(CANCER_COLORS.keys())


# -------------------------- load best seed pairs ------------------------

panels = pd.read_csv(OUT / "pairseed_no_cascade_per_cancer_panels.csv")
k2 = panels[panels["k_target"] == 2]
best_pairs = {r["cancer"]: r["proteins"].split(",")
              for _, r in k2.iterrows()}
print("Top seed pair per cancer:")
for c in ALL_12:
    print(f"  {c:>6}: {' + '.join(best_pairs[c])}")


# -------------------------- load cohort ---------------------------------

# ALL patients, per-panel complete case (NaN kept)
X_use, labels_use, BLOCKS = load_cohort()
print(f"Cohort: {len(X_use)} samples (per-panel complete case)")


# -------------------------- plot ----------------------------------------

N_ROWS, N_COLS = 4, 3
SUBPLOT_W, SUBPLOT_H = 5.2, 4.2
fig, axes = plt.subplots(N_ROWS, N_COLS,
                         figsize=(N_COLS * SUBPLOT_W,
                                  N_ROWS * SUBPLOT_H))
axes = axes.flatten()

for j, target in enumerate(ALL_12):
    ax = axes[j]
    p1, p2 = best_pairs[target]
    x = X_use[p1].to_numpy()
    y = X_use[p2].to_numpy()
    has = ~(np.isnan(x) | np.isnan(y))   # patients with both proteins

    # Non-target samples first (background)
    non_target = (labels_use != target) & has
    ax.scatter(x[non_target], y[non_target],
               c="#bcbcbc", s=10, alpha=0.45,
               linewidths=0, zorder=2)

    # Target samples on top
    target_mask = (labels_use == target) & has
    n_target = int(target_mask.sum())
    ax.scatter(x[target_mask], y[target_mask],
               c=CANCER_COLORS[target], s=55, alpha=0.92,
               edgecolors="black", linewidths=0.6, zorder=4)

    ax.set_xlabel(p1, fontsize=AXIS_LABEL_FONT, fontweight="bold")
    ax.set_ylabel(p2, fontsize=AXIS_LABEL_FONT, fontweight="bold")
    ax.set_title(f"{target}  (n = {n_target})",
                 fontsize=TITLE_FONT, fontweight="bold")
    ax.tick_params(labelsize=TICK_FONT)
    ax.grid(True, alpha=0.2, zorder=1)

plt.tight_layout()
out_pdf = OUT / "fig_S5_seed_pairs_scatter.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_pdf}")
