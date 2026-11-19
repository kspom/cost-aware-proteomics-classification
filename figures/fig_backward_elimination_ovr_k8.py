"""Backward-elimination accuracy curves for each of the 12 PairSeed-OvR
k=8 panels.

Each subplot shows balanced accuracy of an OvR detector as a function
of panel size, starting at 1 protein and ending at the full k-protein
panel. The trajectory was obtained by greedy backward elimination
(see analysis/backward_elimination_ovr_k8.py): at each step, the
protein whose removal yields the highest balanced accuracy is
dropped. Plotted with n_proteins on the x-axis so the curve rises
left-to-right.

Reads:
    paper_figures/backward_elimination_ovr_k8.csv

Output:
    paper_figures/fig_backward_elimination_ovr_k8.pdf

Layout mirrors Figure 1 (PairSeed-OvR boxplots): 4 rows × 3 columns,
alphabetical order over the 12 cancers.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import fig_style  # noqa: E402

fig_style.apply()

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Layout — match fig_1 boxplots
N_ROWS, N_COLS = 4, 3
SUBPLOT_W, SUBPLOT_H = 4.2, 3.0
FIG_W = N_COLS * SUBPLOT_W
FIG_H = N_ROWS * SUBPLOT_H

TITLE_FONT = 18
TICK_FONT = 14
AXIS_LABEL_FONT = 16
ANNOT_FONT = 12

LINE_COLOR = fig_style.COLORS["pairseed_no_cascade"]
MARKER_COLOR = LINE_COLOR


# -------------------------- load --------------------------------------

df = pd.read_csv(OUT / "backward_elimination_ovr_k8.csv")
ALL_12 = sorted(df["cancer"].unique())


# -------------------------- plot 4×3 grid -----------------------------

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_W, FIG_H),
                         sharey=False)
axes = axes.flatten()

for j, c in enumerate(ALL_12):
    ax = axes[j]
    sub = df[df["cancer"] == c].sort_values("n_proteins")
    xs = sub["n_proteins"].to_numpy()
    ys = sub["bal_acc"].to_numpy()

    ax.plot(xs, ys, "-o", color=LINE_COLOR, markersize=7,
            linewidth=2.2, markerfacecolor=MARKER_COLOR,
            markeredgecolor="black", markeredgewidth=0.6)

    # k_max annotation at the right end
    ax.annotate(f"{ys[-1]:.1f}%",
                xy=(xs[-1], ys[-1]),
                xytext=(-4, -14),
                textcoords="offset points",
                fontsize=ANNOT_FONT, ha="right",
                fontweight="bold", color="black")
    # k=1 annotation at the left end
    ax.annotate(f"{ys[0]:.1f}%",
                xy=(xs[0], ys[0]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=ANNOT_FONT, ha="left",
                fontweight="bold", color="black")

    ax.set_title(c, fontsize=TITLE_FONT, fontweight="bold", pad=4)
    ax.set_xticks(range(1, int(xs.max()) + 1))
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    ax.grid(True, alpha=0.3)

    # Per-subplot y-range: pad below 1-protein point by ~5pp and
    # above full-panel point by ~3pp; MYEL trajectory is in the 98-100
    # band so use a tight window to make the variation visible.
    if c == "MYEL":
        ax.set_ylim(98.0, 100.0)
    else:
        ymin = max(50.0, ys.min() - 5)
        ymax = min(101.0, ys.max() + 3)
        ax.set_ylim(ymin, ymax)
    ax.set_xlim(0.5, xs.max() + 0.5)

    if j % N_COLS == 0:
        ax.set_ylabel("Balanced accuracy (%)",
                      fontsize=AXIS_LABEL_FONT)
    if j // N_COLS == N_ROWS - 1:
        ax.set_xlabel("Panel size (proteins)",
                      fontsize=AXIS_LABEL_FONT)

plt.tight_layout()
out_pdf = OUT / "fig_backward_elimination_ovr_k8.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved {out_pdf}")
