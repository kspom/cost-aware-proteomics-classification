"""Backward-elimination accuracy curves for the 7 REFEREE cascade tests
(within-bucket OvR disambiguators):
    Trio       — T41_CRC, T42_LUNGC, T43_OVC
    Residual4  — T71_BRC, T72_CVX, T73_ENDC, T74_PRC

Reads paper_figures/backward_elimination_cascade_k8.csv (produced by
analysis/backward_elimination_cascade_k8.py).

Layout: 2 rows × 4 columns. Top row = trio referees (3 cells + 1
empty); bottom row = residual4 referees (4 cells).

Output:
    paper_figures/fig_backward_elimination_cascade_referees.pdf
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

# Top row trio, bottom row residual4 — None marks blank cells.
GRID = [
    "T41_CRC", "T42_LUNGC", "T43_OVC", None,
    "T71_BRC", "T72_CVX", "T73_ENDC", "T74_PRC",
]

N_ROWS, N_COLS = 2, 4
SUBPLOT_W, SUBPLOT_H = 4.2, 3.0
FIG_W = N_COLS * SUBPLOT_W
FIG_H = N_ROWS * SUBPLOT_H

TITLE_FONT = 16
TICK_FONT = 13
AXIS_LABEL_FONT = 15
ANNOT_FONT = 11

LINE_COLOR = fig_style.COLORS["cascade"]
MARKER_COLOR = LINE_COLOR

df = pd.read_csv(OUT / "backward_elimination_cascade_k8.csv")

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_W, FIG_H),
                         sharey=False)
axes = axes.flatten()

TIGHT_YLIM = {
    "T74_PRC": (87.0, 100.5),
}

for j, tid in enumerate(GRID):
    ax = axes[j]
    if tid is None:
        ax.axis("off")
        continue
    sub = df[df["test_id"] == tid].sort_values("n_proteins")
    xs = sub["n_proteins"].to_numpy()
    ys = sub["bal_acc"].to_numpy()

    ax.plot(xs, ys, "-o", color=LINE_COLOR, markersize=7,
            linewidth=2.2, markerfacecolor=MARKER_COLOR,
            markeredgecolor="black", markeredgewidth=0.6)

    ax.annotate(f"{ys[-1]:.1f}%",
                xy=(xs[-1], ys[-1]),
                xytext=(-4, -14), textcoords="offset points",
                fontsize=ANNOT_FONT, ha="right",
                fontweight="bold", color="black")
    ax.annotate(f"{ys[0]:.1f}%",
                xy=(xs[0], ys[0]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=ANNOT_FONT, ha="left",
                fontweight="bold", color="black")

    ax.set_title(tid, fontsize=TITLE_FONT, fontweight="bold", pad=4)
    ax.set_xticks(range(1, int(xs.max()) + 1))
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    ax.grid(True, alpha=0.3)

    if tid in TIGHT_YLIM:
        ax.set_ylim(*TIGHT_YLIM[tid])
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
out_pdf = OUT / "fig_backward_elimination_cascade_referees.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved {out_pdf}")
