"""Supplementary Figure S9 — leave-one-disease-out
routing matrices. (A) PairSeed-no-cascade, (B) PairSeedCascade-with-referees.

Rows: disease removed from training; columns: predicted class ("unknown"
first). Cell value: % of the held-out disease's patients with complete
protein data routed to that prediction. Framed cells: favourable outcomes
("unknown" in A; another member of the same bucket in B).

Inputs:
    paper_figures/honest_lodo_pairseed_nocascade_summary.csv
    paper_figures/honest_lodo_cascade_with_referees_summary.csv
Run from the repository root:
    python figures/fig_S9_lodo_matrices.py
Output:
    paper_figures/fig_S9_lodo_matrices.pdf
"""
import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
import fig_style  # noqa: E402

fig_style.apply()
OUT = ROOT / "paper_figures"

AXIS_LABEL_FONT, TICK_FONT, CELL_FONT, PANEL_FONT = 22, 16, 13, 30
ROW_ORDER = ["AML", "MYEL", "CLL", "LYMPH", "GLIOM",
             "CRC", "LUNGC", "OVC", "BRC", "CVX", "ENDC", "PRC"]
COL_ORDER = ["unknown"] + ROW_ORDER
TRIO = {"CRC", "LUNGC", "OVC"}
RES4 = {"BRC", "CVX", "ENDC", "PRC"}


def matrix(fname):
    s = pd.read_csv(OUT / fname)
    M = np.zeros((len(ROW_ORDER), len(COL_ORDER)))
    ns = []
    for i, r in enumerate(ROW_ORDER):
        sub = s[s.held_out == r].iloc[0]
        n = int(sub.n)
        ns.append(n)
        rt = ast.literal_eval(sub.routing)
        for j, c in enumerate(COL_ORDER):
            M[i, j] = 100.0 * rt.get(c, 0) / n
    return M, ns


def draw(ax, M, ns, cmap, xlabel, frames):
    nr, nc = M.shape
    ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(nc))
    ax.set_xticklabels(COL_ORDER, rotation=45, ha="right",
                       fontsize=TICK_FONT, fontweight="bold")
    ax.set_yticks(range(nr))
    ax.set_yticklabels([f"{r} (n={ns[i]})" for i, r in enumerate(ROW_ORDER)],
                       fontsize=TICK_FONT, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT, labelpad=10)
    ax.set_ylabel("Disease removed from training",
                  fontsize=AXIS_LABEL_FONT, labelpad=10)
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            if v < 0.5:
                continue
            ax.text(j, i, "<1%" if v < 1 else f"{v:.0f}%", ha="center",
                    va="center", fontsize=CELL_FONT, fontweight="bold",
                    color="white" if v > 55 else "black")
    for i, j in frames:
        ax.add_patch(patches.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor="black", linewidth=2.5,
                                       zorder=5))
    ax.set_xticks(np.arange(-.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nr, 1), minor=True)
    ax.grid(which="minor", color="grey", linewidth=.4, alpha=.5)
    ax.tick_params(which="minor", length=0)


fig, (axA, axB) = plt.subplots(2, 1, figsize=(13.8, 19.5))
MA, nA = matrix("honest_lodo_pairseed_nocascade_summary.csv")
draw(axA, MA, nA, "Greens", "Predicted class",
     [(i, 0) for i in range(len(ROW_ORDER))])
MB, nB = matrix("honest_lodo_cascade_with_referees_summary.csv")
fr = [(i, COL_ORDER.index(s)) for i, r in enumerate(ROW_ORDER)
      for s in ((TRIO if r in TRIO else RES4 if r in RES4 else set()) - {r})]
draw(axB, MB, nB, "Blues", "Predicted class", fr)
for ax, lab in ((axA, "A"), (axB, "B")):
    ax.text(-0.17, 1.02, lab, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=PANEL_FONT, fontweight="bold")
fig.tight_layout(h_pad=3)
fig.savefig(OUT / "fig_S9_lodo_matrices.pdf", bbox_inches="tight")
print("Saved paper_figures/fig_S9_lodo_matrices.pdf")
