"""Figure 7 — PairSeed cascade before and after the
additional typicality check (per-protein reference ranges, alpha = 0.05).

20 categories, non-stacked bars (grey = before, blue = after):
  1     known diseases: % classified as "unknown" (inner 5-fold CV)
  12    disease removed from training: % classified as "unknown"
  7     bucketed disease removed from training: % assigned to another
        member of the same (correct) bucket

Input:  paper_figures/open_set_check_predictions.csv
        (analysis/open_set_typicality_check.py)
Run from the repository root:
    python figures/fig_7_open_set_check.py
Output: paper_figures/fig_7_open_set_check.pdf (+ .png preview)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
import fig_style  # noqa: E402

fig_style.apply()
OUT = ROOT / "paper_figures"
ALPHA = 0.05
DIS = ["AML", "MYEL", "CLL", "LYMPH", "GLIOM",
       "CRC", "LUNGC", "OVC", "BRC", "CVX", "ENDC", "PRC"]
TRIO = {"CRC", "LUNGC", "OVC"}
RES4 = {"BRC", "CVX", "ENDC", "PRC"}
BKT = [d for d in DIS if d in TRIO | RES4]
BEFORE, AFTER = "#9a9a9a", fig_style.COLORS["cascade"]


def bucket(c):
    return "trio" if c in TRIO else "res4" if c in RES4 else c


D = pd.read_csv(OUT / "open_set_check_predictions.csv")
D = D[D.arch == "cascade"]


def values(pred):
    k = D.kind == "known"
    v = [100 * (pred[k] == "unknown").mean()]
    for d in DIS:
        m = (D.kind == "heldout") & (D.true == d)
        v.append(100 * (pred[m] == "unknown").mean())
    for d in BKT:
        m = (D.kind == "heldout") & (D.true == d)
        v.append(100 * (pred[m].map(bucket) == bucket(d)).mean())
    return np.array(v)


after = D.pred.where(D.p_protein >= ALPHA, "unknown")
cats = ["known\ndiseases"] + DIS + BKT
gap = 0.7
x = np.arange(len(cats)) + np.array([0] + [1.5 * gap] * len(DIS)
                                    + [2.5 * gap] * len(BKT))
w = 0.34
fig, ax = plt.subplots(figsize=(16, 6))
for k, (pred, col, lab) in enumerate(((D.pred, BEFORE, "before the check"),
                                      (after, AFTER, "after the check"))):
    v = values(pred)
    xx = x + (k - 0.5) * w
    ax.bar(xx, v, w, color=col, edgecolor="black", lw=.5, label=lab)
    for xi, vi in zip(xx, v):
        ax.text(xi, vi + 1, f"{vi:.0f}", ha="center", va="bottom", fontsize=9)
for a, b in ((0, 1), (len(DIS), len(DIS) + 1)):
    ax.axvline((x[a] + x[b]) / 2, color="grey", ls="--", lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=12, fontweight="bold", rotation=45,
                   ha="right")
ax.set_xlim(x[0] - 1.1, x[-1] + .6)
ax.set_ylim(0, 118)
ax.set_ylabel("Patients (%)", fontsize=15)
ax.tick_params(axis="y", labelsize=12)
ax.grid(axis="y", alpha=.3)
ax.set_axisbelow(True)
for xs, txt in ((x[0], "classified\nas\n\"unknown\""),
                (x[1:len(DIS) + 1].mean(), "disease removed from training:\ncorrectly classified as \"unknown\""),
                (x[len(DIS) + 1:].mean(), "disease removed from training:\nassigned to the correct bucket")):
    ax.text(xs, 117, txt, ha="center", va="top", fontsize=16.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2,
          fontsize=18, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_7_open_set_check.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_7_open_set_check.png", dpi=100,
            bbox_inches="tight")
print("Saved paper_figures/fig_7_open_set_check.pdf")
