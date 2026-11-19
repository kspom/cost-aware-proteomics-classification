"""Supplementary Note S1, Figure N4: selection frequency of proteins in the
panel-stability analysis (no-cascade one-vs-rest tests, k = 8; 30 random
stratified 80% subsamples of the 1,273 patients).

One panel per cancer: the six most frequently selected proteins, bar length =
% of the 30 repeats in which the protein was selected. Dark bars: protein is
in the k = 8 panel of Figure 1 (full cohort); light bars: not in that panel.
Dashed line: 50%. Panel title: mean held-out balanced accuracy.

Inputs: paper_figures/pairseed_panel_stability_{runs,frequency}.csv
Output: paper_figures/fig_noteS1_stability.pdf (+ .png preview)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
TOP = 6

runs = pd.read_csv(OUT / "pairseed_panel_stability_runs.csv")
freq = pd.read_csv(OUT / "pairseed_panel_stability_frequency.csv")
ba = runs.groupby("cancer").heldout_bal_acc.mean()

fig, axes = plt.subplots(3, 4, figsize=(17, 11))
for ax, (c, g) in zip(axes.ravel(), freq.groupby("cancer")):
    g = g.sort_values("frequency", ascending=False).head(TOP)[::-1]
    col = ["#2b6cb0" if m else "#a9c4e4" for m in g.in_main_panel]
    ax.barh(g.protein, 100 * g.frequency, color=col)
    ax.axvline(50, color="black", ls="--", lw=1)
    ax.set_xlim(0, 100)
    ax.set_title(f"{c}  (held-out BA {ba[c]:.0f}%)", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("Selected in % of 30 repeats", fontsize=13)
handles = [plt.Rectangle((0, 0), 1, 1, color="#2b6cb0"),
           plt.Rectangle((0, 0), 1, 1, color="#a9c4e4")]
fig.legend(handles, ["in the full-cohort panel (Figure 1)",
                     "not in the full-cohort panel"],
           loc="lower center", ncol=2, fontsize=14, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "fig_noteS1_stability.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_noteS1_stability.png", dpi=70, bbox_inches="tight")
print("Saved", OUT / "fig_noteS1_stability.pdf")
