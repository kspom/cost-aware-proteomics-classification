"""Figure 2 (two panels) — strict 12-class accuracy of classifier families
(A) against the number of unique proteins and (B) against the number of
deployable parameters (log scale).

Families (same colours/markers in both panels):
  - PairSeed-no-cascade family              (green +, k = 2..24)
  - Alvez-like family                       (orange diamonds, scale s)
  - Alvez classifier reproduction           (purple star, 83 proteins)
  - PairSeedCascade-with-referees family    (circles, k = 2..14)

Deployable parameters = nonzero weights of the deployed formulas plus one
intercept per formula (PairSeed: sum of panel sizes + one per test;
Alvez-like / Alvez: nonzero glmnet coefficients of the 12 class formulas
+ 12). Alvez-like points are not connected in panel B because the
elastic-net hyperparameters are re-tuned for every member.

Run from the repository root:
    python figures/fig_2_accuracy_proteins_parameters.py [side|stack]
Output:
    paper_figures/fig_2_accuracy_proteins_parameters.pdf        (side)
    paper_figures/fig_2_accuracy_proteins_parameters_stack.pdf  (stack)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
import fig_style  # noqa: E402

fig_style.apply()
OUT = ROOT / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

LAYOUT = sys.argv[1] if len(sys.argv) > 1 else "side"

# Font sizes are in points at the final printed size (figure width ~ page
# width, so no further downscaling is expected).
if LAYOUT == "side":
    FIGSIZE = (14, 6.6)
    AXIS_LABEL_FONT, TICK_FONT, LEGEND_FONT = 17, 14, 14
    ZONE_LABEL_FONT, PANEL_FONT, MARKER_SCALE = 11, 22, 0.9
else:
    FIGSIZE = (13, 16.5)
    AXIS_LABEL_FONT, TICK_FONT, LEGEND_FONT = 28, 22, 20
    ZONE_LABEL_FONT, PANEL_FONT, MARKER_SCALE = 16, 34, 1.5

YLIM = (40, 95)
C, M = fig_style.COLORS, fig_style.MARKERS

ZONE_LABEL_COLORS = {
    "screening": "#a03030", "clinical": "#2f6a2f",
    "research": "#5a3a73", "high_protein": "#404060",
}

# ---- data ---------------------------------------------------------------
nc = pd.read_csv(OUT / "pairseed_no_cascade_family.csv")
nc_pan = pd.read_csv(OUT / "pairseed_no_cascade_per_cancer_panels.csv")
nc_par = (nc_pan.groupby("k_target")["n_proteins"].sum()
          + nc_pan.groupby("k_target").size()).rename("params")
nc = nc.merge(nc_par, left_on="k_target", right_index=True)

cas = pd.read_csv(ROOT / "PairSeed_cascades" /
                  "cascade_with_referees_family.csv")
alv = pd.read_csv(OUT / "alvez_mimicked_family.csv")
star = pd.read_csv(OUT / "alvez_83_protein_classifier.csv")
star = star[star["name"].str.contains("multiclass glmnet")].iloc[0]


def draw(ax, x_nc, x_alv, x_star, x_cas, connect_alvez):
    ms = MARKER_SCALE
    ax.plot(nc[x_nc], nc["strict_acc"], "-", color=C["pairseed_no_cascade"],
            alpha=0.35, linewidth=1.3, zorder=2)
    ax.scatter(nc[x_nc], nc["strict_acc"], c=C["pairseed_no_cascade"],
               marker=M["pairseed_no_cascade"][0],
               s=M["pairseed_no_cascade"][1] * ms, linewidths=2.0,
               label="PairSeed-no-cascade family", zorder=4)
    if connect_alvez:
        ax.plot(alv[x_alv], alv["strict_acc"], "-", color=C["alvez_mimicked"],
                alpha=0.35, linewidth=1.3, zorder=2)
    ax.scatter(alv[x_alv], alv["strict_acc"], c=C["alvez_mimicked"],
               marker=M["alvez_mimicked"][0], s=M["alvez_mimicked"][1] * ms,
               edgecolor="black", linewidths=0.6,
               label="Alvez-like family", zorder=5)
    ax.scatter([star[x_star]], [star["strict_acc"]], c=C["alvez_star"],
               marker=M["alvez_star"][0], s=M["alvez_star"][1] * ms,
               edgecolor="black", linewidths=0.7,
               label="Alvez classifier reproduction (83 proteins)", zorder=6)
    ax.plot(cas[x_cas], cas["strict_acc"], color=C["cascade"],
            alpha=0.35, linewidth=1.3, zorder=3)
    ax.scatter(cas[x_cas], cas["strict_acc"], c=C["cascade"], marker="o",
               s=M["alvez_star"][1] * ms * 0.7, edgecolor="black",
               linewidths=0.6, label="PairSeedCascade-with-referees family",
               zorder=7)
    ax.set_ylim(*YLIM)
    ax.set_ylabel("Strict 12-class accuracy (%)", fontsize=AXIS_LABEL_FONT)
    ax.tick_params(axis="both", labelsize=TICK_FONT)


# ---- figure -------------------------------------------------------------
if LAYOUT == "side":
    fig, (axA, axB) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
else:
    fig, (axA, axB) = plt.subplots(2, 1, figsize=FIGSIZE)

# A: proteins
axA.set_xlim(0, 350)
fig_style.add_pareto_zones(axA, ymin=YLIM[0], ymax=YLIM[1], x_axis_max=350,
                           show_labels=False)
draw(axA, "n_proteins", "n_proteins", "n_proteins", "n_proteins_unique",
     connect_alvez=True)
axA.set_xlabel("Total unique proteins in classifier", fontsize=AXIS_LABEL_FONT)
axA.grid(True, alpha=0.25, zorder=1)
ZONE_TEXT = {
    "screening":    ("screening (<20 p)",              10, 93, 90),
    "clinical":     ("clinical sweet spot\n(20–50 p)", 35, 93, 90),
    "research":     ("research-grade\n(50–110 p)",     80, 94, 0),
    "high_protein": ("high-protein regime\n(>110 p)", 230, 94, 0),
}
for name, (text, x, y, rot) in ZONE_TEXT.items():
    axA.text(x, y, text, ha="center", va="top", fontsize=ZONE_LABEL_FONT,
             color=ZONE_LABEL_COLORS[name], fontweight="bold", rotation=rot)

# B: parameters
draw(axB, "params", "n_deployable_parameters", "n_deployable_parameters",
     "n_deployable_parameters", connect_alvez=False)
axB.set_xscale("log")
axB.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000])
axB.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axB.set_xlim(20, 5000)
axB.set_xlabel("Number of parameters in classifier (log scale)",
               fontsize=AXIS_LABEL_FONT)
axB.grid(True, which="both", alpha=0.25, zorder=1)
if LAYOUT == "side":
    axB.set_ylabel("")
    axB.tick_params(axis="y", labelleft=False)

for ax, lab in ((axA, "A"), (axB, "B")):
    ax.text(-0.02, 1.02, lab, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=PANEL_FONT, fontweight="bold")

# One shared legend below both panels
handles, labels = axA.get_legend_handles_labels()
fig.tight_layout()
if LAYOUT == "side":
    fig.subplots_adjust(bottom=0.26, wspace=0.10)
    fig.legend(handles, labels, loc="lower center", ncols=2,
               fontsize=LEGEND_FONT, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    name = "fig_2_accuracy_proteins_parameters.pdf"
else:
    fig.subplots_adjust(bottom=0.12, hspace=0.25)
    fig.legend(handles, labels, loc="lower center", ncols=2,
               fontsize=LEGEND_FONT, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    name = "fig_2_accuracy_proteins_parameters_stack.pdf"
fig.savefig(OUT / name, bbox_inches="tight")
print(f"Saved paper_figures/{name}")
