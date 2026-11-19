"""Boxplots for the 7 within-bucket OvR referee tests
(T41, T42, T43 = trio referees; T71, T72, T73, T74 = residual4
referees). Same conventions as Figure 4 (cascade per-test boxplots):

    * positive class                → blue   (target of the referee)
    * negative classes within the
      bucket                        → red    (trained-against)
    * cancers outside the bucket    → light grey (never seen
                                                  during fit)

Each referee is a binary LDA test trained ONLY on its bucket
cohort (no out-of-bucket negatives), at k = 8.

Layout: 3 rows × 3 cols. Row 1 = trio referees (T41, T42, T43).
Row 2 = residual4 referees (T71, T72, T73). Row 3 = T74 plus a
protein-panel legend in the bottom-right empty cells.

Input:
    paper_figures/bucket_referee_panels.csv  (from
        analysis/pairseed_bucket_referees.py)
    filtered_wide.csv

Output:
    paper_figures/fig_bucket_referees_boxplots.pdf
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
from pairseed_lib import load_cohort, panel_scores  # noqa: E402

fig_style.apply()

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Per-figure fonts (match Figure 4)
TITLE_FONT = 18
TICK_FONT = 16
YLABEL_FONT = 20
PANEL_LIST_FONT = 14

TARGET_COLOR = fig_style.COLORS["cascade"]   # blue
NEG_TRAIN_COLOR = "#e63946"                  # red (within-bucket neg)
OUT_OF_COHORT_COLOR = "#cfd5dd"              # grey (out of bucket)

TRIO = {"CRC", "LUNGC", "OVC"}
RES4 = {"BRC", "CVX", "ENDC", "PRC"}


# -------------------------- load test definitions ----------------------

tests_df = pd.read_csv(OUT / "bucket_referee_panels.csv")
TESTS_ORDER = [
    "T41_CRC", "T42_LUNGC", "T43_OVC",      # row 1 (trio)
    "T71_BRC", "T72_CVX", "T73_ENDC",       # row 2 (residual4)
    "T74_PRC",                              # row 3 (residual4 cont.)
]


# -------------------------- load cohort + score every sample ----------

# All 1273 patients, NaN kept: each test is fitted on, and plotted for,
# the patients with complete data on its own panel.
X_use, labels_use, BLOCKS = load_cohort()
COMPLETE = BLOCKS[1] == 0   # boxplots show the 1094 complete patients
X_raw = X_use.to_numpy(dtype=float)
P_INDEX = {p: i for i, p in enumerate(X_use.columns)}
print(f"Cohort: {len(X_use)} samples")

ALL_12 = sorted(np.unique(labels_use))

# Compute score per sample for each referee
score = {}
panel_str_short = {}
panel_str_full = {}
positives_set = {}
bucket_of = {}
for _, r in tests_df.iterrows():
    panel = r["panel"].split(",")
    w = np.array([float(x) for x in r["weights"].split(",")])
    b = float(r["intercept"])
    score[r["test_id"]] = panel_scores(X_raw, [P_INDEX[p] for p in panel], w, b)
    panel_str_short[r["test_id"]] = " + ".join(panel)
    panel_str_full[r["test_id"]] = " + ".join(panel)
    positives_set[r["test_id"]] = set(r["positives"].split(","))
    bucket_of[r["test_id"]] = r["bucket"]


# -------------------------- titles -----------------------------------

TEST_TITLE = {
    "T41_CRC":   "T41: CRC vs {LUNGC, OVC}",
    "T42_LUNGC": "T42: LUNGC vs {CRC, OVC}",
    "T43_OVC":   "T43: OVC vs {CRC, LUNGC}",
    "T71_BRC":   "T71: BRC vs {CVX, ENDC, PRC}",
    "T72_CVX":   "T72: CVX vs {BRC, ENDC, PRC}",
    "T73_ENDC":  "T73: ENDC vs {BRC, CVX, PRC}",
    "T74_PRC":   "T74: PRC vs {BRC, CVX, ENDC}",
}


# -------------------------- plot 3 × 3 grid ---------------------------

N_ROWS, N_COLS = 3, 3
SUBPLOT_W, SUBPLOT_H = 5.5, 3.6
fig, axes = plt.subplots(N_ROWS, N_COLS,
                         figsize=(N_COLS * SUBPLOT_W, N_ROWS * SUBPLOT_H))
axes = axes.flatten()

for j, tid in enumerate(TESTS_ORDER):
    ax = axes[j]
    positives = positives_set[tid]
    bucket = bucket_of[tid]
    bucket_members = TRIO if bucket == "trio" else RES4

    by_disease = [v[~np.isnan(v)] for v in
                  (score[tid][(labels_use == d) & COMPLETE] for d in ALL_12)]
    bp = ax.boxplot(
        by_disease, labels=ALL_12, showmeans=False,
        patch_artist=True, widths=0.65,
        medianprops=dict(color="black", linewidth=1.4),
        flierprops=dict(marker=".", markersize=2.5,
                        markerfacecolor="grey", markeredgecolor="grey"),
    )
    for k, patch in enumerate(bp["boxes"]):
        d = ALL_12[k]
        if d in positives:
            patch.set_facecolor(TARGET_COLOR)
            patch.set_alpha(0.85)
        elif d in bucket_members:
            patch.set_facecolor(NEG_TRAIN_COLOR)
            patch.set_alpha(0.75)
        else:
            patch.set_facecolor(OUT_OF_COHORT_COLOR)
            patch.set_alpha(0.55)

    ax.axhline(0, color=fig_style.COLORS["grid"],
               linewidth=0.8, linestyle="--", zorder=0)

    ax.set_title(TEST_TITLE[tid], fontsize=TITLE_FONT,
                 fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=TICK_FONT)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top")
    ax.tick_params(axis="y", labelsize=TICK_FONT)
    ax.grid(True, axis="y", alpha=0.25)
    if j % N_COLS == 0:
        ax.set_ylabel("LDA score", fontsize=YLABEL_FONT)

# Hide unused cells
for idx in range(len(TESTS_ORDER), N_ROWS * N_COLS):
    axes[idx].set_visible(False)

# Protein-panel legend in bottom-right empty cells
panel_lines = ["Protein panels:"]
for tid in TESTS_ORDER:
    short = tid.split("_")[0]
    panel_lines.append(f"  {short}: {panel_str_short[tid]}")

fig.text(0.40, 0.31, "\n".join(panel_lines),
         ha="left", va="top",
         fontsize=PANEL_LIST_FONT, family="monospace",
         linespacing=1.4,
         bbox=dict(facecolor="#f7f7f7",
                   edgecolor="#999999",
                   boxstyle="round,pad=0.7"))

plt.tight_layout()
out_pdf = OUT / "fig_bucket_referees_boxplots.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved {out_pdf}")
