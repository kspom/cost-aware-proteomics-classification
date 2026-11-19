"""Figure 4 — Cascade per-test boxplots.

7-panel grid arranged as **3 rows × 3 columns** (with one unused
cell), one subplot per cascade test. Each subplot shows the LDA
score of that cascade test across all 12 disease groups, colored
according to its role in the test:
    * positive class(es)               -> blue   (target of the test)
    * negative class(es) in the
      training cohort                  -> red    (what the test is
                                                  trained to push down)
    * cancers outside the training
      cohort                           -> light grey (never seen
                                                      during fit;
                                                      routed in by
                                                      the cascade)
The subplot title gives the test_id, its positive class(es),
training cohort, and the actual list of proteins.

The k = 8 cap of the cascade architecture means each individual
test can recruit up to 8 proteins. The actual panel size found by
PairSeed is shown in the title (it may be smaller than 8 if
plateau was reached earlier).

Input:
    paper_figures/pairseed_cascade_k8_panels.csv     (from analysis/pairseed_cascade_k8.py)
    PairSeed_cascades/architecture.json              (input)
    filtered_wide.csv                                (from explore_missingness.py)

Output:
    paper_figures/fig_4_cascade_test_boxplots.pdf
"""

import json
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

# Visual layout
N_ROWS, N_COLS = 3, 3            # 9 cells, 7 cascade tests + 2 for legend
SUBPLOT_W, SUBPLOT_H = 5.5, 3.6
FIG_W = N_COLS * SUBPLOT_W
FIG_H = N_ROWS * SUBPLOT_H

TITLE_FONT = 18
TICK_FONT = 16
YLABEL_FONT = 20

TARGET_COLOR = fig_style.COLORS["cascade"]      # positive class
NEG_TRAIN_COLOR = "#e63946"                     # negatives the test
                                                # actually discriminates
OUT_OF_COHORT_COLOR = "#cfd5dd"                 # routed in but not
                                                # part of this test's fit


# -------------------------- load architecture ----------------------------

with open("PairSeed_cascades/architecture.json") as f:
    ARCH = json.load(f)
TESTS_ORDER = [t["test_id"] for t in ARCH["tests"]]
TEST_DEFS = {t["test_id"]: t for t in ARCH["tests"]}


# -------------------------- load panels ---------------------------------

panels_df = pd.read_csv(OUT / "pairseed_cascade_k8_panels.csv")
panels = {r["test_id"]: r["panel"].split(" + ")
          for _, r in panels_df.iterrows()}


# -------------------------- load cohort ---------------------------------

# All 1273 patients, NaN kept: each test is fitted on, and plotted for,
# the patients with complete data on its own panel.
X_use, labels_use, BLOCKS = load_cohort()
COMPLETE = BLOCKS[1] == 0   # boxplots show the 1094 complete patients
X_raw = X_use.to_numpy(dtype=float)
P_INDEX = {p: i for i, p in enumerate(X_use.columns)}
print(f"Cohort: {len(X_use)} samples (per-panel complete case)")

ALL_12 = sorted(np.unique(labels_use))


def cohort_mask(spec, labels_arr):
    if spec == "full":
        return np.ones(len(labels_arr), dtype=bool)
    if spec.startswith("only:"):
        keep_set = set(spec[len("only:"):].split(","))
        return np.isin(labels_arr, list(keep_set))
    if spec.startswith("excl:"):
        excl_set = set(spec[len("excl:"):].split(","))
        return ~np.isin(labels_arr, list(excl_set))
    raise ValueError(spec)


# -------------------------- fit each test and score ---------------------

score = {}
for tid in TESTS_ORDER:
    test_def = TEST_DEFS[tid]
    panel = panels[tid]
    pidx = [P_INDEX[p] for p in panel]
    mask = cohort_mask(test_def["training_cohort"], labels_use) \
        & panel_mask(pidx, BLOCKS)
    Xtr = X_raw[np.ix_(mask.nonzero()[0], pidx)]
    assert not np.isnan(Xtr).any()
    ytr = np.isin(
        labels_use[mask], test_def["positive_classes"]).astype(int)
    w, b = lda_fit(Xtr, ytr)
    score[tid] = panel_scores(X_raw, pidx, w, b)
    print(f"  {tid}: plotted {int((~np.isnan(score[tid])).sum())} patients")


# -------------------------- plot 3×3 grid -------------------------------

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_W, FIG_H))
axes = axes.flatten()

# Concise per-test title — accurate w.r.t. training cohort:
#   T1, T2, T3, T4 train on the full cohort     -> "vs all"
#   T5 trains excluding haem cancers            -> "vs non-haem cancers"
#   T6 trains only on CLL+LYMPH                 -> "vs CLL"
#   T7 trains excluding haem + GLIOM + OVC + PRC -> "vs other epithelial cancers"
TEST_TITLE = {
    "T1_MYEL":       "T1: MYEL vs all",
    "T2_AML":        "T2: AML vs all",
    "T3_CLL":        "T3: CLL vs all",
    "T4_trio":       "T4: trio (CRC+LUNGC+OVC)\nvs all",
    "T5_GLIOM":      "T5: GLIOM\nvs non-haem cancers",
    "T6_CLLvsLYMPH": "T6: LYMPH vs CLL",
    "T7_LYMPH":      "T7: LYMPH\nvs other epithelial cancers",
}

for j, tid in enumerate(TESTS_ORDER):
    ax = axes[j]
    test_def = TEST_DEFS[tid]
    positives = set(test_def["positive_classes"])

    # Which disease groups are inside this test's training cohort?
    cohort_diseases = set(
        ALL_12[k] for k, d in enumerate(ALL_12)
        if cohort_mask(test_def["training_cohort"],
                       np.array([d]))[0]
    )

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
        elif d in cohort_diseases:
            patch.set_facecolor(NEG_TRAIN_COLOR)
            patch.set_alpha(0.75)
        else:
            patch.set_facecolor(OUT_OF_COHORT_COLOR)
            patch.set_alpha(0.55)

    ax.axhline(0, color=fig_style.COLORS["grid"],
               linewidth=0.8, linestyle="--", zorder=0)

    ax.set_title(f"{TEST_TITLE[tid]}, n={len(panels[tid])}",
                 fontsize=TITLE_FONT,
                 fontweight="bold", pad=6)

    ax.tick_params(axis="x", labelsize=TICK_FONT)
    plt.setp(ax.get_xticklabels(), rotation=90,
             ha="center", va="top")
    ax.tick_params(axis="y", labelsize=TICK_FONT)
    ax.grid(True, axis="y", alpha=0.25)

    if j % N_COLS == 0:
        ax.set_ylabel("LDA score", fontsize=YLABEL_FONT)


# Hide the unused 8th and 9th cells (cells 7 and 8 in 0-indexed flatten).
for idx in range(len(TESTS_ORDER), N_ROWS * N_COLS):
    axes[idx].set_visible(False)

# Use the bottom-middle + bottom-right empty area for a compact
# "protein panels" legend. fig.text in figure-relative coords spans
# both empty cells naturally.
PANEL_LIST_FONT = 16

panel_lines = ["Protein panels:"]
for tid in TESTS_ORDER:
    tnum = tid.split("_")[0]  # "T1", "T2", ...
    panel_lines.append(f"  {tnum}: {' + '.join(panels[tid])}")

fig.text(0.40, 0.31, "\n".join(panel_lines),
         ha="left", va="top",
         fontsize=PANEL_LIST_FONT, family="monospace",
         linespacing=1.4,
         bbox=dict(facecolor="#f7f7f7",
                   edgecolor="#999999",
                   boxstyle="round,pad=0.7"))

plt.tight_layout()
out_pdf = OUT / "fig_4_cascade_test_boxplots.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved {out_pdf}")
