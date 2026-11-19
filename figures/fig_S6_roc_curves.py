"""Supplementary Figure S6 — Per-cancer ROC curves at k = 8.

For each of the 12 cancers, the OvR PairSeed detector at k = 8 is
scored under 5-fold stratified CV (held-out scores), and the ROC
curve is drawn with its AUC. Layout: 4 rows × 3 columns, cancers in
the same order as Supplementary Figure S5, square axes, each cancer in
its own colour. The Youden operating point is marked by a black ×.

Inputs:
    paper_figures/pairseed_no_cascade_per_cancer_panels.csv  (from analysis/pairseed_no_cascade_family.py;
        falls back to paper_figures/pairseed_per_cancer_panels_v2.csv
        when the new file is absent)
    filtered_wide.csv                                        (from explore_missingness.py)

Output:
    paper_figures/fig_S6_roc_curves.pdf
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import lda_fit, load_cohort, panel_mask, panel_scores  # noqa: E402
import fig_style  # noqa: E402

fig_style.apply()

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

TITLE_FONT = 15
AXIS_LABEL_FONT = 14
TICK_FONT = 12
AUC_LABEL_FONT = 13

# Subplot order + display name + colour. Matches Alvez 2023 Fig. 5b.
ALV_ORDER = [
    ("AML",   "AML",         "#9ec9e0"),
    ("CLL",   "CLL",         "#1f6fb4"),
    ("LYMPH", "DLBCL",       "#1f5e54"),
    ("MYEL",  "Myeloma",     "#4cc5a3"),
    ("LUNGC", "Lung",        "#a8c43d"),
    ("CRC",   "Colorectal",  "#a08864"),
    ("GLIOM", "Glioma",      "#e9c836"),
    ("PRC",   "Prostate",    "#e0843f"),
    ("BRC",   "Breast",      "#e9aaaa"),
    ("CVX",   "Cervical",    "#a91828"),
    ("ENDC",  "Endometrial", "#c79bd0"),
    ("OVC",   "Ovarian",     "#5a2d6a"),
]


# -------------------------- load panels ---------------------------------

def load_panels_at_k8():
    new = OUT / "pairseed_no_cascade_per_cancer_panels.csv"
    legacy = OUT / "pairseed_per_cancer_panels_v2.csv"
    if new.exists():
        print(f"  loaded {new}")
        df = pd.read_csv(new)
    elif legacy.exists():
        print(f"  loaded {legacy}  (legacy fallback)")
        df = pd.read_csv(legacy)
    else:
        raise FileNotFoundError(
            "Neither pairseed_no_cascade_per_cancer_panels.csv nor "
            "pairseed_per_cancer_panels_v2.csv is in paper_figures/")
    df = df[df["k_target"] == 8]
    return {r["cancer"]: r["proteins"].split(",") for _, r in df.iterrows()}


print("Loading panels...")
panels = load_panels_at_k8()


# -------------------------- load cohort ---------------------------------

# ALL patients, per-panel complete case (NaN kept)
X_use, y_use, BLOCKS = load_cohort()
print(f"Cohort: {len(X_use)} samples (per-panel complete case)")


# -------------------------- compute held-out scores --------------------

ALL_12_INTERNAL = sorted(set(t[0] for t in ALV_ORDER))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
y_int = pd.Categorical(y_use, categories=ALL_12_INTERNAL).codes

print("Computing held-out CV scores per cancer...")
held_out = {}
for code, _, _ in ALV_ORDER:
    panel = panels[code]
    y_bin = (y_use == code).astype(int)
    idx = [X_use.columns.get_loc(p) for p in panel]
    m = panel_mask(idx, BLOCKS)
    XNP = X_use.to_numpy(dtype=float)
    scores = np.full(len(y_use), np.nan)
    for tr_idx, te_idx in skf.split(XNP, y_int):
        tr = tr_idx[m[tr_idx]]
        ytr = y_bin[tr]
        if ytr.sum() < 2 or (ytr == 0).sum() < 2:
            continue
        try:
            w, b = lda_fit(XNP[tr][:, idx], ytr)
            scores[te_idx] = panel_scores(XNP[te_idx], idx, w, b)
        except np.linalg.LinAlgError:
            pass
    # evaluated on the 1094 complete patients
    ok = (BLOCKS[1] == 0) & ~np.isnan(scores)
    auc = roc_auc_score(y_bin[ok], scores[ok])
    fpr, tpr, _ = roc_curve(y_bin[ok], scores[ok])
    youden = tpr - fpr
    idx_best = int(np.argmax(youden))
    held_out[code] = (fpr, tpr, auc, fpr[idx_best], tpr[idx_best])
    print(f"  {code:>6}: AUC = {auc:.3f}")


# -------------------------- plot 4×3 grid -----------------------------

ORDER = sorted(ALV_ORDER, key=lambda t: t[0])        # as in Figure S5
NROW, NCOL = 4, 3
fig, axes = plt.subplots(NROW, NCOL, figsize=(9.5, 12.5),
                         sharey=True, sharex=True)
for i, (code, _, colour) in enumerate(ORDER):
    ax = axes.flat[i]
    fpr, tpr, auc, fpr_y, tpr_y = held_out[code]

    ax.plot([0, 1], [0, 1], "k--", alpha=0.55, linewidth=1.0)
    ax.plot(fpr, tpr, color=colour, linewidth=2.0)
    ax.plot(fpr_y, tpr_y, "x", color="black", markersize=10,
            markeredgewidth=2.0)

    ax.text(0.95, 0.05, f"AUC: {auc:.2f}",
            ha="right", va="bottom",
            fontsize=AUC_LABEL_FONT,
            transform=ax.transAxes)

    ax.set_title(code, fontsize=TITLE_FONT, fontweight="bold",
                 color=colour, pad=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    if i % NCOL == 0:
        ax.set_ylabel("Sensitivity", fontsize=AXIS_LABEL_FONT)
    if i // NCOL == NROW - 1:
        ax.set_xlabel("1 − Specificity", fontsize=AXIS_LABEL_FONT)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

plt.tight_layout(pad=0.6)
out_pdf = OUT / "fig_S6_roc_curves.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_pdf}")
