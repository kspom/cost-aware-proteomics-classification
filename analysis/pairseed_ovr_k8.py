"""PairSeed one-vs-rest classifier at k = 8.

Cohort: ALL 1273 patients, per-panel complete case. Panels are
identical to pairseed_no_cascade_family.py at k_target = 8 (same
search; depth cap 8 vs 16 does not change the depth-<=8 panels).

This is the **single-operating-point** version of PairSeed-no-cascade:
it builds 12 parallel one-vs-rest LDA detectors, each with a panel
of up to 8 proteins, and combines them by argmax of their LDA scores.
This is the basic classifier presented first in the manuscript's
Results section, before any sweep over panel size or comparison
against the cascade.

Algorithm (for each of the 12 cancer classes C):
  Step 1 — `seed_pair_search`              (pairseed_lib)
              exhaustive 2-protein LDA scan over all P·(P−1)/2 pairs;
              positives = samples with cancer C, negatives = the
              other 11 cancers.
  Step 2 — `greedy_branch_with_intermediates` (pairseed_lib)
              greedy-forward expansion from each seed pair, branching
              on ties and stopping on plateau, capped at panel size 8.
              Returns the best-CV panel for each depth k = 2..8.

We take the k = 8 panel from each cancer's search, then evaluate the
combined 12-OvR argmax classifier under 5-fold stratified CV and
report strict 12-class accuracy.

Outputs:
  paper_figures/pairseed_ovr_k8_panels.csv      (one row per cancer)
  paper_figures/pairseed_ovr_k8_accuracy.csv    (single-row summary)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from pairseed_nan_eval import zscore_df, ovr_argmax_cv  # noqa: E402
from pairseed_lib import (  # noqa: E402
    seed_pair_search,
    greedy_branch_with_intermediates,
    select_parsimony_panel,
    lda_fit,
    load_cohort,
    panel_mask,
    RANDOM_STATE,
)

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
K_TARGET = 8

# Cohort: ALL patients, per-panel complete case (NaN kept).
X_use_df, labels_arr_all, BLOCKS = load_cohort()
labels_use_s = pd.Series(labels_arr_all, index=X_use_df.index)
ALL_12 = sorted(np.unique(labels_arr_all))
proteins_list = list(X_use_df.columns)
p_index = {p: i for i, p in enumerate(proteins_list)}
XN_ALL = zscore_df(X_use_df).to_numpy(dtype=np.float64)
print(f"Cohort: {len(X_use_df)} samples × {len(proteins_list)} proteins (NaN kept)")
print(f"Building 12 one-vs-rest PairSeed panels at k = {K_TARGET}\n")


def search_one_class_k8(target):
    """PairSeed search at max_depth=8 for a single cancer's OvR
    detector. Returns the best CV panel found (panel size ≤ 8)."""
    Xn = XN_ALL
    y = (labels_use_s == target).astype(int).to_numpy()

    pairs = seed_pair_search(Xn, y, proteins_list, top_k=5,
                             blocks=BLOCKS)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(Xn, y))

    # Run greedy expansion from every seed; track best-CV panel at
    # each depth, then apply the parsimony rule (smallest k whose CV
    # error is within PLATEAU_TOL of the minimum).
    combined_best = {}
    for pi, pj, _ in pairs:
        seed = (p_index[pi], p_index[pj])
        completed, bpd = greedy_branch_with_intermediates(
            Xn, y, seed, splits, max_depth=K_TARGET, blocks=BLOCKS)
        for d, (S, e) in bpd.items():
            if (d not in combined_best) or (e < combined_best[d][1]):
                combined_best[d] = (S, e)
        for S, err, _, _ in completed:
            d = len(S)
            if (d not in combined_best) or (err < combined_best[d][1]):
                combined_best[d] = (S, err)

    combined_best = {d: combined_best[d] for d in combined_best
                     if d <= K_TARGET}
    S, cv_err = select_parsimony_panel(combined_best, n_pos=int(y.sum()))
    panel = [proteins_list[i] for i in S]
    return panel, float(cv_err), len(S)


# Per-cancer search
t0 = time.perf_counter()
panels = {}
print(f"  {'cancer':>6}  {'k':>3}  {'CV bal.err':>11}  panel")
for c in ALL_12:
    panel, cv_err, k_found = search_one_class_k8(c)
    panels[c] = panel
    n_used = int(panel_mask([p_index[p] for p in panel], BLOCKS).sum())
    print(f"  {c:>6}  {k_found:>3}  {cv_err:>10.4f}   "
          f"{' + '.join(panel)}   (patients {n_used}/{len(labels_arr_all)})")
print(f"\nAll 12 OvR searches done in {time.perf_counter()-t0:.0f}s")

# Save per-cancer panels
panel_rows = [{
    "cancer": c,
    "k_target": K_TARGET,
    "n_proteins": len(panels[c]),
    "proteins": ",".join(panels[c]),
} for c in ALL_12]
pd.DataFrame(panel_rows).to_csv(
    OUT / "pairseed_ovr_k8_panels.csv", index=False)
print(f"\nSaved {OUT/'pairseed_ovr_k8_panels.csv'}")


# 5-fold stratified CV evaluation of the 12-OvR argmax classifier
# over ALL patients (unavailable tests skipped in the argmax)
strict_all, preds = ovr_argmax_cv(X_use_df.to_numpy(dtype=np.float64),
                              labels_arr_all, panels, p_index, BLOCKS,
                              ALL_12)
# reported accuracy: 1094 complete patients
_c = BLOCKS[1] == 0
strict = 100.0 * np.mean(preds[_c] == labels_arr_all[_c])
n_unique = len(set().union(*panels.values()))

print(f"\n=== PairSeed-OvR-k8 single-point classifier ===")
print(f"  Unique proteins across 12 panels : {n_unique}")
print(f"  Strict 12-class accuracy         : {strict:5.1f}%")

pd.DataFrame([{
    "k_target": K_TARGET,
    "n_proteins_unique": n_unique,
    "strict_acc": strict, "strict_acc_all_1273": strict_all,
    "n_samples": int(_c.sum()),
}]).to_csv(OUT / "pairseed_ovr_k8_accuracy.csv", index=False)
print(f"\nSaved {OUT/'pairseed_ovr_k8_accuracy.csv'}")
