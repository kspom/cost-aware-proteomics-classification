"""Evaluate the PairSeed-without-cascade family at k_target=2..16.

Cohort: ALL 1273 patients, per-panel complete case (pairseed_lib
`load_cohort` / `blocks`); accuracy over all 1273 patients.

Identical algorithm to `pairseed_v2_family_evaluate.py`, but imports
the two-step search primitives (seed-pair scan and greedy expansion
with branching) from the new central library `pairseed_lib.py`
instead of from `dual_haem_pipeline.py` + `greedy_forward.py`. The
output CSVs and the per-cancer panels are byte-for-byte equivalent
to the v2 run (same MAX_DEPTH=16, same TIE_TOL, PLATEAU_TOL,
RANDOM_STATE).

For each of the 12 cancer classes, runs:
  Step 1 — top-5 seed-pair search       (pairseed_lib.seed_pair_search)
  Step 2 — greedy forward expansion     (pairseed_lib.greedy_branch_with_intermediates)
            with branching, plateau stop, beam width MAX_BRANCHES,
            tracking the best panel at every depth k=2..16.
Then evaluates the 12-class argmax-over-OvR-detector classifier
under 5-fold stratified CV for each k_target.

Outputs:
  paper_figures/pairseed_no_cascade_per_cancer_panels.csv
  paper_figures/pairseed_no_cascade_family.csv
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
    RANDOM_STATE,
)

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
K_GRID = list(range(2, 17))
MAX_DEPTH_FAMILY = 16

# Per-panel complete case: ALL patients, NaN kept (see pairseed_lib)
X_use_df, labels_arr_all, BLOCKS = load_cohort()
labels_use_s = pd.Series(labels_arr_all, index=X_use_df.index)
labels = labels_use_s
proteins_list = list(X_use_df.columns)
p_index = {p: i for i, p in enumerate(proteins_list)}
XN_ALL = zscore_df(X_use_df).to_numpy(dtype=np.float64)
ALL_12 = sorted(np.unique(labels_arr_all))
print(f"Cohort: {len(X_use_df)} samples × {len(proteins_list)} proteins (NaN kept)")
print(f"k_target grid: {K_GRID}")


def search_one_class_all_k(target):
    """OvR PairSeed search for a single cancer. Builds positives = the
    target class, negatives = all other 11 cancers, then runs the
    two-step pipeline imported from pairseed_lib."""
    Xn = XN_ALL
    y = (labels_use_s == target).astype(int).to_numpy()
    pairs = seed_pair_search(Xn, y, proteins_list, top_k=5,
                             blocks=BLOCKS)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(Xn, y))
    combined_best = {}
    completed_all = []
    for pi, pj, _ in pairs:
        seed = (p_index[pi], p_index[pj])
        completed, bpd = greedy_branch_with_intermediates(
            Xn, y, seed, splits, max_depth=MAX_DEPTH_FAMILY,
            blocks=BLOCKS)
        for d, (S, e) in bpd.items():
            if (d not in combined_best) or (e < combined_best[d][1]):
                combined_best[d] = (S, e)
        completed_all.extend(completed)
    for S, err, _, _ in completed_all:
        d = len(S)
        if (d not in combined_best) or (err < combined_best[d][1]):
            combined_best[d] = (S, err)
    panels_at_k = {}
    n_pos_total = int(y.sum())
    for k in K_GRID:
        capped = {d: combined_best[d] for d in combined_best if d <= k}
        if not capped:
            continue
        S, _ = select_parsimony_panel(capped, n_pos=n_pos_total)
        panels_at_k[k] = [proteins_list[i] for i in S]
    return panels_at_k, sorted(combined_best.keys())


print("\nRunning PairSeed search (max_depth=16) for each of 12 cancers...")
t0 = time.perf_counter()
panels_per_cancer = {}
for c in ALL_12:
    pak, depths = search_one_class_all_k(c)
    panels_per_cancer[c] = pak
    print(f"  {c:>6}: depths reached = {depths}")
print(f"All searches done in {time.perf_counter()-t0:.0f}s")

# Save per-cancer panels in long format
rows = []
for c in ALL_12:
    for k in K_GRID:
        rows.append({"cancer": c, "k_target": k,
                     "n_proteins": len(panels_per_cancer[c][k]),
                     "proteins": ",".join(panels_per_cancer[c][k])})
pd.DataFrame(rows).to_csv(
    OUT / "pairseed_no_cascade_per_cancer_panels.csv", index=False)
print(f"\nSaved {OUT/'pairseed_no_cascade_per_cancer_panels.csv'}")
from pairseed_lib import panel_mask  # noqa: E402
print("Patients with complete data per k=8 panel:")
for c in ALL_12:
    idx = [p_index[p] for p in panels_per_cancer[c][8]]
    print(f"  {c:>6}: {int(panel_mask(idx, BLOCKS).sum())} / {len(labels_arr_all)}")


# Multiclass argmax CV evaluation


def repeated_5fold_ovr_argmax(panels_per_cancer_at_k):
    """Strict 12-class accuracy (unavailable tests skipped in argmax).
    Training: per-panel complete case (all 1273). Reported accuracy:
    the 1094 complete patients (BLOCKS[1] == 0); all-1273 kept
    as a second value.
    Returns (acc_complete_1094, acc_all_1273)."""
    strict_all, preds = ovr_argmax_cv(X_use_df.to_numpy(dtype=np.float64),
                                      labels_arr_all, panels_per_cancer_at_k,
                                      p_index, BLOCKS, ALL_12)
    c = BLOCKS[1] == 0
    return 100.0 * np.mean(preds[c] == labels_arr_all[c]), strict_all


print("\nEvaluating multiclass argmax accuracy at each k_target...")
print(f"  {'k':>3} {'n_unique':>9} {'strict':>8}")
results = []
for k in K_GRID:
    panels_at_k = {c: panels_per_cancer[c][k] for c in ALL_12}
    n_unique = len(set().union(*panels_at_k.values()))
    strict, strict_all = repeated_5fold_ovr_argmax(panels_at_k)
    results.append({
        "name": f"PairSeed-no-cascade k={k}",
        "k_target": k, "n_proteins": n_unique,
        "strict_acc": strict, "strict_acc_all_1273": strict_all,
        "n_samples": int((BLOCKS[1] == 0).sum()),
    })
    print(f"  {k:>3} {n_unique:>9} {strict:>7.1f}%")

pd.DataFrame(results).to_csv(OUT / "pairseed_no_cascade_family.csv", index=False)
print(f"\nSaved {OUT/'pairseed_no_cascade_family.csv'}")
