"""Cohort: ALL 1273 patients, per-panel complete case.

Extend the PairSeed-no-cascade family to k = 18, 20, 22, 24
(vs the default k = 2..16 from pairseed_no_cascade_family.py). Same
two-step search (seed-pair + greedy expansion with branching,
plateau stop), only the panel-size cap is larger.

PairSeed has a natural plateau: each cancer's greedy search stops
when adding more proteins doesn't reduce CV error (PLATEAU_TOL).
Cancers with strong distinctive signatures (MYEL, CLL, AML) plateau
very early; cancers with diffuse signal (LUNGC, CRC, BRC) keep
growing. The realistic ceiling is ~200-250 unique proteins.

Rows for k = 18, 20, 22, 24 are appended to the existing CSVs
written by pairseed_no_cascade_family.py.
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
from pairseed_lib import (
    seed_pair_search, greedy_branch_with_intermediates,
    select_parsimony_panel, lda_fit, load_cohort,
    RANDOM_STATE,
)

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
MAX_DEPTH_EXTEND = 24
NEW_K_GRID = [18, 20, 22, 24]

# Per-panel complete case: ALL patients, NaN kept (see pairseed_lib)
X_use_df, labels_arr_all, BLOCKS = load_cohort()
labels_use_s = pd.Series(labels_arr_all, index=X_use_df.index)
labels = labels_use_s
proteins_list = list(X_use_df.columns)
p_index = {p: i for i, p in enumerate(proteins_list)}
XN_ALL = zscore_df(X_use_df).to_numpy(dtype=np.float64)
print(f"Cohort: {len(X_use_df)} × {len(proteins_list)} proteins (NaN kept)")
ALL_12 = sorted(np.unique(labels_arr_all))


def search_one_class_all_k(target):
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
            Xn, y, seed, splits, max_depth=MAX_DEPTH_EXTEND,
            blocks=BLOCKS)
        for d, (S, e) in bpd.items():
            if (d not in combined_best) or (e < combined_best[d][1]):
                combined_best[d] = (S, e)
        completed_all.extend(completed)
    for S, err, _, _ in completed_all:
        d = len(S)
        if (d not in combined_best) or (err < combined_best[d][1]):
            combined_best[d] = (S, err)
    panels = {}
    n_pos_total = int(y.sum())
    for k in NEW_K_GRID:
        capped = {d: combined_best[d] for d in combined_best if d <= k}
        if not capped:
            continue
        S, _ = select_parsimony_panel(capped, n_pos=n_pos_total)
        panels[k] = [proteins_list[i] for i in S]
    return panels


print(f"\nSearching with MAX_DEPTH={MAX_DEPTH_EXTEND} for each cancer...")
t0 = time.perf_counter()
panels_per_cancer = {}
for c in ALL_12:
    panels_per_cancer[c] = search_one_class_all_k(c)
    print(f"  {c}: k=24 panel size {len(panels_per_cancer[c][24])}")
print(f"All searches done in {time.perf_counter()-t0:.0f}s")

# Save per-cancer panels (append to existing)
existing_panels = pd.read_csv(
    OUT / "pairseed_no_cascade_per_cancer_panels.csv")
new_rows = []
for c in ALL_12:
    for k in NEW_K_GRID:
        new_rows.append({"cancer": c, "k_target": k,
                         "n_proteins": len(panels_per_cancer[c][k]),
                         "proteins": ",".join(panels_per_cancer[c][k])})
combined_panels = pd.concat(
    [existing_panels, pd.DataFrame(new_rows)], ignore_index=True)
combined_panels.to_csv(
    OUT / "pairseed_no_cascade_per_cancer_panels.csv", index=False)


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


print(f"\n  {'k':>3} {'n_unique':>9} {'strict':>8}")
new_results = []
for k in NEW_K_GRID:
    panels_at_k = {c: panels_per_cancer[c][k] for c in ALL_12}
    n_unique = len(set().union(*panels_at_k.values()))
    strict, strict_all = repeated_5fold_ovr_argmax(panels_at_k)
    print(f"  {k:>3} {n_unique:>9} {strict:>7.1f}%")
    new_results.append({
        "name": f"PairSeed-no-cascade k={k}", "k_target": k,
        "n_proteins": n_unique,
        "strict_acc": strict, "strict_acc_all_1273": strict_all,
        "n_samples": int((BLOCKS[1] == 0).sum()),
    })

existing_fam = pd.read_csv(OUT / "pairseed_no_cascade_family.csv")
combined_fam = pd.concat([existing_fam, pd.DataFrame(new_results)],
                         ignore_index=True)
combined_fam.to_csv(OUT / "pairseed_no_cascade_family.csv", index=False)
print(f"\nUpdated {OUT/'pairseed_no_cascade_family.csv'}")
