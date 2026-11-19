"""Cohort: ALL 1273 patients, per-panel complete case (each subset
is evaluated on the patients complete for it).

Backward elimination on each of the 12 PairSeed-OvR k=8 panels.

For every cancer's k=8 panel (from `pairseed_ovr_k8_panels.csv`):
  - start with the full k-protein panel
  - at each step, evaluate 5-fold CV balanced accuracy after removing
    each remaining protein in turn, drop the one whose removal yields
    the **highest** balanced accuracy (equivalently: smallest balanced
    error)
  - record the trajectory from k proteins down to 1

The resulting curve answers: for each OvR test, how compressible is
the panel?  A flat plateau over the right-hand side of the curve =
"elbow" (small subset already captures most of the signal); a steep
monotone climb = every protein contributes.

Splits: 5-fold StratifiedKFold(random_state=42), same protocol as the
PairSeed search itself, so the trajectory is comparable to the
PairSeed objective.

Output:
    paper_figures/backward_elimination_ovr_k8.csv
        columns: cancer, n_proteins, removed_at_this_step,
                 retained_panel, bal_acc, bal_err
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
from pairseed_lib import cv_balanced_error, load_cohort, RANDOM_STATE  # noqa: E402

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)


# -------------------------- cohort ------------------------------------

# ALL patients, per-panel complete case (NaN kept)
X_use_df, labels_use, BLOCKS = load_cohort()
print(f"Cohort: {len(X_use_df)} samples x {X_use_df.shape[1]} proteins (NaN kept)")

# z-score (same convention as PairSeed search)
X_z = ((X_use_df - X_use_df.mean(axis=0)) /
       X_use_df.std(axis=0, ddof=0).replace(0, 1.0))
proteins_list = list(X_z.columns)
p_index = {p: i for i, p in enumerate(proteins_list)}


# -------------------------- panels ------------------------------------

panels_df = pd.read_csv(OUT / "pairseed_ovr_k8_panels.csv")
panels = {r["cancer"]: r["proteins"].split(",")
          for _, r in panels_df.iterrows()}
ALL_12 = sorted(panels.keys())


# -------------------------- backward elimination ----------------------

def backward_eliminate(target, panel):
    """Greedy removal: at each step drop the protein whose absence
    minimises CV balanced error.  Returns list of trajectory rows."""
    y = (labels_use == target).astype(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X_z.values, y))
    # train: per-panel complete case; evaluate: 1094 complete patients
    splits = [(tr, te[BLOCKS[1][te] == 0]) for tr, te in splits]
    panel_idx = [p_index[p] for p in panel]

    # Full panel
    err_full = cv_balanced_error(X_z.values, y, panel_idx, splits, BLOCKS)
    rows = [{
        "cancer": target,
        "n_proteins": len(panel),
        "removed_at_this_step": "",
        "retained_panel": ",".join(panel),
        "bal_acc": 100.0 * (1 - err_full),
        "bal_err": err_full,
    }]

    current = list(panel_idx)
    current_names = list(panel)
    while len(current) > 1:
        best_err = np.inf
        best_drop = None
        for i in range(len(current)):
            S = current[:i] + current[i+1:]
            err = cv_balanced_error(X_z.values, y, S, splits, BLOCKS)
            if err < best_err:
                best_err = err
                best_drop = i
        dropped = current_names[best_drop]
        current = current[:best_drop] + current[best_drop+1:]
        current_names = (current_names[:best_drop]
                         + current_names[best_drop+1:])
        rows.append({
            "cancer": target,
            "n_proteins": len(current),
            "removed_at_this_step": dropped,
            "retained_panel": ",".join(current_names),
            "bal_acc": 100.0 * (1 - best_err),
            "bal_err": best_err,
        })
    return rows


# -------------------------- run for all 12 ----------------------------

t0 = time.perf_counter()
all_rows = []
print(f"\n{'cancer':>6}  {'k':>2}  {'bal_acc':>8}  removed")
for c in ALL_12:
    rows = backward_eliminate(c, panels[c])
    all_rows.extend(rows)
    for r in rows:
        rm = r["removed_at_this_step"] or "(full)"
        print(f"  {c:>6}  {r['n_proteins']:>2}  "
              f"{r['bal_acc']:>7.2f}%  {rm}")
print(f"\nElapsed: {time.perf_counter()-t0:.1f}s")

df = pd.DataFrame(all_rows)
out_csv = OUT / "backward_elimination_ovr_k8.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")
