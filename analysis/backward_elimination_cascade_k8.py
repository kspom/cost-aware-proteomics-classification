"""Backward elimination for the 14-test cascade-with-referees architecture.

For every cascade test (7 main + 7 within-bucket referees):
  - start with the test's k=8 panel (some main tests stopped early
    on plateau, e.g. T1 ended at k=4)
  - at each step, evaluate 5-fold CV balanced accuracy of the OvR
    LDA fit after removing each remaining protein, drop the one
    whose removal yields the highest balanced accuracy
  - record the trajectory from k proteins down to 1

Each test is evaluated on its own training cohort and OvR target
(positive classes vs the rest of the cohort), matching the protocol
used by PairSeed to search the panel in the first place. Referees
are scored only within their bucket cohort (trio = CRC/LUNGC/OVC;
residual4 = BRC/CVX/ENDC/PRC).

Splits: StratifiedKFold(5, shuffle=True, random_state=42).

Output:
    paper_figures/backward_elimination_cascade_k8.csv
        columns: test_id, group (main|referee), n_proteins,
                 removed_at_this_step, retained_panel, bal_acc, bal_err
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import cv_balanced_error, RANDOM_STATE, load_cohort  # noqa: E402

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
ARCH_PATH = Path("PairSeed_cascades") / "architecture_with_referees.json"

MAIN_TESTS = {"T1_MYEL", "T2_AML", "T3_CLL", "T4_trio",
              "T5_GLIOM", "T6_CLLvsLYMPH", "T7_LYMPH"}


# -------------------------- cohort ------------------------------------

# All 1273 patients, NaN kept; every subset S is scored on the
# patients complete for S (per-panel complete case).
X_use_df, y_use, BLOCKS = load_cohort()
X_z = ((X_use_df - X_use_df.mean(axis=0)) /
       X_use_df.std(axis=0, ddof=0).replace(0, 1.0))
proteins_list = list(X_z.columns)
p_index = {p: i for i, p in enumerate(proteins_list)}
X_z_arr = X_z.values
print(f"Cohort: {len(X_use_df)} samples × {X_use_df.shape[1]} proteins")


# -------------------------- training cohort mask ----------------------

def cohort_mask(spec):
    if spec == "full":
        return np.ones(len(y_use), dtype=bool)
    if spec.startswith("excl:"):
        excl = set(spec[len("excl:"):].split(","))
        return ~np.isin(y_use, list(excl))
    if spec.startswith("only:"):
        keep_set = set(spec[len("only:"):].split(","))
        return np.isin(y_use, list(keep_set))
    raise ValueError(spec)


# -------------------------- load architecture & panels ---------------

with open(ARCH_PATH) as f:
    ARCH = json.load(f)

cas_panels = pd.read_csv(OUT / "pairseed_cascade_k8_panels.csv")
ref_panels = pd.read_csv(OUT / "bucket_referee_panels.csv")

panel_for = {}
for _, r in cas_panels.iterrows():
    panel_for[r["test_id"]] = r["panel"].split(" + ")
for _, r in ref_panels.iterrows():
    panel_for[r["test_id"]] = r["panel"].split(",")


# -------------------------- backward elimination ----------------------

def backward_eliminate(test_id):
    spec = next(t for t in ARCH["tests"] if t["test_id"] == test_id)
    positives = set(spec["positive_classes"])
    cohort = cohort_mask(spec["training_cohort"])
    panel = panel_for[test_id]
    panel_idx = [p_index[p] for p in panel]

    yt = np.isin(y_use[cohort], list(positives)).astype(int)
    Xc = X_z_arr[cohort]
    bt = (BLOCKS[0], BLOCKS[1][cohort])

    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(np.zeros(len(yt)), yt))
    # train: per-panel complete case; evaluate: 1094 complete patients
    splits = [(tr, te[bt[1][te] == 0]) for tr, te in splits]

    err_full = cv_balanced_error(Xc, yt, panel_idx, splits, bt)
    rows = [{
        "test_id": test_id,
        "group": "main" if test_id in MAIN_TESTS else "referee",
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
            err = cv_balanced_error(Xc, yt, S, splits, bt)
            if err < best_err:
                best_err = err
                best_drop = i
        dropped = current_names[best_drop]
        current = current[:best_drop] + current[best_drop+1:]
        current_names = (current_names[:best_drop]
                         + current_names[best_drop+1:])
        rows.append({
            "test_id": test_id,
            "group": "main" if test_id in MAIN_TESTS else "referee",
            "n_proteins": len(current),
            "removed_at_this_step": dropped,
            "retained_panel": ",".join(current_names),
            "bal_acc": 100.0 * (1 - best_err),
            "bal_err": best_err,
        })
    return rows


# -------------------------- run --------------------------------------

t0 = time.perf_counter()
all_rows = []
print(f"\n{'test_id':>14}  group    {'k':>2}  {'bal_acc':>8}  removed")
for td in ARCH["tests"]:
    rows = backward_eliminate(td["test_id"])
    all_rows.extend(rows)
    grp = "main   " if td["test_id"] in MAIN_TESTS else "referee"
    for r in rows:
        rm = r["removed_at_this_step"] or "(full)"
        print(f"  {td['test_id']:>14}  {grp}  {r['n_proteins']:>2}  "
              f"{r['bal_acc']:>7.2f}%  {rm}")
print(f"\nElapsed: {time.perf_counter()-t0:.1f}s")

pd.DataFrame(all_rows).to_csv(
    OUT / "backward_elimination_cascade_k8.csv", index=False)
print(f"\nSaved {OUT/'backward_elimination_cascade_k8.csv'}")
