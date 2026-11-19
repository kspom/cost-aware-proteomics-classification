"""Family of PairSeed cascade classifiers across target_k.

Same architecture as `pairseed_cascade_k8.py` (7-test hierarchical
cascade), but varies the per-test `target_k` cap from 2 up to 14 to
trace out a Pareto curve in the (panel-size, accuracy) plane. Each
family point uses the same panel-fixed 5-fold CV evaluation as the
canonical k = 8 deployable and the other family scripts
(`pairseed_no_cascade_family.py`, `alvez_mimicked_family.py`), so all
points are directly comparable on the Pareto plot.

Procedure for each `target_k` in the grid:
  1. For each of the 7 tests, run PairSeed (seed-pair + greedy
     expansion, plateau stop, max_depth = target_k) on the test's
     training cohort and pick the smallest panel at the global CV
     minimum (smallest-k-at-plateau).
  2. Refit equal-prior LDA per test on the full training cohort.
  3. Outer 5-fold stratified CV with panels fixed; cascade decision
     logic applied in priority order; report strict
     accuracy.

Inputs:
    filtered_wide.csv
    PairSeed_cascades/architecture.json

Outputs (in `PairSeed_cascades/`):
    cascade_family_panels.csv
        One row per (target_k, test_id) with that test's panel found
        on the full cohort.
    cascade_family.csv
        One row per target_k: n_proteins (union across the 7 tests),
        n_deployable_parameters, strict_acc.

Runtime: ~40 min on a workstation (one PairSeed search per test per
target_k, plus a fast 5-fold CV per target_k).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import (  # noqa: E402
    seed_pair_search, greedy_branch_with_intermediates,
    lda_fit, RANDOM_STATE, load_cohort, panel_mask, panel_scores,
)


N_FOLDS = 5
PLATEAU_TOL = 0.0005
TARGET_K_GRID = list(range(2, 15))  # k = 2, 3, ..., 14 (13 values)


# ----------------------------- load architecture ---------------------------

with open("PairSeed_cascades/architecture.json") as f:
    ARCH = json.load(f)

TESTS = {t["test_id"]: t for t in ARCH["tests"]}
TEST_IDS_ORDERED = [t["test_id"] for t in ARCH["tests"]]
print(f"Cascade family: {ARCH['name']}  ({ARCH['n_tests']} tests)")
print(f"target_k grid: {TARGET_K_GRID}\n")


# ----------------------------- load cohort ---------------------------------

# All 1273 patients x 1462 proteins; NaN kept, per-panel complete case.
X_use_df, labels_use, BLOCKS = load_cohort()
X_raw = X_use_df.to_numpy(dtype=np.float64)
print(f"Cohort: {len(X_use_df)} samples × {X_use_df.shape[1]} proteins\n")

ALL_12 = sorted(np.unique(labels_use))


mu_z = X_use_df.mean(axis=0)
sigma_z = X_use_df.std(axis=0, ddof=0).replace(0, 1.0)
Xz = ((X_use_df - mu_z) / sigma_z).to_numpy(dtype=np.float64)
proteins_list = list(X_use_df.columns)
p_index = {p: i for i, p in enumerate(proteins_list)}


# ----------------------------- helpers -------------------------------------

def cohort_mask(spec, labels_arr):
    if spec == "full":
        return np.ones(len(labels_arr), dtype=bool)
    if spec.startswith("only:"):
        keep_set = set(spec[len("only:"):].split(","))
        return np.isin(labels_arr, list(keep_set))
    if spec.startswith("excl:"):
        excl_set = set(spec[len("excl:"):].split(","))
        return ~np.isin(labels_arr, list(excl_set))
    raise ValueError(f"Unknown cohort spec: {spec}")


def search_one_test(test_def, row_idx, target_k):
    """PairSeed search for one binary test on the indicated rows,
    using the supplied max-depth cap. Returns the smallest panel at
    plateau."""
    c_mask = cohort_mask(test_def["training_cohort"], labels_use[row_idx])
    sel = row_idx[c_mask]
    Xt = Xz[sel]
    yt = np.isin(
        labels_use[sel], test_def["positive_classes"]).astype(int)
    if yt.sum() < 2 or (yt == 0).sum() < 2:
        return None
    bt = (BLOCKS[0], BLOCKS[1][sel])
    pairs = seed_pair_search(Xt, yt, proteins_list, top_k=5, blocks=bt)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(np.zeros(len(yt)), yt))
    best_per_depth = {}
    for pi, pj, _ in pairs:
        seed = (p_index[pi], p_index[pj])
        _, bpd = greedy_branch_with_intermediates(
            Xt, yt, seed, splits, max_depth=target_k, blocks=bt)
        for d, (S, e) in bpd.items():
            if d not in best_per_depth or e < best_per_depth[d][1]:
                best_per_depth[d] = (S, e)
    if not best_per_depth:
        return None
    tol = 0.5 / int(yt.sum())
    min_err = min(e for _, e in best_per_depth.values())
    best_d = min(d for d, (_, e) in best_per_depth.items()
                 if e <= min_err + tol)
    S, _ = best_per_depth[best_d]
    return [proteins_list[i] for i in S]


def fit_lda_for_test(panel, test_def, row_idx):
    c_mask = cohort_mask(test_def["training_cohort"], labels_use[row_idx])
    sel = row_idx[c_mask]
    pidx = [p_index[p] for p in panel]
    sel = sel[panel_mask(pidx, BLOCKS)[sel]]       # complete on this panel
    Xt = X_raw[np.ix_(sel, pidx)]
    assert not np.isnan(Xt).any()
    yt = np.isin(
        labels_use[sel], test_def["positive_classes"]).astype(int)
    if yt.sum() < 2 or (yt == 0).sum() < 2:
        return np.zeros(len(panel)), -np.inf
    w, b = lda_fit(Xt, yt)
    return w, b


def cascade_predict(scores):
    if scores["T1_MYEL"] > 0:
        return "MYEL"
    if scores["T3_CLL"] > 0:
        return "LYMPH" if scores["T6_CLLvsLYMPH"] > 0 else "CLL"
    if scores["T2_AML"] > 0:
        return "AML"
    if scores["T4_trio"] > 0:
        return "trio"
    if scores["T5_GLIOM"] > 0:
        return "GLIOM"
    if scores["T7_LYMPH"] > 0:
        return "LYMPH"
    return "residual4"


# ----------------------------- main loop over target_k ---------------------

all_idx = np.arange(len(labels_use))
y_int = pd.Categorical(labels_use, categories=ALL_12).codes

panel_rows = []
family_rows = []

for target_k in TARGET_K_GRID:
    print(f"\n=== target_k = {target_k} ===")
    t0 = time.perf_counter()

    # Step 1 — per-test search on the full cohort
    panels = {}
    for test_id in TEST_IDS_ORDERED:
        test_def = TESTS[test_id]
        panel = search_one_test(test_def, all_idx, target_k=target_k)
        if panel is None:
            raise RuntimeError(f"{test_id} @ k={target_k}: cohort too small")
        panels[test_id] = panel
        panel_rows.append({
            "target_k": target_k,
            "test_id": test_id,
            "training_cohort": test_def["training_cohort"],
            "positive_classes": "|".join(test_def["positive_classes"]),
            "panel_size": len(panel),
            "panel": " + ".join(panel),
        })
        print(f"  {test_id:>15}  k={len(panel)}  "
              f"panel={' + '.join(panel)}")

    union = sorted({p for panel in panels.values() for p in panel})
    n_proteins = len(union)
    n_params = sum(len(p) for p in panels.values()) + len(panels)

    # Step 2 — panel-fixed outer 5-fold CV
    skf_outer = StratifiedKFold(
        N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    preds = np.empty(len(labels_use), dtype=object)
    for tr_idx, te_idx in skf_outer.split(np.zeros(len(y_int)), y_int):
        fold_models = {}
        for test_id, test_def in TESTS.items():
            w, b = fit_lda_for_test(panels[test_id], test_def, tr_idx)
            fold_models[test_id] = (panels[test_id], w, b)
        fs = {t: panel_scores(X_raw[te_idx], [p_index[p] for p in pan], w, b)
              for t, (pan, w, b) in fold_models.items()}
        for k_, i_abs in enumerate(te_idx):
            # NaN score = test not available -> falls through
            preds[i_abs] = cascade_predict({t: v[k_] for t, v in fs.items()})

    strict_correct = sum(preds[i] == labels_use[i]
                         for i in range(len(preds)))
    strict_acc = 100 * strict_correct / len(preds)

    family_rows.append({
        "target_k": target_k,
        "n_proteins": n_proteins,
        "n_deployable_parameters": n_params,
        "strict_acc": strict_acc,
        "n_samples": len(labels_use),
    })
    print(f"  → union={n_proteins} proteins, params={n_params}, "
          f"strict={strict_acc:.1f}%, "
          f"elapsed={time.perf_counter()-t0:.0f}s")


# ----------------------------- save ---------------------------------------

OUT = Path("PairSeed_cascades")
OUT.mkdir(parents=True, exist_ok=True)

pd.DataFrame(panel_rows).to_csv(OUT / "cascade_family_panels.csv", index=False)
print(f"\nSaved {OUT/'cascade_family_panels.csv'}")

pd.DataFrame(family_rows).to_csv(OUT / "cascade_family.csv", index=False)
print(f"Saved {OUT/'cascade_family.csv'}")

print("\n=== Family summary ===")
print(pd.DataFrame(family_rows).round(2).to_string(index=False))
