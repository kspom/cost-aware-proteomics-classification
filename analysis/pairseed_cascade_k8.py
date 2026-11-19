"""PairSeedCascade at k ≤ 8 — canonical hierarchical cascade classifier.

Self-contained deployable. Reads only the cascade *architecture*
(`PairSeed_cascades/architecture.json` — positive classes, training-
cohort restriction, target panel size per test) and derives the
protein panels and LDA weights from `filtered_wide.csv` at run time
by running PairSeed search for each of the 7 binary tests.

Pipeline:

  1. Load architecture.json (7 tests, target_k = 8 each).
  2. For each test, run PairSeed (seed-pair + greedy expansion,
     plateau stop) on the test's training cohort, pick the smallest
     panel at the global CV-error minimum (smallest-k-at-plateau).
  3. Refit equal-prior LDA per test on its full training cohort
     (deployable formulas).
  4. Outer 5-fold stratified CV with the panels fixed (LDA refit per
     fold). Same panel-fixed CV protocol as `pairseed_ovr_k8.py` and
     `alvez_mimicked_family.py`, so the three classifiers' Pareto
     points are directly comparable.

For a strictly honest nested-CV variant (per-fold re-search; ~50 min),
see `pairseed_cascade_k8_CV5.py`.

Decision logic (priority order):

    T1_MYEL    > 0 → MYEL
    T3_CLL     > 0 → if T6_CLLvsLYMPH > 0: LYMPH else CLL
    T2_AML     > 0 → AML
    T4_trio    > 0 → trio  (CRC, LUNGC, OVC)
    T5_GLIOM   > 0 → GLIOM
    T7_LYMPH   > 0 → LYMPH
    else            → residual4  (BRC, CVX, ENDC, PRC)

Inputs:
    filtered_wide.csv
    PairSeed_cascades/architecture.json

Outputs:
    paper_figures/pairseed_cascade_k8_panels.csv
    paper_figures/pairseed_cascade_k8_formulas.csv
    paper_figures/pairseed_cascade_k8_accuracy.csv
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


# ----------------------------- load architecture ---------------------------

with open("PairSeed_cascades/architecture.json") as f:
    ARCH = json.load(f)

TESTS = {t["test_id"]: t for t in ARCH["tests"]}
print(f"Cascade: {ARCH['name']}  ({ARCH['n_tests']} tests)\n")


# ----------------------------- load cohort ---------------------------------

# All 1273 patients x 1462 proteins; missing values kept as NaN and
# handled per panel (complete case per panel, see pairseed_lib).
X_use_df, labels_use, BLOCKS = load_cohort()
X_raw = X_use_df.to_numpy(dtype=np.float64)
print(f"Cohort: {len(X_use_df)} samples × {X_use_df.shape[1]} proteins "
      f"({int((BLOCKS[1] > 0).sum())} with one missing block)\n")

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


def search_one_test(test_def, row_idx):
    """PairSeed search for one binary test on the indicated rows.
    Returns the best panel (smallest k at plateau)."""
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
            Xt, yt, seed, splits, max_depth=test_def["target_k"], blocks=bt)
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


def n_used(panel, test_def, row_idx):
    c_mask = cohort_mask(test_def["training_cohort"], labels_use[row_idx])
    sel = row_idx[c_mask]
    return int(panel_mask([p_index[p] for p in panel], BLOCKS)[sel].sum()), len(sel)


def cascade_predict(scores):
    # NaN score (panel protein missing) compares False: the test is
    # "not fired" and the patient falls through to the next test.
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


# ----------------------------- (1) full-cohort panel search ----------------

all_idx = np.arange(len(labels_use))
panels = {}
formulas = {}

print("=== Step 1: PairSeed search per test on the full cohort ===")
t0 = time.perf_counter()
for test_id in [t["test_id"] for t in ARCH["tests"]]:
    test_def = TESTS[test_id]
    panel = search_one_test(test_def, all_idx)
    if panel is None:
        raise RuntimeError(f"{test_id}: cohort too small")
    panels[test_id] = panel
    w, b = fit_lda_for_test(panel, test_def, all_idx)
    formulas[test_id] = {"w": w, "b": b}
    nu, nc = n_used(panel, test_def, all_idx)
    print(f"  {test_id:>15}  k={len(panel)}  patients {nu}/{nc}  "
          f"cohort={test_def['training_cohort']}  "
          f"panel={' + '.join(panel)}")
print(f"  search done in {time.perf_counter()-t0:.0f}s")

union_proteins = sorted({p for panel in panels.values() for p in panel})
n_proteins = len(union_proteins)
n_params = sum(len(p) for p in panels.values()) + len(panels)
print(f"\n  unique proteins across 7 tests : {n_proteins}")
print(f"  deployable parameters          : {n_params}")


# ----------------------------- (2) panel-fixed outer 5-fold CV -------------

print("\n=== Step 2: panel-fixed outer 5-fold CV (LDA refit per fold) ===")
y_int = pd.Categorical(labels_use, categories=ALL_12).codes
skf_outer = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
preds = np.empty(len(labels_use), dtype=object)

for fold_i, (tr_idx, te_idx) in enumerate(skf_outer.split(np.zeros(len(y_int)), y_int)):
    fold_models = {}
    for test_id, test_def in TESTS.items():
        # Panel is fixed (from Step 1). Only LDA weights are refit on
        # the training fold.
        panel = panels[test_id]
        w, b = fit_lda_for_test(panel, test_def, tr_idx)
        fold_models[test_id] = (panel, w, b)

    fold_scores = {t: panel_scores(X_raw[te_idx], [p_index[p] for p in pan], w, b)
                   for t, (pan, w, b) in fold_models.items()}
    for k, i_abs in enumerate(te_idx):
        preds[i_abs] = cascade_predict(
            {t: v[k] for t, v in fold_scores.items()})
    print(f"  fold {fold_i+1}/{N_FOLDS} done")


strict_correct = int(sum(preds[i] == labels_use[i]
                         for i in range(len(preds))))
strict_acc = 100 * strict_correct / len(preds)

print(f"\n=== PairSeedCascade (k ≤ 8, panel-fixed CV) ===")
print(f"  Strict 12-class accuracy : {strict_acc:5.1f}%")


# ----------------------------- (3) save -----------------------------------

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

panel_rows = [{
    "test_id": test_id,
    "training_cohort": TESTS[test_id]["training_cohort"],
    "positive_classes": "|".join(TESTS[test_id]["positive_classes"]),
    "panel_size": len(panels[test_id]),
    "panel": " + ".join(panels[test_id]),
} for test_id in [t["test_id"] for t in ARCH["tests"]]]
pd.DataFrame(panel_rows).to_csv(
    OUT / "pairseed_cascade_k8_panels.csv", index=False)
print(f"\nSaved {OUT/'pairseed_cascade_k8_panels.csv'}")

formula_rows = []
for test_id in [t["test_id"] for t in ARCH["tests"]]:
    f = formulas[test_id]
    row = {
        "test_id": test_id,
        "training_cohort": TESTS[test_id]["training_cohort"],
        "positive_classes": "|".join(TESTS[test_id]["positive_classes"]),
        "panel_size": len(panels[test_id]),
        "intercept": float(f["b"]),
    }
    for p in union_proteins:
        row[p] = 0.0
    for p, weight in zip(panels[test_id], f["w"]):
        row[p] = float(weight)
    formula_rows.append(row)
pd.DataFrame(formula_rows).to_csv(
    OUT / "pairseed_cascade_k8_formulas.csv", index=False)
print(f"Saved {OUT/'pairseed_cascade_k8_formulas.csv'}")

pd.DataFrame([{
    "classifier": "PairSeedCascade k<=8",
    "n_proteins_unique": n_proteins,
    "n_deployable_parameters": n_params,
    "strict_acc": strict_acc,
    "n_samples": len(labels_use),
}]).to_csv(OUT / "pairseed_cascade_k8_accuracy.csv", index=False)
print(f"Saved {OUT/'pairseed_cascade_k8_accuracy.csv'}")
