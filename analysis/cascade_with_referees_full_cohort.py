"""Apply the cascade-WITH-referees architecture to the full cohort
(no LODO held-out) and compute strict 12-class accuracy. Uses
already-trained T1-T7 panels (from analysis/pairseed_cascade_k8.py)
and the within-bucket referees T41-T43, T71-T74 (from
analysis/pairseed_bucket_referees.py).

Decision logic (cascade-with-referees):
    T1_MYEL   > 0  -> MYEL
    T3_CLL    > 0  -> if T6 > 0: LYMPH else CLL
    T2_AML    > 0  -> AML
    T4_trio   > 0  -> argmax of {T41, T42, T43}
    T5_GLIOM  > 0  -> GLIOM
    T7_LYMPH  > 0  -> LYMPH
    else            -> argmax of {T71, T72, T73, T74}

5-fold stratified CV (panel-fixed): panels selected once on the
full cohort; LDA weights refit per fold. Matches the protocol of
the no-cascade family for fair comparison.

Output:
    paper_figures/cascade_with_referees_full_cohort.csv
        one-row summary: n_proteins_unique, n_deployable_parameters,
        strict_acc.
    paper_figures/cascade_with_referees_predictions.csv
        per-sample predictions (Sample_ID, true_label, predicted).
"""
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import lda_fit, load_cohort, panel_mask, panel_scores  # noqa: E402

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
ARCH_PATH = Path("PairSeed_cascades") / "architecture_with_referees.json"
RANDOM_STATE = 42


# -------------------------- load cohort --------------------------------

# All 1273 patients, NaN kept; each test uses the patients complete on
# its own panel (fit) and is "not available" (NaN score) otherwise.
X_use_df, y_use, BLOCKS = load_cohort()
X_raw = X_use_df.to_numpy(dtype=float)
P_INDEX = {p: i for i, p in enumerate(X_use_df.columns)}
print(f"Cohort: {len(X_use_df)} samples")


# -------------------------- load test definitions ----------------------

with open(ARCH_PATH) as f:
    ARCH = json.load(f)
TEST_DEFS = ARCH["tests"]


def cohort_mask_for(spec):
    """Training cohort mask (same convention as honest_lodo_cascade.py)."""
    if spec == "full":
        return np.ones(len(y_use), dtype=bool)
    if spec.startswith("excl:"):
        excl = set(spec[len("excl:"):].split(","))
        return ~np.isin(y_use, list(excl))
    if spec.startswith("only:"):
        keep_set = set(spec[len("only:"):].split(","))
        return np.isin(y_use, list(keep_set))
    raise ValueError(spec)


# -------------------------- load panels --------------------------------

cas_panels = pd.read_csv(OUT / "pairseed_cascade_k8_panels.csv")
ref_panels = pd.read_csv(OUT / "bucket_referee_panels.csv")

panel_for = {}
for _, r in cas_panels.iterrows():
    panel_for[r["test_id"]] = r["panel"].split(" + ")
for _, r in ref_panels.iterrows():
    panel_for[r["test_id"]] = r["panel"].split(",")


# -------------------------- score every sample under panel-fixed CV ----

skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
y_strat = pd.Categorical(y_use, categories=sorted(set(y_use))).codes

scores = {tid: np.full(len(y_use), np.nan) for tid in
          [td["test_id"] for td in TEST_DEFS]}

for td in TEST_DEFS:
    pidx = [P_INDEX[p] for p in panel_for[td["test_id"]]]
    m = panel_mask(pidx, BLOCKS) & cohort_mask_for(td["training_cohort"])
    print(f"  {td['test_id']:<15} patients complete on panel: "
          f"{int(m.sum())}/{int(cohort_mask_for(td['training_cohort']).sum())}")

for tr_idx, te_idx in skf.split(np.zeros(len(y_strat)), y_strat):
    for td in TEST_DEFS:
        tid = td["test_id"]
        panel = panel_for[tid]
        pidx = [P_INDEX[p] for p in panel]
        positives = set(td["positive_classes"])
        cohort = cohort_mask_for(td["training_cohort"]) & panel_mask(pidx, BLOCKS)
        train_keep = cohort[tr_idx]
        if train_keep.sum() < 4:
            continue
        Xtr = X_raw[np.ix_(tr_idx[train_keep], pidx)]
        assert not np.isnan(Xtr).any()
        ytr = np.isin(y_use[tr_idx[train_keep]],
                      list(positives)).astype(int)
        if ytr.sum() < 2 or (ytr == 0).sum() < 2:
            continue
        try:
            w, b = lda_fit(Xtr, ytr)
        except np.linalg.LinAlgError:
            continue
        scores[tid][te_idx] = panel_scores(X_raw[te_idx], pidx, w, b)


# -------------------------- cascade decision per sample ----------------

def argmax_available(refs):
    """Referee argmax over available (non-NaN) scores; none -> unknown."""
    refs = [r for r in refs if not np.isnan(r[1])]
    return max(refs, key=lambda x: x[1])[0] if refs else "unknown"


def predict_one(i):
    # NaN score = test not available -> comparison False -> falls through
    if scores["T1_MYEL"][i] > 0:
        return "MYEL"
    if scores["T3_CLL"][i] > 0:
        return "LYMPH" if scores["T6_CLLvsLYMPH"][i] > 0 else "CLL"
    if scores["T2_AML"][i] > 0:
        return "AML"
    if scores["T4_trio"][i] > 0:
        refs = [("CRC", scores["T41_CRC"][i]),
                ("LUNGC", scores["T42_LUNGC"][i]),
                ("OVC", scores["T43_OVC"][i])]
        return argmax_available(refs)
    if scores["T5_GLIOM"][i] > 0:
        return "GLIOM"
    if scores["T7_LYMPH"][i] > 0:
        return "LYMPH"
    refs = [("BRC", scores["T71_BRC"][i]),
            ("CVX", scores["T72_CVX"][i]),
            ("ENDC", scores["T73_ENDC"][i]),
            ("PRC", scores["T74_PRC"][i])]
    return argmax_available(refs)


preds = np.array([predict_one(i) for i in range(len(y_use))])


# -------------------------- accuracy ----------------------------------

# reported accuracy: 1094 complete patients; all-1273 kept
_c = BLOCKS[1] == 0
strict_correct = (preds[_c] == y_use[_c]).mean()
strict_all = (preds == y_use).mean()


# -------------------------- count proteins + params --------------------

all_panels = []
for td in TEST_DEFS:
    all_panels.extend(panel_for[td["test_id"]])
unique_proteins = sorted(set(all_panels))
n_proteins = len(unique_proteins)

# Deployable params = sum of panel sizes + one intercept per test
n_params = sum(len(panel_for[td["test_id"]]) for td in TEST_DEFS) \
           + len(TEST_DEFS)


print(f"\n=== Cascade-with-referees, 5-fold CV (panel-fixed) ===")
print(f"  Unique proteins across 14 tests : {n_proteins}")
print(f"  Deployable parameters           : {n_params}")
print(f"  Strict 12-class accuracy        : {100*strict_correct:.1f}%")

pd.DataFrame([{
    "name": "PairSeedCascade-with-referees",
    "n_tests": len(TEST_DEFS),
    "n_proteins_unique": n_proteins,
    "n_deployable_parameters": n_params,
    "strict_acc": round(100 * strict_correct, 2),
    "strict_acc_all_1273": round(100 * strict_all, 2),
    "n_samples": int(_c.sum()),
}]).to_csv(OUT / "cascade_with_referees_full_cohort.csv", index=False)
print(f"  unknown (no referee available): {(preds == 'unknown').sum()}")

pd.DataFrame({
    "Sample_ID": X_use_df.index,
    "true_label": y_use,
    "predicted": preds,
}).to_csv(OUT / "cascade_with_referees_predictions.csv", index=False)
print(f"\nSaved {OUT/'cascade_with_referees_full_cohort.csv'}")
print(f"Saved {OUT/'cascade_with_referees_predictions.csv'}")
