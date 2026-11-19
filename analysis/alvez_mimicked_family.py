"""Alvez-procedure-mimicked family: vary the top-K scale.

Procedure for each scale factor s:
  - Per-cancer CV-tuned C (fixed once for all scales)
  - Sex-restricted negatives + balanced subsample (fixed)
  - Refit elastic-net at best_C, rank proteins by |std coef| (fixed)
  - Take top-K[c] = max(1, round(s * Alvez_K[c])) per cancer
  - Refit plain L2 on each cancer's top-K via 5-fold outer CV
  - Final classifier (as in Alvez 2023): ONE multiclass glmnet
    (multinomial elastic-net, alpha/lambda tuned by inner CV) on the
    union of the per-cancer top-K panels; see alvez_multiclass.py.
    The legacy final step (12 OvR L2 tests + argmax) is kept in the
    column strict_acc_ovr_argmax for reference.

Scales chosen so the family spans ~15 to ~850 proteins:
  s = 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0
The s=1.0 point is the published Alvez K's exactly (~85 proteins).

Output: paper_figures/alvez_mimicked_family.csv
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, str(Path(__file__).parent))
from alvez_multiclass import evaluate_multiclass  # noqa: E402

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
RAND = 42
C_GRID = [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 1.0]

SEX_RESTRICTED = {
    "BRC": "F", "CVX": "F", "ENDC": "F", "OVC": "F", "PRC": "M",
}
ALVEZ_K = {
    "AML": 3, "BRC": 8, "CLL": 6, "CRC": 10, "CVX": 7, "ENDC": 6,
    "GLIOM": 3, "LUNGC": 18, "LYMPH": 4, "MYEL": 3, "OVC": 3, "PRC": 14,
}
SCALES = [0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]

# Load cohort + sex labels
wide = pd.read_csv("filtered_wide.csv", index_col="Sample_ID")
labels = wide["__Cancer__"]
X_full = wide.drop(columns="__Cancer__")
if "XG" in X_full.columns:
    X_full = X_full.drop(columns="XG")
nan_frac = X_full.isna().mean(axis=0)
DENSE = nan_frac[nan_frac <= 0.05].index.tolist()
keep = X_full[DENSE].notna().all(axis=1)
X_use = X_full.loc[keep, DENSE]
y_use = labels[keep].to_numpy()
sample_ids = X_use.index.to_numpy()
sex = pd.read_csv(
    "voted_sex_labels.csv"
).set_index("Sample_ID").loc[sample_ids, "voted_sex"].to_numpy()
print(f"Cohort: {len(X_use)} × {len(DENSE)} dense proteins")

ALL_12 = sorted(np.unique(y_use))
# Deterministic per-cancer RNG offsets. Python's built-in str hash is
# salted per process (PYTHONHASHSEED), so `hash(c) % 9999` would shift
# the balanced subsample (and therefore the elastic-net coefficients)
# between runs. We use a fixed positional offset instead.
CANCER_OFFSET = {c: i for i, c in enumerate(ALL_12)}

Xnp = X_use.to_numpy()


def balanced_subsample(pos_mask, neg_mask, rng):
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]
    n_pos = len(pos_idx)
    if len(neg_idx) > n_pos:
        neg_idx = rng.choice(neg_idx, size=n_pos, replace=False)
    return np.concatenate([pos_idx, neg_idx])


def cohort_for(c):
    pos_mask = (y_use == c)
    if c in SEX_RESTRICTED:
        target_sex = SEX_RESTRICTED[c]
        neg_mask = (y_use != c) & (sex == target_sex)
    else:
        neg_mask = (y_use != c)
    return pos_mask, neg_mask


# Step 1: Per-cancer CV-tune C (run once for all scales)
print("\n=== Step 1: per-cancer CV-tune C (one-time) ===")
best_Cs = {}
for c in ALL_12:
    pos_mask, neg_mask = cohort_for(c)
    rng = np.random.default_rng(RAND)
    sel = balanced_subsample(pos_mask, neg_mask, rng)
    Xb, yb = Xnp[sel], pos_mask[sel].astype(int)
    best_C, best_auc = None, -np.inf
    for C in C_GRID:
        skf = StratifiedKFold(5, shuffle=True, random_state=RAND)
        aucs = []
        for tr, te in skf.split(Xb, yb):
            sc_inner = StandardScaler().fit(Xb[tr])
            try:
                clf = LogisticRegression(
                    penalty="elasticnet", solver="saga",
                    l1_ratio=0.5, C=C, max_iter=5000,
                    random_state=RAND)
                clf.fit(sc_inner.transform(Xb[tr]), yb[tr])
                aucs.append(roc_auc_score(
                    yb[te],
                    clf.decision_function(sc_inner.transform(Xb[te]))))
            except Exception:
                aucs.append(0.5)
        m = float(np.mean(aucs))
        if m > best_auc:
            best_auc, best_C = m, C
    best_Cs[c] = best_C
    print(f"  {c:>6}: best_C={best_C}, AUC={best_auc:.3f}")


# Step 2: per-cancer ranking from full balanced fit at best_C
print("\n=== Step 2: per-cancer importance ranking ===")
rankings = {}
for c in ALL_12:
    pos_mask, neg_mask = cohort_for(c)
    rng = np.random.default_rng(RAND + CANCER_OFFSET[c])
    sel = balanced_subsample(pos_mask, neg_mask, rng)
    Xb, yb = Xnp[sel], pos_mask[sel].astype(int)
    sc_b = StandardScaler().fit(Xb)
    clf = LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=0.5, C=best_Cs[c], max_iter=5000, random_state=RAND)
    clf.fit(sc_b.transform(Xb), yb)
    coefs = clf.coef_[0]
    # Rank ALL proteins by |coef| (zeros get arbitrary low rank)
    ranking = sorted(range(len(DENSE)),
                     key=lambda i: -abs(coefs[i]))
    rankings[c] = ranking
    n_nonzero = int((np.abs(coefs) > 1e-10).sum())
    print(f"  {c:>6}: {n_nonzero} nonzero proteins available")


def evaluate_panels(panels):
    """Outer 5-fold CV multiclass argmax."""
    skf_outer = StratifiedKFold(5, shuffle=True, random_state=RAND)
    y_int = pd.Categorical(y_use, categories=ALL_12).codes
    preds = np.empty(len(y_use), dtype=object)
    for tr_idx, te_idx in skf_outer.split(Xnp, y_int):
        sc_o = StandardScaler().fit(Xnp[tr_idx])
        Xtr_s = sc_o.transform(Xnp[tr_idx])
        Xte_s = sc_o.transform(Xnp[te_idx])
        scores = np.full((len(te_idx), len(ALL_12)), -np.inf)
        for j, c in enumerate(ALL_12):
            panel = panels[c]
            if not panel:
                continue
            col_idx = [DENSE.index(p) for p in panel]
            ytr = (y_use[tr_idx] == c).astype(int)
            if ytr.sum() < 2 or (ytr == 0).sum() < 2:
                continue
            clf = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs",
                max_iter=5000, random_state=RAND,
                class_weight="balanced")
            clf.fit(Xtr_s[:, col_idx], ytr)
            scores[:, j] = clf.decision_function(Xte_s[:, col_idx])
        for i in range(len(te_idx)):
            preds[te_idx[i]] = ALL_12[np.argmax(scores[i])]
    strict = 100 * (preds == y_use).sum() / len(preds)
    return strict


# Step 3: For each scale, take top-K[c] = round(s * ALVEZ_K[c])
print("\n=== Step 3: vary top-K scale ===")
print(f"  {'scale':>5} {'union':>6} {'strict':>7}")
results = []
for s in SCALES:
    panels = {}
    for c in ALL_12:
        K_c = max(1, round(s * ALVEZ_K[c]))
        # Take top-K from ranking, but only include nonzero proteins
        # (else fallback to plain top-K from full ranking).
        top_idx = rankings[c][:K_c]
        panels[c] = [DENSE[i] for i in top_idx]
    union = sorted(set().union(*panels.values()))
    # Alvez final step: ONE multiclass glmnet on the union panel
    mc = evaluate_multiclass(X_use[union].to_numpy(), y_use)
    # legacy final step (12 OvR L2 tests + argmax), kept for reference
    strict_ovr = evaluate_panels(panels)
    print(f"  {s:>5.2f} {len(union):>6} {mc['strict_acc']:>6.1f}%"
          f"  (OvR argmax {strict_ovr:.1f}%, params "
          f"{mc['n_deployable_parameters']})", flush=True)
    results.append({
        "name": f"Alvez-mimicked s={s}",
        "scale": s, "n_proteins": len(union),
        "strict_acc": mc["strict_acc"],
        "n_proteins_used": mc["n_proteins_used"],
        "n_deployable_parameters": mc["n_deployable_parameters"],
        "C": mc["C"], "l1_ratio": mc["l1_ratio"],
        "strict_acc_ovr_argmax": strict_ovr,
        "per_cancer_K": ",".join(
            f"{c}={max(1, round(s * ALVEZ_K[c]))}"
            for c in ALL_12),
    })

pd.DataFrame(results).to_csv(
    OUT / "alvez_mimicked_family.csv", index=False)
print(f"\nSaved {OUT/'alvez_mimicked_family.csv'}")
