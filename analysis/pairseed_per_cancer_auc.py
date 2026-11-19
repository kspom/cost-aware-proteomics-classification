"""Cohort: ALL 1273 patients for training (per-panel complete case);
AUC over the 1094 complete patients (n_samples).

Compute per-cancer one-vs-rest AUC for PairSeed v2 k=8 panels,
analogous to Alvez 2023's published per-cancer AUC table.
Writes the result as a CSV (Table S1).

For each of 12 cancers:
  1. Take that cancer's PairSeed v2 k=8 panel (from
     pairseed_no_cascade_per_cancer_panels.csv, or
     pairseed_per_cancer_panels_v2.csv as a legacy fallback).
  2. 5-fold stratified CV: fit binary LDA on training fold (target =
     this cancer; rest = all 11 others); decision_function on held-out
     fold.
  3. Compute ROC AUC across pooled held-out scores.

Comparison side-by-side with Alvez's published per-cancer AUCs from
the original paper.

Output:
    paper_figures/pairseed_v2_k8_per_cancer_auc.csv   (Table S1)
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import lda_fit, load_cohort, panel_mask, panel_scores  # noqa: E402

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Alvez's published per-cancer AUC numbers
# (from main text + Fig. 5; same we used to validate our re-implementation)
ALVEZ_AUC = {
    "AML": 1.00, "BRC": 0.86, "CLL": 1.00, "CRC": 0.91,
    "CVX": 0.95, "ENDC": 0.92, "GLIOM": 1.00, "LUNGC": 0.82,
    "LYMPH": 0.93, "MYEL": 0.97, "OVC": 0.93, "PRC": 0.92,
}

new_panels = OUT / "pairseed_no_cascade_per_cancer_panels.csv"
legacy_panels = OUT / "pairseed_per_cancer_panels_v2.csv"
if new_panels.exists():
    per_cancer = pd.read_csv(new_panels)
    print(f"  loaded {new_panels}")
elif legacy_panels.exists():
    per_cancer = pd.read_csv(legacy_panels)
    print(f"  loaded {legacy_panels}  (legacy fallback)")
else:
    raise FileNotFoundError(
        "Neither pairseed_no_cascade_per_cancer_panels.csv nor "
        "pairseed_per_cancer_panels_v2.csv is in paper_figures/")
per_cancer = per_cancer[per_cancer["k_target"] == 8]
panels = {r["cancer"]: r["proteins"].split(",")
          for _, r in per_cancer.iterrows()}
ALL_12 = sorted(panels.keys())

# ALL patients, per-panel complete case (NaN kept)
X_use, y_use, BLOCKS = load_cohort()
print(f"Cohort: {len(X_use)} samples (per-panel complete case)")
Xnp = X_use.to_numpy(dtype=float)
p_index = {p: i for i, p in enumerate(X_use.columns)}

skf = StratifiedKFold(5, shuffle=True, random_state=42)
y_int = pd.Categorical(y_use, categories=ALL_12).codes

results = []
print(f"\n{'cancer':>6} {'panel':>50} {'PairSeed AUC':>13} "
      f"{'Alvez AUC':>12} {'Delta':>7}")
for c in ALL_12:
    panel = panels[c]
    y_binary = (y_use == c).astype(int)
    idx = [p_index[p] for p in panel]
    m = panel_mask(idx, BLOCKS)
    held_out_scores = np.full(len(y_use), np.nan)
    for tr_idx, te_idx in skf.split(Xnp, y_int):
        tr = tr_idx[m[tr_idx]]
        ytr = y_binary[tr]
        if ytr.sum() < 2 or (ytr == 0).sum() < 2:
            continue
        try:
            w, b = lda_fit(Xnp[tr][:, idx], ytr)
            held_out_scores[te_idx] = panel_scores(Xnp[te_idx], idx, w, b)
        except np.linalg.LinAlgError:
            pass
    # evaluated on the 1094 complete patients
    ok = (BLOCKS[1] == 0) & ~np.isnan(held_out_scores)
    auc = roc_auc_score(y_binary[ok], held_out_scores[ok])
    alv = ALVEZ_AUC[c]
    delta = auc - alv
    panel_str = "+".join(panel)
    if len(panel_str) > 50:
        panel_str = panel_str[:47] + "..."
    print(f"{c:>6} {panel_str:>50} {auc:>12.3f}  "
          f"{alv:>11.2f}  {delta:>+7.3f}")
    results.append({
        "cancer": c, "panel_size": len(panel),
        "panel": ",".join(panel),
        "pairseed_auc": auc, "alvez_auc": alv, "delta": delta,
        "n_samples": int(ok.sum()),
    })

mean_pairseed = np.mean([r["pairseed_auc"] for r in results])
mean_alvez = np.mean([r["alvez_auc"] for r in results])
print(f"\nMean PairSeed v2 k=8 per-cancer AUC: {mean_pairseed:.3f}")
print(f"Mean Alvez       per-cancer AUC: {mean_alvez:.3f}")
print(f"Mean |delta|: "
      f"{np.mean([abs(r['delta']) for r in results]):.3f}")
n_better = sum(1 for r in results if r["delta"] > 0)
n_equal = sum(1 for r in results if abs(r["delta"]) <= 0.005)
print(f"PairSeed strictly better: {n_better}/{len(results)}; "
      f"within ±0.005: {n_equal}/{len(results)}")

res_df = pd.DataFrame(results)
csv_path = OUT / "pairseed_v2_k8_per_cancer_auc.csv"
res_df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")
