"""Build the voted-sex label table used by `alvez_mimicked_family.py`.

Self-contained pipeline that takes `filtered_wide.csv` and writes
`voted_sex_labels.csv` (plus a per-cancer
summary). Merges the work that was previously split across
`sex_differences_analysis.py` (panel search + multi-CV ranking) and
`sex_voted_boxplots.py` (vote across top panels) into one run.

Procedure
---------
1. Use the cancer label as a gold-standard sex proxy: PRC = male,
   OVC ∪ ENDC ∪ CVX = female. This gives 133 M + 299 F = 432 samples
   with definite sex.
2. On these 432 samples, run the PairSeed two-step search
   (seed-pair scan + greedy expansion) for the binary M-vs-F task.
3. Rank all candidate panels by repeated 5-fold balanced error
   (10 fold-seeds, then average).
4. Take the top-10 panels. For each, fit equal-prior LDA on all 432
   labelled samples and apply to the full 1273-sample cohort.
5. Majority vote across the 10 classifiers (votes_M ≥ 6 → M, else F).

Input:
    filtered_wide.csv

Outputs:
    voted_sex_labels.csv
        per-sample table: Sample_ID, cancer, votes_M, votes_F, voted_sex
    voted_sex_summary.csv
        per-cancer F/M counts (sanity-check that female-only and
        male-only cancers come out 100 % the expected sex)

The boxplot generation that lives in `sex_voted_boxplots.py` (per-
cancer F-vs-M split visualisation of the top-12 recurring proteins
etc.) is intentionally OMITTED here — left as commented stub at the
bottom of the file. Uncomment if you want the diagnostic figures.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from pairseed_lib import (
    seed_pair_search, greedy_branch_with_intermediates,
    lda_fit, predict, bal_err, RANDOM_STATE,
)


OUT = Path(".")

N_FOLDS = 5
N_VOTERS = 10
SEEDS_FOR_REPEATED_CV = [0, 1, 2, 7, 13, 42, 100, 999, 12345, 27182]

MALE_CLASSES = {"PRC"}
FEMALE_CLASSES = {"OVC", "ENDC", "CVX"}


# ---- load cohort ----
wide = pd.read_csv("filtered_wide.csv", index_col="Sample_ID")
labels = wide["__Cancer__"]
X_full = wide.drop(columns="__Cancer__").dropna(axis=1, how="any")
if "XG" in X_full.columns:
    X_full = X_full.drop(columns="XG")

mask_M = labels.isin(MALE_CLASSES)
mask_F = labels.isin(FEMALE_CLASSES)
mask_MF = mask_M | mask_F
X_MF = X_full[mask_MF]
y_MF = mask_M[mask_MF].astype(int).to_numpy()

n_M, n_F = int(y_MF.sum()), int((y_MF == 0).sum())
print(f"Gold-standard M (PRC):              {n_M}")
print(f"Gold-standard F (OVC + ENDC + CVX): {n_F}")
print(f"Total labelled for M-vs-F training: {len(y_MF)}")
print(f"Full cohort to label:               {len(X_full)}")

proteins = X_full.columns.tolist()


# ---- step 1: PairSeed two-step search on M vs F ----
print("\n=== Step 1: seed-pair search ===")
mu, sigma = X_MF.mean(), X_MF.std(ddof=0)
Xz = (X_MF - mu) / sigma
X = Xz.to_numpy(dtype=np.float64)

pairs = seed_pair_search(X, y_MF, proteins, top_k=5)
for pi, pj, e in pairs:
    print(f"  {pi} + {pj}  err={e:.4f}")

print("\n=== Step 2: greedy expansion from each seed ===")
splits = list(StratifiedKFold(
    N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y_MF))
p_index = {p: i for i, p in enumerate(proteins)}

all_completed = []
for s_rank, (pi, pj, _) in enumerate(pairs):
    seed = (p_index[pi], p_index[pj])
    completed, _ = greedy_branch_with_intermediates(X, y_MF, seed, splits)
    for S, err, reason, traj in completed:
        all_completed.append(
            {"S": S, "err": err, "reason": reason, "seed_rank": s_rank})

by_set = {}
for r in all_completed:
    key = tuple(sorted(r["S"]))
    if key not in by_set or r["err"] < by_set[key]["err"]:
        by_set[key] = r
unique = sorted(by_set.values(), key=lambda r: r["err"])
print(f"  {len(unique)} unique panels (best CV bal.err = "
      f"{unique[0]['err']:.4f})")


# ---- step 3: repeated 5-fold ranking ----
print("\n=== Step 3: repeated 5-fold ranking ===")


def repeated_5fold(panel_indices, X_full_z, y):
    Xp = X_full_z[:, panel_indices]
    means = []
    for s in SEEDS_FOR_REPEATED_CV:
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=s)
        errs = []
        for tr, te in skf.split(Xp, y):
            w, b = lda_fit(Xp[tr], y[tr])
            errs.append(bal_err(y[te], predict(Xp[te], w, b)))
        means.append(np.mean(errs))
    return float(np.mean(means)), float(np.std(means))


cv_rows = []
for r in unique:
    rep_m, rep_s = repeated_5fold(list(r["S"]), X, y_MF)
    cv_rows.append({
        "k": len(r["S"]),
        "panel": " + ".join(proteins[i] for i in r["S"]),
        "single_5fold_balerr": float(r["err"]),
        "rep_mean": rep_m,
        "rep_std": rep_s,
    })
cv_df = pd.DataFrame(cv_rows).sort_values(
    "rep_mean").reset_index(drop=True)

print(f"  Top {N_VOTERS} panels by repeated-CV mean:")
for i in range(min(N_VOTERS, len(cv_df))):
    r = cv_df.iloc[i]
    print(f"    #{i+1}  k={int(r['k'])}  rep_mean={r['rep_mean']:.4f} "
          f"± {r['rep_std']:.4f}  panel: {r['panel']}")


# ---- step 4: top-N classifiers applied to the full cohort + vote ----
print(f"\n=== Step 4: vote with top-{N_VOTERS} panels ===")
top_panels = cv_df.head(N_VOTERS).reset_index(drop=True)

votes_M = np.zeros(len(X_full), dtype=int)
for _, row in top_panels.iterrows():
    panel = row["panel"].split(" + ")
    Xp_train = X_MF[panel].to_numpy()
    w, b = lda_fit(Xp_train, y_MF)
    idx_per_sample = X_full[panel].to_numpy() @ w + b
    votes_M += (idx_per_sample > 0).astype(int)

voted_M = votes_M > (N_VOTERS / 2)
voted_sex = np.where(voted_M, "M", "F")

vote_df = pd.DataFrame({
    "Sample_ID": X_full.index,
    "cancer": labels.loc[X_full.index].values,
    "votes_M": votes_M,
    "votes_F": N_VOTERS - votes_M,
    "voted_sex": voted_sex,
})
vote_df.to_csv(OUT / "voted_sex_labels.csv", index=False)
print(f"Saved {OUT/'voted_sex_labels.csv'}")

summary = vote_df.groupby("cancer").agg(
    n=("voted_sex", "size"),
    n_M=("voted_sex", lambda s: (s == "M").sum()),
    n_F=("voted_sex", lambda s: (s == "F").sum()),
    pct_M=("voted_sex", lambda s: 100 * (s == "M").mean()),
)
summary.to_csv(OUT / "voted_sex_summary.csv")
print(f"Saved {OUT/'voted_sex_summary.csv'}")
print("\nPer-cancer voted-sex distribution:")
print(summary.round(1).to_string())


# ----------------------------------------------------------------------
# Boxplot diagnostics (intentionally commented out).
#
# The original `sex_voted_boxplots.py` also drew three PDF figures
# splitting each cancer into F/M sub-boxes (top-20 DE proteins,
# top-12 panel-recurring proteins, top-5 multi-protein classifier
# indices). Those are diagnostic, not used downstream by the
# Alvez-mimicked family. Uncomment below if you want them.
#
# import matplotlib.pyplot as plt
# ... plotting code from sex_voted_boxplots.py here ...
# ----------------------------------------------------------------------
