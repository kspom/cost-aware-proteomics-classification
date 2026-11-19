"""Family of PairSeedCascade-with-referees classifiers across
target_k = 2..14 (panel-fixed 5-fold CV — same protocol as the
no-cascade family). For each target_k:

  1. Use the already-cached T1–T7 panels from
     PairSeed_cascades/cascade_family_panels.csv (produced by
     analysis/pairseed_cascade_family.py).
  2. Search referee panels (T41–T43 trio, T71–T74 residual4) at
     that same target_k on their bucket-only cohorts — fast,
     ~30 s total per target_k.
  3. Apply cascade-with-referees decision logic to all samples
     under panel-fixed 5-fold CV (panels fixed, LDA weights refit
     per fold). Compute strict 12-class accuracy
     accuracy + unique protein count + deployable parameters.

Output:
    PairSeed_cascades/cascade_with_referees_family.csv
        rows: target_k, n_proteins_unique, n_deployable_parameters,
              strict_acc.
    PairSeed_cascades/cascade_with_referees_family_panels.csv
        per (target_k, test_id) panel record.

Runtime: ~10 minutes (referee searches dominate; T1–T7 panels reused).
"""
import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import (
    seed_pair_search, greedy_branch_with_intermediates,
    select_parsimony_panel, lda_fit,
    RANDOM_STATE, load_cohort, panel_mask, panel_scores,
)

CAS = Path("PairSeed_cascades")
OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
K_RANGE = list(range(2, 15))   # 2..14

TRIO = ["CRC", "LUNGC", "OVC"]
RES4 = ["BRC", "CVX", "ENDC", "PRC"]

# Load canonical class lists + cohort specs from the JSON
# (the CSV uses '|' as separator for multi-class positives, which is
# why we read JSON instead and avoid string-parsing surprises).
with open(CAS / "architecture_with_referees.json") as f:
    ARCH = json.load(f)
TEST_META = {td["test_id"]: td for td in ARCH["tests"]}


# -------------------------- load cohort --------------------------------

# All 1273 patients x 1462 proteins, NaN kept (per-panel complete case).
X_use_df, y_use, BLOCKS = load_cohort()
X_raw = X_use_df.to_numpy(dtype=float)
DENSE = list(X_use_df.columns)          # name kept: all proteins
P_INDEX = {p: i for i, p in enumerate(DENSE)}
proteins_list = list(range(len(DENSE)))
print(f"Cohort: {len(X_use_df)} samples, {len(DENSE)} proteins")


# -------------------------- load T1-T7 family panels -------------------

t1_7 = pd.read_csv(CAS / "cascade_family_panels.csv")


def search_referee(positives, bucket_members, target_k):
    """Binary PairSeed search at target_k, trained on bucket cohort."""
    cohort_mask = np.isin(y_use, bucket_members)
    Xc = X_use_df.iloc[cohort_mask.nonzero()[0]]
    yc = y_use[cohort_mask]
    Xn = ((Xc - Xc.mean(axis=0)) /
          Xc.std(axis=0, ddof=0).replace(0, 1.0)).to_numpy()
    y_bin = np.isin(yc, list(positives)).astype(int)
    bt = (BLOCKS[0], BLOCKS[1][cohort_mask])
    pairs = seed_pair_search(Xn, y_bin, proteins_list, top_k=5, blocks=bt)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(np.zeros(len(y_bin)), y_bin))
    combined_best = {}
    for pi, pj, _ in pairs:
        seed = (pi, pj)
        seed_branches, bpd = greedy_branch_with_intermediates(
            Xn, y_bin, seed, splits, max_depth=target_k, blocks=bt)
        for d, (S_, e_) in bpd.items():
            if d not in combined_best or e_ < combined_best[d][1]:
                combined_best[d] = (S_, e_)
        for S_, err, _, _ in seed_branches:
            d = len(S_)
            if d not in combined_best or err < combined_best[d][1]:
                combined_best[d] = (S_, err)
    combined_best = {d: combined_best[d] for d in combined_best
                     if d <= target_k}
    best_S, best_err = select_parsimony_panel(
        combined_best, n_pos=int(y_bin.sum()))
    return [DENSE[i] for i in best_S], best_err


REFEREES = [
    ("T41_CRC",   ["CRC"],   TRIO),
    ("T42_LUNGC", ["LUNGC"], TRIO),
    ("T43_OVC",   ["OVC"],   TRIO),
    ("T71_BRC",   ["BRC"],   RES4),
    ("T72_CVX",   ["CVX"],   RES4),
    ("T73_ENDC",  ["ENDC"],  RES4),
    ("T74_PRC",   ["PRC"],   RES4),
]


# -------------------------- per-target_k loop --------------------------

family_rows = []
panel_rows = []
y_strat = pd.Categorical(y_use, categories=sorted(set(y_use))).codes
skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

for k_target in K_RANGE:
    print(f"\n=== target_k = {k_target} ===")
    t0 = time.perf_counter()

    panels = {}

    # T1-T7 from cached family
    sub = t1_7[t1_7["target_k"] == k_target]
    for _, r in sub.iterrows():
        panels[r["test_id"]] = r["panel"].split(" + ")

    # Referees: search at this target_k
    for tid, positives, members in REFEREES:
        panel, err = search_referee(positives, members, k_target)
        panels[tid] = panel
        panel_rows.append({
            "target_k": k_target, "test_id": tid,
            "bucket": "trio" if members is TRIO else "residual4",
            "positives": ",".join(positives),
            "panel_size": len(panel),
            "panel": ",".join(panel),
            "search_err": round(err, 4),
        })

    # Panel-fixed 5-fold CV scoring
    scores = {tid: np.full(len(y_use), np.nan) for tid in panels}
    for tr_idx, te_idx in skf.split(np.zeros(len(y_strat)), y_strat):
        for tid, panel in panels.items():
            pidx = [P_INDEX[p] for p in panel]
            # canonical class list + cohort spec from JSON
            meta = TEST_META[tid]
            cohort_spec = meta["training_cohort"]
            pos = set(meta["positive_classes"])
            if cohort_spec == "full":
                train_mask = np.ones(len(y_use), dtype=bool)
            elif cohort_spec.startswith("excl:"):
                excl = set(cohort_spec[len("excl:"):].split(","))
                train_mask = ~np.isin(y_use, list(excl))
            else:
                keep_set = set(cohort_spec[len("only:"):].split(","))
                train_mask = np.isin(y_use, list(keep_set))
            train_mask = train_mask & panel_mask(pidx, BLOCKS)
            tr_keep = train_mask[tr_idx]
            if tr_keep.sum() < 4:
                continue
            Xtr = X_raw[np.ix_(tr_idx[tr_keep], pidx)]
            assert not np.isnan(Xtr).any()
            ytr = np.isin(y_use[tr_idx[tr_keep]], list(pos)).astype(int)
            if ytr.sum() < 2 or (ytr == 0).sum() < 2:
                continue
            try:
                w, b = lda_fit(Xtr, ytr)
            except np.linalg.LinAlgError:
                continue
            scores[tid][te_idx] = panel_scores(X_raw[te_idx], pidx, w, b)

    # Apply cascade-with-referees. NaN score = test not available:
    # main tests fall through; referees argmax over available scores.
    def amax(refs):
        refs = [r for r in refs if not np.isnan(r[1])]
        return max(refs, key=lambda x: x[1])[0] if refs else "unknown"

    n = len(y_use)
    preds = np.empty(n, dtype=object)
    for i in range(n):
        if scores["T1_MYEL"][i] > 0:
            preds[i] = "MYEL"
        elif scores["T3_CLL"][i] > 0:
            preds[i] = ("LYMPH" if scores["T6_CLLvsLYMPH"][i] > 0
                        else "CLL")
        elif scores["T2_AML"][i] > 0:
            preds[i] = "AML"
        elif scores["T4_trio"][i] > 0:
            preds[i] = amax(
                [("CRC", scores["T41_CRC"][i]),
                 ("LUNGC", scores["T42_LUNGC"][i]),
                 ("OVC", scores["T43_OVC"][i])])
        elif scores["T5_GLIOM"][i] > 0:
            preds[i] = "GLIOM"
        elif scores["T7_LYMPH"][i] > 0:
            preds[i] = "LYMPH"
        else:
            preds[i] = amax(
                [("BRC", scores["T71_BRC"][i]),
                 ("CVX", scores["T72_CVX"][i]),
                 ("ENDC", scores["T73_ENDC"][i]),
                 ("PRC", scores["T74_PRC"][i])])

    # Accuracy
    # reported: 1094 complete patients; all-1273 kept
    _c = BLOCKS[1] == 0
    strict = (preds[_c] == y_use[_c]).mean() * 100
    strict_all = (preds == y_use).mean() * 100

    # Counts
    all_panels = sum(panels.values(), [])
    n_proteins = len(set(all_panels))
    n_params = sum(len(p) for p in panels.values()) + len(panels)

    family_rows.append({
        "target_k": k_target,
        "n_proteins_unique": n_proteins,
        "n_deployable_parameters": n_params,
        "strict_acc": round(strict, 2),
        "strict_acc_all_1273": round(strict_all, 2),
        "n_samples": int(_c.sum()),
    })

    elapsed = time.perf_counter() - t0
    print(f"  k={k_target}: {n_proteins}p, {n_params} params, "
          f"strict={strict:.1f}%  ({elapsed:.0f}s)")


# -------------------------- write outputs ------------------------------

fam_csv = CAS / "cascade_with_referees_family.csv"
pan_csv = CAS / "cascade_with_referees_family_panels.csv"
pd.DataFrame(family_rows).to_csv(fam_csv, index=False)
pd.DataFrame(panel_rows).to_csv(pan_csv, index=False)
print(f"\nSaved {fam_csv}")
print(f"Saved {pan_csv}")
