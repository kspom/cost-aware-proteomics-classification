"""Within-bucket OvR referee tests (clarifying tests that fire after
a bucket has been assigned by T4_trio or T8_RES4 to drill down to a
specific cancer). Each referee is trained ONLY on the bucket cohort
(no out-of-bucket negatives), at k = 8.

Trio referees (after T4_trio > 0):
    T41_CRC    : CRC   vs {LUNGC, OVC}
    T42_LUNGC  : LUNGC vs {CRC,   OVC}
    T43_OVC    : OVC   vs {CRC,   LUNGC}

Residual4 referees (after T8_RES4 > 0, or after the cascade falls
through to residual4 in the no-T8 architecture):
    T71_BRC    : BRC  vs {CVX, ENDC, PRC}
    T72_CVX    : CVX  vs {BRC, ENDC, PRC}
    T73_ENDC   : ENDC vs {BRC, CVX,  PRC}
    T74_PRC    : PRC  vs {BRC, CVX,  ENDC}

Output:
    paper_figures/bucket_referee_panels.csv
        one row per test:
        test_id, bucket, positives, negatives, panel_size, panel,
        search_err, intercept, weights
"""
import warnings
warnings.filterwarnings("ignore")

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
    RANDOM_STATE, load_cohort, panel_mask,
)

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
K_TARGET = 8

TRIO = ["CRC", "LUNGC", "OVC"]
RES4 = ["BRC", "CVX", "ENDC", "PRC"]


# -------------------------- load cohort ---------------------------------

# All 1273 patients x 1462 proteins, NaN kept (per-panel complete case).
X_use_df, y_use, BLOCKS = load_cohort()
DENSE = list(X_use_df.columns)          # name kept for minimal diff: all proteins
proteins_list = list(range(len(DENSE)))
print(f"Cohort: {len(X_use_df)} samples, {len(DENSE)} proteins")


def search_referee(positives, bucket_members):
    """Binary PairSeed search at k=K_TARGET, trained only on the
    bucket cohort: positives vs the other bucket members."""
    cohort_mask = np.isin(y_use, bucket_members)
    Xc = X_use_df.iloc[cohort_mask.nonzero()[0]]
    yc = y_use[cohort_mask]
    Xn = ((Xc - Xc.mean(axis=0)) /
          Xc.std(axis=0, ddof=0).replace(0, 1.0)).to_numpy()
    y_bin = np.isin(yc, list(positives)).astype(int)
    n_pos = int(y_bin.sum())
    n_neg = int(len(y_bin) - n_pos)

    bt = (BLOCKS[0], BLOCKS[1][cohort_mask])
    pairs = seed_pair_search(Xn, y_bin, proteins_list, top_k=5, blocks=bt)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(np.zeros(len(y_bin)), y_bin))
    combined_best = {}
    for pi, pj, _ in pairs:
        seed = (pi, pj)
        seed_branches, bpd = greedy_branch_with_intermediates(
            Xn, y_bin, seed, splits, max_depth=K_TARGET, blocks=bt)
        for d, (S_, e_) in bpd.items():
            if d not in combined_best or e_ < combined_best[d][1]:
                combined_best[d] = (S_, e_)
        for S_, err, _, _ in seed_branches:
            d = len(S_)
            if d not in combined_best or err < combined_best[d][1]:
                combined_best[d] = (S_, err)
    combined_best = {d: combined_best[d] for d in combined_best
                     if d <= K_TARGET}
    best_S, best_err = select_parsimony_panel(combined_best, n_pos=n_pos)
    panel = [DENSE[i] for i in best_S]

    ok = panel_mask(list(best_S), BLOCKS)[cohort_mask]   # complete on panel
    Xp = X_use_df[panel].iloc[cohort_mask.nonzero()[0][ok]].to_numpy()
    assert not np.isnan(Xp).any()
    print(f"    {panel}: fit on {int(ok.sum())}/{len(ok)} bucket patients")
    w, b = lda_fit(Xp, y_bin[ok])
    return panel, best_err, w, b, n_pos, n_neg


# -------------------------- run all 7 referees -------------------------

rows = []
referees = [
    ("T41_CRC",   "trio",      ["CRC"],   TRIO),
    ("T42_LUNGC", "trio",      ["LUNGC"], TRIO),
    ("T43_OVC",   "trio",      ["OVC"],   TRIO),
    ("T71_BRC",   "residual4", ["BRC"],   RES4),
    ("T72_CVX",   "residual4", ["CVX"],   RES4),
    ("T73_ENDC",  "residual4", ["ENDC"],  RES4),
    ("T74_PRC",   "residual4", ["PRC"],   RES4),
]

print(f"\n=== Within-bucket referee PairSeed search at k = {K_TARGET} ===")
for test_id, bucket, positives, members in referees:
    t0 = time.perf_counter()
    panel, err, w, b, n_pos, n_neg = search_referee(positives, members)
    elapsed = time.perf_counter() - t0
    negatives = [c for c in members if c not in positives]
    rows.append({
        "test_id": test_id,
        "bucket": bucket,
        "positives": ",".join(positives),
        "negatives": ",".join(negatives),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "panel_size": len(panel),
        "panel": ",".join(panel),
        "search_err": round(err, 4),
        "intercept": float(b),
        "weights": ",".join(f"{x:.4f}" for x in w),
    })
    print(f"  {test_id:<10} ({n_pos:3d}+ / {n_neg:3d}-) panel ({len(panel)}): "
          f"{'+'.join(panel)},  err={err:.3f}  ({elapsed:.0f}s)")

out_csv = OUT / "bucket_referee_panels.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")
