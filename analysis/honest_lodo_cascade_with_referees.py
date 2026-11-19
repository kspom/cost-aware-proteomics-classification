"""Honest LODO for the cascade-WITH-referees architecture.

For each held-out class X:
  1. Drop every cascade test (T1-T7) whose positive class becomes
     empty when X is removed.
  2. Drop every referee whose positive class is X.
  3. For every surviving test, re-run PairSeed search FROM SCRATCH
     on the test's cohort restricted to (full/only/excl) minus X.
  4. Apply the cascade-with-referees decision logic to all samples:
       T1..T7 priority as before;
       within trio bucket — argmax over surviving T41/T42/T43;
       within residual4 bucket — argmax over surviving T71-T74.
  5. Record routing of held-out samples per class.

Architecture spec: PairSeed_cascades/architecture_with_referees.json

Outputs (in paper_figures/):
    honest_lodo_cascade_with_referees_summary.csv
    honest_lodo_cascade_with_referees_panels.csv

This is the slow canonical script: ~3 hours on workstation hardware
because each held-out class re-runs PairSeed for up to 14 surviving
tests.
"""
import json
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
    RANDOM_STATE, load_cohort, panel_mask, panel_scores,
)

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
ARCH_PATH = Path("PairSeed_cascades") / "architecture_with_referees.json"
K_TARGET = 8

TRIO = {"CRC", "LUNGC", "OVC"}
RES4 = {"BRC", "CVX", "ENDC", "PRC"}


# -------------------------- load cohort --------------------------------

# All 1273 patients x 1462 proteins, NaN kept (per-panel complete case).
X_use_df, y_use, BLOCKS = load_cohort()
X_raw = X_use_df.to_numpy(dtype=float)
DENSE = list(X_use_df.columns)          # name kept: all proteins
P_INDEX = {p: i for i, p in enumerate(DENSE)}
proteins_list = list(range(len(DENSE)))
ALL_12 = sorted(set(y_use))
print(f"Cohort: {len(X_use_df)} samples, {len(DENSE)} proteins")


# -------------------------- architecture --------------------------------

with open(ARCH_PATH) as f:
    ARCH = json.load(f)
TEST_DEFS = ARCH["tests"]


def cohort_mask_for(spec, held_out):
    """Training cohort minus held_out, further filtered by spec."""
    mask = (y_use != held_out)
    if spec == "full":
        return mask
    if spec.startswith("excl:"):
        excl = set(spec[len("excl:"):].split(","))
        return mask & ~np.isin(y_use, list(excl))
    if spec.startswith("only:"):
        keep_set = set(spec[len("only:"):].split(","))
        return mask & np.isin(y_use, list(keep_set))
    raise ValueError(spec)


def search_panel(positives, cohort_mask):
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
    best_S, best_err = select_parsimony_panel(
        combined_best, n_pos=int(y_bin.sum()))
    return [DENSE[i] for i in best_S], best_err


def fit_test_lda(panel, positives, cohort_mask):
    m = cohort_mask & panel_mask([P_INDEX[p] for p in panel], BLOCKS)
    Xtr = X_raw[np.ix_(m.nonzero()[0], [P_INDEX[p] for p in panel])]
    assert not np.isnan(Xtr).any()
    ytr = np.isin(y_use[m], list(positives)).astype(int)
    if ytr.sum() < 2 or (ytr == 0).sum() < 2:
        return None, None
    return lda_fit(Xtr, ytr)


def build_cascade_for(held_out):
    """Drop tests whose positive class becomes empty after removing
    held_out; for T6, drop if held_out is CLL or LYMPH."""
    tests = []
    for td in TEST_DEFS:
        new_pos = set(td["positive_classes"]) - {held_out}
        if not new_pos:
            continue
        if td["test_id"] == "T6_CLLvsLYMPH":
            if held_out in ("CLL", "LYMPH"):
                continue
        mask = cohort_mask_for(td["training_cohort"], held_out)
        tests.append((td["test_id"], new_pos, mask))
    return tests


def predict(scores, held_out):
    """Cascade-with-referees decision per sample."""
    n = next(iter(scores.values())).shape[0]
    preds = np.empty(n, dtype=object)

    trio_alive = [(t, c) for t, c in [
        ("T41_CRC", "CRC"), ("T42_LUNGC", "LUNGC"),
        ("T43_OVC", "OVC")] if c != held_out]
    res4_alive = [(t, c) for t, c in [
        ("T71_BRC", "BRC"), ("T72_CVX", "CVX"),
        ("T73_ENDC", "ENDC"), ("T74_PRC", "PRC")] if c != held_out]

    def s_of(tid, i):
        return scores.get(tid, np.full(n, -np.inf))[i]

    for i in range(n):
        if s_of("T1_MYEL", i) > 0:
            preds[i] = "MYEL"
        elif s_of("T3_CLL", i) > 0:
            preds[i] = ("LYMPH"
                        if s_of("T6_CLLvsLYMPH", i) > 0 else "CLL")
        elif s_of("T2_AML", i) > 0:
            preds[i] = "AML"
        elif s_of("T4_trio", i) > 0:
            if trio_alive:
                rs = [(s_of(t, i), c) for t, c in trio_alive
                      if not np.isnan(s_of(t, i))]
                preds[i] = max(rs)[1] if rs else "unknown"
            else:
                preds[i] = "trio"  # all referees dropped — bucket only
        elif s_of("T5_GLIOM", i) > 0:
            preds[i] = "GLIOM"
        elif s_of("T7_LYMPH", i) > 0:
            preds[i] = "LYMPH"
        else:
            if res4_alive:
                rs = [(s_of(t, i), c) for t, c in res4_alive
                      if not np.isnan(s_of(t, i))]
                preds[i] = max(rs)[1] if rs else "unknown"
            else:
                preds[i] = "residual4"
    return preds


# -------------------------- main loop ---------------------------------

print("\n=== Honest LODO for cascade-with-referees ===")
summary_rows = []
panel_rows = []
for held in ALL_12:
    print(f"\n=== Hold out {held} (n={int((y_use == held).sum())}) ===")
    t0 = time.perf_counter()
    tests = build_cascade_for(held)
    print(f"  surviving tests: {[t[0] for t in tests]}")

    scores = {}
    for tid, positives, cohort_mask in tests:
        panel, err = search_panel(positives, cohort_mask)
        w, b = fit_test_lda(panel, positives, cohort_mask)
        if w is None:
            continue
        scores[tid] = panel_scores(X_raw, [P_INDEX[p] for p in panel], w, b)
        panel_rows.append({
            "held_out": held, "test_id": tid,
            "positives": ",".join(sorted(positives)),
            "panel_size": len(panel),
            "panel": ",".join(panel),
            "search_err": round(err, 4),
            "intercept": float(b),
            "weights": ",".join(f"{x:.4f}" for x in w),
        })

    preds = predict(scores, held)
    # routing reported on the 1094 complete patients
    held_mask = (y_use == held) & (BLOCKS[1] == 0)
    held_preds = preds[held_mask]
    routing = pd.Series(held_preds).value_counts().to_dict()
    n_held = int(held_mask.sum())

    # Strict-class correct: predicted == held_out
    # (= 0 by definition since held-out wasn't trained, but the test
    # still asks: did the cascade route the sample to its TRUE class?
    # The true class IS the held-out — answer is always 0 / impossible.
    # We instead report the correct-bucket fraction.)
    n_correct_bucket = 0
    if held in TRIO:
        n_correct_bucket = sum(routing.get(c, 0) for c in
                                 TRIO if c != held)
        pct_correct_bucket = 100 * n_correct_bucket / n_held
    elif held in RES4:
        n_correct_bucket = sum(routing.get(c, 0) for c in
                                 RES4 if c != held)
        pct_correct_bucket = 100 * n_correct_bucket / n_held
    else:
        pct_correct_bucket = None

    cnt = pd.Series(held_preds).value_counts()
    top_confused = ", ".join(
        f"{c} ({100*cnt[c]/n_held:.0f}%)"
        for c in cnt.index[:3])

    summary_rows.append({
        "held_out": held, "n": n_held,
        "n_samples": int((BLOCKS[1] == 0).sum()),
        "n_tests_retained": len(scores),
        "pct_correct_bucket": (round(pct_correct_bucket, 1)
                               if pct_correct_bucket is not None
                               else "—"),
        "top_routes": top_confused,
        "routing": str(routing),
    })
    elapsed = time.perf_counter() - t0
    print(f"  routing top: {top_confused}   "
          f"correct_bucket {pct_correct_bucket}    ({elapsed:.0f}s)")

pd.DataFrame(summary_rows).to_csv(
    OUT / "honest_lodo_cascade_with_referees_summary.csv", index=False)
pd.DataFrame(panel_rows).to_csv(
    OUT / "honest_lodo_cascade_with_referees_panels.csv", index=False)
print(f"\nSaved {OUT/'honest_lodo_cascade_with_referees_summary.csv'}")
print(f"Saved {OUT/'honest_lodo_cascade_with_referees_panels.csv'}")
