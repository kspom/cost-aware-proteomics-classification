"""Cohort: ALL 1273 patients, per-panel complete case; a test whose
panel is incomplete for a patient counts as not fired.

Honest LODO for PairSeed-no-cascade: for each held-out class X,
re-run PairSeed search from scratch on the 11-class cohort
(disease X excluded from training) for each of the 11 retained
OvR detectors. Apply argmax over the 11 OvR LDA scores; a sample
is classified as "unknown" iff the maximum score is <= 0 (the
natural LDA decision boundary — same threshold convention as the
cascade). NO learned tau threshold.

Output:
  paper_figures/honest_lodo_pairseed_nocascade_summary.csv
  paper_figures/honest_lodo_pairseed_nocascade_panels.csv  (per-fold,
    per-detector panel proteins for transparency)
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import (  # noqa: E402
    seed_pair_search, greedy_branch_with_intermediates,
    select_parsimony_panel, lda_fit, load_cohort, panel_mask,
    panel_scores, RANDOM_STATE,
)

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
K_TARGET = 8

TRIO = {"CRC", "LUNGC", "OVC"}
RES4 = {"BRC", "CVX", "ENDC", "PRC"}
def bucket_of(c):
    if c in TRIO: return "trio"
    if c in RES4: return "residual4"
    return c

# ALL patients, per-panel complete case (NaN kept)
X_use_df, y_use, BLOCKS = load_cohort()
proteins_list = list(X_use_df.columns)
p_index = {p: i for i, p in enumerate(proteins_list)}
XNP = X_use_df.to_numpy(dtype=float)
print(f"Cohort: {len(X_use_df)} × {len(proteins_list)} proteins (NaN kept)")


ALL_12 = sorted(np.unique(y_use))


def pairseed_search_for(target_class, held_out_class):
    """Run PairSeed search at k=K_TARGET on cohort minus held_out_class,
    target = target_class. Returns best panel (k_target proteins).
    """
    cohort_mask = (y_use != held_out_class)
    Xc = X_use_df.iloc[cohort_mask]
    yc = y_use[cohort_mask]
    # Z-normalise on this cohort
    Xn = ((Xc - Xc.mean(axis=0)) /
          Xc.std(axis=0, ddof=0).replace(0, 1.0)).to_numpy()
    y_bin = (yc == target_class).astype(int)
    blocks_c = (BLOCKS[0], BLOCKS[1][cohort_mask])
    pairs = seed_pair_search(Xn, y_bin, proteins_list, top_k=5,
                             blocks=blocks_c)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(Xn, y_bin))
    combined_best = {}
    for pi, pj, _ in pairs:
        seed = (p_index[pi], p_index[pj])
        seed_branches, bpd = greedy_branch_with_intermediates(
            Xn, y_bin, seed, splits, max_depth=K_TARGET,
            blocks=blocks_c)
        for d, (S_, e_) in bpd.items():
            if d not in combined_best or e_ < combined_best[d][1]:
                combined_best[d] = (S_, e_)
        for S_, err, _, _ in seed_branches:
            d = len(S_)
            if d not in combined_best or err < combined_best[d][1]:
                combined_best[d] = (S_, err)
    combined_best = {d: combined_best[d] for d in combined_best
                     if d <= K_TARGET}
    best_S, _ = select_parsimony_panel(
        combined_best, n_pos=int(y_bin.sum()))
    return [proteins_list[i] for i in best_S]


def fit_and_score(panel, target_class, held_out_class):
    """Fit binary LDA on cohort minus held-out (panel-complete patients
    only), score every sample (NaN where the panel is incomplete)."""
    idx = [p_index[p] for p in panel]
    rows = np.where((y_use != held_out_class)
                    & panel_mask(idx, BLOCKS))[0]
    ytr = (y_use[rows] == target_class).astype(int)
    if ytr.sum() < 2 or (ytr == 0).sum() < 2:
        return None, None, None
    Xtr = XNP[rows][:, idx]
    assert not np.isnan(Xtr).any()
    w, b = lda_fit(Xtr, ytr)
    return w, b, panel_scores(XNP, idx, w, b)


# ---- Main loop: 12 LODO folds ----
print(f"\n=== Honest PairSeed-no-cascade LODO at k_target={K_TARGET} ===")
summary_rows = []
panel_rows = []
for held in ALL_12:
    print(f"\n=== Hold out {held} (n={int((y_use == held).sum())}) ===")
    t0 = time.perf_counter()
    retained = [c for c in ALL_12 if c != held]

    # Re-search panels for 11 detectors
    panels = {}
    for det in retained:
        panel = pairseed_search_for(det, held)
        panels[det] = panel
        panel_rows.append({
            "held_out": held, "detector": det,
            "panel_size": len(panel),
            "panel": ",".join(panel),
        })
    print(f"  re-searched 11 panels in {time.perf_counter()-t0:.0f}s")

    # Refit LDA per detector and score all samples
    score_matrix = np.full((len(X_use_df), len(retained)), -np.inf)
    for j, det in enumerate(retained):
        w, b, s = fit_and_score(panels[det], det, held)
        if s is not None:
            # NaN (test not available) = not fired
            score_matrix[:, j] = np.where(np.isnan(s), -np.inf, s)
    max_scores = score_matrix.max(axis=1)
    argmax_class = np.array([retained[i]
                              for i in score_matrix.argmax(axis=1)])

    # Natural threshold: each binary OvR LDA test has decision boundary
    # at 0. A sample is "unknown" iff no detector positively identifies
    # it as its target class (max score <= 0). This matches the cascade
    # convention (each cascade test also uses the LDA-design threshold 0).
    pred = np.where(max_scores <= 0, "unknown", argmax_class)

    # routing reported on the 1094 complete patients
    held_mask = (y_use == held) & (BLOCKS[1] == 0)
    held_preds = pred[held_mask]
    routing = pd.Series(held_preds).value_counts().to_dict()
    n_held = int(held_mask.sum())
    pct_unknown = 100 * routing.get("unknown", 0) / n_held
    pct_wrong = 100 - pct_unknown  # everything not "unknown" is wrong

    # Top confused with (excluding "unknown")
    other_preds = [p for p in held_preds if p != "unknown"]
    top_confused = "—"
    if other_preds:
        cnt = pd.Series(other_preds).value_counts()
        top_n = min(3, len(cnt))
        top_confused = ", ".join(
            f"{c} ({100*cnt[c]/n_held:.0f}%)"
            for c in cnt.index[:top_n])

    summary_rows.append({
        "held_out": held, "n": n_held,
        "n_samples": int((BLOCKS[1] == 0).sum()),
        "pct_unknown": round(pct_unknown, 1),
        "pct_wrong": round(pct_wrong, 1),
        "top_confused_with": top_confused,
        "routing": str(routing),
    })
    print(f"  unknown {pct_unknown:.1f}%, wrong {pct_wrong:.1f}%, "
          f"top: {top_confused}")

pd.DataFrame(summary_rows).to_csv(
    OUT / "honest_lodo_pairseed_nocascade_summary.csv", index=False)
pd.DataFrame(panel_rows).to_csv(
    OUT / "honest_lodo_pairseed_nocascade_panels.csv", index=False)
print(f"\nSaved {OUT/'honest_lodo_pairseed_nocascade_summary.csv'}")
print(f"Saved {OUT/'honest_lodo_pairseed_nocascade_panels.csv'}")
