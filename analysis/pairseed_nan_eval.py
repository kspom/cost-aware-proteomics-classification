"""Shared evaluation helpers for the per-panel complete-case cohort
(all 1273 patients, NaN kept). Used by the no-cascade PairSeed scripts.

Rules
-----
* A test's LDA weights are fitted on the training rows that have
  complete data for that test's panel (pairseed_lib.panel_mask).
* A test's score is NaN for a patient missing any panel protein
  (pairseed_lib.panel_scores): the test is "not available".
* Multiclass (no-cascade) prediction = argmax over the available
  scores; if no score is available the prediction is 'unknown'
  (counted as wrong).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import (  # noqa: E402
    lda_fit, panel_mask, panel_scores, RANDOM_STATE)


def zscore_df(X_df):
    """z-score each protein over all patients, NaN-skipping, NaN kept."""
    return ((X_df - X_df.mean(axis=0)) /
            X_df.std(axis=0, ddof=0).replace(0, 1.0))


def fit_panel(X, y, idx, blocks, rows=None):
    """Equal-prior LDA on `rows` ∩ panel-complete samples.
    Returns (w, b, n_used) or (None, None, n_used) if singular."""
    m = panel_mask(idx, blocks)
    use = np.where(m)[0] if rows is None else rows[m[rows]]
    Xp = X[use][:, idx]
    assert not np.isnan(Xp).any(), "NaN reached an LDA fit"
    try:
        w, b = lda_fit(Xp, y[use])
    except np.linalg.LinAlgError:
        return None, None, len(use)
    return w, b, len(use)


def argmax_available(scores, classes):
    """Row-wise argmax over non-NaN scores; 'unknown' if none."""
    out = np.empty(scores.shape[0], dtype=object)
    for i, row in enumerate(scores):
        out[i] = "unknown" if np.isnan(row).all() \
            else classes[int(np.nanargmax(row))]
    return out


def ovr_argmax_cv(X, labels, panels, p_index, blocks, classes,
                  random_state=RANDOM_STATE):
    """5-fold stratified CV (panel-fixed) strict accuracy of the
    12-test OvR argmax classifier over ALL rows of X.
    Returns (strict_acc_percent, predictions)."""
    skf = StratifiedKFold(5, shuffle=True, random_state=random_state)
    y_int = pd.Categorical(labels, categories=classes).codes
    preds = np.empty(len(labels), dtype=object)
    for tr, te in skf.split(X, y_int):
        scores = np.full((len(te), len(classes)), np.nan)
        for j, c in enumerate(classes):
            idx = [p_index[p] for p in panels[c]]
            w, b, _ = fit_panel(X, (labels == c).astype(int), idx,
                                blocks, rows=tr)
            if w is not None:
                scores[:, j] = panel_scores(X[te], idx, w, b)
        preds[te] = argmax_available(scores, classes)
    return 100.0 * float(np.mean(preds == labels)), preds
