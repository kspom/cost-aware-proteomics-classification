"""PairSeed core library — basic functions for the two-step
protein-panel search used throughout the manuscript.

Self-contained: imports only `numpy`. All per-cancer panel-search
scripts in this project import their primitives from this file.

Algorithm overview
------------------
Every per-cancer one-vs-rest detector in the manuscript is built by
the same two-step procedure:

  Step 1.  Seed-pair search   — exhaustive 2-protein LDA scan over all
                                P·(P−1)/2 pairs; returns the top-K
                                pairs with the lowest balanced error.
                                Vectorised via Schur complement.
                                See `seed_pair_search`.

  Step 2.  Greedy expansion   — from each seed pair, add one protein
                                at a time (the one that minimises CV
                                balanced error), branching on near-ties
                                (`TIE_TOL`) and stopping on plateau
                                (`PLATEAU_TOL`) or `max_depth`. Keeps
                                the best panel at every intermediate
                                size, so the output contains 3-protein,
                                4-protein, ... up to `max_depth` panels.
                                See `greedy_branch_with_intermediates`.

Supporting primitives
---------------------
  lda_fit / predict / bal_err   — equal-prior binary LDA, sign-rule
                                  prediction, balanced error.
  fit_eval_fixed / cv_balanced_error
                                — CV balanced error for a fixed panel.
  eval_candidates_one_fold / cv_evaluate_candidates
                                — CV balanced error of (S ∪ {q}) for
                                  every candidate q, vectorised across
                                  candidates via Schur complement.
"""

import numpy as np


# Shared search hyper-parameters
MAX_DEPTH = 16            # default maximum panel size for greedy expansion
MAX_BRANCHES = 20         # beam width: surviving tied branches per step
TIE_TOL = 0.005           # error tolerance for treating candidates as tied
PLATEAU_TOL = 0.0005      # error tolerance for declaring a branch finished
N_FOLDS = 5               # CV fold count
RANDOM_STATE = 42         # seed for CV fold splits
DET_EPS = 1e-10           # Schur-complement degeneracy guard
TOP_K_SEEDS = 5           # how many seed pairs to keep from step 1


# ------------------- missing values: per-panel complete case ----------------
#
# In this cohort every missing value belongs to one of a few disjoint
# protein "blocks", and each incomplete sample lacks exactly one block.
# A panel S can therefore use every sample except those whose missing
# block intersects S. Functions below accept an optional
# `blocks = (protein_block, sample_block)` (from `missing_blocks`);
# with blocks=None they behave exactly as before (complete matrix).

def missing_blocks(X):
    """Return (protein_block, sample_block) int arrays; 0 = no missing.
    Raises if missing-value patterns overlap (blocks not disjoint)."""
    M = np.isnan(X)
    protein_block = np.zeros(X.shape[1], dtype=int)
    sample_block = np.zeros(X.shape[0], dtype=int)
    patterns = {}
    for s in np.where(M.any(axis=1))[0]:
        key = M[s].tobytes()
        if key not in patterns:
            patterns[key] = len(patterns) + 1
            if (protein_block[M[s]] != 0).any():
                raise ValueError("missing-value blocks overlap")
            protein_block[M[s]] = patterns[key]
        sample_block[s] = patterns[key]
    return protein_block, sample_block


def panel_mask(S, blocks):
    """Boolean mask of samples with complete data on protein set S."""
    protein_block, sample_block = blocks
    used = set(protein_block[list(S)].tolist()) - {0}
    if not used:
        return np.ones(len(sample_block), dtype=bool)
    return ~np.isin(sample_block, list(used))


def masked_splits(splits, mask):
    """Restrict each (train, test) index pair to samples in `mask`."""
    return [(tr[mask[tr]], te[mask[te]]) for tr, te in splits]


def load_cohort(path="filtered_wide.csv"):
    """Full analysis cohort: ALL patients (1273) x ALL proteins (1462),
    missing values kept as NaN. Returns (X_df, labels, blocks) where
    blocks = missing_blocks(X_df.to_numpy()). Use with per-panel
    complete case: fit each panel on panel_mask(panel, blocks)."""
    import pandas as pd
    wide = pd.read_csv(path, index_col="Sample_ID")
    labels = wide["__Cancer__"].to_numpy()
    X_df = wide.drop(columns="__Cancer__")
    if "XG" in X_df.columns:
        X_df = X_df.drop(columns="XG")
    return X_df, labels, missing_blocks(X_df.to_numpy(dtype=float))


def panel_scores(X, panel_idx, w, b):
    """LDA score X[:, panel]·w + b; NaN for samples missing any panel
    protein (such a test is 'not available' for that sample)."""
    Xp = X[:, panel_idx]
    out = np.full(X.shape[0], np.nan)
    ok = ~np.isnan(Xp).any(axis=1)
    out[ok] = Xp[ok] @ w + b
    return out


# ------------------------------ LDA primitives -------------------------------

def lda_fit(X, y):
    """Equal-prior binary LDA. Returns (w, b) such that the decision
    rule is sign(X·w + b)."""
    p = y == 1
    mu_p, mu_n = X[p].mean(0), X[~p].mean(0)
    Xc = X.copy()
    Xc[p] -= mu_p
    Xc[~p] -= mu_n
    Sigma = (Xc.T @ Xc) / (len(y) - 2)
    w = np.linalg.solve(Sigma, mu_p - mu_n)
    return w, -w @ ((mu_p + mu_n) / 2.0)


def predict(X, w, b):
    return (X @ w + b > 0).astype(int)


def bal_err(yt, yp):
    """Balanced error = ½(FNR + FPR)."""
    n_p = (yt == 1).sum()
    n_n = (yt == 0).sum()
    if n_p == 0 or n_n == 0:
        return np.nan
    fn = ((yt == 1) & (yp == 0)).sum() / n_p
    fp = ((yt == 0) & (yp == 1)).sum() / n_n
    return 0.5 * (fn + fp)


# ----------------------------- CV helpers -----------------------------------

def fit_eval_fixed(X_tr, y_tr, X_te, y_te, S):
    """Equal-prior LDA on protein subset S, fit on training fold,
    return balanced error on test fold (or 0.5 if degenerate)."""
    XS_tr = X_tr[:, S]
    pos = y_tr == 1
    neg = ~pos
    if pos.sum() < 2 or neg.sum() < 2:
        return 0.5
    mu_p = XS_tr[pos].mean(axis=0)
    mu_n = XS_tr[neg].mean(axis=0)
    Xc = XS_tr.copy()
    Xc[pos] -= mu_p
    Xc[neg] -= mu_n
    Sigma = (Xc.T @ Xc) / (len(y_tr) - 2)
    try:
        w = np.linalg.solve(Sigma, mu_p - mu_n)
    except np.linalg.LinAlgError:
        return 0.5
    thr = w @ ((mu_p + mu_n) / 2.0)
    scores = X_te[:, S] @ w
    pred = scores > thr
    y_te_pos = y_te == 1
    y_te_neg = ~y_te_pos
    if y_te_pos.sum() == 0 or y_te_neg.sum() == 0:
        return 0.5
    fnr = (~pred[y_te_pos]).sum() / y_te_pos.sum()
    fpr = pred[y_te_neg].sum() / y_te_neg.sum()
    return 0.5 * (fnr + fpr)


def cv_balanced_error(X, y, S, splits, blocks=None):
    """Mean CV balanced error of LDA fit on protein subset S
    (per-panel complete case when `blocks` is given)."""
    if blocks is not None:
        splits = masked_splits(splits, panel_mask(S, blocks))
    return float(np.mean([fit_eval_fixed(X[tr], y[tr], X[te], y[te], S)
                          for tr, te in splits]))


def select_parsimony_panel(best_per_depth, n_pos=None, tol=None):
    """Parsimony rule: from {k -> (panel, cv_err)}, return the panel
    at the smallest k whose CV balanced error is within one
    positive-sample's contribution of the minimum.

    Threshold rationale: a single positive sample contributes
    0.5 / n_pos to the CV balanced error (since bal_err =
    ½ × (FNR + FPR) and 1 misclassified positive raises FNR by
    1/n_pos). Setting `tol = 0.5 / n_pos` means: stop adding
    proteins as soon as the next protein would not, on average,
    correct one additional positive case — the natural integer-count
    plateau in raw misclassification space.

    Pass `n_pos` (positive-class count in the training cohort) and
    `tol` is computed automatically; or pass `tol` explicitly to
    override (legacy behaviour, e.g. for ablation studies).
    """
    if not best_per_depth:
        return None, None
    if tol is None:
        if n_pos is None or n_pos <= 0:
            raise ValueError("Pass n_pos (positive-class count) or "
                             "an explicit tol value.")
        tol = 0.5 / n_pos
    min_err = min(e for _, e in best_per_depth.values())
    chosen_k = min(k for k, (_, e) in best_per_depth.items()
                   if e <= min_err + tol)
    panel, err = best_per_depth[chosen_k]
    return panel, err


def eval_candidates_one_fold(X_tr, y_tr, X_te, y_te, S, cand_idx):
    """For each candidate q in cand_idx, fit equal-prior LDA on
    (S ∪ {q}) and return balanced test error. Vectorised across
    candidates via block-matrix Schur complement."""
    XS_tr = X_tr[:, S]
    Xq_tr = X_tr[:, cand_idx]
    pos = y_tr == 1
    neg = ~pos
    n_tr = len(y_tr)
    if pos.sum() < 2 or neg.sum() < 2:
        return np.full(len(cand_idx), np.inf)

    mu_p_S = XS_tr[pos].mean(axis=0)
    mu_n_S = XS_tr[neg].mean(axis=0)
    delta_S = mu_p_S - mu_n_S
    mid_S = 0.5 * (mu_p_S + mu_n_S)

    mu_p_q = Xq_tr[pos].mean(axis=0)
    mu_n_q = Xq_tr[neg].mean(axis=0)
    delta_q = mu_p_q - mu_n_q
    mid_q = 0.5 * (mu_p_q + mu_n_q)

    XSc = XS_tr.copy()
    XSc[pos] -= mu_p_S
    XSc[neg] -= mu_n_S
    Xqc = Xq_tr.copy()
    Xqc[pos] -= mu_p_q
    Xqc[neg] -= mu_n_q

    A = (XSc.T @ XSc) / (n_tr - 2)
    B = (XSc.T @ Xqc) / (n_tr - 2)
    d = (Xqc * Xqc).sum(axis=0) / (n_tr - 2)

    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.full(len(cand_idx), np.inf)

    v = A_inv @ delta_S
    u = A_inv @ B
    s = d - (B * u).sum(axis=0)
    valid = s > DET_EPS
    s_safe = np.where(valid, s, 1.0)

    Btv = B.T @ v
    w_q = (delta_q - Btv) / s_safe
    c_q = -w_q

    v_dot_mid = v @ mid_S
    u_dot_mid = mid_S @ u
    thr = v_dot_mid + u_dot_mid * c_q + w_q * mid_q

    XS_te = X_te[:, S]
    Xq_te = X_te[:, cand_idx]
    Xv = XS_te @ v
    Xu = XS_te @ u
    scores = Xv[:, None] + Xu * c_q[None, :] + Xq_te * w_q[None, :]
    pred = scores > thr[None, :]

    y_te_pos = y_te == 1
    y_te_neg = ~y_te_pos
    if y_te_pos.sum() == 0 or y_te_neg.sum() == 0:
        return np.full(len(cand_idx), 0.5)

    fn = (~pred[y_te_pos]).sum(axis=0).astype(float)
    fp = pred[y_te_neg].sum(axis=0).astype(float)
    bal = 0.5 * (fn / y_te_pos.sum() + fp / y_te_neg.sum())
    bal = np.where(valid, bal, np.inf)
    return bal


def cv_evaluate_candidates(X, y, S, cand_idx, splits, blocks=None):
    """Average balanced error across folds for each candidate q.
    With `blocks`, candidates are grouped by missing-value block so
    that each (S ∪ {q}) is evaluated on its own complete samples."""
    cand_idx = np.asarray(cand_idx)
    if blocks is None:
        accum = np.zeros(len(cand_idx))
        for tr, te in splits:
            accum += eval_candidates_one_fold(X[tr], y[tr], X[te], y[te],
                                              S, cand_idx)
        return accum / len(splits)
    out = np.empty(len(cand_idx))
    cand_block = blocks[0][cand_idx]
    for b in np.unique(cand_block):
        sel = np.where(cand_block == b)[0]
        m_splits = masked_splits(splits,
                                 panel_mask(list(S) + [int(cand_idx[sel[0]])],
                                            blocks))
        accum = np.zeros(len(sel))
        for tr, te in m_splits:
            accum += eval_candidates_one_fold(X[tr], y[tr], X[te], y[te],
                                              S, cand_idx[sel])
        out[sel] = accum / len(m_splits)
    return out


# ---------------------- Step 1: seed-pair search ----------------------------

def _pair_error_grid(X, y):
    """(P, P) grid of training balanced error of every 2-protein LDA
    (upper triangle; inf elsewhere). X must be complete."""
    n, P = X.shape
    n_pos = int(y.sum())
    n_neg = n - n_pos
    sw = np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg)
    mu_p = X[y == 1].mean(0)
    mu_n = X[y == 0].mean(0)
    delta = mu_p - mu_n
    mid = 0.5 * (mu_p + mu_n)
    Xc = X.copy()
    Xc[y == 1] -= mu_p
    Xc[y == 0] -= mu_n
    Sigma = (Xc.T @ Xc) / (n - 2)
    diag = np.diag(Sigma).copy()
    err_grid = np.full((P, P), np.inf)
    for i in range(P - 1):
        s_jj = diag[i + 1:]
        s_ij = Sigma[i, i + 1:]
        det = diag[i] * s_jj - s_ij ** 2
        valid = det > DET_EPS
        det_safe = np.where(valid, det, 1.0)
        d_i = delta[i]
        d_j = delta[i + 1:]
        wx = (s_jj * d_i - s_ij * d_j) / det_safe
        wy = (diag[i] * d_j - s_ij * d_i) / det_safe
        thr = wx * mid[i] + wy * mid[i + 1:]
        scores = X[:, i:i + 1] * wx[None, :] + X[:, i + 1:] * wy[None, :]
        pred = scores > thr[None, :]
        err = ((pred != y[:, None]) * sw[:, None]).sum(0)
        err_grid[i, i + 1:] = np.where(valid, err, np.inf)
    return err_grid


def seed_pair_search(X, y, proteins, top_k=TOP_K_SEEDS, blocks=None):
    """Exhaustive 2-protein LDA scan. Returns the top_k pairs with the
    lowest balanced error, vectorised via Schur complement.

    Parameters
    ----------
    X : (n_samples, n_proteins) ndarray, z-scored
    y : (n_samples,) binary 0/1 ndarray
    proteins : list of protein names of length n_proteins
    top_k : number of best pairs to return

    Returns
    -------
    list of (protein_i, protein_j, balanced_error), sorted by error.
    """
    if blocks is None:
        err_grid = _pair_error_grid(X, y)
    else:
        # every pair (i, j) is scored on the samples complete for both
        protein_block = blocks[0]
        P = X.shape[1]
        err_grid = np.full((P, P), np.inf)
        ids = np.unique(protein_block)
        for a_i, a in enumerate(ids):
            for b in ids[a_i:]:
                cols = np.where(np.isin(protein_block, [a, b]))[0]
                m = panel_mask(cols, blocks)
                sub = _pair_error_grid(X[np.ix_(m, cols)], y[m])
                ca, cb = protein_block[cols] == a, protein_block[cols] == b
                keep = (ca[:, None] & cb[None, :]) | (cb[:, None] & ca[None, :])
                gi, gj = np.meshgrid(cols, cols, indexing="ij")
                sel = keep & np.isfinite(sub)
                err_grid[gi[sel], gj[sel]] = sub[sel]
    P = X.shape[1]
    flat = err_grid.ravel()
    top_idx = np.argpartition(flat, top_k)[:top_k]
    top_idx = top_idx[np.argsort(flat[top_idx])]
    pairs = []
    for k in top_idx:
        i, j = divmod(int(k), P)
        pairs.append((proteins[i], proteins[j], float(flat[k])))
    return pairs


# --------------- Step 2: greedy forward expansion with branching -----------

def greedy_branch_with_intermediates(X, y, seed, splits, max_depth=MAX_DEPTH,
                                     blocks=None):
    """Greedy-forward expansion from a 2-protein seed.

    At each step, evaluates all remaining proteins as candidates,
    picks the one that minimises CV balanced error, and BRANCHES on
    near-ties (CV error within `TIE_TOL` of the minimum). At most
    `MAX_BRANCHES` branches survive per step (beam search). A branch
    stops when no candidate improves CV error by more than
    `PLATEAU_TOL`, on zero error, or at panel size `max_depth`.

    Parameters
    ----------
    X : (n_samples, n_proteins) ndarray, z-scored
    y : (n_samples,) binary 0/1 ndarray
    seed : 2-tuple of protein indices (into X's columns)
    splits : list of (train_idx, test_idx) from StratifiedKFold
    max_depth : maximum panel size to expand to (default: MAX_DEPTH=16)

    Returns
    -------
    completed : list of (S, err, stop_reason, trajectory) for every
                completed branch.
    best_per_depth : {depth: (S, err)} the lowest-error panel reached
                     at each panel size from len(seed) up to max_depth.
                     This is the dictionary used by the per-cancer
                     family scripts to extract a "best panel at k=2..K"
                     curve in a single search.
    """
    seed_S = list(seed)
    seed_err = cv_balanced_error(X, y, seed_S, splits, blocks)
    active = [(tuple(seed_S), seed_err)]
    traj = {tuple(seed_S): [seed_err]}
    completed = []
    best_per_depth = {len(seed_S): (tuple(seed_S), seed_err)}
    P = X.shape[1]
    for depth in range(len(seed_S) + 1, max_depth + 1):
        nxt = []
        for S, cur_err in active:
            cand = np.array([q for q in range(P) if q not in S])
            errs = cv_evaluate_candidates(X, y, list(S), cand, splits, blocks)
            min_err = errs.min()
            if min_err >= cur_err - PLATEAU_TOL:
                completed.append((S, cur_err, "plateau", traj[S]))
                continue
            tied_mask = errs <= min_err + TIE_TOL
            for q, e in zip(cand[tied_mask], errs[tied_mask]):
                new_S = tuple(list(S) + [int(q)])
                new_traj = traj[S] + [float(e)]
                traj[new_S] = new_traj
                d = len(new_S)
                if (d not in best_per_depth) or (e < best_per_depth[d][1]):
                    best_per_depth[d] = (new_S, float(e))
                if e <= 1e-12:
                    completed.append((new_S, float(e),
                                      "zero error", new_traj))
                else:
                    nxt.append((new_S, float(e)))
        nxt.sort(key=lambda r: r[1])
        active = nxt[:MAX_BRANCHES]
        if not active:
            break
    for S, e in active:
        completed.append((S, e, "max depth", traj[S]))
    return completed, best_per_depth
