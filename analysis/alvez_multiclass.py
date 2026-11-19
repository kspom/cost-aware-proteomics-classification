"""Final classifier of the Alvez et al. (2023) pipeline: ONE multiclass
glmnet model trained jointly on all 12 cancers using a fixed protein
panel (their 'multiclassification setting', Fig. 5; Methods:
caret::train(method = "glmnet")).

Python equivalent used here:
  * multinomial logistic regression with elastic-net penalty
    (sklearn LogisticRegressionCV, solver = 'saga');
  * proteins standardised inside each training fold (glmnet does this
    internally);
  * alpha (= l1_ratio) and lambda (= 1/C) tuned by inner 5-fold CV on
    the training fold only. The alpha grid {0.1, 0.55, 1.0} is caret's
    default glmnet grid;
  * no class balancing (none is described for the multiclass setting).

Evaluation: outer 5-fold stratified CV, strict 12-class accuracy.
Deployable parameters: nonzero coefficients of the full-cohort fit
(tuned the same way) + 12 intercepts — zero weights need not be
deployed, so this is the most favourable count for the multiclass model.
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

L1_RATIOS = [0.1, 0.55, 1.0]
CS = np.logspace(-2, 1, 7)          # 0.01 ... 10
RAND = 42


def _model():
    return LogisticRegressionCV(
        Cs=CS, l1_ratios=L1_RATIOS, penalty="elasticnet", solver="saga",
        cv=StratifiedKFold(5, shuffle=True, random_state=RAND),
        scoring="accuracy", max_iter=3000, n_jobs=-1,
        random_state=RAND)


def fit_multiclass(X, y):
    """Tune + fit on (X, y). Returns (scaler, fitted model)."""
    sc = StandardScaler().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = _model().fit(sc.transform(X), y)
    return sc, clf


def evaluate_multiclass(X, y):
    """Outer 5-fold CV strict accuracy + full-cohort parameter count.

    X : (n_samples, n_proteins) array, y : array of class labels.
    Returns dict(strict_acc, n_proteins_input, n_proteins_used,
                 n_deployable_parameters, C, l1_ratio).
    """
    outer = StratifiedKFold(5, shuffle=True, random_state=RAND)
    preds = np.empty(len(y), dtype=object)
    for tr, te in outer.split(X, y):
        sc, clf = fit_multiclass(X[tr], y[tr])
        preds[te] = clf.predict(sc.transform(X[te]))
    strict = 100.0 * float(np.mean(preds == y))

    _, full = fit_multiclass(X, y)
    nz = np.abs(full.coef_) > 1e-10
    return {
        "strict_acc": strict,
        "n_proteins_input": X.shape[1],
        "n_proteins_used": int(nz.any(axis=0).sum()),
        "n_deployable_parameters": int(nz.sum()) + full.coef_.shape[0],
        "C": float(np.atleast_1d(full.C_)[0]),
        "l1_ratio": float(np.atleast_1d(full.l1_ratio_)[0]),
    }
