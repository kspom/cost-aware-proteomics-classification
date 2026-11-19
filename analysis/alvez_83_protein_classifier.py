"""Reproduction of the Alvez et al. (2023) panel-based classifier.

12 binary one-vs-rest classifiers, each fitted on ONLY its own
per-cancer protein subset taken from Alvez Supplementary Data 5.
Multiclass prediction = argmax over the 12 binary scores. The 83
unique proteins are partitioned per-cancer (3 to 18 proteins each);
each cancer's binary classifier sees only its own subset.

This is the "star" point on the Pareto plot — fixed panels from the
external paper, no panel search of our own.

Two fits reported:
  (a) Per-cancer OvR LDA — equal-prior LDA, comparable to our
      PairSeed-no-cascade architecture.
  (b) Per-cancer OvR L2 logistic — closest to Alvez's glmnet
      ("when the panel is fixed, additional L1 isn't needed").

Each fit reports strict 12-class accuracy from 5-fold
stratified CV.

Self-contained: no project-specific imports. Equal-prior LDA is
defined inline below since this reproduction is conceptually
independent of the PairSeed pipeline.

Inputs:
  filtered_wide.csv
  41467_2023_39765_MOESM7_ESM.xlsx   (Alvez Supplementary Data 5)

Output:
  paper_figures/alvez_83_protein_classifier.csv
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


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


OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Load Alvez panel and per-cancer mapping
ALV = pd.read_excel("41467_2023_39765_MOESM7_ESM.xlsx", sheet_name="Sheet1")
NAMEMAP = {
    "AML": "AML", "Breast": "BRC", "CLL": "CLL", "Cervical": "CVX",
    "Colorectal": "CRC", "DLBCL": "LYMPH", "Endometrial": "ENDC",
    "Glioma": "GLIOM", "Lung": "LUNGC", "Myeloma": "MYEL",
    "Ovarian": "OVC", "Prostate": "PRC",
}
ALV["our_code"] = ALV["Cancer"].map(NAMEMAP)
per_cancer_panels = {}
for c in sorted(set(ALV["our_code"])):
    sub = ALV.loc[ALV["our_code"] == c, "Protein"].tolist()
    # drop duplicates if a protein is listed twice for the same cancer
    per_cancer_panels[c] = sorted(set(sub))

print("Alvez per-cancer panels:")
for c in sorted(per_cancer_panels):
    panel = per_cancer_panels[c]
    print(f"  {c:>6} (n={len(panel):>2}): {' + '.join(panel)}")
total_unique = len(set().union(*per_cancer_panels.values()))
total_listed = sum(len(v) for v in per_cancer_panels.values())
print(f"\nTotal unique: {total_unique}, total listed: {total_listed}")


# Load cohort
wide = pd.read_csv("filtered_wide.csv", index_col="Sample_ID")
labels = wide["__Cancer__"]
X_full = wide.drop(columns="__Cancer__")
if "XG" in X_full.columns:
    X_full = X_full.drop(columns="XG")

# All 83 proteins are in the dense pool; restrict to complete-case
all_panel_proteins = sorted(set().union(*per_cancer_panels.values()))
keep = X_full[all_panel_proteins].notna().all(axis=1)
X_use = X_full.loc[keep]
y_use = labels[keep].to_numpy()
print(f"\nCohort: {len(X_use)} complete-case samples on Alvez panel")

ALL_12 = sorted(per_cancer_panels.keys())


def cv_eval_lda():
    """5-fold CV: for each fold, fit per-cancer binary LDA on the
    cancer's own panel; multiclass argmax."""
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    preds = np.empty(len(X_use), dtype=object)
    y_int = pd.Categorical(y_use, categories=ALL_12).codes
    for tr_idx, te_idx in skf.split(X_use, y_int):
        scores = np.full((len(te_idx), len(ALL_12)), -np.inf)
        for j, c in enumerate(ALL_12):
            panel = per_cancer_panels[c]
            Xtr = X_use[panel].iloc[tr_idx].to_numpy()
            Xte = X_use[panel].iloc[te_idx].to_numpy()
            ytr = (y_use[tr_idx] == c).astype(int)
            try:
                w, b = lda_fit(Xtr, ytr)
                scores[:, j] = Xte @ w + b
            except np.linalg.LinAlgError:
                scores[:, j] = -np.inf
        for i in range(len(te_idx)):
            preds[te_idx[i]] = ALL_12[np.argmax(scores[i])]
    return preds


def cv_eval_logistic(class_weight=None):
    """5-fold CV: per-cancer binary L2 logistic regression on the
    cancer's panel."""
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    preds = np.empty(len(X_use), dtype=object)
    y_int = pd.Categorical(y_use, categories=ALL_12).codes
    for tr_idx, te_idx in skf.split(X_use, y_int):
        scores = np.full((len(te_idx), len(ALL_12)), -np.inf)
        for j, c in enumerate(ALL_12):
            panel = per_cancer_panels[c]
            Xtr = X_use[panel].iloc[tr_idx].to_numpy()
            Xte = X_use[panel].iloc[te_idx].to_numpy()
            ytr = (y_use[tr_idx] == c).astype(int)
            if ytr.sum() < 2:
                continue
            clf = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs",
                max_iter=5000, random_state=42,
                class_weight=class_weight)
            clf.fit(Xtr, ytr)
            scores[:, j] = clf.decision_function(Xte)
        for i in range(len(te_idx)):
            preds[te_idx[i]] = ALL_12[np.argmax(scores[i])]
    return preds


def metrics(preds, name):
    strict = 100 * (preds == y_use).sum() / len(preds)
    print(f"  {name:>50}: strict={strict:>5.1f}%")
    return {"name": name, "strict_acc": strict}


print("\n=== Faithful Alvez OvR architecture: 12 binary classifiers, "
      "each on its per-cancer subset ===\n")
results = []
results.append(metrics(cv_eval_lda(),
                       "Alvez faithful (per-cancer OvR LDA)"))
results.append(metrics(cv_eval_logistic(class_weight=None),
                       "Alvez faithful (per-cancer OvR L2 logistic)"))
results.append(metrics(cv_eval_logistic(class_weight="balanced"),
                       "Alvez faithful (OvR L2 logistic, balanced)"))

# Alvez's actual final classifier: ONE multiclass glmnet on the panel
import sys
sys.path.insert(0, str(Path(__file__).parent))
from alvez_multiclass import evaluate_multiclass  # noqa: E402

print("\n=== Faithful Alvez final classifier: one multiclass glmnet "
      "on the 83-protein panel ===")
mc = evaluate_multiclass(X_use[all_panel_proteins].to_numpy(), y_use)
print(f"  strict={mc['strict_acc']:.1f}%  proteins used="
      f"{mc['n_proteins_used']}  params={mc['n_deployable_parameters']}  "
      f"C={mc['C']:.3g}  l1_ratio={mc['l1_ratio']}")

n_proteins = total_unique
for r in results:
    r["n_proteins"] = n_proteins
    r["n_deployable_parameters"] = total_listed + len(ALL_12)
results.append({"name": "Alvez faithful (multiclass glmnet, 83-protein panel)",
                "n_proteins": n_proteins, **mc})
pd.DataFrame(results).to_csv(
    OUT / "alvez_83_protein_classifier.csv", index=False)
print(f"\nSaved {OUT/'alvez_83_protein_classifier.csv'}")

print(f"\nFor reference (already on Pareto):")
print(f"  Variant B (multinomial L2 on all 83): strict 68.3%")
print(f"  Elastic-net family C=0.01 (auto-select 86p, multinomial L1): "
      f"strict 51.1%")


# ----------------------------------------------------------------------
# Boxplots: full-cohort fit, per-cancer score across the 12 disease
# groups, for both LDA and L2 logistic. Deployable formulas are also
# saved to CSV (intercept + per-protein weight).
# ----------------------------------------------------------------------

all_panel_proteins_sorted = sorted(set().union(*per_cancer_panels.values()))


def fit_full_lda(panel, target_class):
    Xp = X_use[panel].to_numpy()
    y = (y_use == target_class).astype(int)
    return lda_fit(Xp, y)


def fit_full_l2(panel, target_class):
    Xp = X_use[panel].to_numpy()
    y = (y_use == target_class).astype(int)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, random_state=42)
    clf.fit(Xp, y)
    return clf.coef_[0], float(clf.intercept_[0])


def compute_scores_and_formulas(fit_fn):
    """Returns (scores [N, 12], formula_rows [12 dicts])."""
    scores = np.zeros((len(X_use), len(ALL_12)))
    formula_rows = []
    for j, c in enumerate(ALL_12):
        panel = per_cancer_panels[c]
        w, b = fit_fn(panel, c)
        Xp = X_use[panel].to_numpy()
        scores[:, j] = Xp @ w + b
        row = {"class": c, "panel_size": len(panel), "intercept": float(b)}
        for p in all_panel_proteins_sorted:
            row[p] = 0.0
        for p, weight in zip(panel, w):
            row[p] = float(weight)
        formula_rows.append(row)
    return scores, formula_rows


def plot_boxplots(scores, title, out_pdf):
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    axes = axes.flatten()
    for j, c in enumerate(ALL_12):
        ax = axes[j]
        by_disease = [scores[y_use == d, j] for d in ALL_12]
        bp = ax.boxplot(by_disease, labels=ALL_12, showmeans=False,
                        patch_artist=True, widths=0.65,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker=".", markersize=3,
                                        markerfacecolor="grey",
                                        markeredgecolor="grey"))
        for k, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#9b3f9b" if ALL_12[k] == c else "#d6c0d8")
            patch.set_alpha(0.85 if ALL_12[k] == c else 0.55)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        panel = per_cancer_panels[c]
        title_lines = [f"{c} (k={len(panel)})"]
        chunk = []
        for p in panel:
            chunk.append(p)
            if len(" + ".join(chunk)) > 32:
                title_lines.append(" + ".join(chunk))
                chunk = []
        if chunk:
            title_lines.append(" + ".join(chunk))
        ax.set_title("\n".join(title_lines),
                     fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=60, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    plt.savefig(OUT / out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT / out_pdf}")


for fit_name, fit_fn, suptitle in [
    ("lda",
     fit_full_lda,
     f"Alvez panel — per-cancer OvR LDA at k_unique={total_unique}. "
     "Each subplot: one cancer's binary LDA score across the 12 "
     "disease groups (purple = target)."),
    ("l2",
     fit_full_l2,
     f"Alvez panel — per-cancer OvR L2 logistic at k_unique={total_unique}. "
     "Each subplot: one cancer's binary decision-function score across "
     "the 12 disease groups (purple = target)."),
]:
    scores, formula_rows = compute_scores_and_formulas(fit_fn)
    pd.DataFrame(formula_rows).to_csv(
        OUT / f"alvez_83_protein_classifier_{fit_name}_formulas.csv",
        index=False)
    print(f"Saved {OUT/f'alvez_83_protein_classifier_{fit_name}_formulas.csv'}")
    plot_boxplots(scores, suptitle,
                  f"alvez_83_protein_classifier_{fit_name}_boxplots.pdf")
