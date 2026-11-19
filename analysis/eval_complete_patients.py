"""Evaluation on the 1094 COMPLETE patients (manuscript convention).

Training convention (unchanged): every LDA test is fitted on all
training patients who have every protein of that test's panel
(per-panel complete case, up to 1273 patients).

Reporting convention: accuracy and LODO routing are counted only over
the 1094 patients with no missing protein — the same patients on which
the Alvez models are fitted and evaluated.

This script re-evaluates the SAVED panels (no panel search is
repeated). Each block first reproduces the stored all-1273 number as a
consistency check, then rewrites the CSV with:
    strict_acc            accuracy over the 1094 complete patients
    strict_acc_all_1273   accuracy over all 1273 (for the text)
    n_samples             1094

Files rewritten:
    paper_figures/pairseed_no_cascade_family.csv
    paper_figures/pairseed_ovr_k8_accuracy.csv
    PairSeed_cascades/cascade_with_referees_family.csv
    paper_figures/cascade_with_referees_full_cohort.csv
    paper_figures/honest_lodo_pairseed_nocascade_summary.csv
    paper_figures/honest_lodo_cascade_with_referees_summary.csv

Run from the repository root (~2 min):
    .venv/bin/python analysis/eval_complete_patients.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from pairseed_lib import (  # noqa: E402
    load_cohort, panel_mask, panel_scores, lda_fit, RANDOM_STATE)
from pairseed_nan_eval import ovr_argmax_cv  # noqa: E402

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
CAS = Path("PairSeed_cascades")
TRIO = ["CRC", "LUNGC", "OVC"]
RES4 = ["BRC", "CVX", "ENDC", "PRC"]

X_df, y, BLOCKS = load_cohort()
X = X_df.to_numpy(dtype=float)
P = {p: i for i, p in enumerate(X_df.columns)}
ALL_12 = sorted(set(y))
COMPLETE = BLOCKS[1] == 0
N_ALL, N_C = len(y), int(COMPLETE.sum())
print(f"Cohort: {N_ALL} patients, {N_C} complete, "
      f"{N_ALL - N_C} with one missing protein block")
assert N_C == 1094


def acc(pred, mask):
    return 100.0 * float(np.mean(pred[mask] == y[mask]))


def check(label, new, old, tol=0.06):
    flag = "ok" if abs(new - old) <= tol else "MISMATCH"
    print(f"    check {label}: recomputed {new:.2f} vs stored {old:.2f} "
          f"[{flag}]")
    if flag != "ok":
        raise SystemExit(f"Stored result not reproduced for {label}")


# ======================================================================
# 1. No-cascade family
# ======================================================================
print("\n[1] No-cascade family")
fam = pd.read_csv(OUT / "pairseed_no_cascade_family.csv")
pan = pd.read_csv(OUT / "pairseed_no_cascade_per_cancer_panels.csv")
old_all = fam.get("strict_acc_all_1273", fam["strict_acc"])
new_c, new_a = [], []
for (_, r), old in zip(fam.iterrows(), old_all):
    k = r["k_target"]
    sub = pan[pan.k_target == k]
    panels = {c: sub[sub.cancer == c].proteins.iloc[0].split(",")
              for c in ALL_12}
    _, pred = ovr_argmax_cv(X, y, panels, P, BLOCKS, ALL_12)
    a_all, a_c = acc(pred, np.ones(N_ALL, bool)), acc(pred, COMPLETE)
    check(f"k={k}", a_all, old)
    print(f"    k={k:>2}: all {a_all:.1f}%  complete {a_c:.1f}%  "
          f"incomplete {acc(pred, ~COMPLETE):.1f}%")
    new_a.append(a_all)
    new_c.append(a_c)
fam["strict_acc"] = new_c
fam["strict_acc_all_1273"] = new_a
fam["n_samples"] = N_C
fam.to_csv(OUT / "pairseed_no_cascade_family.csv", index=False)

k8 = pd.read_csv(OUT / "pairseed_ovr_k8_accuracy.csv")
row8 = fam[fam.k_target == 8].iloc[0]
k8["strict_acc"] = row8["strict_acc"]
k8["strict_acc_all_1273"] = row8["strict_acc_all_1273"]
k8["n_samples"] = N_C
k8.to_csv(OUT / "pairseed_ovr_k8_accuracy.csv", index=False)


# ======================================================================
# 2. Cascade-with-referees family (same CV as
#    pairseed_cascade_with_referees_family.py)
# ======================================================================
print("\n[2] Cascade-with-referees family")
with open(CAS / "architecture_with_referees.json") as f:
    META = {td["test_id"]: td for td in json.load(f)["tests"]}
t17 = pd.read_csv(CAS / "cascade_family_panels.csv")
refs = pd.read_csv(CAS / "cascade_with_referees_family_panels.csv")
cfam = pd.read_csv(CAS / "cascade_with_referees_family.csv")
y_strat = pd.Categorical(y, categories=ALL_12).codes
skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
SPLITS = list(skf.split(np.zeros(N_ALL), y_strat))


def train_mask_of(tid):
    spec = META[tid]["training_cohort"]
    if spec == "full":
        return np.ones(N_ALL, bool)
    if spec.startswith("excl:"):
        return ~np.isin(y, spec[5:].split(","))
    return np.isin(y, spec[5:].split(","))


def cascade_cv_predict(panels):
    scores = {t: np.full(N_ALL, np.nan) for t in panels}
    for tr, te in SPLITS:
        for tid, panel in panels.items():
            pidx = [P[p] for p in panel]
            m = train_mask_of(tid) & panel_mask(pidx, BLOCKS)
            rows = tr[m[tr]]
            if len(rows) < 4:
                continue
            ytr = np.isin(y[rows], META[tid]["positive_classes"]).astype(int)
            if ytr.sum() < 2 or (ytr == 0).sum() < 2:
                continue
            try:
                w, b = lda_fit(X[np.ix_(rows, pidx)], ytr)
            except np.linalg.LinAlgError:
                continue
            scores[tid][te] = panel_scores(X[te], pidx, w, b)

    def amax(lst):
        lst = [r for r in lst if not np.isnan(r[1])]
        return max(lst, key=lambda r: r[1])[0] if lst else "unknown"

    s = scores
    pred = np.empty(N_ALL, dtype=object)
    for i in range(N_ALL):
        if s["T1_MYEL"][i] > 0:
            pred[i] = "MYEL"
        elif s["T3_CLL"][i] > 0:
            pred[i] = "LYMPH" if s["T6_CLLvsLYMPH"][i] > 0 else "CLL"
        elif s["T2_AML"][i] > 0:
            pred[i] = "AML"
        elif s["T4_trio"][i] > 0:
            pred[i] = amax([(c, s[f"T4{j+1}_{c}"][i])
                            for j, c in enumerate(TRIO)])
        elif s["T5_GLIOM"][i] > 0:
            pred[i] = "GLIOM"
        elif s["T7_LYMPH"][i] > 0:
            pred[i] = "LYMPH"
        else:
            pred[i] = amax([(c, s[f"T7{j+1}_{c}"][i])
                            for j, c in enumerate(RES4)])
    return pred


old_all = cfam.get("strict_acc_all_1273", cfam["strict_acc"])
new_c, new_a, pred_k8 = [], [], None
for (_, r), old in zip(cfam.iterrows(), old_all):
    k = r["target_k"]
    panels = {rr.test_id: rr.panel.split(" + ")
              for rr in t17[t17.target_k == k].itertuples()}
    panels.update({rr.test_id: rr.panel.split(",")
                   for rr in refs[refs.target_k == k].itertuples()})
    assert len(panels) == 14, (k, sorted(panels))
    pred = cascade_cv_predict(panels)
    a_all, a_c = acc(pred, np.ones(N_ALL, bool)), acc(pred, COMPLETE)
    check(f"k={k}", a_all, old)
    print(f"    k={k:>2}: all {a_all:.1f}%  complete {a_c:.1f}%  "
          f"incomplete {acc(pred, ~COMPLETE):.1f}%")
    new_a.append(round(a_all, 2))
    new_c.append(round(a_c, 2))
    if k == 8:
        pred_k8 = pred
cfam["strict_acc"] = new_c
cfam["strict_acc_all_1273"] = new_a
cfam["n_samples"] = N_C
cfam.to_csv(CAS / "cascade_with_referees_family.csv", index=False)

full = pd.read_csv(OUT / "cascade_with_referees_full_cohort.csv")
row8 = cfam[cfam.target_k == 8].iloc[0]
print(f"    k=8 family {row8['strict_acc_all_1273']:.1f}% vs full-cohort "
      f"file {full['strict_acc'].iloc[0]:.1f}% (all 1273)")
full["strict_acc_all_1273"] = full["strict_acc"]
full["strict_acc"] = row8["strict_acc"]
full["n_samples"] = N_C
full.to_csv(OUT / "cascade_with_referees_full_cohort.csv", index=False)


# ======================================================================
# 3. No-cascade LODO (refit LDA on saved panels; same rule as
#    honest_lodo_pairseed_nocascade.py), counted on complete held-out
# ======================================================================
print("\n[3] No-cascade LODO")
lp = pd.read_csv(OUT / "honest_lodo_pairseed_nocascade_panels.csv")
old = pd.read_csv(OUT / "honest_lodo_pairseed_nocascade_summary.csv")
old_unk = old.get("pct_unknown_all_1273", old["pct_unknown"])
rows = []
for held, ou in zip(ALL_12, old_unk):
    retained = [c for c in ALL_12 if c != held]
    S = np.full((N_ALL, len(retained)), -np.inf)
    for j, det in enumerate(retained):
        panel = lp[(lp.held_out == held) & (lp.detector == det)
                   ].panel.iloc[0].split(",")
        idx = [P[p] for p in panel]
        tr = np.where((y != held) & panel_mask(idx, BLOCKS))[0]
        w, b = lda_fit(X[np.ix_(tr, idx)], (y[tr] == det).astype(int))
        s = panel_scores(X, idx, w, b)
        S[:, j] = np.where(np.isnan(s), -np.inf, s)
    pred = np.where(S.max(1) <= 0, "unknown",
                    np.array(retained)[S.argmax(1)])
    hm_all = y == held
    check(f"{held} unknown", 100 * np.mean(pred[hm_all] == "unknown"), ou,
          tol=0.06)
    hm = hm_all & COMPLETE
    hp = pred[hm]
    n = int(hm.sum())
    routing = pd.Series(hp).value_counts().to_dict()
    pu = 100 * routing.get("unknown", 0) / n
    other = pd.Series([p for p in hp if p != "unknown"]).value_counts()
    top = ", ".join(f"{c} ({100*other[c]/n:.0f}%)"
                    for c in other.index[:3]) if len(other) else "—"
    rows.append({"held_out": held, "n": n, "n_all_1273": int(hm_all.sum()),
                 "n_samples": N_C, "pct_unknown": round(pu, 1),
                 "pct_wrong": round(100 - pu, 1),
                 "pct_unknown_all_1273": round(float(ou), 1),
                 "top_confused_with": top, "routing": str(routing)})
    print(f"    {held:>6}: n={n:>3} unknown {pu:5.1f}%  top {top}")
pd.DataFrame(rows).to_csv(
    OUT / "honest_lodo_pairseed_nocascade_summary.csv", index=False)


# ======================================================================
# 4. Cascade LODO (saved weights; same rule as
#    reclassify_lodo_cascade_with_referees.py), complete held-out only
# ======================================================================
print("\n[4] Cascade-with-referees LODO")
lc =pd.read_csv(OUT / "honest_lodo_cascade_with_referees_panels.csv")
old = pd.read_csv(OUT / "honest_lodo_cascade_with_referees_summary.csv")
old_unk = dict(zip(old.held_out,
                   old.get("pct_unknown_all_1273", old["pct_unknown"])))


def cascade_predict_with_unknown(scores, held_out):
    n = N_ALL
    trio = [(f"T4{j+1}_{c}", c) for j, c in enumerate(TRIO)]
    res4 = [(f"T7{j+1}_{c}", c) for j, c in enumerate(RES4)]
    trio = [(t, c) for t, c in trio if c != held_out and t in scores]
    res4 = [(t, c) for t, c in res4 if c != held_out and t in scores]

    def s_of(t, i):
        return scores.get(t, np.full(n, -np.inf))[i]

    def dis(alive, i):
        rs = [(s_of(t, i), c) for t, c in alive
              if not np.isnan(s_of(t, i))]
        if not rs:
            return "unknown"
        bs, bc = max(rs)
        return bc if bs > 0 else "unknown"

    pred = np.empty(n, dtype=object)
    for i in range(n):
        if s_of("T1_MYEL", i) > 0:
            pred[i] = "MYEL"
        elif s_of("T3_CLL", i) > 0:
            pred[i] = "LYMPH" if s_of("T6_CLLvsLYMPH", i) > 0 else "CLL"
        elif s_of("T2_AML", i) > 0:
            pred[i] = "AML"
        elif s_of("T4_trio", i) > 0:
            pred[i] = dis(trio, i)
        elif s_of("T5_GLIOM", i) > 0:
            pred[i] = "GLIOM"
        elif s_of("T7_LYMPH", i) > 0:
            pred[i] = "LYMPH"
        else:
            pred[i] = dis(res4, i)
    return pred


rows = []
for held in ALL_12:
    sub = lc[lc.held_out == held]
    scores = {}
    for r in sub.itertuples():
        w = np.array([float(v) for v in r.weights.split(",")])
        scores[r.test_id] = panel_scores(
            X, [P[p] for p in r.panel.split(",")], w, float(r.intercept))
    pred = cascade_predict_with_unknown(scores, held)
    hm_all = y == held
    check(f"{held} unknown", 100 * np.mean(pred[hm_all] == "unknown"),
          old_unk[held])
    hm = hm_all & COMPLETE
    hp = pred[hm]
    n = int(hm.sum())
    routing = pd.Series(hp).value_counts().to_dict()
    pu = 100 * routing.get("unknown", 0) / n
    bucket = TRIO if held in TRIO else RES4 if held in RES4 else None
    pcb = (round(100 * sum(routing.get(c, 0) for c in bucket if c != held)
                 / n, 1) if bucket else "—")
    cnt = pd.Series(hp).value_counts()
    top = ", ".join(f"{c} ({100*cnt[c]/n:.0f}%)" for c in cnt.index[:3])
    rows.append({"held_out": held, "n": n, "n_all_1273": int(hm_all.sum()),
                 "n_samples": N_C, "n_tests_retained": len(sub),
                 "pct_unknown": round(pu, 1), "pct_correct_bucket": pcb,
                 "pct_unknown_all_1273": round(float(old_unk[held]), 1),
                 "top_routes": top, "routing": str(routing)})
    print(f"    {held:>6}: n={n:>3} unknown {pu:5.1f}%  bucket {pcb}  "
          f"top {top}")
pd.DataFrame(rows).to_csv(
    OUT / "honest_lodo_cascade_with_referees_summary.csv", index=False)
print("\nAll CSVs rewritten (strict_acc / LODO on 1094 complete patients).")
