"""Open-set analysis: an additional typicality check after classification.

Can the PairSeed classifiers answer "unknown" for a disease absent from
training without adding proteins? Uses the leave-one-disease-out (LODO)
panels (k = 8) written by analysis/honest_lodo_pairseed_nocascade.py and
analysis/honest_lodo_cascade_with_referees.py.

Check (manuscript, "Additional typicality check"): for every test on the
decision path that led to the predicted disease, each panel protein is
compared with its distribution among the training patients of the
predicted disease (normal, sample mean and SD); two-sided p-values are
Bonferroni-corrected over the proteins and the tests on the path, and the
prediction is replaced by "unknown" if the corrected p-value is < ALPHA.
Variants mentioned in the text: (b) LDA-score check, (d) joint check of the
panel (Mahalanobis distance, Ledoit-Wolf covariance, chi-square with k
degrees of freedom), and for the no-cascade classifier the rule "exactly
one test positive".

Evaluation inside every LODO experiment (held-out disease H):
  * known patients with complete protein data (11 trained diseases) are
    split 5-fold; LDA weights and reference ranges are fitted on 4/5
    (patients with incomplete data: training only) and applied to 1/5;
  * held-out patients are classified by each of the 5 models (pooled).
full_fit_check(): fitted on all known patients without the check, the
same code reproduces the LODO routing of Additional file 1, Figure S9.

Run from the repository root (< 1 min):
    python analysis/open_set_typicality_check.py
Outputs (paper_figures/):
    open_set_check_baseline.csv     reproduction check of Figure S9
    open_set_check_predictions.csv  one row per patient x fold x model
    open_set_check_summary.csv      rates used in the text and Figure 7
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.stats import chi2, norm
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pairseed_lib import lda_fit, load_cohort, panel_mask  # noqa: E402

PF = ROOT / "paper_figures"
TRIO = {"CRC", "LUNGC", "OVC"}
RES4 = {"BRC", "CVX", "ENDC", "PRC"}
GATES = ["score", "protein", "mahal"]
RAND = 42

X_df, y, BLOCKS = load_cohort(ROOT / "filtered_wide.csv")
X = X_df.to_numpy(dtype=float)
PIDX = {p: i for i, p in enumerate(X_df.columns)}
COMPLETE = BLOCKS[1] == 0
ALL_12 = sorted(set(y))
ARCH = json.load(open(ROOT / "PairSeed_cascades" /
                      "architecture_with_referees.json"))
COHORT_SPEC = {t["test_id"]: t["training_cohort"] for t in ARCH["tests"]}

PAN_NC = pd.read_csv(PF / "honest_lodo_pairseed_nocascade_panels.csv")
PAN_CA = pd.read_csv(PF / "honest_lodo_cascade_with_referees_panels.csv")


def bucket(c):
    return "trio" if c in TRIO else "res4" if c in RES4 else c


def cohort_mask(spec, held):
    m = y != held
    if spec == "full":
        return m
    kind, lst = spec.split(":")
    s = lst.split(",")
    return m & (~np.isin(y, s) if kind == "excl" else np.isin(y, s))


# ------------------------------------------------------------------ tests
class Test:
    """One binary LDA test with its panel, fitted on rows `train`."""

    def __init__(self, tid, panel, positives, cmask, train):
        self.tid, self.positives = tid, set(positives)
        self.idx = [PIDX[p] for p in panel]
        rows = cmask & train & panel_mask(self.idx, BLOCKS)
        yy = np.isin(y[rows], list(self.positives)).astype(int)
        self.ok = yy.sum() >= 2 and (yy == 0).sum() >= 2
        if self.ok:
            self.w, self.b = lda_fit(X[rows][:, self.idx], yy)
        self.train = train
        self._stats = {}

    def score(self, rows):
        Xp = X[rows][:, self.idx]
        s = Xp @ self.w + self.b
        s[np.isnan(s)] = -np.inf          # unavailable test = not fired
        return s

    def stats(self, cls):
        """Gate statistics of class `cls` on this panel (training rows)."""
        if cls not in self._stats:
            rows = self.train & (y == cls) & panel_mask(self.idx, BLOCKS)
            Z = X[rows][:, self.idx]
            lw = LedoitWolf().fit(Z)
            sc = Z @ self.w + self.b
            self._stats[cls] = dict(
                s_mu=sc.mean(), s_sd=sc.std(ddof=1),
                z_mu=Z.mean(0), z_sd=Z.std(0, ddof=1),
                mu=lw.location_, P=np.linalg.inv(lw.covariance_))
        return self._stats[cls]

    def pvals(self, cls, x):
        """Parametric p-values of one sample for the three gates:
        (b) two-sided normal p of the LDA score,
        (c) Bonferroni over the panel of two-sided normal p per protein,
        (d) chi-square(k) p of the squared Mahalanobis distance."""
        st, z = self.stats(cls), x[self.idx]
        k = len(self.idx)
        p_score = 2 * norm.sf(abs(z @ self.w + self.b - st["s_mu"]) / st["s_sd"])
        p_prot = min(1.0, k * (2 * norm.sf(
            np.abs(z - st["z_mu"]) / st["z_sd"])).min())
        d = z - st["mu"]
        p_mah = chi2.sf(d @ st["P"] @ d, df=k)
        return np.array([p_score, p_prot, p_mah])


# ------------------------------------------------------------ predictors
def nocascade_models(held, train):
    pan = PAN_NC[PAN_NC.held_out == held]
    tests = {}
    for _, r in pan.iterrows():
        t = Test(r.detector, r.panel.split(","), [r.detector],
                 y != held, train)
        if t.ok:
            tests[r.detector] = t
    return tests


def nocascade_predict(tests, rows):
    """Returns per sample: base prediction, #positive tests, path."""
    cls = list(tests)
    S = np.column_stack([tests[c].score(rows) for c in cls])
    out = []
    for i, r in enumerate(rows):
        j = int(np.argmax(S[i]))
        npos = int((S[i] > 0).sum())
        if S[i, j] <= 0:
            out.append(("unknown", npos, [], False))
        else:
            out.append((cls[j], npos, [(tests[cls[j]], cls[j])], npos > 1))
    return out


def cascade_models(held, train):
    pan = PAN_CA[PAN_CA.held_out == held]
    tests = {}
    for _, r in pan.iterrows():
        t = Test(r.test_id, r.panel.split(","), r.positives.split(","),
                 cohort_mask(COHORT_SPEC[r.test_id], held), train)
        if t.ok:
            tests[r.test_id] = t
    return tests


TRIO_REF = [("T41_CRC", "CRC"), ("T42_LUNGC", "LUNGC"), ("T43_OVC", "OVC")]
RES4_REF = [("T71_BRC", "BRC"), ("T72_CVX", "CVX"), ("T73_ENDC", "ENDC"),
            ("T74_PRC", "PRC")]


def cascade_predict(tests, rows):
    """Same decision logic as analysis/reclassify_lodo_cascade_with_referees.py
    (a bucket outputs "unknown" when every surviving referee scores <= 0).
    Returns per sample: (prediction, -, path, sign_violation)."""
    S = {t: tests[t].score(rows) for t in tests}

    def s(t, i):
        return S[t][i] if t in S else -np.inf

    out = []
    for i in range(len(rows)):
        T = tests.get
        if s("T1_MYEL", i) > 0:
            out.append(("MYEL", 0, [(T("T1_MYEL"), "MYEL")], False))
        elif s("T3_CLL", i) > 0:
            c = "LYMPH" if s("T6_CLLvsLYMPH", i) > 0 else "CLL"
            path = [(T("T3_CLL"), c)]
            if "T6_CLLvsLYMPH" in tests:
                path.append((T("T6_CLLvsLYMPH"), c))
            out.append((c, 0, path, False))
        elif s("T2_AML", i) > 0:
            out.append(("AML", 0, [(T("T2_AML"), "AML")], False))
        elif s("T4_trio", i) > 0:
            alive = [(t, c) for t, c in TRIO_REF if t in tests]
            sc = [(s(t, i), t, c) for t, c in alive]
            v, t, c = max(sc)
            npos = sum(x[0] > 0 for x in sc)
            if v <= 0:
                out.append(("unknown", 0, [], False))
            else:
                out.append((c, 0, [(T("T4_trio"), c), (T(t), c)], npos > 1))
        elif s("T5_GLIOM", i) > 0:
            out.append(("GLIOM", 0, [(T("T5_GLIOM"), "GLIOM")], False))
        elif s("T7_LYMPH", i) > 0:
            out.append(("LYMPH", 0, [(T("T7_LYMPH"), "LYMPH")], False))
        else:
            alive = [(t, c) for t, c in RES4_REF if t in tests]
            sc = [(s(t, i), t, c) for t, c in alive]
            v, t, c = max(sc)
            npos = sum(x[0] > 0 for x in sc)
            if v <= 0:
                out.append(("unknown", 0, [], False))
            else:
                out.append((c, 0, [(T(t), c)], npos > 1))
    return out


def path_pvalues(path, x):
    """Bonferroni-combined p-value of the path for each gate."""
    if not path:
        return np.ones(3)
    P = np.array([t.pvals(c, x) for t, c in path])
    return np.minimum(1.0, len(path) * P.min(axis=0))


# --------------------------------------------------------------- summary
ALPHA = 0.05
DISTINCT = ["AML", "CLL", "GLIOM", "LYMPH", "MYEL"]
BUCKETED = ["CRC", "LUNGC", "OVC", "BRC", "CVX", "ENDC", "PRC"]
VARIANTS = {"none": None, "protein": "p_protein", "score": "p_score",
            "mahal": "p_mahal", "one_positive": "sign_violation"}


def summary(df):
    """Rates (%) per architecture and check variant (alpha = ALPHA)."""
    rows = []
    for arch in ("nc", "cascade"):
        d = df[df.arch == arch]
        for v, col in VARIANTS.items():
            if v == "one_positive" and arch != "nc":
                continue
            p = d.pred.copy()
            if v == "one_positive":
                p[d.sign_violation] = "unknown"
            elif col:
                p[d[col] < ALPHA] = "unknown"
            k, h = d.kind == "known", d.kind == "heldout"
            r = dict(arch=arch, check=v,
                     known_unknown=100 * (p[k] == "unknown").mean(),
                     known_strict_acc=100 * (p[k] == d.true[k]).mean())
            for dis in DISTINCT + BUCKETED:
                m = h & (d.true == dis)
                r[f"unknown_{dis}"] = 100 * (p[m] == "unknown").mean()
                if dis in BUCKETED:
                    r[f"bucket_{dis}"] = 100 * (
                        p[m].map(bucket) == bucket(dis)).mean()
            r["unknown_distinct_mean"] = np.mean(
                [r[f"unknown_{x}"] for x in DISTINCT])
            r["unknown_bucketed_mean"] = np.mean(
                [r[f"unknown_{x}"] for x in BUCKETED])
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def run(arch):
    models, predict = ((nocascade_models, nocascade_predict) if arch == "nc"
                       else (cascade_models, cascade_predict))
    recs = []
    for held in ALL_12:
        known = np.where((y != held) & COMPLETE)[0]
        hrows = np.where((y == held) & COMPLETE)[0]
        skf = StratifiedKFold(5, shuffle=True, random_state=RAND)
        for f, (tr, te) in enumerate(skf.split(known, y[known])):
            train = np.zeros(len(y), bool)
            train[(y != held) & ~COMPLETE] = True   # incomplete: train only
            train[known[tr]] = True
            tests = models(held, train)
            for kind, rows in (("known", known[te]), ("heldout", hrows)):
                for r, (pred, npos, path, viol) in zip(
                        rows, predict(tests, rows)):
                    p = path_pvalues(path, X[r])
                    recs.append(dict(arch=arch, held=held, fold=f, kind=kind,
                                     true=y[r], pred=pred, npos=npos,
                                     sign_violation=viol, p_score=p[0],
                                     p_protein=p[1], p_mahal=p[2]))
        print(f"  {arch} held-out {held} done", flush=True)
    return pd.DataFrame(recs)


def full_fit_check():
    """Baseline reproduction: models fitted on all known samples must give
    the routing of Figure S9."""
    rows = []
    for arch, models, predict, f in (
            ("nc", nocascade_models, nocascade_predict,
             "honest_lodo_pairseed_nocascade_summary.csv"),
            ("cascade", cascade_models, cascade_predict,
             "honest_lodo_cascade_with_referees_summary.csv")):
        ref = pd.read_csv(PF / f)
        for held in ALL_12:
            tests = models(held, y != held)
            hrows = np.where((y == held) & COMPLETE)[0]
            pr = pd.Series([o[0] for o in predict(tests, hrows)])
            got = pr.value_counts().to_dict()
            want = eval(ref[ref.held_out == held].routing.iloc[0])
            rows.append(dict(arch=arch, held=held, match=got == want))
    df = pd.DataFrame(rows)
    print(df.groupby("arch").match.agg(["sum", "count"]))
    return df


if __name__ == "__main__":
    full_fit_check().to_csv(PF / "open_set_check_baseline.csv", index=False)
    df = pd.concat([run("nc"), run("cascade")], ignore_index=True)
    df.to_csv(PF / "open_set_check_predictions.csv", index=False)
    S = summary(df)
    S.round(2).to_csv(PF / "open_set_check_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(1).to_string(index=False))
