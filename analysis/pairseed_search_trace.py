"""Record the full PairSeed search (k = 8) for one one-vs-rest test:
top-5 seed pairs, every completed greedy branch with its CV-error
trajectory, the best panel per size and the panel selected by the
parsimony rule. Used for Supplementary Note S1, Figure N1.

Usage (from the repository root):
    python analysis/pairseed_search_trace.py MYEL
    python analysis/pairseed_search_trace.py LUNGC
Output: paper_figures/search_trace_<CANCER>.json
"""
import json
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from pairseed_lib import (seed_pair_search, greedy_branch_with_intermediates,  # noqa: E402
                          select_parsimony_panel, load_cohort, RANDOM_STATE)
from pairseed_nan_eval import zscore_df  # noqa: E402

K = 8
c = sys.argv[1]
X, y, B = load_cohort()
prot = list(X.columns)
P = {p: i for i, p in enumerate(prot)}
Xn = zscore_df(X).to_numpy(float)
yb = (y == c).astype(int)
pairs = seed_pair_search(Xn, yb, prot, top_k=5, blocks=B)
splits = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(Xn, yb))
out = {"cancer": c, "seeds": [], "best_per_depth": {}}
comb = {}
for pi, pj, e in pairs:
    done, bpd = greedy_branch_with_intermediates(Xn, yb, (P[pi], P[pj]), splits,
                                                 max_depth=K, blocks=B)
    out["seeds"].append({"pair": [pi, pj], "train_err": float(e), "branches": [
        {"panel": [prot[i] for i in S], "err_traj": [round(float(v), 4) for v in tr],
         "stop": why} for S, err, why, tr in done]})
    for d, (S, er) in bpd.items():
        if d not in comb or er < comb[d][1]:
            comb[d] = (S, er)
    for S, er, _, _ in done:
        if len(S) not in comb or er < comb[len(S)][1]:
            comb[len(S)] = (S, er)
comb = {d: v for d, v in comb.items() if d <= K}
S, er = select_parsimony_panel(comb, n_pos=int(yb.sum()))
out["best_per_depth"] = {d: {"panel": [prot[i] for i in v[0]], "err": round(float(v[1]), 4)}
                         for d, v in sorted(comb.items())}
out["selected"] = {"panel": [prot[i] for i in S], "err": round(float(er), 4),
                   "tol": round(0.5 / yb.sum(), 4)}
dst = Path("paper_figures") / f"search_trace_{c}.json"
dst.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(dst, "w"), indent=1)
print("Saved", dst, "selected:", out["selected"])
