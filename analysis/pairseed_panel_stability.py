"""Stability of PairSeed no-cascade panels (k = 8).

For R random stratified subsamples of 80% of the 1273 patients, the
full PairSeed search (seed-pair scan + greedy branching + parsimony
rule, k = 8) is repeated for each of the 12 one-vs-rest tests.
Training uses per-panel complete case on the subsample (as in the
main analysis); z-scoring is recomputed on the subsample.

For every (repeat, cancer) the selected panel and its balanced
accuracy on the remaining 20% (complete patients only, LDA fitted on
the panel-complete subsample) are recorded.

Outputs (paper_figures/):
  pairseed_panel_stability_runs.csv       one row per (repeat, cancer)
  pairseed_panel_stability_frequency.csv  selection frequency per
                                          (cancer, protein); flags
                                          proteins in the main k = 8
                                          panel (Figure 1)

Resumable: completed (repeat, cancer) jobs are read back from the
runs file and skipped.

Run from the repository root:
    OMP_NUM_THREADS=1 .venv/bin/python analysis/pairseed_panel_stability.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from pairseed_lib import (  # noqa: E402
    seed_pair_search, greedy_branch_with_intermediates,
    select_parsimony_panel, lda_fit, load_cohort, panel_mask,
    RANDOM_STATE)

OUT = Path("paper_figures")
OUT.mkdir(parents=True, exist_ok=True)
RUNS = OUT / "pairseed_panel_stability_runs.csv"
FREQ = OUT / "pairseed_panel_stability_frequency.csv"
R = 30
FRAC = 0.8
K = 8
N_WORKERS = int(os.environ.get("N_WORKERS", "12"))

X_df, y_all, BLOCKS = load_cohort()
PROTEINS = list(X_df.columns)
P = {p: i for i, p in enumerate(PROTEINS)}
X_raw = X_df.to_numpy(dtype=float)
ALL_12 = sorted(set(y_all))
COMPLETE = BLOCKS[1] == 0
SPLITS = list(StratifiedShuffleSplit(
    n_splits=R, train_size=FRAC, random_state=RANDOM_STATE
).split(np.zeros(len(y_all)), y_all))


def job(args):
    r, cancer = args
    t0 = time.perf_counter()
    tr, te = SPLITS[r]
    Xs = X_df.iloc[tr]
    Xn = ((Xs - Xs.mean(0)) / Xs.std(0, ddof=0).replace(0, 1.0)).to_numpy()
    y = (y_all[tr] == cancer).astype(int)
    bl = (BLOCKS[0], BLOCKS[1][tr])
    pairs = seed_pair_search(Xn, y, PROTEINS, top_k=5, blocks=bl)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
                  .split(Xn, y))
    best = {}
    for pi, pj, _ in pairs:
        done, bpd = greedy_branch_with_intermediates(
            Xn, y, (P[pi], P[pj]), splits, max_depth=K, blocks=bl)
        for d, (S, e) in bpd.items():
            if d not in best or e < best[d][1]:
                best[d] = (S, e)
        for S, e, _, _ in done:
            if len(S) not in best or e < best[len(S)][1]:
                best[len(S)] = (S, e)
    best = {d: v for d, v in best.items() if d <= K}
    S, err = select_parsimony_panel(best, n_pos=int(y.sum()))
    # held-out balanced accuracy (raw NPX, complete held-out patients)
    m = panel_mask(S, BLOCKS)
    rows = tr[m[tr]]
    w, b = lda_fit(X_raw[np.ix_(rows, S)],
                   (y_all[rows] == cancer).astype(int))
    tt = te[COMPLETE[te]]
    pred = X_raw[np.ix_(tt, S)] @ w + b > 0
    yt = y_all[tt] == cancer
    ba = 0.5 * (pred[yt].mean() + (~pred[~yt]).mean())
    return {"repeat": r, "cancer": cancer, "n_proteins": len(S),
            "proteins": ",".join(PROTEINS[i] for i in S),
            "cv_bal_err": round(float(err), 4),
            "heldout_bal_acc": round(100 * float(ba), 2),
            "seconds": round(time.perf_counter() - t0, 1)}


def write_frequency(runs):
    main = pd.read_csv(OUT / "pairseed_ovr_k8_panels.csv")
    main = {r.cancer: set(r.proteins.split(",")) for r in main.itertuples()}
    rows = []
    for c, g in runs.groupby("cancer"):
        n = len(g)
        cnt = pd.Series(",".join(g.proteins).split(",")).value_counts()
        for p, k in cnt.items():
            rows.append({"cancer": c, "protein": p, "n_repeats": n,
                         "frequency": round(k / n, 3),
                         "in_main_panel": p in main[c]})
        for p in main[c] - set(cnt.index):
            rows.append({"cancer": c, "protein": p, "n_repeats": n,
                         "frequency": 0.0, "in_main_panel": True})
    pd.DataFrame(rows).sort_values(
        ["cancer", "frequency"], ascending=[True, False]
    ).to_csv(FREQ, index=False)


if __name__ == "__main__":
    done = pd.read_csv(RUNS) if RUNS.exists() else pd.DataFrame()
    have = set(zip(done.get("repeat", []), done.get("cancer", [])))
    todo = [(r, c) for r in range(R) for c in ALL_12 if (r, c) not in have]
    print(f"{len(todo)} jobs to run ({len(have)} already done), "
          f"{N_WORKERS} workers", flush=True)
    t0 = time.perf_counter()
    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(job, todo), 1):
            pd.DataFrame([res]).to_csv(RUNS, mode="a", index=False,
                                       header=not RUNS.exists())
            el = time.perf_counter() - t0
            print(f"[{i}/{len(todo)}] r={res['repeat']} {res['cancer']:>6} "
                  f"n={res['n_proteins']} BA={res['heldout_bal_acc']:.1f} "
                  f"({res['seconds']:.0f}s; elapsed {el/60:.0f} min, "
                  f"eta {el/i*(len(todo)-i)/60:.0f} min)", flush=True)
    runs = pd.read_csv(RUNS).drop_duplicates(["repeat", "cancer"])
    write_frequency(runs)
    print(f"Saved {RUNS} and {FREQ}")
