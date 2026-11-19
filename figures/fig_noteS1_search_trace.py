"""Supplementary Note S1, Figure N1: PairSeed search trace for MYEL and LUNGC.

Reads the traces written by the trace step (one JSON per cancer, k = 8,
current cohort): paper_figures/search_trace_{MYEL,LUNGC}.json (analysis/pairseed_search_trace.py).
For each of the top-5 seed pairs, every completed greedy branch is drawn
as CV balanced error vs panel size (colour = seed pair). The shaded band
is the parsimony tolerance (minimum error + 0.5/n_pos); the star marks
the selected panel (smallest size within the band).

Output: paper_figures/fig_noteS1_search_trace.pdf (+ .png preview)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / "paper_figures"
OUT = ROOT / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
for ax, c, title in zip(axes, ["MYEL", "LUNGC"],
                        ["MYEL vs all others (easy)", "LUNGC vs all others (difficult)"]):
    d = json.load(open(TR / f"search_trace_{c}.json"))
    errs_all = []
    for k, s in enumerate(d["seeds"]):
        lab = f"{' + '.join(s['pair'])} ({len(s['branches'])} branches)"
        for j, b in enumerate(s["branches"]):
            xs = list(range(2, 2 + len(b["err_traj"])))
            ys = [100 * e for e in b["err_traj"]]
            errs_all += ys
            ax.plot(xs, ys, color=COLORS[k], alpha=0.45, lw=1.2,
                    label=lab if j == 0 else None)
    best = min(v["err"] for v in d["best_per_depth"].values())
    tol = d["selected"]["tol"]
    thr = 100 * (best + tol)
    # y-range: from just below the lowest error to clearly above the
    # threshold / the highest branch, so the top edge of the grey region
    # is visible and the region visibly starts at the bottom of the plot
    lo = min(errs_all) - 0.05 * (max(errs_all) - min(errs_all))
    hi = max(max(errs_all), thr) + 0.1 * (max(max(errs_all), thr) - lo)
    ax.set_ylim(lo, hi)
    ax.axhspan(lo, thr, color="0.85", zorder=0,
               label="acceptable: error ≤ min + 0.5/n_pos")
    sel = d["selected"]
    ax.plot(len(sel["panel"]), 100 * sel["err"], marker="*", ms=28,
            color="gold", mec="black", zorder=5, label="selected panel")
    # protein labels along the selected path
    path = sel["panel"]
    if c == "LUNGC":
        txt = (f"{path[0]} + {path[1]} → {path[2]} → {path[3]}\n"
               + " → ".join(["", *path[4:]]).lstrip())
    else:
        txt = " → ".join([f"{path[0]} + {path[1]}"] + path[2:])
    ax.set_title(f"{title}\nselected: {txt}", fontsize=16.5)
    ax.set_xlabel("Panel size (number of proteins)", fontsize=18)
    ax.set_ylabel("5-fold CV balanced error (%)", fontsize=18)
    xmax = 6 if c == "MYEL" else 8
    ax.set_xticks(range(2, xmax + 1))
    ax.set_xlim(1.8, xmax + 0.2)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12, loc="upper right", title="seed pair", title_fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig_noteS1_search_trace.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_noteS1_search_trace.png", dpi=80, bbox_inches="tight")
print("Saved", OUT / "fig_noteS1_search_trace.pdf")
