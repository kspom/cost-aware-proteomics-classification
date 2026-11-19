"""Supplementary Figure S10 — the data of Figure S9 in a simpler form.

For every disease removed from training: % of its patients (complete
protein data) with a favourable outcome, for the no-cascade and the
cascade classifier. Dark part of a bar: assigned to "unknown"; light
part: assigned to another member of the same bucket (bucketed diseases
only). The two leftmost categories pool the members of each bucket.

Inputs: the two honest_lodo_*_summary.csv files (as Figure S9).
Run from the repository root:
    python figures/fig_S10_lodo_favorable_bars.py
Output: paper_figures/fig_S10_lodo_favorable_bars.pdf (+ .png preview)
"""
import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))
import fig_style  # noqa: E402

fig_style.apply()
OUT = ROOT / "paper_figures"
C = fig_style.COLORS
DIS = ["AML", "MYEL", "CLL", "LYMPH", "GLIOM",
       "CRC", "LUNGC", "OVC", "BRC", "CVX", "ENDC", "PRC"]
BUCKETS = {"trio": {"CRC", "LUNGC", "OVC"},
           "residual4": {"BRC", "CVX", "ENDC", "PRC"}}
CATS = list(BUCKETS) + DIS
ARCH = [("no cascade", "honest_lodo_pairseed_nocascade_summary.csv",
         C["pairseed_no_cascade"]),
        ("cascade", "honest_lodo_cascade_with_referees_summary.csv",
         C["cascade"])]


def counts(fname):
    """per disease: (n, n_unknown, n_same_bucket)"""
    s = pd.read_csv(OUT / fname)
    out = {}
    for d in DIS:
        r = s[s.held_out == d].iloc[0]
        rt = ast.literal_eval(r.routing)
        sib = next((b - {d} for b in BUCKETS.values() if d in b), set())
        out[d] = (int(r.n), rt.get("unknown", 0),
                  sum(rt.get(x, 0) for x in sib))
    for b, mem in BUCKETS.items():
        out[b] = tuple(sum(out[d][k] for d in mem) for k in range(3))
    return out


fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(CATS)) + np.array([0, 0] + [0.6] * len(DIS))
w = 0.38
for k, (name, f, col) in enumerate(ARCH):
    cnt = counts(f)
    unk = np.array([100 * cnt[c][1] / cnt[c][0] for c in CATS])
    bkt = np.array([100 * cnt[c][2] / cnt[c][0] for c in CATS])
    xx = x + (k - 0.5) * w
    ax.bar(xx, unk, w, color=col, edgecolor="black", lw=.6)
    ax.bar(xx, bkt, w, bottom=unk, color=col, alpha=.35,
           edgecolor="black", lw=.6)
    for xi, v in zip(xx, unk + bkt):
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=10)
ax.axvline((x[1] + x[2]) / 2, color="grey", ls="--", lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels(CATS, fontsize=13, fontweight="bold", rotation=0)
ax.set_xlim(x[0] - .7, x[-1] + .7)
ax.set_ylim(0, 105)
ax.set_ylabel("Favorable outcome (%)", fontsize=15)
ax.set_xlabel("Disease removed from training", fontsize=15)
ax.tick_params(axis="y", labelsize=12)
ax.grid(axis="y", alpha=.3)
ax.set_axisbelow(True)
leg = [Patch(fc=C["pairseed_no_cascade"], ec="k", label="no cascade: unknown"),
       Patch(fc=C["pairseed_no_cascade"], alpha=.35, ec="k",
             label="no cascade: same bucket"),
       Patch(fc=C["cascade"], ec="k", label="cascade: unknown"),
       Patch(fc=C["cascade"], alpha=.35, ec="k", label="cascade: same bucket")]
ax.legend(handles=leg, ncols=4, loc="upper center",
          bbox_to_anchor=(0.5, -0.13), fontsize=12, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_S10_lodo_favorable_bars.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_S10_lodo_favorable_bars.png", dpi=100,
            bbox_inches="tight")
print("Saved paper_figures/fig_S10_lodo_favorable_bars.pdf")
