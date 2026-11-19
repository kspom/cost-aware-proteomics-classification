"""Scatter of the two first cascade tests: T1 (MYEL) vs T3 (CLL) LDA scores.

Same conventions as Figure 4: each test is fitted on its training cohort
(per-panel complete case, up to 1273 patients); points shown are the
1094 patients with complete protein data. Quadrants show the cascade
decision: T1 > 0 -> MYEL; T1 <= 0 and T3 > 0 -> CLL (or LYMPH via T6);
both <= 0 -> passed on to the next tests.

Output: paper_figures/fig_cascade_T1_vs_T3_scatter.pdf (+ .png preview)
Run from the repository root.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pairseed_lib import lda_fit, load_cohort, panel_mask, panel_scores  # noqa: E402

OUT = ROOT / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
X_df, y, BLOCKS = load_cohort(ROOT / "filtered_wide.csv")
X = X_df.to_numpy(dtype=float)
P = {p: i for i, p in enumerate(X_df.columns)}
COMPLETE = BLOCKS[1] == 0

pan = pd.read_csv(OUT / "pairseed_cascade_k8_panels.csv").set_index("test_id")


def score(tid, positive):
    assert pan.loc[tid, "training_cohort"] == "full"
    panel = pan.loc[tid, "panel"].split(" + ")
    idx = [P[p] for p in panel]
    m = panel_mask(idx, BLOCKS)
    w, b = lda_fit(X[np.ix_(m.nonzero()[0], idx)], (y[m] == positive).astype(int))
    return panel_scores(X, idx, w, b), panel


s1, p1 = score("T1_MYEL", "MYEL")
s3, p3 = score("T3_CLL", "CLL")
ok = COMPLETE

fig, ax = plt.subplots(figsize=(9, 7.5))
others = [c for c in sorted(set(y)) if c not in ("MYEL", "CLL")]
cmap = plt.get_cmap("tab10")
for k, c in enumerate(others):
    m = ok & (y == c)
    ax.scatter(s1[m], s3[m], s=9, color=cmap(k % 10), alpha=0.35,
               linewidths=0, label=f"{c} (n={m.sum()})", zorder=2)
m = ok & (y == "MYEL")
ax.scatter(s1[m], s3[m], s=60, marker="o", color="#8e6bbf",
           edgecolor="black", linewidths=0.7, label=f"MYEL (n={m.sum()})", zorder=4)
m = ok & (y == "CLL")
ax.scatter(s1[m], s3[m], s=70, marker="D", color="#22b8cf",
           edgecolor="black", linewidths=0.7, label=f"CLL (n={m.sum()})", zorder=4)

ax.axvline(0, color="black", ls="--", lw=0.9, zorder=1)
ax.axhline(0, color="black", ls="--", lw=0.9, zorder=1)
box = dict(boxstyle="square,pad=0.4", lw=0.8)
ax.text(0.02, 0.97, "T1 ≤ 0, T3 > 0\n→ CLL (caught by T3)", transform=ax.transAxes,
        va="top", fontsize=10, bbox={**box, "fc": "#d6f3f7", "ec": "#22b8cf"})
ax.text(0.98, 0.03, "T1 > 0\n→ MYEL (caught by T1)", transform=ax.transAxes,
        va="bottom", ha="right", fontsize=10,
        bbox={**box, "fc": "#ebe3f5", "ec": "#8e6bbf"})
ax.text(0.02, 0.03, "T1 ≤ 0, T3 ≤ 0\npassed to the next tests", transform=ax.transAxes,
        va="bottom", fontsize=9, color="0.3", bbox={**box, "fc": "white", "ec": "0.6"})

ax.set_xlabel(f"T1 (MYEL test) LDA score\npanel: {' + '.join(p1)}", fontsize=12)
ax.set_ylabel(f"T3 (CLL test) LDA score\npanel: {' + '.join(p3)}", fontsize=12)
ax.grid(alpha=0.25)
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9, title="cancer")
fig.tight_layout()
fig.savefig(OUT / "fig_cascade_T1_vs_T3_scatter.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_cascade_T1_vs_T3_scatter.png", dpi=80, bbox_inches="tight")
print("Saved", OUT / "fig_cascade_T1_vs_T3_scatter.pdf")
print("MYEL with T1>0:", int(((s1 > 0) & ok & (y == "MYEL")).sum()), "/",
      int((ok & (y == "MYEL")).sum()),
      "| non-MYEL with T1>0:", int(((s1 > 0) & ok & (y != "MYEL")).sum()))
print("CLL with T1<=0 & T3>0:", int(((s1 <= 0) & (s3 > 0) & ok & (y == "CLL")).sum()),
      "/", int((ok & (y == "CLL")).sum()),
      "| non-CLL there:", int(((s1 <= 0) & (s3 > 0) & ok & (y != "CLL")).sum()))
