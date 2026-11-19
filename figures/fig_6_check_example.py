"""Figure 6 — an example of the additional typicality check.

GLIOM is removed from training and the PairSeed cascade is fitted on all
known patients (as in Figure S9B); 41 GLIOM patients are assigned to PRC by
the residual4 referee T74_PRC (7 proteins). (A) all 7 proteins in SD units
from the PRC mean, grey band = reference range; (B) two of the proteins with
the PRC reference box. Range limit: |z| <= norm.isf(0.05 / 7 / 2) = 2.69.

Uses the models of analysis/open_set_typicality_check.py.
Run from the repository root:
    python figures/fig_6_check_example.py
Output: paper_figures/fig_6_check_example.pdf (+ .png preview)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import norm
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_figures"
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT / "figures")]
sys.argv = sys.argv[:1]
import open_set_typicality_check as O  # noqa: E402
import fig_style  # noqa: E402
fig_style.apply()

HELD, PRED, TID = "GLIOM", "PRC", "T74_PRC"
tests = O.cascade_models(HELD, O.y != HELD)
t = tests[TID]
prot = list(O.X_df.columns[t.idx]); k = len(prot)
st = t.stats(PRED)
ZT = norm.isf(0.05 / k / 2)            # per-protein limit (Bonferroni)

hr = np.where((O.y == HELD) & O.COMPLETE)[0]
out = O.cascade_predict(tests, hr)
pred = np.array([o[0] for o in out])
Zh = (O.X[hr][:, t.idx] - st["z_mu"]) / st["z_sd"]
rej = np.abs(Zh).max(1) > ZT
toP = pred == PRED
ref_rows = (O.y == PRED) & O.panel_mask(t.idx, O.BLOCKS)   # PRC training
Zr = (O.X[ref_rows][:, t.idx] - st["z_mu"]) / st["z_sd"]
print(f"{TID}: {k} proteins {prot}; limit |z| > {ZT:.2f}")
print(f"{HELD}: {len(hr)} patients, {toP.sum()} -> {PRED}, "
      f"{(toP & rej).sum()} of them -> unknown by the check")
print(f"{PRED} training patients outside: {(np.abs(Zr).max(1) > ZT).mean():.1%}")

C_REF, C_KEEP, C_REJ = "#b5b5b5", "#d62728", "#2a7ab9"


# ---- combined draft: A = two proteins, B = all proteins -----------------
fig, (axB, axA) = plt.subplots(1, 2, figsize=(17, 7.2),
                               gridspec_kw=dict(width_ratios=[1.75, 1]))
viol = (np.abs(Zh[toP]) > ZT).sum(0)
a, b = np.argsort(-viol)[:2]
raw = lambda Z, j: Z[:, j] * st["z_sd"][j] + st["z_mu"][j]
axA.scatter(raw(Zr, a), raw(Zr, b), s=22, c=C_REF)
for msk, c, mk in ((toP & ~rej, C_KEEP, "o"), (toP & rej, C_REJ, "s")):
    axA.scatter(raw(Zh[msk], a), raw(Zh[msk], b), s=60, c=c, marker=mk,
                edgecolor="k", lw=.5)
lo = st["z_mu"] - ZT * st["z_sd"]; hi = st["z_mu"] + ZT * st["z_sd"]
axA.add_patch(Rectangle((lo[a], lo[b]), hi[a] - lo[a], hi[b] - lo[b],
                        fill=False, ec="k", lw=2, ls="--"))
axA.set_xlabel(f"{prot[a]} (NPX)", fontsize=24)
axA.set_ylabel(f"{prot[b]} (NPX)", fontsize=24)

rng = np.random.default_rng(1)
axB.axhspan(-ZT, ZT, color="#e8e8e8", zorder=0)
for j in range(k):
    axB.scatter(j - .2 + rng.uniform(-.1, .1, len(Zr)), Zr[:, j], s=8,
                c=C_REF, zorder=2)
    for msk, c, mk in ((toP & ~rej, C_KEEP, "o"), (toP & rej, C_REJ, "s")):
        axB.scatter(j + .2 + rng.uniform(-.1, .1, msk.sum()), Zh[msk, j],
                    s=26, c=c, marker=mk, edgecolor="k", lw=.4, zorder=3)
for s_ in (-ZT, ZT):
    axB.axhline(s_, color="k", ls="--", lw=1.2)
axB.set_xticks(range(k))
axB.set_xticklabels(prot, fontsize=14, fontweight="bold", rotation=30)
axB.set_ylabel(f"Deviation from the {PRED} mean (SD)", fontsize=24)
for ax, L in ((axB, "A"), (axA, "B")):
    ax.tick_params(labelsize=13)
    ax.text(-0.12, 1.03, L, transform=ax.transAxes, fontsize=26,
            fontweight="bold", ha="right", va="bottom")
h = [Line2D([], [], ls="", marker="o", ms=6, color=C_REF, label=f"{PRED} (training)"),
     Line2D([], [], ls="", marker="o", ms=9, mfc=C_KEEP, mec="k",
            label=f"{HELD} → {PRED}, kept"),
     Line2D([], [], ls="", marker="s", ms=9, mfc=C_REJ, mec="k",
            label=f"{HELD} → {PRED} → \"unknown\" after the check"),
     Line2D([], [], ls="--", color="k", lw=1.5,
            label=f"reference range of {PRED}")]
fig.legend(handles=h, loc="lower center", ncols=2, fontsize=21, frameon=False,
           bbox_to_anchor=(.5, -.02))
fig.tight_layout(rect=(0, .14, 1, 1), w_pad=4)
fig.savefig(OUT / "fig_6_check_example.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_6_check_example.png", dpi=100, bbox_inches="tight")
print("Saved paper_figures/fig_6_check_example.pdf")
