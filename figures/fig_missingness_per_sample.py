"""Number of missing proteins per patient in the deposited dataset
(BioStudies S-BSST935, 1,477 patients x 1,463 proteins), with the
exclusion threshold used in explore_missingness.py (> 110 missing).

Read-only: does NOT rewrite filtered_wide.csv.
Output: paper_figures/fig_missingness_per_sample.pdf (+ .png preview)
Run from the repository root (~1 min; reads the 138 MB raw file).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLD = 110   # same as explore_missingness.SAMPLE_MISSING_THRESHOLD

df = pd.read_csv(ROOT / "pancancer_olink_data_biostudies_v2.txt",
                 sep=r"\s+", engine="c", usecols=[0, 1, 2])
df.columns = ["Sample_ID", "Cancer", "Assay"]
for c in df.columns:
    df[c] = df[c].str.strip('"')
n_prot = df.Assay.nunique()
miss = (n_prot - df.groupby("Sample_ID").size()).sort_values()
n_drop = int((miss > THRESHOLD).sum())

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(miss))
kept = miss.to_numpy() <= THRESHOLD
ax.bar(x[kept], miss.to_numpy()[kept], width=1.0, color="#4c72b0",
       label=f"kept (n = {kept.sum()})")
ax.bar(x[~kept], miss.to_numpy()[~kept], width=1.0, color="#c44e52",
       label=f"excluded (n = {n_drop})")
ax.axhline(THRESHOLD, color="black", ls="--", lw=1)
ax.text(10, THRESHOLD + 12, f"threshold: {THRESHOLD} missing proteins",
        fontsize=10)
ax.set_xlim(-5, len(miss) + 5)
ax.set_xlabel(f"Patients, sorted by number of missing proteins (n = {len(miss)})",
              fontsize=12)
ax.set_ylabel(f"Missing proteins (of {n_prot})", fontsize=12)
ax.legend(loc="upper left", fontsize=10, bbox_to_anchor=(0, 0.85))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_missingness_per_sample.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_missingness_per_sample.png", dpi=80, bbox_inches="tight")
print(f"{len(miss)} patients x {n_prot} proteins; zero missing: "
      f"{int((miss == 0).sum())}; 1-{THRESHOLD}: "
      f"{int(((miss > 0) & (miss <= THRESHOLD)).sum())}; excluded: {n_drop}")
