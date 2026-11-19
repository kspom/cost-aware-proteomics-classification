"""PairSeed-no-cascade panel composition analysis.

Parallel to `pairseed_cascade_composition.py` but for the 12-OvR
no-cascade family. Reads the per-(cancer, k_target) panels produced
by `pairseed_no_cascade_family.py` and computes:

  - Union panel at each `k_target` (across all 12 OvR detectors).
  - Intersection matrix across the `k_target` grid (PDF heatmap +
    CSV).
  - Per-protein presence frequency across union panels.
  - Strict-core (proteins in EVERY union panel) and near-core
    (proteins present in most union panels — by default in the top
    four frequency tiers: N, N-1, N-2, N-3, where N is the number
    of union panels; near-core spans N-4..N-1).
  - Two heatmap "tables": rows = 12 cancer OvR detectors, columns =
    core / near-core proteins, cells = "+" if the protein is ever
    recruited by that detector across the `k_target` grid.

Inputs:
    paper_figures/pairseed_no_cascade_per_cancer_panels.csv
        Per-(cancer, k_target) panels produced by
        `pairseed_no_cascade_family.py`. Must exist — run the family
        script first. Columns: cancer, k_target, n_proteins, proteins
        (proteins is a comma-separated list).

Outputs (in `PairSeed_no_cascade/`):
    no_cascade_panel_lists_per_k.csv
        Union of the 12 detectors' panels at each k_target.
    no_cascade_intersection_matrix.csv / .pdf
        N × N matrix where cell (i, j) = |union_i ∩ union_j|.
    no_cascade_protein_presence_frequency.csv
        For every protein ever recruited: how many of the N union
        panels contain it.
    no_cascade_table_A_core.csv / .pdf
        Strict-core proteins (in every union panel) × 12 OvR
        detectors, with "+" where ever recruited.
    no_cascade_table_B_near_core.csv / .pdf
        Near-core proteins × 12 OvR detectors, same convention.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


IN_PATH = Path("paper_figures/pairseed_no_cascade_per_cancer_panels.csv")
OUT = Path("PairSeed_no_cascade")
OUT.mkdir(parents=True, exist_ok=True)


# ---- load per-(cancer, k_target) panels ----------------------------------

if not IN_PATH.exists():
    raise SystemExit(
        f"Input not found: {IN_PATH}\n"
        f"Run `python pairseed_no_cascade_family.py` first.")

df = pd.read_csv(IN_PATH)
panels = {(r["cancer"], int(r["k_target"])): r["proteins"].split(",")
          for _, r in df.iterrows()}
ALL_12 = sorted(df["cancer"].unique())
K_GRID = sorted(df["k_target"].unique())
print(f"k_target grid loaded from {IN_PATH}: {K_GRID}")
print(f"12 cancers: {ALL_12}\n")


# ---- union panel per k_target --------------------------------------------

union_panel = {}
for k in K_GRID:
    proteins = set()
    for c in ALL_12:
        proteins.update(panels[(c, k)])
    union_panel[k] = sorted(proteins)
    print(f"k_target={k:>2}: {len(proteins):>3} unique proteins across "
          f"the 12 detectors")

pd.DataFrame([
    {"k_target": k, "n_proteins": len(union_panel[k]),
     "proteins": ",".join(union_panel[k])}
    for k in K_GRID
]).to_csv(OUT / "no_cascade_panel_lists_per_k.csv", index=False)
print(f"\nSaved {OUT/'no_cascade_panel_lists_per_k.csv'}")


# ---- intersection matrix -------------------------------------------------

labels_col = [f"k={k} (n={len(union_panel[k])})" for k in K_GRID]
M = np.zeros((len(K_GRID), len(K_GRID)), dtype=int)
sets = [set(union_panel[k]) for k in K_GRID]
for i in range(len(K_GRID)):
    for j in range(len(K_GRID)):
        M[i, j] = len(sets[i] & sets[j])

pd.DataFrame(M, index=labels_col, columns=labels_col).to_csv(
    OUT / "no_cascade_intersection_matrix.csv")
print(f"Saved {OUT/'no_cascade_intersection_matrix.csv'}")


# ---- per-protein presence frequency --------------------------------------

all_proteins = sorted(set().union(*sets))
freq = {p: sum(p in sets[i] for i in range(len(K_GRID)))
        for p in all_proteins}
pd.DataFrame([
    {"protein": p, "n_panels_present": freq[p],
     "fraction_present": freq[p] / len(K_GRID)}
    for p in all_proteins
]).sort_values("n_panels_present", ascending=False).to_csv(
    OUT / "no_cascade_protein_presence_frequency.csv", index=False)
print(f"Saved {OUT/'no_cascade_protein_presence_frequency.csv'}")


# ---- strict-core and near-core -------------------------------------------

N = len(K_GRID)
strict_core = sorted([p for p in all_proteins if freq[p] == N])
near_n_minus_1 = sorted([p for p in all_proteins if freq[p] == N - 1])
near_n_minus_2 = sorted([p for p in all_proteins if freq[p] == N - 2])
near_n_minus_3 = sorted([p for p in all_proteins if freq[p] == N - 3])
near_n_minus_4 = sorted([p for p in all_proteins if freq[p] == N - 4])
NEAR_CORE = sorted(set(near_n_minus_1 + near_n_minus_2 +
                       near_n_minus_3 + near_n_minus_4))

print(f"\n=== No-cascade family core / near-core ===")
print(f"Total unique proteins across {N} union panels: {len(all_proteins)}")
print(f"Strict core (in EVERY one of {N} panels): "
      f"{len(strict_core)} proteins")
print(f"  {strict_core}")
print(f"In {N-1}/{N} panels: {len(near_n_minus_1)} -> {near_n_minus_1}")
print(f"In {N-2}/{N} panels: {len(near_n_minus_2)} -> {near_n_minus_2}")
print(f"In {N-3}/{N} panels: {len(near_n_minus_3)} -> {near_n_minus_3}")
print(f"In {N-4}/{N} panels: {len(near_n_minus_4)} -> {near_n_minus_4}")
print(f"Near-core (in {N-4}..{N-1} of {N} panels): "
      f"{len(NEAR_CORE)} proteins")


# ---- intersection matrix heatmap PDF -------------------------------------

fig, ax = plt.subplots(figsize=(14, 11))
im = ax.imshow(M, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(K_GRID)))
ax.set_yticks(range(len(K_GRID)))
ax.set_xticklabels(labels_col, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(labels_col, fontsize=10)
for i in range(len(K_GRID)):
    for j in range(len(K_GRID)):
        v = M[i, j]
        color = "white" if v < M.max() * 0.55 else "black"
        ax.text(j, i, f"{v}", ha="center", va="center",
                fontsize=10, color=color)
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("Number of shared proteins", fontsize=12)
ax.set_title(
    f"PairSeed-no-cascade panel intersection matrix\n"
    f"Diagonal = panel size; off-diagonal = proteins shared between "
    f"two k_target union panels.\n"
    f"Strict core ({len(strict_core)} proteins, in EVERY of {N} union "
    f"panels): {', '.join(strict_core) if strict_core else '—'}.",
    fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "no_cascade_intersection_matrix.pdf", bbox_inches="tight")
plt.close()
print(f"\nSaved {OUT/'no_cascade_intersection_matrix.pdf'}")


# ---- presence tables ------------------------------------------------------

def build_matrix(proteins):
    """Cell = 1 if the protein is in this detector's panel at ANY
    k_target in the grid (i.e., ever recruited by this OvR detector)."""
    M2 = np.zeros((len(ALL_12), len(proteins)), dtype=int)
    for i, c in enumerate(ALL_12):
        ever_in_panel = set()
        for k in K_GRID:
            ever_in_panel.update(panels[(c, k)])
        for j, p in enumerate(proteins):
            M2[i, j] = 1 if p in ever_in_panel else 0
    return M2


def render_table(M2, proteins, row_labels, fname, title, cmap_color):
    fig, ax = plt.subplots(figsize=(0.6 * len(proteins) + 5,
                                     0.6 * len(row_labels) + 1.5))
    cmap = mcolors.ListedColormap(["white", cmap_color])
    ax.imshow(M2, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(proteins)))
    ax.set_xticklabels(proteins, rotation=45, ha="right",
                       fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]):
            if M2[i, j]:
                ax.text(j, i, "+", ha="center", va="center",
                        fontsize=15, fontweight="bold", color="black")
    ax.set_xticks(np.arange(-0.5, len(proteins), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig(OUT / fname, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT/fname}")


if strict_core:
    M_core = build_matrix(strict_core)
    pd.DataFrame(np.where(M_core, "+", ""),
                 index=ALL_12, columns=strict_core).to_csv(
        OUT / "no_cascade_table_A_core.csv")
    render_table(
        M_core, strict_core, ALL_12,
        "no_cascade_table_A_core.pdf",
        f"PairSeed-no-cascade Table A. Strict core "
        f"({len(strict_core)} proteins, in every one of {N} OvR union "
        f"panels) — '+' if recruited by that OvR detector at any "
        f"k_target in the grid",
        "#1f7a3c")
else:
    print("(strict core is empty — skipping Table A)")

if NEAR_CORE:
    M_near = build_matrix(NEAR_CORE)
    pd.DataFrame(np.where(M_near, "+", ""),
                 index=ALL_12, columns=NEAR_CORE).to_csv(
        OUT / "no_cascade_table_B_near_core.csv")
    render_table(
        M_near, NEAR_CORE, ALL_12,
        "no_cascade_table_B_near_core.pdf",
        f"PairSeed-no-cascade Table B. Near-core "
        f"({len(NEAR_CORE)} proteins, in {N-4}–{N-1} of {N} OvR union "
        f"panels) — '+' if recruited by that OvR detector at any "
        f"k_target in the grid",
        "#6db088")
else:
    print("(near-core is empty — skipping Table B)")


# ---- console summary: which OvR detector recruits each strict-core protein

print(f"\nStrict-core protein → OvR detector mapping (any k_target):")
for p in strict_core:
    detectors_with_p = [c for c in ALL_12
                        if any(p in panels[(c, k)] for k in K_GRID)]
    print(f"  {p:>10}  →  {', '.join(detectors_with_p)}")
