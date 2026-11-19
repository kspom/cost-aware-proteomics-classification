"""Cascade panel composition analysis (parallel to the
PairSeed-no-cascade `pairseed_panel_intersection_matrix_v2.py`).

Reads the per-target-k cascade panels produced by
`pairseed_cascade_family.py` and produces:

  - intersection matrix across the target_k grid (PDF heatmap +
    CSV).
  - per-protein presence frequency across union panels.
  - strict-core (proteins in EVERY union panel) and near-core
    (proteins present in most union panels).
  - two heatmap "tables": rows = the 7 cascade tests, columns =
    core / near-core proteins, cells = "+" if the protein is ever
    recruited by that test across the target_k grid.

Inputs:
    PairSeed_cascades/architecture.json
        Cascade architecture spec (used only for test ordering and
        per-test positive-class labels).
    PairSeed_cascades/cascade_family_panels.csv
        Per-(target_k, test_id) panels produced by
        `pairseed_cascade_family.py`. Must exist — run the family
        script first.

Outputs (in `PairSeed_cascades/`):
    cascade_panel_lists_per_k.csv
        Union of the 7 tests' panels at each target_k.
    cascade_intersection_matrix.csv
        N × N matrix where cell (i, j) = |union_i ∩ union_j|.
    cascade_intersection_matrix.pdf
        Heatmap of the above.
    cascade_protein_presence_frequency.csv
        For every protein ever recruited: how many of the N union
        panels contain it.
    cascade_table_A_core.csv + .pdf
        Strict-core proteins (in every union panel) × 7 cascade
        tests, with "+" where ever recruited.
    cascade_table_B_near_core.csv + .pdf
        Near-core proteins × 7 cascade tests, same convention.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


OUT = Path("PairSeed_cascades")
OUT.mkdir(parents=True, exist_ok=True)


# ---- load architecture for test ordering ---------------------------------

with open(OUT / "architecture.json") as f:
    arch = json.load(f)

TEST_ORDER = [t["test_id"] for t in arch["tests"]]
TEST_LABELS = {
    t["test_id"]: f"{t['test_id']} ({','.join(t['positive_classes'])})"
    for t in arch["tests"]
}


# ---- load per-(target_k, test_id) panels ---------------------------------

panels_csv = OUT / "cascade_family_panels.csv"
if not panels_csv.exists():
    raise SystemExit(
        f"Input not found: {panels_csv}\n"
        f"Run `python pairseed_cascade_family.py` first.")

df = pd.read_csv(panels_csv)
# panel field is " + "-separated (matches pairseed_cascade_family.py output)
panels = {(r["test_id"], r["target_k"]): r["panel"].split(" + ")
          for _, r in df.iterrows()}
K_GRID = sorted(df["target_k"].unique())
print(f"target_k grid loaded from {panels_csv}: {K_GRID}")
print(f"tests: {TEST_ORDER}\n")


# ---- union panel per target_k --------------------------------------------

union_panel = {}
for k in K_GRID:
    proteins = set()
    for tid in TEST_ORDER:
        proteins.update(panels[(tid, k)])
    union_panel[k] = sorted(proteins)
    print(f"target_k={k:>2}: {len(proteins):>3} unique proteins across "
          f"the 7 tests")

pd.DataFrame([
    {"target_k": k, "n_proteins": len(union_panel[k]),
     "proteins": ",".join(union_panel[k])}
    for k in K_GRID
]).to_csv(OUT / "cascade_panel_lists_per_k.csv", index=False)
print(f"\nSaved {OUT/'cascade_panel_lists_per_k.csv'}")


# ---- intersection matrix -------------------------------------------------

labels_col = [f"k={k} (n={len(union_panel[k])})" for k in K_GRID]
M = np.zeros((len(K_GRID), len(K_GRID)), dtype=int)
sets = [set(union_panel[k]) for k in K_GRID]
for i in range(len(K_GRID)):
    for j in range(len(K_GRID)):
        M[i, j] = len(sets[i] & sets[j])

pd.DataFrame(M, index=labels_col, columns=labels_col).to_csv(
    OUT / "cascade_intersection_matrix.csv")
print(f"Saved {OUT/'cascade_intersection_matrix.csv'}")


# ---- per-protein presence frequency --------------------------------------

all_proteins = sorted(set().union(*sets))
freq = {p: sum(p in sets[i] for i in range(len(K_GRID)))
        for p in all_proteins}
pd.DataFrame([
    {"protein": p, "n_panels_present": freq[p],
     "fraction_present": freq[p] / len(K_GRID)}
    for p in all_proteins
]).sort_values("n_panels_present", ascending=False).to_csv(
    OUT / "cascade_protein_presence_frequency.csv", index=False)
print(f"Saved {OUT/'cascade_protein_presence_frequency.csv'}")


# ---- strict-core and near-core -------------------------------------------

N = len(K_GRID)
strict_core = sorted([p for p in all_proteins if freq[p] == N])
near_n_minus_1 = sorted([p for p in all_proteins if freq[p] == N - 1])
near_n_minus_2 = sorted([p for p in all_proteins if freq[p] == N - 2])
near_n_minus_3 = sorted([p for p in all_proteins if freq[p] == N - 3])
near_n_minus_4 = sorted([p for p in all_proteins if freq[p] == N - 4])
NEAR_CORE = sorted(set(near_n_minus_1 + near_n_minus_2 +
                       near_n_minus_3 + near_n_minus_4))

print(f"\n=== Cascade family core / near-core ===")
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

fig, ax = plt.subplots(figsize=(13, 11))
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
    f"PairSeedCascade panel intersection matrix\n"
    f"Diagonal = panel size; off-diagonal = proteins shared between "
    f"two target_k union panels.\n"
    f"Strict core ({len(strict_core)} proteins, in EVERY of {N} union "
    f"panels): {', '.join(strict_core) if strict_core else '—'}.",
    fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "cascade_intersection_matrix.pdf", bbox_inches="tight")
plt.close()
print(f"\nSaved {OUT/'cascade_intersection_matrix.pdf'}")


# ---- presence tables ------------------------------------------------------

def build_matrix(proteins):
    """Cell = 1 if the protein is in this test's panel at ANY target_k
    in the grid (i.e., ever recruited by this test)."""
    M2 = np.zeros((len(TEST_ORDER), len(proteins)), dtype=int)
    for i, tid in enumerate(TEST_ORDER):
        ever_in_panel = set()
        for k in K_GRID:
            ever_in_panel.update(panels[(tid, k)])
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


row_labels = [TEST_LABELS[tid] for tid in TEST_ORDER]

if strict_core:
    M_core = build_matrix(strict_core)
    pd.DataFrame(np.where(M_core, "+", ""),
                  index=row_labels, columns=strict_core).to_csv(
        OUT / "cascade_table_A_core.csv")
    render_table(
        M_core, strict_core, row_labels,
        "cascade_table_A_core.pdf",
        f"Cascade Table A. Strict core ({len(strict_core)} proteins, "
        f"in every one of {N} cascade union panels) — '+' if "
        f"recruited by that cascade test at any target_k in the grid",
        "#3b66c4")
else:
    print("(strict core is empty — skipping Table A)")

if NEAR_CORE:
    M_near = build_matrix(NEAR_CORE)
    pd.DataFrame(np.where(M_near, "+", ""),
                  index=row_labels, columns=NEAR_CORE).to_csv(
        OUT / "cascade_table_B_near_core.csv")
    render_table(
        M_near, NEAR_CORE, row_labels,
        "cascade_table_B_near_core.pdf",
        f"Cascade Table B. Near-core ({len(NEAR_CORE)} proteins, "
        f"in {N-4}–{N-1} of {N} cascade union panels) — '+' if "
        f"recruited by that cascade test at any target_k in the grid",
        "#7b9bd0")
else:
    print("(near-core is empty — skipping Table B)")


# ---- console summary: which test recruits each strict-core protein -------

print(f"\nStrict-core protein → cascade test mapping (any target_k):")
for p in strict_core:
    tests_with_p = [tid for tid in TEST_ORDER
                    if any(p in panels[(tid, k)] for k in K_GRID)]
    print(f"  {p:>10}  →  {', '.join(tests_with_p)}")
