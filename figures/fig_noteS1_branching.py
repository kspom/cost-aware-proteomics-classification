"""Schematic of the PairSeed branching for one one-vs-rest test
(Supplementary Note S1): one seed pair, then at each step up to 2 next
proteins and a "..." branch for the remaining alternatives.

The tree is taken from the recorded search (the same rules as
figures/fig_noteS1_sankey_input.py): next proteins are ordered by the number
of completed branches below them; a protein set already shown in another
order, or a protein already shown elsewhere, is folded into "...".

Usage (from the repository root):
    python figures/fig_noteS1_branching.py MYEL [MAX_DEPTH] [SHOW]
Input:  paper_figures/search_trace_<CANCER>.json (analysis/pairseed_search_trace.py)
Output: paper_figures/fig_noteS1_branching_<CANCER>.pdf/.png
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

c = sys.argv[1] if len(sys.argv) > 1 else "MYEL"
MAX_DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SHOW = int(sys.argv[3]) if len(sys.argv) > 3 else 2
d = json.load(open(Path("paper_figures") / f"search_trace_{c}.json"))
sel = set(d["selected"]["panel"])
seed = next(s for s in d["seeds"] if set(s["pair"]) <= sel)
panels = [tuple(b["panel"]) for b in seed["branches"]]

# ---- tree -------------------------------------------------------------
shown, used = set(), set()


def build(prefix, label):
    """Node = dict(label, children); None if only repeats lie below."""
    node = {"label": label, "children": []}
    k = len(prefix)
    if k >= MAX_DEPTH:
        return node
    nxt = Counter(p[k] for p in panels if len(p) > k and p[:k] == prefix)
    if not nxt:
        return node                                   # the branch ends here
    hidden = 0
    for prot, _ in nxt.most_common():
        key = frozenset(prefix + (prot,))
        if key in shown or prot in used or len(node["children"]) >= SHOW:
            hidden += 1
            continue
        shown.add(key)
        used.add(prot)
        ch = build(prefix + (prot,), "+" + prot)
        if ch is None:
            hidden += 1
            continue
        node["children"].append(ch)
    if not node["children"]:
        return None
    if hidden:
        node["children"].append({"label": "...", "children": [], "dots": True})
    return node


used.update(seed["pair"])
root = build(tuple(seed["pair"]), " +\n".join(seed["pair"]))

# ---- layout: leaves get unit slots, inner nodes are centred ----------
W = 0.62                     # ribbon width (slot = 1)
BOX_W, GAP = 1.55, 1.25      # node box width, horizontal gap between columns
slot = [0]


def place(n, depth, branch):
    n["depth"], n["branch"] = depth, branch
    for i, ch in enumerate(n["children"]):
        place(ch, depth + 1, i if depth == 0 else branch)
    if n["children"]:
        n["y"] = np.mean([ch["y"] for ch in n["children"]])
    else:
        n["y"] = slot[0]
        slot[0] += 1
    n["h"] = W * max(1, len(n["children"]))


place(root, 0, -1)
n_leaf = slot[0]
COL = ["#1b6ca8", "#d1495b", "#9a9a9a"]


def color(n):
    if n.get("dots"):
        return "#bdbdbd"
    return "#444444" if n["branch"] < 0 else COL[min(n["branch"], 1)]


def x0(n):
    return n["depth"] * (BOX_W + GAP)


fig, ax = plt.subplots(figsize=(9, 0.5 * n_leaf + 1.2))


def draw(n):
    k = len(n["children"])
    for i, ch in enumerate(n["children"]):
        ys = n["y"] - n["h"] / 2 + W * i              # ribbon slice in parent
        ye = ch["y"] - W / 2                          # ribbon end at child
        xa, xb = x0(n) + BOX_W, x0(ch)
        xm = (xa + xb) / 2
        verts = [(xa, ys), (xm, ys), (xm, ye), (xb, ye),
                 (xb, ye + W), (xm, ye + W), (xm, ys + W), (xa, ys + W), (xa, ys)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=color(ch), ec="none",
                               alpha=0.35, zorder=1))
        draw(ch)
    col = color(n)
    ax.add_patch(FancyBboxPatch((x0(n), n["y"] - n["h"] / 2), BOX_W, n["h"],
                                boxstyle="round,pad=0,rounding_size=0.12",
                                fc="white", ec=col, lw=1.6, zorder=2))
    ax.text(x0(n) + BOX_W / 2, n["y"], n["label"], ha="center", va="center",
            fontsize=11, color="#555" if n.get("dots") else "black",
            fontweight="bold" if n["depth"] == 0 else "normal", zorder=3)


draw(root)
depth_max = max(len(p) for p in panels if set(seed["pair"]) <= set(p))
heads = ["Seed pair"] + [f"{i}rd protein" if i == 3 else f"{i}th protein"
                         for i in range(3, MAX_DEPTH + 1)]
for i, h in enumerate(heads):
    ax.text(i * (BOX_W + GAP) + BOX_W / 2, -0.9, h, ha="center", va="center",
            fontsize=11, color="#555")
ax.set_xlim(-0.1, (len(heads) - 1) * (BOX_W + GAP) + BOX_W + 0.1)
ax.set_ylim(n_leaf - 0.4, -1.3)
ax.axis("off")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(Path("paper_figures") / f"fig_noteS1_branching_{c}.{ext}", dpi=150)
print("Saved paper_figures/fig_noteS1_branching_%s.pdf/png" % c)
