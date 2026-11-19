"""Common matplotlib styling for all manuscript figures.

Centralises font sizes, the classifier-family color palette, marker
shapes, figure dimensions, and the Pareto-plot background-zone
shading so that all `fig_*.py` scripts produce visually consistent
PDFs.

The classifier-family colors and markers (`COLORS`, `MARKERS`) are
the only style elements that **must** stay consistent across figures
— they let the reader recognise the same classifier on different
plots. Everything else (font sizes, line widths, layout) is a sane
default that individual figure scripts may override locally if a
specific figure needs different formatting.

Usage
-----
    import fig_style

    fig_style.apply()                              # set rcParams once
    fig, ax = plt.subplots(figsize=fig_style.SIZE["manuscript_wide"])

    ax.scatter(xs, ys,
               c=fig_style.COLORS["cascade"],
               marker=fig_style.MARKERS["cascade"][0],
               s=fig_style.MARKERS["cascade"][1],
               label="PairSeedCascade family")

    fig_style.add_pareto_zones(ax, ymin=40, ymax=98)
"""

import matplotlib.pyplot as plt


# -------------------------- font sizes ------------------------------------

FONT = {
    "axis_label": 14,   # x / y label text
    "tick_label": 12,   # tick labels
    "title":      14,   # axis-level title (use sparingly; usually caption)
    "legend":     11,   # legend entries
    "annotation": 10,   # in-figure text (point labels, zone names)
}


# -------------------------- classifier-family colors ---------------------

COLORS = {
    # PairSeed-no-cascade family — green crosses
    "pairseed_no_cascade": "#1f7a3c",
    # PairSeed-cascade family — blue circles
    "cascade":             "#3b66c4",
    # Alvez-mimicked family — yellow / orange diamonds
    "alvez_mimicked":      "#e0843f",
    # Alvez 2023 published classifier — purple star
    "alvez_star":          "#9b3f9b",
    # Multinomial elastic-net demo point — red plus
    "elastic_net":         "#e63946",
    # Neutral / annotation
    "grid":                "#cccccc",
    "axis":                "#444444",
    "text":                "#222222",
}


# -------------------------- marker shapes & sizes ------------------------
# (marker_glyph, marker_size_in_points_squared) — matches scatter() s=...

MARKERS = {
    "pairseed_no_cascade": ("+", 130),
    "cascade":             ("o", 110),
    "alvez_mimicked":      ("D", 100),
    "alvez_star":          ("*", 230),
    "elastic_net":         ("P", 180),
}


# -------------------------- Pareto-plot background zones -----------------
# Vertical bands along the protein-count axis. Adjust ZONE boundaries to
# match the manuscript's "screening / clinical / research-grade /
# high-protein" framing.

ZONE = {
    "screening":    (0,   20),
    "clinical":     (20,  50),
    "research":     (50,  110),
    "high_protein": (110, 1500),
}
ZONE_COLORS = {
    "screening":    "#fbe2e2",
    "clinical":     "#dff0db",
    "research":     "#e5dbe9",
    "high_protein": "#ecedf2",
}
ZONE_LABELS = {
    "screening":    "screening\n(<20 p)",
    "clinical":     "clinical sweet spot\n(20–50 p)",
    "research":     "research-grade\n(50–110 p)",
    "high_protein": "high-protein regime",
}


# -------------------------- figure dimensions (inches) --------------------

SIZE = {
    "single_column":   (3.5, 2.8),
    "double_column":   (7.2, 4.5),
    "manuscript_wide": (10.0, 6.0),  # used for Pareto with annotations
    "supplementary":   (8.0, 6.0),
}


# -------------------------- entry points ----------------------------------

def apply():
    """Apply the common rcParams to matplotlib globally.

    Call once at the top of each `fig_*.py` script, before any
    `plt.subplots()`. Sets sans-serif font, hides top/right spines,
    embeds TrueType (PDF type 42) so Illustrator/Affinity can edit
    text post-hoc, and saves figures with tight bounding box.
    """
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize":    FONT["axis_label"],
        "axes.titlesize":    FONT["title"],
        "xtick.labelsize":   FONT["tick_label"],
        "ytick.labelsize":   FONT["tick_label"],
        "legend.fontsize":   FONT["legend"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "axes.edgecolor":    COLORS["axis"],
        "savefig.bbox":      "tight",
        "savefig.format":    "pdf",
        "pdf.fonttype":      42,
    })


def add_pareto_zones(ax, ymin=40, ymax=98, x_axis_max=None,
                     show_labels=True):
    """Draw the four coloured background zones on a Pareto-style plot.

    Call after setting xlim / ylim on `ax`. Boundaries are taken from
    `ZONE`; colors from `ZONE_COLORS`; labels from `ZONE_LABELS`.

    Parameters
    ----------
    ax : matplotlib Axes
    ymin, ymax : float
        y-range of the plot (used to anchor the zone labels at the top).
    x_axis_max : float, optional
        Right edge for the "high_protein" band. Defaults to the current
        ax.xlim()[1] so the band fills the visible plot.
    show_labels : bool
        Draw the zone names at the top of each band.
    """
    if x_axis_max is None:
        x_axis_max = ax.get_xlim()[1]
    for name, (x0, x1) in ZONE.items():
        x1c = min(x1, x_axis_max)
        if x0 >= x_axis_max:
            continue
        ax.axvspan(x0, x1c, color=ZONE_COLORS[name],
                   alpha=0.55, zorder=0)
        if show_labels:
            x_mid = 0.5 * (x0 + x1c)
            ax.text(x_mid, ymax * 0.985, ZONE_LABELS[name],
                    ha="center", va="top",
                    fontsize=FONT["annotation"],
                    color=COLORS["text"],
                    bbox=dict(facecolor="white", alpha=0.6,
                              edgecolor="none", pad=2))
