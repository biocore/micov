"""Drawing for `micov depth-plot`: layout helpers and the linear plot.

Everything here takes one genome's tables from `_depth.genome_statistics`.
x is ``position - 1``, so base p occupies [p - 1, p) and a bin's edges are
its half-open coordinates less one.
"""

import re

import matplotlib as mpl
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ._depth import orf_segments, overview_row_bp
from ._plot import GROUP_COLORS, group_style

#: Up to this many groups share one set of axes; more get a lane each.
OVERLAY_MAX_GROUPS = 3

#: The depth axis is linear below this and logarithmic above.
DEPTH_LINTHRESH = 2.0
DEPTH_LINSCALE = 0.6

#: Label lanes per strand; labels that fit in none are counted instead.
LABEL_LANES = 2

#: ORFs are drawn as arrows when the panel spans at most this many bases.
ARROW_MAX_BP = 150_000

#: The contrast colour scale is clipped at +-this (an 8-fold difference).
CONTRAST_LIMIT = 3.0

INK = "#262626"
MUTED = "#6b6b6b"
GRID = "#e8e8e8"
ORF_NEUTRAL = "#bdbdbd"
ORF_OTHER = "#d4d4d4"
ORF_OTHER_LABEL = "Other"

#: `--orf-color-by`'s top three values. Colour-by allows at most two groups,
#: which take the first two group colours, so the categories take the next
#: three; `test_depth_plot.OrfCategoryTests` checks they stay distinct.
ORF_CATEGORY_COLORS = GROUP_COLORS[2:5]

#: No contrast. The scale runs from the first group's colour through this to
#: the second's.
CONTRAST_MID = "#e6e6e6"

#: ORF boxes span these heights above the backbone (+) and below it (-); an
#: unstranded ORF straddles it, +-`UNSTRANDED_Y`.
ORF_Y = (0.13, 0.62)
UNSTRANDED_Y = 0.25

#: Applied with `matplotlib.rc_context` around each drawing, never globally:
#: micov is also a library, and its callers' rcParams are theirs.
STYLE = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
}
DPI = 150
FIGURE_WIDTH = 12
LABEL_FONTSIZE = 6.5
BAND_ALPHA = 0.18
DEPTH_LABEL = "depth, alignments\n(median, IQR, mean)"
PREVALENCE_LABEL = "prevalence\n(fraction of samples)"


def bp_formatter(span):
    """Return a tick formatter in Mb, kb or bp, whichever suits `span`."""
    scale, unit = ((1e6, "Mb") if span > 600_000
                   else (1e3, "kb") if span > 10_000 else (1, "bp"))

    def label(value, _=None):
        return "0" if value == 0 else f"{value / scale:,.6g} {unit}"

    return label


def nice_ticks(a, b, n=8):
    """About `n` ticks over [a, b] at 1, 2, 2.5 or 5 times a power of ten."""
    raw = (b - a) / n
    magnitude = 10 ** np.floor(np.log10(raw))
    step = min(s * magnitude for s in (1, 2, 2.5, 5, 10) if s * magnitude >= raw)
    return np.arange(np.ceil(a / step) * step, b + step / 2, step)


def symlog_ticks(ymax):
    """Ticks for the depth axis: linear to 2, then 1-2-5 per decade.

    Thinned to the decades when that would make more than seven.
    """
    if ymax < 5:
        return [0, 1, 2]
    ticks = [0, 2]
    decade = 1
    while 5 * decade <= ymax:
        ticks += [t for t in (5 * decade, 10 * decade, 20 * decade) if t <= ymax]
        decade *= 10
    if len(ticks) > 7:
        ticks = [0, 2] + [10**k for k in range(1, 20) if 10**k <= ymax]
    return ticks


def depth_ymax(tables):
    """Return the depth axis's top: the highest Q3 or mean, at least 2."""
    return max([DEPTH_LINTHRESH] + [float(max(t["q3"].max(), t["mean"].max()))
                                    for t in tables if len(t["q3"])])


def true_runs(mask):
    """Start and stop (half-open) indices of each run of True in `mask`."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def merge_spans(starts, stops, gap=0):
    """Merge spans that overlap, touch, or lie within `gap` of each other."""
    order = np.argsort(starts, kind="stable")
    merged = []
    for start, stop in zip(starts[order], stops[order], strict=True):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    merged = np.array(merged, dtype=np.int64).reshape(-1, 2)
    return merged[:, 0], merged[:, 1]


def clip_spans(starts, stops, a, b):
    """Return the parts of the spans that fall in [a, b)."""
    keep = (starts < b) & (stops > a)
    return np.maximum(starts[keep], a), np.minimum(stops[keep], b)


def parse_highlight(text):
    """Parse ``KEY=VALUE`` (equal) or ``KEY~REGEX`` (search) into a triple.

    The first `=` or `~` is the operator, so the value may contain either.
    """
    match = re.match(r"([^=~]+)([=~])(.+)", text)
    if match is None:
        raise ValueError(
            f"{text!r} is not KEY=VALUE or KEY~REGEX: KEY is a GFF column or "
            "attribute, = matches the value exactly and ~ searches it."
        )
    key, operator, value = match.groups()
    if operator == "~":
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError(
                f"{text!r} is not KEY=VALUE or KEY~REGEX: {value!r} is not a "
                f"regular expression ({error})."
            ) from None
    return key, operator, value


def orf_value(orfs, key):
    """Each ORF's `key`: its GFF column if it has one, else its attribute."""
    if key in ("type", "strand"):
        return orfs[key]
    return np.array([attributes.get(key) for attributes in orfs["attributes"]],
                    dtype=object)


def highlight_mask(orfs, highlights):
    """Which ORFs any of the parsed `highlights` matches."""
    mask = np.zeros(len(orfs["start"]), bool)
    for key, operator, value in highlights:
        if operator == "=":
            matches = [v == value for v in orf_value(orfs, key)]
        else:
            pattern = re.compile(value)
            matches = [v is not None and pattern.search(v) is not None
                       for v in orf_value(orfs, key)]
        mask |= np.array(matches, bool)
    return mask


def orf_categories(values):
    """Label each ORF with its value if among the three commonest, else Other.

    Ties go to the value that sorts first, so the result does not depend on
    the order of the ORFs. A missing or empty value is never a category.

    Returns
    -------
    labels : np.ndarray
    top : list
        The categories, commonest first.
    """
    present = [v for v in values if v]
    counts = {v: present.count(v) for v in set(present)}
    top = sorted(counts, key=lambda v: (-counts[v], v))[:len(ORF_CATEGORY_COLORS)]
    labels = np.array([v if v in top else ORF_OTHER_LABEL for v in values],
                      dtype=object)
    return labels, top


def contrast_colors(contrast):
    """Colour each ORF by its contrast, as RGBA.

    The first group's colour at -`CONTRAST_LIMIT` or below, the second's at
    +`CONTRAST_LIMIT` or above, and `CONTRAST_MID` at 0 or NaN.
    """
    # interpolated directly: a matplotlib colormap's 256-entry table has no
    # entry at exactly 0, so no contrast would be off-grey
    t = np.clip(np.nan_to_num(contrast) / CONTRAST_LIMIT, -1, 1)
    anchors = np.array([to_rgba(c) for c in
                        (GROUP_COLORS[0], CONTRAST_MID, GROUP_COLORS[1])])
    return np.stack([np.interp(t, (-1, 0, 1), anchors[:, k]) for k in range(4)],
                    axis=1)


def orf_polygons(starts, stops, strands, head_bp):
    """Vertices of each ORF's shape, in x = position - 1.

    A stranded ORF is an arrow pointing along its strand, its head `head_bp`
    long but at most half the ORF (0 draws a box); an unstranded one is a box
    across the backbone.
    """
    y0, y1 = ORF_Y
    mid = (y0 + y1) / 2
    shapes = []
    for start, stop, strand in zip(starts - 1, stops - 1, strands, strict=True):
        head = min(head_bp, (stop - start) / 2)
        if strand == "+":
            shape = [[start, y0], [stop - head, y0], [stop, mid],
                     [stop - head, y1], [start, y1]]
        elif strand == "-":
            shape = [[stop, -y0], [start + head, -y0], [start, -mid],
                     [start + head, -y1], [stop, -y1]]
        else:
            shape = [[start, -UNSTRANDED_Y], [stop, -UNSTRANDED_Y],
                     [stop, UNSTRANDED_Y], [start, UNSTRANDED_Y]]
        shapes.append(np.array(shape, dtype=float))
    return shapes


def assign_label_lanes(centers, widths, strands, lanes=LABEL_LANES):
    """Give each label the first lane on its side it fits in, left to right.

    An unstranded ORF's label goes above, with the + strand. A label that
    fits in no lane gets -1, to be counted rather than dropped silently.
    """
    lane_of = np.full(len(centers), -1)
    ends = {}
    for i in np.argsort(centers, kind="stable"):
        side = strands[i] == "-"
        for lane in range(lanes):
            if centers[i] - widths[i] / 2 > ends.get((side, lane), -np.inf):
                ends[(side, lane)] = centers[i] + widths[i] / 2
                lane_of[i] = lane
                break
    return lane_of


def plot_path(output, target_name, genome, variable, suffix=""):
    """``{output}.{target_name}.{genome}.{variable}.depth-plot[-suffix].png``."""
    tag = f"depth-plot-{suffix}" if suffix else "depth-plot"
    return f"{output}.{target_name}.{genome}.{variable}.{tag}.png"


def check_orf_mode(n_groups, color_by, contrast):
    """Refuse an ORF colouring the groups cannot support.

    Raises
    ------
    ValueError
        If both colourings are asked for, if `color_by` with three or more
        groups, or `contrast` without exactly two.
    """
    if color_by is not None and contrast:
        raise ValueError("--orf-color-by and --orf-contrast both colour the "
                         "ORFs; choose one.")
    if color_by is not None and n_groups > 2:
        raise ValueError(
            f"--orf-color-by needs at most two groups, and there are {n_groups}: "
            "three category colours beside three or more group colours cannot "
            "all stay distinct to a colour-blind reader. Use --highlight to "
            "mark the ORFs of interest instead."
        )
    if contrast and n_groups != 2:
        raise ValueError(f"--orf-contrast compares exactly two groups, and "
                         f"there are {n_groups}.")


def orf_track(orfs, highlight, color_by=None, contrast=None):
    """Decide how each ORF is drawn.

    Neutral grey by default, with highlighted ORFs in ink. Coloured by
    `color_by`'s top three values, or by `contrast`, highlighted ORFs keep
    their colour and gain an ink outline instead.

    Returns
    -------
    dict
        `start`, `stop`, `strand`, `label` and `highlight` per ORF, an RGBA
        `fill` and a boolean `outline`, and the `legend`: None,
        ``("categories", top)`` or ``("contrast",)``.
    """
    if color_by is not None:
        labels, top = orf_categories(orf_value(orfs, color_by))
        colour_of = dict(zip(top, ORF_CATEGORY_COLORS, strict=False))
        fill = np.array([to_rgba(colour_of.get(v, ORF_OTHER)) for v in labels])
        outline, legend = highlight, ("categories", top)
    elif contrast is not None:
        fill = contrast_colors(contrast)
        outline, legend = highlight, ("contrast",)
    else:
        fill = np.array([to_rgba(INK if on else ORF_NEUTRAL) for on in highlight])
        outline, legend = np.zeros(len(highlight), bool), None
    return {"start": orfs["start"], "stop": orfs["stop"],
            "strand": orfs["strand"], "label": orfs["label"],
            "highlight": highlight, "fill": fill.reshape(-1, 4),
            "outline": outline, "legend": legend}



def linear_plot(path, *, title, length, sizes, overview, regions=(), track=None):
    """Draw one genome's overview, in rows of `_depth.overview_row_bp`, to `path`.

    Each row has depth above (one axes, or a lane per group past
    `OVERLAY_MAX_GROUPS`), the ORFs on the genome line, and breadth below:
    union segments, then prevalence hanging down. Every row and lane shares
    one depth scale, and the last row is masked past the genome's end.

    Parameters
    ----------
    length : int
    sizes : dict
        Each group's number of samples.
    overview : dict
        `_depth.genome_statistics`' overview bins.
    regions : list of (int, int)
        Detail regions, [start, stop), shaded where they fall.
    track : dict, optional
        `orf_track`'s output.

    Raises
    ------
    ValueError
        Before drawing anything, if a column of `overview` has missing values
        or its bins do not tile the same span for every group.
    """
    groups = sorted(sizes)
    _check_bins(overview, groups)
    ymax = depth_ymax([overview])
    width = overview_row_bp(length)
    rows = -(-length // width)
    with mpl.rc_context(STYLE):
        heights = [_panel_height(groups, track)] * rows
        fig = Figure(figsize=(FIGURE_WIDTH, sum(heights) + 0.45 * rows))
        grid = fig.add_gridspec(rows, 1, height_ratios=heights, hspace=0.35)
        for row in range(rows):
            x0, x1 = row * width, (row + 1) * width
            panel = _draw_panel(fig, grid[row], row, groups, sizes, overview,
                                x0, x1, length, ymax, track, legend=row == 0)
            for i, (start, stop) in enumerate(regions):
                a, b = max(start - 1, x0), min(stop - 1, x1)
                if a < b:
                    for ax in panel:
                        ax.axvspan(a, b, color=INK, alpha=0.09, lw=0, zorder=0,
                                   gid=f"region:{i}")
            if x1 > length:
                for ax in panel:
                    ax.axvspan(length, x1, color="white", lw=0, zorder=5,
                               gid="past-end")
            if row == 0:
                panel[0].set_title(title, loc="left", color=INK)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")


def detail_plot(path, *, title, length, sizes, table, track=None):
    """Draw one detail region, the span of `table`'s bins, to `path`.

    As one row of `linear_plot`, at the region's own bins and depth scale,
    with ORFs as arrows when the region is at most `ARROW_MAX_BP` long.
    """
    groups = sorted(sizes)
    _check_bins(table, groups)
    x0, x1 = table["bin_start"][0] - 1, table["bin_stop"][-1] - 1
    with mpl.rc_context(STYLE):
        fig = Figure(figsize=(FIGURE_WIDTH, _panel_height(groups, track)))
        grid = fig.add_gridspec(1, 1)
        panel = _draw_panel(fig, grid[0], 0, groups, sizes, table, x0, x1,
                            length, depth_ymax([table]), track, legend=True)
        panel[0].set_title(title, loc="left", color=INK)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")


def _check_bins(table, groups):
    """Refuse bins that cannot be drawn faithfully, before any figure opens."""
    for key, values in table.items():
        if np.ma.isMaskedArray(values):
            raise ValueError(f"The bins' {key} has missing values; they would "
                             "be drawn as gaps.")
    edges = None
    for group in groups:
        own = table["group"] == group
        starts, stops = table["bin_start"][own], table["bin_stop"][own]
        if not np.array_equal(stops[:-1], starts[1:]):
            raise ValueError(f"The bins of {group} do not tile their span.")
        if edges is not None and not (np.array_equal(edges[0], starts)
                                      and np.array_equal(edges[1], stops)):
            raise ValueError("The groups' bins differ.")
        edges = starts, stops


def _panel_height(groups, track):
    """Inches for one row: depth, the ORF track, then breadth."""
    lanes = len(groups) > OVERLAY_MAX_GROUPS
    labels = track is not None and track["highlight"].any()
    depth, breadth = (0.9 * len(groups), 0.45 * len(groups)) if lanes else (2.0, 1.0)
    return depth + (1.6 if labels else 0.55) + breadth


def _draw_panel(fig, slot, row, groups, sizes, table, x0, x1, length, ymax,
                track, legend):
    """Draw one row, [x0, x1), into `slot`; return its axes, top to bottom."""
    lanes = len(groups) > OVERLAY_MAX_GROUPS
    names = groups if lanes else [None]
    labels = track is not None and track["highlight"].any()
    lane_height = (0.9, 0.45) if lanes else (2.0, 1.0)
    ratios = ([lane_height[0]] * len(names) + [1.6 if labels else 0.55]
              + [lane_height[1]] * len(names))
    grid = slot.subgridspec(len(ratios), 1, height_ratios=ratios,
                            hspace=0.3 if lanes else 0.0)
    axes = [fig.add_subplot(cell) for cell in grid]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    depth_axes = axes[:len(names)]
    orf_ax = axes[len(names)]
    breadth_axes = axes[len(names) + 1:]
    suffix = [""] if not lanes else [f":{g}" for g in groups]

    keep = (table["bin_start"] - 1 < x1) & (table["bin_stop"] - 1 > x0)
    shown = {key: values[keep] for key, values in table.items()}
    for g, group in enumerate(groups):
        own = {key: values[shown["group"] == group]
               for key, values in shown.items()}
        edges = np.append(own["bin_start"], own["bin_stop"][-1:]) - 1
        lane = g if lanes else 0
        _draw_depth(depth_axes[lane], own, edges, group, g)
        _draw_breadth(breadth_axes[lane], own, edges, group, g,
                      0 if lanes else g, 1 if lanes else len(groups))

    for ax, name in zip(depth_axes, suffix, strict=True):
        ax.set_gid(f"depth:{row}{name}")
        ax.set_yscale("symlog", linthresh=DEPTH_LINTHRESH,
                      linscale=DEPTH_LINSCALE)
        ax.set_ylim(0, ymax * 1.15)
        ticks = symlog_ticks(ymax)
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    for ax, name in zip(breadth_axes, suffix, strict=True):
        ax.set_gid(f"breadth:{row}{name}")
        ticks = [0, 1] if lanes else [0, 0.5, 1]
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    orf_ax.set_gid(f"orfs:{row}")
    orf_ax.axis("off")

    if lanes:
        for ax, group in zip(depth_axes, groups, strict=True):
            ax.text(0.006, 0.95, f"{group}  n={sizes[group]}",
                    transform=ax.transAxes, fontsize=8, color=INK, va="top",
                    gid=f"lane:{group}",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})
        depth_axes[len(groups) // 2].set_ylabel(DEPTH_LABEL)
        breadth_axes[len(groups) // 2].set_ylabel(PREVALENCE_LABEL)
    else:
        depth_axes[0].set_ylabel(DEPTH_LABEL)
        breadth_axes[0].set_ylabel(PREVALENCE_LABEL)
        if legend:
            handles = [Line2D([], [], color=group_style(g)[0],
                              ls=group_style(g)[1], lw=2.2)
                       for g in range(len(groups))]
            handles.append(Line2D([], [], color=MUTED, lw=1.0, ls=":"))
            depth_axes[0].legend(
                handles, [f"{g}  n={sizes[g]}" for g in groups] + ["group mean"],
                loc="upper right", frameon=False, fontsize=8, handlelength=1.6,
                ncols=len(groups) + 1,
            )

    for ax in axes[:-1]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)
    axes[-1].set_xticks(nice_ticks(x0, x1))
    axes[-1].xaxis.set_major_formatter(bp_formatter(x1 - x0))
    axes[0].set_xlim(x0, x1)

    if track is not None:
        bp_per_pt = (x1 - x0) / (orf_ax.get_position().width
                                 * fig.get_figwidth() * 72)
        _draw_orfs(orf_ax, track, x0, x1, length, bp_per_pt, labels)
        _draw_highlights(depth_axes + breadth_axes, track, x0, x1, bp_per_pt)
        if legend:
            _draw_orf_legend(orf_ax, track, groups)
    else:
        orf_ax.axhline(0, color=INK, lw=1.5, zorder=3, gid="backbone")
        orf_ax.set_ylim(-0.9, 0.9)
    return axes


def _draw_depth(ax, own, edges, group, index):
    colour, linestyle = group_style(index)
    ax.stairs(own["q3"], edges, baseline=own["q1"], fill=True, color=colour,
              alpha=BAND_ALPHA, lw=0, gid=f"iqr:{group}")
    ax.stairs(own["median"], edges, color=colour, lw=1.3, ls=linestyle,
              gid=f"median:{group}")
    ax.stairs(own["mean"], edges, color=colour, lw=0.9, ls=":",
              gid=f"mean:{group}")


def _draw_breadth(ax, own, edges, group, index, slot, slots):
    """Union as segments in a strip above 0, prevalence hanging from 0."""
    colour = group_style(index)[0]
    gap = 0.075
    starts, stops = true_runs(own["union"])
    ax.hlines(np.full(len(starts), -(slot + 0.5) * gap), edges[starts],
              edges[stops], color=colour, lw=2.0, gid=f"union:{group}")
    ax.stairs(own["prevalence"], edges, color=colour, lw=1.1,
              gid=f"prevalence:{group}")
    ax.set_ylim(1.04, -slots * gap - 0.03)


def _draw_orfs(ax, track, x0, x1, length, bp_per_pt, labels):
    """Draw the genome line, its ORFs, and the highlighted ORFs' labels.

    An ORF across a circular genome's origin is drawn as its two parts,
    boxes rather than arrows, since neither part holds both ends.
    """
    ax.axhline(0, color=INK, lw=1.5, zorder=3, gid="backbone")
    head = 4 * bp_per_pt if x1 - x0 <= ARROW_MAX_BP else 0
    index, starts, stops = orf_segments(track["start"], track["stop"], length)
    split = np.bincount(index, minlength=len(track["start"]))[index] > 1
    shown = (starts - 1 < x1) & (stops - 1 > x0)
    for strand in ("+", "-", "."):
        on = shown & (track["strand"][index] == strand)
        if not on.any():
            continue
        shapes = (orf_polygons(starts[on & ~split], stops[on & ~split],
                               track["strand"][index][on & ~split], head)
                  + orf_polygons(starts[on & split], stops[on & split],
                                 track["strand"][index][on & split], 0))
        order = np.concatenate([np.flatnonzero(on & ~split),
                                np.flatnonzero(on & split)])
        outline = track["outline"][index][order]
        ax.add_collection(PolyCollection(
            shapes, facecolors=track["fill"][index][order],
            edgecolors=[INK if o else "none" for o in outline],
            linewidths=0.8, zorder=2, gid=f"orfs:{strand}",
        ))
    ax.set_ylim(-2.4, 2.4) if labels else ax.set_ylim(-0.9, 0.9)
    if not labels:
        return
    on = track["highlight"] & (track["start"] - 1 < x1) & (track["stop"] - 1 > x0)
    names = track["label"][on]
    centers = (np.maximum(track["start"][on] - 1, x0)
               + np.minimum(track["stop"][on] - 1, x1)) / 2
    widths = np.array([(len(n) * LABEL_FONTSIZE * 0.62 + 5) * bp_per_pt
                       for n in names])
    strands = track["strand"][on]
    lanes = assign_label_lanes(centers, widths, strands)
    for name, center, strand, lane in zip(names, centers, strands, lanes,
                                          strict=True):
        if lane < 0:
            continue
        sign = -1 if strand == "-" else 1
        ax.annotate(name, (center, sign * ORF_Y[1]),
                    (center, sign * (ORF_Y[1] + 0.33 + lane * 0.62)),
                    fontsize=LABEL_FONTSIZE, color=INK, ha="center",
                    va="top" if sign < 0 else "bottom", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "lw": 0.4, "color": MUTED,
                                "shrinkA": 0, "shrinkB": 0},
                    gid="label")
    dropped = int((lanes < 0).sum())
    if dropped:
        ax.text(1.0, 0.0, f"+{dropped} unlabelled", transform=ax.transAxes,
                ha="right", va="top", fontsize=LABEL_FONTSIZE, color=MUTED,
                gid="unlabelled")


def _draw_highlights(axes, track, x0, x1, bp_per_pt):
    """Shade depth and breadth faintly under each highlighted ORF."""
    on = track["highlight"]
    starts, stops = merge_spans(track["start"][on] - 1, track["stop"][on] - 1,
                                gap=bp_per_pt)
    starts, stops = clip_spans(starts, stops, x0, x1)
    for ax in axes:
        for start, stop in zip(starts, stops, strict=True):
            ax.axvspan(start, stop, color=INK, alpha=0.06, lw=0, zorder=0,
                       gid="highlight")


def _draw_orf_legend(ax, track, groups):
    """Show the categories' colours, or the contrast's scale."""
    if track["legend"] is None:
        return
    if track["legend"][0] == "categories":
        top = track["legend"][1]
        colours = [*ORF_CATEGORY_COLORS[:len(top)], ORF_OTHER]
        # in the right margin, beside the ORFs: above them it would sit on
        # the depth panel, which has no gap above the track
        legend = ax.legend([Patch(color=c) for c in colours],
                           [*top, ORF_OTHER_LABEL], loc="center left",
                           frameon=False, fontsize=7, bbox_to_anchor=(1.005, 0.5),
                           handlelength=1.0, borderaxespad=0)
        legend.set_gid("orf-legend")
        return
    scale = ax.inset_axes([1.01, 0.42, 0.12, 0.16])
    scale.set_gid("contrast-scale")
    ramp = np.linspace(-CONTRAST_LIMIT, CONTRAST_LIMIT, 256)
    scale.imshow(contrast_colors(ramp)[None, :, :], aspect="auto")
    fold = f"\u00d7{2 ** CONTRAST_LIMIT:g}"
    scale.set_xticks([0, 127.5, 255],
                     [f"{groups[0]} {fold}", "equal", f"{groups[1]} {fold}"],
                     fontsize=6.5)
    scale.tick_params(length=0, pad=1)
    scale.set_yticks([])
    for spine in scale.spines.values():
        spine.set_visible(False)
