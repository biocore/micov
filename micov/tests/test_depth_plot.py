"""Tests for `depth-plot`'s drawing.

The pure helpers -- ticks, spans, highlights, ORF colours and shapes, label
lanes -- are tested on their own. The drawings are tested by capturing the
figure at `savefig` and reading back the artists, which carry gids, so a test
asserts what was drawn rather than how a PNG looks.
"""

import itertools
import random
import shutil
import unittest
from tempfile import mkdtemp
from typing import ClassVar
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from micov._depth import ROW_BP, display_bin_edges, overview_bin_bp
from micov._depth_plot import (
    CONTRAST_MID,
    DEPTH_LINTHRESH,
    INK,
    MAX_ARC,
    ORF_CATEGORY_COLORS,
    ORF_NEUTRAL,
    ORF_OTHER,
    ORF_OTHER_LABEL,
    ORF_Y,
    UNSTRANDED_Y,
    assign_label_lanes,
    bp_formatter,
    check_orf_mode,
    circular_layout,
    circular_plot,
    clip_spans,
    contrast_colors,
    densify,
    depth_ymax,
    detail_plot,
    highlight_mask,
    linear_plot,
    merge_spans,
    nice_ticks,
    orf_categories,
    orf_polygons,
    orf_track,
    orf_value,
    parse_highlight,
    place_ring_labels,
    plot_path,
    polar_steps,
    radial_symlog,
    radial_text,
    spread_angles,
    symlog_ticks,
    tangential_text,
    theta,
    true_runs,
)
from micov._plot import GROUP_COLORS, group_style
from micov.tests.test_plot import (
    CVD_MIN_DELTA_E,
    MACHADO,
    NORMAL_MIN_DELTA_E,
    delta_e,
)


def orfs_of(*rows):
    """ORFs as `_depth.genome_orfs` returns them, plus `attributes`.

    Each row is (start, stop, strand, label, attributes).
    """
    def text(values):
        return np.array(values, dtype=object)

    return {
        "orf_id": text([f"o{i}" for i in range(len(rows))]),
        "label": text([r[3] for r in rows]),
        "type": text([r[4].pop("type", "CDS") for r in rows]),
        "start": np.array([r[0] for r in rows], np.int64),
        "stop": np.array([r[1] for r in rows], np.int64),
        "strand": text([r[2] for r in rows]),
        "attributes": text([r[4] for r in rows]),
    }


class AxisTests(unittest.TestCase):
    def test_bp_labels_take_the_unit_of_the_span(self):
        for span, value, label in (
            (2_000_000, 1_500_000, "1.5 Mb"),
            (2_000_000, 2_000_000, "2 Mb"),
            (2_000_000, 0, "0"),
            (85_000, 1_920_000, "1,920 kb"),
            (85_000, 1_902_500, "1,902.5 kb"),
            (500, 1_250, "1,250 bp"),
        ):
            with self.subTest(span=span, value=value):
                self.assertEqual(bp_formatter(span)(value), label)

    def test_nice_ticks(self):
        self.assertEqual(
            nice_ticks(0, 2_000_000).tolist(),
            list(range(0, 2_000_001, 250_000)),
        )
        self.assertEqual(
            nice_ticks(1000, 1500).tolist(), list(range(1000, 1501, 100))
        )

    def test_symlog_ticks(self):
        """Linear to 2, then 1-2-5 decades, thinned to decades if crowded."""
        for ymax, ticks in ((2, [0, 1, 2]), (4, [0, 1, 2]),
                            (60, [0, 2, 5, 10, 20, 50]), (500, [0, 2, 10, 100]),
                            (5000, [0, 2, 10, 100, 1000])):
            with self.subTest(ymax=ymax):
                self.assertEqual(symlog_ticks(ymax), ticks)

    def test_depth_ymax_is_the_highest_q3_or_mean(self):
        """The mean can pass Q3 where one sample is deep; both must show."""
        def table(q3, mean):
            return {"q3": np.array(q3, float), "mean": np.array(mean, float)}

        self.assertEqual(
            depth_ymax([table([1, 7], [2, 3]), table([4], [9.5])]), 9.5
        )

    def test_depth_ymax_never_collapses(self):
        """Nothing aligned still needs the linear part of the axis."""
        table = {"q3": np.zeros(3), "mean": np.zeros(3)}
        self.assertEqual(depth_ymax([table]), 2.0)


class SpanTests(unittest.TestCase):
    def test_true_runs(self):
        for mask, runs in (
            ([False, True, True, False, True], ([1, 4], [3, 5])),
            ([False, False], ([], [])),
            ([True, True, True], ([0], [3])),
        ):
            with self.subTest(mask=mask):
                starts, stops = true_runs(np.array(mask))
                self.assertEqual((starts.tolist(), stops.tolist()), runs)

    def test_merge_spans(self):
        starts, stops = np.array([10, 1, 3, 8]), np.array([12, 5, 8, 8])
        merged = merge_spans(starts, stops)
        self.assertEqual([m.tolist() for m in merged], [[1, 10], [8, 12]])
        merged = merge_spans(starts, stops, gap=2)
        self.assertEqual([m.tolist() for m in merged], [[1], [12]])

    def test_clip_spans(self):
        """[0, 4) ends where [4, 10) starts, so has no part in it."""
        clipped = clip_spans(np.array([1, 8, 20, 0]), np.array([5, 12, 30, 4]),
                             4, 10)
        self.assertEqual([c.tolist() for c in clipped], [[4, 8], [5, 10]])


class HighlightTests(unittest.TestCase):
    ORFS = staticmethod(lambda: orfs_of(
        (1, 301, "+", "dnaA", {"ID": "gc_1", "gene": "dnaA"}),
        (901, 1101, "-", "GC_0002", {"ID": "gc_2", "product": "5' nucleotidase"}),
        (1201, 1401, "+", "rrsA", {"ID": "gc_3", "type": "rRNA"}),
        (2601, 2801, "+", "nTest", {"ID": "gc_5", "product": "phage spliced"}),
    ))

    def test_parse(self):
        for text, parsed in (
            ("product=5' nucleotidase", ("product", "=", "5' nucleotidase")),
            ("product~phage", ("product", "~", "phage")),
            ("product~a=b", ("product", "~", "a=b")),
            ("type=rRNA", ("type", "=", "rRNA")),
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_highlight(text), parsed)

    def test_parse_rejects(self):
        for text in ("phage", "=x", "product=", "product~("):
            with self.subTest(text=text), self.assertRaisesRegex(
                ValueError, "KEY=VALUE"
            ):
                parse_highlight(text)

    def test_a_gff_column_before_an_attribute(self):
        orfs = self.ORFS()
        self.assertEqual(orf_value(orfs, "type").tolist(),
                         ["CDS", "CDS", "rRNA", "CDS"])
        self.assertEqual(orf_value(orfs, "product").tolist(),
                         [None, "5' nucleotidase", None, "phage spliced"])

    def test_any_expression_highlights(self):
        orfs = self.ORFS()
        mask = highlight_mask(
            orfs, [parse_highlight("product~phage"), parse_highlight("type=rRNA")]
        )
        self.assertEqual(mask.tolist(), [False, False, True, True])

    def test_equals_is_exact_and_tilde_searches(self):
        orfs = self.ORFS()
        self.assertEqual(
            highlight_mask(orfs, [parse_highlight("product=phage")]).tolist(),
            [False] * 4,
        )
        self.assertEqual(
            highlight_mask(orfs, [parse_highlight("product~5' nuc")]).tolist(),
            [False, True, False, False],
        )

    def test_nothing_highlighted(self):
        self.assertEqual(highlight_mask(self.ORFS(), []).tolist(), [False] * 4)


class OrfCategoryTests(unittest.TestCase):
    VALUES: ClassVar[list] = ["a", "b", "a", "c", "d", "b", "a", None, "c", "e",
                              ""]

    def test_the_three_commonest_values_then_other(self):
        """Counts a 3, b 2, c 2, d 1, e 1: ties go to the earlier value, and a
        missing value is never a category."""
        labels, top = orf_categories(np.array(self.VALUES, dtype=object))
        self.assertEqual(top, ["a", "b", "c"])
        self.assertEqual(
            labels.tolist(),
            ["a", "b", "a", "c", ORF_OTHER_LABEL, "b", "a", ORF_OTHER_LABEL, "c",
             ORF_OTHER_LABEL, ORF_OTHER_LABEL],
        )

    def test_independent_of_row_order(self):
        rng = random.Random(3)
        for _ in range(20):
            order = list(range(len(self.VALUES)))
            rng.shuffle(order)
            labels, top = orf_categories(
                np.array([self.VALUES[i] for i in order], dtype=object)
            )
            self.assertEqual(top, ["a", "b", "c"])
            expected, _ = orf_categories(np.array(self.VALUES, dtype=object))
            self.assertEqual(labels.tolist(), [expected[i] for i in order])

    def test_a_tie_for_a_place_goes_to_the_value_sorting_first(self):
        """Twenty values seen once each: the three that sort first, whatever
        order the ORFs, or a set of their values, hold them in. String hashes
        change between runs, so a tie left to a set would change too."""
        values = [f"COG{i:02d}" for i in range(20)]
        random.Random(5).shuffle(values)
        _, top = orf_categories(np.array(values, dtype=object))
        self.assertEqual(top, ["COG00", "COG01", "COG02"])

    def test_category_colours_stay_distinct(self):
        """With two groups (colour-by refuses three or more) the categories
        take the palette slots the groups leave free, plus a grey Other."""
        self.assertEqual(ORF_CATEGORY_COLORS, GROUP_COLORS[2:5])
        colours = (*ORF_CATEGORY_COLORS, ORF_OTHER)
        for a, b in itertools.combinations(colours, 2):
            for deficiency in MACHADO:
                with self.subTest(pair=(a, b), vision=deficiency):
                    self.assertGreaterEqual(
                        delta_e(a, b, deficiency), CVD_MIN_DELTA_E
                    )
            with self.subTest(pair=(a, b), vision="normal"):
                self.assertGreaterEqual(delta_e(a, b), NORMAL_MIN_DELTA_E)


class ContrastColourTests(unittest.TestCase):
    def test_the_scale_runs_from_the_first_group_to_the_second(self):
        """Clipped at +-3 (8-fold); no contrast is the neutral midpoint."""
        colours = contrast_colors(np.array([-5.0, -3.0, 0.0, 3.0, np.nan]))
        self.assertEqual(
            [to_hex(c) for c in colours],
            [GROUP_COLORS[0].lower()] * 2 + [CONTRAST_MID]
            + [GROUP_COLORS[1].lower(), CONTRAST_MID],
        )


class OrfShapeTests(unittest.TestCase):
    def test_strands(self):
        """x is position - 1, so [11, 21) spans 10 to 20; + points right,
        - left, and an unstranded ORF straddles the backbone."""
        y0, y1 = ORF_Y
        mid = (y0 + y1) / 2
        u = UNSTRANDED_Y
        shapes = orf_polygons(np.array([11, 11, 11]), np.array([21, 21, 21]),
                              np.array(["+", "-", "."], dtype=object), 4)
        self.assertEqual(
            [s.tolist() for s in shapes],
            [[[10, y0], [16, y0], [20, mid], [16, y1], [10, y1]],
             [[20, -y0], [14, -y0], [10, -mid], [14, -y1], [20, -y1]],
             [[10, -u], [20, -u], [20, u], [10, u]]],
        )

    def test_the_head_is_at_most_half_the_orf(self):
        (shape,) = orf_polygons(np.array([1]), np.array([5]),
                                np.array(["+"], dtype=object), 100)
        self.assertEqual(shape[:, 0].tolist(), [0, 2, 4, 2, 0])

    def test_no_head_is_a_box(self):
        (shape,) = orf_polygons(np.array([1]), np.array([5]),
                                np.array(["+"], dtype=object), 0)
        self.assertEqual(shape[:, 0].tolist(), [0, 4, 4, 4, 0])


class LabelLaneTests(unittest.TestCase):
    def test_first_fit_then_dropped(self):
        """Three overlapping labels on one strand: two lanes, one left out."""
        lanes = assign_label_lanes(np.array([10.0, 12.0, 14.0]), np.full(3, 5.0),
                                   np.array(["+", "+", "+"], dtype=object))
        self.assertEqual(lanes.tolist(), [0, 1, -1])

    def test_strands_have_their_own_lanes(self):
        """An unstranded label goes above, with the + strand: it collides
        with the + label at 10, not the - label at 30."""
        lanes = assign_label_lanes(np.array([10.0, 30.0, 12.0]), np.full(3, 5.0),
                                   np.array(["+", "-", "."], dtype=object))
        self.assertEqual(lanes.tolist(), [0, 0, 1])

    def test_placed_left_to_right_whatever_the_input_order(self):
        lanes = assign_label_lanes(np.array([30.0, 10.0, 12.0]), np.full(3, 5.0),
                                   np.array(["+", "+", "+"], dtype=object))
        self.assertEqual(lanes.tolist(), [0, 0, 1])


class OrfTrackTests(unittest.TestCase):
    def orfs(self):
        return orfs_of(
            (1, 301, "+", "dnaA", {"COG": "J"}),
            (901, 1101, "-", "b", {"COG": "X"}),
            (1201, 1401, "+", "c", {"COG": "J"}),
            (2001, 2101, ".", "d", {}),
        )

    def fills(self, track):
        return [to_hex(c) for c in track["fill"]]

    def test_neutral_with_highlights_in_ink(self):
        track = orf_track(self.orfs(), np.array([False, True, False, False]))
        self.assertEqual(self.fills(track),
                         [ORF_NEUTRAL, INK, ORF_NEUTRAL, ORF_NEUTRAL])
        self.assertEqual(track["outline"].tolist(), [False] * 4)
        self.assertIsNone(track["legend"])

    def test_colour_by_keeps_fills_and_outlines_highlights(self):
        track = orf_track(self.orfs(), np.array([False, True, False, False]),
                          color_by="COG")
        c = [colour.lower() for colour in ORF_CATEGORY_COLORS]
        self.assertEqual(self.fills(track), [c[0], c[1], c[0], ORF_OTHER])
        self.assertEqual(track["outline"].tolist(), [False, True, False, False])
        self.assertEqual(track["legend"], ("categories", ["J", "X"]))

    def test_contrast(self):
        track = orf_track(self.orfs(), np.zeros(4, bool),
                          contrast=np.array([-3.0, 3.0, 0.0, np.nan]))
        self.assertEqual(
            self.fills(track),
            [GROUP_COLORS[0].lower(), GROUP_COLORS[1].lower(), CONTRAST_MID,
             CONTRAST_MID],
        )
        self.assertEqual(track["legend"], ("contrast",))

    def test_the_rest_of_the_orf_is_carried(self):
        track = orf_track(self.orfs(), np.array([False, True, False, False]))
        self.assertEqual(track["start"].tolist(), [1, 901, 1201, 2001])
        self.assertEqual(track["strand"].tolist(), ["+", "-", "+", "."])
        self.assertEqual(track["label"].tolist(), ["dnaA", "b", "c", "d"])
        self.assertEqual(track["highlight"].tolist(), [False, True, False, False])


class OrfModeTests(unittest.TestCase):
    def test_colour_by_needs_at_most_two_groups(self):
        """Three categories beside three group colours cannot all stay
        distinct to a colour-blind reader."""
        with self.assertRaisesRegex(ValueError, "--highlight"):
            check_orf_mode(3, "product", False)
        check_orf_mode(2, "product", False)

    def test_contrast_needs_exactly_two_groups(self):
        for n in (1, 3):
            with self.subTest(groups=n), self.assertRaisesRegex(
                ValueError, "exactly two"
            ):
                check_orf_mode(n, None, True)
        check_orf_mode(2, None, True)

    def test_one_colouring_at_a_time(self):
        with self.assertRaisesRegex(ValueError, "--orf-contrast"):
            check_orf_mode(2, "product", True)


class PlotPathTests(unittest.TestCase):
    def test_named_like_micovs_other_plots(self):
        """{output}.{target name}.{genome}.{variable}.{tag}.png"""
        self.assertEqual(plot_path("out/run", "E_coli", "GC", "group"),
                         "out/run.E_coli.GC.group.depth-plot.png")
        self.assertEqual(plot_path("out/run", "E_coli", "GC", "group", "circular"),
                         "out/run.E_coli.GC.group.depth-plot-circular.png")
        self.assertEqual(
            plot_path("out/run", "E_coli", "GC", "group", "detail-1001-1501"),
            "out/run.E_coli.GC.group.depth-plot-detail-1001-1501.png",
        )


def bins_table(groups, edges):
    """A `genome_statistics` bins table with a distinct, known value in every
    bin: group g's median at bin i is i % 7 + g."""
    k = len(edges) - 1
    i = np.arange(k)
    table = {key: [] for key in ("group", "bin_start", "bin_stop", "q1",
                                 "median", "q3", "mean", "prevalence", "union")}
    for g, group in enumerate(groups):
        median = (i % 7 + g).astype(float)
        for key, values in (
            ("group", np.full(k, group, dtype=object)),
            ("bin_start", edges[:-1]), ("bin_stop", edges[1:]),
            ("q1", median / 2), ("median", median), ("q3", median + 1),
            ("mean", median + 0.25), ("prevalence", (i % 5) / 4),
            ("union", i % 3 != g % 3),
        ):
            table[key].append(values)
    return {key: np.concatenate(values) for key, values in table.items()}


def group_rows(table, group):
    return {key: values[table["group"] == group] for key, values in table.items()}


def axes_by_gid(fig, gid):
    """The axes, or inset axes, with this gid."""
    (ax,) = [ax for parent in fig.axes for ax in (parent, *parent.child_axes)
             if ax.get_gid() == gid]
    return ax


def artists(ax, gid):
    return [a for a in ax.get_children() if a.get_gid() == gid]


def artist(ax, gid):
    (found,) = artists(ax, gid)
    return found


class DrawingTestCase(unittest.TestCase):
    """Draws with `Figure.savefig` patched, keeping the figure to inspect."""

    SIZES: ClassVar[dict] = {"case": 3, "control": 2}

    def draw(self, plot, **kwargs):
        with mock.patch.object(Figure, "savefig", autospec=True) as save:
            plot("out.png", title="GC", **kwargs)
        save.assert_called_once()
        self.assertEqual(save.call_args.args[1], "out.png")
        return save.call_args.args[0]

    def overview(self, groups=("case", "control"), length=4_500_000, **kwargs):
        sizes = {g: self.SIZES.get(g, 4) for g in groups}
        table = bins_table(groups, display_bin_edges(1, length + 1,
                                                     overview_bin_bp(length)))
        fig = self.draw(linear_plot, length=length, sizes=sizes, overview=table,
                        **kwargs)
        return fig, table


class LinearOverviewTests(DrawingTestCase):
    def test_one_iqr_median_and_mean_per_group_in_its_colour(self):
        """Groups take `group_style` in sorted order; the mean is dotted."""
        fig, _ = self.overview()
        ax = axes_by_gid(fig, "depth:0")
        for g, group in enumerate(("case", "control")):
            colour = group_style(g)[0].lower()
            with self.subTest(group=group):
                self.assertEqual(to_hex(artist(ax, f"median:{group}")
                                        .get_edgecolor()), colour)
                self.assertEqual(to_hex(artist(ax, f"iqr:{group}")
                                        .get_facecolor()), colour)
                self.assertEqual(to_hex(artist(ax, f"mean:{group}")
                                        .get_edgecolor()), colour)
                self.assertEqual(artist(ax, f"mean:{group}").get_linestyle(),
                                 ":")

    def test_steps_are_the_bins_exactly(self):
        """Edges are bin coordinates less one: base 1 spans [0, 1)."""
        fig, table = self.overview()
        rows = group_rows(table, "control")
        first = rows["bin_start"] <= ROW_BP
        ax = axes_by_gid(fig, "depth:0")
        median = artist(ax, "median:control").get_data()
        self.assertEqual(median.values.tolist(), rows["median"][first].tolist())
        self.assertEqual(median.edges.tolist(),
                         [0, *(rows["bin_stop"][first] - 1).tolist()])
        iqr = artist(ax, "iqr:control").get_data()
        self.assertEqual(iqr.values.tolist(), rows["q3"][first].tolist())
        self.assertEqual(iqr.baseline.tolist(), rows["q1"][first].tolist())

    def test_rows_of_two_megabases_masked_past_the_end(self):
        """4.5 Mb is three rows, each spanning exactly 2 Mb, so every row and
        every genome has the same resolution."""
        fig, _ = self.overview()
        for row in range(3):
            for kind in ("depth", "orfs", "breadth"):
                with self.subTest(row=row, axes=kind):
                    self.assertEqual(
                        axes_by_gid(fig, f"{kind}:{row}").get_xlim(),
                        (row * ROW_BP, (row + 1) * ROW_BP),
                    )
        self.assertFalse([ax for ax in fig.axes if ax.get_gid() == "depth:3"])
        for kind in ("depth", "orfs", "breadth"):
            past = artist(axes_by_gid(fig, f"{kind}:2"), "past-end")
            self.assertEqual(past.get_x(), 4_500_000)
            self.assertEqual(past.get_width(), 1_500_000)
        self.assertFalse(artists(axes_by_gid(fig, "depth:1"), "past-end"))

    def test_depth_is_symlog_and_shared_by_every_row(self):
        fig, _ = self.overview()
        axes = [axes_by_gid(fig, f"depth:{row}") for row in range(3)]
        for ax in axes:
            self.assertEqual(ax.get_yscale(), "symlog")
            self.assertEqual(ax.yaxis.get_transform().linthresh, DEPTH_LINTHRESH)
        self.assertEqual(len({ax.get_ylim() for ax in axes}), 1)
        self.assertIn("alignments", axes[1].get_ylabel())

    def test_union_and_prevalence(self):
        """Union is a segment wherever any sample covers; prevalence hangs
        below the genome, 0 at the top."""
        fig, table = self.overview()
        rows = group_rows(table, "case")
        first = rows["bin_start"] <= ROW_BP
        ax = axes_by_gid(fig, "breadth:0")
        segments = artist(ax, "union:case").get_segments()
        starts, stops = true_runs(rows["union"][first])
        self.assertEqual([(s[0][0], s[1][0]) for s in segments],
                         list(zip(starts * 1000, stops * 1000, strict=True)))
        prevalence = artist(ax, "prevalence:case").get_data()
        self.assertEqual(prevalence.values.tolist(),
                         rows["prevalence"][first].tolist())
        bottom, top = ax.get_ylim()
        self.assertGreater(bottom, top)

    def test_a_legend_names_each_group_and_its_size(self):
        fig, _ = self.overview()
        legend = axes_by_gid(fig, "depth:0").get_legend()
        self.assertEqual([t.get_text() for t in legend.get_texts()],
                         ["case  n=3", "control  n=2", "group mean"])

    def test_four_groups_get_a_lane_each_on_one_scale(self):
        groups = ("a", "b", "c", "d")
        fig, _ = self.overview(groups=groups, length=1_500_000)
        lanes = [axes_by_gid(fig, f"depth:0:{g}") for g in groups]
        self.assertEqual(len({ax.get_ylim() for ax in lanes}), 1)
        for g, (group, ax) in enumerate(zip(groups, lanes, strict=True)):
            with self.subTest(group=group):
                self.assertEqual(to_hex(artist(ax, f"median:{group}")
                                        .get_edgecolor()),
                                 group_style(g)[0].lower())
                self.assertEqual(artist(ax, f"lane:{group}").get_text(),
                                 f"{group}  n=4")
                breadth = axes_by_gid(fig, f"breadth:0:{group}")
                self.assertTrue(artists(breadth, f"prevalence:{group}"))
        self.assertFalse([ax for ax in fig.axes if ax.get_gid() == "depth:0"])

    def test_a_genome_of_exactly_one_row(self):
        fig, _ = self.overview(length=ROW_BP)
        self.assertFalse([ax for ax in fig.axes if ax.get_gid() == "depth:1"])
        self.assertFalse(artists(axes_by_gid(fig, "depth:0"), "past-end"))

    def test_three_groups_still_overlay(self):
        fig, _ = self.overview(groups=("a", "b", "c"), length=1_500_000)
        ax = axes_by_gid(fig, "depth:0")
        for group in ("a", "b", "c"):
            self.assertTrue(artists(ax, f"median:{group}"))

    def test_a_genome_shorter_than_a_row_is_one_row_of_its_length(self):
        """A mitochondrion fills the width, rather than a sliver of 2 Mb, and
        nothing is past its end to mask."""
        fig, _ = self.overview(length=3000)
        for kind in ("depth", "orfs", "breadth"):
            with self.subTest(axes=kind):
                self.assertEqual(axes_by_gid(fig, f"{kind}:0").get_xlim(),
                                 (0, 3000))
                self.assertFalse(artists(axes_by_gid(fig, f"{kind}:0"),
                                         "past-end"))
        self.assertFalse([ax for ax in fig.axes if ax.get_gid() == "depth:1"])

    def test_regions_are_shaded_where_they_fall(self):
        fig, _ = self.overview(regions=[(1001, 1501), (2_500_001, 2_600_001)])
        shade = artist(axes_by_gid(fig, "depth:0"), "region:0")
        self.assertEqual((shade.get_x(), shade.get_width()), (1000, 500))
        self.assertTrue(artists(axes_by_gid(fig, "depth:1"), "region:1"))
        self.assertFalse(artists(axes_by_gid(fig, "depth:0"), "region:1"))
        self.assertFalse(artists(axes_by_gid(fig, "depth:1"), "region:0"))

    def test_saved_once_and_tight(self):
        with mock.patch.object(Figure, "savefig", autospec=True) as save:
            linear_plot("out.png", title="GC", length=3000, sizes={"a": 1},
                        overview=bins_table(["a"], np.array([1, 1001, 3001])))
        self.assertEqual(save.call_args.kwargs.get("bbox_inches"), "tight")


TRACK_ORFS = staticmethod(lambda: orfs_of(
    (101, 1101, "+", "dnaA", {}),
    (1201, 2201, "+", "dnaB", {}),
    (1301, 2301, "+", "dnaC", {}),
    (5001, 6001, "-", "rpoB", {}),
    (8001, 9001, ".", "rrs", {}),
))


class OrfDrawingTests(DrawingTestCase):
    def track(self, highlight=(False,) * 5, **kwargs):
        return orf_track(TRACK_ORFS(), np.array(highlight), **kwargs)

    def test_neutral_orfs_by_strand(self):
        fig, _ = self.overview(length=10_000, track=self.track())
        ax = axes_by_gid(fig, "orfs:0")
        self.assertEqual(len(artist(ax, "orfs:+").get_paths()), 3)
        self.assertEqual(len(artist(ax, "orfs:-").get_paths()), 1)
        self.assertEqual(len(artist(ax, "orfs:.").get_paths()), 1)
        self.assertEqual({to_hex(c) for c in artist(ax, "orfs:+").get_facecolor()},
                         {ORF_NEUTRAL})
        self.assertTrue(artists(ax, "backbone"))
        self.assertFalse(artists(ax, "label"))

    def test_highlights_get_ink_a_band_and_labels(self):
        """At 2 Mb a row, the labels of dnaA, dnaB and dnaC (all + strand,
        within 2.3 kb) overlap: two lanes take dnaA and dnaB, and dnaC is
        counted. Their bands are a point apart at most, so merge."""
        track = self.track(highlight=(True, True, True, False, False))
        fig, _ = self.overview(length=ROW_BP, track=track)
        orfs = axes_by_gid(fig, "orfs:0")
        self.assertEqual(
            [to_hex(c) for c in artist(orfs, "orfs:+").get_facecolor()],
            [INK] * 3,
        )
        self.assertEqual(sorted(t.get_text() for t in artists(orfs, "label")),
                         ["dnaA", "dnaB"])
        self.assertEqual(artist(orfs, "unlabelled").get_text(), "+1 unlabelled")
        bands = artists(axes_by_gid(fig, "depth:0"), "highlight")
        self.assertEqual([(b.get_x(), b.get_x() + b.get_width()) for b in bands],
                         [(100, 2300)])

    def test_colour_by_has_a_legend(self):
        track = orf_track(orfs_of((1, 101, "+", "a", {"COG": "J"}),
                                  (201, 301, "-", "b", {"COG": "X"})),
                          np.zeros(2, bool), color_by="COG")
        fig, _ = self.overview(length=1000, track=track)
        legend = artist(axes_by_gid(fig, "orfs:0"), "orf-legend")
        self.assertEqual([t.get_text() for t in legend.get_texts()],
                         ["J", "X", ORF_OTHER_LABEL])

    def test_a_short_genome_shows_which_way_its_genes_run(self):
        """A 16.6 kb mitochondrion's row is within `ARROW_MAX_BP`, so its
        genes are arrows; on a 4.5 Mb genome the same genes are boxes."""
        def shapes(length):
            track = orf_track(orfs_of((101, 1101, "+", "nad1", {}),
                                      (2001, 3001, "-", "nad2", {})),
                              np.zeros(2, bool))
            fig, _ = self.overview(length=length, track=track)
            ax = axes_by_gid(fig, "orfs:0")
            return [len(set(path.vertices[:, 0]))
                    for strand in "+-"
                    for path in artist(ax, f"orfs:{strand}").get_paths()]

        # start, where the head begins, and the tip; a box has only two
        self.assertEqual(shapes(16_569), [3, 3])
        self.assertEqual(shapes(4_500_000), [2, 2])

    def test_contrast_has_a_scale_naming_the_groups(self):
        track = self.track(contrast=np.array([-3.0, 3.0, 0.0, 1.0, np.nan]))
        fig, _ = self.overview(length=10_000, track=track)
        scale = axes_by_gid(fig, "contrast-scale")
        self.assertEqual([t.get_text() for t in scale.get_xticklabels()],
                         ["case \u00d78", "equal", "control \u00d78"])

    def test_an_orf_across_the_origin_is_drawn_at_both_ends(self):
        """9,501-10,500 on a 10,000 bp circular genome: 9,501-10,000 and 1-500,
        both boxes, since neither part holds both of its ends."""
        track = orf_track(orfs_of((9501, 10501, "+", "a", {})), np.zeros(1, bool))
        fig, _ = self.overview(length=10_000, track=track)
        paths = artist(axes_by_gid(fig, "orfs:0"), "orfs:+").get_paths()
        spans = sorted((p.vertices[:, 0].min(), p.vertices[:, 0].max())
                       for p in paths)
        self.assertEqual(spans, [(0, 500), (9500, 10_000)])
        # a detail panel draws arrows, but not for either part
        table = bins_table(["case"], display_bin_edges(9001, 10_001, 1))
        fig = self.draw(detail_plot, length=10_000, sizes={"case": 3},
                        table=table, track=track)
        (part,) = artist(axes_by_gid(fig, "orfs:0"), "orfs:+").get_paths()
        self.assertEqual(sorted(set(part.vertices[:, 0])), [9500, 10_000])

    def test_highlighted_and_coloured_orfs_are_outlined(self):
        track = self.track(highlight=(True, False, False, False, False),
                           contrast=np.zeros(5))
        fig, _ = self.overview(length=10_000, track=track)
        edges = artist(axes_by_gid(fig, "orfs:0"), "orfs:+").get_edgecolor()
        self.assertEqual(to_hex(edges[0]), INK)
        self.assertEqual(edges[1][3], 0.0)


class DetailPlotTests(DrawingTestCase):
    def detail(self, track=None, groups=("case", "control")):
        table = bins_table(groups, display_bin_edges(1001, 1501, 1))
        sizes = {g: self.SIZES.get(g, 4) for g in groups}
        fig = self.draw(detail_plot, length=3000, sizes=sizes, table=table,
                        track=track)
        return fig, table

    def test_single_bases_over_the_region(self):
        fig, _ = self.detail()
        ax = axes_by_gid(fig, "depth:0")
        self.assertEqual(ax.get_xlim(), (1000, 1500))
        median = artist(ax, "median:case").get_data()
        self.assertEqual(len(median.values), 500)
        self.assertEqual((median.edges[0], median.edges[-1]), (1000, 1500))

    def test_orfs_are_arrows_and_highlights_labelled(self):
        track = orf_track(orfs_of((1101, 1201, "+", "dnaA", {}),
                                  (1301, 1401, "-", "dnaB", {})),
                          np.array([True, False]))
        fig, _ = self.detail(track=track)
        ax = axes_by_gid(fig, "orfs:0")
        (arrow,) = artist(ax, "orfs:+").get_paths()
        # start, where the head begins, and the tip; a box has only two
        self.assertEqual(len(set(arrow.vertices[:, 0])), 3)
        self.assertEqual([t.get_text() for t in artists(ax, "label")], ["dnaA"])

    def test_lanes(self):
        fig, _ = self.detail(groups=("a", "b", "c", "d"))
        self.assertTrue(artists(axes_by_gid(fig, "depth:0:d"), "median:d"))


class DrawingGuardTests(DrawingTestCase):
    def test_masked_values_are_refused_before_drawing(self):
        """fetchnumpy hands back NULLs as masked arrays, which matplotlib
        draws as gaps without complaint."""
        table = bins_table(["a"], np.array([1, 1001, 2001]))
        table["median"] = np.ma.masked_array(table["median"], [True, False])
        with mock.patch.object(Figure, "savefig", autospec=True) as save, \
                self.assertRaisesRegex(ValueError, "median"):
            linear_plot("out.png", title="G", length=2000, sizes={"a": 1},
                        overview=table)
        save.assert_not_called()

    def test_bins_must_tile(self):
        gap = bins_table(["a"], np.array([1, 1001, 2001]))
        gap["bin_start"][1] += 1
        disagree = bins_table(["a", "b"], np.array([1, 1001, 2501]))
        disagree["bin_stop"][-1] = 2001
        for why, table in (("a gap between bins", gap),
                           ("the groups' bins disagree", disagree)):
            with self.subTest(why), mock.patch.object(
                Figure, "savefig", autospec=True
            ) as save, self.assertRaisesRegex(ValueError, "bins"):
                linear_plot("out.png", title="G", length=2500,
                            sizes=dict.fromkeys(set(table["group"]), 1),
                            overview=table)
            save.assert_not_called()

    def test_a_failed_save_leaves_no_figure_and_no_style_behind(self):
        before = dict(plt.rcParams)
        with mock.patch.object(Figure, "savefig", side_effect=OSError("full")), \
                self.assertRaises(OSError):
            linear_plot("out.png", title="G", length=2000, sizes={"a": 1},
                        overview=bins_table(["a"], np.array([1, 1001, 2001])))
        self.assertEqual(plt.get_fignums(), [])
        self.assertEqual(dict(plt.rcParams), before)

    def test_a_real_png(self):
        d = mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        path = f"{d}/o'brien.png"
        track = orf_track(TRACK_ORFS(), np.array([True, False, False, True, False]))
        linear_plot(path, title="GC", length=10_000, sizes={"a": 1, "b": 2},
                    overview=bins_table(["a", "b"],
                                        display_bin_edges(1, 10_001, 1000)),
                    regions=[(1001, 2001)], track=track)
        with open(path, "rb") as fp:
            self.assertEqual(fp.read(8), b"\x89PNG\r\n\x1a\n")
        self.assertEqual(plt.get_fignums(), [])


TAU = 2 * np.pi


class RingGeometryTests(unittest.TestCase):
    def test_theta_runs_once_round_per_genome(self):
        """x 0 is the top; an ORF across the origin carries on past 2 pi,
        which the polar axes wrap."""
        np.testing.assert_allclose(
            theta(np.array([0, 2500, 10_000, 10_500]), 10_000),
            [0, TAU / 4, TAU, TAU * 1.05],
        )

    def test_densify_closes_a_ring(self):
        """Polar axes join points with straight chords, so a ring given by
        its two ends alone would not be drawn at all."""
        angles, radii = densify([0, TAU], [0.6, 0.6])
        self.assertEqual((angles[0], angles[-1]), (0, TAU))
        self.assertLessEqual(np.diff(angles).max(), MAX_ARC + 1e-12)
        self.assertTrue((radii == 0.6).all())

    def test_densify_traces_arcs_either_way_and_slants_evenly(self):
        """A polygon's far side runs back round; an arrowhead's edges change
        radius as they turn."""
        angles, radii = densify([TAU, 0], [0.6, 0.6])
        self.assertLessEqual(np.abs(np.diff(angles)).max(), MAX_ARC + 1e-12)
        angles, radii = densify([0, 2 * MAX_ARC], [0.5, 0.7])
        np.testing.assert_allclose(angles, [0, MAX_ARC, 2 * MAX_ARC])
        np.testing.assert_allclose(radii, [0.5, 0.6, 0.7])

    def test_densify_leaves_a_radial_line_alone(self):
        angles, radii = densify([1.0, 1.0], [0.2, 0.9])
        self.assertEqual(angles.tolist(), [1.0, 1.0])
        self.assertEqual(radii.tolist(), [0.2, 0.9])

    def test_polar_steps_hold_each_bin_then_step(self):
        """Bins [0, 10) at 1 and [10, 20) at 3 on a 40 bp genome: an arc at
        1 to a quarter turn, a radial step, then an arc at 3 to a half."""
        angles, values = polar_steps(np.array([0, 10, 20]), np.array([1.0, 3.0]),
                                     40)
        self.assertEqual(angles[0], 0)
        self.assertAlmostEqual(angles[-1], TAU / 2)
        quarter = np.isclose(angles, TAU / 4)
        self.assertEqual(values[quarter].tolist(), [1.0, 3.0])
        self.assertTrue((values[~quarter & (angles < TAU / 4)] == 1).all())
        self.assertTrue((values[~quarter & (angles > TAU / 4)] == 3).all())
        self.assertTrue((np.diff(angles) >= 0).all())
        self.assertLessEqual(np.diff(angles).max(), MAX_ARC + 1e-12)

    def test_rings_run_outside_in_without_touching(self):
        """Labels, coordinates, depth, the ORFs either side of the backbone,
        a union arc per group, then prevalence: strictly outside in, with a
        hole left in the middle."""
        for n in (1, 2, 3):
            with self.subTest(groups=n):
                radii = circular_layout(n)
                values = list(radii.values())
                self.assertTrue(all(a > b for a, b in itertools.pairwise(values)),
                                radii)
                self.assertGreater(values[-1], 0.2)
                self.assertEqual([k for k in radii if k.startswith("union")],
                                 [f"union:{g}" for g in range(n)])

    def test_ring_text_reads_outward_and_never_upside_down(self):
        for angle, rotation, ha in ((0, 90, "left"), (TAU / 4, 0, "left"),
                                    (TAU / 2, 90, "right"),
                                    (3 * TAU / 4, 0, "right")):
            with self.subTest(angle=angle):
                got = radial_text(angle)
                self.assertAlmostEqual(got[0], rotation)
                self.assertEqual(got[1], ha)
        for angle in np.linspace(0, TAU, 73, endpoint=False):
            rotation, ha = radial_text(angle)
            self.assertTrue(-90 < rotation <= 90)
            # the text runs from its anchor away from the centre: clockwise
            # from the top, outward is (sin, cos)
            runs = (1 if ha == "left" else -1) * np.array(
                [np.cos(np.radians(rotation)), np.sin(np.radians(rotation))])
            np.testing.assert_allclose(runs, [np.sin(angle), np.cos(angle)],
                                       atol=1e-12)


    def test_coordinates_run_along_the_ring_inside_it(self):
        """Along the ring, so a label takes one line of the depth band
        however long it is, and hanging inward, clear of the highlighted
        ORFs' leaders outside; never upside down."""
        for angle, rotation, va in ((0, 0, "top"), (TAU / 4, -90, "top"),
                                    (TAU / 2, 0, "bottom"),
                                    (3 * TAU / 4, 90, "top")):
            with self.subTest(angle=angle):
                got = tangential_text(angle)
                self.assertAlmostEqual(got[0], rotation)
                self.assertEqual(got[1], va)
        for angle in np.linspace(0, TAU, 73, endpoint=False):
            rotation, va = tangential_text(angle)
            self.assertTrue(-90 <= rotation <= 90)
            up = np.array([-np.sin(np.radians(rotation)),
                           np.cos(np.radians(rotation))])
            # the text's body lies inward of its anchor: clockwise from the
            # top, inward is (-sin, -cos)
            body = -up if va == "top" else up
            np.testing.assert_allclose(body, [-np.sin(angle), -np.cos(angle)],
                                       atol=1e-12)


class RingLabelTests(unittest.TestCase):
    def test_spaced_labels_stay_put(self):
        angles = np.array([0.5, 1.0, 3.0])
        np.testing.assert_allclose(spread_angles(angles, 0.1), angles)

    def test_a_tie_spreads_evenly_about_its_angle(self):
        """Moved as little as possible: two labels at 1.0, 0.1 apart, go to
        0.95 and 1.05."""
        np.testing.assert_allclose(spread_angles(np.array([1.0, 1.0]), 0.1),
                                   [0.95, 1.05])

    def test_a_cluster_across_the_top_spreads_there(self):
        """0.01 and 2 pi - 0.01 are neighbours across the top. Cut the circle
        anywhere but its widest gap and they would be spread apart from
        either end of the cut, or averaged to pi."""
        np.testing.assert_allclose(
            spread_angles(np.array([0.01, TAU - 0.01]), 0.1),
            [0.05, TAU - 0.05],
        )

    def test_placed_labels_keep_their_gap_and_stay_near_their_orf(self):
        """Random clusters: each placed label is a gap from its neighbours,
        across the top too, and within `max_shift` of its ORF; the rest are
        NaN, to be counted. Past 78 labels the ring is full, and they are
        thinned first."""
        rng = np.random.default_rng(1)
        for trial in range(40):
            n = int(rng.integers(1, 200))
            centres = rng.uniform(0, TAU, 4)
            angles = (rng.choice(centres, n) + rng.normal(0, 0.05, n)) % TAU
            placed = place_ring_labels(angles, 0.08, 0.25)
            on = ~np.isnan(placed)
            with self.subTest(trial=trial, n=n):
                self.assertTrue(on.any())
                ring = np.sort(placed[on])
                gaps = np.diff(np.append(ring, ring[0] + TAU))
                self.assertGreaterEqual(gaps.min(), 0.08 - 1e-9)
                shift = np.abs((placed[on] - angles[on] + np.pi) % TAU - np.pi)
                self.assertLessEqual(shift.max(), 0.25 + 1e-9)

    def test_a_cluster_too_big_loses_its_farthest_labels(self):
        """Nine labels at one angle, 0.1 apart and moving at most 0.32:
        seven fit, from 0.7 to 1.3, and the two pushed farthest are not."""
        placed = place_ring_labels(np.full(9, 1.0), 0.1, 0.32)
        self.assertEqual(int(np.isnan(placed).sum()), 2)
        np.testing.assert_allclose(np.sort(placed[~np.isnan(placed)]),
                                   np.linspace(0.7, 1.3, 7))

    def test_more_labels_than_the_ring_holds_are_thinned_not_moved(self):
        """31 labels evenly round a ring that holds ten: ten stay, about every
        third, each on its own ORF rather than pushed off it."""
        angles = np.arange(31) * TAU / 31
        placed = place_ring_labels(angles, TAU / 10.5, 0.5)
        on = ~np.isnan(placed)
        self.assertEqual(int(on.sum()), 10)
        np.testing.assert_allclose(placed[on], angles[on])

    def test_a_crowd_loses_labels_before_a_label_on_its_own(self):
        """Five labels at 1.0 and one at 1.6, 0.1 apart and moving at most
        0.15: the crowd's ends, moved farthest, go; the loner stays put."""
        angles = np.array([1.0] * 5 + [1.6])
        placed = place_ring_labels(angles, 0.1, 0.15)
        self.assertEqual(placed[-1], 1.6)
        self.assertEqual(int(np.isnan(placed).sum()), 1)

    def test_crowds_that_would_spread_into_each_other_are_thinned(self):
        """Five labels at the top, one a third of the way round, and five
        just short of two thirds. Spread freely, both fives would push past
        the cut and into each other; labels are dropped until none do."""
        angles = np.array([0.0] * 5 + [TAU / 3] + [2 * TAU / 3 - 0.01] * 5)
        placed = place_ring_labels(angles, TAU / 11, 2.0)
        ring = np.sort(placed[~np.isnan(placed)])
        self.assertGreaterEqual(np.diff(np.append(ring, ring[0] + TAU)).min(),
                                TAU / 11 - 1e-9)

    def test_thinning_starts_after_the_widest_gap(self):
        """Forty labels 0.1 apart from 5.0 round past the top, on a ring that
        holds 25 at 0.25 apart: every third from 5.0 stays, on its ORF."""
        angles = (5.0 + 0.1 * np.arange(40)) % TAU
        placed = place_ring_labels(angles, 0.25, 0.5)
        on = ~np.isnan(placed)
        self.assertEqual(np.flatnonzero(on).tolist(), list(range(0, 40, 3)))
        np.testing.assert_allclose(placed[on], angles[on])

    def test_independent_of_order(self):
        angles = np.array([0.2, 0.25, 0.26, 3.0, 6.25, 6.27])
        order = np.array([3, 0, 5, 1, 4, 2])
        np.testing.assert_allclose(place_ring_labels(angles[order], 0.1, 0.3),
                                   place_ring_labels(angles, 0.1, 0.3)[order])


class RingTestCase(DrawingTestCase):
    def ring(self, groups=("case", "control"), length=4_500_000, **kwargs):
        sizes = {g: self.SIZES.get(g, 4) for g in groups}
        table = bins_table(groups, display_bin_edges(1, length + 1,
                                                     overview_bin_bp(length)))
        fig = self.draw(circular_plot, length=length, sizes=sizes,
                        overview=table, **kwargs)
        return fig, table, axes_by_gid(fig, "ring")

    @staticmethod
    def edges(rows):
        return np.append(rows["bin_start"], rows["bin_stop"][-1:]) - 1


class CircularPlotTests(RingTestCase):
    def test_polar_from_the_top_clockwise(self):
        _, _, ax = self.ring()
        self.assertEqual(ax.name, "polar")
        self.assertAlmostEqual(ax.get_theta_offset(), np.pi / 2)
        self.assertEqual(ax.get_theta_direction(), -1)

    def test_one_iqr_median_and_mean_per_group_in_its_colour(self):
        _, _, ax = self.ring()
        for g, group in enumerate(("case", "control")):
            colour = group_style(g)[0].lower()
            with self.subTest(group=group):
                self.assertEqual(to_hex(artist(ax, f"median:{group}").get_color()),
                                 colour)
                self.assertEqual(to_hex(artist(ax, f"iqr:{group}")
                                        .get_facecolor()[0]), colour)
                mean = artist(ax, f"mean:{group}")
                self.assertEqual(to_hex(mean.get_color()), colour)
                self.assertEqual(mean.get_linestyle(), ":")

    def test_the_iqr_runs_from_q1_to_q3(self):
        _, table, ax = self.ring()
        rows = group_rows(table, "control")
        radii = circular_layout(2)
        band = depth_ymax([table]), radii["depth_base"], radii["depth_top"]
        _, q1 = polar_steps(self.edges(rows), rows["q1"], 4_500_000)
        _, q3 = polar_steps(self.edges(rows), rows["q3"], 4_500_000)
        (path,) = artist(ax, "iqr:control").get_paths()
        self.assertAlmostEqual(path.vertices[:, 1].min(),
                               radial_symlog(q1, *band).min())
        self.assertAlmostEqual(path.vertices[:, 1].max(),
                               radial_symlog(q3, *band).max())

    def test_depth_rises_outward_from_the_backbone_on_the_linear_scale(self):
        """The median traces each bin round the ring, at the radius of its
        depth on the same symlog scale as the linear plot."""
        _, table, ax = self.ring()
        rows = group_rows(table, "control")
        radii = circular_layout(2)
        angles, values = polar_steps(self.edges(rows), rows["median"],
                                     4_500_000)
        median = artist(ax, "median:control")
        np.testing.assert_array_equal(median.get_xdata(), angles)
        np.testing.assert_allclose(
            median.get_ydata(),
            radial_symlog(values, depth_ymax([table]), radii["depth_base"],
                          radii["depth_top"]),
        )
        self.assertGreaterEqual(min(median.get_ydata()), radii["depth_base"])

    def test_a_depth_sits_where_the_linear_plot_puts_it(self):
        """The same fraction of the ring's depth band as of the linear plot's
        depth axis, so the two plots of a genome read alike."""
        fig, table = self.overview()
        ax = axes_by_gid(fig, "depth:0")
        ymax = depth_ymax([table])
        for depth in (0, 1, 2, 5, ymax):
            with self.subTest(depth=depth):
                linear = ax.transAxes.inverted().transform(
                    ax.transData.transform((0, depth)))[1]
                ring = (radial_symlog(depth, ymax, 0.5, 0.9) - 0.5) / 0.4
                self.assertAlmostEqual(ring, linear)

    def test_union_arcs_and_prevalence_hang_inward(self):
        """Each group's union arcs on its own ring inside the backbone;
        prevalence 0 just inside them, and 1 farthest in."""
        _, table, ax = self.ring()
        radii = circular_layout(2)
        for g, group in enumerate(("case", "control")):
            rows = group_rows(table, group)
            edges = self.edges(rows)
            with self.subTest(group=group):
                starts, stops = true_runs(rows["union"])
                arcs = artist(ax, f"union:{group}").get_segments()
                np.testing.assert_allclose(
                    [(arc[0, 0], arc[-1, 0]) for arc in arcs],
                    np.column_stack([theta(edges[starts], 4_500_000),
                                     theta(edges[stops], 4_500_000)]),
                )
                self.assertEqual({r for arc in arcs for r in arc[:, 1]},
                                 {radii[f"union:{g}"]})
                prevalence = artist(ax, f"prevalence:{group}")
                _, values = polar_steps(edges, rows["prevalence"], 4_500_000)
                r = np.asarray(prevalence.get_ydata())
                np.testing.assert_allclose(r[values == 0], radii["prevalence_0"])
                np.testing.assert_allclose(r[values == 1], radii["prevalence_1"])

    def test_coordinates_round_the_outside(self):
        _, _, ax = self.ring()
        ticks = artists(ax, "tick")
        self.assertEqual([t.get_text() for t in ticks],
                         ["0", "1 Mb", "2 Mb", "3 Mb", "4 Mb"])
        np.testing.assert_allclose([t.get_position()[0] for t in ticks],
                                   theta(np.arange(5) * 1e6, 4_500_000))

    def test_the_end_is_not_labelled_over_the_start(self):
        """10 kb ends on a tick, which is the top again."""
        _, _, ax = self.ring(length=10_000)
        self.assertEqual([t.get_text() for t in artists(ax, "tick")],
                         ["0", "2,000 bp", "4,000 bp", "6,000 bp", "8,000 bp"])

    def test_a_legend_names_each_group_and_its_size(self):
        fig, _, _ = self.ring()
        (legend,) = fig.legends
        self.assertEqual([t.get_text() for t in legend.get_texts()],
                         ["case  n=3", "control  n=2", "group mean"])

    def test_more_than_three_groups_are_refused_before_drawing(self):
        """A ring of four overlaid groups is unreadable, and lanes do not
        bend round; the linear plot carries them."""
        groups = ("a", "b", "c", "d")
        with mock.patch.object(Figure, "savefig", autospec=True) as save, \
                self.assertRaisesRegex(ValueError, "three"):
            circular_plot("out.png", title="G", length=2000,
                          sizes=dict.fromkeys(groups, 1),
                          overview=bins_table(groups, np.array([1, 1001, 2001])))
        save.assert_not_called()

    def test_masked_values_are_refused_before_drawing(self):
        table = bins_table(["a"], np.array([1, 1001, 2001]))
        table["mean"] = np.ma.masked_array(table["mean"], [True, False])
        with mock.patch.object(Figure, "savefig", autospec=True) as save, \
                self.assertRaisesRegex(ValueError, "mean"):
            circular_plot("out.png", title="G", length=2000, sizes={"a": 1},
                          overview=table)
        save.assert_not_called()

    def test_a_failed_save_leaves_no_figure_and_no_style_behind(self):
        before = dict(plt.rcParams)
        with mock.patch.object(Figure, "savefig", side_effect=OSError("full")), \
                self.assertRaises(OSError):
            circular_plot("out.png", title="G", length=2000, sizes={"a": 1},
                          overview=bins_table(["a"], np.array([1, 1001, 2001])))
        self.assertEqual(plt.get_fignums(), [])
        self.assertEqual(dict(plt.rcParams), before)


class RingOrfTests(RingTestCase):
    def track(self, highlight=(False,) * 5, **kwargs):
        return orf_track(TRACK_ORFS(), np.array(highlight), **kwargs)

    def test_orfs_sit_on_the_backbone_by_strand(self):
        """+ outside the backbone, - inside, . across it, and neutral."""
        _, _, ax = self.ring(length=10_000, track=self.track())
        radii = circular_layout(2)
        sides = {"+": (radii["backbone"], radii["orfs_out"]),
                 "-": (radii["orfs_in"], radii["backbone"])}
        for strand, count in (("+", 3), ("-", 1), (".", 1)):
            with self.subTest(strand=strand):
                orfs = artist(ax, f"orfs:{strand}")
                self.assertEqual(len(orfs.get_paths()), count)
                r = np.concatenate([p.vertices[:, 1] for p in orfs.get_paths()])
                if strand in sides:
                    self.assertGreaterEqual(r.min(), sides[strand][0] - 1e-12)
                    self.assertLessEqual(r.max(), sides[strand][1] + 1e-12)
                else:
                    self.assertLess(r.min(), radii["backbone"])
                    self.assertGreater(r.max(), radii["backbone"])
                self.assertEqual({to_hex(c) for c in orfs.get_facecolor()},
                                 {ORF_NEUTRAL})
        self.assertTrue(artists(ax, "backbone"))

    def test_orfs_follow_the_colour_mode(self):
        contrast = np.array([-3.0, 3.0, 0.0, 1.0, np.nan])
        track = self.track(contrast=contrast)
        fig, _, ax = self.ring(length=10_000, track=track)
        np.testing.assert_allclose(artist(ax, "orfs:+").get_facecolor(),
                                   contrast_colors(contrast[:3]))
        self.assertTrue(axes_by_gid(fig, "contrast-scale"))
        track = orf_track(orfs_of((1, 101, "+", "a", {"COG": "J"}),
                                  (201, 301, "-", "b", {"COG": "X"})),
                          np.zeros(2, bool), color_by="COG")
        _, _, ax = self.ring(length=1000, track=track)
        self.assertEqual(
            [t.get_text() for t in artist(ax, "orf-legend").get_texts()],
            ["J", "X", ORF_OTHER_LABEL],
        )

    def test_an_orf_across_the_origin_is_one_shape(self):
        """On the ring the origin is no edge: 9,501-10,500 on a 10,000 bp
        genome runs on past 2 pi, which the axes wrap."""
        track = orf_track(orfs_of((9501, 10501, "+", "a", {})), np.zeros(1, bool))
        _, _, ax = self.ring(length=10_000, track=track)
        (path,) = artist(ax, "orfs:+").get_paths()
        self.assertAlmostEqual(path.vertices[:, 0].min(), theta(9500, 10_000))
        self.assertAlmostEqual(path.vertices[:, 0].max(), theta(10_500, 10_000))

    def test_a_short_genome_shows_which_way_its_genes_run(self):
        """As on the linear plot, arrows up to `ARROW_MAX_BP`: an arrow's tip
        runs on past the end of its outer edge, where a box's is square."""
        outer = circular_layout(2)["orfs_out"]

        def tip_ahead(length):
            _, _, ax = self.ring(length=length, track=self.track())
            (path, *_) = artist(ax, "orfs:+").get_paths()
            edge = np.isclose(path.vertices[:, 1], outer)
            return path.vertices[:, 0].max() - path.vertices[edge, 0].max()

        self.assertGreater(tip_ahead(16_569), 0)
        self.assertAlmostEqual(tip_ahead(4_500_000), 0)

    def test_highlights_get_ink_a_wedge_and_a_label(self):
        """dnaA, dnaB and dnaC are within a point of each other at the
        backbone, so their wedges merge: one from 100 to 2,300, through
        depth and breadth."""
        track = self.track(highlight=(True, True, True, False, False))
        _, _, ax = self.ring(length=4_500_000, track=track)
        self.assertEqual(
            [to_hex(c) for c in artist(ax, "orfs:+").get_facecolor()], [INK] * 3
        )
        radii = circular_layout(2)
        (wedge,) = artist(ax, "highlight").get_paths()
        np.testing.assert_allclose(
            [wedge.vertices[:, 0].min(), wedge.vertices[:, 0].max()],
            theta(np.array([100, 2300]), 4_500_000),
        )
        np.testing.assert_allclose(
            [wedge.vertices[:, 1].min(), wedge.vertices[:, 1].max()],
            [radii["prevalence_1"], radii["depth_top"]],
        )
        labels = artists(ax, "label")
        unlabelled = artists(ax, "unlabelled")
        self.assertEqual(
            len(labels) + (int(unlabelled[0].get_text().split()[0])
                           if unlabelled else 0), 3)
        self.assertIn("dnaA", [t.get_text() for t in labels])

    def test_labels_that_fit_nowhere_are_counted(self):
        """Sixty highlighted ORFs within 6 kb: some labelled, the rest
        counted, none lost."""
        orfs = orfs_of(*[(1 + 100 * i, 91 + 100 * i, "+", f"g{i}", {})
                         for i in range(60)])
        track = orf_track(orfs, np.ones(60, bool))
        _, _, ax = self.ring(length=4_500_000, track=track)
        labels = artists(ax, "label")
        (unlabelled,) = artists(ax, "unlabelled")
        self.assertGreater(len(labels), 0)
        self.assertEqual(unlabelled.get_text(),
                         f"+{60 - len(labels)} unlabelled")
        # each label sits at the end of its leader, which starts at its ORF
        leaders = artists(ax, "leader")
        self.assertEqual(len(leaders), len(labels))
        names = [t.get_text() for t in labels]
        for label, leader in zip(labels, leaders, strict=True):
            i = int(label.get_text()[1:])
            self.assertAlmostEqual(leader.get_xdata()[0],
                                   theta(100 * i + 45, 4_500_000))
            self.assertAlmostEqual(leader.get_xdata()[-1], label.get_position()[0])
        self.assertEqual(len(set(names)), len(names))

    def test_a_real_png(self):
        d = mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        path = f"{d}/o'brien.png"
        circular_plot(path, title="GC", length=10_000, sizes={"a": 1, "b": 2},
                      overview=bins_table(["a", "b"],
                                          display_bin_edges(1, 10_001, 5)),
                      track=self.track(highlight=(True, False, False, True,
                                                  False)))
        with open(path, "rb") as fp:
            self.assertEqual(fp.read(8), b"\x89PNG\r\n\x1a\n")
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
