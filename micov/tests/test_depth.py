"""Tests for `depth-plot`'s computation.

Depth and breadth may come from different files -- metatranscriptomic depth
over metagenomic breadth, say -- so before anything is computed the two layers,
the metadata and the features have to be reconciled. Anything left out is
reported by name: a sample quietly missing from one layer would change a
group's median with nothing to show why.

The `dp_*` fixtures are described in `test_data/README.md`.
"""

import itertools
import re
import shutil
import unittest
from pathlib import Path
from tempfile import mkdtemp
from typing import ClassVar
from unittest import mock

import numpy as np

from micov import _depth
from micov._depth import (
    BREADTH_INTERVALS_TABLE,
    DEPTH_ALIGNMENTS_TABLE,
    DETAIL_MAX_BINS,
    GENOMES_TABLE,
    ROSTER_TABLE,
    WINDOW_CELLS,
    coverage_counts,
    detail_bin_bp,
    display_bin_edges,
    genome_bins,
    intersect_layers,
    quartiles_x4,
    stage_breadth,
    stage_depth,
    window_depth,
    window_size,
)
from micov._io import (
    load_alignment_layer,
    load_depth_features,
    load_orfs,
    load_sample_groups,
)
from micov._miint import connection

DATA = Path(__file__).parent / "test_data"


class DepthTestCase(unittest.TestCase):
    """The `dp_*` fixtures, read and reconciled the way the command will."""

    def setUp(self):
        self.d = mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.con = connection()
        self.addCleanup(self.con.close)

    def write(self, name, text):
        path = f"{self.d}/{name}"
        with open(path, "w") as fp:
            fp.write(text)
        return path

    def layer(self, name, select):
        """Write a layer Parquet derived from `dp_depth.parquet` by `select`."""
        path = f"{self.d}/{name}.parquet"
        self.con.sql(f"""COPY ({select.format(depth=f"'{DATA}/dp_depth.parquet'")})
                         TO '{path}' (FORMAT PARQUET)""")
        return path

    def resolve(self, depth=None, breadth=None, metadata=None, column="group",
                features=None, orfs=None):
        load_alignment_layer(
            self.con, str(depth or DATA / "dp_depth.parquet"), "depth_layer"
        )
        load_alignment_layer(
            self.con, str(breadth or DATA / "dp_breadth.parquet"), "breadth_layer"
        )
        load_sample_groups(
            self.con, str(metadata or DATA / "dp_metadata.tsv"), column
        )
        load_depth_features(self.con, str(features or DATA / "dp_regions.tsv"))
        if orfs is not None:
            load_orfs(self.con, str(orfs))
        intersect_layers(
            self.con, "depth_layer", "breadth_layer", orfs=orfs is not None
        )

    def resolve_quietly(self):
        """The two-file run, whose left-out samples and genomes are warned of."""
        with self.assertLogs("micov", level="WARNING"):
            self.resolve()


class IntersectLayersTests(DepthTestCase):
    def roster(self):
        return self.con.sql(f"FROM {ROSTER_TABLE} ORDER BY sample_idx").fetchall()

    def genomes(self):
        return self.con.sql(f"FROM {GENOMES_TABLE} ORDER BY 1").fetchall()

    def test_samples_are_those_in_the_metadata_and_both_layers(self):
        """S6 is breadth only, S7 depth only, S8 neither; S9 has no metadata."""
        with self.assertLogs("micov", level="WARNING"):
            self.resolve()
        self.assertEqual(
            self.roster(),
            [("S1", 0, "case"), ("S2", 1, "case"), ("S3", 2, "case"),
             ("S4", 3, "control"), ("S5", 4, "control")],
        )

    def test_genomes_are_those_in_the_features_and_both_layers(self):
        with self.assertLogs("micov", level="WARNING"):
            self.resolve()
        self.assertEqual(self.genomes(), [("GC", 3000, True), ("GL", 2000, False)])

    def test_every_sample_and_genome_left_out_is_named(self):
        with self.assertLogs("micov", level="WARNING") as logged:
            self.resolve()
        message = "\n".join(logged.output)
        for name in ("S6", "S7", "S8", "GX", "GB"):
            with self.subTest(left_out=name):
                self.assertIn(name, message)

    def test_unaligned_reads_and_unlisted_samples_are_not_reported(self):
        """`*` is no genome, and S9 was left out by the metadata, not by micov."""
        with self.assertLogs("micov", level="WARNING") as logged:
            self.resolve()
        message = "\n".join(logged.output)
        self.assertNotIn("*", message)
        self.assertNotIn("S9", message)

    def test_a_placed_unmapped_read_does_not_put_a_genome_in_a_layer(self):
        """S4's unmapped read sits on GC, but aligned nowhere."""
        depth = self.layer(
            "depth", "SELECT * FROM {depth} WHERE read_id = 'S4:both:d4'"
        )
        with self.assertRaisesRegex(ValueError, "sample"):
            self.resolve(depth=depth)

    def test_the_same_file_for_both_layers(self):
        """`--breadth` defaults to `--depth`: then no layer can lack anything.

        S7 and GX, depth-only in the two-file run, are now in both; S6 is in
        neither, which is still reported.
        """
        path = DATA / "dp_depth.parquet"
        with self.assertLogs("micov", level="WARNING") as logged:
            self.resolve(depth=path, breadth=path)
        self.assertEqual(
            [row[0] for row in self.roster()], ["S1", "S2", "S3", "S4", "S5", "S7"]
        )
        self.assertEqual([row[0] for row in self.genomes()], ["GC", "GL", "GX"])
        message = "\n".join(logged.output)
        self.assertNotIn("layer only", message)
        self.assertIn("S6", message)

    def test_no_sample_in_common_is_an_error(self):
        breadth = self.layer(
            "breadth", "SELECT * REPLACE ('Z' || sample_id AS sample_id) FROM {depth}"
        )
        with self.assertRaisesRegex(ValueError, "sample"):
            self.resolve(breadth=breadth)

    def test_no_genome_in_common_is_an_error(self):
        features = self.write("f.tsv", "genome_id\tlength\nGX\t1000\nGB\t1000\n")
        with self.assertRaisesRegex(ValueError, "genome"):
            self.resolve(features=features)

    def test_more_than_ten_groups_is_an_error_naming_them(self):
        """Ten is as many as can be drawn distinctly (see `_plot.group_style`)."""
        depth = self.layer(
            "eleven",
            "SELECT * REPLACE ('T' || k AS sample_id) "
            "FROM {depth}, range(11) t(k) WHERE read_id = 'S1:both:a1'",
        )
        metadata = self.write(
            "m.tsv", "sample_id\tgroup\n"
            + "".join(f"T{k}\tg{k:02d}\n" for k in range(11))
        )
        with self.assertRaises(ValueError) as raised:
            self.resolve(depth=depth, breadth=depth, metadata=metadata)
        for k in range(11):
            self.assertIn(f"g{k:02d}", str(raised.exception))

    def test_an_alignment_beyond_the_length_is_an_error(self):
        """A read at 2,951 on a 2,000 bp genome means the length is wrong."""
        features = self.write("f.tsv", "genome_id\tlength\nGC\t2000\nGL\t2000\n")
        with self.assertRaisesRegex(ValueError, "GC.*2000"):
            self.resolve(features=features)

    def test_orfs_on_no_plotted_genome_are_an_error(self):
        """An empty ORF track would look like a genome with no genes."""
        gff = self.write(
            "x.gff", "##gff-version 3\nGQ\tt\tCDS\t1\t30\t.\t+\t0\tID=q1\n"
        )
        orfs = f"{self.d}/orfs.parquet"
        self.con.sql(f"COPY (FROM read_gff('{gff}')) TO '{orfs}' (FORMAT PARQUET)")
        with self.assertRaisesRegex(ValueError, "GQ"):
            self.resolve(orfs=orfs)

    def test_orfs_on_other_genomes_are_ignored(self):
        """dp_orfs also has ORFs on GX (left out) and GQ (not a feature)."""
        with self.assertLogs("micov", level="WARNING"):
            self.resolve(orfs=DATA / "dp_orfs.parquet")
        self.assertEqual(len(self.roster()), 5)



class SyntheticTestCase(unittest.TestCase):
    """One genome, G, built from literal reads, bypassing the readers.

    Each read is `(sample_id, position, stop_position, cigar)`, with the stop
    worked out by hand the way htslib reports it: the position plus the
    reference the CIGAR consumes (M, D, N, = and X).
    """

    def setUp(self):
        self.con = connection()
        self.addCleanup(self.con.close)

    def build(self, groups, reads, length=8):
        """`groups` maps each sample to its group; reads go in both layers."""
        self.n = len(groups)
        self.con.execute(f"""CREATE OR REPLACE TABLE {ROSTER_TABLE}
                             (sample_id VARCHAR, sample_idx INTEGER,
                              group_name VARCHAR)""")
        self.con.executemany(
            f"INSERT INTO {ROSTER_TABLE} VALUES (?, ?, ?)",
            [(s, i, groups[s]) for i, s in enumerate(sorted(groups))],
        )
        self.con.execute(
            f"""CREATE OR REPLACE TABLE {GENOMES_TABLE} AS
                SELECT 'G' AS genome_id, ?::BIGINT AS length,
                       false AS is_circular""",
            [length],
        )
        self.con.execute("""CREATE OR REPLACE TABLE layer
                            (sample_id VARCHAR, reference VARCHAR,
                             position BIGINT, stop_position BIGINT,
                             cigar VARCHAR)""")
        self.con.executemany(
            "INSERT INTO layer VALUES (?, 'G', ?, ?, ?)", reads
        )
        stage_breadth(self.con, "layer")

    def bins(self, edge_sets, length=8):
        return genome_bins(self.con, "layer", "G", length, edge_sets)


class StageDepthTests(DepthTestCase):
    def test_one_genomes_aligned_reads_of_used_samples_by_position(self):
        """The window pass reads this table once per window, so it holds only
        what can add depth there: S4's placed unmapped read d4 would otherwise
        reach the window query, and S7 (depth only) and S9 (no metadata) are
        not plotted. Both of S1's alignments at 1 and 1201 stay -- a
        secondary alignment is still a read on that genome.
        """
        self.resolve_quietly()
        stage_depth(self.con, "depth_layer", "GC")
        rows = self.con.sql(f"FROM {DEPTH_ALIGNMENTS_TABLE}").fetchall()
        positions = [row[1] for row in rows]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(
            sorted(rows),
            [(0, 1, 101, "100M"), (0, 951, 1051, "100M"),
             (0, 1201, 1301, "100M"), (0, 2951, 3001, "50M"),
             (1, 1001, 1101, "50M10D40M"), (1, 1251, 1351, "100M"),
             (2, 1201, 1301, "100M"), (2, 2601, 2761, "30M100N30M"),
             (3, 101, 201, "100M"), (3, 1221, 1321, "100M"),
             (3, 1221, 1321, "100M"), (4, 2001, 2101, "100M")],
        )

    def test_coordinates_are_narrow(self):
        """About 28 bytes a read: a 10 Mb genome's million reads stay small."""
        self.resolve_quietly()
        stage_depth(self.con, "depth_layer", "GL")
        self.assertEqual(
            self.con.sql(f"DESCRIBE {DEPTH_ALIGNMENTS_TABLE}").fetchall(),
            [("sample_idx", "INTEGER", "YES", None, None, None),
             ("position", "UINTEGER", "YES", None, None, None),
             ("stop_position", "UINTEGER", "YES", None, None, None),
             ("cigar", "VARCHAR", "YES", None, None, None)],
        )


class WindowDepthTests(SyntheticTestCase):
    """Per-base depth in one window: row i is sample_idx i, column j base w0+j."""

    def depth(self, reads, w0=1, w1=9, groups=None, length=8):
        self.build(groups or {"S0": "a"}, reads, length)
        stage_depth(self.con, "layer", "G")
        return window_depth(self.con, self.n, w0, w1).tolist()

    def test_coordinates_are_half_open(self):
        """[2, 5) covers 2, 3 and 4."""
        self.assertEqual(
            self.depth([("S0", 2, 5, "3M")]), [[0, 1, 1, 1, 0, 0, 0, 0]]
        )

    def test_a_deletion_counts(self):
        """The read spans the deleted bases; depth is how much DNA was there."""
        self.assertEqual(
            self.depth([("S0", 1, 7, "2M2D2M")]), [[1, 1, 1, 1, 1, 1, 0, 0]]
        )

    def test_a_skip_does_not(self):
        """N is an intron or a gap between mates: nothing was sequenced there."""
        self.assertEqual(
            self.depth([("S0", 1, 7, "2M2N2M")]), [[1, 1, 0, 0, 1, 1, 0, 0]]
        )

    def test_clips_and_insertions_consume_no_reference(self):
        self.assertEqual(
            self.depth([("S0", 1, 4, "2S3M2H")]), [[1, 1, 1, 0, 0, 0, 0, 0]]
        )
        self.assertEqual(
            self.depth([("S0", 1, 5, "2M2I2M")]), [[1, 1, 1, 1, 0, 0, 0, 0]]
        )

    def test_a_window_edge_inside_a_deletion_or_skip(self):
        """Windows are an implementation detail; they must not show."""
        for cigar, expected in (
            ("2M2D2M", [1, 1, 1, 1, 1, 1, 0, 0]),
            ("2M2N2M", [1, 1, 0, 0, 1, 1, 0, 0]),
        ):
            with self.subTest(cigar=cigar):
                reads = [("S0", 1, 7, cigar)]
                left = self.depth(reads, 1, 4)[0]
                right = self.depth(reads, 4, 9)[0]
                self.assertEqual(left + right, expected)

    def test_a_read_starting_before_the_window(self):
        self.assertEqual(self.depth([("S0", 1, 7, "6M")], 4, 6), [[1, 1]])

    def test_an_overhang_past_the_genome_is_clipped(self):
        self.assertEqual(
            self.depth([("S0", 7, 11, "4M")]), [[0, 0, 0, 0, 0, 0, 1, 1]]
        )

    def test_reads_stack_and_samples_stay_apart(self):
        reads = [("S0", 2, 5, "3M"), ("S0", 3, 6, "3M"), ("S1", 1, 3, "2M")]
        self.assertEqual(
            self.depth(reads, groups={"S0": "a", "S1": "a"}),
            [[0, 1, 2, 2, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]],
        )

    def test_a_sample_with_no_reads_in_the_window_is_zeros(self):
        """It still counts towards its group's quantiles, as depth 0."""
        reads = [("S0", 1, 3, "2M"), ("S2", 2, 4, "2M"), ("S1", 7, 9, "2M")]
        groups = {"S0": "a", "S1": "a", "S2": "a"}
        self.assertEqual(
            self.depth(reads, 1, 5, groups),
            [[1, 1, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]],
        )

    def test_a_read_miint_skips_is_zeros(self):
        """miint skips a NULL CIGAR, and a sample whose only reads it skips
        aggregates to NULL rather than zeros."""
        reads = [("S0", 1, 3, None), ("S1", 1, 3, "2M")]
        self.assertEqual(
            self.depth(reads, 1, 5, {"S0": "a", "S1": "a"}),
            [[0, 0, 0, 0], [1, 1, 0, 0]],
        )


class WindowSizeTests(unittest.TestCase):
    def test_a_window_holds_at_most_window_cells(self):
        """Memory is samples x window, whatever the genome's length."""
        for n in (1, 3, 300, 10**6):
            with self.subTest(samples=n):
                self.assertLessEqual(window_size(n, 10**9) * n, WINDOW_CELLS)
        self.assertEqual(window_size(4, 10**9), WINDOW_CELLS // 4)

    def test_a_short_genome_is_one_window(self):
        self.assertEqual(window_size(5, 3000), 3000)

    def test_never_empty(self):
        self.assertEqual(window_size(WINDOW_CELLS * 2, 3000), 1)


class StageBreadthTests(DepthTestCase):
    def test_each_samples_reads_merge_into_intervals(self):
        """A sample counts once toward prevalence however many of its reads
        overlap: S4's two reads at 1221 are one interval, as are S1's
        overlapping reads at 101 and 151 on GL. S4's unmapped read at 1500
        covers nothing (it would otherwise widen to [0, 1500)), and S3's
        skip counts: breadth is where the read is, depth where its bases are.
        GX, GB and S6 to S9 are not plotted.
        """
        self.resolve_quietly()
        stage_breadth(self.con, "breadth_layer")
        self.assertEqual(
            self.con.sql(f"FROM {BREADTH_INTERVALS_TABLE} ORDER BY ALL").fetchall(),
            [("GC", 0, 1, 101), ("GC", 0, 951, 1051), ("GC", 0, 1201, 1301),
             ("GC", 0, 2951, 3001),
             ("GC", 1, 1001, 1101), ("GC", 1, 1251, 1351),
             ("GC", 2, 1201, 1301), ("GC", 2, 2601, 2761),
             ("GC", 3, 101, 201), ("GC", 3, 1221, 1321),
             ("GC", 4, 2001, 2101),
             ("GL", 0, 101, 251), ("GL", 0, 1801, 1851),
             ("GL", 1, 1, 101), ("GL", 2, 1001, 1101),
             ("GL", 3, 501, 601), ("GL", 4, 501, 601)],
        )


class CoverageCountsTests(unittest.TestCase):
    """How many of the given (merged, so one per sample) intervals cover each
    base of [w0, w1). The caller sorts the starts and stops, as
    `genome_bins` does once per genome."""

    @staticmethod
    def counts(starts, stops, w0, w1):
        return coverage_counts(
            np.sort(np.array(starts, np.int64)), np.sort(np.array(stops, np.int64)),
            w0, w1,
        ).tolist()

    def test_counts_covering_intervals(self):
        self.assertEqual(
            self.counts([1, 3], [5, 8], 1, 9), [1, 1, 2, 2, 1, 1, 1, 0]
        )

    def test_touching_intervals_do_not_double_count(self):
        self.assertEqual(
            self.counts([1, 5], [5, 8], 1, 9), [1, 1, 1, 1, 1, 1, 1, 0]
        )

    def test_clipped_to_the_window(self):
        """[1, 4) ends where the window starts, and [10, 12) is beyond it."""
        self.assertEqual(self.counts([1, 10, 1], [10, 12, 4], 4, 6), [1, 1])

    def test_no_intervals(self):
        self.assertEqual(self.counts([], [], 1, 4), [0, 0, 0])


class QuartilesTests(unittest.TestCase):
    """Q1, median and Q3 across samples (rows) at each base, times four.

    numpy's default (linear) method on integer depths lands on multiples of
    0.25, so four times it is an exact integer, and bin sums of it do not
    depend on the order bases were added in.
    """

    def quartiles(self, column):
        return quartiles_x4(np.array([[d] for d in column], np.uint32))[:, 0]

    def test_two_samples_interpolate(self):
        """{1, 4}: 1.75, 2.5, 3.25."""
        self.assertEqual(self.quartiles([4, 1]).tolist(), [7, 10, 13])

    def test_four_samples(self):
        """{0, 0, 2, 6}: 0, 1, 3."""
        self.assertEqual(self.quartiles([6, 0, 2, 0]).tolist(), [0, 4, 12])

    def test_one_sample_of_three_moves_only_the_upper_quartile(self):
        """{0, 0, 3}: the median is 0 though the mean is 1."""
        self.assertEqual(self.quartiles([0, 3, 0]).tolist(), [0, 0, 6])

    def test_numpys_default_method_exactly(self):
        rng = np.random.default_rng(1)
        for n in range(1, 9):
            with self.subTest(samples=n):
                depth = rng.integers(0, 10**6, (n, 50), dtype=np.uint32)
                expected = np.quantile(depth, (0.25, 0.5, 0.75), axis=0) * 4
                self.assertTrue(np.array_equal(expected, np.round(expected)))
                self.assertEqual(quartiles_x4(depth).tolist(), expected.tolist())


class BinEdgesTests(unittest.TestCase):
    def test_overview_bins(self):
        self.assertEqual(
            display_bin_edges(1, 3001, 1000).tolist(), [1, 1001, 2001, 3001]
        )

    def test_the_last_bin_ends_at_the_genome(self):
        self.assertEqual(
            display_bin_edges(1, 2951, 1000).tolist(), [1, 1001, 2001, 2951]
        )

    def test_a_detail_bin_is_as_small_as_the_bin_limit_allows(self):
        """Single bases up to 1,500 of them, so a short region shows each."""
        for start, stop, bp in ((1001, 1501, 1), (1, 1501, 1), (1, 1502, 2),
                                (1, 3001, 2), (1, 4502, 4)):
            with self.subTest(start=start, stop=stop):
                self.assertEqual(detail_bin_bp(start, stop), bp)
                edges = display_bin_edges(start, stop, bp)
                self.assertLessEqual(len(edges) - 1, DETAIL_MAX_BINS)
                self.assertEqual((edges[0], edges[-1]), (start, stop))


class GenomeBinsTests(SyntheticTestCase):
    COLUMNS: ClassVar[list] = ["group", "bin_start", "bin_stop", "q1", "median",
                               "q3", "mean", "prevalence", "union"]

    def bin_of(self, table, group, start):
        (row,) = np.flatnonzero(
            (table["group"] == group) & (table["bin_start"] == start)
        )
        return {key: table[key][row].item() for key in self.COLUMNS[3:]}

    def test_columns(self):
        self.build({"S0": "a"}, [("S0", 1, 3, "2M")])
        (table,) = self.bins([np.array([1, 5, 9])])
        self.assertEqual(list(table), self.COLUMNS)
        self.assertEqual(
            [table[c].dtype.kind for c in self.COLUMNS],
            ["O", "i", "i", "f", "f", "f", "f", "f", "b"],
        )

    def test_rows_are_group_major_in_sorted_order(self):
        self.build({"S0": "b", "S1": "a"}, [("S0", 1, 3, "2M")])
        (table,) = self.bins([np.array([1, 5, 9])])
        self.assertEqual(table["group"].tolist(), ["a", "a", "b", "b"])
        self.assertEqual(table["bin_start"].tolist(), [1, 5, 1, 5])
        self.assertEqual(table["bin_stop"].tolist(), [5, 9, 5, 9])

    def test_a_bin_averages_per_base_quantiles(self):
        """Per base first, then the bin: S0 has depth 3 at base 1 and S2 at
        base 2, so the median is 0 at both, and the bin's is 0. Taking each
        sample's bin mean first (1.5, 0, 1.5) would give 1.5, a median depth
        no base has.
        """
        reads = [("S0", 1, 2, "1M")] * 3 + [("S2", 2, 3, "1M")] * 3
        self.build({"S0": "a", "S1": "a", "S2": "a"}, reads)
        (table,) = self.bins([np.array([1, 3, 9])])
        self.assertEqual(
            self.bin_of(table, "a", 1),
            {"q1": 0.0, "median": 0.0, "q3": 1.5, "mean": 1.0,
             "prevalence": 1 / 3, "union": True},
        )

    def test_one_sample_of_three(self):
        reads = [("S0", 1, 2, "1M")] * 3
        self.build({"S0": "a", "S1": "a", "S2": "a"}, reads)
        (table,) = self.bins([np.array([1, 2])])
        self.assertEqual(
            self.bin_of(table, "a", 1),
            {"q1": 0.0, "median": 0.0, "q3": 1.5, "mean": 1.0,
             "prevalence": 1 / 3, "union": True},
        )

    def test_a_bin_is_in_the_union_if_any_base_is_covered(self):
        """Prevalence is the share of (sample, base) pairs covered."""
        self.build({"S0": "a", "S1": "a"}, [("S0", 4, 5, "1M")])
        (table,) = self.bins([np.array([1, 5, 9])])
        self.assertEqual(self.bin_of(table, "a", 1)["union"], True)
        self.assertEqual(self.bin_of(table, "a", 1)["prevalence"], 1 / 8)
        self.assertEqual(self.bin_of(table, "a", 5)["union"], False)

    def test_a_skip_counts_toward_breadth_not_depth(self):
        self.build({"S0": "a"}, [("S0", 1, 7, "2M2N2M")])
        (table,) = self.bins([np.arange(1, 10)])
        self.assertEqual(table["median"].tolist(), [1, 1, 0, 0, 1, 1, 0, 0])
        self.assertEqual(table["prevalence"].tolist(), [1, 1, 1, 1, 1, 1, 0, 0])
        self.assertEqual(table["union"].tolist(), [True] * 6 + [False] * 2)

    def test_a_sample_in_another_group_adds_nothing(self):
        self.build({"S0": "a", "S1": "b"}, [("S1", 1, 5, "4M")])
        (table,) = self.bins([np.array([1, 5, 9])])
        self.assertEqual(
            self.bin_of(table, "a", 1),
            {"q1": 0.0, "median": 0.0, "q3": 0.0, "mean": 0.0,
             "prevalence": 0.0, "union": False},
        )
        self.assertEqual(self.bin_of(table, "b", 1)["median"], 1.0)

    def test_several_bin_sets_from_one_pass(self):
        """The overview and each detail region come from the same windows."""
        self.build({"S0": "a"}, [("S0", 2, 5, "3M")])
        overview, detail = self.bins([np.array([1, 5, 9]), np.arange(3, 7)])
        self.assertEqual(overview["median"].tolist(), [0.75, 0.0])
        self.assertEqual(detail["median"].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(detail["bin_start"].tolist(), [3, 4, 5])


class FixtureBinsTests(DepthTestCase):
    """The dp fixture, worked by hand from `dp.sam` (README design tables)."""

    def bins(self, genome, length, edge_sets):
        self.resolve_quietly()
        stage_breadth(self.con, "breadth_layer")
        return genome_bins(self.con, "depth_layer", genome, length, edge_sets)

    def assertBins(self, table, expected):
        for (group, start), values in expected.items():
            (row,) = np.flatnonzero(
                (table["group"] == group) & (table["bin_start"] == start)
            )
            for key, value in values.items():
                with self.subTest(group=group, start=start, column=key):
                    self.assertAlmostEqual(table[key][row], value, places=12)

    def test_gl_overview(self):
        """Case (S1-S3, n 3) in [1, 1001): S2 has depth 1 on 1-100, S1 1 on
        101-150, 2 on 151-200 and 1 on 201-250. The per-base Q3 is 0.5, 0.5,
        1 and 0.5 there, summing to 150 over the bin; the depth sums to 300;
        the breadth to 150 + 100. In [1001, 2001), S3 has 100 bases of depth
        1 and S1 50 of breadth only (a6). Control (S4, S5, n 2) both have
        depth 1 on 501-600, then nothing.
        """
        (table,) = self.bins("GL", 2000, [display_bin_edges(1, 2001, 1000)])
        self.assertBins(table, {
            ("case", 1): {"q1": 0, "median": 0, "q3": 0.15, "mean": 0.1,
                          "prevalence": 250 / 3000, "union": True},
            ("case", 1001): {"q1": 0, "median": 0, "q3": 0.05,
                             "mean": 100 / 3000, "prevalence": 150 / 3000,
                             "union": True},
            ("control", 1): {"q1": 0.1, "median": 0.1, "q3": 0.1, "mean": 0.1,
                             "prevalence": 0.1, "union": True},
            ("control", 1001): {"q1": 0, "median": 0, "q3": 0, "mean": 0,
                                "prevalence": 0, "union": False},
        })

    def test_gc_detail_regions_show_single_bases(self):
        """At 1050 S1 (a2) and S2 (b1) both have depth; at 1055 only S2's
        deletion does. At 1250 S1 and S3 do, and S4 twice. At 2650 S3's
        read spans the base with a skip: breadth, no depth.
        """
        tables = self.bins("GC", 3000, [display_bin_edges(1001, 1501, 1),
                                        display_bin_edges(2501, 2901, 1)])
        self.assertBins(tables[0], {
            ("case", 1050): {"q1": 0.5, "median": 1, "q3": 1, "mean": 2 / 3,
                             "prevalence": 2 / 3, "union": True},
            ("case", 1055): {"q1": 0, "median": 0, "q3": 0.5, "mean": 1 / 3,
                             "prevalence": 1 / 3, "union": True},
            ("control", 1055): {"q1": 0, "median": 0, "q3": 0, "mean": 0,
                                "prevalence": 0, "union": False},
            ("case", 1250): {"q1": 0.5, "median": 1, "q3": 1, "mean": 2 / 3,
                             "prevalence": 2 / 3, "union": True},
            ("control", 1250): {"q1": 0.5, "median": 1, "q3": 1.5, "mean": 1,
                                "prevalence": 0.5, "union": True},
        })
        self.assertBins(tables[1], {
            ("case", 2650): {"q1": 0, "median": 0, "q3": 0, "mean": 0,
                             "prevalence": 1 / 3, "union": True},
        })


def random_reads(rng, samples, length, count):
    """Reads with every CIGAR operation depth treats differently, some
    running past the genome's end."""
    def sizes(k):
        return [int(x) for x in rng.integers(1, 6, k)]

    reads = []
    for _ in range(count):
        a, b, c = sizes(3)
        cigar = [f"{a}M", f"{a}M{b}D{c}M", f"{a}M{b}N{c}M",
                 f"{b}S{a}M{c}I{b}M", f"{a}M{c}S"][rng.integers(5)]
        consumed = sum(int(k) for k, op in re.findall(r"(\d+)([MDN=X])", cigar))
        position = int(rng.integers(1, length + 1))
        reads.append((samples[rng.integers(len(samples))], position,
                      position + consumed, cigar))
    return reads


def oracle_bins(groups, reads, length, edges):
    """Per-base depth by walking each CIGAR in Python, then binned with
    numpy: independent of miint and of the windows."""
    samples = sorted(groups)
    depth = np.zeros((len(samples), length + 1), np.int64)
    cover = np.zeros((len(samples), length + 1), bool)
    for sample, position, stop, cigar in reads:
        i, at = samples.index(sample), position
        for count, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
            if op in "M=XD":
                depth[i, at : at + int(count)] += 1
            if op in "M=XDN":
                at += int(count)
        cover[i, position:stop] = True
    depth, cover = depth[:, 1:], cover[:, 1:]
    table = {key: [] for key in GenomeBinsTests.COLUMNS}
    for group in sorted(set(groups.values())):
        rows = [i for i, s in enumerate(samples) if groups[s] == group]
        q1, median, q3 = np.quantile(depth[rows], (0.25, 0.5, 0.75), axis=0)
        per_base = {"q1": q1, "median": median, "q3": q3,
                    "mean": depth[rows].mean(axis=0),
                    "prevalence": cover[rows].mean(axis=0)}
        for start, stop in itertools.pairwise(edges):
            table["group"].append(group)
            table["bin_start"].append(start)
            table["bin_stop"].append(stop)
            for key, values in per_base.items():
                table[key].append(values[start - 1 : stop - 1].mean())
            table["union"].append(cover[rows, start - 1 : stop - 1].any())
    return table


class WindowInvarianceTests(SyntheticTestCase):
    LENGTH = 61
    GROUPS: ClassVar[dict] = {"S0": "a", "S1": "b", "S2": "a", "S3": "b",
                              "S4": "a"}

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(2026)
        self.reads = random_reads(rng, sorted(self.GROUPS), self.LENGTH, 40)
        self.build(self.GROUPS, self.reads, self.LENGTH)
        self.edge_sets = [display_bin_edges(1, self.LENGTH + 1, 7),
                          display_bin_edges(10, 40, 1)]

    def test_any_window_size_gives_identical_bins(self):
        """Bit for bit: every bin is integer sums divided once, so the order
        windows add them in cannot show."""
        n = len(self.GROUPS)
        whole = self.bins(self.edge_sets, self.LENGTH)
        for width in (1, 2, 3, 7):
            with mock.patch.object(_depth, "WINDOW_CELLS", width * n):
                self.assertEqual(window_size(n, self.LENGTH), width)
                windowed = self.bins(self.edge_sets, self.LENGTH)
            for tables in zip(whole, windowed, strict=True):
                for key in GenomeBinsTests.COLUMNS:
                    with self.subTest(width=width, column=key):
                        self.assertEqual(
                            tables[0][key].tolist(), tables[1][key].tolist()
                        )

    def test_bins_match_a_per_base_oracle(self):
        with mock.patch.object(_depth, "WINDOW_CELLS", 3 * len(self.GROUPS)):
            tables = self.bins(self.edge_sets, self.LENGTH)
        for edges, table in zip(self.edge_sets, tables, strict=True):
            expected = oracle_bins(self.GROUPS, self.reads, self.LENGTH, edges)
            for key in GenomeBinsTests.COLUMNS:
                with self.subTest(bins=len(edges) - 1, column=key):
                    if table[key].dtype.kind == "f":
                        np.testing.assert_allclose(
                            table[key], expected[key], rtol=1e-12, atol=1e-15
                        )
                    else:
                        self.assertEqual(table[key].tolist(), expected[key])


if __name__ == "__main__":
    unittest.main()
