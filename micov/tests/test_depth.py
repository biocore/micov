"""Tests for `depth-plot`'s computation.

Depth and breadth may come from different files -- metatranscriptomic depth
over metagenomic breadth, say -- so before anything is computed the two layers,
the metadata and the features have to be reconciled. Anything left out is
reported by name: a sample quietly missing from one layer would change a
group's median with nothing to show why.

The `dp_*` fixtures are described in `test_data/README.md`.
"""

import itertools
import math
import re
import shutil
import unittest
import warnings
from pathlib import Path
from tempfile import mkdtemp
from typing import ClassVar
from unittest import mock

import numpy as np

from micov import _depth
from micov._depth import (
    BREADTH_INTERVALS_TABLE,
    DEPTH_ALIGNMENTS_TABLE,
    DEPTH_READS_TABLE,
    DETAIL_MAX_BINS,
    GENOMES_TABLE,
    OVERVIEW_ROW_BINS,
    ROSTER_TABLE,
    ROW_BP,
    WINDOW_CELLS,
    coverage_counts,
    detail_bin_bp,
    display_bin_edges,
    genome_orfs,
    genome_statistics,
    intersect_layers,
    orf_contrast,
    orf_segments,
    overview_bin_bp,
    overview_row_bp,
    quartiles_x4,
    stage_breadth,
    stage_depth,
    stage_depth_reads,
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
from micov._utils import sql_string

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
        depth = sql_string(DATA / "dp_depth.parquet")
        self.con.sql(f"""COPY ({select.format(depth=depth)})
                         TO {sql_string(path)} (FORMAT PARQUET)""")
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
        """With the layer it lacks: that is the file to look at."""
        with self.assertLogs("micov", level="WARNING") as logged:
            self.resolve()
        for name, where in (("S6", "breadth layer only"),
                            ("S7", "depth layer only"), ("S8", "neither layer"),
                            ("GX", "depth layer only"),
                            ("GB", "breadth layer only")):
            with self.subTest(left_out=name):
                (line,) = [line for line in logged.output if name in line]
                self.assertIn(where, line)

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
        self.con.sql(f"COPY (FROM read_gff({sql_string(gff)})) "
                     f"TO {sql_string(orfs)} (FORMAT PARQUET)")
        with self.assertRaisesRegex(ValueError, "GQ"):
            self.resolve(orfs=orfs)

    def orfs_from_gff(self, lines):
        gff = self.write("o.gff", "##gff-version 3\n" + lines)
        path = f"{self.d}/o.parquet"
        self.con.sql(f"COPY (FROM read_gff({sql_string(gff)})) "
                     f"TO {sql_string(path)} (FORMAT PARQUET)")
        return path

    def test_orfs_outside_their_genome_are_an_error_naming_them(self):
        """Checked before any genome is computed, so a run that fails here
        writes nothing. Only a circular genome's ORF may run past its end,
        across the origin (gc_7 in dp_orfs does, and is fine)."""
        for line, why in (
            ("GL\tt\tCDS\t1951\t2050\t.\t+\t0\tID=bad\n", "past a linear end"),
            ("GC\tt\tCDS\t3001\t3100\t.\t+\t0\tID=bad\n", "starts beyond"),
            ("GC\tt\tCDS\t2001\t5100\t.\t+\t0\tID=bad\n", "longer than GC"),
        ):
            with self.subTest(why):
                orfs = self.orfs_from_gff(
                    "GL\tt\tCDS\t1\t30\t.\t+\t0\tID=fine\n" + line
                )
                with (self.assertLogs("micov", level="WARNING"),
                      self.assertRaisesRegex(ValueError, r"bad") as raised):
                    self.resolve(orfs=orfs)
                self.assertNotIn("fine", str(raised.exception))

    def test_orfs_on_other_genomes_are_ignored(self):
        """dp_orfs also has ORFs on GX (left out) and GQ (not a feature)."""
        with self.assertLogs("micov", level="WARNING"):
            self.resolve(orfs=DATA / "dp_orfs.parquet")
        self.assertEqual(len(self.roster()), 5)

    def test_an_orf_ending_where_it_starts_or_before_is_an_error(self):
        """`read_gff` passes such a line through, and its statistics would be
        a sum taken backwards, or over no bases: -0.0 or NaN in the table."""
        for line, why in (
            ("GL\tt\tCDS\t301\t250\t.\t+\t0\tID=bad\n", "ends before it starts"),
            ("GL\tt\tCDS\t31\t30\t.\t+\t0\tID=bad\n", "spans no base"),
            ("GL\tt\tCDS\t0\t30\t.\t+\t0\tID=bad\n", "starts before base 1"),
        ):
            with self.subTest(why):
                orfs = self.orfs_from_gff(
                    "GL\tt\tCDS\t1\t30\t.\t+\t0\tID=fine\n" + line
                )
                with (self.assertLogs("micov", level="WARNING"),
                      self.assertRaisesRegex(ValueError, r"bad") as raised):
                    self.resolve(orfs=orfs)
                self.assertNotIn("fine", str(raised.exception))

    def test_an_orf_without_an_id_on_a_plotted_genome_is_an_error(self):
        """The ID is the row's key in the per-ORF table."""
        orfs = self.orfs_from_gff("GL\tt\tCDS\t1\t30\t.\t+\t0\tgene=abc\n")
        with (self.assertLogs("micov", level="WARNING"),
              self.assertRaisesRegex(ValueError, "ID.*GL:1")):
            self.resolve(orfs=orfs)

    def test_an_orf_without_an_id_elsewhere_is_ignored(self):
        """A database-wide GFF is fine: ORFs on genomes not plotted are not
        read, so they cannot stop the run."""
        orfs = self.orfs_from_gff("GL\tt\tCDS\t1\t30\t.\t+\t0\tID=fine\n"
                                  "GQ\tt\tCDS\t1\t30\t.\t+\t0\tgene=abc\n")
        with self.assertLogs("micov", level="WARNING"):
            self.resolve(orfs=orfs)
        self.assertEqual(len(self.genomes()), 2)

    def test_a_genome_only_unlisted_samples_align_to_is_left_out_and_named(self):
        """S9 has no metadata, so its reads on GZ add nothing: plotted, GZ
        would be a flat line that looked like a genome no sample carries."""
        add_s9 = ("SELECT * FROM read_parquet({path}) UNION ALL "
                  "SELECT * REPLACE ('GZ' AS reference) "
                  "FROM read_parquet({path}) WHERE sample_id = 'S9'")
        depth = self.layer(
            "d", add_s9.format(path=sql_string(DATA / "dp_depth.parquet")))
        breadth = self.layer(
            "b", add_s9.format(path=sql_string(DATA / "dp_breadth.parquet")))
        features = self.write("f.tsv", "genome_id\tlength\nGC\t3000\nGL\t2000\n"
                                       "GZ\t3000\n")
        with self.assertLogs("micov", level="WARNING") as logged:
            self.resolve(depth=depth, breadth=breadth, features=features)
        self.assertEqual([row[0] for row in self.genomes()], ["GC", "GL"])
        (line,) = [line for line in logged.output if "GZ" in line]
        self.assertIn("from the samples used", line)



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
        stage_depth_reads(self.con, "layer")

    def bins(self, edge_sets, length=8):
        return genome_statistics(self.con, "G", length, edge_sets)[0]


class StageDepthTests(DepthTestCase):
    def test_one_genomes_aligned_reads_of_used_samples_by_position(self):
        """The window pass reads this table once per window, so it holds only
        what can add depth there: S4's placed unmapped read d4 would otherwise
        reach the window query, and S7 (depth only) and S9 (no metadata) are
        not plotted. Both of S1's alignments at 1 and 1201 stay -- a
        secondary alignment is still a read on that genome.
        """
        self.resolve_quietly()
        stage_depth_reads(self.con, "depth_layer")
        stage_depth(self.con, "GC")
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
        stage_depth_reads(self.con, "depth_layer")
        stage_depth(self.con, "GL")
        self.assertEqual(
            self.con.sql(f"DESCRIBE {DEPTH_ALIGNMENTS_TABLE}").fetchall(),
            [("sample_idx", "INTEGER", "YES", None, None, None),
             ("position", "UINTEGER", "YES", None, None, None),
             ("stop_position", "UINTEGER", "YES", None, None, None),
             ("cigar", "VARCHAR", "YES", None, None, None)],
        )


class StageDepthReadsTests(DepthTestCase):
    """The depth layer, staged once for every genome."""

    def test_plotted_reads_stored_by_genome_then_position(self):
        """Each genome's `stage_depth` then reads only its own row groups.
        Scanning the layer for every genome made each genome cost more the
        bigger the whole input was. Only what can add depth is kept: not
        S4's placed unmapped read, not S7 (depth only) or S9 (no metadata),
        not GX, which is left out."""
        self.resolve_quietly()
        stage_depth_reads(self.con, "depth_layer")
        rows = self.con.sql(f"""SELECT sample_idx, reference, position
                                FROM {DEPTH_READS_TABLE}
                                ORDER BY rowid""").fetchall()
        self.assertEqual([row[1:] for row in rows],
                         sorted(row[1:] for row in rows))
        self.assertEqual({row[1] for row in rows}, {"GC", "GL"})
        # S1 to S5, numbered in sample_id order (`ROSTER_TABLE`)
        self.assertEqual({row[0] for row in rows}, {0, 1, 2, 3, 4})
        self.assertNotIn((3, "GC", 1500), rows)


class WindowDepthTests(SyntheticTestCase):
    """Per-base depth in one window: row i is sample_idx i, column j base w0+j."""

    def depth(self, reads, w0=1, w1=9, groups=None, length=8):
        self.build(groups or {"S0": "a"}, reads, length)
        stage_depth(self.con, "G")
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
    `genome_statistics` does once per genome."""

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

    def test_an_overview_row_is_the_genome_up_to_two_megabases(self):
        """A mitochondrion is one row of its own length, not a sliver of a
        2 Mb row; a larger genome wraps into rows of `ROW_BP`."""
        for length, row in ((16_569, 16_569), (ROW_BP, ROW_BP),
                            (10_000_000, ROW_BP)):
            with self.subTest(length=length):
                self.assertEqual(overview_row_bp(length), row)

    def test_an_overview_row_has_at_most_two_thousand_bins(self):
        """1 kb on any genome of 2 Mb or more; a 16.6 kb mitochondrion gets
        9 bp, resolving its genes rather than 17 blocks."""
        for length, bp in ((1, 1), (2_000, 1), (2_001, 2), (16_569, 9),
                           (50_000, 25), (ROW_BP, 1_000), (ROW_BP + 1, 1_000),
                           (10_000_000, 1_000)):
            with self.subTest(length=length):
                self.assertEqual(overview_bin_bp(length), bp)
                row = overview_row_bp(length)
                self.assertLessEqual(len(display_bin_edges(1, row + 1, bp)) - 1,
                                     OVERVIEW_ROW_BINS)


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
        stage_depth_reads(self.con, "depth_layer")
        return genome_statistics(self.con, genome, length, edge_sets)[0]

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


def oracle_per_base(groups, reads, length):
    """Per-base depth and breadth by walking each CIGAR in Python:
    independent of miint and of the windows. Rows are samples in order."""
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
    return samples, depth[:, 1:], cover[:, 1:]


def oracle_bins(groups, reads, length, edges):
    samples, depth, cover = oracle_per_base(groups, reads, length)
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



ORF_COLUMNS = ["genome_id", "orf_id", "label", "type", "start", "stop", "strand",
               "group", "n_samples", "depth_q1", "depth_median", "depth_q3",
               "depth_mean", "prevalence", "union_breadth", "contrast"]


def synthetic_orfs(spans):
    """ORFs o0, o1, ... over the given [start, stop) spans, as `genome_orfs`
    returns them."""
    def text(values):
        return np.array(values, dtype=object)

    return {"orf_id": text([f"o{i}" for i in range(len(spans))]),
            "label": text([f"L{i}" for i in range(len(spans))]),
            "type": text(["CDS"] * len(spans)),
            "start": np.array([a for a, _ in spans], np.int64),
            "stop": np.array([b for _, b in spans], np.int64),
            "strand": text(["+"] * len(spans))}


class OrfSegmentsTests(unittest.TestCase):
    def test_an_orf_within_the_genome_is_one_segment(self):
        index, starts, stops = orf_segments(
            np.array([1, 2901]), np.array([301, 3001]), 3000
        )
        self.assertEqual(
            (index.tolist(), starts.tolist(), stops.tolist()),
            ([0, 1], [1, 2901], [301, 3001]),
        )

    def test_an_orf_across_the_origin_splits_there(self):
        """GFF3 writes an ORF across a circular genome's origin with an end
        past the length: gc_7, 2951-3050 on the 3,000 bp GC."""
        index, starts, stops = orf_segments(
            np.array([101, 2951]), np.array([201, 3051]), 3000
        )
        self.assertEqual(
            sorted(zip(index.tolist(), starts.tolist(), stops.tolist(),
                       strict=True)),
            [(0, 101, 201), (1, 1, 51), (1, 2951, 3001)],
        )


class OrfStatisticsTests(SyntheticTestCase):
    """One row per ORF and group; each sample's mean depth over the ORF first,
    then the group's quantiles of those."""

    def orf_table(self, spans, length=8, **kwargs):
        _, table, _ = genome_statistics(
            self.con, "G", length,
            [display_bin_edges(1, length + 1, length)], synthetic_orfs(spans),
            **kwargs,
        )
        return table

    def test_a_missing_contrast_is_reported_only_when_asked_for(self):
        """The table always has a contrast column, but a run that did not ask
        for --orf-contrast should not be told why it is empty."""
        self.build({"S0": "a", "S1": "b"}, [("S1", 1, 3, "2M")])
        with self.assertNoLogs("micov"):
            table = self.orf_table([(1, 3), (3, 5)])
        self.assertTrue(np.isnan(table["contrast"]).all())
        with self.assertLogs("micov", level="WARNING") as logged:
            table = self.orf_table([(1, 3), (3, 5)], warn_contrast=True)
        self.assertTrue(np.isnan(table["contrast"]).all())
        self.assertIn("No ORF contrast for G", "\n".join(logged.output))

    def test_each_orfs_contrast_once(self):
        """The ORF track is coloured by it, one colour an ORF; the table
        repeats it for every group. b has twice a's depth on o0 and the same
        on o1 and o2, so both norms are 1."""
        self.build({"S0": "a", "S1": "b"},
                   [("S0", 1, 7, "6M"), ("S1", 1, 7, "6M"), ("S1", 1, 3, "2M")])
        _, table, contrast = genome_statistics(
            self.con, "G", 8, [display_bin_edges(1, 9, 8)],
            synthetic_orfs([(1, 3), (3, 5), (5, 7)]),
        )
        np.testing.assert_allclose(contrast, [math.log2(2.05 / 1.05), 0, 0])
        np.testing.assert_array_equal(table["contrast"], np.repeat(contrast, 2))

    def row(self, table, orf_id, group):
        (i,) = np.flatnonzero(
            (table["orf_id"] == orf_id) & (table["group"] == group)
        )
        return {key: table[key][i].item() if hasattr(table[key][i], "item")
                else table[key][i] for key in ORF_COLUMNS}

    def test_columns_and_rows(self):
        """ORF by ORF, the groups sorted within each."""
        self.build({"S0": "b", "S1": "a", "S2": "b", "S3": "c"},
                   [("S0", 1, 3, "2M")])
        table = self.orf_table([(1, 3), (5, 7)])
        self.assertEqual(list(table), ORF_COLUMNS)
        self.assertEqual(table["orf_id"].tolist(), ["o0"] * 3 + ["o1"] * 3)
        self.assertEqual(table["group"].tolist(), ["a", "b", "c"] * 2)
        self.assertEqual(table["n_samples"].tolist(), [1, 2, 1] * 2)
        self.assertEqual(table["genome_id"].tolist(), ["G"] * 6)
        self.assertEqual(table["label"].tolist(), ["L0"] * 3 + ["L1"] * 3)
        self.assertEqual(table["start"].tolist(), [1, 1, 1, 5, 5, 5])
        self.assertEqual(table["stop"].tolist(), [3, 3, 3, 7, 7, 7])

    def test_the_median_is_of_each_samples_mean_depth(self):
        """Over the two bases, S0 has depth 2 then 0, S1 0 then 2, and S2 1
        then 0: means 1, 1 and 0.5, whose median is 1. Per base first would
        give medians 1 and 0, so 0.5 -- a typical sample this ORF does not
        have."""
        reads = ([("S0", 1, 2, "1M")] * 2 + [("S1", 2, 3, "1M")] * 2
                 + [("S2", 1, 2, "1M")])
        self.build({"S0": "a", "S1": "a", "S2": "a"}, reads)
        row = self.row(self.orf_table([(1, 3)]), "o0", "a")
        self.assertEqual(
            [row[k] for k in ("depth_q1", "depth_median", "depth_q3")],
            [0.75, 1.0, 1.0],
        )
        self.assertEqual(row["depth_mean"], 5 / 6)

    def test_an_orf_without_coverage_is_zero_not_nan(self):
        self.build({"S0": "a"}, [("S0", 1, 3, "2M")])
        row = self.row(self.orf_table([(5, 8)]), "o0", "a")
        self.assertEqual(
            [row[k] for k in ORF_COLUMNS[9:15]], [0.0] * 6
        )

    def test_prevalence_and_union_breadth(self):
        """S0 covers 1-4 with a skip (depth only at 1 and 4), S1 covers 2-4:
        7 of the 8 (sample, base) pairs, and all 4 bases."""
        self.build({"S0": "a", "S1": "a"},
                   [("S0", 1, 5, "1M2N1M"), ("S1", 2, 5, "3M")])
        row = self.row(self.orf_table([(1, 5)]), "o0", "a")
        self.assertEqual(row["prevalence"], 7 / 8)
        self.assertEqual(row["union_breadth"], 1.0)
        self.assertEqual(row["depth_mean"], 5 / 8)

    def test_an_orf_across_the_origin_counts_both_ends(self):
        """[7, 11) on an 8 bp genome is bases 7, 8, 1 and 2."""
        self.build({"S0": "a"}, [("S0", 1, 3, "2M"), ("S0", 7, 9, "2M")])
        row = self.row(self.orf_table([(7, 11)]), "o0", "a")
        self.assertEqual(
            [row[k] for k in ("depth_median", "prevalence", "union_breadth")],
            [1.0, 1.0, 1.0],
        )

    def test_a_genome_without_orfs_has_an_empty_table(self):
        """--orfs may annotate some plotted genomes and not others."""
        self.build({"S0": "a", "S1": "b"}, [("S0", 1, 3, "2M")])
        with warnings.catch_warnings(), self.assertNoLogs("micov"):
            warnings.simplefilter("error")
            table = self.orf_table([])
        self.assertEqual(list(table), ORF_COLUMNS)
        self.assertEqual({len(column) for column in table.values()}, {0})

    def test_two_groups_have_a_contrast(self):
        """b has twice a's depth on every ORF: no contrast, once each group
        is scaled by its typical ORF."""
        reads = []
        for sample, scale in (("S0", 1), ("S1", 2)):
            for base in (1, 2, 3):
                reads += [(sample, base, base + 1, "1M")] * (base * scale)
        self.build({"S0": "a", "S1": "b"}, reads)
        table = self.orf_table([(1, 2), (2, 3), (3, 4)])
        self.assertEqual(table["contrast"].tolist(), [0.0] * 6)


class OrfContrastTests(unittest.TestCase):
    """log2((B / norm B + 0.05) / (A / norm A + 0.05)) per ORF, from each
    group's per-ORF median; A and B are the groups in sorted order, and a
    group's norm is the median of its per-ORF medians on this genome."""

    def test_a_constant_ratio_is_no_contrast(self):
        """Deeper sequencing of one group is not a difference between them."""
        median = np.array([[1.0, 2.0, 3.0, 0.5], [2.0, 4.0, 6.0, 1.0]])
        self.assertEqual(
            orf_contrast(median, ["a", "b"], "G").tolist(), [0.0] * 4
        )

    def test_positive_where_the_second_group_is_higher(self):
        """Norms 2 and 4: o0 is 0.5 of typical in a and 1 in b."""
        median = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
        contrast = orf_contrast(median, ["a", "b"], "G")
        for got, expected in zip(
            contrast,
            [math.log2(1.05 / 0.55), 0.0, math.log2(1.05 / 1.55)],
            strict=True,
        ):
            self.assertAlmostEqual(got, expected, places=12)
        self.assertGreater(contrast[0], 0)
        self.assertLess(contrast[2], 0)

    def test_a_zero_norm_is_no_contrast_and_is_named(self):
        """Most of the genome's ORFs have median 0 in case, so there is no
        typical depth to scale by."""
        median = np.array([[0.0, 0.0, 3.0], [1.0, 2.0, 3.0]])
        with self.assertLogs("micov", level="WARNING") as logged:
            contrast = orf_contrast(median, ["case", "control"], "GQ1",
                                    warn=True)
        self.assertTrue(np.isnan(contrast).all())
        message = "\n".join(logged.output)
        self.assertIn("GQ1", message)
        self.assertIn("case", message)
        self.assertNotIn("control", message)

    def test_only_two_groups_have_a_contrast(self):
        for groups in (["a"], ["a", "b", "c"]):
            with self.subTest(groups=groups), self.assertNoLogs("micov"):
                median = np.ones((len(groups), 3))
                self.assertTrue(
                    np.isnan(orf_contrast(median, groups, "G", warn=True)).all()
                )


class FixtureOrfTests(DepthTestCase):
    """The dp ORFs, worked by hand from `dp.sam` and `dp.gff`. Case is S1-S3
    (n 3), control S4 and S5 (n 2)."""

    def orf_table(self, genome, length, warn_contrast=False):
        with self.assertLogs("micov", level="WARNING"):
            self.resolve(orfs=DATA / "dp_orfs.parquet")
        stage_breadth(self.con, "breadth_layer")
        stage_depth_reads(self.con, "depth_layer")
        orfs = genome_orfs(self.con, genome)
        if not warn_contrast:
            with self.assertNoLogs("micov"):
                _, table, _ = genome_statistics(
                    self.con, genome, length,
                    [display_bin_edges(1, length + 1, overview_bin_bp(length))],
                    orfs,
                )
            return table, ""
        with self.assertLogs("micov", level="WARNING") as logged:
            _, table, _ = genome_statistics(
                self.con, genome, length,
                [display_bin_edges(1, length + 1, overview_bin_bp(length))],
                orfs, warn_contrast=True,
            )
        return table, "\n".join(logged.output)

    def assertRows(self, table, expected):
        for (orf_id, group), values in expected.items():
            (row,) = np.flatnonzero(
                (table["orf_id"] == orf_id) & (table["group"] == group)
            )
            for key, value in zip(ORF_COLUMNS[9:15], values, strict=True):
                with self.subTest(orf=orf_id, group=group, column=key):
                    self.assertAlmostEqual(table[key][row], value, places=12)

    def test_genome_orfs(self):
        """Only the ORF types, by position; the gene and region lines are not."""
        with self.assertLogs("micov", level="WARNING"):
            self.resolve(orfs=DATA / "dp_orfs.parquet")
        orfs = genome_orfs(self.con, "GC")
        self.assertEqual(
            list(zip(*(orfs[k].tolist() for k in
                       ("orf_id", "label", "type", "start", "stop", "strand")),
                     strict=True)),
            [("gc_1", "dnaA", "CDS", 1, 301, "+"),
             ("gc_2", "GC_0002", "CDS", 901, 1101, "-"),
             ("gc_3", "rrsA", "rRNA", 1201, 1401, "+"),
             ("gc_4", "gc_4", "ncRNA", 2001, 2101, "."),
             ("gc_5", "nTest", "CDS", 2601, 2801, "+"),
             ("gc_6", "gc_6", "CDS", 2901, 3001, "-"),
             ("gc_7", "gc_7", "CDS", 2951, 3051, "+")],
        )

    def test_genome_orfs_carry_their_attributes(self):
        """`--highlight` and `--orf-color-by` match GFF attributes."""
        with self.assertLogs("micov", level="WARNING"):
            self.resolve(orfs=DATA / "dp_orfs.parquet")
        orfs = genome_orfs(self.con, "GC", attributes=True)
        self.assertEqual(orfs["attributes"][1],
                         {"ID": "gc_2", "locus_tag": "GC_0002",
                          "product": "5' nucleotidase"})
        self.assertEqual([a.get("gene") for a in orfs["attributes"]],
                         ["dnaA", None, "rrsA", None, "nTest", None, None])

    def test_genome_orfs_leave_attributes_out_unless_asked(self):
        """A dict per ORF costs 77 ms a genome at 10,000 ORFs, and only
        `--highlight` and `--orf-color-by` read them."""
        with self.assertLogs("micov", level="WARNING"):
            self.resolve(orfs=DATA / "dp_orfs.parquet")
        self.assertNotIn("attributes", genome_orfs(self.con, "GC"))

    def test_gc(self):
        """Columns: depth Q1, median, Q3, mean, prevalence, union breadth.

        - gc_2 (200 bp): S1 (a2) and S2 (b1) each cover 100 bases, means 0.5,
          0.5 and 0; their breadth joins up to 150 bases.
        - gc_3 (200 bp): S1, S2 and S3 each 100 bases, means 0.5; S4's two
          reads stack to depth 2 over 100 bases, mean 1, beside S5's 0.
        - gc_4 (100 bp): S5 covers it all.
        - gc_5 (200 bp): S3's read has 60 bases of depth and spans 160.
        - gc_7 (2951-3050, across the origin): S1's a3 covers 2951-3000 and
          a1 1-50, so all 100 bases.
        """
        table, _ = self.orf_table("GC", 3000)
        self.assertRows(table, {
            ("gc_2", "case"): (0.25, 0.5, 0.5, 1 / 3, 1 / 3, 0.75),
            ("gc_2", "control"): (0, 0, 0, 0, 0, 0),
            ("gc_3", "case"): (0.5, 0.5, 0.5, 0.5, 0.5, 0.75),
            ("gc_3", "control"): (0.25, 0.5, 0.75, 0.5, 0.25, 0.5),
            ("gc_4", "control"): (0.25, 0.5, 0.75, 0.5, 0.5, 1.0),
            ("gc_5", "case"): (0, 0, 0.15, 0.1, 4 / 15, 0.8),
            ("gc_7", "case"): (0, 0, 0.5, 1 / 3, 1 / 3, 1.0),
        })
        self.assertEqual(len(table["orf_id"]), 14)

    def test_gc_has_no_contrast(self):
        """Most GC ORFs have median 0 in both groups: nothing to scale by."""
        table, warned = self.orf_table("GC", 3000, warn_contrast=True)
        self.assertTrue(np.isnan(table["contrast"]).all())
        for name in ("GC", "case", "control"):
            self.assertIn(name, warned)

    def test_gl_breadth_without_depth(self):
        """gl_5 has breadth (S1's a6, 50 bases) but no depth."""
        table, _ = self.orf_table("GL", 2000)
        self.assertRows(table, {
            ("gl_5", "case"): (0, 0, 0, 0, 1 / 6, 0.5),
            ("gl_1", "case"): (0, 0, 2 / 3, 4 / 9, 1 / 3, 1.0),
            ("gl_2", "control"): (1, 1, 1, 1, 1, 1),
        })


def random_orfs(rng, length, count):
    """Spans up to 15 bp, some running past the end (so across the origin)."""
    starts = rng.integers(1, length + 1, count)
    return [(int(a), int(a + b)) for a, b in
            zip(starts, rng.integers(1, 16, count), strict=True)]


def oracle_orfs(groups, reads, length, spans):
    samples, depth, cover = oracle_per_base(groups, reads, length)
    names = sorted(set(groups.values()))
    table = {key: [] for key in ORF_COLUMNS[7:15]}
    medians = []
    for start, stop in spans:
        bases = [(b - 1) % length for b in range(start, stop)]
        for group in names:
            rows = [i for i, s in enumerate(samples) if groups[s] == group]
            d, c = depth[np.ix_(rows, bases)], cover[np.ix_(rows, bases)]
            q1, median, q3 = np.quantile(d.mean(axis=1), (0.25, 0.5, 0.75))
            values = {"group": group, "n_samples": len(rows), "depth_q1": q1,
                      "depth_median": median, "depth_q3": q3,
                      "depth_mean": d.mean(), "prevalence": c.mean(),
                      "union_breadth": c.any(axis=0).mean()}
            for key, value in values.items():
                table[key].append(value)
            medians.append(median)
    medians = np.array(medians).reshape(len(spans), len(names)).T
    a, b = medians / np.median(medians, axis=1, keepdims=True)
    table["contrast"] = np.repeat(np.log2((b + 0.05) / (a + 0.05)), len(names))
    return table


class OrfWindowInvarianceTests(SyntheticTestCase):
    """ORF sums carried across windows, including over a window edge and an
    ORF across the origin: bit-identical for any window, and equal to the
    Python oracle."""

    LENGTH = 61
    GROUPS: ClassVar[dict] = {"S0": "a", "S1": "b", "S2": "a", "S3": "b",
                              "S4": "a"}

    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(5)
        self.reads = random_reads(rng, sorted(self.GROUPS), self.LENGTH, 200)
        self.spans = random_orfs(rng, self.LENGTH, 12)
        self.assertTrue(any(stop > self.LENGTH + 1 for _, stop in self.spans))
        self.build(self.GROUPS, self.reads, self.LENGTH)

    def orf_table(self, width):
        with mock.patch.object(_depth, "WINDOW_CELLS", width * len(self.GROUPS)):
            _, table, _ = genome_statistics(
                self.con, "G", self.LENGTH,
                [display_bin_edges(1, self.LENGTH + 1, 7)],
                synthetic_orfs(self.spans),
            )
        return table

    def test_any_window_size_gives_identical_orf_statistics(self):
        whole = self.orf_table(self.LENGTH)
        for width in (1, 2, 3, 7):
            windowed = self.orf_table(width)
            for key in ORF_COLUMNS:
                with self.subTest(width=width, column=key):
                    self.assertEqual(whole[key].tolist(), windowed[key].tolist())

    def test_orf_statistics_match_a_per_base_oracle(self):
        table = self.orf_table(3)
        expected = oracle_orfs(self.GROUPS, self.reads, self.LENGTH, self.spans)
        for key, values in expected.items():
            with self.subTest(column=key):
                if key in ("group", "n_samples"):
                    self.assertEqual(table[key].tolist(), values)
                else:
                    np.testing.assert_allclose(
                        table[key], values, rtol=1e-12, atol=1e-15
                    )


if __name__ == "__main__":
    unittest.main()
