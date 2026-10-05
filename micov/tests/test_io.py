"""Tests for micov's input parsing.

`load_genome_lengths` is the one that matters here. It is on the live
`compress` and `cov-to-parquet` paths -- it supplies every coverage
denominator, and doubles as htslib's reference map -- and until M5 it had no
tests at all. Its four validation branches were covered only through
`parse_genome_lengths`, the polars twin it was written to replace, so deleting
that twin without moving these across would have silently stripped the
coverage from the surviving function.

The error messages are asserted verbatim, including the stray trailing quote in
"is not integer'", because they are what a user sees when a length file is
wrong and because they were the evidence that the two implementations agreed.
"""

import tempfile
import unittest
from pathlib import Path

import duckdb

from micov._constants import (
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_START,
    COLUMN_STOP,
)
from micov._io import (
    FEATURE_ID_COLUMNS,
    SAMPLE_ID_COLUMNS,
    load_alignment_layer,
    load_bed_cov,
    load_depth_features,
    load_genome_lengths,
    load_orfs,
    load_sample_groups,
    read_tsv_with_header,
)
from micov._miint import connection


class GenomeLengthsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.name = self.temp_dir.name + "/foo.tsv"
        self.con = connection()
        self.addCleanup(self.con.close)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write(self, data):
        with open(self.name, "w") as fp:
            fp.write(data)
        return self.name

    def loaded(self):
        return self.con.sql(
            f"SELECT {COLUMN_GENOME_ID}, {COLUMN_LENGTH} "
            "FROM genome_lengths ORDER BY 1"
        ).fetchall()

    def test_reads_a_headered_file(self):
        """Columns are taken by position, not by name.

        A length file is whatever the user had lying around -- the header here
        is `foo/bar/baz`, and the third column is ignored entirely.
        """
        self.write("foo\tbar\tbaz\na\t10\txyz\nb\t20\txyz\nc\t30\txyz\n")
        load_genome_lengths(self.con, self.name)
        self.assertEqual(self.loaded(), [("a", 10), ("b", 20), ("c", 30)])

    def test_reads_a_headerless_file(self):
        """Header detection is a guess micov has to make.

        There is no way to require a header -- released micov accepted both --
        so the first line is sniffed, and a headerless file must not lose its
        first genome to being mistaken for column names.
        """
        self.write("a\t10\txyz\nb\t20\txyz\nc\t30\txyz\n")
        load_genome_lengths(self.con, self.name)
        self.assertEqual(self.loaded(), [("a", 10), ("b", 20), ("c", 30)])

    def test_a_genome_whose_id_is_part_of_genome_id_is_not_a_header(self):
        """`id` is a substring of `genome_id`, and that is all it has in common.

        Header detection used to test ``first_field in "genome_id"`` -- a
        substring test on a plain string -- so a headerless file whose first
        genome was `id`, `genome` or `e` read as having a header, and that
        genome's row was silently dropped from every denominator.
        """
        for genome in ("id", "genome", "e"):
            with self.subTest(first_genome=genome):
                self.con.sql("DROP TABLE IF EXISTS genome_lengths")
                self.write(f"{genome}\t10\nb\t20\n")
                load_genome_lengths(self.con, self.name)
                self.assertEqual(self.loaded(), sorted([(genome, 10), ("b", 20)]))

    def test_the_canonical_header_is_a_header(self):
        self.write("genome_id\tlength\na\t10\nb\t20\n")
        load_genome_lengths(self.con, self.name)
        self.assertEqual(self.loaded(), [("a", 10), ("b", 20)])

    def test_a_commented_header_is_a_header(self):
        self.write("#genome_id\tlength\na\t10\nb\t20\n")
        load_genome_lengths(self.con, self.name)
        self.assertEqual(self.loaded(), [("a", 10), ("b", 20)])

    def test_renames_to_the_canonical_columns(self):
        """Everything downstream joins on `genome_id` and divides by `length`."""
        self.write("foo\tbar\tbaz\na\t10\txyz\n")
        load_genome_lengths(self.con, self.name)
        described = self.con.sql("DESCRIBE genome_lengths").fetchall()
        self.assertEqual(
            [(row[0], row[1]) for row in described],
            [(COLUMN_GENOME_ID, "VARCHAR"), (COLUMN_LENGTH, "BIGINT")],
        )

    def test_non_integer_lengths_are_rejected(self):
        """A non-numeric length means the file's columns are not what micov thinks.

        Left alone it would produce a nonsense denominator rather than an
        error, and the message names the column as the *file* labelled it.
        """
        self.write("foo\tbar\tbaz\na\t10\txyz\nb\tXXX\txyz\nc\t30\txyz\n")
        with self.assertRaisesRegex(ValueError, "'bar' is not integer"):
            load_genome_lengths(self.con, self.name)

    def test_duplicate_genome_ids_are_rejected(self):
        """A repeated genome silently multiplies rows at every join downstream."""
        self.write("foo\tbar\tbaz\na\t10\txyz\nb\t20\txyz\nb\t30\txyz\n")
        with self.assertRaisesRegex(ValueError, "'foo' is not unique"):
            load_genome_lengths(self.con, self.name)

    def test_non_positive_lengths_are_rejected(self):
        """Breadth divides by length; zero or negative is not a denominator."""
        self.write("foo\tbar\tbaz\na\t10\txyz\nb\t-5\txyz\nc\t30\txyz\n")
        with self.assertRaisesRegex(ValueError, "Lengths of zero or less"):
            load_genome_lengths(self.con, self.name)


class BedCovTests(unittest.TestCase):
    """Reading `.cov` / BED3, which is `position-plot`'s only input.

    A `.cov` file may or may not carry a header -- `micov compress` wrote one,
    hand-made and third-party files often do not, and micov has always
    accepted both. Getting that wrong does not fail loudly: a header row read
    as data becomes a genome literally named "genome_id", and a data row read
    as a header silently loses the first interval.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.name = self.temp_dir.name + "/s.cov"
        self.con = connection()
        self.addCleanup(self.con.close)

    def load(self, data):
        with open(self.name, "w") as fp:
            fp.write(data)
        load_bed_cov(self.con, self.name)
        return self.con.sql(
            f"SELECT {COLUMN_GENOME_ID}, {COLUMN_START}, {COLUMN_STOP} "
            "FROM bed_positions ORDER BY 1, 2"
        ).fetchall()

    ROWS = [("G1", 10, 20), ("G2", 5, 9)]  # noqa: RUF012

    def test_headered(self):
        self.assertEqual(
            self.load("genome_id\tstart\tstop\nG1\t10\t20\nG2\t5\t9\n"),
            self.ROWS,
        )

    def test_headerless(self):
        self.assertEqual(self.load("G1\t10\t20\nG2\t5\t9\n"), self.ROWS)

    def test_commented_header(self):
        """`#`-prefixed headers are what a BED file conventionally carries."""
        self.assertEqual(
            self.load("#genome_id\tstart\tstop\nG1\t10\t20\nG2\t5\t9\n"),
            self.ROWS,
        )

    def test_interval_bounds_are_unsigned(self):
        """The rest of micov works in UINTEGER; a BIGINT here leaks into joins."""
        self.load("genome_id\tstart\tstop\nG1\t10\t20\n")
        described = self.con.sql("DESCRIBE bed_positions").fetchall()
        self.assertEqual(
            [(row[0], row[1]) for row in described],
            [
                (COLUMN_GENOME_ID, "VARCHAR"),
                (COLUMN_START, "UINTEGER"),
                (COLUMN_STOP, "UINTEGER"),
            ],
        )



class ReadTsvWithHeaderTests(unittest.TestCase):
    """The header rule, as a function every reader shares.

    It moved out of `View` so that inputs which are not the coverage Parquet
    pair -- `depth-plot`'s features and metadata -- enforce the same rule. A
    headerless file has its first row read as column names, and that genome or
    sample then vanishes from every output; insisting on the first column's
    name is what makes the missing header detectable.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.con = connection()
        self.addCleanup(self.con.close)

    def rows(self, text, rename, first_column, all_varchar=False):
        path = f"{self.temp_dir.name}/in.tsv"
        with open(path, "w") as fp:
            fp.write(text)
        rel = self.con.sql(
            read_tsv_with_header(self.con, path, rename, first_column, all_varchar)
        )
        return rel.columns, rel.fetchall()

    def test_leading_columns_are_renamed_and_the_rest_pass_through(self):
        columns, rows = self.rows(
            "sample_name\tdog\tage\nS1\tbeagle\t3\n",
            [COLUMN_GENOME_ID],
            ("sample_name",),
        )
        self.assertEqual(columns, [COLUMN_GENOME_ID, "dog", "age"])
        self.assertEqual(rows, [("S1", "beagle", 3)])

    def test_sample_name_is_accepted_for_metadata(self):
        columns, _ = self.rows("sample_name\tdog\nS1\tYes\n", ["sample_id"],
                               SAMPLE_ID_COLUMNS)
        self.assertEqual(columns, ["sample_id", "dog"])

    def test_a_headerless_file_is_refused_by_name(self):
        """The first data row would otherwise become the header, silently."""
        with self.assertRaisesRegex(ValueError, "in.tsv.*'genome_id'.*'G1'"):
            self.rows("G1\t100\nG2\t200\n", [COLUMN_GENOME_ID],
                      FEATURE_ID_COLUMNS)

    def test_all_varchar_keeps_category_values_as_written(self):
        """`example/`'s `dog` column is Yes/No, which the sniffer makes boolean.

        A group would then be named `True` rather than `Yes`, in legends,
        file contents and `.ks.csv` labels.
        """
        _, rows = self.rows("sample_id\tdog\nS1\tYes\nS2\tNo\n", ["sample_id"],
                            SAMPLE_ID_COLUMNS, all_varchar=True)
        self.assertEqual(rows, [("S1", "Yes"), ("S2", "No")])


DATA = Path(__file__).parent / "test_data"


class DepthInputTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.con = connection()
        self.addCleanup(self.con.close)

    def write(self, name, text):
        path = f"{self.temp_dir.name}/{name}"
        with open(path, "w") as fp:
            fp.write(text)
        return path


class AlignmentLayerTests(DepthInputTestCase):
    """`depth-plot` reads `read_alignments` output saved as Parquet.

    `read_alignments` has no sample column, so a user saving its output
    directly has a file that cannot be grouped. That has to fail with a message
    saying how to add one, not with a DuckDB binder error.
    """

    def without(self, column):
        path = f"{self.temp_dir.name}/layer.parquet"
        duckdb.sql(f"""COPY (SELECT * EXCLUDE ({column})
                              FROM '{DATA}/dp_depth.parquet')
                       TO '{path}' (FORMAT PARQUET)""")
        return path

    def test_the_fixture_loads_as_a_view(self):
        load_alignment_layer(self.con, str(DATA / "dp_depth.parquet"), "depth_layer")
        self.assertEqual(
            self.con.sql("SELECT count(*) FROM depth_layer").fetchone(), (23,)
        )

    def test_a_missing_sample_id_says_how_to_add_one(self):
        with self.assertRaisesRegex(ValueError, "sample_id.*read_alignments"):
            load_alignment_layer(self.con, self.without("sample_id"), "depth_layer")

    def test_a_missing_cigar_is_named(self):
        with self.assertRaisesRegex(ValueError, "cigar"):
            load_alignment_layer(self.con, self.without("cigar"), "depth_layer")


class DepthFeaturesTests(DepthInputTestCase):
    """`--features-to-keep` for `depth-plot` carries the genome lengths.

    The alignment Parquet has none, and the depth arrays are as long as the
    length says, so a wrong length is wrong data rather than a cosmetic
    problem. Region rows become detail panels; an empty region means none.
    """

    HEADER = "genome_id\tlength\tis_circular\tstart\tstop\n"

    def load(self, path):
        load_depth_features(self.con, path)
        return self.con.sql(
            "SELECT * FROM depth_features ORDER BY genome_id, start NULLS FIRST"
        ).fetchall()

    def test_regions_and_genomes_without_regions(self):
        self.assertEqual(
            self.load(str(DATA / "dp_regions.tsv")),
            [
                ("GB", 1000, False, None, None),
                ("GC", 3000, True, 1001, 1501),
                ("GC", 3000, True, 2501, 2901),
                ("GL", 2000, False, None, None),
                ("GX", 1000, False, None, None),
            ],
        )

    def test_without_region_columns_there_are_no_regions(self):
        rows = self.load(str(DATA / "dp_features.tsv"))
        self.assertEqual([row[3:] for row in rows], [(None, None)] * 4)

    def test_is_circular_is_optional_and_defaults_to_linear(self):
        path = self.write("f.tsv", "genome_id\tlength\nG1\t100\n")
        self.assertEqual(self.load(path), [("G1", 100, False, None, None)])

    def test_an_empty_is_circular_means_linear(self):
        path = self.write("f.tsv", "genome_id\tlength\tis_circular\n"
                                   "G1\t100\ttrue\nG2\t100\t\n")
        self.assertEqual([row[2] for row in self.load(path)], [True, False])

    def test_a_missing_length_is_refused(self):
        path = self.write("f.tsv", "genome_id\tis_circular\nG1\ttrue\n")
        with self.assertRaisesRegex(ValueError, "length"):
            self.load(path)

    def test_a_non_integer_length_is_refused(self):
        path = self.write("f.tsv", "genome_id\tlength\nG1\t3kb\n")
        with self.assertRaisesRegex(ValueError, "length"):
            self.load(path)

    def test_a_length_of_zero_is_refused(self):
        path = self.write("f.tsv", "genome_id\tlength\nG1\t0\n")
        with self.assertRaisesRegex(ValueError, "G1"):
            self.load(path)

    def test_is_circular_must_be_true_or_false(self):
        path = self.write("f.tsv", "genome_id\tlength\tis_circular\nG1\t100\tmaybe\n")
        with self.assertRaisesRegex(ValueError, "is_circular"):
            self.load(path)

    def test_start_without_stop_is_refused(self):
        path = self.write("f.tsv", "genome_id\tlength\tstart\nG1\t100\t5\n")
        with self.assertRaisesRegex(ValueError, "stop"):
            self.load(path)

    def test_half_a_region_is_refused(self):
        path = self.write("f.tsv", self.HEADER + "G1\t100\tfalse\t5\t\n")
        with self.assertRaisesRegex(ValueError, "G1"):
            self.load(path)

    def test_an_empty_region_is_refused(self):
        """Regions are half-open, so stop must exceed start."""
        path = self.write("f.tsv", self.HEADER + "G1\t100\tfalse\t50\t50\n")
        with self.assertRaisesRegex(ValueError, "G1.*50"):
            self.load(path)

    def test_a_region_past_the_end_is_refused(self):
        """[1, length + 1) is the whole genome; anything further is not in it."""
        path = self.write("f.tsv", self.HEADER + "G1\t100\tfalse\t90\t102\n")
        with self.assertRaisesRegex(ValueError, "G1"):
            self.load(path)

    def test_the_whole_genome_is_a_valid_region(self):
        path = self.write("f.tsv", self.HEADER + "G1\t100\tfalse\t1\t101\n")
        self.assertEqual(self.load(path), [("G1", 100, False, 1, 101)])

    def test_a_genome_with_two_lengths_is_refused(self):
        path = self.write("f.tsv", self.HEADER + "G1\t100\tfalse\t1\t10\n"
                                                 "G1\t101\tfalse\t20\t30\n")
        with self.assertRaisesRegex(ValueError, "G1"):
            self.load(path)

    def test_a_repeated_region_is_refused(self):
        """Each region is a detail panel and a file name; two would collide."""
        path = self.write("f.tsv", self.HEADER + "G1\t100\tfalse\t1\t10\n"
                                                 "G1\t100\tfalse\t1\t10\n")
        with self.assertRaisesRegex(ValueError, "G1"):
            self.load(path)

    def test_a_headerless_file_is_refused(self):
        path = self.write("f.tsv", "G1\t100\n")
        with self.assertRaisesRegex(ValueError, "genome_id"):
            self.load(path)


class SampleGroupsTests(DepthInputTestCase):
    def load(self, path, column):
        load_sample_groups(self.con, path, column)
        return self.con.sql("SELECT * FROM sample_groups ORDER BY 1").fetchall()

    def test_the_fixture(self):
        self.assertEqual(
            self.load(str(DATA / "dp_metadata.tsv"), "site"),
            [("S1", "o'hare"), ("S2", "o'hare"), ("S3", "o'hare"),
             ("S4", "midway"), ("S5", "midway"), ("S6", "midway"),
             ("S7", "midway"), ("S8", "midway")],
        )

    def test_values_are_kept_as_written(self):
        """Yes/No would otherwise become a group named True."""
        path = self.write("m.tsv", "sample_name\tdog\nS1\tYes\nS2\tNo\n")
        self.assertEqual(self.load(path, "dog"), [("S1", "Yes"), ("S2", "No")])

    def test_a_missing_column_names_the_ones_there_are(self):
        with self.assertRaisesRegex(ValueError, "colour.*group"):
            self.load(str(DATA / "dp_metadata.tsv"), "colour")

    def test_a_column_name_with_a_space(self):
        """The column name is spliced into SQL as an identifier."""
        path = self.write("m.tsv", "sample_id\tdog breed\nS1\tbeagle\n")
        self.assertEqual(self.load(path, "dog breed"), [("S1", "beagle")])

    def test_a_sample_without_a_value_is_left_out_and_reported(self):
        """It belongs to no group, so it cannot be counted in any."""
        path = self.write("m.tsv", "sample_id\tdog\nS1\tYes\nS2\t\n")
        with self.assertLogs("micov", level="WARNING") as logged:
            rows = self.load(path, "dog")
        self.assertEqual(rows, [("S1", "Yes")])
        self.assertIn("S2", "\n".join(logged.output))


class OrfTests(DepthInputTestCase):
    """ORFs are `read_gff` output saved as Parquet."""

    def load(self, path=None):
        load_orfs(self.con, str(path or DATA / "dp_orfs.parquet"))
        return self.con.sql(
            "SELECT genome_id, orf_id, label, type, start, stop, strand "
            "FROM orfs ORDER BY genome_id, start, orf_id"
        ).fetchall()

    def gff(self, *lines):
        gff = self.write("in.gff", "##gff-version 3\n" + "".join(lines))
        path = f"{self.temp_dir.name}/orfs.parquet"
        self.con.sql(f"COPY (FROM read_gff('{gff}')) TO '{path}' (FORMAT PARQUET)")
        return path

    def test_the_fixture(self):
        """`gene` and `region` rows are dropped: `gene` duplicates its CDS."""
        rows = self.load()
        self.assertEqual(
            [row[1] for row in rows],
            ["gc_1", "gc_2", "gc_3", "gc_4", "gc_5", "gc_6", "gc_7",
             "gl_1", "gl_2", "gl_3", "gl_4", "gl_5", "gq_1", "gx_1"],
        )

    def test_coordinates_are_kept_half_open(self):
        """`read_gff` already wrote GFF end + 1; adding 1 again would be wrong."""
        self.assertIn(("GC", "gc_1", "dnaA", "CDS", 1, 301, "+"), self.load())

    def test_labels_fall_back_from_gene_to_locus_tag_to_id(self):
        labels = {row[1]: row[2] for row in self.load()}
        self.assertEqual(
            (labels["gc_1"], labels["gc_2"], labels["gc_4"]),
            ("dnaA", "GC_0002", "gc_4"),
        )

    def test_an_unstranded_orf_is_dot_not_null(self):
        """A NULL strand reaches numpy as a masked array, which compares oddly."""
        strands = {row[1]: row[6] for row in self.load()}
        self.assertEqual(strands["gc_4"], ".")

    def test_an_empty_gene_falls_through_to_the_next_label(self):
        path = self.gff("G1\tt\tCDS\t1\t30\t.\t+\t0\tID=x1;gene=;locus_tag=LT1\n")
        self.assertEqual(self.load(path)[0][2], "LT1")

    def test_an_orf_without_an_id_is_refused(self):
        """The ID is the row's key in the per-ORF table."""
        path = self.gff("G1\tt\tCDS\t1\t30\t.\t+\t0\tgene=abc\n")
        with self.assertRaisesRegex(ValueError, "ID"):
            self.load(path)

    def test_a_missing_column_is_named(self):
        path = f"{self.temp_dir.name}/orfs.parquet"
        self.con.sql(f"""COPY (SELECT * EXCLUDE (attributes)
                               FROM '{DATA}/dp_orfs.parquet')
                         TO '{path}' (FORMAT PARQUET)""")
        with self.assertRaisesRegex(ValueError, "attributes"):
            self.load(path)


if __name__ == "__main__":
    unittest.main()
