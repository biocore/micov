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

from micov._constants import (
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_START,
    COLUMN_STOP,
)
from micov._io import load_bed_cov, load_genome_lengths
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


if __name__ == "__main__":
    unittest.main()
