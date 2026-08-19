"""Tests for alignment ingest through miint.

micov's own CIGAR walker and numba interval merge are gone; `read_alignments`
and `compress_intervals` do that work now. These tests pin the conventions that
would otherwise be silently inherited from miint rather than chosen -- half-open
coordinates with no `+1`, touching intervals merging, and a reference map that
does not quietly drop reads.
"""

import gzip
import os
import shutil
import unittest
from pathlib import Path
from tempfile import mkdtemp

from micov._io import (
    ALIGNMENT_POSITIONS_TABLE,
    compress_alignments,
    load_genome_lengths,
    write_coverage_parquet,
)
from micov._miint import connection

HERE = Path(__file__).parent
DATA = HERE / "test_data"


def sam(*records):
    """Build headerless SAM text from (name, ref, pos, cigar) tuples."""
    return "".join(
        f"{name}\t0\t{ref}\t{pos}\t1\t{cigar}\t*\t0\t0\t*\t*\n"
        for name, ref, pos, cigar in records
    )


class AlignmentIngestTests(unittest.TestCase):
    def setUp(self):
        self.d = mkdtemp()
        self.con = connection()
        self.addCleanup(self.con.close)
        self.addCleanup(shutil.rmtree, self.d)

    def lengths(self, **genomes):
        """Write a lengths TSV and load it as the `genome_lengths` table."""
        path = f"{self.d}/lengths.tsv"
        with open(path, "w") as fp:
            fp.write("genome_id\tlength\n")
            for genome, length in genomes.items():
                fp.write(f"{genome}\t{length}\n")
        load_genome_lengths(self.con, path)
        return path

    def write_sam(self, text, name="in.sam"):
        path = f"{self.d}/{name}"
        with open(path, "w") as fp:
            fp.write(text)
        return path

    def positions(self, path, sample_id="S1", disable_compression=False):
        compress_alignments(
            self.con, path, sample_id, disable_compression=disable_compression
        )
        return sorted(
            self.con.sql(f"FROM {ALIGNMENT_POSITIONS_TABLE}").fetchall()
        )

    def test_stop_is_half_open_with_no_plus_one(self):
        """`stop = POS + reference span`, and breadth is the plain difference.

        A `+1` here would inflate every genome's breadth by one base per
        interval, which is small enough to look plausible and would move every
        published coverage value.
        """
        self.lengths(X=1000)
        path = self.write_sam(sam(("A", "X", 1, "50M")))

        self.assertEqual(self.positions(path), [("X", 1, 51, "S1")])

    def test_deletions_and_gaps_advance_the_reference(self):
        """`D` and `N` consume reference, so they count as covered.

        micov keeps them deliberately -- a deletion inside a read is still a
        region the read spans -- and this is the convention `_convert.py`
        encoded before miint took the job over.
        """
        self.lengths(X=1000)
        path = self.write_sam(
            sam(("A", "X", 1, "10M5D10M"), ("B", "X", 100, "10M5N10M"))
        )

        self.assertEqual(
            self.positions(path),
            [("X", 1, 26, "S1"), ("X", 100, 125, "S1")],
        )

    def test_touching_intervals_merge(self):
        """[400,500) and [500,505) become [400,505).

        micov's documented behaviour and the thing most likely to be lost
        silently in a merge-implementation swap: it changes breadth only where
        intervals abut, so most fixtures cannot see it.
        """
        self.lengths(X=1000)
        path = self.write_sam(sam(("A", "X", 400, "100M"), ("B", "X", 500, "5M")))

        self.assertEqual(self.positions(path), [("X", 400, 505, "S1")])

    def test_overlapping_and_nested_intervals_merge(self):
        self.lengths(X=1000)
        path = self.write_sam(
            sam(("A", "X", 1, "10M"), ("B", "X", 5, "10M"), ("C", "X", 6, "3M"))
        )

        self.assertEqual(self.positions(path), [("X", 1, 15, "S1")])

    def test_disjoint_intervals_are_kept_apart(self):
        """A one-base gap is a gap; only touching collapses."""
        self.lengths(X=1000)
        path = self.write_sam(sam(("A", "X", 1, "10M"), ("B", "X", 12, "5M")))

        self.assertEqual(
            self.positions(path), [("X", 1, 11, "S1"), ("X", 12, 17, "S1")]
        )

    def test_disable_compression_keeps_every_interval(self):
        self.lengths(X=1000)
        path = self.write_sam(sam(("A", "X", 400, "100M"), ("B", "X", 500, "5M")))

        self.assertEqual(
            self.positions(path, disable_compression=True),
            [("X", 400, 500, "S1"), ("X", 500, 505, "S1")],
        )

    def test_gzipped_input_is_read(self):
        self.lengths(X=1000)
        path = f"{self.d}/in.sam.gz"
        with gzip.open(path, "wt") as fp:
            fp.write(sam(("A", "X", 1, "50M")))

        self.assertEqual(self.positions(path), [("X", 1, 51, "S1")])


class ReferenceMapGuardTests(unittest.TestCase):
    """A reference missing from `--lengths` must not vanish quietly.

    htslib resolves RNAME through the supplied map and reports anything absent
    from it as `*`. The row count does not change, no error is raised, and the
    reads are simply attributed to nothing -- so a `--lengths` file covering
    half the references silently halves the coverage. Rule 10: this has to be
    loud.
    """

    def setUp(self):
        self.d = mkdtemp()
        self.con = connection()
        self.addCleanup(self.con.close)
        self.addCleanup(shutil.rmtree, self.d)

    def write(self, text, name):
        path = f"{self.d}/{name}"
        with open(path, "w") as fp:
            fp.write(text)
        return path

    def test_unattributed_alignments_are_reported(self):
        lengths = self.write("genome_id\tlength\nX\t1000\n", "lengths.tsv")
        load_genome_lengths(self.con, lengths)
        path = self.write(
            sam(("A", "X", 1, "50M"), ("B", "Y", 1, "50M"), ("C", "Z", 1, "50M")),
            "in.sam",
        )

        with self.assertLogs("micov", level="WARNING") as logged:
            compress_alignments(self.con, path, "S1")

        message = "\n".join(logged.output)
        self.assertIn("--lengths", message)
        # the count is the whole guarantee: htslib has already collapsed the
        # reference names to '*', so micov can say how many were lost but not
        # which references they were
        self.assertIn("2", message)
        # and they are excluded from the positions rather than written as a
        # genome literally named '*'
        self.assertEqual(
            self.con.sql(f"FROM {ALIGNMENT_POSITIONS_TABLE}").fetchall(),
            [("X", 1, 51, "S1")],
        )

    def test_complete_map_is_silent(self):
        """The negative control: the guard must not fire on good input."""
        lengths = self.write("genome_id\tlength\nX\t1000\nY\t1000\n", "lengths.tsv")
        load_genome_lengths(self.con, lengths)
        path = self.write(sam(("A", "X", 1, "50M"), ("B", "Y", 1, "50M")), "in.sam")

        with self.assertNoLogs("micov", level="WARNING"):
            compress_alignments(self.con, path, "S1")

    def test_unaligned_reads_do_not_trip_the_guard(self):
        """A genuinely unaligned read carries `*` legitimately.

        Only references present in the data but absent from the map are an
        error; a read that did not align anywhere is ordinary SAM.
        """
        lengths = self.write("genome_id\tlength\nX\t1000\n", "lengths.tsv")
        load_genome_lengths(self.con, lengths)
        path = self.write(
            "A\t0\tX\t1\t1\t50M\t*\t0\t0\t*\t*\nB\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*\n",
            "in.sam",
        )

        with self.assertLogs("micov", level="WARNING") as logged:
            compress_alignments(self.con, path, "S1")
        self.assertIn("1", "\n".join(logged.output))


class CoverageParquetTests(unittest.TestCase):
    """The two-file parquet pair is a frozen output format.

    `compress` and `nonqiita-to-parquet` both write it, so it is produced from
    one place; these pin the column names, order and types that Qiita and
    released micov versions read.
    """

    def setUp(self):
        self.d = mkdtemp()
        self.con = connection()
        self.addCleanup(self.con.close)
        self.addCleanup(shutil.rmtree, self.d)

    def test_written_schema_matches_the_frozen_contract(self):
        lengths = f"{self.d}/lengths.tsv"
        with open(lengths, "w") as fp:
            fp.write("genome_id\tlength\nX\t1000\n")
        load_genome_lengths(self.con, lengths)

        positions = (
            "SELECT 'X' AS genome_id, 1::UINTEGER AS start, "
            "83::UINTEGER AS stop, 'S1' AS sample_id"
        )
        base = f"{self.d}/out"
        write_coverage_parquet(self.con, positions, base)

        self.assertTrue(os.path.exists(f"{base}.covered_positions.parquet"))
        self.assertTrue(os.path.exists(f"{base}.coverage.parquet"))

        pos = self.con.sql(f"FROM '{base}.covered_positions.parquet'")
        self.assertEqual(
            list(zip(pos.columns, (str(t) for t in pos.types), strict=True)),
            [
                ("genome_id", "VARCHAR"),
                ("start", "UINTEGER"),
                ("stop", "UINTEGER"),
                ("sample_id", "VARCHAR"),
            ],
        )

        cov = self.con.sql(f"FROM '{base}.coverage.parquet'")
        self.assertEqual(
            list(zip(cov.columns, (str(t) for t in cov.types), strict=True)),
            [
                ("sample_id", "VARCHAR"),
                ("genome_id", "VARCHAR"),
                ("covered", "UINTEGER"),
                ("length", "BIGINT"),
                ("percent_covered", "DOUBLE"),
            ],
        )

        # 8.200000000000001, not 8.2: percent is `(covered / length) * 100` and
        # (82 / 1000) * 100 is not exactly 8.2 in a double. Pinning the literal
        # pins the order of operations -- reassociating to
        # `covered * 100 / length` would be caught rather than tolerated, and
        # it would move every published coverage value.
        self.assertEqual(
            cov.fetchall(), [("S1", "X", 82, 1000, 8.200000000000001)]
        )


if __name__ == "__main__":
    unittest.main()
