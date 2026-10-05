"""Paths and IDs containing a single quote.

micov builds SQL by interpolating paths -- and `compress --sample-id` -- into
string literals. Until M11b none of them was escaped, so an ordinary macOS home
such as ``/Users/o'brien`` made every command fail with a DuckDB ``Parser
Error`` that named neither the path nor micov. These values come from the
user's own command line, so this was never a security boundary; it was a
correctness one, and the fix is the same either way: every literal goes
through `_utils.sql_string`.

Each test drives a different interpolation site, so reverting the helper to
the identity turns them all red.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb

from micov._miint import MIINT_EXTENSION_PATH_VARIABLE, connection
from micov._utils import sql_string
from micov._view import View
from micov.tests.test_equivalence import DATA, MicovCliTestCase, requires_micov
from micov.tests.test_miint import installed_extension_path, requires_miint_build

#: A directory name with the character that broke every literal, and a space
#: for good measure.
AWKWARD = "o'brien data"


def distinct_samples(parquet):
    """Read `sample_id`s with a bound parameter, independent of micov."""
    rows = duckdb.execute(
        "SELECT DISTINCT sample_id FROM read_parquet(?) ORDER BY 1", [str(parquet)]
    ).fetchall()
    return [row[0] for row in rows]


class SqlStringTests(unittest.TestCase):
    def test_plain_text_is_quoted(self):
        self.assertEqual(sql_string("abc"), "'abc'")

    def test_embedded_quote_is_doubled(self):
        self.assertEqual(sql_string("a'b"), "'a''b'")

    def test_quote_alone(self):
        self.assertEqual(sql_string("'"), "''''")

    def test_empty(self):
        self.assertEqual(sql_string(""), "''")

    def test_accepts_a_path(self):
        self.assertEqual(sql_string(Path("/x/o'b")), "'/x/o''b'")

    def test_round_trips_through_duckdb(self):
        """The real claim: DuckDB reads back exactly the original value."""
        for value in ("o'brien", "''", "a'b'c", "'", "plain"):
            with self.subTest(value=value):
                (got,) = duckdb.sql(f"SELECT {sql_string(value)}").fetchone()
                self.assertEqual(got, value)


@requires_micov
class ApostropheCliTests(MicovCliTestCase):
    def setUp(self):
        super().setUp()
        self.awkward = self.tmp / AWKWARD
        self.awkward.mkdir()

    def stage(self, *names):
        for name in names:
            shutil.copy(DATA / name, self.awkward / name)

    def test_compress_under_an_awkward_directory_and_sample_id(self):
        """Covers read_alignments, the sample-id literal and both COPY TOs."""
        self.stage("test.sam.xz", "lengths.tsv")
        out = self.awkward / "out"
        self.micov(
            "compress",
            "--data", self.awkward / "test.sam.xz",
            "--lengths", self.awkward / "lengths.tsv",
            "--sample-id", "o'brien",
            "--output", out,
        )
        for suffix in ("coverage", "covered_positions"):
            with self.subTest(file=suffix):
                self.assertEqual(
                    distinct_samples(f"{out}.{suffix}.parquet"), ["o'brien"]
                )

    def test_cov_to_parquet_under_an_awkward_directory(self):
        """Covers the read_csv glob and the lengths reader."""
        self.stage("mini_sampleA.cov", "mini_sampleB.cov", "mini_lengths.tsv")
        out = self.awkward / "mini"
        self.micov(
            "cov-to-parquet",
            "--pattern", self.awkward / "*.cov",
            "--lengths", self.awkward / "mini_lengths.tsv",
            "--output", out,
        )
        self.assertEqual(
            distinct_samples(f"{out}.coverage.parquet"),
            ["mini_sampleA", "mini_sampleB"],
        )

    def build_parquet(self):
        self.stage(
            "mini_sampleA.cov",
            "mini_sampleB.cov",
            "mini_lengths.tsv",
            "mini_sample_metadata.tsv",
            "mini_regions.tsv",
        )
        out = self.awkward / "mini"
        self.micov(
            "cov-to-parquet",
            "--pattern", self.awkward / "*.cov",
            "--lengths", self.awkward / "mini_lengths.tsv",
            "--output", out,
        )
        return out

    def test_view_with_regions_under_an_awkward_directory(self):
        """Covers `View`'s metadata reader and its position-level sources."""
        out = self.build_parquet()
        view = View(
            str(out),
            str(self.awkward / "mini_sample_metadata.tsv"),
            str(self.awkward / "mini_regions.tsv"),
        )
        self.addCleanup(view.close)
        presence = dict(view.sample_presence_absence().fetchall())
        self.assertEqual(presence["mini_sampleA"], "present")

    def test_view_with_genomes_under_an_awkward_directory(self):
        """Covers `View`'s genome-level (no start/stop) sources."""
        out = self.build_parquet()
        features = self.awkward / "genomes.tsv"
        features.write_text("genome_id\nG000000001\n")
        view = View(
            str(out), str(self.awkward / "mini_sample_metadata.tsv"), str(features)
        )
        self.addCleanup(view.close)
        (count,) = view.con.sql("SELECT COUNT(*) FROM coverage").fetchone()
        self.assertGreater(count, 0)


@requires_miint_build
class ApostropheExtensionPathTests(unittest.TestCase):
    def test_override_path_with_an_apostrophe_loads(self):
        """Covers `LOAD` of `MICOV_MIINT_EXTENSION_PATH`."""
        source = installed_extension_path()
        if source is None or not os.path.exists(source):
            self.skipTest("no installed miint build to point the override at")

        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        awkward = tmp / AWKWARD
        awkward.mkdir()
        target = awkward / os.path.basename(source)
        shutil.copy(source, target)

        with mock.patch.dict(os.environ, {MIINT_EXTENSION_PATH_VARIABLE: str(target)}):
            con = connection()
        self.addCleanup(con.close)
        (version,) = con.sql("SELECT miint_version()").fetchone()
        self.assertTrue(version)


if __name__ == "__main__":
    unittest.main()
