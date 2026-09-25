"""Negative controls for the golden comparators.

A harness that cannot fail is worse than no harness, because it reads as
coverage. Every comparator in ``_golden`` is exercised here in *both*
directions:

- too strict: a perturbation it is supposed to tolerate must still pass
- too loose: a perturbation it is supposed to catch must raise

The too-strict half is not padding. During the M0 audit two artifacts
compared byte-identical to their goldens on the first run despite being
genuinely nondeterministic, so a comparator calibrated by watching one run
of micov would have been built to demand byte equality and then flaked
forever. These tests pin the tolerance deliberately instead.

Fixtures here are synthetic and built in a tmpdir, so this module never
depends on ``example/`` (which ``MANIFEST.in`` prunes from the sdist) and
always runs in the fast tier.
"""

import gzip
import tempfile
import unittest
from pathlib import Path

from micov.tests._golden import (
    assert_file_set,
    assert_gzip_text_equal,
    assert_ks_equal,
    assert_parquet_equal,
    assert_png_plausible,
    assert_tsv_equal_unordered,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

KS_HEADER = "label_A,label_B,ks-statistic,ks-pvalue,ks-pvalue-bonferroni"
# two deterministic rows, so the Bonferroni family is m = 2; the second row's
# p * 2 exceeds 1 and is capped
KS_DETERMINISTIC = [
    "No,Yes,0.3,0.24244968766417713,0.48489937532835425",
    "No,not provided,0.3,0.5691054572613793,1.0",
]
# Monte Carlo rows are not corrected, so their last field is empty
KS_MONTE = [
    "No,Monte Carlo unfocused (n=20),0.1,0.9999923931635496,",
    "Yes,Monte Carlo unfocused (n=20),0.23684210526315788,0.5351657978006094,",
]

BIN_HEADER = "genome_id\tbin_idx\tbin_start\tbin_stop\tsample_hits_std"
# rows 2 and 3 are a deliberate tie on sample_hits_std -- polars orders ties
# arbitrarily, so their relative order must not matter
BIN_ROWS = [
    "G000154205\t0\t0\t100\t1.5",
    "G000154205\t1\t100\t200\t0.5",
    "G000154205\t2\t200\t300\t0.5",
]
BIN_KEYS = ("genome_id", "bin_idx")


class GoldenSelfTestBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def write(self, name, lines):
        """Write newline-terminated text and return its path."""
        path = self.tmp / name
        path.write_text("\n".join(lines) + "\n")
        return path

    def write_gz(self, name, lines, mtime):
        """Write gzip text with an explicit mtime, so bytes are controllable."""
        path = self.tmp / name
        with gzip.GzipFile(path, "wb", mtime=mtime) as fp:
            fp.write(("\n".join(lines) + "\n").encode())
        return path

    def write_parquet(self, name, select):
        """Write a parquet file from a literal SELECT, preserving column order."""
        import duckdb

        path = self.tmp / name
        con = duckdb.connect()
        try:
            con.execute(f"COPY ({select}) TO '{path}' (FORMAT PARQUET)")
        finally:
            con.close()
        return path


class TestAssertGzipTextEqual(GoldenSelfTestBase):
    """`.tsv.gz` bytes carry an mtime (source #2); content is stable."""

    def test_differing_bytes_same_content_passes(self):
        a = self.write_gz("a.tsv.gz", ["group\tx\ty", "d\t0\t1.5"], mtime=1)
        b = self.write_gz("b.tsv.gz", ["group\tx\ty", "d\t0\t1.5"], mtime=99999)
        self.assertNotEqual(a.read_bytes(), b.read_bytes())
        assert_gzip_text_equal(a, b)

    def test_changed_content_fails(self):
        a = self.write_gz("a.tsv.gz", ["group\tx\ty", "d\t0\t1.5"], mtime=1)
        b = self.write_gz("b.tsv.gz", ["group\tx\ty", "d\t0\t1.6"], mtime=1)
        with self.assertRaises(AssertionError):
            assert_gzip_text_equal(a, b)


class TestAssertParquetEqual(GoldenSelfTestBase):
    """Row order is unstable (source #6); column order is contractual."""

    #: `sample_id` last -- the column order `qiita-to-parquet` wrote, which
    #: `example/parquet/` carried until M4 regenerated it. Kept as the fixture
    #: because the pair below is what makes the ordered-schema check bite.
    TRAILING_ID = (
        "SELECT 'G1' AS genome_id, 10 AS covered, 100 AS length, "
        "10.0 AS percent_covered, 's1' AS sample_id"
    )
    TRAILING_ID_TWO_ROWS = TRAILING_ID + " UNION ALL SELECT 'G2', 20, 100, 20.0, 's2'"

    def test_identical_passes(self):
        a = self.write_parquet("a.parquet", self.TRAILING_ID_TWO_ROWS)
        b = self.write_parquet("b.parquet", self.TRAILING_ID_TWO_ROWS)
        assert_parquet_equal(a, b)

    def test_row_order_reversed_passes(self):
        a = self.write_parquet("a.parquet", self.TRAILING_ID_TWO_ROWS)
        reversed_rows = (
            "SELECT 'G2' AS genome_id, 20 AS covered, 100 AS length, "
            "20.0 AS percent_covered, 's2' AS sample_id "
            "UNION ALL SELECT 'G1', 10, 100, 10.0, 's1'"
        )
        b = self.write_parquet("b.parquet", reversed_rows)
        assert_parquet_equal(a, b)

    def test_column_order_change_fails(self):
        """This is exactly how micov's two Parquet producers used to differ."""
        a = self.write_parquet("a.parquet", self.TRAILING_ID)
        leading_id = (
            "SELECT 's1' AS sample_id, 'G1' AS genome_id, 10 AS covered, "
            "100 AS length, 10.0 AS percent_covered"
        )
        b = self.write_parquet("b.parquet", leading_id)
        with self.assertRaises(AssertionError):
            assert_parquet_equal(a, b)

    def test_column_order_change_with_identical_values_fails(self):
        """Isolate the *ordered* schema check.

        Same column name set, same values in the same positions -- only the
        names are swapped. The row comparison matches by position and so
        sees nothing wrong; only an order-sensitive schema check catches it.
        Without this, `test_column_order_change_fails` would still pass if
        the schema were compared unordered.
        """
        a = self.write_parquet("a.parquet", "SELECT 1 AS x, 1 AS y")
        b = self.write_parquet("b.parquet", "SELECT 1 AS y, 1 AS x")
        with self.assertRaises(AssertionError):
            assert_parquet_equal(a, b)

    def test_changed_float_fails(self):
        a = self.write_parquet("a.parquet", self.TRAILING_ID)
        changed = self.TRAILING_ID.replace(
            "10.0 AS percent_covered", "10.5 AS percent_covered"
        )
        b = self.write_parquet("b.parquet", changed)
        with self.assertRaises(AssertionError):
            assert_parquet_equal(a, b)

    def test_row_count_change_fails(self):
        a = self.write_parquet("a.parquet", self.TRAILING_ID)
        b = self.write_parquet("b.parquet", self.TRAILING_ID_TWO_ROWS)
        with self.assertRaises(AssertionError):
            assert_parquet_equal(a, b)

    def test_duplicate_multiplicity_change_fails(self):
        """Same row *set* and same count, different multiplicities.

        Only a multiset comparison catches this -- plain `EXCEPT` would
        report both sides as equal.
        """
        two_x_one_y = (
            "SELECT 'G1' AS genome_id, 10 AS covered, 100 AS length, "
            "10.0 AS percent_covered, 's1' AS sample_id "
            "UNION ALL SELECT 'G1', 10, 100, 10.0, 's1' "
            "UNION ALL SELECT 'G2', 20, 100, 20.0, 's2'"
        )
        one_x_two_y = (
            "SELECT 'G1' AS genome_id, 10 AS covered, 100 AS length, "
            "10.0 AS percent_covered, 's1' AS sample_id "
            "UNION ALL SELECT 'G2', 20, 100, 20.0, 's2' "
            "UNION ALL SELECT 'G2', 20, 100, 20.0, 's2'"
        )
        a = self.write_parquet("a.parquet", two_x_one_y)
        b = self.write_parquet("b.parquet", one_x_two_y)
        with self.assertRaises(AssertionError):
            assert_parquet_equal(a, b)

    def test_changed_dtype_fails(self):
        a = self.write_parquet("a.parquet", self.TRAILING_ID)
        retyped = self.TRAILING_ID.replace(
            "10 AS covered", "CAST(10 AS BIGINT) AS covered"
        )
        b = self.write_parquet("b.parquet", retyped)
        with self.assertRaises(AssertionError):
            assert_parquet_equal(a, b)


class TestAssertTsvEqualUnordered(GoldenSelfTestBase):
    """Binning tie order is unstable (sources #5, #7); values are not."""

    def test_tie_block_reordered_passes(self):
        a = self.write("a.tsv", [BIN_HEADER, *BIN_ROWS])
        swapped = [BIN_ROWS[0], BIN_ROWS[2], BIN_ROWS[1]]
        b = self.write("b.tsv", [BIN_HEADER, *swapped])
        assert_tsv_equal_unordered(a, b, sort_keys=BIN_KEYS)

    def test_changed_value_fails(self):
        a = self.write("a.tsv", [BIN_HEADER, *BIN_ROWS])
        bad = [BIN_ROWS[0], BIN_ROWS[1], "G000154205\t2\t200\t300\t0.6"]
        b = self.write("b.tsv", [BIN_HEADER, *bad])
        with self.assertRaises(AssertionError):
            assert_tsv_equal_unordered(a, b, sort_keys=BIN_KEYS)

    def test_changed_header_fails(self):
        a = self.write("a.tsv", [BIN_HEADER, *BIN_ROWS])
        b = self.write("b.tsv", [BIN_HEADER.replace("bin_idx", "idx"), *BIN_ROWS])
        with self.assertRaises(AssertionError):
            assert_tsv_equal_unordered(a, b, sort_keys=BIN_KEYS)

    def test_row_count_change_fails(self):
        a = self.write("a.tsv", [BIN_HEADER, *BIN_ROWS])
        b = self.write("b.tsv", [BIN_HEADER, *BIN_ROWS[:2]])
        with self.assertRaises(AssertionError):
            assert_tsv_equal_unordered(a, b, sort_keys=BIN_KEYS)

    def test_unknown_sort_key_fails(self):
        """A typo in sort_keys must be loud, not silently unsorted."""
        a = self.write("a.tsv", [BIN_HEADER, *BIN_ROWS])
        b = self.write("b.tsv", [BIN_HEADER, *BIN_ROWS])
        with self.assertRaises(AssertionError):
            assert_tsv_equal_unordered(a, b, sort_keys=("nope",))


class TestAssertTsvFloatTolerance(GoldenSelfTestBase):
    """`sample_hits_std` drifts a few ULP outside polars (source #9).

    The tolerance has to absorb that and nothing else, so these pin both
    edges: what it must let through, and what it must still catch.
    """

    STD = "sample_hits_std"

    def rows(self, last):
        return [
            "G000154205\t0\t0\t100\t1.5",
            "G000154205\t1\t100\t200\t0.5",
            f"G000154205\t2\t200\t300\t{last}",
        ]

    def test_ulp_drift_passes(self):
        # the exact drift measured against the committed binning golden
        a = self.write("a.tsv", [BIN_HEADER, *self.rows("1.1547005383792515")])
        b = self.write("b.tsv", [BIN_HEADER, *self.rows("1.1547005383792517")])
        assert_tsv_equal_unordered(
            a, b, sort_keys=BIN_KEYS, float_columns=(self.STD,)
        )

    def test_drift_beyond_tolerance_fails(self):
        # 1e-12 relative -- far below anything a reader would notice, and
        # still caught
        a = self.write("a.tsv", [BIN_HEADER, *self.rows("1.154700538379")])
        b = self.write("b.tsv", [BIN_HEADER, *self.rows("1.1547005383792517")])
        with self.assertRaises(AssertionError):
            assert_tsv_equal_unordered(
                a, b, sort_keys=BIN_KEYS, float_columns=(self.STD,)
            )

    def test_exact_columns_still_exact(self):
        """Naming one column approximate must not relax the others."""
        a = self.write("a.tsv", [BIN_HEADER, *BIN_ROWS])
        bad = [BIN_ROWS[0], BIN_ROWS[1], "G000154205\t2\t200\t301\t0.5"]
        b = self.write("b.tsv", [BIN_HEADER, *bad])
        with self.assertRaises(AssertionError):
            assert_tsv_equal_unordered(
                a, b, sort_keys=BIN_KEYS, float_columns=(self.STD,)
            )

    def test_tie_block_reordered_still_passes(self):
        """Drifted values must not pair two different rows against each other.

        Rows 2 and 3 tie on the approximate column, so it is excluded from the
        sort tie-break; otherwise a drift could align row 2 with row 3.
        """
        a = self.write("a.tsv", [BIN_HEADER, *BIN_ROWS])
        swapped = [BIN_ROWS[0], BIN_ROWS[2], BIN_ROWS[1]]
        b = self.write("b.tsv", [BIN_HEADER, *swapped])
        assert_tsv_equal_unordered(
            a, b, sort_keys=BIN_KEYS, float_columns=(self.STD,)
        )

    def test_unknown_float_column_fails(self):
        """A typo must be loud, not a silently exact comparison."""
        a = self.write("a.tsv", [BIN_HEADER, *BIN_ROWS])
        b = self.write("b.tsv", [BIN_HEADER, *BIN_ROWS])
        with self.assertRaises(AssertionError):
            assert_tsv_equal_unordered(
                a, b, sort_keys=BIN_KEYS, float_columns=("nope",)
            )

    def test_non_numeric_in_float_column_fails(self):
        a = self.write("a.tsv", [BIN_HEADER, *self.rows("0.5")])
        b = self.write("b.tsv", [BIN_HEADER, *self.rows("nan-ish")])
        with self.assertRaises(AssertionError):
            assert_tsv_equal_unordered(
                a, b, sort_keys=BIN_KEYS, float_columns=(self.STD,)
            )


class TestAssertKsEqual(GoldenSelfTestBase):
    """Monte Carlo rows are unseeded (source #1); the rest are published."""

    def test_identical_passes(self):
        rows = [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE]
        assert_ks_equal(self.write("a.ks.csv", rows), self.write("b.ks.csv", rows))

    def test_monte_pvalue_change_passes(self):
        a = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        drifted = [
            "No,Monte Carlo unfocused (n=20),0.15,0.8123456789,",
            "Yes,Monte Carlo unfocused (n=20),0.2,0.4999999999,",
        ]
        b = self.write("b.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *drifted])
        assert_ks_equal(a, b)

    def test_deterministic_pvalue_change_fails(self):
        """The tolerance is for ULP drift, not a moved p-value.

        Was ``...799`` against ``...713`` until M9, 3.5e-15 relative. That
        now sits *inside* ``TSV_FLOAT_REL_TOL`` by design, so the control
        moved to a difference the tolerance must not absorb.
        """
        a = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        bad = ["No,Yes,0.3,0.2424496876642,0.4848993753284", KS_DETERMINISTIC[1]]
        b = self.write("b.ks.csv", [KS_HEADER, *bad, *KS_MONTE])
        with self.assertRaises(AssertionError):
            assert_ks_equal(a, b)

    def test_deterministic_pvalue_ulp_drift_passes(self):
        """miint's p-value, which differs from scipy's by 1 ULP (source #10)."""
        a = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        drifted = [
            "No,Yes,0.3,0.24244968766417715,0.4848993753283543",
            KS_DETERMINISTIC[1],
        ]
        b = self.write("b.ks.csv", [KS_HEADER, *drifted, *KS_MONTE])
        assert_ks_equal(a, b)

    def test_deterministic_statistic_ulp_change_fails(self):
        """The statistic is exact -- the tolerance must not leak onto it.

        ``0.30000000000000004`` is precisely what a pre-#257 miint returns
        for this row, so this is the regression the exactness guards.
        """
        a = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        bad = [
            "No,Yes,0.30000000000000004,0.24244968766417713,0.48489937532835425",
            KS_DETERMINISTIC[1],
        ]
        b = self.write("b.ks.csv", [KS_HEADER, *bad, *KS_MONTE])
        with self.assertRaises(AssertionError):
            assert_ks_equal(a, b)

    def test_deterministic_label_change_fails(self):
        """Tolerating the p-value must not decouple it from its label pair."""
        a = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        bad = [
            "No,Maybe,0.3,0.24244968766417713,0.48489937532835425",
            KS_DETERMINISTIC[1],
        ]
        b = self.write("b.ks.csv", [KS_HEADER, *bad, *KS_MONTE])
        with self.assertRaises(AssertionError):
            assert_ks_equal(a, b)

    def test_deterministic_statistic_change_fails(self):
        a = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        bad = [
            "No,Yes,0.4,0.24244968766417713,0.48489937532835425",
            KS_DETERMINISTIC[1],
        ]
        b = self.write("b.ks.csv", [KS_HEADER, *bad, *KS_MONTE])
        with self.assertRaises(AssertionError):
            assert_ks_equal(a, b)

    def test_dropped_monte_row_fails(self):
        """Tolerating the values must not tolerate losing the comparison."""
        a = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        b = self.write("b.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, KS_MONTE[0]])
        with self.assertRaises(AssertionError):
            assert_ks_equal(a, b)

    def test_monte_pvalue_out_of_range_fails(self):
        """The range check guards the *observed* output, so it goes first."""
        bad = [
            "No,Monte Carlo unfocused (n=20),0.1,1.5,",
            KS_MONTE[1],
        ]
        observed = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *bad])
        golden = self.write("b.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        with self.assertRaises(AssertionError):
            assert_ks_equal(observed, golden)

    def test_monte_statistic_out_of_range_fails(self):
        bad = [
            "No,Monte Carlo unfocused (n=20),-0.1,0.5,",
            KS_MONTE[1],
        ]
        observed = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *bad])
        golden = self.write("b.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        with self.assertRaises(AssertionError):
            assert_ks_equal(observed, golden)

    def test_non_numeric_monte_value_fails(self):
        bad = [
            "No,Monte Carlo unfocused (n=20),nan,inf,",
            KS_MONTE[1],
        ]
        observed = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *bad])
        golden = self.write("b.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        with self.assertRaises(AssertionError):
            assert_ks_equal(observed, golden)

    def test_malformed_row_fails(self):
        """A short row must raise AssertionError, not ValueError from zip."""
        observed = self.write(
            "a.ks.csv", [KS_HEADER, "No,Yes,0.3", *KS_MONTE]
        )
        golden = self.write("b.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        with self.assertRaises(AssertionError):
            assert_ks_equal(observed, golden)

    def test_wrong_header_fails(self):
        observed = self.write("a.ks.csv", ["a,b,c,d,e", *KS_DETERMINISTIC])
        golden = self.write("b.ks.csv", [KS_HEADER, *KS_DETERMINISTIC])
        with self.assertRaises(AssertionError):
            assert_ks_equal(observed, golden)

    def test_dropped_deterministic_row_fails(self):
        a = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        b = self.write("b.ks.csv", [KS_HEADER, KS_DETERMINISTIC[0], *KS_MONTE])
        with self.assertRaises(AssertionError):
            assert_ks_equal(a, b)

    def test_bonferroni_ulp_drift_passes(self):
        """Derived from the p-value, so it inherits the p-value's tolerance."""
        a = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        drifted = [
            "No,Yes,0.3,0.24244968766417713,0.4848993753283543",
            KS_DETERMINISTIC[1],
        ]
        b = self.write("b.ks.csv", [KS_HEADER, *drifted, *KS_MONTE])
        assert_ks_equal(a, b)

    def test_bonferroni_change_fails(self):
        """A wrong family size is exactly this: p is right, p * m is not."""
        a = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        # m = 4 instead of 2, as if the Monte Carlo rows had been counted
        bad = ["No,Yes,0.3,0.24244968766417713,0.9697987506567085", KS_DETERMINISTIC[1]]
        b = self.write("b.ks.csv", [KS_HEADER, *bad, *KS_MONTE])
        with self.assertRaisesRegex(AssertionError, "bonferroni. differs"):
            assert_ks_equal(a, b)

    def test_corrected_monte_carlo_row_fails(self):
        """Monte Carlo comparisons are outside the family and stay empty."""
        bad = [
            "No,Monte Carlo unfocused (n=20),0.1,0.2,0.4",
            KS_MONTE[1],
        ]
        observed = self.write("a.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *bad])
        golden = self.write("b.ks.csv", [KS_HEADER, *KS_DETERMINISTIC, *KS_MONTE])
        with self.assertRaisesRegex(AssertionError, "Monte Carlo.*bonferroni"):
            assert_ks_equal(observed, golden)

    def test_pre_bonferroni_header_fails(self):
        """A file written before M11a has four columns, and is not current."""
        old = "label_A,label_B,ks-statistic,ks-pvalue"
        observed = self.write(
            "a.ks.csv", [old, "No,Yes,0.3,0.24244968766417713"]
        )
        golden = self.write("b.ks.csv", [KS_HEADER, KS_DETERMINISTIC[0]])
        with self.assertRaisesRegex(AssertionError, "header"):
            assert_ks_equal(observed, golden)


class TestAssertPngPlausible(GoldenSelfTestBase):
    def test_valid_png_passes(self):
        path = self.tmp / "a.png"
        path.write_bytes(PNG_MAGIC + b"\x00" * 64)
        assert_png_plausible(path)

    def test_empty_fails(self):
        path = self.tmp / "a.png"
        path.write_bytes(b"")
        with self.assertRaises(AssertionError):
            assert_png_plausible(path)

    def test_not_a_png_fails(self):
        path = self.tmp / "a.png"
        path.write_bytes(b"<html>oops</html>")
        with self.assertRaises(AssertionError):
            assert_png_plausible(path)

    def test_missing_fails(self):
        with self.assertRaises(AssertionError):
            assert_png_plausible(self.tmp / "absent.png")

    def test_magic_only_fails(self):
        """A truncated write is a real failure, not a plausible plot."""
        path = self.tmp / "a.png"
        path.write_bytes(PNG_MAGIC)
        with self.assertRaises(AssertionError):
            assert_png_plausible(path)


class TestAssertFileSet(GoldenSelfTestBase):
    def setUp(self):
        super().setUp()
        self.outdir = self.tmp / "out"
        self.outdir.mkdir()
        for name in ("a.png", "b.ks.csv"):
            (self.outdir / name).write_text("x")

    def test_exact_match_passes(self):
        assert_file_set(self.outdir, {"a.png", "b.ks.csv"})

    def test_extra_file_fails(self):
        with self.assertRaises(AssertionError):
            assert_file_set(self.outdir, {"a.png"})

    def test_missing_file_fails(self):
        with self.assertRaises(AssertionError):
            assert_file_set(self.outdir, {"a.png", "b.ks.csv", "c.tsv"})


class TestComparatorsRejectDirectories(GoldenSelfTestBase):
    """A path that exists but is not a file must not read as success."""

    def test_directory_is_not_a_valid_artifact(self):
        d = self.tmp / "adir"
        d.mkdir()
        with self.assertRaises(AssertionError):
            assert_png_plausible(d)


if __name__ == "__main__":
    unittest.main()
