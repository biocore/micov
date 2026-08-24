import csv
import shutil
import unittest
from tempfile import mkdtemp

import duckdb

from micov._constants import (
    ABSENT,
    COLUMN_COVERED,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_NAME,
    COLUMN_PERCENT_COVERED,
    COLUMN_REGION_ID,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
    NOT_APPLICABLE,
    PRESENT,
)
from micov._view import View

# The View hands back DuckDB relations, so expectations are plain rows plus the
# SQL type of each column. Since M5 these are the *only* statement of the
# fixture types: `_constants.py` no longer carries dtypes, and `make_cov_pos`
# below builds the parquet from these same tuples, so a type can't drift
# between what is written and what is asserted.
METADATA_SCHEMA = (
    (COLUMN_SAMPLE_ID, "VARCHAR"),
    ("foo", "VARCHAR"),
)
COVERAGE_SCHEMA = (
    (COLUMN_GENOME_ID, "VARCHAR"),
    (COLUMN_SAMPLE_ID, "VARCHAR"),
    (COLUMN_COVERED, "UINTEGER"),
    (COLUMN_LENGTH, "UINTEGER"),
    (COLUMN_PERCENT_COVERED, "DOUBLE"),
)
POSITION_SCHEMA = (
    (COLUMN_GENOME_ID, "VARCHAR"),
    (COLUMN_SAMPLE_ID, "VARCHAR"),
    (COLUMN_START, "UINTEGER"),
    (COLUMN_STOP, "UINTEGER"),
)
FEATURE_METADATA_SCHEMA = (
    (COLUMN_GENOME_ID, "VARCHAR"),
    (COLUMN_START, "UINTEGER"),
    (COLUMN_STOP, "UINTEGER"),
    (COLUMN_LENGTH, "UINTEGER"),
    (COLUMN_REGION_ID, "VARCHAR"),
)
FEATURE_NAME_SCHEMA = (
    (COLUMN_GENOME_ID, "VARCHAR"),
    (COLUMN_NAME, "VARCHAR"),
)

# the fixture corpus, shared by the parquet writer and the expectations so the
# two cannot drift apart
COVERAGE_ROWS = [
    ("G1", "S1", 8, 100, 8.0),
    ("G2", "S1", 20, 100, 20.0),
    ("G3", "S1", 30, 100, 30.0),
    ("G3", "S2", 11, 100, 11.0),
    ("G4", "S2", 21, 100, 21.0),
    ("G2", "S3", 52, 100, 52.0),
    ("G3", "S3", 62, 100, 62.0),
    ("G4", "S3", 72, 100, 72.0),
    ("G5", "S3", 82, 100, 82.0),
    ("G4", "S1", 20, 100, 20.0),
]
POSITION_ROWS = [
    ("G1", "S1", 1, 5),
    ("G1", "S1", 6, 10),
    ("G2", "S1", 30, 50),
    ("G3", "S1", 15, 30),
    ("G3", "S1", 75, 90),
    ("G4", "S1", 5, 15),
    ("G4", "S1", 45, 55),
    ("G3", "S2", 1, 11),
    ("G4", "S2", 30, 51),
    ("G2", "S3", 1, 52),
    ("G3", "S3", 1, 62),
    ("G4", "S3", 8, 13),
    ("G4", "S3", 20, 87),
    ("G5", "S3", 10, 92),
]


def _write_parquet(con, path, schema, rows):
    """Write `rows` to parquet with exactly the types `schema` declares."""
    columns = ", ".join(f'"{name}" {sql_type}' for name, sql_type in schema)
    placeholders = ", ".join("?" * len(schema))
    con.execute(f"CREATE OR REPLACE TABLE fixture ({columns})")
    con.executemany(f"INSERT INTO fixture VALUES ({placeholders})", rows)
    con.execute(f"COPY fixture TO '{path}' (FORMAT PARQUET)")


def make_cov_pos(d, name):
    """Build the parquet pair the View reads.

    Written with DuckDB since M5. It was polars, which is the library released
    micov used to write these files with -- so nothing in the suite now
    produces parquet the way those released versions did. R7 checked that
    round-trip in both directions and `example/coverages/*.cov.gz` remain as
    independent artifacts, but the loss is real and deliberate.
    """
    con = duckdb.connect()
    try:
        _write_parquet(
            con, f"{d}/{name}.coverage.parquet", COVERAGE_SCHEMA, COVERAGE_ROWS
        )
        _write_parquet(
            con,
            f"{d}/{name}.covered_positions.parquet",
            POSITION_SCHEMA,
            POSITION_ROWS,
        )
    finally:
        con.close()


def by_sample(rows, samples):
    return [row for row in rows if row[1] in samples]


def by_genome(rows, genomes):
    return [row for row in rows if row[0] in genomes]


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.d = mkdtemp()
        self.name = "testdata"
        make_cov_pos(self.d, self.name)

        # (header, rows). Metadata reaches View as a TSV on disk, so there is
        # nothing for a dataframe to do here but hold the rows on the way out;
        # the types below are the file's, and View re-types on read.
        self.md = (
            (COLUMN_SAMPLE_ID, "foo"),
            [["S1", "a"], ["S2", "b"], ["S3", "c"], ["S4", "d"], ["S5", "e"]],
        )
        self.feat = (
            (COLUMN_GENOME_ID,),
            [["G1"], ["G2"], ["G3"], ["G4"], ["G5"], ["G6"]],
        )

    def tsv(self, header, rows):
        """Write a fixture out as a TSV and return its path.

        View reads its metadata straight from disk with DuckDB rather than
        accepting a parsed frame, so fixtures are materialized the same way a
        user supplies them.
        """
        self._tsv_count = getattr(self, "_tsv_count", 0) + 1
        path = f"{self.d}/fixture{self._tsv_count}.tsv"
        with open(path, "w", newline="") as fp:
            writer = csv.writer(fp, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def tearDown(self):
        shutil.rmtree(self.d)

    def assert_relation(self, relation, schema, rows):
        """Compare a View relation against expected columns, types and rows.

        Column order and row order are both insensitive -- the View's three
        branches emit the same columns in different orders -- but the column
        names, their SQL types and the values are all checked. The types carry
        their own weight: a UINTEGER quietly widening to BIGINT changes a
        parquet column type, which is a frozen output contract, and is
        invisible in the values.

        `schema` fixes the order `rows` are written in, so the expectations do
        not have to track whichever order the relation happens to use.
        """
        columns = list(relation.columns)
        observed_schema = dict(
            zip(columns, (str(t) for t in relation.types), strict=True)
        )
        self.assertEqual(observed_schema, dict(schema))

        order = [columns.index(name) for name, _ in schema]
        observed = sorted(tuple(row[i] for i in order) for row in relation.fetchall())
        self.assertEqual(observed, sorted(rows))

    def test_view_sample_superset(self):
        v = View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*self.feat))

        # S4 and S5 have no coverage, so they drop out of the metadata
        self.assert_relation(
            v.metadata(), METADATA_SCHEMA, [("S1", "a"), ("S2", "b"), ("S3", "c")]
        )
        self.assert_relation(v.coverages(), COVERAGE_SCHEMA, COVERAGE_ROWS)
        self.assert_relation(v.positions(), POSITION_SCHEMA, POSITION_ROWS)

    def test_view_sample_subset(self):
        header, rows = self.md
        md = (header, [row for row in rows if row[0] in ("S1", "S3", "S5")])
        v = View(f"{self.d}/{self.name}", self.tsv(*md), None)

        kept = ("S1", "S3")
        self.assert_relation(
            v.metadata(), METADATA_SCHEMA, [("S1", "a"), ("S3", "c")]
        )
        self.assert_relation(
            v.coverages(), COVERAGE_SCHEMA, by_sample(COVERAGE_ROWS, kept)
        )
        self.assert_relation(
            v.positions(), POSITION_SCHEMA, by_sample(POSITION_ROWS, kept)
        )
        self.assert_relation(
            v.feature_metadata(),
            FEATURE_METADATA_SCHEMA,
            [
                ("G1", 0, 100, 100, "G1_0_100"),
                ("G2", 0, 100, 100, "G2_0_100"),
                ("G3", 0, 100, 100, "G3_0_100"),
                ("G4", 0, 100, 100, "G4_0_100"),
                ("G5", 0, 100, 100, "G5_0_100"),
            ],
        )
        self.assert_relation(
            v.feature_names(),
            FEATURE_NAME_SCHEMA,
            [("G1", "G1"), ("G2", "G2"), ("G3", "G3"), ("G4", "G4"), ("G5", "G5")],
        )

    def test_view_constrain_features(self):
        header, rows = self.feat
        feat = (header, [row for row in rows if row[0] in ("G1", "G5", "G6")])

        v = View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

        # G6 is requested but has no coverage, so it never appears
        kept = ("G1", "G5")
        self.assert_relation(
            v.metadata(), METADATA_SCHEMA, [("S1", "a"), ("S2", "b"), ("S3", "c")]
        )
        self.assert_relation(
            v.coverages(), COVERAGE_SCHEMA, by_genome(COVERAGE_ROWS, kept)
        )
        self.assert_relation(
            v.positions(), POSITION_SCHEMA, by_genome(POSITION_ROWS, kept)
        )
        self.assert_relation(
            v.feature_metadata(),
            FEATURE_METADATA_SCHEMA,
            [("G1", 0, 100, 100, "G1_0_100"), ("G5", 0, 100, 100, "G5_0_100")],
        )

    def test_view_constrain_positions_full(self):
        feat = (
            (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP),
            [["G1", 0, 1000], ["G5", 0, 1000]],
        )
        v = View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

        self.assert_relation(
            v.metadata(), METADATA_SCHEMA, [("S1", "a"), ("S2", "b"), ("S3", "c")]
        )
        # breadth is unchanged but is now against the 1000bp region rather than
        # the 100bp genome, so every percent drops by a factor of ten.
        #
        # 8.200000000000001, not 8.2: the percent is `(covered / length) * 100`
        # and (82 / 1000) * 100 is not exactly 8.2 in a double. The literal is
        # deliberate -- it pins the order of operations, so reassociating to
        # `covered * 100 / length` would be caught rather than tolerated.
        self.assert_relation(
            v.coverages(),
            COVERAGE_SCHEMA,
            [
                ("G1", "S1", 8, 1000, 0.8),
                ("G5", "S3", 82, 1000, 8.200000000000001),
            ],
        )
        self.assert_relation(
            v.positions(), POSITION_SCHEMA, by_genome(POSITION_ROWS, ("G1", "G5"))
        )
        # the user is requesting a stop position outside of the size the genome but ok?
        self.assert_relation(
            v.feature_metadata(),
            FEATURE_METADATA_SCHEMA,
            [
                ("G1", 0, 1000, 1000, "G1_0_1000"),
                ("G5", 0, 1000, 1000, "G5_0_1000"),
            ],
        )

    def test_view_constrain_positions_none(self):
        feat = (
            (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP),
            [["G1", 1000, 2000], ["G5", 1000, 2000]],
        )

        with self.assertRaisesRegex(ValueError, "No positions left"):
            View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

    def test_view_constrain_positions_bounds_simple(self):
        feat = (
            (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP),
            [["G1", 7, 9], ["G5", 0, 20]],
        )
        v = View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

        self.assert_relation(
            v.metadata(), METADATA_SCHEMA, [("S1", "a"), ("S2", "b"), ("S3", "c")]
        )
        self.assert_relation(
            v.coverages(),
            COVERAGE_SCHEMA,
            [("G1", "S1", 2, 2, 100.0), ("G5", "S3", 10, 20, 50.0)],
        )
        # both start and stop are clipped for G1/S1
        # left bound of G5/S3 is outside of its interval so verify the correct
        # start is retained
        self.assert_relation(
            v.positions(),
            POSITION_SCHEMA,
            [("G1", "S1", 7, 9), ("G5", "S3", 10, 20)],
        )
        self.assert_relation(
            v.feature_metadata(),
            FEATURE_METADATA_SCHEMA,
            [("G1", 7, 9, 2, "G1_7_9"), ("G5", 0, 20, 20, "G5_0_20")],
        )

    def test_view_constrain_positions_bounds_complex(self):
        feat = (
            (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP),
            [
                ["G1", 0, 100],
                ["G2", 40, 60],
                ["G3", 40, 60],
                ["G4", 90, 100],
                ["G5", 40, 60],
            ],
        )
        v = View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

        self.assert_relation(
            v.metadata(), METADATA_SCHEMA, [("S1", "a"), ("S2", "b"), ("S3", "c")]
        )
        self.assert_relation(
            v.coverages(),
            COVERAGE_SCHEMA,
            [
                ("G1", "S1", 8, 100, 8.0),
                ("G2", "S1", 10, 20, 50.0),
                ("G2", "S3", 12, 20, 60.0),
                ("G3", "S3", 20, 20, 100.0),
                ("G5", "S3", 20, 20, 100.0),
            ],
        )
        self.assert_relation(
            v.positions(),
            POSITION_SCHEMA,
            [
                ("G1", "S1", 1, 5),
                ("G1", "S1", 6, 10),
                ("G2", "S1", 40, 50),
                ("G2", "S3", 40, 52),
                ("G3", "S3", 40, 60),
                ("G5", "S3", 40, 60),
            ],
        )
        # G4's region is requested but no sample has coverage inside it
        self.assert_relation(
            v.feature_metadata(),
            FEATURE_METADATA_SCHEMA,
            [
                ("G1", 0, 100, 100, "G1_0_100"),
                ("G2", 40, 60, 20, "G2_40_60"),
                ("G3", 40, 60, 20, "G3_40_60"),
                ("G5", 40, 60, 20, "G5_40_60"),
            ],
        )

    # A region is half-open [start, stop), so an interval beginning exactly at
    # `stop` lies outside it. micov's overlap predicate was `pos.start <=
    # fc.stop`, which admitted that interval, clipped it to a zero-width
    # [stop, stop) contributing no bases, and still called the sample present.
    # These three tests pin the corrected `<`, decided as part of adopting
    # miint's region primitives -- the predicate lives inside them, so it was
    # not separable from the migration.
    #
    # G3's region ends at 15 and S1's G3 interval starts at 15. S2 and S3 do
    # cover the region, so it survives the SEMI JOIN and stays observable
    # rather than vanishing along with S1.
    ABUTTING_REGION = (
        (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP),
        [["G3", 1, 15]],
    )

    def test_abutting_interval_is_outside_the_region(self):
        """An interval starting at the region's exclusive end must not count.

        S1's only G3 intervals are [15, 30) and [75, 90); the region is
        [1, 15). S1 covers none of it, so it must contribute neither a
        position nor a coverage row -- previously it contributed both, with
        `covered` of 0.
        """
        v = View(
            f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*self.ABUTTING_REGION)
        )

        self.assert_relation(
            v.positions(),
            POSITION_SCHEMA,
            [("G3", "S2", 1, 11), ("G3", "S3", 1, 15)],
        )
        self.assert_relation(
            v.coverages(),
            COVERAGE_SCHEMA,
            [
                ("G3", "S2", 10, 14, 71.42857142857143),
                ("G3", "S3", 14, 14, 100.0),
            ],
        )

    def test_abutting_interval_is_not_present(self):
        """The same interval must not read as present either.

        This is the user-visible half: `extract-sample-presence` reported S1
        present in a region it covers zero bases of. `absent` rather than
        `not applicable` because S1 does have G3 coverage -- just not here.
        """
        v = View(
            f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*self.ABUTTING_REGION)
        )

        self.assert_relation(
            v.sample_presence_absence(),
            ((COLUMN_SAMPLE_ID, "VARCHAR"), ("G3_1_15", "VARCHAR")),
            [("S1", ABSENT), ("S2", PRESENT), ("S3", PRESENT)],
        )

    def test_no_zero_width_intervals_reach_positions(self):
        """Clipping must not manufacture empty intervals.

        Stated separately from the row-level expectation above because it is
        the invariant the rest of micov relies on: `_quant.py` bins these
        intervals and `_plot.py` draws them, and a [stop, stop) segment is
        meaningless to both.
        """
        v = View(
            f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*self.ABUTTING_REGION)
        )

        # by name, not by position: the region branch emits `positions` in a
        # different column order from the other two
        rows = v.positions().select(f"{COLUMN_START}, {COLUMN_STOP}").fetchall()
        self.assertTrue(rows)
        for start, stop in rows:
            with self.subTest(start=start, stop=stop):
                self.assertLess(start, stop)

    def test_zero_width_region_is_rejected(self):
        """A region with no width has no denominator, and no user intent.

        It reached `ValueError: No positions left after filtering.` before --
        true, but it describes the symptom rather than the malformed row, and
        it is the same message a perfectly good region that happens to match
        nothing produces.
        """
        feat = ((COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP), [["G1", 5, 5]])
        with self.assertRaisesRegex(ValueError, r"'G1'.*\[5, 5\)"):
            View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

    def test_inverted_region_is_rejected(self):
        """Reversed coordinates fail today, but not in a way anyone can act on.

        `feature_metadata` computes `stop - start` on UINTEGER columns, so a
        transposed pair of columns surfaces as `Out of Range Error: Overflow in
        subtraction of UINT32 (40 - 60)` -- from DuckDB, naming neither the
        genome, the file, nor the fact that a region is half-open.
        """
        feat = (
            (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP),
            [["G1", 0, 100], ["G2", 60, 40]],
        )
        with self.assertRaisesRegex(ValueError, r"'G2'.*\[60, 40\)"):
            View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

    def test_two_regions_on_one_genome_are_scored_separately(self):
        """Each region gets its own numerator, not the genome's total.

        Two regions on one genome is an ordinary thing to ask for -- two genes,
        say -- and micov summed the clipped intervals across *both* before
        dividing by *each* region's length. That reported S3 as covering 21
        bases of a 20bp region: `percent_covered` of 105.0, and S1 at 100%
        of a region it covers a quarter of.

        Every value below is hand-checkable against POSITION_ROWS, which is the
        point -- G3 holds [15,30) and [75,90) for S1, [1,11) for S2, [1,62)
        for S3.
        """
        feat = (
            (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP),
            [["G3", 0, 20], ["G3", 60, 100]],
        )
        v = View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

        self.assert_relation(
            v.coverages(),
            COVERAGE_SCHEMA,
            [
                # region [0, 20): S1 covers [15,20), S2 [1,11), S3 [1,20)
                ("G3", "S1", 5, 20, 25.0),
                ("G3", "S2", 10, 20, 50.0),
                ("G3", "S3", 19, 20, 95.0),
                # region [60, 100): S1 covers [75,90), S3 [60,62), S2 nothing
                ("G3", "S1", 15, 40, 37.5),
                ("G3", "S3", 2, 40, 5.0),
            ],
        )
        # the intervals stay separate islands rather than merging across the
        # gap between the two regions
        self.assert_relation(
            v.positions(),
            POSITION_SCHEMA,
            [
                ("G3", "S1", 15, 20),
                ("G3", "S1", 75, 90),
                ("G3", "S2", 1, 11),
                ("G3", "S3", 1, 20),
                ("G3", "S3", 60, 62),
            ],
        )

    def test_breadth_never_exceeds_the_region(self):
        """The invariant the case above violated, stated on its own.

        A percentage over 100 is not a rounding question -- it means the
        numerator and the denominator are measuring different things.
        """
        feat = (
            (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP),
            [["G3", 0, 20], ["G3", 60, 100]],
        )
        v = View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

        rows = v.coverages().select(
            f"{COLUMN_SAMPLE_ID}, {COLUMN_COVERED}, {COLUMN_LENGTH}, "
            f"{COLUMN_PERCENT_COVERED}"
        ).fetchall()
        self.assertTrue(rows)
        for sample, covered, length, percent in rows:
            with self.subTest(sample=sample):
                self.assertLessEqual(covered, length)
                self.assertLessEqual(percent, 100.0)

    def test_presence_covers_only_the_constrained_samples(self):
        """`extract-sample-presence` must respect the metadata it was given.

        Presence was derived from the unfiltered position table, so a metadata
        file naming one sample still produced a row for every sample in the
        parquet -- the one constraint the user actually asked for, ignored.
        S1 and S3 both have G3 coverage, so they are what leaked.
        """
        header, rows = self.md
        md = (header, [row for row in rows if row[0] == "S2"])
        v = View(
            f"{self.d}/{self.name}", self.tsv(*md), self.tsv(*self.ABUTTING_REGION)
        )

        self.assert_relation(
            v.sample_presence_absence(),
            ((COLUMN_SAMPLE_ID, "VARCHAR"), ("G3_1_15", "VARCHAR")),
            [("S2", PRESENT)],
        )

    def test_sample_presence_absence_no_regions(self):
        v = View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*self.feat))
        with self.assertRaisesRegex(ValueError, r"^Cannot calculate"):
            v.sample_presence_absence()

    def test_sample_presence_absence_single_region(self):
        feat = (
            (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP),
            [
                ["G1", 0, 100],
                ["G2", 40, 60],
                ["G3", 40, 60],
                ["G4", 90, 100],
                ["G5", 40, 60],
            ],
        )
        v = View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

        schema = (
            (COLUMN_SAMPLE_ID, "VARCHAR"),
            ("G1_0_100", "VARCHAR"),
            ("G2_40_60", "VARCHAR"),
            ("G3_40_60", "VARCHAR"),
            ("G5_40_60", "VARCHAR"),
        )
        # absent means the sample covers the genome but not the region; not
        # applicable means it has no coverage of that genome at all
        self.assert_relation(
            v.sample_presence_absence(),
            schema,
            [
                ("S1", PRESENT, PRESENT, ABSENT, NOT_APPLICABLE),
                ("S2", NOT_APPLICABLE, NOT_APPLICABLE, ABSENT, NOT_APPLICABLE),
                ("S3", NOT_APPLICABLE, PRESENT, PRESENT, PRESENT),
            ],
        )

    def test_integrity_checks(self):
        feat = (
            (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP),
            [
                ["G1", 0, 100],
                ["G2", 40, 60],
                ["G3", 40, 60],
                ["G3", 40, 60],
                ["G4", 90, 100],
                ["G5", 40, 60],
            ],
        )
        with self.assertRaisesRegex(ValueError, "Region IDs are not unique"):
            View(f"{self.d}/{self.name}", self.tsv(*self.md), self.tsv(*feat))

    def test_feature_names(self):
        names = ((COLUMN_GENOME_ID, COLUMN_NAME), [["G1", "foo"], ["G2", "bar"]])

        v = View(
            f"{self.d}/{self.name}",
            self.tsv(*self.md),
            self.tsv(*self.feat),
            self.tsv(*names),
        )

        # a genome with no supplied name falls back to its own id
        self.assert_relation(
            v.feature_names(),
            FEATURE_NAME_SCHEMA,
            [
                ("G1", "foo"),
                ("G2", "bar"),
                ("G3", "G3"),
                ("G4", "G4"),
                ("G5", "G5"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
