import shutil
import unittest
from tempfile import mkdtemp

import polars as pl

from micov._constants import (
    ABSENT,
    COLUMN_COVERED,
    COLUMN_COVERED_DTYPE,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_LENGTH_DTYPE,
    COLUMN_NAME,
    COLUMN_PERCENT_COVERED,
    COLUMN_PERCENT_COVERED_DTYPE,
    COLUMN_REGION_ID,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_START_DTYPE,
    COLUMN_STOP,
    COLUMN_STOP_DTYPE,
    NOT_APPLICABLE,
    PRESENT,
)
from micov._view import View

# The View hands back DuckDB relations, so expectations are plain rows plus the
# SQL type of each column. The types restate `_constants.py` in DuckDB's terms:
# UINTEGER is pl.UInt32, DOUBLE is float, VARCHAR is str.
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


def make_cov_pos(d, name):
    (
        pl.LazyFrame(
            COVERAGE_ROWS,
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, str),
                (COLUMN_SAMPLE_ID, str),
                (COLUMN_COVERED, COLUMN_COVERED_DTYPE),
                (COLUMN_LENGTH, COLUMN_LENGTH_DTYPE),
                (COLUMN_PERCENT_COVERED, COLUMN_PERCENT_COVERED_DTYPE),
            ],
        ).sink_parquet(f"{d}/{name}.coverage.parquet")
    )

    (
        pl.LazyFrame(
            POSITION_ROWS,
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, str),
                (COLUMN_SAMPLE_ID, str),
                (COLUMN_START, COLUMN_START_DTYPE),
                (COLUMN_STOP, COLUMN_STOP_DTYPE),
            ],
        ).sink_parquet(f"{d}/{name}.covered_positions.parquet")
    )


def by_sample(rows, samples):
    return [row for row in rows if row[1] in samples]


def by_genome(rows, genomes):
    return [row for row in rows if row[0] in genomes]


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.d = mkdtemp()
        self.name = "testdata"
        make_cov_pos(self.d, self.name)

        self.md = pl.DataFrame(
            [["S1", "a"], ["S2", "b"], ["S3", "c"], ["S4", "d"], ["S5", "e"]],
            orient="row",
            schema=[(COLUMN_SAMPLE_ID, str), ("foo", str)],
        )
        self.feat = pl.DataFrame(
            [["G1"], ["G2"], ["G3"], ["G4"], ["G5"], ["G6"]],
            orient="row",
            schema=[(COLUMN_GENOME_ID, str)],
        )

    def tsv(self, df):
        """Write a fixture frame out as a TSV and return its path.

        View reads its metadata straight from disk with DuckDB rather than
        accepting a parsed frame, so fixtures are materialized the same way a
        user supplies them.
        """
        self._tsv_count = getattr(self, "_tsv_count", 0) + 1
        path = f"{self.d}/fixture{self._tsv_count}.tsv"
        df.write_csv(path, separator="\t")
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
        v = View(f"{self.d}/{self.name}", self.tsv(self.md), self.tsv(self.feat))

        # S4 and S5 have no coverage, so they drop out of the metadata
        self.assert_relation(
            v.metadata(), METADATA_SCHEMA, [("S1", "a"), ("S2", "b"), ("S3", "c")]
        )
        self.assert_relation(v.coverages(), COVERAGE_SCHEMA, COVERAGE_ROWS)
        self.assert_relation(v.positions(), POSITION_SCHEMA, POSITION_ROWS)

    def test_view_sample_subset(self):
        md = self.md.filter(pl.col(COLUMN_SAMPLE_ID).is_in(["S1", "S3", "S5"]))
        v = View(f"{self.d}/{self.name}", self.tsv(md), None)

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
        feat = self.feat.filter(pl.col(COLUMN_GENOME_ID).is_in(["G1", "G5", "G6"]))

        v = View(f"{self.d}/{self.name}", self.tsv(self.md), self.tsv(feat))

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
        feat = pl.DataFrame(
            [["G1", 0, 1000], ["G5", 0, 1000]],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, str),
                (COLUMN_START, COLUMN_START_DTYPE),
                (COLUMN_STOP, COLUMN_STOP_DTYPE),
            ],
        )
        v = View(f"{self.d}/{self.name}", self.tsv(self.md), self.tsv(feat))

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
        feat = pl.DataFrame(
            [["G1", 1000, 2000], ["G5", 1000, 2000]],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, str),
                (COLUMN_START, COLUMN_START_DTYPE),
                (COLUMN_STOP, COLUMN_STOP_DTYPE),
            ],
        )

        with self.assertRaisesRegex(ValueError, "No positions left"):
            View(f"{self.d}/{self.name}", self.tsv(self.md), self.tsv(feat))

    def test_view_constrain_positions_bounds_simple(self):
        feat = pl.DataFrame(
            [["G1", 7, 9], ["G5", 0, 20]],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, str),
                (COLUMN_START, COLUMN_START_DTYPE),
                (COLUMN_STOP, COLUMN_STOP_DTYPE),
            ],
        )
        v = View(f"{self.d}/{self.name}", self.tsv(self.md), self.tsv(feat))

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
        feat = pl.DataFrame(
            [
                ["G1", 0, 100],
                ["G2", 40, 60],
                ["G3", 40, 60],
                ["G4", 90, 100],
                ["G5", 40, 60],
            ],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, str),
                (COLUMN_START, COLUMN_START_DTYPE),
                (COLUMN_STOP, COLUMN_STOP_DTYPE),
            ],
        )
        v = View(f"{self.d}/{self.name}", self.tsv(self.md), self.tsv(feat))

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

    def test_sample_presence_absence_no_regions(self):
        v = View(f"{self.d}/{self.name}", self.tsv(self.md), self.tsv(self.feat))
        with self.assertRaisesRegex(ValueError, r"^Cannot calculate"):
            v.sample_presence_absence()

    def test_sample_presence_absence_single_region(self):
        feat = pl.DataFrame(
            [
                ["G1", 0, 100],
                ["G2", 40, 60],
                ["G3", 40, 60],
                ["G4", 90, 100],
                ["G5", 40, 60],
            ],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, str),
                (COLUMN_START, COLUMN_START_DTYPE),
                (COLUMN_STOP, COLUMN_STOP_DTYPE),
            ],
        )
        v = View(f"{self.d}/{self.name}", self.tsv(self.md), self.tsv(feat))

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
        feat = pl.DataFrame(
            [
                ["G1", 0, 100],
                ["G2", 40, 60],
                ["G3", 40, 60],
                ["G3", 40, 60],
                ["G4", 90, 100],
                ["G5", 40, 60],
            ],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, str),
                (COLUMN_START, COLUMN_START_DTYPE),
                (COLUMN_STOP, COLUMN_STOP_DTYPE),
            ],
        )
        with self.assertRaisesRegex(ValueError, "Region IDs are not unique"):
            View(f"{self.d}/{self.name}", self.tsv(self.md), self.tsv(feat))

    def test_feature_names(self):
        names = pl.DataFrame(
            [["G1", "foo"], ["G2", "bar"]],
            orient="row",
            schema=[(COLUMN_GENOME_ID, str), (COLUMN_NAME, str)],
        )

        v = View(
            f"{self.d}/{self.name}",
            self.tsv(self.md),
            self.tsv(self.feat),
            self.tsv(names),
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
