import os

import duckdb
import polars as pl

from micov._constants import (
    ABSENT,
    COLUMN_COVERED,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_LENGTH_DTYPE,
    COLUMN_NAME,
    COLUMN_REGION_ID,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
    NOT_APPLICABLE,
    PRESENT,
)
from micov._cov import compress_per_sample, coverage_percent_per_sample


class View:
    """View subsets of coverage data."""

    def __init__(
        self,
        dbbase,
        sample_metadata,
        features_to_keep,
        feature_names=None,
        threads=1,
        memory="8gb",
    ):
        self.dbbase = dbbase
        self.sample_metadata = sample_metadata
        self.features_to_keep = features_to_keep
        self.feature_names_df = feature_names

        self.constrain_positions = False
        self.constrain_features = False

        self.con = duckdb.connect(
            ":memory:", config={"threads": threads, "memory_limit": f"{memory}"}
        )
        self._init()

    def close(self):
        self.con.close()

    def __del__(self):
        self.close()

    def _feature_filters(self):
        if self.features_to_keep is None:
            return

        if COLUMN_START in self.features_to_keep.columns:
            if COLUMN_STOP not in self.features_to_keep.columns:
                raise KeyError(f"'{COLUMN_START}' found but missing '{COLUMN_STOP}'")
            self.constrain_positions = True
        elif COLUMN_STOP in self.features_to_keep.columns:
            if COLUMN_START not in self.features_to_keep.columns:
                raise KeyError(f"'{COLUMN_STOP}' found but missing '{COLUMN_START}'")
        else:
            self.features_to_keep = self.features_to_keep.with_columns(
                pl.lit(None).alias(COLUMN_START), pl.lit(None).alias(COLUMN_STOP)
            )

        if len(self.features_to_keep) > 0:
            self.constrain_features = True

    def _init(self):
        self._feature_filters()
        self._load_db()

    def _load_db(self):
        coverage = f"{self.dbbase}.coverage.parquet"
        positions = f"{self.dbbase}.covered_positions.parquet"

        if not os.path.exists(coverage):
            raise OSError(f"'{coverage}' not found")

        if not os.path.exists(positions):
            raise OSError(f"'{positions}' not found")

        # constrain the metadata before any feature filtering as the unfocused
        # monte carlo curve assumes access to _any_ sample with _any_ coverage
        md_df = self.sample_metadata  # noqa: F841
        self.con.sql(f"""CREATE TABLE metadata AS
                         SELECT md.*
                         FROM md_df md
                             SEMI JOIN '{coverage}' cov
                                 ON md.{COLUMN_SAMPLE_ID}=cov.{COLUMN_SAMPLE_ID}""")

        feat_df = self.features_to_keep
        if feat_df is None:
            self.con.sql(f"""CREATE TABLE feature_constraint AS
                             SELECT DISTINCT {COLUMN_GENOME_ID},
                                    NULL AS {COLUMN_START},
                                    NULL AS {COLUMN_STOP}
                             FROM '{coverage}'""")
        else:
            self.con.sql("CREATE TABLE feature_constraint AS FROM feat_df")

        # views are "free". Let's establish a common reference point for unmodified
        # position data'
        self.con.sql(f"CREATE VIEW unconstrained_positions AS FROM '{positions}'")

        if self.constrain_positions:
            # limit the samples considered
            # limit the set of features considered
            # limit the intervals considered
            # force the min/max of the intervals to be of the defined bounds
            # to cover the case where the requested interval range is contained
            # within a wider interal.
            self.con.sql(f"""CREATE VIEW positions AS
                             SELECT pos.* EXCLUDE ({COLUMN_START}, {COLUMN_STOP}),
                                    LEAST(pos.{COLUMN_STOP},
                                          fc.{COLUMN_STOP}) AS {COLUMN_STOP},
                                    GREATEST(pos.{COLUMN_START},
                                             fc.{COLUMN_START}) AS {COLUMN_START}
                             FROM unconstrained_positions pos
                                 JOIN feature_constraint fc
                                     ON pos.{COLUMN_GENOME_ID}=fc.{COLUMN_GENOME_ID}
                                         AND pos.{COLUMN_START} <= fc.{COLUMN_STOP}
                                         AND pos.{COLUMN_STOP} > fc.{COLUMN_START}
                                 JOIN metadata md
                                     ON pos.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")

            # pull the new position data, compress, and reconstruct the view
            # we "wrap" a table so "positions" is a consistent entity in the database
            # TODO: replace with duckdb native per sample compression
            #   Do we stream to parquet? this could be large
            positions_df = self.con.sql(f"""SELECT * FROM positions
                                            ORDER BY {COLUMN_SAMPLE_ID},
                                                     {COLUMN_GENOME_ID},
                                                     {COLUMN_START}""").pl()

            if len(positions_df) == 0:
                msg = "No positions left after filtering."
                raise ValueError(msg)

            positions_df = compress_per_sample(positions_df)
            self.con.sql("CREATE TABLE recompressed_positions AS FROM positions_df")
            self.con.sql("""CREATE OR REPLACE VIEW positions AS
                            SELECT * FROM recompressed_positions""")

            # obtain the length of the constrained regions for computing coverage
            # percent
            diff = pl.col(COLUMN_STOP) - pl.col(COLUMN_START)
            lengths = (
                self.features_to_keep.lazy()
                .with_columns(diff.cast(COLUMN_LENGTH_DTYPE).alias(COLUMN_LENGTH))
                .drop([COLUMN_START, COLUMN_STOP])
                .collect()
            )

            coverage_df = coverage_percent_per_sample(positions_df, lengths)  # noqa: F841
            self.con.sql("CREATE TABLE recomputed_coverage AS FROM coverage_df")
            self.con.sql("""CREATE OR REPLACE VIEW coverage AS
                            SELECT * FROM recomputed_coverage""")

            self.con.sql(f"""CREATE TABLE feature_metadata AS
                             SELECT *,
                                {COLUMN_STOP} - {COLUMN_START} AS {COLUMN_LENGTH},
                                CONCAT_WS('_',
                                          {COLUMN_GENOME_ID},
                                          {COLUMN_START},
                                          {COLUMN_STOP}) AS {COLUMN_REGION_ID}
                             FROM feature_constraint fc
                                 SEMI JOIN coverage cov USING ({COLUMN_GENOME_ID})""")

        elif self.constrain_features:
            # limit the samples considered
            # limit the set of features considered
            # express start/stop of the genomes as the full genome
            self.con.sql(f"""CREATE VIEW coverage AS
                             SELECT cov.*
                             FROM '{coverage}' cov
                                 JOIN feature_constraint fc
                                     ON cov.{COLUMN_GENOME_ID}=fc.{COLUMN_GENOME_ID}
                                 JOIN metadata md
                                     ON cov.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")
            self.con.sql(f"""CREATE VIEW positions AS
                             SELECT pos.*
                             FROM '{positions}' pos
                                 JOIN feature_constraint fc
                                     ON pos.{COLUMN_GENOME_ID}=fc.{COLUMN_GENOME_ID}
                                 JOIN metadata md
                                     ON pos.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")
            self.con.sql(f"""CREATE VIEW genome_lengths AS
                             SELECT {COLUMN_GENOME_ID},
                                 FIRST({COLUMN_LENGTH}) AS {COLUMN_LENGTH}
                             FROM coverage
                             GROUP BY {COLUMN_GENOME_ID};

                             CREATE TABLE feature_metadata AS
                             SELECT fc.{COLUMN_GENOME_ID},
                                 0::UINTEGER AS {COLUMN_START},
                                 gl.{COLUMN_LENGTH} AS {COLUMN_STOP},
                                 gl.{COLUMN_LENGTH},
                                 CONCAT_WS('_',
                                           fc.{COLUMN_GENOME_ID},
                                           0,
                                           {COLUMN_LENGTH}) AS {COLUMN_REGION_ID}
                             FROM feature_constraint fc
                                 JOIN genome_lengths gl
                                     ON fc.{COLUMN_GENOME_ID}=gl.{COLUMN_GENOME_ID}
                    """)
        else:
            # limit the samples considered
            self.con.sql(f"""CREATE VIEW coverage AS
                             SELECT cov.*
                             FROM '{coverage}' cov
                                 JOIN metadata md
                                     ON cov.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")
            self.con.sql(f"""CREATE VIEW positions AS
                             SELECT pos.*
                             FROM '{positions}' pos
                                 JOIN metadata md
                                     ON pos.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")

            # TODO: the query below is identifical to the one in the
            #   `elif self.constrain_features` branch. it probably should
            #   be decomposed, however that suggests considering a management
            #   strategy for the other sql queries.
            #
            # use the existing length data from coverage to set the start/stop
            # positions in feature_metadata
            self.con.sql(f"""CREATE VIEW genome_lengths AS
                                    SELECT {COLUMN_GENOME_ID},
                                        FIRST({COLUMN_LENGTH}) AS {COLUMN_LENGTH}
                                    FROM coverage
                                    GROUP BY {COLUMN_GENOME_ID};
                                CREATE TABLE feature_metadata AS
                                    SELECT fc.{COLUMN_GENOME_ID},
                                        0::UINTEGER AS {COLUMN_START},
                                        gl.{COLUMN_LENGTH} AS {COLUMN_STOP},
                                        gl.{COLUMN_LENGTH},
                                        CONCAT_WS('_',
                                                  fc.{COLUMN_GENOME_ID},
                                                  0,
                                                  {COLUMN_LENGTH}) AS {COLUMN_REGION_ID}
                                 FROM feature_constraint fc
                                     JOIN genome_lengths gl
                                         ON fc.{COLUMN_GENOME_ID}=gl.{COLUMN_GENOME_ID}
                    """)
        self._integrity_checks()

    def _integrity_checks(self):
        region_id_uniqueness = self.con.sql(f"""
            SELECT
                CASE
                    WHEN COUNT(DISTINCT {COLUMN_REGION_ID}) == COUNT({COLUMN_REGION_ID})
                    THEN 'OK'
                    ELSE 'FAIL'
                END AS region_id_uniqueness
            FROM feature_metadata
        """).fetchone()[0]
        if region_id_uniqueness == "FAIL":
            raise ValueError("Region IDs are not unique.")

    def metadata(self):
        return self.con.sql("SELECT * FROM metadata")

    def feature_metadata(self):
        return self.con.sql("SELECT * FROM feature_metadata")

    def coverages(self):
        schema = self.con.sql(f"""DESCRIBE SELECT {COLUMN_COVERED}, {COLUMN_LENGTH}
                                  FROM coverage""").pl()
        schema = schema.filter(pl.col("column_type") == "BIGINT")
        if len(schema) > 0:
            # old files used int64
            # UINTEGER is UInt32
            # TODO: guarentee we are consistent with _constansts.py
            return self.con.sql(f"""SELECT
                                        * EXCLUDE ({COLUMN_COVERED}, {COLUMN_LENGTH}),
                                        {COLUMN_COVERED}::UINTEGER AS {COLUMN_COVERED},
                                        {COLUMN_LENGTH}::UINTEGER AS {COLUMN_LENGTH}
                                    FROM coverage""")
        else:
            return self.con.sql("SELECT * from coverage")

    def positions(self):
        schema = self.con.sql(f"""DESCRIBE SELECT {COLUMN_START}, {COLUMN_STOP}
                                  FROM positions""").pl()
        schema = schema.filter(pl.col("column_type") == "BIGINT")
        if len(schema) > 0:
            # old files used int64
            # UINTEGER is UInt32
            # TODO: guarentee we are consistent with _constansts.py
            return self.con.sql(f"""SELECT
                                        * EXCLUDE ({COLUMN_START}, {COLUMN_STOP}),
                                        {COLUMN_START}::UINTEGER AS {COLUMN_START},
                                        {COLUMN_STOP}::UINTEGER AS {COLUMN_STOP}
                                    FROM positions""")
        else:
            return self.con.sql("SELECT * from positions")

    def feature_names(self):
        if self.feature_names_df is None:
            return self.con.sql(f"""
                SELECT DISTINCT {COLUMN_GENOME_ID}, {COLUMN_GENOME_ID} AS {COLUMN_NAME}
                FROM feature_metadata
            """)
        else:
            feature_names = self.feature_names_df  # noqa
            return self.con.sql(f"""
                SELECT DISTINCT
                    fm.{COLUMN_GENOME_ID},
                    COALESCE(fn.{COLUMN_NAME}, fm.{COLUMN_GENOME_ID}) AS {COLUMN_NAME}
                FROM feature_metadata fm
                LEFT JOIN feature_names fn
                    USING ({COLUMN_GENOME_ID})""")

    def sample_presence_absence(self):
        if not self.constrain_positions:
            raise ValueError("Cannot calculate presence/absence without positions.")

        self.con.sql(f"""
            -- define a view which describes whether a sample is present in a particular
            -- region.
            CREATE OR REPLACE VIEW has_region AS (
                SELECT
                    pos.{COLUMN_SAMPLE_ID},
                    fm.{COLUMN_REGION_ID},
                    CASE
                        WHEN pos.{COLUMN_START} <= fm.{COLUMN_STOP}
                            AND pos.{COLUMN_STOP} > fm.{COLUMN_START}
                        THEN '{PRESENT}'
                        ELSE '{ABSENT}'
                    END AS painfo
                FROM unconstrained_positions pos
                    LEFT JOIN feature_metadata fm
                        ON pos.{COLUMN_GENOME_ID}=fm.{COLUMN_GENOME_ID}
            );
        """)

        self.con.sql(f"""
            -- extract the samples which are "present" and "absent" and the
            -- regions they are present -- in. Note that a sample is present in a
            -- region if it has coverage in that -- region. It is considered absent
            -- if it nas nonzero coverage for the genome -- AND lacks coverage
            -- within the focus region.

            -- n.b. we have to materialize as pivot elements cannot be used in views
            -- without explicilty naming the columns. Since we do not know the regions
            -- in advance, we cannot readily define the columns. As far as I know,
            -- the only way would be a clunky dynamic SQL query.
            CREATE OR REPLACE TABLE present AS (
                SELECT
                    {COLUMN_SAMPLE_ID},
                    CASE
                        WHEN COLUMNS(* EXCLUDE {COLUMN_SAMPLE_ID}) > 0
                        THEN '{PRESENT}'
                        ELSE NULL
                    END
                FROM (PIVOT (SELECT * EXCLUDE (painfo)
                             FROM has_region
                             WHERE painfo='{PRESENT}')
                      ON {COLUMN_REGION_ID})
            );
            CREATE OR REPLACE TABLE absent AS (
                SELECT
                    {COLUMN_SAMPLE_ID},
                    CASE
                        WHEN COLUMNS(* EXCLUDE {COLUMN_SAMPLE_ID}) > 0
                        THEN '{ABSENT}'
                        ELSE NULL
                    END
                FROM (PIVOT (SELECT * EXCLUDE (painfo)
                             FROM has_region
                             WHERE painfo='{ABSENT}')
                      ON {COLUMN_REGION_ID})
            );
            """)

        # Joining, coalescing, and filling nulls as far as I could tell requires
        # clunky dynamic SQL in order to determine the set of columns to
        # coalesce. It's easy to do within Polars.'
        present = self.con.sql("SELECT * FROM present").pl()
        absent = self.con.sql("SELECT * FROM absent").pl()

        # columns in common are ones where there is a mix of samples which are present
        # and absent
        common = (set(present.columns) & set(absent.columns)) - {
            COLUMN_SAMPLE_ID,
        }

        # when there are duplicates, the left column receives the original name
        # and the right column is suffixed. The default suffix is "_right".
        exprs = [pl.coalesce([c, c + "_right"]).alias(c) for c in common]

        # after we coalesce, the right columns are unnecessary
        drops = [c + "_right" for c in common]

        joined = (  # noqa
            present.lazy()
            .join(absent.lazy(), on={COLUMN_SAMPLE_ID}, how="full", coalesce=True)
            .with_columns(exprs)
            .drop(drops)
            .fill_null(pl.lit(NOT_APPLICABLE))
            .collect()
        )

        # clean up, and createa an object which can be pulled like the other access
        # methods of this class.
        self.con.sql("""
            DROP VIEW has_region;
            DROP TABLE present;
            DROP TABLE absent;
            CREATE OR REPLACE TABLE sample_presence_absence AS FROM joined;
            """)

        return self.con.sql("SELECT * FROM sample_presence_absence")
