import os

import duckdb
import polars as pl

from micov._constants import (
    COLUMN_COVERED,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_LENGTH_DTYPE,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)
from micov._cov import compress_per_sample, coverage_percent_per_sample


class View:
    """View subsets of coverage data."""

    def __init__(
        self, dbbase, sample_metadata, features_to_keep, threads=1, memory="8gb"
    ):
        self.dbbase = dbbase
        self.sample_metadata = sample_metadata
        self.features_to_keep = features_to_keep

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
                             FROM '{positions}' pos
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
                             SELECT *, {COLUMN_STOP} - {COLUMN_START} AS {COLUMN_LENGTH}
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
                                    SELECT f.{COLUMN_GENOME_ID},
                                        0::UINTEGER AS {COLUMN_START},
                                        g.{COLUMN_LENGTH} AS {COLUMN_STOP},
                                        g.{COLUMN_LENGTH}
                                    FROM feature_constraint f
                                        JOIN genome_lengths g
                                            ON f.{COLUMN_GENOME_ID}=g.{COLUMN_GENOME_ID}
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
                                    SELECT f.{COLUMN_GENOME_ID},
                                        0::UINTEGER AS {COLUMN_START},
                                        g.{COLUMN_LENGTH} AS {COLUMN_STOP},
                                        g.{COLUMN_LENGTH}
                                    FROM feature_constraint f
                                        JOIN genome_lengths g
                                            ON f.{COLUMN_GENOME_ID}=g.{COLUMN_GENOME_ID}
                    """)

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

    def target_names(self, target_names):
        # TODO: integrate into feature_metadata()
        if target_names is not None:
            target_names = dict(
                pl.scan_csv(
                    target_names,
                    separator="\t",
                    new_columns=["feature-id", "lineage"],
                    has_header=False,
                )
                .with_columns(
                    pl.col("lineage")
                    .str.split(";")
                    .list.get(-1)
                    .str.replace_all(r" |\[|\]", "_")
                    .alias("species")
                )
                .select("feature-id", "species")
                .collect()
                .iter_rows()
            )
        else:
            sql = "SELECT DISTINCT genome_id FROM coverage"
            target_names = {k[0]: k[0] for k in self.con.sql(sql).fetchall()}
        return target_names
