import os

import duckdb
import polars as pl

from micov import MEMORY, THREADS
from micov._constants import (
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

    def __init__(self, dbbase, sample_metadata, features_to_keep):
        self.dbbase = dbbase
        self.sample_metadata = sample_metadata
        self.features_to_keep = features_to_keep

        self.constrain_positions = False
        self.constrain_features = False

        self.con = duckdb.connect(
            ":memory:", config={"threads": THREADS, "memory_limit": f"{MEMORY}gb"}
        )
        self._init()

    def close(self):
        self.con.close()

    def __del__(self):
        self.close()

    def _feature_filters(self):
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
            raise IOError(f"'{coverage}' not found")

        if not os.path.exists(positions):
            raise IOError(f"'{positions}' not found")

        feat_df = self.features_to_keep
        md_df = self.sample_metadata
        self.con.sql("CREATE TABLE feature_constraint AS FROM feat_df")
        self.con.sql(f"""CREATE TABLE metadata AS
                         SELECT md.*
                         FROM md_df md
                             SEMI JOIN '{coverage}' cov
                                 ON md.{COLUMN_SAMPLE_ID}=cov.{COLUMN_SAMPLE_ID}""")

        if self.constrain_positions and self.constrain_features:
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
                                         AND pos.{COLUMN_START} < fc.{COLUMN_STOP}
                                         AND pos.{COLUMN_STOP} > fc.{COLUMN_START}
                                 JOIN metadata md
                                     ON pos.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")

            # pull the new position data, compress, and reconstruct the view
            # we "wrap" a table so "positions" is a consistent entity in the database
            positions = self.con.sql(f"""SELECT * FROM positions
                                         ORDER BY {COLUMN_SAMPLE_ID},
                                                  {COLUMN_GENOME_ID},
                                                  {COLUMN_START}""").pl()

            if len(positions) == 0:
                raise ValueError("No positions left after filtering.")

            positions = compress_per_sample(positions)
            self.con.sql("CREATE TABLE recompressed_positions AS FROM positions")
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
            coverage = coverage_percent_per_sample(positions, lengths)
            self.con.sql("CREATE TABLE recomputed_coverage AS FROM coverage")
            self.con.sql("""CREATE OR REPLACE VIEW coverage AS
                            SELECT * FROM recomputed_coverage""")

        elif self.constrain_features:
            # limit the samples considered
            # limit the set of features considered
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
        else:
            # limit the samples considered
            self.con.sql(f"""CREATE VIEW coverage AS
                             SELECT cov.*
                             FROM '{coverage}'
                                 JOIN metadata md
                                     ON cov.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")
            self.con.sql(f"""CREATE VIEW positions AS
                             SELECT pos.*
                             FROM '{positions}'
                                 JOIN metadata md
                                     ON pos.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")

    def metadata(self):
        return self.con.sql("SELECT * FROM metadata")

    def coverages(self):
        return self.con.sql("SELECT * from coverage")

    def positions(self):
        return self.con.sql("SELECT * from positions")
