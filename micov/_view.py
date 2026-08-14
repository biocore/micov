import os

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
        self.feature_names_source = feature_names

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

    def _read_tsv(self, path, rename, all_varchar=False):
        """Build a SELECT over a TSV, renaming its leading columns.

        micov's metadata files identify their key columns by *position*, not by
        name, so the first column -- and for feature names the second -- is
        renamed to the canonical name whatever the file happened to call it.

        Returns SQL rather than a relation so callers can compose it into a
        larger statement.
        """
        varchar = ", all_varchar=true" if all_varchar else ""
        source = f"read_csv('{path}', delim='\t', header=true{varchar})"
        columns = [row[0] for row in self.con.sql(f"DESCRIBE FROM {source}").fetchall()]
        # not strict: `rename` covers only the leading columns, and the file
        # carries however many more it likes
        selected = [
            f'"{old}" AS {new}' for old, new in zip(columns, rename, strict=False)
        ]
        selected += [f'"{column}"' for column in columns[len(rename) :]]
        return f"SELECT {', '.join(selected)} FROM {source}"

    def _feature_filters(self):
        """Load the feature constraints and decide which filter mode applies.

        Three modes: no constraint, genome-level, and sub-genome region. Only
        the last needs interval clipping, and it is selected by the presence of
        a `start`/`stop` pair in the feature file.
        """
        coverage = f"{self.dbbase}.coverage.parquet"

        if self.features_to_keep is None:
            self.con.sql(f"""CREATE TABLE feature_constraint AS
                             SELECT DISTINCT {COLUMN_GENOME_ID},
                                    NULL AS {COLUMN_START},
                                    NULL AS {COLUMN_STOP}
                             FROM '{coverage}'""")
            return

        query = self._read_tsv(self.features_to_keep, [COLUMN_GENOME_ID])
        columns = [row[0] for row in self.con.sql(f"DESCRIBE {query}").fetchall()]

        if COLUMN_START in columns:
            if COLUMN_STOP not in columns:
                raise KeyError(f"'{COLUMN_START}' found but missing '{COLUMN_STOP}'")
            self.constrain_positions = True
            # read_csv infers BIGINT for the interval bounds, but the rest of
            # the View works in UINTEGER and feature_metadata's dtypes are
            # visible to downstream consumers.
            query = (
                f"SELECT * EXCLUDE ({COLUMN_START}, {COLUMN_STOP}), "
                f"{COLUMN_START}::UINTEGER AS {COLUMN_START}, "
                f"{COLUMN_STOP}::UINTEGER AS {COLUMN_STOP} FROM ({query})"
            )
        elif COLUMN_STOP in columns:
            raise KeyError(f"'{COLUMN_STOP}' found but missing '{COLUMN_START}'")
        else:
            # the downstream SQL always names start/stop, so supply them as
            # typed NULLs. INTEGER matches what a bare NULL literal resolves to
            # in the `features_to_keep is None` branch above.
            query = (f"SELECT *, NULL::INTEGER AS {COLUMN_START}, "
                     f"NULL::INTEGER AS {COLUMN_STOP} FROM ({query})")

        self.con.sql(f"CREATE TABLE feature_constraint AS {query}")

        count = self.con.sql("SELECT COUNT(*) FROM feature_constraint").fetchone()[0]
        if count > 0:
            self.constrain_features = True

    def _init(self):
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
        metadata = self._read_tsv(
            self.sample_metadata, [COLUMN_SAMPLE_ID], all_varchar=True
        )
        self.con.sql(f"""CREATE TABLE metadata AS
                         SELECT md.*
                         FROM ({metadata}) md
                             SEMI JOIN '{coverage}' cov
                                 ON md.{COLUMN_SAMPLE_ID}=cov.{COLUMN_SAMPLE_ID}""")

        self._feature_filters()

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

            # clipping to the region bounds can leave overlapping intervals, so
            # re-compress per sample. We "wrap" a table so "positions" is a
            # consistent entity in the database.
            #
            # Gaps-and-islands: an interval opens a new island only when it
            # starts strictly beyond every stop seen so far in its partition.
            # `>` rather than `>=` is what merges *touching* intervals --
            # [400,500) and [500,505) become [400,505) -- which is micov's
            # documented behaviour and is pinned by test_cov.py.
            self.con.sql(f"""CREATE TABLE recompressed_positions AS
                WITH ordered AS (
                    SELECT {COLUMN_SAMPLE_ID}, {COLUMN_GENOME_ID},
                           {COLUMN_START}, {COLUMN_STOP},
                           MAX({COLUMN_STOP}) OVER (
                               PARTITION BY {COLUMN_SAMPLE_ID}, {COLUMN_GENOME_ID}
                               ORDER BY {COLUMN_START}, {COLUMN_STOP}
                               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                           ) AS prior_stop
                    FROM positions
                ),
                islands AS (
                    SELECT *,
                           SUM(CASE
                                   WHEN prior_stop IS NULL
                                        OR {COLUMN_START} > prior_stop
                                   THEN 1 ELSE 0
                               END) OVER (
                               PARTITION BY {COLUMN_SAMPLE_ID}, {COLUMN_GENOME_ID}
                               ORDER BY {COLUMN_START}, {COLUMN_STOP}
                               ROWS UNBOUNDED PRECEDING
                           ) AS island
                    FROM ordered
                )
                SELECT {COLUMN_GENOME_ID},
                       MIN({COLUMN_START})::UINTEGER AS {COLUMN_START},
                       MAX({COLUMN_STOP})::UINTEGER AS {COLUMN_STOP},
                       {COLUMN_SAMPLE_ID}
                FROM islands
                GROUP BY {COLUMN_SAMPLE_ID}, {COLUMN_GENOME_ID}, island""")

            empty = self.con.sql(
                "SELECT COUNT(*) FROM recompressed_positions"
            ).fetchone()[0]
            if empty == 0:
                msg = "No positions left after filtering."
                raise ValueError(msg)

            self.con.sql("""CREATE OR REPLACE VIEW positions AS
                            SELECT * FROM recompressed_positions""")

            # breadth against the *region* length rather than the genome length.
            # Every genome reaching this point came through the join against
            # feature_constraint above, so the inner join cannot drop one --
            # which is why there is no "unrepresented genome" check here.
            self.con.sql(f"""CREATE TABLE recomputed_coverage AS
                SELECT pos.{COLUMN_GENOME_ID},
                       SUM(pos.{COLUMN_STOP} - pos.{COLUMN_START})::UINTEGER
                           AS {COLUMN_COVERED},
                       fc.{COLUMN_LENGTH},
                       (SUM(pos.{COLUMN_STOP} - pos.{COLUMN_START})
                        / fc.{COLUMN_LENGTH}) * 100 AS {COLUMN_PERCENT_COVERED},
                       pos.{COLUMN_SAMPLE_ID}
                FROM recompressed_positions pos
                    JOIN (SELECT {COLUMN_GENOME_ID},
                                 ({COLUMN_STOP} - {COLUMN_START})::UINTEGER
                                     AS {COLUMN_LENGTH}
                          FROM feature_constraint) fc
                        USING ({COLUMN_GENOME_ID})
                GROUP BY pos.{COLUMN_SAMPLE_ID}, pos.{COLUMN_GENOME_ID},
                         fc.{COLUMN_LENGTH}""")
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
                                  FROM coverage""").fetchall()
        if any(column_type == "BIGINT" for _, column_type, *_ in schema):
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
                                  FROM positions""").fetchall()
        if any(column_type == "BIGINT" for _, column_type, *_ in schema):
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
        if self.feature_names_source is None:
            return self.con.sql(f"""
                SELECT DISTINCT {COLUMN_GENOME_ID}, {COLUMN_GENOME_ID} AS {COLUMN_NAME}
                FROM feature_metadata
            """)
        else:
            names = self._read_tsv(
                self.feature_names_source, [COLUMN_GENOME_ID, COLUMN_NAME]
            )
            # A name that looks like a lineage keeps only its last element.
            # '^.*; ' is greedy, so it consumes through the *final* delimiter
            # and leaves a plain name untouched -- both cases in one pass.
            names = (
                f"SELECT {COLUMN_GENOME_ID}, "
                f"regexp_replace(regexp_replace({COLUMN_NAME}, '^.*; ', ''), "
                r"'[ \[\]]', '_', 'g')"
                f" AS {COLUMN_NAME} FROM ({names})"
            )
            return self.con.sql(f"""
                SELECT DISTINCT
                    fm.{COLUMN_GENOME_ID},
                    COALESCE(fn.{COLUMN_NAME}, fm.{COLUMN_GENOME_ID}) AS {COLUMN_NAME}
                FROM feature_metadata fm
                LEFT JOIN ({names}) fn
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
            -- One row per sample, one column per region. A sample is present in
            -- a region if it has coverage there; absent if it has coverage of
            -- the genome but none within the region; and not applicable if it
            -- has no coverage of that genome at all -- the last of which shows
            -- up as a sample/region pair that never reached has_region, so the
            -- PIVOT leaves a NULL for COALESCE to fill.

            -- A sample can be both present and absent in one region, via two
            -- intervals of which only one overlaps. BOOL_OR resolves that to
            -- present, matching the precedence the previous implementation got
            -- from coalescing the present column first.

            -- n.b. we have to materialize as pivot elements cannot be used in views
            -- without explicilty naming the columns. Since we do not know the regions
            -- in advance, we cannot readily define the columns. As far as I know,
            -- the only way would be a clunky dynamic SQL query.
            CREATE OR REPLACE TABLE sample_presence_absence AS (
                SELECT
                    {COLUMN_SAMPLE_ID},
                    COALESCE(COLUMNS(* EXCLUDE {COLUMN_SAMPLE_ID}),
                             '{NOT_APPLICABLE}')
                FROM (
                    PIVOT (
                        SELECT
                            {COLUMN_SAMPLE_ID},
                            {COLUMN_REGION_ID},
                            CASE
                                WHEN BOOL_OR(painfo = '{PRESENT}')
                                THEN '{PRESENT}'
                                ELSE '{ABSENT}'
                            END AS painfo
                        FROM has_region
                        GROUP BY {COLUMN_SAMPLE_ID}, {COLUMN_REGION_ID}
                    ) ON {COLUMN_REGION_ID} USING FIRST(painfo)
                )
            );
            """)

        # kept in its own call: PIVOT resolves its columns at bind time, and
        # batching the DROP alongside it makes has_region unresolvable
        self.con.sql("DROP VIEW has_region")

        return self.con.sql("SELECT * FROM sample_presence_absence")
