import os

from micov._constants import (
    COLUMN_COVERED,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_NAME,
    COLUMN_PERCENT_COVERED,
    COLUMN_REGION_ID,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)
from micov._miint import connection
from micov._utils import sql_string


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

        self.con = connection(memory=memory, threads=threads)
        self._init()

    def close(self):
        self.con.close()

    def __del__(self):
        # opening the connection can now fail -- it requires the miint
        # extension -- which leaves `con` unset. A finaliser that raises buries
        # micov's authored message under "Exception ignored in __del__".
        if getattr(self, "con", None) is not None:
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
        source = f"read_csv({sql_string(path)}, delim='\t', header=true{varchar})"
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
                             FROM {sql_string(coverage)}""")
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

        if self.constrain_positions:
            # A region is half-open [start, stop), so stop has to exceed start.
            # Checked here, with the rest of the feature-file validation, and
            # in the user's terms: a transposed pair of columns used to surface
            # as `Out of Range Error: Overflow in subtraction of UINT32
            # (40 - 60)` from `feature_metadata` below, and a zero-width region
            # as `No positions left after filtering.` -- neither of which names
            # the offending row. miint rejects both too, but phrases its error
            # as advice about its own SQL API, which is not what the user wrote.
            malformed = self.con.sql(f"""SELECT {COLUMN_GENOME_ID},
                                                {COLUMN_START}, {COLUMN_STOP}
                                         FROM feature_constraint
                                         WHERE {COLUMN_STOP} <= {COLUMN_START}
                                         ORDER BY ALL
                                         LIMIT 1""").fetchall()
            if malformed:
                genome_id, start, stop = malformed[0]
                raise ValueError(
                    f"Region for '{genome_id}' is not a valid interval: "
                    f"[{start}, {stop}). Regions are half-open, so 'stop' must "
                    "be greater than 'start'."
                )

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
                             SEMI JOIN {sql_string(coverage)} cov
                                 ON md.{COLUMN_SAMPLE_ID}=cov.{COLUMN_SAMPLE_ID}""")

        self._feature_filters()

        # views are "free". Let's establish a common reference point for unmodified
        # position data'
        self.con.sql(
            f"CREATE VIEW unconstrained_positions AS FROM {sql_string(positions)}"
        )

        if self.constrain_positions:
            # `region_coverage` and `region_presence` are table macros built on
            # `query_table()`, so each argument has to name a table or a view
            # -- a subquery is a Binder Error. Hence the two views below rather
            # than inlining either of them at the call site.
            self.con.sql(f"""CREATE VIEW selected_positions AS
                             SELECT pos.*
                             FROM unconstrained_positions pos
                                 JOIN metadata md
                                     ON pos.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")

            # `region_id` is spelled out here rather than read from
            # feature_metadata, which cannot exist yet -- it is filtered by the
            # coverage this view is about to produce. The two use the same
            # expression, so the ids agree.
            self.con.sql(f"""CREATE VIEW regions AS
                             SELECT {COLUMN_GENOME_ID},
                                    {COLUMN_START} AS region_start,
                                    {COLUMN_STOP} AS region_stop,
                                    CONCAT_WS('_',
                                              {COLUMN_GENOME_ID},
                                              {COLUMN_START},
                                              {COLUMN_STOP}) AS {COLUMN_REGION_ID}
                             FROM feature_constraint""")

            # Clip to the region bounds, so an interval wider than the region
            # contributes only the part inside it.
            #
            # `pos.start < fc.stop`, not `<=`: the region is half-open, so an
            # interval beginning exactly at `stop` falls outside it. The `<=`
            # this replaces admitted that interval, clipped it to a zero-width
            # [stop, stop) worth no bases, and still reported the sample
            # present in the region. `<` is also the predicate miint's macros
            # use, so micov's positions and its coverage agree on what overlaps.
            self.con.sql(f"""CREATE VIEW clipped_positions AS
                             SELECT pos.{COLUMN_SAMPLE_ID}, pos.{COLUMN_GENOME_ID},
                                    GREATEST(pos.{COLUMN_START},
                                             fc.{COLUMN_START}) AS {COLUMN_START},
                                    LEAST(pos.{COLUMN_STOP},
                                          fc.{COLUMN_STOP}) AS {COLUMN_STOP}
                             FROM selected_positions pos
                                 JOIN feature_constraint fc
                                     ON pos.{COLUMN_GENOME_ID}=fc.{COLUMN_GENOME_ID}
                                         AND pos.{COLUMN_START} < fc.{COLUMN_STOP}
                                         AND pos.{COLUMN_STOP} > fc.{COLUMN_START}""")

            # Clipping can leave overlapping intervals, so re-compress per
            # sample. `compress_intervals` is the same primitive
            # `_io.compress_alignments` uses -- which is the point of doing it
            # this way: micov has one interval merge, not a second hand-written
            # one that has to be kept agreeing with the first. Touching
            # intervals collapse, as test_cov.py pins.
            #
            # We "wrap" a table so "positions" is a consistent entity in the
            # database.
            self.con.sql(f"""CREATE TABLE recompressed_positions AS
                SELECT {COLUMN_GENOME_ID},
                       interval.start::UINTEGER AS {COLUMN_START},
                       interval.stop::UINTEGER AS {COLUMN_STOP},
                       {COLUMN_SAMPLE_ID}
                FROM (SELECT {COLUMN_SAMPLE_ID}, {COLUMN_GENOME_ID},
                             UNNEST(compress_intervals({COLUMN_START},
                                                       {COLUMN_STOP}))
                                 AS interval
                      FROM clipped_positions
                      GROUP BY {COLUMN_SAMPLE_ID}, {COLUMN_GENOME_ID})""")

            empty = self.con.sql(
                "SELECT COUNT(*) FROM recompressed_positions"
            ).fetchone()[0]
            if empty == 0:
                msg = "No positions left after filtering."
                raise ValueError(msg)

            self.con.sql("""CREATE VIEW positions AS
                            SELECT * FROM recompressed_positions""")

            # breadth against the *region* length rather than the genome
            # length. `region_coverage` clips and merges internally, so it
            # takes the unclipped positions rather than the view above.
            #
            # `proportion_covered` is 0..1 where micov reports 0..100, and the
            # `* 100` is applied after the division rather than folded into it.
            # `covered * 100 / length` is a different double -- test_view.py
            # pins the difference with a literal.
            self.con.sql(f"""CREATE TABLE recomputed_coverage AS
                SELECT {COLUMN_GENOME_ID},
                       covered::UINTEGER AS {COLUMN_COVERED},
                       region_length::UINTEGER AS {COLUMN_LENGTH},
                       proportion_covered * 100 AS {COLUMN_PERCENT_COVERED},
                       {COLUMN_SAMPLE_ID}
                FROM region_coverage(selected_positions, regions)""")
            self.con.sql("""CREATE VIEW coverage AS
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
                             FROM {sql_string(coverage)} cov
                                 JOIN feature_constraint fc
                                     ON cov.{COLUMN_GENOME_ID}=fc.{COLUMN_GENOME_ID}
                                 JOIN metadata md
                                     ON cov.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")
            self.con.sql(f"""CREATE VIEW positions AS
                             SELECT pos.*
                             FROM {sql_string(positions)} pos
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
                             FROM {sql_string(coverage)} cov
                                 JOIN metadata md
                                     ON cov.{COLUMN_SAMPLE_ID}=md.{COLUMN_SAMPLE_ID}""")
            self.con.sql(f"""CREATE VIEW positions AS
                             SELECT pos.*
                             FROM {sql_string(positions)} pos
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

        # feature_metadata, not the `regions` view: this is the set of regions
        # that survived the coverage filter, and so the set the wide output has
        # a column for. `regions` would add a column for every requested region
        # no sample covers.
        self.con.sql(f"""CREATE OR REPLACE VIEW presence_regions AS
                         SELECT {COLUMN_GENOME_ID},
                                {COLUMN_START} AS region_start,
                                {COLUMN_STOP} AS region_stop,
                                {COLUMN_REGION_ID}
                         FROM feature_metadata""")

        self.con.sql(f"""
            -- One row per sample, one column per region.
            --
            -- `region_presence` classifies every sample against every region:
            -- present if it has coverage inside the region, absent if it
            -- covers the genome but nothing within the region, and not
            -- applicable if it has no coverage of that genome at all. It emits
            -- those three strings verbatim, so `_constants.py`'s values are
            -- what lands in the file rather than a translation of them.
            --
            -- A sample can be both present and absent in one region, via two
            -- intervals of which only one overlaps; the macro resolves that to
            -- present, which is the precedence micov has always had.
            --
            -- Every sample/region pair gets a row, so there is nothing for the
            -- PIVOT to leave NULL and no COALESCE here. The previous
            -- implementation needed one: it derived presence from a join that
            -- simply had no row for a sample lacking the genome, and filled
            -- the resulting hole with 'not applicable' afterwards.
            --
            -- n.b. we have to materialize as pivot elements cannot be used in
            -- views without explicilty naming the columns. Since we do not
            -- know the regions in advance, we cannot readily define the
            -- columns. As far as I know, the only way would be a clunky
            -- dynamic SQL query.
            CREATE OR REPLACE TABLE sample_presence_absence AS (
                PIVOT (
                    SELECT {COLUMN_SAMPLE_ID}, {COLUMN_REGION_ID}, state
                    FROM region_presence(selected_positions,
                                         presence_regions,
                                         metadata)
                ) ON {COLUMN_REGION_ID} USING FIRST(state)
            );
            """)

        return self.con.sql("SELECT * FROM sample_presence_absence")
