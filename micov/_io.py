import bz2
import lzma
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager

from ._constants import (
    COLUMN_COVERED,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_NAME,
    COLUMN_PERCENT_COVERED,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)
from ._utils import logger, sql_string

#: What the first column of a feature file (`--features-to-keep`,
#: `--target-names`) must be called. Region files already had their `start`
#: and `stop` matched by name; this makes the id column consistent with them.
FEATURE_ID_COLUMNS = (COLUMN_GENOME_ID,)

#: What the first column of `--sample-metadata` may be called: micov's own
#: name, and `sample_name`, which Qiita exports and `example/` use.
SAMPLE_ID_COLUMNS = (COLUMN_SAMPLE_ID, "sample_name")


def read_tsv_with_header(con, path, rename, first_column, all_varchar=False):
    """Build a SELECT over a TSV, renaming its leading columns.

    The leading columns are renamed to micov's canonical names -- for
    feature names the second column too -- but the file must have a
    header, and its first column must be one of `first_column`.

    That requirement is the fix for a silent loss. A headerless file, such
    as a taxonomy `lineages.txt`, had its first *row* read as column names,
    so that genome or sample disappeared from every output with no error.
    Insisting on the name is what makes a missing header detectable at
    all: a data row's first field is not `genome_id`.

    Returns SQL rather than a relation so callers can compose it into a
    larger statement.
    """
    varchar = ", all_varchar=true" if all_varchar else ""
    source = f"read_csv({sql_string(path)}, delim='\t', header=true{varchar})"
    columns = [row[0] for row in con.sql(f"DESCRIBE FROM {source}").fetchall()]
    if columns[0] not in first_column:
        expected = " or ".join(repr(name) for name in first_column)
        raise ValueError(
            f"'{path}' must begin with a header line whose first column is "
            f"named {expected}, but its first column is {columns[0]!r}. If "
            "the file has no header, add one: micov would otherwise read "
            "the first row as column names and silently drop it."
        )
    # not strict: `rename` covers only the leading columns, and the file
    # carries however many more it likes
    selected = [
        f'"{old}" AS {new}' for old, new in zip(columns, rename, strict=False)
    ]
    selected += [f'"{column}"' for column in columns[len(rename) :]]
    return f"SELECT {', '.join(selected)} FROM {source}"


def target_names_query(con, path):
    """Build a SELECT of `genome_id` and the name to give its plot files.

    A name that looks like a lineage keeps only its last element, and
    spaces and square brackets become underscores, since the name goes into
    file names.
    """
    names = read_tsv_with_header(
        con, path, [COLUMN_GENOME_ID, COLUMN_NAME], FEATURE_ID_COLUMNS
    )
    # '^.*; ' is greedy, so it consumes through the *final* delimiter and
    # leaves a plain name untouched -- both cases in one pass.
    return (
        f"SELECT {COLUMN_GENOME_ID}, "
        f"regexp_replace(regexp_replace({COLUMN_NAME}, '^.*; ', ''), "
        r"'[ \[\]]', '_', 'g')"
        f" AS {COLUMN_NAME} FROM ({names})"
    )


#: `load_bed_cov` leaves the BED3 intervals here.
BED_POSITIONS_TABLE = "bed_positions"

#: The `read_alignments` rows that cover anything. An unmapped read keeps any
#: RNAME and POS it was given -- aligners place an unmapped mate beside its
#: partner -- and is reported with stop_position 0, a backwards interval that
#: `compress_intervals` would widen to [0, POS).
ALIGNED_ROWS = "stop_position > position"


@contextmanager
def positions_path(positions):
    """Yield a filesystem path for BED3 input, spooling stdin if it must.

    DuckDB's CSV reader **always** sniffs the dialect, even when every column
    is specified, and sniffing consumes the stream. On a seekable file it
    rewinds and parses; on a pipe there is nothing left to rewind to, and it
    returns **zero rows without raising**. So `micov position-plot < foo.cov`
    cannot read the pipe directly and the input has to become a file first.

    That is the opposite of `micov compress`, which streams stdin the whole
    way -- but `compress` goes through htslib, not DuckDB's CSV reader, and a
    SAM file is orders of magnitude larger than the BED3 for one sample.
    """
    if positions is not None:
        yield positions
        return

    handle, path = tempfile.mkstemp(suffix=".cov")
    try:
        with os.fdopen(handle, "wb") as target:
            shutil.copyfileobj(sys.stdin.buffer, target)
        yield path
    finally:
        os.unlink(path)


def load_bed_cov(con, positions):
    """Load a BED3 `.cov` file into DuckDB.

    Columns are taken by position and renamed, so a file may carry any header
    or none -- released micov wrote one, and hand-made files often do not.
    Header detection is DuckDB's sniffer rather than micov's `_test_has_header`
    because it also handles a `#`-prefixed header, which real BED files carry
    and which `_test_has_header` only recognises by accident.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        A connection from `micov._miint.connection`.
    positions : str
        Path to a `.cov`/BED3 file. Use `positions_path` for stdin.

    """
    source = f"read_csv({sql_string(positions)}, delim='\t')"
    described = con.sql(f"DESCRIBE FROM {source}").fetchall()

    if len(described) < 3:
        raise ValueError(
            f"'{positions}' has {len(described)} column(s); BED3 needs three "
            "(genome_id, start, stop). Check that it is tab-delimited."
        )

    genome_id_col, start_col, stop_col = (row[0] for row in described[:3])
    con.sql(f"""CREATE OR REPLACE TABLE {BED_POSITIONS_TABLE} AS
                SELECT "{genome_id_col}" AS {COLUMN_GENOME_ID},
                       "{start_col}"::UINTEGER AS {COLUMN_START},
                       "{stop_col}"::UINTEGER AS {COLUMN_STOP}
                FROM {source}""")


def _test_has_header(line):
    """Test whether a line appears to be a header."""
    if isinstance(line, bytes):
        line = line.decode("utf-8")

    # `==`, not `in`: COLUMN_GENOME_ID is a plain string, so `in` was a
    # substring test, and a headerless file whose first genome was `id` or
    # `genome` lost that genome's row to being read as a header
    if (
        line.startswith("#")
        or line.split("\t")[0] == COLUMN_GENOME_ID
        or not line.split("\t")[1].strip().isdigit()
    ):
        has_header = True
    else:
        has_header = False

    return has_header


def load_genome_lengths(con, lengths):
    """Load a TSV of feature and length information into DuckDB.

    Was the SQL twin of a polars `parse_genome_lengths`, kept message-for-
    message identical to it so that either path rejected the same file the
    same way. M5 moved the last polars consumer to SQL and deleted the twin;
    `micov/tests/test_io.py` carries the validation coverage that used to
    live on it.
    """
    with open(lengths) as fp:
        first_line = fp.readline()

    header = "true" if _test_has_header(first_line) else "false"
    source = f"read_csv({sql_string(lengths)}, delim='\t', header={header})"

    # the columns are identified by position, so report problems using
    # whatever the file happened to call them
    described = con.sql(f"DESCRIBE FROM {source}").fetchall()
    genome_id_col, length_col = described[0][0], described[1][0]

    if "INT" not in described[1][1]:
        raise ValueError(f"'{length_col}' is not integer'")

    con.sql(f"""CREATE TABLE genome_lengths AS
                SELECT "{genome_id_col}" AS {COLUMN_GENOME_ID},
                       "{length_col}" AS {COLUMN_LENGTH}
                FROM {source}""")

    distinct, total, smallest = con.sql(f"""
        SELECT COUNT(DISTINCT {COLUMN_GENOME_ID}),
               COUNT({COLUMN_GENOME_ID}),
               MIN({COLUMN_LENGTH})
        FROM genome_lengths""").fetchone()

    if distinct != total:
        raise ValueError(f"'{genome_id_col}' is not unique")

    if smallest <= 0:
        raise ValueError("Lengths of zero or less cannot be used")


#: The table `compress_alignments` leaves behind: the frozen covered-positions
#: shape, ready to hand to `write_coverage_parquet`.
ALIGNMENT_POSITIONS_TABLE = "alignment_positions"

#: Extensions htslib cannot open itself. Plain SAM, bgzf `.gz`, `.bam` and
#: `.cram` it reads natively; `.xz` and `.bz2` it does not, and micov's own
#: example data and test fixtures are `.sam.xz`.
_UNREADABLE_BY_HTSLIB = {".xz": lzma.open, ".bz2": bz2.open}


@contextmanager
def _htslib_readable(sam):
    """Yield a path htslib can open, decompressing first only if it must.

    Decompressing costs a temp file the size of the decompressed input, so it
    is done only for the formats htslib genuinely cannot read. It never applies
    to the documented pipe idiom -- `xzcat foo.sam.xz | micov compress` hands
    DuckDB `/dev/stdin`, already decompressed, and streams the whole way.
    """
    _, extension = os.path.splitext(sam)
    opener = _UNREADABLE_BY_HTSLIB.get(extension)

    if opener is None:
        yield sam
        return

    handle, path = tempfile.mkstemp(suffix=".sam")
    try:
        with opener(sam, "rb") as source, os.fdopen(handle, "wb") as target:
            shutil.copyfileobj(source, target)
        yield path
    finally:
        os.unlink(path)


def compress_alignments(con, sam, sample_id, disable_compression=False):
    """Read SAM/BAM through miint and leave merged intervals in a table.

    Reads `sam` exactly **once**. That is not an optimisation -- micov's
    documented idiom pipes SAM in on stdin, which cannot be rewound, so
    anything needing a second look at the input is unimplementable. The
    reference-map check below therefore has to be answerable from this same
    scan.

    Requires the `genome_lengths` table, which `load_genome_lengths` creates;
    it doubles as htslib's reference map, since headerless SAM carries no
    header to resolve RNAME against.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        A connection from `micov._miint.connection`, so miint is loaded.
    sam : str
        Path to SAM/BAM, or a path DuckDB can read such as ``/dev/stdin``.
    sample_id : str
        Written into every row; `coverage.parquet` is keyed by it.
    disable_compression : bool, optional
        Keep every interval instead of merging overlaps.

    Raises
    ------
    ValueError
        If no alignments could be read at all -- an empty input, or a file
        that is not SAM/BAM. Reads whose reference is absent from
        `genome_lengths` are *warned* about rather than raised on; see
        `_report_unattributed` for why they cannot be told apart from reads
        that simply did not align.

    """
    if disable_compression:
        # same shape compress_intervals returns, so the unnest below is common
        intervals = "list({'start': position, 'stop': stop_position})"
    else:
        intervals = "compress_intervals(position, stop_position)"

    intervals += f" FILTER (WHERE {ALIGNED_ROWS})"

    with _htslib_readable(sam) as readable:
        con.sql(f"""CREATE OR REPLACE TABLE alignment_groups AS
                    SELECT reference,
                           {intervals} AS intervals,
                           COUNT(*) AS aligned_reads
                    FROM read_alignments({sql_string(readable)},
                                         reference_lengths := genome_lengths)
                    GROUP BY reference""")

    _report_unattributed(con, sam)

    # htslib parses a non-SAM file as SAM without complaining and simply
    # yields nothing, so BED3 handed to `compress` used to produce two empty
    # parquet files and exit 0. The polars path this replaced errored here
    # too, and an empty result is never what the caller meant by asking to
    # compress something. A genome whose reads were all unmapped has no
    # intervals (NULL), so it does not count as aligned either.
    attributed = con.sql("""SELECT COUNT(*) FROM alignment_groups
                            WHERE reference != '*'
                                AND len(intervals) > 0""").fetchone()[0]
    if attributed == 0:
        raise ValueError(
            f"No alignments were read from '{sam}'. `micov compress` takes "
            "SAM/BAM; BED3 (.cov/.cov.gz) input goes to `micov "
            "cov-to-parquet` instead. If the input really is SAM, check "
            "that it is not empty and that --lengths names its references."
        )

    # '*' is dropped only after the check above has accounted for it
    con.sql(f"""CREATE OR REPLACE TABLE {ALIGNMENT_POSITIONS_TABLE} AS
                SELECT reference AS {COLUMN_GENOME_ID},
                       interval.start::UINTEGER AS {COLUMN_START},
                       interval.stop::UINTEGER AS {COLUMN_STOP},
                       {sql_string(sample_id)} AS {COLUMN_SAMPLE_ID}
                FROM (SELECT reference, UNNEST(intervals) AS interval
                      FROM alignment_groups
                      WHERE reference != '*')""")


def _report_unattributed(con, sam):
    """Warn about alignments that could not be attributed to a genome.

    Two different things arrive here as reference `*`, and **htslib does not
    let micov tell them apart**: a read that did not align at all, and a read
    that aligned to a reference absent from `--lengths`. Verified -- given a
    map covering only `X`, reads against `Y` and `Z` come back not merely as
    `*` but with FLAG rewritten to 4 (unmapped), which is exactly what a
    genuinely unaligned read carries. The original RNAME is gone by then, and
    recovering it would need a second pass the stdin idiom cannot afford.

    So this warns rather than raises. Raising would abort on ordinary SAM,
    which is full of unaligned reads; staying silent would hide a `--lengths`
    file that covers half the references. Note this is not new loss: micov
    already dropped genomes missing from `--lengths`, at the join against
    `genome_lengths`, since a genome with no length has no breadth denominator.
    """
    unattributed = con.sql("""SELECT aligned_reads FROM alignment_groups
                              WHERE reference = '*'""").fetchall()

    if not unattributed or unattributed[0][0] == 0:
        return

    count = unattributed[0][0]
    logger.warning(
        f"{count} alignment(s) in '{sam}' were not attributed to any genome. "
        "They either did not align, or aligned to a reference absent from "
        "--lengths; htslib reports both identically, so micov cannot "
        "distinguish them. If this count is unexpectedly large, check that "
        "--lengths covers every reference the data was aligned against."
    )


def write_coverage_parquet(con, positions, output):
    """Write the frozen two-file parquet pair from a covered-positions source.

    Both `micov compress` and `micov cov-to-parquet` land here, so the
    format exists in one place. The split is load-bearing: `coverage.parquet`
    is one row per sample per genome and drives ordering and filtering, while
    `covered_positions.parquet` is large and only ever scanned with pushdown
    predicates.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        A connection with the `genome_lengths` table loaded.
    positions : str
        SQL yielding `genome_id`, `start`, `stop`, `sample_id` in that order.
    output : str
        Base path; the two suffixes are appended to it.

    """
    covered_positions = sql_string(f"{output}.covered_positions.parquet")
    coverage = sql_string(f"{output}.coverage.parquet")

    con.sql(f"""COPY ({positions})
                TO {covered_positions}
                    (FORMAT PARQUET, PARQUET_VERSION V2, COMPRESSION zstd)""")

    # `(covered / length) * 100`, not `covered * 100 / length`. The two differ
    # in the last bits and the published coverage values were computed this
    # way; test_alignments.py pins a case where it shows.
    con.sql(f"""
        COPY (WITH covered_amount AS (
                  SELECT {COLUMN_SAMPLE_ID},
                         {COLUMN_GENOME_ID},
                         SUM({COLUMN_STOP} - {COLUMN_START})::UINTEGER
                             AS {COLUMN_COVERED}
                  FROM read_parquet({covered_positions})
                  GROUP BY {COLUMN_SAMPLE_ID}, {COLUMN_GENOME_ID})
              SELECT {COLUMN_SAMPLE_ID},
                     {COLUMN_GENOME_ID},
                     {COLUMN_COVERED},
                     {COLUMN_LENGTH},
                     ({COLUMN_COVERED} / {COLUMN_LENGTH}) * 100
                         AS {COLUMN_PERCENT_COVERED}
              FROM covered_amount JOIN genome_lengths USING ({COLUMN_GENOME_ID}))
        TO {coverage}
            (FORMAT PARQUET, PARQUET_VERSION V2, COMPRESSION zstd)""")


#: Columns `depth-plot` reads from `read_alignments` output saved as Parquet.
#: `sample_id` is not one of `read_alignments`' own; the user adds it.
ALIGNMENT_LAYER_COLUMNS = (
    COLUMN_SAMPLE_ID,
    "reference",
    "position",
    "stop_position",
    "cigar",
)

#: `depth-plot`'s genomes, lengths and detail regions; see `load_depth_features`.
DEPTH_FEATURES_TABLE = "depth_features"

#: `sample_id` and the value of the stratifying metadata column.
SAMPLE_GROUPS_TABLE = "sample_groups"

#: ORFs from a `read_gff` Parquet; see `load_orfs`.
ORFS_TABLE = "orfs"

#: Columns read from a `read_gff` Parquet.
ORF_COLUMNS = ("seqid", "type", "position", "stop_position", "strand", "attributes")

#: GFF types drawn as ORFs. `gene` repeats its CDS, and `region` is the whole
#: sequence.
ORF_TYPES = ("CDS", "rRNA", "tRNA", "tmRNA", "ncRNA")


def _missing_columns(con, source, required):
    present = {row[0] for row in con.sql(f"DESCRIBE FROM {source}").fetchall()}
    return [column for column in required if column not in present]


def _examples(rows, limit=5):
    """Render offending rows for an error message, at most `limit` of them."""
    shown = ", ".join(str(row[0]) for row in rows[:limit])
    more = f" and {len(rows) - limit} more" if len(rows) > limit else ""
    return shown + more


def load_alignment_layer(con, path, view):
    """Expose an alignment Parquet as the view `view`, checking its columns.

    The file is `read_alignments` output with a `sample_id` column added.
    `read_alignments` has no sample column of its own, so a user who saves its
    output as is has a file that cannot be grouped; that is refused here with
    the fix, rather than surfacing as a binder error later.
    """
    source = f"read_parquet({sql_string(path)})"
    missing = _missing_columns(con, source, ALIGNMENT_LAYER_COLUMNS)
    if missing:
        hint = ""
        if COLUMN_SAMPLE_ID in missing:
            hint = (
                " `read_alignments` has no sample column: add one when saving "
                "its output, for example from `include_filepath := true`."
            )
        raise ValueError(
            f"'{path}' has no {', '.join(repr(c) for c in missing)} column.{hint}"
        )
    con.sql(f"""CREATE OR REPLACE VIEW {view} AS
                SELECT {", ".join(ALIGNMENT_LAYER_COLUMNS)} FROM {source}""")


def load_depth_features(con, path):
    """Load `depth-plot`'s `--features-to-keep` into `DEPTH_FEATURES_TABLE`.

    Columns: `genome_id`, then a required `length`, an optional `is_circular`
    (missing or empty is linear) and optional `start`/`stop`. A row with a
    region is a detail panel; a row without one only names the genome. The
    genome's own columns must agree across its rows.

    The file is read as text and every column converted explicitly, so what a
    value means does not depend on what DuckDB's sniffer guessed from it.

    Raises
    ------
    ValueError
        Naming the genomes concerned, for a missing or non-positive length, an
        `is_circular` that is not true or false, half a region, an empty or
        inverted region, a region beyond the genome, a genome given two
        lengths or two circularities, or a region given twice.
    """
    query = read_tsv_with_header(
        con, path, [COLUMN_GENOME_ID], FEATURE_ID_COLUMNS, all_varchar=True
    )
    columns = [row[0] for row in con.sql(f"DESCRIBE {query}").fetchall()]

    if COLUMN_LENGTH not in columns:
        raise ValueError(
            f"'{path}' has no '{COLUMN_LENGTH}' column. depth-plot takes each "
            "genome's length from it, since alignments do not carry one."
        )
    regions = COLUMN_START in columns
    if regions != (COLUMN_STOP in columns):
        present, absent = (
            (COLUMN_START, COLUMN_STOP) if regions else (COLUMN_STOP, COLUMN_START)
        )
        raise ValueError(f"'{path}' has a '{present}' column but no '{absent}'")

    def text(column):
        return f'"{column}"' if column in columns else "NULL"

    con.sql(f"""CREATE OR REPLACE TEMP TABLE depth_features_text AS
                SELECT {COLUMN_GENOME_ID},
                       {text(COLUMN_LENGTH)} AS {COLUMN_LENGTH},
                       {text("is_circular")} AS is_circular,
                       {text(COLUMN_START)} AS {COLUMN_START},
                       {text(COLUMN_STOP)} AS {COLUMN_STOP}
                FROM ({query})""")

    def offending(where, show):
        return con.sql(f"""SELECT DISTINCT {show} FROM depth_features_text
                           WHERE {where} ORDER BY 1""").fetchall()

    checks = (
        (
            f"TRY_CAST({COLUMN_LENGTH} AS BIGINT) IS NULL "
            f"OR TRY_CAST({COLUMN_LENGTH} AS BIGINT) <= 0",
            f"{COLUMN_GENOME_ID} || ' (' || coalesce({COLUMN_LENGTH}, 'empty') || ')'",
            "a length that is not a positive integer",
        ),
        (
            "is_circular IS NOT NULL AND TRY_CAST(is_circular AS BOOLEAN) IS NULL",
            f"{COLUMN_GENOME_ID} || ' (' || is_circular || ')'",
            "an is_circular that is not true or false",
        ),
        (
            f"({COLUMN_START} IS NULL) != ({COLUMN_STOP} IS NULL) "
            f"OR ({COLUMN_START} IS NOT NULL "
            f"AND TRY_CAST({COLUMN_START} AS BIGINT) IS NULL) "
            f"OR ({COLUMN_STOP} IS NOT NULL "
            f"AND TRY_CAST({COLUMN_STOP} AS BIGINT) IS NULL)",
            f"{COLUMN_GENOME_ID}",
            "a region whose start and stop are not both integers",
        ),
        (
            f"{COLUMN_STOP}::BIGINT <= {COLUMN_START}::BIGINT",
            f"{COLUMN_GENOME_ID} || ' [' || {COLUMN_START} || ', ' "
            f"|| {COLUMN_STOP} || ')'",
            "an empty region: regions are half-open, so stop must exceed start",
        ),
        (
            f"{COLUMN_START}::BIGINT < 1 "
            f"OR {COLUMN_STOP}::BIGINT > {COLUMN_LENGTH}::BIGINT + 1",
            f"{COLUMN_GENOME_ID} || ' [' || {COLUMN_START} || ', ' "
            f"|| {COLUMN_STOP} || ')'",
            "a region outside the genome, which is [1, length + 1)",
        ),
    )
    for where, show, problem in checks:
        rows = offending(where, show)
        if rows:
            raise ValueError(f"'{path}' has {problem}: {_examples(rows)}")

    con.sql(f"""CREATE OR REPLACE TABLE {DEPTH_FEATURES_TABLE} AS
                SELECT {COLUMN_GENOME_ID},
                       {COLUMN_LENGTH}::BIGINT AS {COLUMN_LENGTH},
                       coalesce(is_circular::BOOLEAN, false) AS is_circular,
                       {COLUMN_START}::BIGINT AS {COLUMN_START},
                       {COLUMN_STOP}::BIGINT AS {COLUMN_STOP}
                FROM depth_features_text""")
    con.sql("DROP TABLE depth_features_text")

    rows = con.sql(f"""SELECT {COLUMN_GENOME_ID} FROM {DEPTH_FEATURES_TABLE}
                       GROUP BY {COLUMN_GENOME_ID}
                       HAVING COUNT(DISTINCT ({COLUMN_LENGTH}, is_circular)) > 1
                       ORDER BY 1""").fetchall()
    if rows:
        raise ValueError(
            f"'{path}' gives a genome more than one length or is_circular: "
            f"{_examples(rows)}"
        )
    rows = con.sql(f"""SELECT {COLUMN_GENOME_ID} FROM {DEPTH_FEATURES_TABLE}
                       GROUP BY {COLUMN_GENOME_ID}, {COLUMN_START}, {COLUMN_STOP}
                       HAVING COUNT(*) > 1
                       ORDER BY 1""").fetchall()
    if rows:
        raise ValueError(f"'{path}' lists a region twice: {_examples(rows)}")


def load_sample_groups(con, path, column):
    """Load each sample's value of `column` into `SAMPLE_GROUPS_TABLE`.

    Values are read as written: as text, so `Yes` stays `Yes` rather than
    becoming a group named `True`. A sample with no value belongs to no group;
    it is left out and reported.
    """
    query = read_tsv_with_header(
        con, path, [COLUMN_SAMPLE_ID], SAMPLE_ID_COLUMNS, all_varchar=True
    )
    columns = [row[0] for row in con.sql(f"DESCRIBE {query}").fetchall()]
    if column not in columns[1:]:
        raise ValueError(
            f"'{path}' has no column {column!r}; its columns are "
            f"{', '.join(columns[1:])}"
        )
    con.sql(f"""CREATE OR REPLACE TABLE {SAMPLE_GROUPS_TABLE} AS
                SELECT {COLUMN_SAMPLE_ID}, "{column}" AS group_name
                FROM ({query})""")
    rows = con.sql(f"""SELECT {COLUMN_SAMPLE_ID} FROM {SAMPLE_GROUPS_TABLE}
                       WHERE group_name IS NULL ORDER BY 1""").fetchall()
    if rows:
        logger.warning(
            f"{len(rows)} sample(s) in '{path}' have no value for {column!r} "
            f"and are left out: {_examples(rows, limit=len(rows))}"
        )
        con.sql(f"DELETE FROM {SAMPLE_GROUPS_TABLE} WHERE group_name IS NULL")


def load_orfs(con, path):
    """Load ORFs from a `read_gff` Parquet into `ORFS_TABLE`, by genome.

    Only `ORF_TYPES` are kept. Coordinates stay as `read_gff` wrote them,
    already half-open (GFF end + 1). Each ORF needs an `ID`, which keys the
    per-ORF table; its label is `gene`, else `locus_tag`, else `ID`. An
    unstranded ORF gets strand `.` rather than NULL, which numpy would
    otherwise hand back as a masked array.
    """
    source = f"read_parquet({sql_string(path)})"
    missing = _missing_columns(con, source, ORF_COLUMNS)
    if missing:
        raise ValueError(
            f"'{path}' has no {', '.join(repr(c) for c in missing)} column; ORFs "
            "are read_gff output saved as Parquet."
        )
    types = ", ".join(sql_string(orf_type) for orf_type in ORF_TYPES)
    con.sql(f"""CREATE OR REPLACE TABLE {ORFS_TABLE} AS
                SELECT seqid AS {COLUMN_GENOME_ID},
                       nullif(attributes['ID'], '') AS orf_id,
                       coalesce(nullif(attributes['gene'], ''),
                                nullif(attributes['locus_tag'], ''),
                                nullif(attributes['ID'], '')) AS label,
                       type,
                       position::BIGINT AS {COLUMN_START},
                       stop_position::BIGINT AS {COLUMN_STOP},
                       coalesce(strand, '.') AS strand,
                       attributes
                FROM {source}
                WHERE type IN ({types})
                -- read one genome at a time: in order, each read touches
                -- only that genome's row groups
                ORDER BY {COLUMN_GENOME_ID}, {COLUMN_START}""")
    rows = con.sql(f"""SELECT type || ' at ' || {COLUMN_GENOME_ID} || ':'
                              || {COLUMN_START}
                       FROM {ORFS_TABLE} WHERE orf_id IS NULL
                       ORDER BY {COLUMN_GENOME_ID}, {COLUMN_START}""").fetchall()
    if rows:
        raise ValueError(
            f"{len(rows)} ORF(s) in '{path}' have no ID attribute, which keys "
            f"the per-ORF table: {_examples(rows)}"
        )



#: `depth-plot`'s per-ORF table: its columns and their types, in order.
ORF_STATISTICS_COLUMNS = (
    (COLUMN_GENOME_ID, "VARCHAR"), ("orf_id", "VARCHAR"), ("label", "VARCHAR"),
    ("type", "VARCHAR"), (COLUMN_START, "BIGINT"), (COLUMN_STOP, "BIGINT"),
    ("strand", "VARCHAR"), ("group", "VARCHAR"), ("n_samples", "BIGINT"),
    ("depth_q1", "DOUBLE"), ("depth_median", "DOUBLE"), ("depth_q3", "DOUBLE"),
    ("depth_mean", "DOUBLE"), ("prevalence", "DOUBLE"),
    ("union_breadth", "DOUBLE"), ("contrast", "DOUBLE"),
)

#: `add_orf_table` collects every genome's per-ORF statistics here.
ORF_STATISTICS_TABLE = "depth_orf_statistics"

#: `add_orf_table` registers one genome's under this name while it copies them.
_ORF_GENOME_RELATION = "depth_orf_genome"


def add_orf_table(con, table):
    """Add one genome's per-ORF statistics to `ORF_STATISTICS_TABLE`.

    Copied in as each genome is computed, so a run over thousands of genomes
    holds them in DuckDB, which can spill to disk, rather than in memory.
    A missing contrast is NaN in the table and NULL in the file: DuckDB reads
    a numpy NaN as NULL.

    Parameters
    ----------
    table : dict of np.ndarray
        `_depth.genome_statistics`' ORF table: the `ORF_STATISTICS_COLUMNS`.
    """
    schema = ", ".join(f'"{name}" {kind}' for name, kind in ORF_STATISTICS_COLUMNS)
    casts = ", ".join(f'"{name}"::{kind}' for name, kind in ORF_STATISTICS_COLUMNS)
    con.sql(f"CREATE TEMP TABLE IF NOT EXISTS {ORF_STATISTICS_TABLE} ({schema})")
    con.register(_ORF_GENOME_RELATION, table)
    try:
        con.sql(f"""INSERT INTO {ORF_STATISTICS_TABLE}
                    SELECT {casts} FROM {_ORF_GENOME_RELATION}""")
    finally:
        con.unregister(_ORF_GENOME_RELATION)


def write_orf_table(con, output, variable):
    """Write every genome's per-ORF statistics, as added, to one Parquet.

    The columns, in order (`ORF_STATISTICS_COLUMNS`): `genome_id`, `orf_id`,
    `label`, `type`, `start`, `stop` (half-open, as `read_gff` gives them),
    `strand`, `group`, `n_samples`, then the group's `depth_q1`,
    `depth_median`, `depth_q3`, `depth_mean`, `prevalence`, `union_breadth`,
    and the ORF's `contrast`. Requires `ORF_STATISTICS_TABLE`.

    Returns
    -------
    str
        The path written, ``{output}.{variable}.depth-plot-orfs.parquet``.
    """
    path = f"{output}.{variable}.depth-plot-orfs.parquet"
    con.sql(f"""COPY {ORF_STATISTICS_TABLE} TO {sql_string(path)}
                    (FORMAT PARQUET, PARQUET_VERSION V2, COMPRESSION zstd)""")
    return path
