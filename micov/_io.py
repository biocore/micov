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
    COLUMN_PERCENT_COVERED,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)
from ._utils import logger

#: `load_bed_cov` leaves the BED3 intervals here.
BED_POSITIONS_TABLE = "bed_positions"


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
    source = f"read_csv('{positions}', delim='\t')"
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

    genome_id_columns = COLUMN_GENOME_ID

    if (
        line.startswith("#")
        or line.split("\t")[0] in genome_id_columns
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
    source = f"read_csv('{lengths}', delim='\t', header={header})"

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

    with _htslib_readable(sam) as readable:
        con.sql(f"""CREATE OR REPLACE TABLE alignment_groups AS
                    SELECT reference,
                           {intervals} AS intervals,
                           COUNT(*) AS aligned_reads
                    FROM read_alignments('{readable}',
                                         reference_lengths := genome_lengths)
                    GROUP BY reference""")

    _report_unattributed(con, sam)

    # htslib parses a non-SAM file as SAM without complaining and simply
    # yields nothing, so BED3 handed to `compress` used to produce two empty
    # parquet files and exit 0. The polars path this replaced errored here
    # too, and an empty result is never what the caller meant by asking to
    # compress something.
    attributed = con.sql("""SELECT COUNT(*) FROM alignment_groups
                            WHERE reference != '*'""").fetchone()[0]
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
                       '{sample_id}' AS {COLUMN_SAMPLE_ID}
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
    con.sql(f"""COPY ({positions})
                TO '{output}.covered_positions.parquet'
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
                  FROM read_parquet('{output}.covered_positions.parquet')
                  GROUP BY {COLUMN_SAMPLE_ID}, {COLUMN_GENOME_ID})
              SELECT {COLUMN_SAMPLE_ID},
                     {COLUMN_GENOME_ID},
                     {COLUMN_COVERED},
                     {COLUMN_LENGTH},
                     ({COLUMN_COVERED} / {COLUMN_LENGTH}) * 100
                         AS {COLUMN_PERCENT_COVERED}
              FROM covered_amount JOIN genome_lengths USING ({COLUMN_GENOME_ID}))
        TO '{output}.coverage.parquet'
            (FORMAT PARQUET, PARQUET_VERSION V2, COMPRESSION zstd)""")
