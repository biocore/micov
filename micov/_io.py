import bz2
import gzip
import io
import lzma
import math
import os
import shutil
import tarfile
import tempfile
import time
from contextlib import contextmanager

import polars as pl

from ._constants import (
    BED_COV_SCHEMA,
    COLUMN_COVERED,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_NAME,
    COLUMN_PERCENT_COVERED,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
    COLUMN_TAXONOMY,
    GENOME_COVERAGE_SCHEMA,
)
from ._cov import compress, coverage_percent
from ._utils import logger


class SetOfAll:
    """A universal set."""

    def __contains__(self, other):
        return True


def parse_bed_cov_to_df(data):
    """BED3 -> DataFrame.

    Parameters
    ----------
    data : IO-like
        The data to parse

    Returns
    -------
    pl.DataFrame
        The BED3 data expressed within a DataFrame

    """
    return _parse_bed_cov(data, None, None, False)


def _parse_bed_cov(data, feature_drop, feature_keep, lazy):
    """BED3 -> DataFrame.

    Parameters
    ----------
    data : IO-like
        The data to parse
    feature_drop : iterable
        Any features to explicitly drop (all others are kept)
    feature_keep : iterable
        Any features to explicitly keep (all others are dropped)
    lazy : bool
        Return LazyFrame or DataFrame

    """
    first_line = data.readline()
    data.seek(0)

    if len(first_line) == 0:
        return None

    if _test_has_header(first_line):
        skip_rows = 1
    else:
        skip_rows = 0

    frame = pl.read_csv(
        data.read(),
        separator="\t",
        new_columns=BED_COV_SCHEMA.columns,
        schema_overrides=BED_COV_SCHEMA.dtypes_dict,
        has_header=False,
        skip_rows=skip_rows,
    ).lazy()

    if feature_drop is not None:
        frame = frame.filter(~pl.col(COLUMN_GENOME_ID).is_in(feature_drop))

    if feature_keep is not None:
        frame = frame.filter(pl.col(COLUMN_GENOME_ID).is_in(feature_keep))

    if lazy:
        return frame
    else:
        return frame.collect()


def parse_qiita_coverages(tgzs, *args, **kwargs):
    """Parse a Qiita-style coverages.tgz file.

    Parameters
    ----------
    tgzs : iterable of str
        The file paths to process
    *args : stuff or None
        Forwarded to _parse_qiita_coverages
    **kwargs : dict, optional
        Forwarded to _parse_qiita_coverages

    """
    if not isinstance(tgzs, list | tuple | set | frozenset):
        tgzs = [
            tgzs,
        ]

    compress_size = kwargs.get("compress_size", 50_000_000)

    if compress_size is not None:
        assert isinstance(compress_size, int)
        assert compress_size >= 0
    else:
        compress_size = math.inf
        kwargs["compress_size"] = compress_size

    frame = _parse_qiita_coverages(tgzs[0], *args, **kwargs)

    if len(tgzs) == 1:
        # short circuit, already compressed
        return frame

    for tgz in tgzs[1:]:
        next_frame = _parse_qiita_coverages(tgz, *args, **kwargs)
        frame = _single_df(_check_and_compress([frame, next_frame], compress_size))

    if compress_size == math.inf:
        return frame
    else:
        return _single_df(
            _check_and_compress(
                [
                    frame,
                ],
                compress_size=0,
            )
        )


def _parse_qiita_coverages(
    tgz,
    compress_size=50_000_000,
    sample_keep=None,
    sample_drop=None,
    feature_keep=None,
    feature_drop=None,
    append_sample_id=False,
):
    """Parse an individual Qiita-style coverages.tgz file.

    A coverages.tgz file contains BED-3 style coverage information per sample.

    Parameters
    ----------
    tgz : str
        The path to process
    compress_size : int, optional
        The number of records to buffer until a compression occurs
    sample_keep : iterable, optional
        Samples to explicitly keep (all others are dropped)
    sample_drop : iterable, optional
        Samples to explicitly drop (all others are kept)
    feature_keep : iterable, optional
        Features to explicitly keep (all others are dropped)
    feature_drop : iterable, optiona;
        Features to explicilty drop (all others are kept)
    append_sample_id : bool
        Whether to include in the resulting DataFrame the detected sample IDs

    Returns
    -------
    pl.DataFrame
        A dataframe representing the coverage data

    """
    # compress_size=None to disable compression
    fp = tarfile.open(tgz)

    try:
        fp.extractfile("coverage_percentage.txt")
    except KeyError as e:
        raise KeyError(f"{tgz} does not look like a Qiita coverage tgz") from e

    if sample_keep is None:
        sample_keep = SetOfAll()

    if sample_drop is None:
        sample_drop = set()

    coverages = []
    for name in fp.getnames():
        if "coverages/" not in name:
            continue

        _, filename = name.split("/")
        sample_id = filename.rsplit(".", 1)[0]

        if sample_id in sample_drop:
            continue

        if sample_id not in sample_keep:
            continue

        data = fp.extractfile(name)
        frame = _parse_bed_cov(data, feature_drop, feature_keep, lazy=True)

        if frame is None:
            continue

        if append_sample_id:
            frame = frame.with_columns(pl.lit(sample_id).alias(COLUMN_SAMPLE_ID))

        coverages.append(frame.collect())
        coverages = _check_and_compress(coverages, compress_size)

    if compress_size == math.inf:
        return _single_df(coverages)
    else:
        return _single_df(_check_and_compress(coverages, compress_size=0))


def _single_df(coverages):
    """Map [pl.DataFrame, ...] -> pl.DataFrame."""
    if len(coverages) > 1:
        df = pl.concat(coverages, rechunk=True)
    elif len(coverages) == 0:
        raise ValueError("No coverages")
    else:
        df = coverages[0]

    return df


def _check_and_compress(coverages, compress_size):
    """Check whether we have buffered enough, if so compress."""
    rowcount = sum([len(df) for df in coverages])
    if rowcount > compress_size:
        df = compress(_single_df(coverages))
        coverages = [
            df,
        ]
    return coverages


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


def _test_has_header_taxonomy(line):
    """Test whether a line appears to be a taxonomy header."""
    if isinstance(line, bytes):
        line = line.decode("utf-8")

    genome_id_columns = COLUMN_GENOME_ID
    taxonomy_columns = COLUMN_TAXONOMY

    if (
        line.startswith("#")
        or (
            line.split("\t")[0] in genome_id_columns
            and line.split("\t")[1] in taxonomy_columns
        )
    ):
        has_header = True
    else:
        has_header = False

    return has_header


def parse_genome_lengths(lengths):
    """Parse a TSV representing feature and length information."""
    with open(lengths) as fp:
        first_line = fp.readline()

    has_header = _test_has_header(first_line)
    df = pl.read_csv(lengths, separator="\t", has_header=has_header)
    genome_id_col = df.columns[0]
    length_col = df.columns[1]

    genome_ids = df[genome_id_col]
    if len(genome_ids) != len(set(genome_ids)):
        raise ValueError(f"'{genome_id_col}' is not unique")

    if not df[length_col].dtype.is_integer():
        raise ValueError(f"'{length_col}' is not integer'")

    if df[length_col].min() <= 0:
        raise ValueError("Lengths of zero or less cannot be used")

    rename = {genome_id_col: COLUMN_GENOME_ID, length_col: COLUMN_LENGTH}
    return df[[genome_id_col, length_col]].rename(rename)


def load_genome_lengths(con, lengths):
    """Load a TSV of feature and length information into DuckDB.

    The SQL twin of `parse_genome_lengths`, which callers still working in
    polars continue to use. Validation and its messages are deliberately
    identical, so either path rejects the same file the same way; the two
    converge into one when the last polars consumer moves to SQL.
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


def parse_taxonomy(taxonomy):
    """Parse a TSV representing feature and taxonomy information."""
    with open(taxonomy) as fp:
        first_line = fp.readline()

    has_header = _test_has_header_taxonomy(first_line)
    df = pl.read_csv(taxonomy, separator="\t", has_header=has_header)
    genome_id_col = df.columns[0]
    taxonomy_col = df.columns[1]

    genome_ids = df[genome_id_col]
    if len(genome_ids) != len(set(genome_ids)):
        raise ValueError(f"'{genome_id_col}' is not unique")

    rename = {genome_id_col: COLUMN_GENOME_ID, taxonomy_col: COLUMN_TAXONOMY}

    return df[[genome_id_col, taxonomy_col]].rename(rename)


def set_taxonomy_as_id(coverages, taxonomy):
    """Add taxonomy information to a coverages DataFrame."""
    missing = set(coverages[COLUMN_GENOME_ID]) - set(taxonomy[COLUMN_GENOME_ID])
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} genome(s) appear unrepresented in "
            f"the taxonomy information, examples: "
            f"{sorted(missing)[:5]}"
        )

    return coverages.join(taxonomy, on=COLUMN_GENOME_ID, how="inner").select(
        COLUMN_TAXONOMY, pl.exclude(COLUMN_TAXONOMY)
    )


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
    # parquet files and exit 0. The old polars path errored here as well (via
    # `_single_df`'s "No coverages"), and an empty result is never what the
    # caller meant by asking to compress something.
    attributed = con.sql("""SELECT COUNT(*) FROM alignment_groups
                            WHERE reference != '*'""").fetchone()[0]
    if attributed == 0:
        raise ValueError(
            f"No alignments were read from '{sam}'. `micov compress` takes "
            "SAM/BAM; BED3 (.cov/.cov.gz) input goes to `micov "
            "nonqiita-to-parquet` instead. If the input really is SAM, check "
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
    already dropped genomes missing from `--lengths`, at the join in
    `coverage_percent`, since it has no length to compute breadth against.
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

    Both `micov compress` and `micov nonqiita-to-parquet` land here, so the
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


def _add_file(tf, name, data):
    """Add a file to a tgz."""
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = int(time.time())
    tf.addfile(ti, io.BytesIO(data))


def write_qiita_cov(name, paths, lengths):
    """Construct a Qiita-style coverages.tgz.

    Parameters
    ----------
    name : str
        The path of the tgz to write.
    paths : iterable
        The paths of the coverage data to include in the tgz.
    lengths : pl.DataFrame
        The genome -> length information.

    """
    tf = tarfile.open(name, "w:gz")

    coverages = []
    for p in paths:
        with open(p, "rb") as fp:
            data = fp.read()

        if len(data) == 0:
            continue

        base = os.path.basename(p)
        if base.endswith(".cov.gz"):
            data = gzip.decompress(data)
            name = base.rsplit(".", 2)[0] + ".cov"
        elif base.endswith(".cov"):
            name = base
        else:
            name = base + ".cov"

        name = f"coverages/{name}"

        _add_file(tf, name, data)
        current_coverage = _parse_bed_cov(io.BytesIO(data), None, None, False)
        coverages.append(current_coverage)
        coverages = _check_and_compress(coverages, compress_size=50_000_000)

    coverage = _single_df(_check_and_compress(coverages, compress_size=0))

    covdataname = "artifact.cov"
    covdata = io.BytesIO()
    coverage.write_csv(covdata, separator="\t", include_header=True)
    covdata.seek(0)
    _add_file(tf, covdataname, covdata.read())

    genome_coverage = coverage_percent(coverage, lengths).collect()
    pername = "coverage_percentage.txt"
    perdata = io.BytesIO()
    genome_coverage.write_csv(perdata, separator="\t", include_header=True)
    perdata.seek(0)
    _add_file(tf, pername, perdata.read())

    tf.close()


def parse_features_to_keep(path):
    if path is None:
        return None

    df = pl.read_csv(path, separator="\t")
    return df.rename({df.columns[0]: COLUMN_GENOME_ID})


def parse_feature_names(path):
    """Parse a TSV of feature names.

    We assume the file has a header, and has two columns. The first is the
    feature ID and second is the name for the feature.

    If the feature name appears to be a lineage, in that it contains "; ",
    the lineage will be split and the last name retained.
    """
    if path is None:
        return None

    df = pl.read_csv(path, separator="\t")

    return (
        df.lazy()
        .rename({df.columns[0]: COLUMN_GENOME_ID, df.columns[1]: COLUMN_NAME})
        .with_columns(
            pl.when(pl.col(COLUMN_NAME).str.contains("; "))
            .then(pl.col(COLUMN_NAME).str.split("; ").list.get(-1))
            .otherwise(pl.col(COLUMN_NAME))
            .alias(COLUMN_NAME)
        )
        .with_columns(pl.col(COLUMN_NAME).str.replace_all(r" |\[|\]", "_"))
        .select([COLUMN_GENOME_ID, COLUMN_NAME])
        .collect()
    )


def parse_sample_metadata(path):
    """Naively parse sample metadata, do not infer types."""
    df = pl.read_csv(path, separator="\t", infer_schema_length=0)
    return df.rename({df.columns[0]: COLUMN_SAMPLE_ID})


def parse_coverage(data, features_to_keep):
    """Parse a simple TSV descriving total coverage."""
    cov_df = pl.read_csv(
        data.read(),
        separator="\t",
        new_columns=GENOME_COVERAGE_SCHEMA.columns,
        schema_overrides=GENOME_COVERAGE_SCHEMA.dtypes_dict,
    ).lazy()

    if features_to_keep is not None:
        cov_df = cov_df.filter(pl.col(COLUMN_GENOME_ID).is_in(features_to_keep))

    return cov_df


def _first_col_as_set(fp):
    df = pl.read_csv(fp, separator="\t", infer_schema_length=0)
    return set(df[df.columns[0]])


def combine_pos_metadata_length(
    sample_metadata, length, covered_positions, features_to_keep
):
    df_md = parse_sample_metadata(sample_metadata).lazy()
    df_length = parse_genome_lengths(length).lazy()
    df_pos = pl.scan_parquet(covered_positions)

    df_pos_md = df_pos.join(df_md, on=COLUMN_SAMPLE_ID, how="left").join(
        df_length, on=COLUMN_GENOME_ID, how="left"
    )

    if features_to_keep:
        features_to_keep = _first_col_as_set(features_to_keep)
        df_pos_md = df_pos_md.filter(pl.col(COLUMN_GENOME_ID).is_in(features_to_keep))

    return df_pos_md
