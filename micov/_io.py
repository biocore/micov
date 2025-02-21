import gzip
import io
import lzma
import math
import os
import sys
import tarfile
import time
from contextlib import contextmanager

import polars as pl

from ._constants import (
    BED_COV_SCHEMA,
    COLUMN_CIGAR,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_NAME,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_START_DTYPE,
    COLUMN_STOP,
    COLUMN_STOP_DTYPE,
    COLUMN_TAXONOMY,
    GENOME_COVERAGE_SCHEMA,
    SAM_SUBSET_SCHEMA,
)
from ._convert import cigar_to_lens
from ._cov import compress, coverage_percent


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
        or line.split("\t")[0] in genome_id_columns
        and line.split("\t")[1] in taxonomy_columns
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


# TODO: this is not the greatest method name
def parse_sam_to_df(sam):
    """Minimally parse SAM and compute stop coordinates from CIGAR."""
    # scan_csv does not seem to support this juggling
    # and it seems we cannot pass in a Schema while also specifying the column
    # indices and names. there probably is a better way here.
    df = (
        pl.read_csv(
            sam,
            separator="\t",
            has_header=False,
            columns=SAM_SUBSET_SCHEMA.column_indices,
            comment_prefix="@",
            new_columns=SAM_SUBSET_SCHEMA.columns,
        )
        .lazy()
        .with_columns(
            pl.col(COLUMN_START).cast(COLUMN_START_DTYPE),
            pl.col(COLUMN_CIGAR)
            .map_elements(cigar_to_lens, return_dtype=COLUMN_STOP_DTYPE)
            .alias(COLUMN_STOP),
        )
        .with_columns(
            (pl.col(COLUMN_STOP) + pl.col(COLUMN_START))
            .cast(COLUMN_STOP_DTYPE)
            .alias(COLUMN_STOP)
        )
    )
    return df.collect()


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


@contextmanager
def _reader(sam):
    """Indirection to support reading from stdin or a file."""
    if sam == "-" or sam is None:
        data = sys.stdin.buffer
        yield data
    elif isinstance(sam, io.BytesIO):
        yield sam
    else:
        if sam.endswith(".sam.xz"):
            with lzma.open(sam, mode="rb") as fp:
                yield fp
        elif sam.endswith(".sam.gz"):
            with gzip.open(sam, mode="rb") as fp:
                yield fp
        else:
            with open(sam, "rb") as fp:
                yield fp


def _flatten_buf(buf):
    """Map [data_1, ... data_N] -> IOobject(all_data) via simple join."""
    if isinstance(buf[0], str):
        return io.StringIO("".join(buf))
    else:
        return io.BytesIO(b"".join(buf))


def _subset_sam_to_bed(df):
    """Pull a subset of specific columns from a dataframe."""
    return df[list(BED_COV_SCHEMA.columns)]


def compress_from_stream(sam, bufsize=100_000_000, disable_compression=False):
    """Compress SAM-like or BED3-like data.

    Parameters
    ----------
    sam : file path of buffer (e.g., sys.stdin)
        The data to consume.
    bufsize : int, optional
        The number of records to buffer before a compressing (i.e., collapsing
        overlapping intervls).
    disable_compression : bool, optional
        If true, do not compress the intervals.

    Returns
    -------
    pl.DataFrame
        A BED-3 like dataframe describing the feature, start and stop regions
        represented by the input SAM data.

    """
    if disable_compression:
        compress_f = _subset_sam_to_bed
    else:
        compress_f = compress

    current_df = pl.DataFrame([], schema=BED_COV_SCHEMA.dtypes_flat)
    with _reader(sam) as data:
        buf = data.readlines(bufsize)

        if len(buf) == 0:
            return None

        line = buf[0]
        if isinstance(line, str):
            delim = "\t"
        elif isinstance(line, bytes):
            delim = b"\t"
        else:
            raise ValueError(f"Unexpected buffer type: {type(line)}")

        if len(line.split(delim)) == 3:
            parse_f = parse_bed_cov_to_df
        else:
            parse_f = parse_sam_to_df

        while len(buf) > 0:
            next_df = compress_f(parse_f(_flatten_buf(buf)))
            current_df = compress_f(pl.concat([current_df, next_df]))
            buf = data.readlines(bufsize)

    return current_df.rechunk()


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
