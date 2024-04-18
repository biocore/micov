from contextlib import contextmanager
import lzma
import polars as pl
import os
import sys
import tarfile
import time
import math
import io
import gzip

from ._cov import compress, coverage_percent
from ._constants import (BED_COV_SCHEMA, COLUMN_GENOME_ID, COLUMN_LENGTH,
                         SAM_SUBSET_SCHEMA, COLUMN_CIGAR, COLUMN_STOP,
                         COLUMN_START, COLUMN_SAMPLE_ID)
from ._convert import cigar_to_lens


class SetOfAll:
    # forgot the formal name for this
    def __contains__(self, other):
        return True


def _parse_bed_cov(data, feature_drop, feature_keep, lazy):
    first_line = data.readline()
    data.seek(0)

    if len(first_line) == 0:
        return None

    if _test_has_header(first_line):
        skip_rows = 1
    else:
        skip_rows = 0

    frame = pl.read_csv(data.read(), separator='\t',
                        new_columns=BED_COV_SCHEMA.columns,
                        dtypes=BED_COV_SCHEMA.dtypes_dict,
                        has_header=False, skip_rows=skip_rows).lazy()

    if feature_drop is not None:
        frame = frame.filter(~pl.col(COLUMN_GENOME_ID).is_in(feature_drop))

    if feature_keep is not None:
        frame = frame.filter(pl.col(COLUMN_GENOME_ID).is_in(feature_keep))

    if lazy:
        return frame
    else:
        return frame.collect()


def parse_qiita_coverages(tgzs, *args, **kwargs):
    if not isinstance(tgzs, (list, tuple, set, frozenset)):
        tgzs = [tgzs, ]

    compress_size = kwargs.get('compress_size', 50_000_000)

    if compress_size is not None:
        assert isinstance(compress_size, int) and compress_size >= 0
    else:
        compress_size = math.inf
        kwargs['compress_size'] = compress_size

    frame = _parse_qiita_coverages(tgzs[0], *args, **kwargs)
    for tgz in tgzs[1:]:
        next_frame = _parse_qiita_coverages(tgz, *args, **kwargs)
        frame = _single_df(_check_and_compress([frame, next_frame],
                                               compress_size))

    if compress_size == math.inf:
        return frame
    else:
        return _single_df(_check_and_compress([frame, ], compress_size=0))


def _parse_qiita_coverages(tgz, compress_size=50_000_000, sample_keep=None,
                           sample_drop=None, feature_keep=None,
                           feature_drop=None, append_sample_id=False):
    # compress_size=None to disable compression
    fp = tarfile.open(tgz)

    try:
        fp.extractfile('coverage_percentage.txt')
    except KeyError:
        raise KeyError(f"{tgz} does not look like a Qiita coverage tgz")

    if sample_keep is None:
        sample_keep = SetOfAll()

    if sample_drop is None:
        sample_drop = set()

    coverages = []
    for name in fp.getnames():
        if 'coverages/' not in name:
            continue

        _, filename = name.split('/')
        sample_id = filename.rsplit('.', 1)[0]

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
    if len(coverages) > 1:
        df = pl.concat(coverages)
    elif len(coverages) == 0:
        raise ValueError("No coverages")
    else:
        df = coverages[0]

    return df


def _check_and_compress(coverages, compress_size):
    rowcount = sum([len(df) for df in coverages])
    if rowcount > compress_size:
        df = compress(_single_df(coverages))
        coverages = [df, ]
    return coverages


def _test_has_header(line):
    if isinstance(line, bytes):
        line = line.decode('utf-8')

    genome_id_columns = ('genome-id', 'genome_id', 'feature-id',
                         'feature_id')

    if line.startswith('#'):
        has_header = True
    elif line.split('\t')[0] in genome_id_columns:
        has_header = True
    elif not line.split('\t')[1].strip().isdigit():
        has_header = True
    else:
        has_header = False

    return has_header


def parse_genome_lengths(lengths):
    with open(lengths) as fp:
        first_line = fp.readline()

    has_header = _test_has_header(first_line)
    df = pl.read_csv(lengths, separator='\t', has_header=has_header)
    genome_id_col = df.columns[0]
    length_col = df.columns[1]

    genome_ids = df[genome_id_col]
    if len(genome_ids) != len(set(genome_ids)):
        raise ValueError(f"'{genome_id_col}' is not unique")

    if not df[length_col].dtype.is_integer():
        raise ValueError(f"'{length_col}' is not integer'")

    if df[length_col].min() <= 0:
        raise ValueError(f"Lengths of zero or less cannot be used")

    rename = {genome_id_col: COLUMN_GENOME_ID,
              length_col: COLUMN_LENGTH}
    return df[[genome_id_col, length_col]].rename(rename)


# TODO: this is not the greatest method name
def parse_sam_to_df(sam):
    df = pl.read_csv(sam, separator='\t', has_header=False,
                     columns=SAM_SUBSET_SCHEMA.column_indices,
                     comment_prefix='@',
                     new_columns=SAM_SUBSET_SCHEMA.columns).lazy()

    return (df
             .with_columns(stop=pl.col(COLUMN_CIGAR).map_elements(cigar_to_lens))
             .with_columns(stop=pl.col(COLUMN_STOP) + pl.col(COLUMN_START))
             .collect())


def _add_file(tf, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = int(time.time())
    tf.addfile(ti, io.BytesIO(data))


def write_qiita_cov(name, paths, lengths):
    tf = tarfile.open(name, "w:gz")

    coverages = []
    for p in paths:
        with open(p, 'rb') as fp:
            data = fp.read()

        if len(data) == 0:
            continue

        base = os.path.basename(p)
        if base.endswith('.cov.gz'):
            data = gzip.decompress(data)
            name = base.rsplit('.', 2)[0] + '.cov'
        elif base.endswith('.cov'):
            name = base
        else:
            name = base + '.cov'

        name = f'coverages/{name}'

        _add_file(tf, name, data)
        current_coverage = _parse_bed_cov(io.BytesIO(data), None, None, False)
        coverages.append(current_coverage)
        coverages = _check_and_compress(coverages, compress_size=50_000_000)

    coverage = _single_df(_check_and_compress(coverages, compress_size=0))

    covdataname = 'artifact.cov'
    covdata = io.BytesIO()
    coverage.write_csv(covdata, separator='\t', include_header=True)
    covdata.seek(0)
    _add_file(tf, covdataname, covdata.read())

    genome_coverage = coverage_percent(coverage, lengths).collect()
    pername = 'coverage_percentage.txt'
    perdata = io.BytesIO()
    genome_coverage.write_csv(perdata, separator='\t', include_header=True)
    perdata.seek(0)
    _add_file(tf, pername, perdata.read())

    tf.close()


def parse_sample_metadata(path):
    df = pl.read_csv(path, separator='\t', infer_schema_length=0)
    return df.rename({df.columns[0]: COLUMN_SAMPLE_ID})


@contextmanager
def _reader(sam):
    """Indirection to support reading from stdin or a file."""
    if sam == '-' or sam is None:
        data = sys.stdin.buffer
        yield data
    elif isinstance(sam, io.BytesIO):
        yield sam
    else:
        with lzma.open(sam) as fp:
            data = fp.read()
            yield data


def _buf_to_bytes(buf):
    return io.BytesIO(b''.join(buf))


def _subset_sam_to_bed(df):
    return df[list(BED_COV_SCHEMA.columns)]


def compress_from_stream(sam, bufsize=1_000_000_000, disable_compression=False):
    if disable_compression:
        compress_f = _subset_sam_to_bed
    else:
        compress_f = compress

    current_df = pl.DataFrame([], schema=BED_COV_SCHEMA.dtypes_flat)
    with _reader(sam) as data:
        buf = data.readlines(bufsize)

        if len(buf) == 0:
            return None

        while len(buf) > 0:
            next_df = compress_f(parse_sam_to_df(_buf_to_bytes(buf)))
            current_df = compress_f(pl.concat([current_df, next_df]))
            buf = data.readlines(bufsize)

    return current_df
