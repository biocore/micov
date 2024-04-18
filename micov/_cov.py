import numba
import polars as pl
from ._constants import (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP,
                         COLUMN_COVERED, BED_COV_SCHEMA,
                         COLUMN_LENGTH, COLUMN_PERCENT_COVERED)


def coverage_percent(coverages, lengths):
    missing = (set(coverages[COLUMN_GENOME_ID]) -
               set(lengths[COLUMN_GENOME_ID]))
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} genome(s) appear unrepresented in "
                         f"the length information, examples: "
                         f"{sorted(missing)[:5]}")

    return (coverages
               .lazy()
               .with_columns((pl.col(COLUMN_STOP) -
                              pl.col(COLUMN_START)).alias(COLUMN_COVERED))
               .group_by([COLUMN_GENOME_ID, ])
               .agg(pl.col(COLUMN_COVERED).sum())
               .join(lengths.lazy(), on=COLUMN_GENOME_ID)
               .with_columns(((pl.col(COLUMN_COVERED) /
                               pl.col(COLUMN_LENGTH)) * 100).alias(COLUMN_PERCENT_COVERED)))  # noqa


@numba.jit(nopython=True)
def _compress(genome, rows):
    # derived from zebra
    # https://github.com/biocore/zebra_filter/blob/master/cover.py#L14

    new_ranges = []
    start_val = None
    end_val = None

    for start, stop in rows:
        if end_val is None:
            # case 1: no active range, start active range.
            start_val = start
            end_val = stop
        elif end_val >= start:
            # case 2: active range continues through this range
            # extend active range
            end_val = max(end_val, stop)
        else:  # if end_val < r[0] - 1:
            # case 3: active range ends before this range begins
            # write new range out, then start new active range
            new_range = (genome, start_val, end_val)
            new_ranges.append(new_range)
            start_val = start
            end_val = stop

    if end_val is not None:
        new_range = (genome, start_val, end_val)
        new_ranges.append(new_range)

    return new_ranges


def compress(df):
    compressed = []
    for (genome, ), grp in df.group_by([COLUMN_GENOME_ID, ]):
        rows = (grp
                 .lazy()
                 .select([COLUMN_START, COLUMN_STOP])
                 .sort(COLUMN_START)
                 .collect()
                 .to_numpy())
        grp_compressed = _compress(genome, rows)
        grp_compressed_df = pl.DataFrame(grp_compressed,
                                         schema=BED_COV_SCHEMA.dtypes_flat)
        compressed.append(grp_compressed_df)

    return pl.concat(compressed)
