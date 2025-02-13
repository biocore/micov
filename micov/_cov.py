import numba
import polars as pl

from ._constants import (
    BED_COV_SCHEMA,
    COLUMN_COVERED,
    COLUMN_COVERED_DTYPE,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_LENGTH_DTYPE,
    COLUMN_PERCENT_COVERED,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)


def coverage_percent_per_sample(coverages, lengths):
    """Compute coverage percent per sample."""
    frames = []
    for (sample,), sample_df in coverages.group_by(
        [
            COLUMN_SAMPLE_ID,
        ]
    ):
        cov = coverage_percent(sample_df, lengths)
        cov = cov.with_columns(pl.lit(sample).alias(COLUMN_SAMPLE_ID))
        frames.append(cov)

    if frames:
        return pl.concat(frames).collect()
    else:
        return pl.DataFrame()


def coverage_percent(coverages, lengths):
    """Compute the percent coverage per genome.

    Parameters
    ----------
    coverages : pl.DataFrame
        Compressed covered region data
    lengths : pl.DataFrame
        The corresponding genome lengths

    Returns
    -------
    pl.LazyFrame
        The genome coverages

    """
    missing = set(coverages[COLUMN_GENOME_ID]) - set(lengths[COLUMN_GENOME_ID])
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} genome(s) appear unrepresented in "
            f"the length information, examples: "
            f"{sorted(missing)[:5]}"
        )

    return (
        coverages.lazy()
        .with_columns(
            (pl.col(COLUMN_STOP) - pl.col(COLUMN_START)).alias(COLUMN_COVERED)
        )
        .group_by(
            [
                COLUMN_GENOME_ID,
            ]
        )
        .agg(pl.col(COLUMN_COVERED).sum())
        .join(lengths.lazy(), on=COLUMN_GENOME_ID)
        .with_columns(
            ((pl.col(COLUMN_COVERED) / pl.col(COLUMN_LENGTH)) * 100).alias(
                COLUMN_PERCENT_COVERED
            )
        )
    )


# TODO: replace compression logic with a duckdb query
# we should obtain benefit of parallelization natively
# code was generated and then minorly adapted using chatgpt 4o
# by firsts providing the "_compress" function below and requesting
# it be expressed as duckdb compatible SQL. The resulting code
# passes the current unit tests (without having provided them)
# NOTE: operation _including_ sample_id has only been loosely checked
# NOTE: this needs to be checked on _large_ data. while the query
#   engine is good, it is not perfect and could trigger large
#   use of tmp
#
# WITH sorted_ranges AS (
#     SELECT
#         *,
#         LAG(stop) OVER (ORDER BY start) AS prev_stop
#     FROM ranges
# ),
# grouped_ranges AS (
#     SELECT
#         *,
#         CASE
#             WHEN prev_stop IS NULL OR start > prev_stop THEN 1
#             ELSE 0
#         END AS new_group_flag
#     FROM sorted_ranges
# ),
# cumulative_groups AS (
#     SELECT
#         *,
#         SUM(new_group_flag) OVER (ORDER BY start
#                                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
#                                  AS group_id
#     FROM grouped_ranges
# )
# SELECT
#     sample_id,
#     genome_id,
#     MIN(start) AS merged_start,
#     MAX(stop) AS merged_stop
# FROM cumulative_groups
# GROUP BY sample_id, genome_id, group_id
# ORDER BY sample_id, genome_id, merged_start;
@numba.jit(nopython=True)
def _compress(rows):
    # derived from zebra
    # https://github.com/biocore/zebra_filter/blob/master/cover.py#L14

    new_ranges = []
    start_val = None
    end_val = None

    # case 1: no active range, start active range.
    start_val, end_val = rows[0]
    for start, stop in rows[1:]:
        if end_val >= start:
            # case 2: active range continues through this range
            # extend active range
            end_val = max(end_val, stop)
        else:  # if end_val < r[0] - 1:
            # case 3: active range ends before this range begins
            # write new range out, then start new active range
            new_range = (start_val, end_val)
            new_ranges.append(new_range)
            start_val = start
            end_val = stop

    if end_val is not None:
        new_range = (start_val, end_val)
        new_ranges.append(new_range)

    return new_ranges


def compress_per_sample(df):
    """Compress data per sample."""
    frames = []
    for (sample,), sample_df in df.group_by(
        [
            COLUMN_SAMPLE_ID,
        ]
    ):
        compressed = compress(sample_df).with_columns(
            pl.lit(sample).alias(COLUMN_SAMPLE_ID)
        )
        frames.append(compressed)

    if frames:
        return pl.concat(frames)
    else:
        return pl.DataFrame([], schema=df.collect_schema())


def compress(df):
    """Compress overlapping intervals into contiguous intervals.

    Parameters
    ----------
    df : pl.DataFrame:
        Genome regions covered

    Notes
    -----
    Intervals are start inclusive and stop exclusive. We currently require
    at least one position of overlap to collapse an interval. The purpose of
    the collapse is to reduce the number of regions tracked to reduce the
    amount of memory to describe coverage.

    1) intervals which overlap are collapsed

        [1, 10) and [5, 15) become [1, 15)

    2) intervals which are nested are collapsed

        [1, 10) and [5, 8) become [1, 10)

    3) immediately adjacent intervals are not collapsed

        [1, 10) and [10, 20) remain unchanged

    A visual depiction:

    123456789012345678901234567890
    ---
       ----
        --
            ---
                 ----
                   -----

                         ---
                           ---
                              ---

    Would reduce to

    123456789012345678901234567890
    -------
            ---
                 -------
                         -----
                              ---

    Returns
    -------
    pl.DataFrame
        The covered genome regions such that overlapping regions, and regions
        represented by another region, are described by a single interval.

    """

    def make_frame(data, genome):
        frame = pl.LazyFrame(
            data,
            schema=[BED_COV_SCHEMA.dtypes_flat[1], BED_COV_SCHEMA.dtypes_flat[2]],
            orient="row",
        )
        return (
            frame.with_columns(pl.lit(genome).cast(str).alias(COLUMN_GENOME_ID))
            .select(BED_COV_SCHEMA.columns)
            .collect()
        )

    compressed = []
    for (genome,), grp in df.group_by(
        [
            COLUMN_GENOME_ID,
        ]
    ):
        rows = (
            grp.lazy()
            .select([COLUMN_START, COLUMN_STOP])
            .sort(COLUMN_START)
            .collect()
            .to_numpy(order="c")
        )

        grp_compressed = _compress(rows)
        grp_compressed_df = make_frame(grp_compressed, genome)
        compressed.append(grp_compressed_df)

    if not compressed:
        return make_frame([], None)
    else:
        return pl.concat(compressed)


def ordered_coverage(coverage, grp, target, length):
    """Gather coverage information and order based on total coverage.

    Parameters
    ----------
    coverage : pl.DataFrame
        A frame that describes the per sample per genome coverage.
    grp : pl.DataFrame
        Sample metadata
    target : str
        The target genome to gather coverage against.
    length :int
        The length of the genome

    Notes
    -----
    Samples in `grp` which we do not have coverage of for the `target` will
    implicitly be remarked as having a 0.0 reported coverage

    Returns
    -------
    pl.DataFrame
        The coverage data sorted by coverage, and augmented with rank values

    """
    coverage = coverage.lazy()
    grp = grp.lazy()

    on_target = (
        coverage.join(grp, on=COLUMN_SAMPLE_ID)
        .filter(pl.col(COLUMN_GENOME_ID) == target)
        .collect()
    )

    on_target_sids = on_target[COLUMN_SAMPLE_ID]

    off_target = (
        grp.filter(~(pl.col(COLUMN_SAMPLE_ID).is_in(on_target_sids)))
        .with_columns(
            pl.lit(0.0).alias(COLUMN_PERCENT_COVERED),
            pl.lit(0).cast(COLUMN_COVERED_DTYPE).alias(COLUMN_COVERED),
            pl.lit(length).cast(COLUMN_LENGTH_DTYPE).alias(COLUMN_LENGTH),
            pl.lit(target).alias(COLUMN_GENOME_ID),
        )
        .select(on_target.columns)
    )

    return (
        pl.concat([on_target.lazy(), off_target])
        .sort(COLUMN_PERCENT_COVERED)
        .with_row_index()
        .with_columns(x=pl.col("index") / pl.len(), x_unscaled=pl.col("index"))
        .drop(pl.col("index"))
        .collect()
    )


def slice_positions(positions, id_):
    """Obtain the genome positions for a sample.

    Parameters
    ----------
    positions : pl.DataFrame
        The per sample per genome covered regions
    id_ : str
        The sample ID to constrain

    Returns
    -------
    pl.LazyFrame
        The subset of positions

    """
    return (
        positions.lazy()
        .filter(pl.col(COLUMN_SAMPLE_ID) == id_)
        .select(pl.col(COLUMN_GENOME_ID), pl.col(COLUMN_START), pl.col(COLUMN_STOP))
    )


def compute_cumulative(coverage, grp, target, target_positions, lengths):
    """Accumulate coverage, from samples with the least to most coverage.

    Parameters
    ----------
    coverage : pl.DataFrame
        The total per sample per target coverage data
    grp : pl.DataFrame
        Sample metadata
    target : str
        The target genome to accumulae coverage over
    target_positions : pl.DataFrame
        The per sample per target regions covered
    lengths : pl.DataFrame
        Per target lengths

    Notes
    -----
    The general approach is to stack all regions covered from samples
    [x, ..., x_n], compress and calculate coverage. This is repeated with
    [x, ..., x_n, x_n + 1].

    """
    length = lengths[COLUMN_LENGTH].item(0)

    current = pl.DataFrame([], schema=BED_COV_SCHEMA.dtypes_flat)
    grp_coverage = ordered_coverage(coverage, grp, target, length)

    if len(grp_coverage) == 0:
        return None, None

    cur_y = []
    cur_x = grp_coverage["x_unscaled"]
    for id_ in grp_coverage[COLUMN_SAMPLE_ID]:
        next_ = slice_positions(target_positions, id_).collect()
        current = compress(pl.concat([current, next_]))
        per_cov = coverage_percent(current, lengths).collect()

        # no observed coverage can occur in the unfocused monte carlo simulation
        # in which case the coverage is zero
        if len(per_cov) == 0:
            val = 0.0
        else:
            val = per_cov[COLUMN_PERCENT_COVERED].item(0)

        cur_y.append(val)
    return cur_x, cur_y


@numba.jit(nopython=True)
def get_covered(x_start_stop):
    """Remap (x, y1, y1) into [(x, y1), (x, y2)]."""
    return [[(x, start), (x, stop)] for (x, start, stop) in x_start_stop]
