from ._constants import (
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)


def bin_list_sql(lengths, bin_num):
    """Build the SQL for every genome's bins: genome_id, bin_idx, bin/start/stop.

    Reproduces the table `pl.Series([0, length]).hist(bin_count=n)` produced:
    breakpoints at ``i * (length / n)`` for ``i`` in 1..n, rounded **half away
    from zero** -- which for these non-negative values is ``floor(x + 0.5)``.
    Both the breakpoint formula and the rounding mode were checked against
    polars directly rather than read off its documentation; numpy rounds half
    to even and would shift a bin edge by one base.

    Bins are 1-indexed because `bin_idx` is a frozen output column. The first
    bin starts at 0, and the last stops at ``length + 1`` so a read ending on
    the final base still falls inside a bin.

    Parameters
    ----------
    lengths : str
        A SQL source yielding one `genome_id` and `length` per genome.
    bin_num : int
        The number of bins to divide each genome into.

    """
    length = f"lengths.{COLUMN_LENGTH}"
    width = f"(CAST({length} AS DOUBLE) / CAST({bin_num} AS DOUBLE))"

    def breakpoint_at(index):
        return f"CAST(floor(CAST({index} AS DOUBLE) * {width} + 0.5) AS UINTEGER)"

    return f"""
        SELECT lengths.{COLUMN_GENOME_ID},
               CAST(i AS UINTEGER) AS bin_idx,
               CASE WHEN i = 1
                    THEN CAST(0 AS UINTEGER)
                    ELSE {breakpoint_at('i - 1')}
               END AS bin_start,
               CASE WHEN i = {bin_num}
                    THEN CAST({length} + 1 AS UINTEGER)
                    ELSE {breakpoint_at('i')}
               END AS bin_stop
        FROM ({lengths}) lengths, range(1, {bin_num} + 1) t(i)
    """


def create_bin_list(con, genome_length, bin_num):
    """Materialize the bins for a single genome length."""
    lengths = (
        f"SELECT 'genome' AS {COLUMN_GENOME_ID}, "
        f"{genome_length} AS {COLUMN_LENGTH}"
    )
    return con.sql(
        f"SELECT bin_idx, bin_start, bin_stop "
        f"FROM ({bin_list_sql(lengths, bin_num)})"
    )


def pos_to_bins(pos, lengths, variable, bin_num):
    """Build the SQL counting read and sample hits per bin.

    Parameters
    ----------
    pos : str
        A SQL source yielding covered positions already joined to the sample
        metadata: `genome_id`, `start`, `stop`, `sample_id` and `variable`.
    lengths : str
        A SQL source yielding one `genome_id` and `length` per genome.
    variable : str
        The metadata column to stratify on.
    bin_num : int
        The number of bins to divide each genome into.

    Notes
    -----
    A read is counted in every bin it spans, so `read_hits` summed over bins
    exceeds the number of reads. The bins a read spans are those its interval
    overlaps: `bin_stop > start AND bin_start < stop`. Bins are contiguous and
    intervals are half open, so this is exactly the inclusive run of bins from
    the one holding `start` to the one holding `stop` -- a read ending flush on
    a bin boundary does not reach into the next bin.

    Every genome is binned in one statement, against its own bin bounds, so
    the result no longer depends on the order genomes are visited in.

    Returns
    -------
    str
        SQL yielding one row per genome, stratification value and bin.
        `samples` is already rendered as ``[a,b,c]``, the frozen output form.

    """
    return f"""
        SELECT pos.{COLUMN_GENOME_ID},
               pos."{variable}",
               bins.bin_idx,
               COUNT(*) AS read_hits,
               COUNT(DISTINCT pos.{COLUMN_SAMPLE_ID}) AS sample_hits,
               '[' || array_to_string(
                   list_sort(list(DISTINCT pos.{COLUMN_SAMPLE_ID})), ','
               ) || ']' AS samples,
               bins.bin_start,
               bins.bin_stop
        FROM ({pos}) pos
            JOIN ({bin_list_sql(lengths, bin_num)}) bins
                ON bins.{COLUMN_GENOME_ID} = pos.{COLUMN_GENOME_ID}
                    AND bins.bin_stop > pos.{COLUMN_START}
                    AND bins.bin_start < pos.{COLUMN_STOP}
        GROUP BY pos.{COLUMN_GENOME_ID}, pos."{variable}",
                 bins.bin_idx, bins.bin_start, bins.bin_stop
        ORDER BY pos.{COLUMN_GENOME_ID}, bins.bin_idx, pos."{variable}"
    """
