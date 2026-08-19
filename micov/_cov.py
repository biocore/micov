import numpy as np

from ._constants import (
    COLUMN_COVERED,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_PERCENT_COVERED,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)


def mask_table(table, keep):
    """Select rows of a table with a boolean mask or an array of row indices.

    A "table" here is a `dict` of equal length numpy arrays -- the shape
    `DuckDBPyRelation.fetchnumpy()` hands back, and what the plotting path
    carries now that it holds no polars frames.
    """
    return {column: values[keep] for column, values in table.items()}


def ordered_coverage(coverage, grp, target, length):
    """Gather coverage information and order based on total coverage.

    Parameters
    ----------
    coverage : dict of np.ndarray
        A table that describes the per sample per genome coverage.
    grp : dict of np.ndarray
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
    dict of np.ndarray
        The coverage data sorted by coverage, and augmented with rank values

    """
    grp_samples = grp[COLUMN_SAMPLE_ID]
    on_target = mask_table(
        coverage,
        (coverage[COLUMN_GENOME_ID] == target)
        & np.isin(coverage[COLUMN_SAMPLE_ID], grp_samples),
    )

    # the join this replaces carried `grp`'s columns onto each covered sample,
    # so index back into `grp` rather than dropping them
    grp_row = {sample: row for row, sample in enumerate(grp_samples)}
    on_rows = np.array(
        [grp_row[sample] for sample in on_target[COLUMN_SAMPLE_ID]], dtype=np.intp
    )
    covered_samples = set(on_target[COLUMN_SAMPLE_ID].tolist())
    off_rows = np.array(
        [
            row
            for row, sample in enumerate(grp_samples)
            if sample not in covered_samples
        ],
        dtype=np.intp,
    )

    fill = {
        COLUMN_GENOME_ID: target,
        COLUMN_COVERED: 0,
        COLUMN_LENGTH: length,
        COLUMN_PERCENT_COVERED: 0.0,
    }

    ordered = {}
    for column, values in on_target.items():
        if column == COLUMN_SAMPLE_ID:
            absent = grp_samples[off_rows]
        else:
            # KeyError rather than a silent gap if `coverage` grows a column
            absent = np.full(len(off_rows), fill[column], dtype=values.dtype)
        ordered[column] = np.concatenate([values, absent])

    for column, values in grp.items():
        if column != COLUMN_SAMPLE_ID:
            ordered[column] = np.concatenate([values[on_rows], values[off_rows]])

    # stable, so samples tied on coverage keep covered-then-absent order. That
    # is what the polars sort did, and the plot goldens were frozen with it.
    ordered = mask_table(
        ordered, np.argsort(ordered[COLUMN_PERCENT_COVERED], kind="stable")
    )

    n = len(ordered[COLUMN_SAMPLE_ID])
    ordered["x_unscaled"] = np.arange(n, dtype=np.uint64)
    ordered["x"] = np.arange(n, dtype=np.float64) / n
    return ordered


def merge_intervals(starts, stops):
    """Merge overlapping, nested and touching intervals.

    The numpy counterpart of `compress`, for the accumulation path where
    intervals arrive as arrays rather than as a frame. Semantics are identical,
    including that *touching* intervals collapse -- [400, 500) and [500, 505)
    become [400, 505) -- which is why a new interval opens on `>` and not `>=`.

    Parameters
    ----------
    starts, stops : np.ndarray
        Interval bounds, 1-based half-open, in any order.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The merged bounds, ordered by start.

    """
    if len(starts) == 0:
        return starts, stops

    order = np.lexsort((stops, starts))
    starts = np.asarray(starts)[order]
    stops = np.asarray(stops)[order]

    # an interval opens a new merged run only when it begins strictly beyond
    # every stop seen so far, so a run absorbs nested intervals too
    highest_stop = np.maximum.accumulate(stops)
    opens = np.empty(len(starts), dtype=bool)
    opens[0] = True
    opens[1:] = starts[1:] > highest_stop[:-1]

    heads = np.flatnonzero(opens)
    return starts[heads], np.maximum.reduceat(stops, heads)


def slice_positions(positions, id_):
    """Obtain the genome positions for a sample.

    Parameters
    ----------
    positions : dict of np.ndarray
        The per sample per genome covered regions
    id_ : str
        The sample ID to constrain

    Returns
    -------
    dict of np.ndarray
        The subset of positions

    """
    keep = positions[COLUMN_SAMPLE_ID] == id_
    return {
        column: positions[column][keep]
        for column in (COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP)
    }


def compute_cumulative(coverage, grp, target, target_positions, lengths):
    """Accumulate coverage, from samples with the least to most coverage.

    Parameters
    ----------
    coverage : dict of np.ndarray
        The total per sample per target coverage data
    grp : dict of np.ndarray
        Sample metadata
    target : str
        The target genome to accumulae coverage over
    target_positions : dict of np.ndarray
        The per sample per target regions covered
    lengths : dict of np.ndarray
        Per target lengths

    Notes
    -----
    The general approach is to stack all regions covered from samples
    [x, ..., x_n], compress and calculate coverage. This is repeated with
    [x, ..., x_n, x_n + 1].

    """
    # n.b. row 0's length, not the target's. That is what the polars
    # implementation passed, and it only fills the `length` column of samples
    # with no coverage, which nothing reads. Preserved rather than quietly
    # corrected; recorded in MIGRATE-TO-MIINT.md section 8.
    length = lengths[COLUMN_LENGTH][0]

    grp_coverage = ordered_coverage(coverage, grp, target, length)

    if len(grp_coverage[COLUMN_SAMPLE_ID]) == 0:
        return None, None

    # the percent is against the target's own length, which is what joining
    # the accumulated intervals to `lengths` on genome_id used to supply
    target_length = lengths[COLUMN_LENGTH][lengths[COLUMN_GENOME_ID] == target][0]

    starts = target_positions[COLUMN_START][:0]
    stops = target_positions[COLUMN_STOP][:0]

    cur_y = []
    cur_x = grp_coverage["x_unscaled"]
    for id_ in grp_coverage[COLUMN_SAMPLE_ID]:
        next_ = slice_positions(target_positions, id_)
        starts, stops = merge_intervals(
            np.concatenate([starts, next_[COLUMN_START]]),
            np.concatenate([stops, next_[COLUMN_STOP]]),
        )

        # no observed coverage can occur in the unfocused monte carlo simulation
        # in which case the coverage is zero
        covered = int((stops - starts).sum())
        cur_y.append((covered / target_length) * 100)
    return cur_x, cur_y


def get_covered(x_start_stop):
    """Remap (x, y1, y1) into [(x, y1), (x, y2)]."""
    return [[(x, start), (x, stop)] for (x, start, stop) in x_start_stop]
