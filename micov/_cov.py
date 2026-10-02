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

    # ties on coverage are broken by sample_id, never by row order: the rows
    # come from a parallel Parquet scan whose order changes run to run (R20a)
    ordered = mask_table(
        ordered,
        np.lexsort((ordered[COLUMN_SAMPLE_ID], ordered[COLUMN_PERCENT_COVERED])),
    )

    n = len(ordered[COLUMN_SAMPLE_ID])
    ordered["x_unscaled"] = np.arange(n, dtype=np.uint64)
    ordered["x"] = np.arange(n, dtype=np.float64) / n
    return ordered


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


#: Name the accumulation input is registered under. `View`'s catalog already
#: holds `positions`, `coverage`, `metadata` and `regions`, and `_cov` may be
#: handed that same connection, so this is deliberately prefixed.
CURVE_INPUT_RELATION = "micov_curve_input"


def cumulative_covered(con, n_iterations, n_ranks, iterations, ranks, starts, stops):
    """Accumulate covered bases per rank, for one or more iterations.

    Ranks accumulate from 0 upward: the value at rank *k* is the breadth of
    every interval belonging to ranks 0..k merged together, which is what makes
    the curve a curve rather than a sorted list of per-sample breadths.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        A connection carrying the miint extension.
    n_iterations, n_ranks : int
        The shape of the result. Every (iteration, rank) pair is accounted for
        whether or not any interval was supplied for it.
    iterations, ranks, starts, stops : np.ndarray
        One row per covered interval, concatenated across iterations.

    Returns
    -------
    np.ndarray
        Covered base counts, shape `(n_iterations, n_ranks)`.

    Notes
    -----
    The roster is generated in SQL and LEFT JOINed to the intervals rather than
    supplied from `ranks`, so a sample with no coverage still gets its rank.
    miint requires ranks to be contiguous `0..N-1` and a sample contributing no
    intervals would otherwise leave a hole -- and, more importantly, silently
    shorten the curve.

    """
    con.register(
        CURVE_INPUT_RELATION,
        {
            "iteration": iterations,
            "rank": ranks,
            COLUMN_START: starts,
            COLUMN_STOP: stops,
        },
    )
    try:
        rows = con.sql(f"""
            SELECT iteration, c.rank AS rank, c.covered AS covered
            FROM (
                SELECT r.iteration AS iteration,
                       UNNEST(cumulative_coverage(r.rank::INTEGER,
                                                  p.{COLUMN_START},
                                                  p.{COLUMN_STOP})) AS c
                FROM (SELECT i.iteration, k.rank
                      FROM range(0, {n_iterations}) i(iteration),
                           range(0, {n_ranks}) k(rank)) r
                    LEFT JOIN {CURVE_INPUT_RELATION} p
                        ON p.iteration = r.iteration AND p.rank = r.rank
                GROUP BY r.iteration
            )
            ORDER BY iteration, rank
        """).fetchall()
    finally:
        con.unregister(CURVE_INPUT_RELATION)

    # ordered by (iteration, rank) above, so the reshape is positional. Asked
    # for explicitly rather than trusting UNNEST to preserve list order.
    covered = np.array([row[2] for row in rows], dtype=np.int64)
    return covered.reshape(n_iterations, n_ranks)


def _rank_intervals(grp_coverage, target_positions):
    """Map each interval onto the rank of the sample it belongs to.

    Intervals for samples outside `grp_coverage` are dropped; samples in it
    with no intervals simply contribute none, and get their rank from the
    roster in `cumulative_covered`.
    """
    rank_of = dict(
        zip(
            grp_coverage[COLUMN_SAMPLE_ID],
            grp_coverage["x_unscaled"],
            strict=True,
        )
    )
    keep = np.array(
        [sample in rank_of for sample in target_positions[COLUMN_SAMPLE_ID]],
        dtype=bool,
    )
    ranks = np.array(
        [rank_of[sample] for sample in target_positions[COLUMN_SAMPLE_ID][keep]],
        dtype=np.int32,
    )
    return ranks, target_positions[COLUMN_START][keep], \
        target_positions[COLUMN_STOP][keep]


def cumulative_curves(con, coverage, groups, target, target_positions, lengths):
    """Accumulate a cumulative coverage curve for each of several groups.

    Every group is ranked independently and all of them accumulate in a single
    aggregate call, which is what makes Monte Carlo affordable: the simulation
    runs hundreds of iterations, and each was previously its own O(n^2) pass.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        A connection carrying the miint extension.
    coverage : dict of np.ndarray
        The total per sample per target coverage data
    groups : list of np.ndarray
        Sample IDs per group. Groups must be the same size as each other.
    target : str
        The target genome to accumulate coverage over
    target_positions : dict of np.ndarray
        The per sample per target regions covered
    lengths : dict of np.ndarray
        Per target lengths

    Returns
    -------
    (list of dict, np.ndarray)
        The ranked coverage per group, and a `(len(groups), n)` array of
        percentages -- one curve per group.

    """
    target_length = lengths[COLUMN_LENGTH][lengths[COLUMN_GENOME_ID] == target][0]

    ordered = [
        ordered_coverage(coverage, {COLUMN_SAMPLE_ID: samples}, target, target_length)
        for samples in groups
    ]
    sizes = {len(grp_coverage[COLUMN_SAMPLE_ID]) for grp_coverage in ordered}
    if len(sizes) > 1:
        # the roster is generated once, for one width, so a group of a
        # different size would be silently truncated or silently padded flat
        # rather than raising. Both callers pass equal-sized groups today.
        raise ValueError(f"Groups must be the same size, got sizes {sorted(sizes)}")

    n_ranks = sizes.pop()
    if n_ranks == 0:
        return ordered, np.empty((len(groups), 0), dtype=np.float64)

    iterations, ranks, starts, stops = [], [], [], []
    for iteration, grp_coverage in enumerate(ordered):
        grp_ranks, grp_starts, grp_stops = _rank_intervals(
            grp_coverage, target_positions
        )
        iterations.append(np.full(len(grp_ranks), iteration, dtype=np.int32))
        ranks.append(grp_ranks)
        starts.append(grp_starts)
        stops.append(grp_stops)

    covered = cumulative_covered(
        con,
        len(groups),
        n_ranks,
        np.concatenate(iterations),
        np.concatenate(ranks),
        np.concatenate(starts),
        np.concatenate(stops),
    )

    # `(covered / length) * 100`, not `covered * 100 / length`: the two are
    # different doubles, and the frozen plot goldens carry the former.
    return ordered, (covered / target_length) * 100


def compute_cumulative(con, coverage, grp, target, target_positions, lengths):
    """Accumulate coverage, from samples with the least to most coverage.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        A connection carrying the miint extension.
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
    The accumulation is miint's `cumulative_coverage` aggregate. It replaced a
    Python loop that re-merged a growing interval set once per sample, which
    was O(n^2) in group size -- 13.2s for 1000 samples against the aggregate's
    0.5s, and bit-identical.

    micov supplies the rank itself, from `ordered_coverage`, rather than
    using miint's `cumulative_coverage_curve` macro. Both break breadth ties
    by `sample_id`; whether the macro could replace the ranking outright has
    not been checked.

    """
    ordered, curves = cumulative_curves(
        con, coverage, [grp[COLUMN_SAMPLE_ID]], target, target_positions, lengths
    )

    if len(ordered[0][COLUMN_SAMPLE_ID]) == 0:
        return None, None

    # a list of np.float64, which is what the loop this replaced returned
    return ordered[0]["x_unscaled"], list(curves[0])


def get_covered(x_start_stop):
    """Remap (x, y1, y1) into [(x, y1), (x, y2)]."""
    return [[(x, start), (x, stop)] for (x, start, stop) in x_start_stop]
