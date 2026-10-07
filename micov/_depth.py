"""Depth, breadth and per-ORF statistics for `micov depth-plot`."""

import numpy as np

from ._constants import (
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)
from ._io import (
    ALIGNED_ROWS,
    DEPTH_FEATURES_TABLE,
    ORFS_TABLE,
    SAMPLE_GROUPS_TABLE,
    _examples,
)
from ._plot import MAX_GROUPS
from ._utils import logger

#: The samples `depth-plot` uses: `sample_id`, a dense `sample_idx` in
#: `sample_id` order, and `group_name`.
ROSTER_TABLE = "depth_roster"

#: The genomes `depth-plot` draws: `genome_id`, `length`, `is_circular`.
GENOMES_TABLE = "depth_genomes"

#: `stage_depth` leaves one genome's depth-layer reads here, by position:
#: `sample_idx`, `position`, `stop_position`, `cigar`.
DEPTH_ALIGNMENTS_TABLE = "depth_alignments"

#: `stage_depth_reads` leaves every plotted genome's depth-layer reads here,
#: by genome and position: the layer's columns, coordinates as UINTEGER.
DEPTH_READS_TABLE = "depth_reads"

#: `stage_breadth` leaves each sample's merged breadth-layer intervals here:
#: `genome_id`, `sample_idx`, `start`, `stop`.
BREADTH_INTERVALS_TABLE = "breadth_intervals"

#: Samples x bases in one window of per-base depth, whatever the genome's
#: length or the number of samples. miint's aggregation peaks at six to seven
#: times the 16 MiB array, so a window costs about 110 MiB; a window 16x
#: larger was no faster on 10 Mb x 300 samples, and peaked at 3 GiB.
WINDOW_CELLS = 2**22

#: The overview wraps a genome into rows of this many bases, so every genome
#: at least this long is drawn at one resolution. A shorter genome, such as a
#: mitochondrion, is one row of its own length (`overview_row_bp`).
ROW_BP = 2_000_000

#: The most bins in a row of the overview: 1 kb bins on a genome of 2 Mb or
#: more, 9 bp on a 16.6 kb mitochondrion (`overview_bin_bp`).
OVERVIEW_ROW_BINS = 2_000

#: The most bins a detail panel has; short regions show single bases.
DETAIL_MAX_BINS = 1_500

#: Depth quantiles drawn per group: the IQR's edges and the median.
QUARTILES = (0.25, 0.5, 0.75)

#: Added to both groups' scaled ORF depth before the contrast's ratio, so an
#: ORF with no depth in one group gives a large, finite contrast.
CONTRAST_PSEUDOCOUNT = 0.05


def _ids(con, sql):
    return {row[0] for row in con.sql(sql).fetchall()}


def _report(ids, what):
    if ids:
        ordered = sorted(ids)
        logger.warning(
            f"{len(ordered)} {what}, so are left out: "
            f"{_examples([(i,) for i in ordered], limit=len(ordered))}"
        )


def intersect_layers(con, depth_view, breadth_view, *, orfs):
    """Settle which samples and genomes `depth-plot` uses.

    A sample is used if it is in the metadata and has an aligned read in both
    layers; a genome, if it is in the features and has an aligned read in both
    layers. Everything the metadata or features name but this leaves out is
    reported by name. What they do not name -- a sample with no metadata, a
    genome not asked for -- was left out by the user, and is not.

    Requires `SAMPLE_GROUPS_TABLE`, `DEPTH_FEATURES_TABLE` and the two layer
    views, and `ORFS_TABLE` if `orfs`. Creates `ROSTER_TABLE` and
    `GENOMES_TABLE`.

    Raises
    ------
    ValueError
        If no sample or no genome is left, if more than `MAX_GROUPS` groups
        are, if an aligned read starts beyond its genome's length, or if
        `orfs` and no ORF is on a genome that is left, or an ORF on one is
        outside it.
    """
    def aligned(view, column):
        return _ids(
            con, f"SELECT DISTINCT {column} FROM {view} WHERE {ALIGNED_ROWS}"
        )

    listed = _ids(con, f"SELECT {COLUMN_SAMPLE_ID} FROM {SAMPLE_GROUPS_TABLE}")
    depth = aligned(depth_view, COLUMN_SAMPLE_ID)
    breadth = aligned(breadth_view, COLUMN_SAMPLE_ID)
    samples = listed & depth & breadth
    if not samples:
        raise ValueError(
            "No sample is in the sample metadata and has alignments in both the "
            f"depth and breadth layers (metadata {len(listed)}, depth "
            f"{len(depth)}, breadth {len(breadth)})."
        )
    _report((listed & depth) - breadth, "sample(s) are in the depth layer only")
    _report((listed & breadth) - depth, "sample(s) are in the breadth layer only")
    _report(listed - depth - breadth, "sample(s) have no alignments in either layer")

    features = _ids(con, f"SELECT {COLUMN_GENOME_ID} FROM {DEPTH_FEATURES_TABLE}")
    depth = aligned(depth_view, "reference")
    breadth = aligned(breadth_view, "reference")
    genomes = features & depth & breadth
    if not genomes:
        raise ValueError(
            "No genome is in the features and has alignments in both the depth "
            f"and breadth layers (features {len(features)}, depth {len(depth)}, "
            f"breadth {len(breadth)})."
        )
    _report((features & depth) - breadth, "genome(s) are in the depth layer only")
    _report((features & breadth) - depth, "genome(s) are in the breadth layer only")
    _report(features - depth - breadth, "genome(s) have no alignments in either layer")

    con.execute(
        f"""CREATE OR REPLACE TABLE {ROSTER_TABLE} AS
            SELECT {COLUMN_SAMPLE_ID},
                   (row_number() OVER (ORDER BY {COLUMN_SAMPLE_ID}) - 1)::INTEGER
                       AS sample_idx,
                   group_name
            FROM {SAMPLE_GROUPS_TABLE}
            WHERE list_contains(?, {COLUMN_SAMPLE_ID})""",
        [sorted(samples)],
    )
    con.execute(
        f"""CREATE OR REPLACE TABLE {GENOMES_TABLE} AS
            SELECT DISTINCT {COLUMN_GENOME_ID}, {COLUMN_LENGTH}, is_circular
            FROM {DEPTH_FEATURES_TABLE}
            WHERE list_contains(?, {COLUMN_GENOME_ID})""",
        [sorted(genomes)],
    )

    groups = sorted(_ids(con, f"SELECT group_name FROM {ROSTER_TABLE}"))
    if len(groups) > MAX_GROUPS:
        raise ValueError(
            f"depth-plot draws at most {MAX_GROUPS} groups, and the metadata "
            f"column has {len(groups)}: {', '.join(groups)}"
        )

    for view in (depth_view, breadth_view):
        rows = con.sql(f"""SELECT a.reference || ' (a read at '
                                      || max(a.position) || ', length '
                                      || g.{COLUMN_LENGTH} || ')'
                           FROM {view} a
                               JOIN {GENOMES_TABLE} g
                                   ON a.reference = g.{COLUMN_GENOME_ID}
                           WHERE {ALIGNED_ROWS}
                               AND a.position > g.{COLUMN_LENGTH}
                           GROUP BY a.reference, g.{COLUMN_LENGTH}
                           ORDER BY 1""").fetchall()
        if rows:
            raise ValueError(
                "An aligned read starts beyond its genome's length, so the "
                f"length in the features is wrong: {_examples(rows)}"
            )

    if orfs:
        on_genomes = con.sql(f"""SELECT count(*) FROM {ORFS_TABLE}
                                 JOIN {GENOMES_TABLE} USING ({COLUMN_GENOME_ID})
                              """).fetchone()[0]
        if on_genomes == 0:
            seqids = con.sql(f"""SELECT DISTINCT {COLUMN_GENOME_ID} FROM {ORFS_TABLE}
                                 ORDER BY 1""").fetchall()
            raise ValueError(
                "No ORF is on a genome being plotted; the ORFs' seqids are "
                f"{_examples(seqids)}. A seqid must equal the genome_id."
            )
        # GFF3 writes an ORF across a circular genome's origin with an end
        # past the length; `orf_segments` splits it there
        rows = con.sql(f"""SELECT o.{COLUMN_GENOME_ID} || ':' || o.orf_id
                                      || ' [' || o.{COLUMN_START} || ', '
                                      || o.{COLUMN_STOP} || '), length '
                                      || g.{COLUMN_LENGTH}
                           FROM {ORFS_TABLE} o
                               JOIN {GENOMES_TABLE} g USING ({COLUMN_GENOME_ID})
                           WHERE o.{COLUMN_START} > g.{COLUMN_LENGTH}
                               OR o.{COLUMN_STOP} - o.{COLUMN_START}
                                   > g.{COLUMN_LENGTH}
                               OR (NOT g.is_circular
                                   AND o.{COLUMN_STOP} > g.{COLUMN_LENGTH} + 1)
                           ORDER BY 1""").fetchall()
        if rows:
            raise ValueError(
                "An ORF must lie within its genome; only a circular genome's "
                "may run past the end, across the origin. These do not: "
                f"{_examples(rows)}"
            )


def stage_depth(con, depth_view, genome_id):
    """Copy one genome's aligned depth-layer reads into a small sorted table.

    Every window is a query over this table, so it holds only what can add
    depth -- aligned reads of the samples being plotted -- in about 28 bytes a
    read, sorted by position so each window's range filter skips most of it.

    Requires `ROSTER_TABLE`. Creates `DEPTH_ALIGNMENTS_TABLE`.
    """
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE {DEPTH_ALIGNMENTS_TABLE} AS
            SELECT r.sample_idx,
                   a.position::UINTEGER AS position,
                   a.stop_position::UINTEGER AS stop_position,
                   a.cigar
            FROM {depth_view} a JOIN {ROSTER_TABLE} r USING ({COLUMN_SAMPLE_ID})
            WHERE a.reference = ? AND {ALIGNED_ROWS}
            ORDER BY a.position""",
        [genome_id],
    )


def stage_depth_reads(con, depth_view):
    """Copy the plotted samples' aligned reads on the plotted genomes, once.

    Stored by genome and position, so each genome's `stage_depth` from this
    table reads only that genome's row groups. Read from the layer itself,
    every genome scans the whole input: at 20M reads that was 42 ms a genome
    against 4 ms from here, and grew with the input.

    Requires `ROSTER_TABLE` and `GENOMES_TABLE`. Creates `DEPTH_READS_TABLE`,
    which stands in for the layer as `genome_statistics`' `depth_view`.
    """
    con.sql(f"""CREATE OR REPLACE TEMP TABLE {DEPTH_READS_TABLE} AS
                SELECT a.{COLUMN_SAMPLE_ID}, a.reference,
                       a.position::UINTEGER AS position,
                       a.stop_position::UINTEGER AS stop_position,
                       a.cigar
                FROM {depth_view} a
                    SEMI JOIN {ROSTER_TABLE} r USING ({COLUMN_SAMPLE_ID})
                    SEMI JOIN {GENOMES_TABLE} g
                        ON a.reference = g.{COLUMN_GENOME_ID}
                WHERE {ALIGNED_ROWS}
                ORDER BY a.reference, a.position""")


def window_depth(con, n, w0, w1):
    """Per-base depth of every sample over the bases [w0, w1).

    Deletions count and skips (N) do not, as `compute_coverage_depth`'s
    ``include_deletions`` mode has it. Reads are shifted so that w0 is base 1
    and the window is the whole "reference": miint then clips reads starting
    before w0 or running past w1, so no window needs padding.

    Requires `DEPTH_ALIGNMENTS_TABLE`.

    Returns
    -------
    np.ndarray
        uint32, `n` x (w1 - w0): row i is sample_idx i, column j base w0 + j.
    """
    depth = np.zeros((n, w1 - w0), dtype=np.uint32)
    # a sample whose reads miint all skips (a NULL CIGAR) aggregates to NULL
    rows = con.execute(
        f"""SELECT sample_idx, d FROM (
                SELECT sample_idx,
                       compute_coverage_depth(position::BIGINT - ?,
                                              stop_position::BIGINT - ?,
                                              cigar, ?, ?) AS d
                FROM {DEPTH_ALIGNMENTS_TABLE}
                WHERE position < ? AND stop_position > ?
                GROUP BY sample_idx)
            WHERE d IS NOT NULL""",
        [w0 - 1, w0 - 1, w1 - w0, "include_deletions", w1, w0],
    ).fetchnumpy()
    for i, d in zip(rows["sample_idx"], rows["d"], strict=True):
        depth[i] = d
    return depth


def window_size(n, length):
    """How many bases a window of `n` samples' depth spans."""
    return max(1, min(length, WINDOW_CELLS // n))


def stage_breadth(con, breadth_view):
    """Merge each plotted sample's breadth-layer reads into intervals.

    Merged, so a sample counts once toward prevalence however many of its
    reads overlap a base. A read's whole span counts, skips included: breadth
    is where reads align, depth where their bases are.

    Requires `ROSTER_TABLE` and `GENOMES_TABLE`. Creates
    `BREADTH_INTERVALS_TABLE`.
    """
    con.sql(f"""CREATE OR REPLACE TEMP TABLE {BREADTH_INTERVALS_TABLE} AS
                SELECT {COLUMN_GENOME_ID}, sample_idx,
                       interval.start::UINTEGER AS {COLUMN_START},
                       interval.stop::UINTEGER AS {COLUMN_STOP}
                FROM (SELECT a.reference AS {COLUMN_GENOME_ID}, r.sample_idx,
                             UNNEST(compress_intervals(a.position,
                                                       a.stop_position))
                                 AS interval
                      FROM {breadth_view} a
                          JOIN {ROSTER_TABLE} r USING ({COLUMN_SAMPLE_ID})
                          JOIN {GENOMES_TABLE} g
                              ON a.reference = g.{COLUMN_GENOME_ID}
                      WHERE {ALIGNED_ROWS}
                      GROUP BY a.reference, r.sample_idx)
                ORDER BY ALL""")


def coverage_counts(starts, stops, w0, w1):
    """How many of the intervals [start, stop) cover each base of [w0, w1).

    `starts` and `stops` must each be sorted, and need not stay paired: base
    x is covered by every interval starting at or before x, less those that
    have also stopped by then. Each window then costs a binary search per
    base, however many intervals the genome has.
    """
    bases = np.arange(w0, w1)
    return (np.searchsorted(starts, bases, side="right")
            - np.searchsorted(stops, bases, side="right"))


def quartiles_x4(depth):
    """Four times the `QUARTILES` of each column (base) across rows (samples).

    numpy's default, linear, method on integers lands on multiples of 0.25,
    so four times it is an exact integer. Bins are then integer sums divided
    once, identical however the bases were split into windows.

    Returns
    -------
    np.ndarray
        int64, 3 x columns.
    """
    return (np.quantile(depth, QUARTILES, axis=0) * 4).astype(np.int64)


def display_bin_edges(start, stop, bin_bp):
    """Edges of `bin_bp` bins over [start, stop); the last may be shorter."""
    return np.append(np.arange(start, stop, bin_bp), stop)


def detail_bin_bp(start, stop):
    """Return the narrowest bin keeping [start, stop) to `DETAIL_MAX_BINS`."""
    return max(1, -(-(stop - start) // DETAIL_MAX_BINS))


def overview_row_bp(length):
    """Return the overview's row width: the whole genome, up to `ROW_BP`."""
    return min(length, ROW_BP)


def overview_bin_bp(length):
    """Return the narrowest bin keeping a row to `OVERVIEW_ROW_BINS`."""
    return -(-overview_row_bp(length) // OVERVIEW_ROW_BINS)


def genome_statistics(con, depth_view, genome_id, length, edge_sets, orfs=None,
                      warn_contrast=False):
    """Bin one genome's per-base group statistics, in one pass of windows.

    At each base, each group's depth Q1, median, Q3 and mean are taken across
    its samples (a sample without reads there counts as 0), as are prevalence
    -- the share of its samples whose breadth covers the base -- and the
    union, whether any does. A bin then shows the mean of each over its bases,
    and is in the union if any base is: so the median drawn is a per-base
    median, never a median of per-sample averages.

    With `orfs`, the same pass gives each ORF's statistics per group: each
    sample's mean depth over the ORF, then the group's quantiles of those --
    the typical sample's depth, where the bins' median is the typical base's
    -- and the mean, prevalence and union breadth over the ORF's bases.

    Requires `ROSTER_TABLE` and `BREADTH_INTERVALS_TABLE`. Replaces
    `DEPTH_ALIGNMENTS_TABLE`.

    Parameters
    ----------
    edge_sets : list of np.ndarray
        Bin edges within [1, `length` + 1], one array per table: the overview
        and each detail region.
    orfs : dict of np.ndarray, optional
        This genome's ORFs, as `genome_orfs` returns them.
    warn_contrast : bool, optional
        Warn when the genome has no contrast (`orf_contrast`); for runs that
        asked for --orf-contrast. The column is filled either way.

    Returns
    -------
    bins : list of dict
        One table per edge set, a row per group (sorted) and bin: `group`,
        `bin_start`, `bin_stop`, `q1`, `median`, `q3`, `mean`, `prevalence`
        and `union`, each a numpy array.
    orf_table : dict or None
        With `orfs`, a row per ORF and group (sorted): the columns of
        `_io.write_orf_table`, each a numpy array.
    """
    roster = con.sql(f"""SELECT group_name FROM {ROSTER_TABLE}
                         ORDER BY sample_idx""").fetchnumpy()["group_name"]
    groups, group_of = np.unique(roster, return_inverse=True)
    members = [np.flatnonzero(group_of == g) for g in range(len(groups))]
    intervals = con.execute(
        f"""SELECT sample_idx, {COLUMN_START}, {COLUMN_STOP}
            FROM {BREADTH_INTERVALS_TABLE} WHERE {COLUMN_GENOME_ID} = ?""",
        [genome_id],
    ).fetchnumpy()
    interval_group = group_of[intervals["sample_idx"]]
    ends = [[np.sort(intervals[column][interval_group == g].astype(np.int64))
             for column in (COLUMN_START, COLUMN_STOP)]
            for g in range(len(groups))]
    stage_depth(con, depth_view, genome_id)

    spans = [(edges[:-1], edges[1:]) for edges in edge_sets]
    if orfs is not None:
        segment_orf, *segments = orf_segments(orfs[COLUMN_START],
                                              orfs[COLUMN_STOP], length)
        spans.append(segments)
        sample_depth = np.zeros((len(roster), len(segment_orf)), np.int64)

    # q1, median, q3 (x4), depth, covering samples, covered: integer sums
    totals = [np.zeros((len(groups), 6, len(starts)), np.int64)
              for starts, _ in spans]
    width = window_size(len(roster), length)
    for w0 in range(1, length + 1, width):
        w1 = min(w0 + width, length + 1)
        depth = window_depth(con, len(roster), w0, w1)
        if orfs is not None:
            _add_span_sums(sample_depth, depth, *segments, w0, w1)
        for g, rows in enumerate(members):
            covering = coverage_counts(*ends[g], w0, w1)
            per_base = np.vstack([quartiles_x4(depth[rows]),
                                  depth[rows].sum(axis=0, dtype=np.int64),
                                  covering, covering > 0])
            for (starts, stops), total in zip(spans, totals, strict=True):
                _add_span_sums(total[g], per_base, starts, stops, w0, w1)

    sizes = np.array([len(rows) for rows in members])[:, None]
    tables = []
    for edges, total in zip(edge_sets, totals[: len(edge_sets)], strict=True):
        bp = np.diff(edges)
        tables.append({
            "group": np.repeat(groups, len(bp)),
            "bin_start": np.tile(edges[:-1], len(groups)),
            "bin_stop": np.tile(edges[1:], len(groups)),
            "q1": (total[:, 0] / (4 * bp)).ravel(),
            "median": (total[:, 1] / (4 * bp)).ravel(),
            "q3": (total[:, 2] / (4 * bp)).ravel(),
            "mean": (total[:, 3] / (sizes * bp)).ravel(),
            "prevalence": (total[:, 4] / (sizes * bp)).ravel(),
            "union": (total[:, 5] > 0).ravel(),
        })
    if orfs is None:
        return tables, None

    def per_orf(sums):
        """Add each ORF's segments up: its last axis becomes ORFs."""
        out = np.zeros((*sums.shape[:-1], len(orfs[COLUMN_START])), np.int64)
        np.add.at(out.T, segment_orf, sums.T)
        return out

    depth_sum = per_orf(sample_depth)
    breadth = per_orf(totals[-1])
    bp = orfs[COLUMN_STOP] - orfs[COLUMN_START]
    quartiles = np.stack([np.quantile(depth_sum[rows] / bp, QUARTILES, axis=0)
                          for rows in members])
    columns = {
        "depth_q1": quartiles[:, 0],
        "depth_median": quartiles[:, 1],
        "depth_q3": quartiles[:, 2],
        "depth_mean": np.stack([depth_sum[rows].sum(axis=0) for rows in members])
                      / (sizes * bp),
        "prevalence": breadth[:, 4] / (sizes * bp),
        "union_breadth": breadth[:, 5] / bp,
    }
    # ORF by ORF, the groups within each: transpose (group, ORF) arrays
    orf_table = {
        COLUMN_GENOME_ID: np.full(len(bp) * len(groups), genome_id, dtype=object),
        **{key: np.repeat(orfs[key], len(groups))
           for key in ("orf_id", "label", "type", COLUMN_START, COLUMN_STOP,
                       "strand")},
        "group": np.tile(groups, len(bp)),
        "n_samples": np.tile(sizes[:, 0], len(bp)),
        **{key: values.T.ravel() for key, values in columns.items()},
        "contrast": np.repeat(
            orf_contrast(quartiles[:, 1], groups, genome_id, warn_contrast),
            len(groups),
        ),
    }
    return tables, orf_table


def _add_span_sums(total, per_base, starts, stops, w0, w1):
    """Add `per_base`'s rows, summed over each span's part of [w0, w1).

    `total` is rows x spans. Only the spans the window overlaps are touched,
    so a window costs the same however many spans the genome has.
    """
    here = np.flatnonzero((starts < w1) & (stops > w0))
    prefix = np.zeros((len(per_base), w1 - w0 + 1), np.int64)
    np.cumsum(per_base, axis=1, out=prefix[:, 1:])
    total[:, here] += (prefix[:, np.clip(stops[here], w0, w1) - w0]
                       - prefix[:, np.clip(starts[here], w0, w1) - w0])


def genome_orfs(con, genome_id):
    """Return one genome's ORFs, by position, as numpy arrays.

    The columns are `orf_id`, `label`, `type`, `start`, `stop`, `strand`, and
    `attributes`, a dict per ORF, which highlights and colours match.
    Requires `ORFS_TABLE`.
    """
    return con.execute(
        f"""SELECT orf_id, label, type, {COLUMN_START}, {COLUMN_STOP}, strand,
                   attributes
            FROM {ORFS_TABLE} WHERE {COLUMN_GENOME_ID} = ?
            ORDER BY {COLUMN_START}, {COLUMN_STOP}, orf_id, type""",
        [genome_id],
    ).fetchnumpy()


def orf_segments(starts, stops, length):
    """Split ORFs that run past the genome's end, across its origin.

    GFF3 writes such an ORF, on a circular genome, with an end beyond the
    length: [start, stop) becomes [start, length + 1) and [1, stop - length).

    Returns
    -------
    tuple of np.ndarray
        Each segment's ORF index, start and stop.
    """
    wraps = np.flatnonzero(stops > length + 1)
    return (np.concatenate([np.arange(len(starts)), wraps]),
            np.concatenate([starts, np.ones(len(wraps), starts.dtype)]),
            np.concatenate([np.minimum(stops, length + 1), stops[wraps] - length]))


def orf_contrast(median, groups, genome_id, warn=False):
    """Compare two groups' depth, ORF by ORF, each scaled by its typical ORF.

    ``log2((B / norm B + c) / (A / norm A + c))``, where A and B are the
    groups' per-ORF median depth, in sorted order, a group's norm is the
    median of those over the genome's ORFs, and c is `CONTRAST_PSEUDOCOUNT`.
    Positive where the second group is the higher. Scaling makes a group
    sequenced deeper throughout no different from the other.

    Parameters
    ----------
    median : np.ndarray
        Groups x ORFs.

    warn : bool, optional
        Warn, naming the genome and group, when a norm of 0 leaves no
        contrast.

    Returns
    -------
    np.ndarray
        Per ORF; NaN unless there are exactly two groups and both norms are
        above 0.
    """
    contrast = np.full(median.shape[1], np.nan)
    if len(groups) != 2 or median.shape[1] == 0:
        return contrast
    norm = np.median(median, axis=1)
    zero = [group for group, value in zip(groups, norm, strict=True) if value == 0]
    if zero:
        if warn:
            logger.warning(
                f"No ORF contrast for {genome_id}: its median ORF has median "
                f"depth 0 in {', '.join(zero)}, so there is no typical depth to "
                "scale by."
            )
        return contrast
    a, b = median / norm[:, None]
    return np.log2((b + CONTRAST_PSEUDOCOUNT) / (a + CONTRAST_PSEUDOCOUNT))

