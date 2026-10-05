import csv
import gzip

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

from ._constants import (
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_PERCENT_COVERED,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)
from ._cov import (
    compute_cumulative,
    cumulative_curves,
    get_covered,
    mask_table,
    ordered_coverage,
    slice_positions,
)
from ._io import BED_POSITIONS_TABLE

#: `per_sample_plots`' genome-sorted copy of the positions, sliced per genome.
PLOT_POSITIONS_TABLE = "plot_positions"

#: The narrowest bucket in a scaled position plot. 1/10000 of a 145kb
#: chloroplast is 15bp, narrower than the intervals being binned, and a genome
#: under 10kb got buckets of less than a base.
MIN_BUCKET_WIDTH = 100

#: Header of every `.ks.csv`. The first four names are frozen; the Bonferroni
#: column was appended in M11a so that readers taking columns by position were
#: unaffected.
KS_HEADER = (
    "label_A",
    "label_B",
    "ks-statistic",
    "ks-pvalue",
    "ks-pvalue-bonferroni",
)


def _write_delimited(path, header, rows, delimiter=",", compress=False):
    """Write `rows` as a delimited text file, optionally gzipped.

    Replaces two polars `write_csv` calls, and reproduces them byte for byte
    across all sixteen frozen files under `example/plots/`.

    **Do not reach for `repr()` or an f-string to format the values here.**
    `csv.writer` renders with `str()`, which for a float is the shortest form
    that round-trips -- exactly what polars wrote. The values arriving here are
    numpy scalars (the position values come from `np.histogram`; the KS
    values were `np.float64` too until M9 moved them from scipy to miint),
    and under numpy 2 `repr()` of one of those is the string
    `np.float64(0.3)`. That would corrupt every float in every `.ks.csv` and
    `.tsv.gz` micov writes, and it is the sort of thing a tidy-up refactor
    does without noticing.

    `lineterminator` is set because the csv module defaults to CRLF.
    """
    opener = gzip.open if compress else open
    with opener(path, "wt", newline="") as fp:
        writer = csv.writer(fp, delimiter=delimiter, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def ks_2samp(con, curve_a, curve_b):
    """Two-sample Kolmogorov-Smirnov test, by miint's `ks_2samp`.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        A connection carrying the miint extension.
    curve_a, curve_b : sequence of float
        The two samples.

    Returns
    -------
    tuple of (float, float)
        The statistic and the two-sided exact p-value.

    Notes
    -----
    These numbers are cited in the paper, which computed them with
    `scipy.stats.ks_2samp` (1.17.1). The statistic is reproduced exactly:
    miint returns D as the exact lattice value ``h / lcm(n1, n2)``, as that
    scipy did. The p-value agrees to a few ULP, not bit for bit -- miint
    derives it independently rather than transcribing scipy.

    miint implements only the exact method and raises above 10000
    observations per sample, where scipy fell back to an approximation.
    Here that is 10000 samples in one metadata group.
    """
    return con.execute(
        "SELECT ks.statistic, ks.pvalue "
        "FROM (SELECT ks_2samp(?::DOUBLE[], ?::DOUBLE[]) AS ks)",
        [np.asarray(curve_a, dtype=np.float64), np.asarray(curve_b, dtype=np.float64)],
    ).fetchone()


def ks_table(con, curves, monte_label=None):
    """Compare every pair of curves by KS, Bonferroni-correcting the groups.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        A connection carrying the miint extension.
    curves : dict of str to sequence of float
        Curve per label, in the order the comparisons are written.
    monte_label : str, optional
        The label of the Monte Carlo curve, if there is one.

    Returns
    -------
    list of list
        One row per pair, ordered as `KS_HEADER`.

    Notes
    -----
    The Bonferroni family is the group-vs-group comparisons in this table,
    so ``m`` is their count and the corrected value is ``min(1, p * m)``.
    Comparisons against the Monte Carlo curve are a null-model check rather
    than a hypothesis: they are left out of ``m`` and their corrected field
    is empty. Counting them would give the same pair of groups a different
    corrected p-value depending on whether ``--monte`` was passed.

    The Monte Carlo curve is identified by `monte_label`, not by its label
    text, so a metadata group whose value happens to start "Monte Carlo" is
    corrected like any other group.
    """
    labels = list(curves)
    pairs = [
        (a, b) for idx, a in enumerate(labels) for b in labels[idx + 1 :]
    ]
    m = sum(monte_label not in pair for pair in pairs)

    rows = []
    for label_a, label_b in pairs:
        statistic, pvalue = ks_2samp(con, curves[label_a], curves[label_b])
        if monte_label in (label_a, label_b):
            corrected = ""
        else:
            corrected = min(1.0, pvalue * m)
        rows.append([label_a, label_b, statistic, pvalue, corrected])
    return rows


def per_sample_plots(
    view,
    sample_metadata_column,
    output,
    monte,
    monte_iters,
    percentile,
):
    """Construct plots for all genomes.

    Construct coverage and position plots for all genomes described with
    coverage data.

    Parameters
    ----------
    view : View
        The current View of the coverage data
    sample_metadata_column : str
        The specific column to stratify when plotting. Note it is assumed
        this column is categorical.
    output : str
        A prefix to use on plotting. This can include a directory, for instance,
        "foo/bar/theprefix"
    monte : str or None
        One of (None, 'focused', 'unfocused'). See "add_monte" for more detail.
    monte_iters : int
        The number of Monte Carlo iterations to perform.
    """
    # Positions stay in DuckDB and only one genome's rows are fetched at a
    # time. Fetching the whole table and masking it per genome was four full
    # numpy string scans per genome: on 9,388 genomes and 29.7M intervals
    # that alone was ~2 h, and `per-sample` ran 2.5x slower than the polars
    # release it replaced. Sorted by genome, each genome's rows sit in a
    # row group or two, so `WHERE genome_id = ?` reads only those.
    view.con.sql(f"""CREATE OR REPLACE TEMP TABLE {PLOT_POSITIONS_TABLE} AS
                     SELECT * FROM ({view.positions().sql_query()})
                     ORDER BY {COLUMN_GENOME_ID}""")

    # Coverage is one row per sample x genome, small enough to hold in memory.
    # Its row order does not matter: `ordered_coverage` breaks breadth ties by
    # sample_id.
    all_coverage = view.coverages().fetchnumpy()
    by_genome = np.argsort(all_coverage[COLUMN_GENOME_ID], kind="stable")
    genomes, firsts = np.unique(
        all_coverage[COLUMN_GENOME_ID][by_genome], return_index=True
    )
    bounds = np.append(firsts, len(by_genome))

    # "unfocused" Monte Carlo draws from every sample with any coverage, which
    # no single genome's rows can tell it
    sample_universe = np.unique(all_coverage[COLUMN_SAMPLE_ID])

    metadata = view.metadata().fetchnumpy()
    feature_metadata = view.feature_metadata().fetchnumpy()
    target_lookup = dict(view.feature_names().fetchall())

    if view.constrain_positions:
        n_genomes = len(np.unique(feature_metadata[COLUMN_GENOME_ID]))
        n_regions = len(feature_metadata[COLUMN_GENOME_ID])

        if n_genomes != n_regions:
            raise ValueError(
                "Plotting does not yet support desribing multiple regions."
            )

    for genome, lo, hi in zip(genomes, bounds[:-1], bounds[1:], strict=True):
        target_coverage = mask_table(all_coverage, by_genome[lo:hi])
        target_positions = view.con.execute(
            f"SELECT * FROM {PLOT_POSITIONS_TABLE} WHERE {COLUMN_GENOME_ID} = ?",
            [genome],
        ).fetchnumpy()
        target_name = target_lookup[genome]
        is_genome = feature_metadata[COLUMN_GENOME_ID] == genome
        ymin = feature_metadata[COLUMN_START][is_genome][0]
        ymax = feature_metadata[COLUMN_STOP][is_genome][0]

        coverage_curve(
            view.con,
            metadata,
            target_coverage,
            target_positions,
            genome,
            sample_metadata_column,
            output,
            target_name,
            percentile,
            monte_iters,
            monte,
            False,
            sample_universe=sample_universe,
        )
        coverage_curve(
            view.con,
            metadata,
            target_coverage,
            target_positions,
            genome,
            sample_metadata_column,
            output,
            target_name,
            percentile,
            monte_iters,
            monte,
            True,
            sample_universe=sample_universe,
        )
        position_plot(
            metadata,
            target_coverage,
            target_positions,
            genome,
            sample_metadata_column,
            output,
            target_name,
            ymin,
            ymax,
            scale=None,
        )
        position_plot(
            metadata,
            target_coverage,
            target_positions,
            genome,
            sample_metadata_column,
            output,
            target_name,
            ymin,
            ymax,
            scale=10000,
        )


def add_monte(
    con,
    monte_type,
    ax,
    max_x,
    iters,
    metadata_full,
    target,
    target_positions,
    coverage,
    accumulate,
    lengths,
    percentile,
    sample_universe,
):
    """Perform a Monte Carlo simulation over coverage.

    Parameters
    ----------
    monte_type : str
        The specific approach to take, either "focused" or "unfocused".
        In "focused" mode, only samples with nonzero coverage to the target
        are considered. In "unfocused" mode, any sample with nonzero coverage
        to any target is considered.
    ax : plt.Axes
        A set of axes to plot into
    max_x : int
        The maximum number of samples to sample
    iters : int
        The number of iterations to perform
    metadata_full : dict of np.ndarray
        The metadata for all samples with nonzero coverage to any target
    target : str
        The genome of iterest
    target_positions : dict of np.ndarray
        The per sample per genome regions covered for the target of interest
    coverage : dict of np.ndarray
        The per sample coverage of the target
    accumulate : bool
        If true, construct a cumulative curve. If false, construct a non
        cumulative curve.
    lengths : dict of np.ndarray
        genome to length data
    percentile : bool
        If true, use percentiles (0-100) on x-axis instead of sample counts.
    sample_universe : np.ndarray
        Every sample with coverage of any genome: the "unfocused" pool.

    Notes
    -----
    The Monte Carlo procedure works by (1) picking a random set of samples
    independent of sample metadata (2) computing coverage over those samples
    (3) repeat `monte_iter` times. This gathers a distribution of coverage
    and provides a null for context for interpreration of the true curves.

    The permutation is unseeded, so the envelope does not reproduce run to
    run. That is unchanged from the polars shuffle this replaces, and the
    golden suite treats Monte Carlo rows accordingly.

    """
    length = lengths[COLUMN_LENGTH][lengths[COLUMN_GENOME_ID] == target][0]

    color = "k"
    line_alpha = 0.6
    fill_alpha = 0.1

    is_target = coverage[COLUMN_GENOME_ID] == target

    if monte_type == "focused":
        ls_median = "dotted"
        ls_bound = "--"

        # constrain to the target
        sample_set = coverage[COLUMN_SAMPLE_ID][is_target]

    elif monte_type == "unfocused":
        ls_median = "dashed"
        ls_bound = "-."

        # take all samples
        sample_set = sample_universe
    else:
        raise ValueError(f"Unknown monte_type='{monte_type}'")

    coverage = mask_table(coverage, is_target)

    max_x += 1  # it comes in as zero index but we need count
    monte_x = list(range(max_x))
    rng = np.random.default_rng()

    # The selection stays in numpy. Moving it into SQL would change the RNG,
    # and with it the published Monte Carlo envelopes; only the accumulation
    # needed to move. `np.isin` keeps `sample_set`'s order rather than the
    # permuted one, which is deliberate -- the permutation chooses *which*
    # samples take part, and `ordered_coverage` then ranks them by breadth.
    groups = [
        sample_set[np.isin(sample_set, rng.permutation(sample_set)[:max_x])]
        for _ in range(iters)
    ]

    if accumulate:
        # every iteration accumulates in one aggregate call rather than one
        # O(n^2) pass each -- this is where the Monte Carlo cost was
        _, monte_y = cumulative_curves(
            con, coverage, groups, target, target_positions, lengths
        )
    else:
        # non-cumulative is a per-sample breadth lookup, not an accumulation,
        # so there is nothing here for the aggregate to do
        monte_y = np.asarray([
            ordered_coverage(coverage, {COLUMN_SAMPLE_ID: grp}, target, length)[
                COLUMN_PERCENT_COVERED
            ]
            for grp in groups
        ])
    median = np.median(monte_y, axis=0)
    std = np.std(monte_y, axis=0)

    if percentile and monte_x is not None:
        monte_x = [x * 100 / (len(monte_x) - 1) for x in monte_x]

    ax.plot(
        monte_x, median, color=color, linestyle=ls_median, linewidth=1, alpha=line_alpha
    )
    ax.plot(
        monte_x,
        median + std,
        color=color,
        linestyle=ls_bound,
        linewidth=1,
        alpha=line_alpha,
    )
    ax.plot(
        monte_x,
        median - std,
        color=color,
        linestyle=ls_bound,
        linewidth=1,
        alpha=line_alpha,
    )
    ax.fill_between(monte_x, median - std, median + std, color=color, alpha=fill_alpha)
    return f"Monte Carlo {monte_type} (n={len(monte_x)})", median


def coverage_curve(
    con,
    metadata_full,
    coverage,
    positions,
    target,
    variable,
    output,
    target_name,
    percentile,
    iters=None,
    with_monte=None,
    accumulate=False,
    min_group_size=10,
    *,
    sample_universe,
):
    """Construct coverage curves.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        A connection carrying the miint extension, for the accumulation.
    metadata_full : dict of np.ndarray
        The metadata for all samples with nonzero coverage to any target
    coverage : dict of np.ndarray
        The per sample coverage of the target. Only the target's rows: callers
        slice per genome, since filtering the whole table here for every genome
        was the cost that made `per-sample` slow on studies with many genomes.
    positions : dict of np.ndarray
        The per sample regions covered on the target, likewise only its rows
    target : str
        The genome of interest
    variable : str
        The specific metadata variable to use for stratification
    output : str
        A prefix to use on plotting. This can include a directory, for instance,
        "foo/bar/theprefix"
    target_name : str
        The name of the target
    iters : int, optional
        The number of Monte Carlo iterations to perform
    with_monte : str, optional
        Add in a Monte Carlo curve if 'focused' or 'unfocused'. See `add_monte`
        for more information.
    accumulate : bool
        If true, construct a cumulative curve. If false, construct a non
        cumulative curve.
    sample_universe : np.ndarray
        Every sample with coverage of any genome, for "unfocused" Monte Carlo.
    min_group_size : int, optional
        The minimum number of samples to have coverage against the target
        in order to be plotted
    percentile : bool, optional
        If true, use percentiles (0-100) on x-axis instead of sample counts.
        If false, use sample counts.

    Notes
    -----
    A coverage curve, whether cumulative or non-cumulative, is plotted
    per sample group described by the `variable`.

    """
    if with_monte is not None and iters is None:
        raise ValueError("Running with Monte Carlo but no iterations set")

    if min_group_size < 0:
        raise ValueError("min_group_size must be greater than 0")

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_prop_cycle(None)

    labels = []
    curves = {}

    target_positions = mask_table(positions, positions[COLUMN_GENOME_ID] == target)
    coverage = mask_table(coverage, coverage[COLUMN_GENOME_ID] == target)
    cov_samples = np.unique(coverage[COLUMN_SAMPLE_ID])
    metadata = mask_table(
        metadata_full, np.isin(metadata_full[COLUMN_SAMPLE_ID], cov_samples)
    )

    if len(target_positions[COLUMN_GENOME_ID]) == 0:
        raise ValueError("Target genome has no associated coverage")

    if len(coverage[COLUMN_GENOME_ID]) == 0:
        raise ValueError("No sample has coverage on the target genome")

    # `coverage` is already constrained to the one target genome, so the
    # distinct (genome_id, length) pairs reduce to the distinct lengths
    distinct_lengths = np.unique(coverage[COLUMN_LENGTH])

    if len(distinct_lengths) > 1:
        raise ValueError("More than one length provided for the genome")

    length = distinct_lengths[0]
    lengths = {
        COLUMN_GENOME_ID: np.array([target], dtype=object),
        COLUMN_LENGTH: distinct_lengths,
    }
    value_order = np.unique(metadata[variable])

    max_x = 0
    for name, color in zip(value_order, range(10), strict=False):
        color = f"C{color}"

        grp = mask_table(metadata, metadata[variable] == name)

        n = len(grp[COLUMN_SAMPLE_ID])
        if n < min_group_size:
            continue

        if accumulate:
            cur_x, cur_y = compute_cumulative(
                con, coverage, grp, target, target_positions, lengths
            )
        else:
            grp_coverage = ordered_coverage(coverage, grp, target, length)
            cur_x = grp_coverage["x_unscaled"]
            cur_y = grp_coverage[COLUMN_PERCENT_COVERED]

        if cur_x is None:
            continue

        # int(): the ranks are uint64, and numpy promotes uint64 + int to
        # float64, which `add_monte` then cannot hand to range()
        max_x = max(max_x, int(cur_x.max()))

        labels.append(f"{name} (n={len(cur_x)})")

        if percentile and cur_x is not None:
            cur_x_percentile = cur_x * 100 / (len(cur_x) - 1)
            ax.plot(cur_x_percentile, cur_y, color=color)
        else:
            ax.plot(cur_x, cur_y, color=color)
        curves[name] = cur_y

    if not labels:
        # no group reached `min_group_size`. Most genomes of a large study
        # land here, and an unclosed figure stays alive until the process
        # exits -- thousands of them, before this close.
        plt.close()
        return

    monte_label = None
    if with_monte is not None:
        monte_label, median_curve = add_monte(
            con,
            with_monte,
            ax,
            max_x,
            iters,
            metadata_full,
            target,
            target_positions,
            coverage,
            accumulate,
            lengths,
            percentile,
            sample_universe,
        )
        labels.append(monte_label)
        curves[monte_label] = median_curve

    tag = "cumulative" if accumulate else "non-cumulative"

    ax.set_ylabel("Percent genome covered", fontsize=16)
    if percentile:
        ax.set_xlabel("Within group sample percentile", fontsize=16)
    else:
        ax.set_xlabel("Within group sample rank by coverage", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    if percentile:
        ax.set_xlim(0, 100)
    else:
        ax.set_xlim(0, max_x)
    ax.set_ylim(0, 100)
    ax.set_title((f"{tag}: {target_name}({target}) " f"({length}bp)"), fontsize=16)
    ax.legend(labels, fontsize=14, loc="center left")

    plt.tight_layout()

    if with_monte is not None:
        tag = f"{tag}-monte-{with_monte}"

    plt.savefig(f"{output}.{target_name}.{target}.{variable}.{tag}.png")
    plt.close()

    if accumulate:
        outf = f"{output}.{target_name}.{target}.{variable}.{tag}.ks.csv"
        _write_delimited(outf, KS_HEADER, ks_table(con, curves, monte_label))


def position_plot_segments(con):
    """Compute normalized covered intervals per genome, ready to draw.

    Split out from `single_sample_position_plot` so the numbers can be
    asserted without rendering: `position-plot` writes only PNGs, whose bytes
    are not comparable across matplotlib versions, so before this existed the
    command's values were guarded by nothing at all.

    Requires the `bed_positions` and `genome_lengths` tables, from
    `_io.load_bed_cov` and `_io.load_genome_lengths`.

    Returns
    -------
    dict of str to np.ndarray
        Keyed by genome, each an `(n, 3)` array of `(x, start / length,
        stop / length)` ordered by start. `x` is the constant the plot draws
        every segment at; positions are unit-normalized against **their own**
        genome's length.

    Notes
    -----
    Genomes with coverage but no entry in `genome_lengths` are dropped, not
    reported. That is an inner join and it is what micov has always done here:
    without a length there is no denominator, and passing a length file
    covering only the genomes of interest is an ordinary way to use this
    command.

    """
    genomes = [
        row[0]
        for row in con.sql(f"""SELECT DISTINCT p.{COLUMN_GENOME_ID}
                               FROM {BED_POSITIONS_TABLE} p
                                   JOIN genome_lengths l
                                       USING ({COLUMN_GENOME_ID})
                               ORDER BY 1""").fetchall()
    ]

    segments = {}
    for genome in genomes:
        # ORDER BY is explicit because DuckDB's sort is not stable and these
        # arrive in whatever order the scan produced; the polars sort this
        # replaced was stable, so unordered output would be a silent change.
        columns = con.execute(
            f"""SELECT 0.5 AS x,
                       p.{COLUMN_START} / l.{COLUMN_LENGTH} AS {COLUMN_START},
                       p.{COLUMN_STOP} / l.{COLUMN_LENGTH} AS {COLUMN_STOP}
                FROM {BED_POSITIONS_TABLE} p
                    JOIN genome_lengths l USING ({COLUMN_GENOME_ID})
                WHERE p.{COLUMN_GENOME_ID} = ?
                ORDER BY p.{COLUMN_START}, p.{COLUMN_STOP}""",
            [genome],
        ).fetchnumpy()
        segments[genome] = np.column_stack(
            [columns["x"], columns[COLUMN_START], columns[COLUMN_STOP]]
        )

    return segments


def single_sample_position_plot(con, output):
    """Construct a metadata-independent position plot, one PNG per genome.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        A connection carrying `bed_positions` and `genome_lengths`.
    output : str
        A prefix to use on plotting. This can include a directory, for
        instance, "foo/bar/theprefix"

    Notes
    -----
    The `scale` parameter this used to take was never read by the body and
    never passed by the CLI; it went with the rewrite.

    """
    for genome, coordinates in position_plot_segments(con).items():
        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        lc = mc.LineCollection(get_covered(coordinates), linewidths=2, alpha=0.7)
        ax.add_collection(lc)

        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(0, 1.0)

        ax.set_title(f"Position plot: {genome}", fontsize=20)
        ax.set_ylabel("Unit normalized position", fontsize=20)

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="both", which="minor", labelsize=16)
        plt.tight_layout()
        # `genome` is a scalar here. It used to be the key polars `group_by`
        # yields -- a one-element tuple -- which went straight into this
        # f-string, so every file was named `out.('G1',).position-plot.png`.
        plt.savefig(f"{output}.{genome}.position-plot.png")
        plt.close()


def position_plot(
    metadata,
    coverage,
    positions,
    target,
    variable,
    output,
    target_name,
    ymin,
    ymax,
    scale=None,
):
    """Construct position plots stratified by metadata value.

    Parameters
    ----------
    metadata : dict of np.ndarray
        The metadata for all samples with nonzero coverage to any target
    coverage : dict of np.ndarray
        The per sample coverage of the target; only its rows
    positions : dict of np.ndarray
        The per sample regions covered on the target; only its rows
    target : str
        The genome of interest
    variable : str
        The specific metadata variable to use for stratification
    output : str
        A prefix to use on plotting. This can include a directory, for instance,
        "foo/bar/theprefix"
    target_name : str
        The name of the target
    ymin : int
        For forcing ax.ylim.
    ymax : int
        For forcing ax.ylim.
    scale : int, optional
        If specified, represent the genome as `scale` number of buckets, or as
        `MIN_BUCKET_WIDTH` buckets if those would be narrower. A bucket is
        considered represented if any position within the bucket is covered

    """
    if scale is not None and scale <= 1:
        raise ValueError("`scale` must be greater than 1")

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    labels = []
    colors = []

    length = ymax - ymin

    if scale is not None:
        if length >= scale * MIN_BUCKET_WIDTH:
            # np.histogram's own edges, so long genomes keep the published `y`
            edges = np.linspace(ymin, ymax, scale + 1, dtype=np.float64)
        else:
            # exactly MIN_BUCKET_WIDTH from ymin; the last bucket holds the rest
            edges = np.append(
                np.arange(ymin, ymax, MIN_BUCKET_WIDTH, dtype=np.float64),
                np.float64(ymax),
            )

    target_positions = mask_table(positions, positions[COLUMN_GENOME_ID] == target)

    samples_with_positions = np.unique(target_positions[COLUMN_SAMPLE_ID])
    metadata = mask_table(
        metadata, np.isin(metadata[COLUMN_SAMPLE_ID], samples_with_positions)
    )

    # TODO: expose to allow ordering by a variable rather than coverage
    custom_xorder = None

    # np.unique sorts, so a group's position here is also its color index --
    # which is what joining against a separately sorted color order produced.
    # Groups are then laid out smallest first; the sort is stable, so groups
    # tied on size stay in value order.
    names, counts = np.unique(metadata[variable], return_counts=True)
    max_x = int(counts.sum())
    order = np.argsort(counts, kind="stable")

    label_pos = []
    x_offset = 0
    boundaries = []
    tsv_x = []
    tsv_y = []
    tsv_group = []

    invert = len(order) == 2

    for oidx, row in enumerate(order):
        name = names[row]
        count = int(counts[row])
        grp = mask_table(metadata, metadata[variable] == name)
        color = f"C{row}"

        if custom_xorder is not None:
            has_order = np.array([v is not None for v in grp[custom_xorder]])
            grp_coverage = mask_table(grp, has_order)
            grp_coverage = mask_table(
                grp_coverage, np.argsort(grp_coverage[custom_xorder], kind="stable")
            )
            n = len(grp_coverage[COLUMN_SAMPLE_ID])
            grp_coverage["x_unscaled"] = np.arange(
                x_offset, x_offset + n, dtype=np.uint64
            )
            grp_coverage[COLUMN_GENOME_ID] = np.full(n, target, dtype=object)
        else:
            grp_coverage = ordered_coverage(coverage, grp, target, length)
            # np.uint64 rather than a plain int: numpy promotes uint64 + int
            # to float64, and the ranks must stay integral
            grp_coverage["x_unscaled"] += np.uint64(x_offset)

        n = len(grp_coverage[COLUMN_SAMPLE_ID])
        if n == 0:
            continue

        # reverse plot order if we have two groups and in second group
        if invert and oidx == 1:
            grp_coverage = mask_table(grp_coverage, np.arange(n - 1, -1, -1))
            grp_coverage["x_unscaled"] = np.arange(
                x_offset, x_offset + n, dtype=np.uint64
            )

        colors.append(color)

        hist_x = []
        hist_y = []

        # the ranks are unique per sample, so the join this replaces only ever
        # paired a sample's intervals with its own x -- which the loop has
        for sid, x in zip(
            grp_coverage[COLUMN_SAMPLE_ID], grp_coverage["x_unscaled"], strict=True
        ):
            x = int(x)
            cur_positions = slice_positions(target_positions, sid)
            starts = cur_positions[COLUMN_START]
            stops = cur_positions[COLUMN_STOP]

            if scale is None:
                covered_positions = get_covered(
                    np.column_stack([np.full(len(starts), x), starts, stops])
                )
                lc = mc.LineCollection(
                    covered_positions, color=color, linewidths=0.5, alpha=0.7
                )
                ax.add_collection(lc)
            else:
                # every bucket [start, stop) overlaps: from the one holding
                # `start` to the last one whose left edge is before `stop`.
                # Above 1Mb edges are fractional, and one can fall inside the
                # last base, so "the bucket holding stop - 1" would miss it. An
                # alignment can run off the end of the genome; only the part
                # before `ymax` is plotted.
                first = np.searchsorted(edges, starts, side="right") - 1
                last = np.searchsorted(edges, np.minimum(stops, ymax),
                                       side="left") - 1
                touched = np.zeros(len(edges), dtype=np.int64)
                np.add.at(touched, first, 1)
                np.add.at(touched, last + 1, -1)
                obs_bins = edges[:-1][np.cumsum(touched[:-1]) > 0]
                hist_x.extend([x for _ in obs_bins])
                hist_y.extend(obs_bins)

        if scale is not None:
            ax.scatter(hist_x, hist_y, s=0.2, color=color, alpha=0.7)
            tsv_x += hist_x
            tsv_y += hist_y
            tsv_group += [name] * len(hist_x)

        label_pos.append(x_offset + (count // 2))
        labels.append(name)
        x_offset += count
        boundaries.append(x_offset)

    ax.set_xlim(0, max_x)
    ax.set_ylim(ymin, ymax)

    for x in boundaries[:-1]:
        ax.plot([x, x], [ymin, ymax], color="k", ls="--", alpha=0.6)

    if scale is None:
        ax.set_title(f"Position plot: {target} ({length}bp)", fontsize=20)
        ax.set_ylabel("Genome position", fontsize=20)
        scaletag = ""
    else:
        filename = (
            f"{output}.{target_name}.{target}.{variable}."
            "position-plot-scaled.tsv.gz"
        )
        _write_delimited(
            filename,
            ("group", "x", "y"),
            zip(tsv_group, tsv_x, tsv_y, strict=True),
            delimiter="\t",
            compress=True,
        )
        ax.set_title(f"Scaled position plot: {target} ({length}bp)", fontsize=20)
        ax.set_ylabel(f"Coverage ({edges[1] - edges[0]:.0f}bp buckets)", fontsize=20)
        scaletag = "-scaled"

    ax.set_xlabel("Within group sample rank by coverage", fontsize=16)
    ax.set_xticks(label_pos, labels, rotation=45, ha="right", fontsize=16)

    ax.tick_params(axis="y", which="major", labelsize=16)
    ax.tick_params(axis="y", which="minor", labelsize=16)
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(axis="y", ls="--", alpha=1, color="k")

    plt.tight_layout()
    plt.savefig(
        f"{output}.{target_name}.{target}.{variable}.position-plot{scaletag}.png"
    )
    plt.close()
