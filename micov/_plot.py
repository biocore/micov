import gzip

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats as ss
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
    get_covered,
    ordered_coverage,
)


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
    all_covered_positions = view.positions().pl()
    all_coverage = view.coverages().pl()
    metadata = view.metadata().pl()
    feature_metadata = view.feature_metadata().pl()
    feature_names = view.feature_names().pl()
    target_lookup = dict(feature_names.iter_rows())

    if view.constrain_positions:
        n_genomes = len(feature_metadata[COLUMN_GENOME_ID].unique())
        n_regions = len(feature_metadata)

        if n_genomes != n_regions:
            raise ValueError(
                "Plotting does not yet support desribing multiple regions."
            )

    for genome in all_coverage[COLUMN_GENOME_ID].unique():
        target_name = target_lookup[genome]
        ymin, ymax = (
            feature_metadata.filter(pl.col(COLUMN_GENOME_ID) == genome)
            .select([COLUMN_START, COLUMN_STOP])
            .row(0)
        )

        coverage_curve(
            metadata,
            all_coverage,
            all_covered_positions,
            genome,
            sample_metadata_column,
            output,
            target_name,
            percentile,
            monte_iters,
            monte,
            False,
        )
        coverage_curve(
            metadata,
            all_coverage,
            all_covered_positions,
            genome,
            sample_metadata_column,
            output,
            target_name,
            percentile,
            monte_iters,
            monte,
            True,
        )
        position_plot(
            metadata,
            all_coverage,
            all_covered_positions,
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
            all_coverage,
            all_covered_positions,
            genome,
            sample_metadata_column,
            output,
            target_name,
            ymin,
            ymax,
            scale=10000,
        )


def add_monte(
    monte_type,
    ax,
    max_x,
    iters,
    metadata_full,
    target,
    target_positions,
    coverage_full,
    accumulate,
    lengths,
    percentile,
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
    metadata_full : pl.DataFrame
        The metadata for all samples with nonzero coverage to any target
    target : str
        The genome of iterest
    target_positions : pl.DataFrame
        The per sample per genome regions covered for the target of interest
    coverage_full : pl.DataFrame
        The per sample per genome coverage for all samples and genomes
    accumulate : bool
        If true, construct a cumulative curve. If false, construct a non
        cumulative curve.
    lengths : pl.DataFrame
        genome to length data
    percentile : bool
        If true, use percentiles (0-100) on x-axis instead of sample counts.

    Notes
    -----
    The Monte Carlo procedure works by (1) picking a random set of samples
    independent of sample metadata (2) computing coverage over those samples
    (3) repeat `monte_iter` times. This gathers a distribution of coverage
    and provides a null for context for interpreration of the true curves.

    """
    length = (
        lengths.filter(pl.col(COLUMN_GENOME_ID) == target)
        .select(pl.col(COLUMN_LENGTH))
        .row(0)[0]
    )

    color = "k"
    line_alpha = 0.6
    fill_alpha = 0.1

    if monte_type == "focused":
        ls_median = "dotted"
        ls_bound = "--"

        # constrain to the target
        sample_set = (
            coverage_full.lazy()
            .filter(pl.col(COLUMN_GENOME_ID) == target)
            .select(pl.col(COLUMN_SAMPLE_ID))
            .collect()
        )

    elif monte_type == "unfocused":
        ls_median = "dashed"
        ls_bound = "-."

        # take all samples
        sample_set = coverage_full.select(pl.col(COLUMN_SAMPLE_ID).unique())
    else:
        raise ValueError(f"Unknown monte_type='{monte_type}'")

    coverage = coverage_full.filter(pl.col(COLUMN_GENOME_ID) == target)

    max_x += 1  # it comes in as zero index but we need count
    monte_y = []
    monte_x = list(range(max_x))

    for _ in range(iters):
        monte = (sample_set.select(pl.col(COLUMN_SAMPLE_ID).shuffle()).head(max_x))[
            COLUMN_SAMPLE_ID
        ]
        grp_monte = sample_set.filter(pl.col(COLUMN_SAMPLE_ID).is_in(monte))
        if accumulate:
            _, cur_y = compute_cumulative(
                coverage, grp_monte, target, target_positions, lengths
            )
        else:
            grp_coverage = ordered_coverage(coverage, grp_monte, target, length)
            cur_y = grp_coverage[COLUMN_PERCENT_COVERED].to_list()
        monte_y.append(cur_y)

    monte_y = np.asarray(monte_y)
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
    metadata_full,
    coverage_full,
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
):
    """Construct coverage curves.

    Parameters
    ----------
    metadata_full : pl.DataFrame
        The metadata for all samples with nonzero coverage to any target
    coverage_full : pl.DataFrame
        The per sample per genome coverage for all samples and genomes
    positions : pl.DataFrame
        The per sample per genome regions covered
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

    target_positions = positions.filter(pl.col(COLUMN_GENOME_ID) == target)
    coverage = coverage_full.filter(pl.col(COLUMN_GENOME_ID) == target)
    cov_samples = coverage.select(pl.col(COLUMN_SAMPLE_ID).unique())[COLUMN_SAMPLE_ID]
    metadata = metadata_full.filter(pl.col(COLUMN_SAMPLE_ID).is_in(cov_samples))

    if len(target_positions) == 0:
        raise ValueError("Target genome has no associated coverage")

    if len(coverage) == 0:
        raise ValueError("No sample has coverage on the target genome")

    lengths = coverage[[COLUMN_GENOME_ID, COLUMN_LENGTH]].unique()

    if len(lengths) > 1:
        raise ValueError("More than one length provided for the genome")

    length = lengths[COLUMN_LENGTH].item(0)
    value_order = metadata.select(pl.col(variable).unique().sort())[variable]

    max_x = 0
    for name, color in zip(value_order, range(10), strict=False):
        color = f"C{color}"

        grp = metadata.filter(pl.col(variable) == name)

        n = len(grp)
        if n < min_group_size:
            continue

        if accumulate:
            cur_x, cur_y = compute_cumulative(
                coverage, grp, target, target_positions, lengths
            )
        else:
            grp_coverage = ordered_coverage(coverage, grp, target, length)
            cur_x = grp_coverage["x_unscaled"]
            cur_y = grp_coverage[COLUMN_PERCENT_COVERED]

        if cur_x is None:
            continue

        max_x = max(max_x, cur_x.max())

        labels.append(f"{name} (n={len(cur_x)})")

        if percentile and cur_x is not None:
            cur_x_percentile = cur_x * 100 / (len(cur_x) - 1)
            ax.plot(cur_x_percentile, cur_y, color=color)
        else:
            ax.plot(cur_x, cur_y, color=color)
        curves[name] = cur_y

    if not labels:
        return

    if with_monte is not None:
        label, median_curve = add_monte(
            with_monte,
            ax,
            max_x,
            iters,
            metadata_full,
            target,
            target_positions,
            coverage_full,
            accumulate,
            lengths,
            percentile,
        )
        labels.append(label)
        curves[label] = median_curve

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
        ksresults = []
        curve_items = list(curves.items())

        for idx, (label_a, curve_a) in enumerate(curve_items):
            for label_b, curve_b in curve_items[idx + 1 :]:
                ks = ss.ks_2samp(curve_a, curve_b)
                ksresults.append([label_a, label_b, ks.statistic, ks.pvalue])

        outf = f"{output}.{target_name}.{target}.{variable}.{tag}.ks.tsv"
        pl.DataFrame(
            ksresults,
            schema=[
                ("label_A", str),
                ("label_B", str),
                ("ks-statistic", float),
                ("ks-pvalue", float),
            ],
            orient="row",
        ).write_csv(outf)


def single_sample_position_plot(positions, lengths, output, scale=None):
    """Construct a metadata-independent position plot.

    Parameters
    ----------
    positions : pl.DataFrame
        The genome positions to plot
    lengths : pl.DataFrame
        The genome lengths
    output : str
        A prefix to use on plotting. This can include a directory, for instance,
        "foo/bar/theprefix"
    scale : int, optional
        If specified, represent the genome as `scale` number of buckets. A
        bucket is considered represented if any position within the bucket
        is covered

    """
    positions = (
        positions.lazy().join(lengths.lazy(), on=COLUMN_GENOME_ID).with_columns(x=0.5)
    ).collect()
    for name, grp in positions.group_by(COLUMN_GENOME_ID):
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        grp = (
            grp.lazy()
            .sort(by=COLUMN_START)
            .select(
                pl.col("x"),
                pl.col(COLUMN_START) / pl.col(COLUMN_LENGTH),
                pl.col(COLUMN_STOP) / pl.col(COLUMN_LENGTH),
            )
        )

        covered_positions = get_covered(grp.collect().to_numpy())
        lc = mc.LineCollection(covered_positions, linewidths=2, alpha=0.7)
        ax.add_collection(lc)

        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(0, 1.0)

        ax.set_title(f"Position plot: {name}", fontsize=20)
        ax.set_ylabel("Unit normalized position", fontsize=20)

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.tick_params(axis="both", which="minor", labelsize=16)
        plt.tight_layout()
        plt.savefig(f"{output}.{name}.position-plot.png")
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
    metadata : pl.DataFrame
        The metadata for all samples with nonzero coverage to any target
    coverage : pl.DataFrame
        The per sample per genome coverage for all samples and genomes
    positions : pl.DataFrame
        The per sample per genome regions covered
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
        If specified, represent the genome as `scale` number of buckets. A
        bucket is considered represented if any position within the bucket
        is covered

    """
    if scale is not None and scale <= 1:
        raise ValueError("`scale` must be greater than 1")

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    labels = []
    colors = []

    length = ymax - ymin

    target_positions = positions.filter(pl.col(COLUMN_GENOME_ID) == target).lazy()

    samples_with_positions = (
        target_positions.select(pl.col(COLUMN_SAMPLE_ID)).unique().collect()
    )[COLUMN_SAMPLE_ID]
    metadata = metadata.filter(pl.col(COLUMN_SAMPLE_ID).is_in(samples_with_positions))

    # TODO: expose to allow ordering by a variable rather than coverage
    custom_xorder = None

    group_order = metadata.group_by(variable).len().sort(by="len")
    max_x = group_order["len"].sum()

    color_order = (
        metadata.select(pl.col(variable).unique().sort())
        .with_row_index(name="color")
        .select([pl.col(variable), pl.col("color")])
    )
    order = group_order.join(color_order, on=variable).sort(by="len")

    label_pos = []
    x_offset = 0
    boundaries = []
    tsv_x = []
    tsv_y = []
    tsv_group = []

    invert = len(order) == 2

    selection = [variable, "len", "color"]
    for oidx, (name, count, color) in enumerate(order[selection].iter_rows()):
        grp = metadata.filter(pl.col(variable) == name)
        color = f"C{color}"

        if custom_xorder is not None:
            grp_coverage = (
                grp.filter(pl.col(custom_xorder).is_not_null())
                .sort(by=custom_xorder)
                .with_row_index(name="x_unscaled", offset=x_offset)
                .with_columns(pl.lit(target).alias(COLUMN_GENOME_ID))
            )
        else:
            grp_coverage = ordered_coverage(coverage, grp, target, length)
            grp_coverage = grp_coverage.with_columns(pl.col("x_unscaled") + x_offset)

        if len(grp_coverage) == 0:
            continue

        # reverse plot order if we have two groups and in second group
        if invert and oidx == 1:
            grp_coverage = (
                grp_coverage.sort(by="x_unscaled", descending=True)
                .with_row_index("x_rev", offset=x_offset)
                .drop(pl.col("x_unscaled"))
                .with_columns(pl.col("x_rev").alias("x_unscaled"))
            )

        colors.append(color)

        hist_x = []
        hist_y = []

        col_selection = [COLUMN_SAMPLE_ID, "x_unscaled"]
        for sid, x in grp_coverage[col_selection].rows():
            cur_positions = (
                target_positions.filter(pl.col(COLUMN_SAMPLE_ID) == sid)
                .join(grp_coverage.lazy(), on=COLUMN_SAMPLE_ID)
                .select(
                    pl.col("x_unscaled"),
                    pl.col(COLUMN_START),
                    pl.col(COLUMN_STOP),
                )
            )

            if scale is None:
                covered_positions = get_covered(cur_positions.collect().to_numpy())
                lc = mc.LineCollection(
                    covered_positions, color=color, linewidths=0.5, alpha=0.7
                )
                ax.add_collection(lc)
            else:
                # obs_bins = position_histogram(cur_positions, scale, ymin, ymax)
                covered_positions = pl.concat(
                    [
                        cur_positions.select(pl.col("start").alias("common")),
                        cur_positions.select(pl.col("stop").alias("common")),
                    ]
                ).collect()

                obs_count, obs_bins = np.histogram(
                    covered_positions, bins=scale, range=(ymin, ymax)
                )
                obs_bins = obs_bins[:-1][obs_count > 0]
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
        df = pl.DataFrame({"group": tsv_group, "x": tsv_x, "y": tsv_y})
        filename = (
            f"{output}.{target_name}.{target}.{variable}."
            f"position-plot-1_{scale}th-scale.tsv.gz"
        )
        with gzip.open(filename, "wb") as fo:
            df.write_csv(fo, separator="\t")
        ax.set_title(f"Scaled position plot: {target} ({length}bp)", fontsize=20)
        ax.set_ylabel(f"Coverage (1/{scale})th scale", fontsize=20)
        scaletag = f"-1_{scale}th-scale"

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
