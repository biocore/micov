import matplotlib.pyplot as plt
import numba
import numpy as np
from matplotlib import collections as mc
import polars as pl
import scipy.stats as ss
from ._cov import coverage_percent, compress
from ._constants import (COLUMN_SAMPLE_ID, COLUMN_GENOME_ID,
                         COLUMN_PERCENT_COVERED, COLUMN_LENGTH,
                         COLUMN_COVERED,
                         BED_COV_SCHEMA, COLUMN_START, COLUMN_STOP)


def ordered_coverage(coverage, grp, target, length):
    coverage = coverage.lazy()
    grp = grp.lazy()

    on_target = (coverage.join(grp, on=COLUMN_SAMPLE_ID)
                         .filter(pl.col(COLUMN_GENOME_ID) == target)
                         .collect())

    on_target_sids = on_target[COLUMN_SAMPLE_ID]

    off_target = (grp.filter(~(pl.col(COLUMN_SAMPLE_ID)
                                 .is_in(on_target_sids)))
                     .with_columns(pl.lit(0.)
                                     .alias(COLUMN_PERCENT_COVERED),
                                   pl.lit(0)
                                     .cast(int)
                                     .alias(COLUMN_COVERED),
                                   pl.lit(length)
                                     .cast(int)
                                     .alias(COLUMN_LENGTH),
                                   pl.lit(target)
                                     .alias(COLUMN_GENOME_ID))
                     .select(on_target.columns))

    return (pl.concat([on_target.lazy(), off_target])
              .sort(COLUMN_PERCENT_COVERED)
              .with_row_index()
              .with_columns(x=pl.col('index') / pl.len(),
                            x_unscaled=pl.col('index'))
              .collect())


def slice_positions(positions, id_):
    return (positions
                .lazy()
                .filter(pl.col(COLUMN_SAMPLE_ID) == id_)
                .select(pl.col(COLUMN_GENOME_ID), pl.col(COLUMN_START),
                        pl.col(COLUMN_STOP)))


def per_sample_plots(all_coverage, all_covered_positions, metadata,
                     sample_metadata_column, output, monte, monte_iters,
                     target_lookup):
    for genome in all_coverage[COLUMN_GENOME_ID].unique():
        target_name = target_lookup[genome]

        coverage_curve(metadata, all_coverage, all_covered_positions, genome,
                       sample_metadata_column, output, target_name, monte_iters,
                       monte, False)
        coverage_curve(metadata, all_coverage, all_covered_positions, genome,
                       sample_metadata_column, output, target_name, monte_iters,
                       monte, True)
        position_plot(metadata, all_coverage, all_covered_positions, genome,
                      sample_metadata_column, output, target_name, scale=None)
        position_plot(metadata, all_coverage, all_covered_positions, genome,
                      sample_metadata_column, output, target_name, scale=10000)


def per_sample_plots_monte(all_coverage, all_covered_positions, metadata,
                     sample_metadata_column, output, target_lookup, iters):
    for genome in all_coverage[COLUMN_GENOME_ID].unique():
        target_name = target_lookup[genome]
        cumulative_monte(metadata, all_coverage, all_covered_positions, genome,
                         sample_metadata_column, output, target_name, iters)


def compute_cumulative(coverage, grp, target, target_positions, lengths):
    lengths = coverage[[COLUMN_GENOME_ID, COLUMN_LENGTH]].unique()

    if len(lengths) > 1:
        raise ValueError("More than one length provided for the genome")

    length = lengths[COLUMN_LENGTH].item(0)

    current = pl.DataFrame([], schema=BED_COV_SCHEMA.dtypes_flat)
    grp_coverage = ordered_coverage(coverage, grp, target, length)

    if len(grp_coverage) == 0:
        return None, None

    cur_y = []
    cur_x = grp_coverage['x_unscaled']
    for id_ in grp_coverage[COLUMN_SAMPLE_ID]:
        next_ = slice_positions(target_positions, id_).collect()
        current = compress(pl.concat([current, next_]))
        per_cov = coverage_percent(current, lengths).collect()

        if len(per_cov) == 0:
            val = 0.
        else:
            val = per_cov[COLUMN_PERCENT_COVERED].item(0)

        cur_y.append(val)
    return cur_x, cur_y


def add_monte(monte_type, ax, max_x, iters, metadata_full, target,
              target_positions, coverage_full, accumulate, lengths):
    length = (lengths.filter(pl.col(COLUMN_GENOME_ID) == target)
                     .select(pl.col(COLUMN_LENGTH))
                     .row(0)[0])

    color = 'k'
    line_alpha = 0.6
    fill_alpha = 0.1

    if monte_type == 'focused':
        ls_median = 'dotted'
        ls_bound = '--'

        sample_set = (coverage_full.lazy()
                                   .filter(pl.col(COLUMN_GENOME_ID) == target)
                                   .select(pl.col(COLUMN_SAMPLE_ID))
                                   .collect())

    elif monte_type == 'unfocused':
        ls_median = 'dashed'
        ls_bound = '-.'

        sample_set = coverage_full.select(pl.col(COLUMN_SAMPLE_ID).unique())
    else:
        raise ValueError(f"Unknown monte_type='{monte_type}'")

    coverage = coverage_full.filter(pl.col(COLUMN_GENOME_ID) == target)
    metadata = metadata_full.filter(pl.col(COLUMN_SAMPLE_ID)
                                      .is_in(sample_set))

    monte_y = []
    monte_x = list(range(1, max_x + 1))

    for it in range(iters):
        monte = (metadata.select(pl.col(COLUMN_SAMPLE_ID)
                                   .shuffle())
                         .head(max_x))
        grp_monte = metadata.filter(pl.col(COLUMN_SAMPLE_ID)
                                      .is_in(monte))

        if accumulate:
            _, cur_y = compute_cumulative(coverage, grp_monte, target,
                                          target_positions, lengths)
        else:
            grp_coverage = ordered_coverage(coverage, grp_monte, target,
                                            length)
            cur_y = grp_coverage[COLUMN_PERCENT_COVERED].to_list()
        monte_y.append(cur_y)

    monte_y = np.asarray(monte_y)
    median = np.median(monte_y, axis=0)
    std = np.std(monte_y, axis=0)

    ax.plot(monte_x, median, color=color, linestyle=ls_median, linewidth=1,
             alpha=line_alpha)
    ax.plot(monte_x, median + std, color=color, linestyle=ls_bound, linewidth=1,
             alpha=line_alpha)
    ax.plot(monte_x, median - std, color=color, linestyle=ls_bound, linewidth=1,
             alpha=line_alpha)
    ax.fill_between(monte_x, median - std, median + std, color=color,
                    alpha=fill_alpha)
    return f'Monte Carlo {monte_type} (n={len(monte_x)})'

def coverage_curve(metadata_full, coverage_full, positions, target, variable, output,
                   target_name, iters=None, with_monte=None,
                   accumulate=False, min_group_size=10):
    if with_monte is not None and iters is None:
        raise ValueError("Running with Monte Carlo but no iterations set")

    if min_group_size < 0:
        raise ValueError("min_group_size must be greater than 0")

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_prop_cycle(None)

    labels = []

    target_positions = positions.filter(pl.col(COLUMN_GENOME_ID) == target)
    coverage = coverage_full.filter(pl.col(COLUMN_GENOME_ID) == target)
    cov_samples = coverage.select(pl.col(COLUMN_SAMPLE_ID).unique())
    metadata = metadata_full.filter(pl.col(COLUMN_SAMPLE_ID).is_in(cov_samples))

    if len(target_positions) == 0:
        raise ValueError("Target genome has no associated coverage")

    if len(coverage) == 0:
        raise ValueError("No sample has coverage on the target genome")

    lengths = coverage[[COLUMN_GENOME_ID, COLUMN_LENGTH]].unique()

    if len(lengths) > 1:
        raise ValueError("More than one length provided for the genome")

    length = lengths[COLUMN_LENGTH].item(0)
    value_order = metadata.select(pl.col(variable)
                                    .unique()
                                    .sort())[variable]

    max_x = 0
    for name, color in zip(value_order, range(0, 10)):
        color = f'C{color}'

        grp = metadata.filter(pl.col(variable) == name)

        n = len(grp)
        if n < min_group_size:
            continue

        if accumulate:
            cur_x, cur_y = compute_cumulative(coverage, grp, target,
                                              target_positions, lengths)
        else:
            grp_coverage = ordered_coverage(coverage, grp, target, length)
            cur_x = grp_coverage['x_unscaled']
            cur_y = grp_coverage[COLUMN_PERCENT_COVERED]

        if cur_x is None:
            continue

        max_x = max(max_x, cur_x.max())

        labels.append(f"{name} (n={len(cur_x)})")
        ax.plot(cur_x, cur_y, color=color)

    if not labels:
        return

    if with_monte is not None:
        label = add_monte(with_monte, ax, max_x, iters, metadata_full, target,
                          target_positions, coverage_full, accumulate,
                          lengths)
        labels.append(label)

    if accumulate:
        tag = 'cumulative'
    else:
        tag = 'non-cumulative'

    ax.set_ylabel('Percent genome covered', fontsize=16)
    ax.set_xlabel('Within group sample rank by coverage', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, 100)
    ax.set_title((f'{tag}: {target_name}({target}) '
                  f'({length}bp)'), fontsize=16)
    ax.legend(labels, fontsize=14)

    plt.tight_layout()

    if with_monte is not None:
        tag = f"{tag}-monte-{with_monte}"

    plt.savefig(f'{output}.{target_name}.{target}.{variable}.{tag}.png')
    plt.close()


@numba.jit(nopython=True)
def _get_covered(x_start_stop):
    return [[(x, start), (x, stop)] for (x, start, stop) in x_start_stop]


def single_sample_position_plot(positions, lengths, output, scale=None):
    positions = (positions
                    .lazy()
                    .join(lengths.lazy(), on=COLUMN_GENOME_ID)
                    .with_columns(x=0.5)).collect()
    for (name, grp) in positions.group_by(COLUMN_GENOME_ID):
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        grp = (grp.lazy()
                .sort(by=COLUMN_START)
                .select(pl.col('x'),
                        pl.col(COLUMN_START) / pl.col(COLUMN_LENGTH),
                        pl.col(COLUMN_STOP) / pl.col(COLUMN_LENGTH)))

        covered_positions = _get_covered(grp.collect().to_numpy())
        lc = mc.LineCollection(covered_positions,
                               linewidths=2, alpha=0.7)
        ax.add_collection(lc)

        ax.set_xlim(-0.01, 1.0)
        ax.set_ylim(0, 1.0)

        ax.set_title(f'Position plot: {name}', fontsize=20)
        ax.set_ylabel('Unit normalized position', fontsize=20)
        scaletag = ""

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        plt.tight_layout()
        plt.savefig(f'{output}.{name}.position-plot.png')
        plt.close()


def position_plot(metadata, coverage, positions, target, variable, output,
                  target_name, scale=None):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    labels = []
    colors = []

    lengths = coverage.filter(pl.col(COLUMN_GENOME_ID) == target)
    lengths = lengths[[COLUMN_GENOME_ID, COLUMN_LENGTH]].unique()

    if len(lengths) > 1:
        raise ValueError("More than one length provided for the genome")

    length = lengths[COLUMN_LENGTH].item(0)

    target_positions = positions.filter(pl.col(COLUMN_GENOME_ID) == target).lazy()

    value_order = metadata.select(pl.col(variable)
                                    .unique()
                                    .sort())[variable]
    for name, color in zip(value_order, range(0, 10)):
        grp = metadata.filter(pl.col(variable) == name)
        color = f'C{color}'
        grp_coverage = ordered_coverage(coverage, grp, target, length)

        if len(grp_coverage) == 0:
            continue

        labels.append(name)
        colors.append(color)

        hist_x = []
        hist_y = []

        col_selection = [COLUMN_SAMPLE_ID, COLUMN_GENOME_ID, 'x']
        for sid, gid, x in grp_coverage[col_selection].rows():
            cur_positions = (target_positions
                                 .filter(pl.col(COLUMN_SAMPLE_ID) == sid)
                                 .join(grp_coverage.lazy(), on=COLUMN_SAMPLE_ID)
                                 .select(pl.col('x'),
                                         pl.col(COLUMN_START) / length,
                                         pl.col(COLUMN_STOP) / length))

            if scale is None:
                covered_positions = _get_covered(cur_positions.collect().to_numpy())
                lc = mc.LineCollection(covered_positions, color=color,
                                       linewidths=0.5, alpha=0.7)
                ax.add_collection(lc)
            else:
                covered_positions = pl.concat([
                    cur_positions.select(pl.col('start').alias('common')),
                    cur_positions.select(pl.col('stop').alias('common'))
                ]).collect()

                obs_count, obs_bins = np.histogram(covered_positions,
                                                   bins=scale,
                                                   range=(0, 1))
                obs_bins = obs_bins[:-1][obs_count > 0]
                hist_x.extend([x for _ in obs_bins])
                hist_y.extend([b for b in obs_bins])

        if scale is not None:
            ax.scatter(hist_x, hist_y, s=.2, color=color, alpha=0.7)

    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1.0)

    if scale is None:
        ax.set_title(f'Position plot: {target} ({length}bp)', fontsize=20)
        ax.set_ylabel('Unit normalized genome position', fontsize=20)
        scaletag = ""
    else:
        ax.set_title(f'Scaled position plot: {target} ({length}bp)', fontsize=20)
        ax.set_ylabel(f'Coverage (1/{scale})th scale', fontsize=20)
        scaletag = f"-1_{scale}th-scale"

    ax.set_xlabel('Proportion of samples', fontsize=20)

    plt.legend(labels, fontsize=20)
    leg = ax.get_legend()
    for i, lh in enumerate(leg.legend_handles):
        lh.set_color(colors[i])
        lh._sizes = [5.0, ]

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'{output}.{target_name}.{target}.{variable}.position-plot{scaletag}.png')
    plt.close()
