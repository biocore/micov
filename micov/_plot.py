import matplotlib.pyplot as plt
import numba
import numpy as np
from matplotlib import collections as mc
import polars as pl
import scipy.stats as ss
from ._cov import coverage_percent, compress
from ._constants import (COLUMN_SAMPLE_ID, COLUMN_GENOME_ID,
                         COLUMN_PERCENT_COVERED, COLUMN_LENGTH,
                         BED_COV_SCHEMA, COLUMN_START, COLUMN_STOP)


def ordered_coverage(coverage, grp, target):
    return (coverage.lazy()
        .join(grp.lazy(), on=COLUMN_SAMPLE_ID)
        .filter(pl.col(COLUMN_GENOME_ID) == target)
        .sort(COLUMN_PERCENT_COVERED)
        .with_row_index()
        .with_columns(x=pl.col('index') / pl.len())).collect()


def slice_positions(positions, id_):
    return (positions
                .lazy()
                .filter(pl.col(COLUMN_SAMPLE_ID) == id_)
                .select(pl.col(COLUMN_GENOME_ID), pl.col(COLUMN_START),
                        pl.col(COLUMN_STOP)))


def per_sample_plots(all_coverage, all_covered_positions, metadata,
                     sample_metadata_column, output):
    for genome in all_coverage[COLUMN_GENOME_ID].unique():
        non_cumulative(metadata, all_coverage, genome, sample_metadata_column,
                       output)
        cumulative(metadata, all_coverage, all_covered_positions, genome,
                   sample_metadata_column, output)
        position_plot(metadata, all_coverage, all_covered_positions, genome,
                      sample_metadata_column, output, scale=None)
        position_plot(metadata, all_coverage, all_covered_positions, genome,
                      sample_metadata_column, output, scale=10000)


def cumulative(metadata, coverage, positions, target, variable, output):
    plt.figure(figsize=(12, 8))
    labels = []
    covs = []

    target_positions = positions.filter(pl.col(COLUMN_GENOME_ID) == target)
    coverage = coverage.filter(pl.col(COLUMN_GENOME_ID) == target)

    if len(target_positions) == 0:
        raise ValueError()

    if len(coverage) == 0:
        raise ValueError()

    lengths = coverage[[COLUMN_GENOME_ID, COLUMN_LENGTH]].unique()

    if len(lengths) > 1:
        raise ValueError()

    length = lengths[COLUMN_LENGTH].item(0)

    for (name, ), grp in metadata.group_by([variable, ], maintain_order=True):
        current = pl.DataFrame([], schema=BED_COV_SCHEMA.dtypes_flat)

        grp_coverage = ordered_coverage(coverage, grp, target)

        if len(grp_coverage) == 0:
            continue
        labels.append(name)

        cur_y = []
        cur_x = grp_coverage['x']
        for id_ in grp_coverage[COLUMN_SAMPLE_ID]:
            next_ = slice_positions(target_positions, id_).collect()
            current = compress(pl.concat([current, next_]))
            per_cov = coverage_percent(current, lengths).collect()
            cur_y.append(per_cov[COLUMN_PERCENT_COVERED].item(0))

        covs.append(cur_y)
        plt.plot(cur_x, cur_y)

    if len(covs) > 1:
        k, p = ss.kruskal(*covs)
        stat = f'\nKruskal: stat={k:.2f}; p={p:.2e}'
    else:
        stat = ''

    ax = plt.gca()
    ax.set_ylabel('Percent genome covered', fontsize=20)
    ax.set_xlabel('Proportion of samples', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_title((f'Cumulative: {target} '
                  f'({length}bp){stat}'), fontsize=20)
    plt.legend(labels, fontsize=20)

    plt.tight_layout()
    plt.savefig(f'{output}.{target}.{variable}.cumulative.png')
    plt.close()


def non_cumulative(metadata, coverage, target, variable, output):
    plt.figure(figsize=(12, 8))
    labels = []
    covs = []

    for (name, ), grp in metadata.group_by([variable, ], maintain_order=True):
        grp_coverage = ordered_coverage(coverage, grp, target)

        if len(grp_coverage) == 0:
            continue

        labels.append(name)
        covs.append(grp_coverage[COLUMN_PERCENT_COVERED])
        plt.plot(grp_coverage['x'], grp_coverage[COLUMN_PERCENT_COVERED])

        # assumes all the same contig, so if `ordered_coverage` changes than
        # this assumption would not be valid
        # n.b. intentionally leaking `length`
        length = grp_coverage[COLUMN_LENGTH].item(0)

    if len(covs) > 1:
        k, p = ss.kruskal(*covs)
        stat = f'\nKruskal: stat={k:.2f}; p={p:.2e}'
    else:
        stat = ''

    ax = plt.gca()
    ax.set_ylabel('Percent genome covered', fontsize=20)
    ax.set_xlabel('Proportion of samples', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_title((f'Non-cumulative: {target} '
                  f'({length}bp){stat}'), fontsize=20)
    plt.legend(labels, fontsize=20)

    plt.tight_layout()
    plt.savefig(f'{output}.{target}.{variable}.non-cumulative.png')
    plt.close()


@numba.jit(nopython=True)
def _get_covered(x_start_stop):
    return [[(x, start), (x, stop)] for (x, start, stop) in x_start_stop]


def position_plot(metadata, coverage, positions, target, variable, output, scale=None):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    labels = []
    colors = []
    target_positions = positions.filter(pl.col(COLUMN_GENOME_ID) == target).lazy()

    for ((name, ), grp), color in zip(metadata.group_by([variable, ],
                                                        maintain_order=True),
                                      range(0, 10)):
        color = f'C{color}'
        grp_coverage = ordered_coverage(coverage, grp, target)

        labels.append(name)
        colors.append(color)
        length = grp_coverage[COLUMN_LENGTH].item(0)

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
    for i, lh in enumerate(leg.legendHandles):
        lh.set_color(colors[i])

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'{output}.{target}.{variable}.position-plot{scaletag}.png')
    plt.close()
