import polars as pl

from ._constants import COLUMN_SAMPLE_ID
from ._cov import compress, coverage_percent
from ._io import parse_qiita_coverages


def per_sample_coverage(qiita_coverages, current_samples, features_to_keep,
                        features_to_ignore, lengths):
    try:
        coverage = parse_qiita_coverages(qiita_coverages,
                                         sample_keep=current_samples,
                                         feature_keep=features_to_keep,
                                         feature_drop=features_to_ignore,
                                         compress_size=None,
                                         append_sample_id=True)
    except ValueError:
        # we expect this to only occur when requested samples or features
        # are not present
        return None, None

    return coverage, compress_per_sample(coverage, lengths)


def compress_per_sample(coverage, lengths):
    sample_contig_coverage = []
    for (sample, ), sample_grp in coverage.group_by([COLUMN_SAMPLE_ID, ]):
        compressed = compress(sample_grp)
        cov_per = coverage_percent(compressed, lengths)
        cov_per = cov_per.with_columns(pl.lit(sample).alias(COLUMN_SAMPLE_ID))
        sample_contig_coverage.append(cov_per)

    if len(sample_contig_coverage) == 0:
        return None
    else:
        return pl.concat(sample_contig_coverage, rechunk=True)
