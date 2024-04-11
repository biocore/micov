from ._io import parse_qiita_coverages
from ._constants import COLUMN_SAMPLE_ID
from ._cov import coverage_percent
import polars as pl


def per_sample_coverage(qiita_coverages, current_samples, features_to_keep,
                        features_to_ignore, lengths):
    coverage = parse_qiita_coverages(qiita_coverages,
                                     sample_keep=current_samples,
                                     feature_keep=features_to_keep,
                                     feature_drop=features_to_ignore,
                                     compress_size=None, append_sample_id=True)

    sample_contig_coverage = []
    for (sample, ), sample_grp in coverage.group_by([COLUMN_SAMPLE_ID, ]):
        cov_per = coverage_percent(sample_grp, lengths)
        cov_per = cov_per.with_columns(pl.lit(sample).alias(COLUMN_SAMPLE_ID))
        sample_contig_coverage.append(cov_per)

    return coverage, pl.concat(sample_contig_coverage)
