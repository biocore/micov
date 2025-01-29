"""microbiome coverage CLI."""

import click
import polars as pl
import duckdb
import os
import io
import sys
from glob import glob
from ._io import (parse_genome_lengths, parse_taxonomy, set_taxonomy_as_id,
                  parse_qiita_coverages, parse_sam_to_df, write_qiita_cov,
                  parse_sample_metadata, compress_from_stream,
                  parse_bed_cov_to_df, _single_df, _check_and_compress)
from ._cov import coverage_percent
from ._convert import cigar_to_lens
from ._per_sample import per_sample_coverage
from ._plot import (per_sample_plots, per_sample_plots_monte,
                    single_sample_position_plot)
from ._utils import logger
from ._constants import (COLUMN_SAMPLE_ID, COLUMN_GENOME_ID,
                         BED_COV_SAMPLEID_SCHEMA,
                         COLUMN_START, COLUMN_CIGAR, COLUMN_STOP)
from ._quant import pos_to_bins, make_csv_ready


def _first_col_as_set(fp):
    df = pl.read_csv(fp, separator='\t', infer_schema_length=0)
    return set(df[df.columns[0]])


@click.group()
def cli():
    """micov: microbiome coverage."""
    pass


@cli.command()
@click.option('--qiita-coverages', type=click.Path(exists=True), multiple=True,
              required=True, help='Pre-computed Qiita coverage data')
@click.option('--samples-to-keep', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the samples to keep')
@click.option('--samples-to-ignore', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the samples to ignore')
@click.option('--features-to-keep', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the features to keep')
@click.option('--features-to-ignore', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the features to ignore')
@click.option('--output', type=click.Path(exists=False), required=True)
@click.option('--lengths', type=click.Path(exists=True), required=True,
              help="Genome lengths")
def qiita_coverage(qiita_coverages, samples_to_keep, samples_to_ignore,
                   features_to_keep, features_to_ignore, output, lengths):
    """Compute aggregated coverage from one or more Qiita coverage files."""
    if samples_to_keep:
        samples_to_keep = _first_col_as_set(samples_to_keep)

    if samples_to_ignore:
        samples_to_ignore = _first_col_as_set(samples_to_ignore)

    if features_to_keep:
        features_to_keep = _first_col_as_set(features_to_keep)

    if features_to_ignore:
        features_to_ignore = _first_col_as_set(features_to_ignore)

    lengths = parse_genome_lengths(lengths)

    coverage = parse_qiita_coverages(qiita_coverages,
                                     sample_keep=samples_to_keep,
                                     sample_drop=samples_to_ignore,
                                     feature_keep=features_to_keep,
                                     feature_drop=features_to_ignore)
    coverage.write_csv(output + '.covered-positions.tsv', separator='\t',
                       include_header=True)

    genome_coverage = coverage_percent(coverage, lengths).collect()
    genome_coverage.write_csv(output + '.coverage.tsv', separator='\t',
                              include_header=True)


@cli.command()
@click.option('--data', type=click.Path(exists=True), required=False)
@click.option('--output', type=click.Path(exists=False))
@click.option('--disable-compression', is_flag=True, default=False,
              help='Do not compress the regions')
@click.option('--lengths', type=click.Path(exists=True), required=False,
              help='Genome lengths, if provided compute coverage')
@click.option('--taxonomy', type=click.Path(exists=True), required=False,
              help=('Genome taxonomy, if provided show species in coverage '
                    'percentage. Only works when --length is provided'))
def compress(data, output, disable_compression, lengths, taxonomy):
    """Compress BAM/SAM/BED mapping data.

    This command can work with pipes, e.g.:

    samtools view foo.bam | micov coverage | gzip > foo.cov.gz
    """
    if output == '-' or output is None:
        output = sys.stdout

    if lengths is not None:
        lengths = parse_genome_lengths(lengths)

        if taxonomy is not None:
            taxonomy = parse_taxonomy(taxonomy)

    if data is not None and os.path.isdir(data):
        file_list = (glob(data + "/*.sam")
                    + glob(data + '/*.sam.xz')
                    + glob(data + '/*.sam.gz'))
    else:
        file_list = [data]

    dfs = []
    for samfile in file_list:
        df = compress_from_stream(samfile,
                                    disable_compression=disable_compression)
        if df is None or len(df) == 0:
            logger.warning("File appears empty...")
        else:
            dfs.append(df)
    coverage = _single_df(_check_and_compress(dfs, compress_size=0))

    if lengths is None:
        coverage.write_csv(output, separator='\t', include_header=True)
    else:
        genome_coverage = coverage_percent(coverage, lengths).collect()

        if taxonomy is None:
            genome_coverage.write_csv(output, separator='\t',
                                      include_header=True)
        else:
            genome_coverage_with_taxonomy = set_taxonomy_as_id(genome_coverage,
                                                               taxonomy)
            genome_coverage_with_taxonomy.write_csv(output, separator='\t',
                                                    include_header=True)


@cli.command()
@click.option('--positions', type=click.Path(exists=True), required=False,
              help='BED3')
@click.option('--output', type=click.Path(exists=False), required=False)
@click.option('--lengths', type=click.Path(exists=True), required=True,
              help="Genome lengths")
def position_plot(positions, output, lengths):
    """Construct a single sample coverage plot."""
    if positions is None:
        data = sys.stdin
    else:
        data = open(positions, 'rb')

    lengths = parse_genome_lengths(lengths)
    df = parse_bed_cov_to_df(data)
    single_sample_position_plot(df, lengths, output)


@cli.command()
@click.option('--paths', type=click.Path(exists=True), required=True)
@click.option('--output', type=click.Path(exists=False))
@click.option('--lengths', type=click.Path(exists=True), required=True,
              help="Genome lengths")
def consolidate(paths, output, lengths):
    """Consolidate coverage files into a Qiita-like coverage.tgz."""
    paths = [path.strip() for path in open(paths)]
    for path in paths:
        if not os.path.exists(path):
            raise IOError(f"{path} not found")
    lengths = parse_genome_lengths(lengths)
    write_qiita_cov(output, paths, lengths)


@cli.command()
@click.option('--qiita-coverages', type=click.Path(exists=True), multiple=True,
              required=True, help='Pre-computed Qiita coverage data')
@click.option('--output', type=click.Path(exists=False))
@click.option('--lengths', type=click.Path(exists=True), required=True,
              help="Genome lengths")
@click.option('--samples-to-keep', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the sample metadata')
@click.option('--features-to-keep', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the features to keep')
@click.option('--features-to-ignore', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the features to ignore')
def qiita_to_parquet(qiita_coverages, lengths, output, samples_to_keep,
                     features_to_keep, features_to_ignore):
    """Aggregate Qiita coverage to parquet."""
    if features_to_keep:
        features_to_keep = _first_col_as_set(features_to_keep)

    if features_to_ignore:
        features_to_ignore = _first_col_as_set(features_to_ignore)

    if samples_to_keep:
        samples_to_keep = _first_col_as_set(samples_to_keep)

    lengths = parse_genome_lengths(lengths)
    covered_positions, coverage = per_sample_coverage(qiita_coverages,
                                                      samples_to_keep,
                                                      features_to_keep,
                                                      features_to_ignore,
                                                      lengths)

    coverage.collect().write_parquet(output + '.coverage.parquet',
                                     compression='zstd',
                                     compression_level=3)  # default afaik
    covered_positions.write_parquet(output + '.covered_positions.parquet',
                                    compression='zstd',
                                    compression_level=3)  # default afaik


@cli.command()
@click.option('--parquet-coverage', type=click.Path(exists=False),
              required=True, help=('Pre-computed coverage data as parquet. '
                                   'This should be the basename used, i.e. '
                                   'for "foo.coverage.parquet", please use '
                                   '"foo"'))
@click.option('--sample-metadata', type=click.Path(exists=True),
              required=True,
              help='A metadata file with the sample metadata')
@click.option('--sample-metadata-column', type=str,
              required=True,
              help='The column to consider in the sample metadata')
@click.option('--features-to-keep', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the features to keep')
@click.option('--output', type=click.Path(exists=False), required=True)
@click.option('--plot', is_flag=True, default=False,
              help='Generate plots from features')
def per_sample_group(parquet_coverage, sample_metadata, sample_metadata_column,
                     features_to_keep, output, plot):
    """Generate sample group plots and coverage data."""
    _load_db(parquet_coverage, sample_metadata, features_to_keep)

    all_covered_positions = duckdb.sql("SELECT * from covered_positions").pl()
    all_coverage = duckdb.sql("SELECT * FROM coverage").pl()
    metadata_pl = duckdb.sql("SELECT * FROM metadata").pl()

    per_sample_plots(all_coverage, all_covered_positions, metadata_pl,
                     sample_metadata_column, output)


@cli.command()
@click.option('--parquet-coverage', type=click.Path(exists=False),
              required=True, help=('Pre-computed coverage data as parquet. '
                                   'This should be the basename used, i.e. '
                                   'for "foo.coverage.parquet", please use '
                                   '"foo"'))
@click.option('--sample-metadata', type=click.Path(exists=True),
              required=True,
              help='A metadata file with the sample metadata')
@click.option('--sample-metadata-column', type=str,
              required=True,
              help='The column to consider in the sample metadata')
@click.option('--features-to-keep', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the features to keep')
@click.option('--iters', type=int, default=10, required=False)
@click.option('--target-names', type=str, required=False)
@click.option('--output', type=click.Path(exists=False), required=True)
@click.option('--plot', is_flag=True, default=False,
              help='Generate plots from features')
def per_sample_monte(parquet_coverage, sample_metadata, sample_metadata_column,
                     features_to_keep, output, plot, iters, target_names):
    """Generate sample group plots and coverage data with a null curve."""
    _load_db(parquet_coverage, sample_metadata, features_to_keep)

    all_covered_positions = duckdb.sql("SELECT * from covered_positions").pl()
    all_coverage = duckdb.sql("SELECT * FROM coverage").pl()
    metadata_pl = duckdb.sql("SELECT * FROM metadata").pl()

    if target_names is not None:
        target_names = dict(pl.scan_csv(target_names,
                                        separator='\t',
                                        new_columns=['feature-id', 'lineage'],
                                        has_header=False)
                              .with_columns(pl.col('lineage')
                                              .str
                                              .split(';')
                                              .list
                                              .get(-1)
                                              .str
                                              .replace_all(r" |\[|\]", "_")
                                              .alias('species'))
                              .select('feature-id', 'species')
                              .collect()
                              .iter_rows())
    else:
        sql = "SELECT DISTINCT genome_id FROM coverage"
        target_names = {k[0]: k[0] for k in duckdb.sql(sql).fetchall()}

    per_sample_plots_monte(all_coverage, all_covered_positions, metadata_pl,
                           sample_metadata_column, output, target_names, iters)


def _load_db(dbbase, sample_metadata, features_to_keep):
    metadata_pl = parse_sample_metadata(sample_metadata)
    sample_column = metadata_pl.columns[0]
    metadata_pl = metadata_pl.rename({sample_column: COLUMN_SAMPLE_ID})

    samples = tuple(metadata_pl[sample_column].unique())

    sfilt = f'WHERE sample_id IN {samples}'
    if features_to_keep:
        features_to_keep = tuple(_first_col_as_set(features_to_keep))
        sgfilt = f"{sfilt} AND genome_id IN {features_to_keep}"
    else:
        sgfilt = sfilt

    duckdb.sql(f"""CREATE VIEW coverage
                   AS SELECT *
                   FROM '{dbbase}.coverage.parquet'
                   {sgfilt}""")
    duckdb.sql(f"""CREATE VIEW covered_positions
                   AS SELECT *
                   FROM '{dbbase}.covered_positions.parquet'
                   {sgfilt}""")
    duckdb.sql(f"""CREATE TABLE metadata
                   AS SELECT *
                   FROM metadata_pl
                   {sfilt}
                   AND {COLUMN_SAMPLE_ID} IN (SELECT DISTINCT {COLUMN_SAMPLE_ID}
                                              FROM coverage)""")


@cli.command()
@click.option('-pos', '--covered-positions', type=click.Path(exists=True),
              required=True,
              help='Covered positions calculated from one or more samples')
@click.option('-o', '--outdir', type=click.Path(exists=True),
              required=True, help="Output directory. If new, will be created.")
@click.option('-g', '--genome-id', type=str,
              required=True, help="Genome ID of the genome of interest")
@click.option('-l', '--genome-length', type=int,
              required=True, help="Length of the genome of interest")
@click.option('-n', '--bin-num', type=int, default=1000,
              required=False, help="Number of bins")
def binning(covered_positions, outdir, genome_id, genome_length, bin_num):
    """Bin genome positions and quantify read and sample hits across bins."""
    pos = pl.read_csv(covered_positions, separator='\t',
                      new_columns=BED_COV_SAMPLEID_SCHEMA.columns,
                      schema_overrides=BED_COV_SAMPLEID_SCHEMA.dtypes_dict,
                      has_header=False, skip_rows=1).lazy()
    pos = pos.filter(pl.col(COLUMN_GENOME_ID).is_in([genome_id]))

    bin_df, pos_updated = pos_to_bins(pos, genome_length, bin_num)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    bin_df = make_csv_ready(bin_df)
    pos_updated = make_csv_ready(pos_updated)

    bin_df.write_csv(f"{outdir}/bin_stats.tsv", separator="\t", include_header=True)
    pos_updated.write_csv(f"{outdir}/pos_binned.tsv", separator="\t",
                          include_header=True)


if __name__ == '__main__':
    cli()
