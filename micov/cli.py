"""microbiome coverage CLI."""

import click
import polars as pl
import os
import io
import sys
import tqdm
from ._io import (parse_genome_lengths, parse_qiita_coverages, parse_sam_to_df,
                  write_qiita_cov, parse_sample_metadata, compress_from_stream,
                  parse_bed_cov_to_df)
from ._cov import coverage_percent
from ._convert import cigar_to_lens
from ._per_sample import per_sample_coverage
from ._plot import per_sample_plots, single_sample_position_plot


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
              help="Genome lengths, if provided compute coverage")
def compress(data, output, disable_compression, lengths):
    """Compress BAM/SAM/BED mapping data.

    This command can work with pipes, e.g.:

    samtools view foo.bam | micov coverage | gzip > foo.cov.gz
    """
    if output == '-' or output is None:
        output = sys.stdout

    if lengths is not None:
        lengths = parse_genome_lengths(lengths)

    # compress data in blocks to avoid loading full mapping data into memory
    # and compress as we go along.
    df = compress_from_stream(data, disable_compression=disable_compression)
    if df is None or len(df) == 0:
        click.echo("File appears empty...", err=True)
        sys.exit(0)

    if lengths is None:
        df.write_csv(output, separator='\t', include_header=True)
    else:
        genome_coverage = coverage_percent(df, lengths).collect()
        genome_coverage.write_csv(output, separator='\t', include_header=True)


@cli.command()
@click.option('--positions', type=click.Path(exists=True), required=False,
              help='BED3')
@click.option('--output', type=click.Path(exists=False), required=False)
@click.option('--lengths', type=click.Path(exists=True), required=True,
              help="Genome lengths")
def position_plot(positions, output, lengths):
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
@click.option('--sample-metadata', type=click.Path(exists=True),
              required=True,
              help='A metadata file with the sample metadata')
@click.option('--sample-metadata-column', type=str,
              required=True,
              help='The column to consider in the sample metadata')
@click.option('--features-to-keep', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the features to keep')
@click.option('--features-to-ignore', type=click.Path(exists=True),
              required=False,
              help='A metadata file with the features to ignore')
@click.option('--output', type=click.Path(exists=False), required=True)
@click.option('--lengths', type=click.Path(exists=True), required=True,
              help="Genome lengths")
def per_sample_group(qiita_coverages, sample_metadata, sample_metadata_column,
                     features_to_keep, features_to_ignore, output, lengths):
    """Generate sample group plots and coverage data."""
    if features_to_keep:
        features_to_keep = _first_col_as_set(features_to_keep)

    if features_to_ignore:
        features_to_ignore = _first_col_as_set(features_to_ignore)

    lengths = parse_genome_lengths(lengths)
    metadata = parse_sample_metadata(sample_metadata)
    sample_column = metadata.columns[0]

    if sample_metadata_column not in metadata.columns:
        raise KeyError(f"'{sample_metadata_column}' not found")

    if metadata[sample_metadata_column].dtype != pl.String:
        raise ValueError(f"Column must be categorical")

    if len(metadata[sample_metadata_column].unique()) > 10:
        raise ValueError(f"Not sure if this will work will with that many values")

    metadata = metadata[[sample_column, sample_metadata_column]]

    all_covered_positions = []
    all_coverage = []
    for (value, ), grp in metadata.group_by([sample_metadata_column, ]):
        current_samples = set(grp[sample_column].unique())
        covered_positions, coverage = per_sample_coverage(qiita_coverages,
                                                          current_samples,
                                                          features_to_keep,
                                                          features_to_ignore,
                                                          lengths)

        if covered_positions is None or coverage is None:
            click.echo(f"No coverage observed for: '{value}'", err=True)
            continue

        all_covered_positions.append(covered_positions)
        all_coverage.append(coverage)

    all_covered_positions = pl.concat(all_covered_positions)
    all_coverage = pl.concat(all_coverage).collect()

    all_coverage.write_csv(output + '.coverage', separator='\t',
                           include_header=True)
    all_covered_positions.write_csv(output + '.covered_positions', separator='\t',
                                    include_header=True)
    per_sample_plots(all_coverage, all_covered_positions, metadata,
                     sample_metadata_column, output)


if __name__ == '__main__':
    cli()
