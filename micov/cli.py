"""microbiome coverage CLI."""

import os
import sys
from glob import glob

import click
import duckdb
import polars as pl

from ._constants import (
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_SAMPLE_ID,
)
from ._cov import coverage_percent
from ._io import (
    _check_and_compress,
    _first_col_as_set,
    _single_df,
    compress_from_stream,
    parse_bed_cov_to_df,
    parse_feature_names,
    parse_features_to_keep,
    parse_genome_lengths,
    parse_qiita_coverages,
    parse_sample_metadata,
    parse_taxonomy,
    set_taxonomy_as_id,
    write_qiita_cov,
)
from ._per_sample import per_sample_coverage
from ._plot import per_sample_plots, single_sample_position_plot
from ._quant import make_csv_ready, pos_to_bins
from ._utils import logger
from ._view import View


@click.group()
def cli():
    """micov: microbiome coverage."""


@cli.command()
@click.option(
    "--qiita-coverages",
    type=click.Path(exists=True),
    multiple=True,
    required=True,
    help="Pre-computed Qiita coverage data",
)
@click.option(
    "--samples-to-keep",
    type=click.Path(exists=True),
    required=False,
    help="A metadata file with the samples to keep",
)
@click.option(
    "--samples-to-ignore",
    type=click.Path(exists=True),
    required=False,
    help="A metadata file with the samples to ignore",
)
@click.option(
    "--features-to-keep",
    type=click.Path(exists=True),
    required=False,
    help="A metadata file with the features to keep",
)
@click.option(
    "--features-to-ignore",
    type=click.Path(exists=True),
    required=False,
    help="A metadata file with the features to ignore",
)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option(
    "--lengths", type=click.Path(exists=True), required=True, help="Genome lengths"
)
def qiita_coverage(
    qiita_coverages,
    samples_to_keep,
    samples_to_ignore,
    features_to_keep,
    features_to_ignore,
    output,
    lengths,
):
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

    coverage = parse_qiita_coverages(
        qiita_coverages,
        sample_keep=samples_to_keep,
        sample_drop=samples_to_ignore,
        feature_keep=features_to_keep,
        feature_drop=features_to_ignore,
    )
    coverage.write_csv(
        output + ".covered-positions.tsv", separator="\t", include_header=True
    )

    genome_coverage = coverage_percent(coverage, lengths).collect()
    genome_coverage.write_csv(
        output + ".coverage.tsv", separator="\t", include_header=True
    )


@cli.command()
@click.option("--data", type=click.Path(exists=True), required=False)
@click.option("--output", type=click.Path(exists=False))
@click.option(
    "--disable-compression",
    is_flag=True,
    default=False,
    help="Do not compress the regions",
)
@click.option(
    "--lengths",
    type=click.Path(exists=True),
    required=False,
    help="Genome lengths, if provided compute coverage",
)
@click.option(
    "--taxonomy",
    type=click.Path(exists=True),
    required=False,
    help=(
        "Genome taxonomy, if provided show species in coverage "
        "percentage. Only works when --length is provided"
    ),
)
def compress(data, output, disable_compression, lengths, taxonomy):
    """Compress BAM/SAM/BED mapping data.

    This command can work with pipes, e.g.:

    samtools view foo.bam | micov coverage | gzip > foo.cov.gz
    """
    if output == "-" or output is None:
        output = sys.stdout

    if lengths is not None:
        lengths = parse_genome_lengths(lengths)

        if taxonomy is not None:
            taxonomy = parse_taxonomy(taxonomy)

    if data is not None and os.path.isdir(data):
        file_list = (
            glob(data + "/*.sam") + glob(data + "/*.sam.xz") + glob(data + "/*.sam.gz")
        )
    else:
        file_list = [data]

    dfs = []
    for samfile in file_list:
        df = compress_from_stream(samfile, disable_compression=disable_compression)
        if df is None or len(df) == 0:
            logger.warning("File appears empty...")
        else:
            dfs.append(df)
    coverage = _single_df(_check_and_compress(dfs, compress_size=0))

    if lengths is None:
        coverage.write_csv(output, separator="\t", include_header=True)
    else:
        genome_coverage = coverage_percent(coverage, lengths).collect()

        if taxonomy is None:
            genome_coverage.write_csv(output, separator="\t", include_header=True)
        else:
            genome_coverage_with_taxonomy = set_taxonomy_as_id(
                genome_coverage, taxonomy
            )
            genome_coverage_with_taxonomy.write_csv(
                output, separator="\t", include_header=True
            )


@cli.command()
@click.option("--positions", type=click.Path(exists=True), required=False, help="BED3")
@click.option("--output", type=click.Path(exists=False), required=False)
@click.option(
    "--lengths", type=click.Path(exists=True), required=True, help="Genome lengths"
)
def position_plot(positions, output, lengths):
    """Construct a single sample coverage plot."""
    if positions is None:
        data = sys.stdin
    else:
        data = open(positions, "rb")

    lengths = parse_genome_lengths(lengths)
    df = parse_bed_cov_to_df(data)
    single_sample_position_plot(df, lengths, output)


@cli.command()
@click.option("--paths", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(exists=False))
@click.option(
    "--lengths", type=click.Path(exists=True), required=True, help="Genome lengths"
)
def consolidate(paths, output, lengths):
    """Consolidate coverage files into a Qiita-like coverage.tgz."""
    paths = [path.strip() for path in open(paths)]
    for path in paths:
        if not os.path.exists(path):
            raise OSError(f"{path} not found")
    lengths = parse_genome_lengths(lengths)
    write_qiita_cov(output, paths, lengths)


@cli.command()
@click.option(
    "--qiita-coverages",
    type=click.Path(exists=True),
    multiple=True,
    required=True,
    help="Pre-computed Qiita coverage data",
)
@click.option("--output", type=click.Path(exists=False))
@click.option(
    "--lengths", type=click.Path(exists=True), required=True, help="Genome lengths"
)
@click.option(
    "--samples-to-keep",
    type=click.Path(exists=True),
    required=False,
    help="A metadata file with the sample metadata",
)
@click.option(
    "--features-to-keep",
    type=click.Path(exists=True),
    required=False,
    help="A metadata file with the features to keep",
)
@click.option(
    "--features-to-ignore",
    type=click.Path(exists=True),
    required=False,
    help="A metadata file with the features to ignore",
)
def qiita_to_parquet(
    qiita_coverages,
    lengths,
    output,
    samples_to_keep,
    features_to_keep,
    features_to_ignore,
):
    """Aggregate Qiita coverage to parquet."""
    if features_to_keep:
        features_to_keep = _first_col_as_set(features_to_keep)

    if features_to_ignore:
        features_to_ignore = _first_col_as_set(features_to_ignore)

    if samples_to_keep:
        samples_to_keep = _first_col_as_set(samples_to_keep)

    lengths = parse_genome_lengths(lengths)
    covered_positions, coverage = per_sample_coverage(
        qiita_coverages, samples_to_keep, features_to_keep, features_to_ignore, lengths
    )

    coverage.collect().write_parquet(
        output + ".coverage.parquet", compression="zstd", compression_level=3
    )  # default afaik
    covered_positions.write_parquet(
        output + ".covered_positions.parquet", compression="zstd", compression_level=3
    )  # default afaik


@cli.command()
@click.option(
    "--pattern",
    type=str,
    required=True,
    help="Glob pattern for BED3-like files. Must end " "in .cov or .cov.gz",
)
@click.option("--output", type=click.Path(exists=False))
@click.option(
    "--lengths", type=click.Path(exists=True), required=True, help="Genome lengths"
)
@click.option("--memory", type=str, default="16gb", required=False)
@click.option("--threads", type=int, default=4, required=False)
def nonqiita_to_parquet(pattern, lengths, output, memory, threads):
    """Aggregate BED3 files to parquet."""
    lengths = parse_genome_lengths(lengths)

    columns = "{'genome_id': 'VARCHAR', 'start': 'UINTEGER', 'stop': 'UINTEGER'}"
    # TODO: use a connection, passparams in as config dict
    duckdb.sql(f"SET memory_limit TO '{memory}'")
    duckdb.sql(f"SET threads TO {threads}")
    duckdb.sql("CREATE TABLE genome_lengths AS FROM lengths")

    # stream the .cov or .cov.gz files into parquet. Extract the name of the
    # file, without the extension, and store as the sample_id
    duckdb.sql(f"""
        COPY (SELECT genome_id,
                     start,
                     stop,
                     regexp_extract(filename,
                                    '^(.*/)?(.+).cov(.gz)?$', 2) AS sample_id
              FROM read_csv('{pattern}',
                            delim='\t',
                            filename=true,
                            header=true,
                            columns={columns}))
        TO '{output}.covered_positions.parquet'
            (FORMAT PARQUET, PARQUET_VERSION V2,
             COMPRESSION zstd)""")

    # scan the aggregated position information, compute the amount covered
    # and the percent coverage per sample per genome, stream to parquet.
    duckdb.sql(f"""
        COPY (WITH covered_amount AS (
                  SELECT sample_id,
                         genome_id,
                         SUM(stop - start)::UINTEGER AS covered
                  FROM read_parquet('{output}.covered_positions.parquet')
                  GROUP BY sample_id, genome_id)
              SELECT sample_id,
                     genome_id,
                     covered,
                     length,
                     (covered / length) * 100 AS percent_covered
              FROM covered_amount JOIN genome_lengths using (genome_id))
        TO '{output}.coverage.parquet'
            (FORMAT PARQUET, PARQUET_VERSION V2,
             COMPRESSION zstd)""")

    # n.b. a comparable action can be taken with polars. however, polars does
    # not currently allow limiting memory, and in testing, the use exceeded
    # 16gb. Running via the streaming engine may work though.
    # (pl.scan_csv(pattern,
    #             separator='\t',
    #             has_header=True,
    #             schema=pl.Schema({'genome_id': str,
    #                               'start': pl.UInt32,
    #                               'stop': pl.UInt32}),
    #             include_file_paths='filename')
    #   .with_columns(pl.col('filename')
    #                   .str.extract(r"(.+).cov.gz$")
    #                   .alias('sample_id'))
    #   .drop('filename')
    #   .sink_parquet(f"{output}.covered_positions_pl.parquet",
    #                 compression='zstd'))


@cli.command()
@click.option(
    "--parquet-coverage",
    type=click.Path(exists=False),
    required=True,
    help=(
        "Pre-computed coverage data as parquet. "
        "This should be the basename used, i.e. "
        'for "foo.coverage.parquet", please use '
        '"foo"'
    ),
)
@click.option(
    "--sample-metadata",
    type=click.Path(exists=True),
    required=True,
    help="A metadata file with the sample metadata",
)
@click.option(
    "--sample-metadata-column",
    type=str,
    required=True,
    help="The column to consider in the sample metadata",
)
@click.option(
    "--features-to-keep",
    type=click.Path(exists=True),
    required=True,
    help="A metadata file with the features to keep",
)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option(
    "--plot", is_flag=True, default=False, help="Generate plots from features"
)
@click.option(
    "--monte",
    type=click.Choice(["focused", "unfocused"]),
    required=False,
    default=None,
    help="Perform a Monte Carlo simulation for a coverage curve",
)
@click.option(
    "--monte-iters",
    type=int,
    required=False,
    default=100,
    help="The number of permutations to perform",
)
@click.option("--memory", type=str, default="16gb", required=False)
@click.option("--threads", type=int, default=4, required=False)
@click.option("--target-names", type=str, required=False)
@click.option("--percentile", is_flag=True, default=False, help="Use percentile")
def per_sample_group(
    parquet_coverage,
    sample_metadata,
    sample_metadata_column,
    features_to_keep,
    output,
    plot,
    monte,
    monte_iters,
    target_names,
    memory,
    threads,
    percentile,
):
    """Generate sample group plots and coverage data."""
    metadata_pl = parse_sample_metadata(sample_metadata)
    features_pl = parse_features_to_keep(features_to_keep)
    names_pl = parse_feature_names(target_names)
    view = View(
        parquet_coverage,
        metadata_pl,
        features_pl,
        names_pl,
        threads=threads,
        memory=memory,
    )

    per_sample_plots(
        view,
        sample_metadata_column,
        output,
        monte,
        monte_iters,
        percentile,
    )


@cli.command()
@click.option(
    "--parquet-coverage",
    type=str,
    required=True,
    help="Parquet file containing the covered positions data",
)
@click.option(
    "--sample-metadata",
    type=click.Path(exists=True),
    required=True,
    help="A metadata file with the sample metadata",
)
@click.option(
    "--features-to-keep",
    type=click.Path(exists=True),
    required=False,
    help="A file with the features to keep. Must have header",
)
@click.option(
    "--metadata-variable",
    type=str,
    required=True,
    help="The variable to consider in the sample metadata",
)
@click.option(
    "--outdir",
    type=click.Path(exists=False),
    required=True,
    help="Output directory for results",
)
@click.option(
    "--bin-num", type=int, default=1000, help="Number of bins (default: 1000)"
)
@click.option(
    "--rank", is_flag=True, default=False, help="Enable ranking (default: False)"
)
@click.option("--memory", type=str, default="16gb", required=False)
@click.option("--threads", type=int, default=4, required=False)
def binning(
    parquet_coverage,
    sample_metadata,
    features_to_keep,
    metadata_variable,
    outdir,
    bin_num,
    rank,
    memory,
    threads,
):
    """Bin genome positions and quantify read and sample hits across bins."""
    metadata_pl = parse_sample_metadata(sample_metadata)
    features_pl = parse_features_to_keep(features_to_keep)
    view = View(
        parquet_coverage, metadata_pl, features_pl, threads=threads, memory=memory
    )

    all_covered_positions = view.positions()
    metadata = view.metadata().select(COLUMN_SAMPLE_ID, metadata_variable)
    feature_metadata = view.feature_metadata().select(COLUMN_GENOME_ID, COLUMN_LENGTH)

    length_map = dict(feature_metadata.fetchall())
    genomes = set(length_map)

    df_bins_list = []
    for genome_id in genomes:
        length = length_map[genome_id]
        df_pos_md = (
            all_covered_positions.filter(f"{COLUMN_GENOME_ID} = '{genome_id}'")
            .join(metadata, COLUMN_SAMPLE_ID)
            .pl()
            .lazy()
        )
        df_bins = pos_to_bins(df_pos_md, metadata_variable, bin_num, length).collect()
        df_bins_list.append(df_bins)

    df_bins = pl.concat(df_bins_list).lazy()
    df_bins_by_sample_hits = (
        df_bins.group_by(COLUMN_GENOME_ID, "bin_idx", "bin_start", "bin_stop")
        .agg(pl.col("sample_hits").std().alias("sample_hits_std"))
        .fill_null(0)
        .sort("sample_hits_std", descending=True)
    )

    os.makedirs(outdir, exist_ok=True)
    make_csv_ready(df_bins).sink_csv(
        f"{outdir}/stats_bins.tsv",
        separator="\t",
        include_header=True,
    )

    # it's not obvious but this cannot use sink_csv?'
    df_bins_by_sample_hits.collect().write_csv(
        f"{outdir}/stats_by_variance_of_sample_hits.tsv",
        separator="\t",
        include_header=True,
    )


@cli.command()
@click.option(
    "--parquet-coverage",
    type=click.Path(exists=False),
    required=True,
    help=(
        "Pre-computed coverage data as parquet. "
        "This should be the basename used, i.e. "
        'for "foo.coverage.parquet", please use '
        '"foo"'
    ),
)
@click.option(
    "--sample-metadata",
    type=click.Path(exists=True),
    required=True,
    help="A metadata file with the sample metadata",
)
@click.option(
    "--features-to-keep",
    type=click.Path(exists=True),
    required=False,
    help="A metadata file with the features to keep",
)
@click.option("--output", type=click.Path(exists=False), required=True)
@click.option("--memory", type=str, default="16gb", required=False)
@click.option("--threads", type=int, default=4, required=False)
def extract_sample_presence(
    parquet_coverage,
    sample_metadata,
    features_to_keep,
    output,
    memory,
    threads,
):
    """Extract variables for each described feature to keep region."""
    metadata_pl = parse_sample_metadata(sample_metadata)
    features_pl = parse_features_to_keep(features_to_keep)
    view = View(
        parquet_coverage, metadata_pl, features_pl, threads=threads, memory=memory
    )

    view.sample_presence_absence().pl().write_csv(
        output, separator="\t", include_header=True
    )


if __name__ == "__main__":
    cli()
