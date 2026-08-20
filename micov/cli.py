"""microbiome coverage CLI."""

import copy
import os

import click

from ._constants import (
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)
from ._io import (
    ALIGNMENT_POSITIONS_TABLE,
    compress_alignments,
    load_bed_cov,
    load_genome_lengths,
    positions_path,
    write_coverage_parquet,
)
from ._miint import connection
from ._plot import per_sample_plots, single_sample_position_plot
from ._quant import pos_to_bins
from ._view import View


@click.group()
def cli():
    """micov: microbiome coverage."""


@cli.command()
@click.option(
    "--data",
    type=click.Path(exists=True),
    required=False,
    help="SAM/BAM file, or a directory of them. Omit to read stdin.",
)
@click.option(
    "--output",
    type=click.Path(exists=False),
    required=True,
    help="Base path for the .coverage.parquet / .covered_positions.parquet pair",
)
@click.option(
    "--sample-id",
    type=str,
    required=False,
    help=(
        "Sample ID. Defaults to the --data filename with its extensions "
        "stripped; required when reading stdin or a directory."
    ),
)
@click.option(
    "--disable-compression",
    is_flag=True,
    default=False,
    help="Do not merge overlapping intervals",
)
@click.option(
    "--lengths",
    type=click.Path(exists=True),
    required=True,
    help=(
        "Genome lengths. Also serves as the reference map: headerless SAM "
        "carries no header for htslib to resolve RNAME against."
    ),
)
@click.option("--memory", type=str, default="16gb", required=False)
@click.option("--threads", type=int, default=4, required=False)
def compress(data, output, sample_id, disable_compression, lengths, memory, threads):
    """Compress SAM/BAM alignments into per-sample coverage parquet.

    Writes `{output}.coverage.parquet` and `{output}.covered_positions.parquet`.

    This command can work with pipes, e.g.:

    xzcat foo.sam.xz | micov compress --lengths l.tsv --sample-id foo --output foo
    """
    if data is None:
        # DuckDB reads the pipe directly, so the documented xzcat idiom still
        # works without buffering the stream to disk first. That only holds
        # because --lengths supplies the reference map up front -- deriving it
        # from the data would need a second pass stdin cannot give.
        source = "/dev/stdin"
        origin = "stdin"
    elif os.path.isdir(data):
        # A directory is handed to htslib as one glob, so unlike the
        # single-file path there is nowhere to decompress `.xz`/`.bz2` first.
        # Left alone it fails as a bare `IO Error: Failed to open SAM file`
        # naming one arbitrary member, which does not say what to do about it.
        unreadable = sorted(
            name
            for name in os.listdir(data)
            if name.endswith((".xz", ".bz2"))
        )
        if unreadable:
            raise click.UsageError(
                f"{len(unreadable)} file(s) in '{data}' are xz/bz2 compressed "
                f"(e.g. {unreadable[0]}), which htslib cannot read, and a "
                "directory is passed to it as a single glob so micov cannot "
                "decompress them first. Point --data at one file at a time, or "
                "pipe them in: `xzcat f.sam.xz | micov compress --sample-id ...`."
            )
        source = os.path.join(data, "*.sam*")
        origin = "a directory"
    else:
        source = data
        origin = None

    if sample_id is None:
        if origin is not None:
            raise click.UsageError(
                f"--sample-id is required when reading from {origin}: there is "
                "no single filename to take the sample ID from, and "
                "coverage.parquet is keyed by it."
            )
        sample_id = _sample_id_from_path(data)

    con = connection(memory=memory, threads=threads)
    load_genome_lengths(con, lengths)
    compress_alignments(
        con, source, sample_id, disable_compression=disable_compression
    )
    write_coverage_parquet(
        con, f"SELECT * FROM {ALIGNMENT_POSITIONS_TABLE}", output
    )


def _sample_id_from_path(path):
    """Derive a sample ID from an alignment filename.

    `foo/bar/baz.sam.xz` becomes `baz`, matching how
    `cov-to-parquet` reads a sample ID out of a `.cov` filename and how
    the README's loop names its outputs.
    """
    name = os.path.basename(path)
    for suffix in (".gz", ".xz", ".bz2"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for suffix in (".sam", ".bam", ".cram"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


@cli.command()
@click.option("--positions", type=click.Path(exists=True), required=False, help="BED3")
@click.option("--output", type=click.Path(exists=False), required=False)
@click.option(
    "--lengths", type=click.Path(exists=True), required=True, help="Genome lengths"
)
def position_plot(positions, output, lengths):
    """Construct a single sample coverage plot."""
    con = connection()
    load_genome_lengths(con, lengths)
    # stdin is spooled to a file first; DuckDB's CSV reader sniffs the dialect
    # and cannot rewind a pipe afterwards. See `positions_path`.
    with positions_path(positions) as path:
        load_bed_cov(con, path)
    single_sample_position_plot(con, output)


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
def cov_to_parquet(pattern, lengths, output, memory, threads):
    """Aggregate BED3 files to parquet."""
    columns = "{'genome_id': 'VARCHAR', 'start': 'UINTEGER', 'stop': 'UINTEGER'}"

    # was DuckDB's implicit default connection, which made this command
    # non-reentrant: `genome_lengths` was created on a process-global catalog,
    # so a second in-process call hit "Table with name genome_lengths already
    # exists". Going through the helper gives each invocation its own catalog
    # and puts every micov connection in one place.
    con = connection(memory=memory, threads=threads)
    load_genome_lengths(con, lengths)

    # stream the .cov or .cov.gz files into parquet. Extract the name of the
    # file, without the extension, and store as the sample_id
    positions = f"""SELECT {COLUMN_GENOME_ID},
                           {COLUMN_START},
                           {COLUMN_STOP},
                           regexp_extract(filename,
                                          '^(.*/)?(.+).cov(.gz)?$', 2)
                               AS {COLUMN_SAMPLE_ID}
                    FROM read_csv('{pattern}',
                                  delim='\t',
                                  filename=true,
                                  header=true,
                                  columns={columns})"""
    write_coverage_parquet(con, positions, output)
# `cov-to-parquet` was `nonqiita-to-parquet` until micov dropped Qiita support:
# the name only ever meant "not the Qiita one", and there is no Qiita one now.
# The old name stays registered as a hidden alias -- it is what the README
# documented for two releases and what existing pipelines call -- but it is
# kept out of `--help` so it is not what anyone reaches for next. A copy of the
# command object rather than a second declaration, so the two names cannot
# drift into two implementations.
_nonqiita_alias = copy.copy(cov_to_parquet)
_nonqiita_alias.name = "nonqiita-to-parquet"
_nonqiita_alias.hidden = True
cli.add_command(_nonqiita_alias)


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
    view = View(
        parquet_coverage,
        sample_metadata,
        features_to_keep,
        target_names,
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
    view = View(
        parquet_coverage,
        sample_metadata,
        features_to_keep,
        threads=threads,
        memory=memory,
    )

    # named views so one statement can reach all three. The accessors, not the
    # underlying tables, because they carry the dtype normalization for
    # parquet written by older micov versions.
    view.positions().create_view("binning_positions", replace=True)
    view.metadata().create_view("binning_metadata", replace=True)
    view.feature_metadata().create_view("binning_features", replace=True)

    positions = f"""SELECT pos.{COLUMN_GENOME_ID}, pos.{COLUMN_START},
                           pos.{COLUMN_STOP}, pos.{COLUMN_SAMPLE_ID},
                           md."{metadata_variable}"
                    FROM binning_positions pos
                        JOIN binning_metadata md
                            USING ({COLUMN_SAMPLE_ID})"""
    lengths = f"SELECT {COLUMN_GENOME_ID}, {COLUMN_LENGTH} FROM binning_features"

    # materialized because both outputs read it, and re-executing would repeat
    # the scan of the covered positions
    view.con.sql(
        pos_to_bins(positions, lengths, metadata_variable, bin_num)
    ).create("bin_stats")
    df_bins = view.con.table("bin_stats")

    df_bins_by_sample_hits = df_bins.query(
        "stats",
        f"""SELECT {COLUMN_GENOME_ID}, bin_idx, bin_start, bin_stop,
                   COALESCE(stddev_samp(sample_hits), 0) AS sample_hits_std
            FROM stats
            GROUP BY {COLUMN_GENOME_ID}, bin_idx, bin_start, bin_stop
            ORDER BY sample_hits_std DESC""",
    )

    os.makedirs(outdir, exist_ok=True)
    df_bins.write_csv(f"{outdir}/stats_bins.tsv", sep="\t", header=True)
    df_bins_by_sample_hits.write_csv(
        f"{outdir}/stats_by_variance_of_sample_hits.tsv", sep="\t", header=True
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
    view = View(
        parquet_coverage,
        sample_metadata,
        features_to_keep,
        threads=threads,
        memory=memory,
    )

    view.sample_presence_absence().write_csv(output, sep="\t", header=True)


if __name__ == "__main__":
    cli()
