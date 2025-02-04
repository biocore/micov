## micov: aggregate MIcrobiome COVerage

Micov is a program to compute coverage over many genomes and many samples. 
These coverages can be used to filter genomes or contigs which are
not well represented in a dataset, or to examine differential coverage in your
dataset.

## Design

The primary input mapping structure for micov is SAM/BAM or BED (3-column). 
Coverage data can be aggregated into Qiita-like `coverage.tgz` files. Per-sample
coverages can be then be harvested from multiple `coverage.tgz` files.

Why `coverage.tgz` files? Qiita provides a rich set of already computed 
coverage data in a BED3 compatible format. Rather than invent 
*yet-another-format*, we opted to establish functionality on what is readily
available from that resource.

## Installation

We currently recommend creating a separate conda environment, and installing
into that.

```bash
$ conda create -n micov -c conda-forge polars matplotlib scipy click tqdm numba duckdb pyarrow
$ pip install micov
```

## Examples

Compressing covered regions, and computing the per-genome coverage, from existing
Qiita files:

```bash
$ micov qiita-coverage \
    --lengths genome-lengths-in-reference.map \
    --output coverage-example \
    --qiita-coverages /qmounts/qiita_data/BIOM/191463/coverages.tgz \
    --qiita-coverages /qmounts/qiita_data/BIOM/191556/coverages.tgz \
    --qiita-coverages /qmounts/qiita_data/BIOM/191575/coverages.tgz \
    --qiita-coverages /qmounts/qiita_data/BIOM/191879/coverages.tgz \
    --qiita-coverages /qmounts/qiita_data/BIOM/191926/coverages.tgz \
    --qiita-coverages /qmounts/qiita_data/BIOM/191613/coverages.tgz \
    --qiita-coverages /qmounts/qiita_data/BIOM/192511/coverages.tgz \
    --samples-to-keep metadata-with-samples-of-interest.tsv
```

Existing SAM/BAM data can be compressed into a BED-like format. Genome lengths and taxonomy are optional, but useful for downstream analysis:

```bash
$ micov compress \
    --data input.sam \
    --output compressed_output.tsv \
    --lengths genome-lengths.tsv \
    --taxonomy taxonomy.tsv
```

Compressed SAM/BAM data can also be piped in:

```bash
$ xzcat some_data.sam.xz | micov compress > compressed_output.tsv
```

Generate a coverage visualization for a single sample:

```bash
$ micov position-plot \
    --positions covered-positions.tsv \
    --output sample_coverage_plot.png \
    --lengths genome-lengths.tsv
```

Consolidate multiple coverage files into a Qiita-like archive:

```bash
$ micov consolidate \
    --paths /path/to/coverage/files \
    --output consolidated_coverages \
    --lengths genome-lengths.tsv
```

Convert Qiita coverage data to Parquet for efficient querying:

```bash
$ micov qiita-to-parquet \
    --qiita-coverages /path/to/coverage1.tgz \
    --qiita-coverages /path/to/coverage2.tgz \
    --output coverage_data_base \
    --lengths genome-lengths.tsv \
    --samples-to-keep sample_metadata.tsv
```

Generate per-sample group analysis plots from precomputed parquet coverage. Include `--plot` to generate visualizations and `--monte focused` to generate a null coverage curve:

```bash
$ micov per-sample-group \
    --parquet-coverage coverage_data_base \
    --sample-metadata sample_metadata.tsv \
    --sample-metadata-column experimental_group \
    --output per_sample_plots \
    --features-to-keep features_list.tsv \
    --plot \
    --monte focused \
    --monte-iters 100 \
    --target-names target_names.tsv
```

Monte Carlo simulation can also be run as a separate command to generate a null coverage curve:

```bash
$ micov per-sample-monte \
    --parquet-coverage coverage_data_base \
    --sample-metadata sample_metadata.tsv \
    --sample-metadata-column group_column \
    --output monte_results \
    --plot \
    --iters 500 \
    --target-names target_names.tsv
```

Analyze coverage distribution by binning the genome positions for a genome of interest:

```bash
$ micov binning \
    --covered-positions all_samples_covered_positions.tsv \
    --outdir binning_results \
    --genome-id G000005825 \
    --genome-length 4249288 \
    --bin-num 1000
```
