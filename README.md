## micov: aggregate MIcrobiome COVerage

We introduce aggregate MIcrobiome COVerage (micov), a bioinformatic tool that efficiently computes precise, optionally-aggregated, genomic coverage positions across numerous metagenomes and arbitrary sample types. Micov offers three key advantages over conventional tools: rapid sample type-specific cumulative coverage calculations, identification of mobile or polymorphic genetic elements, and detection of strain heterogeneity through coverage variations.

## Design

The primary input mapping structure for micov is per-sample SAM/BAM or BED
(3-column). These data are then consolidated into Parquet files to utilize
pushdown filters.

## Requirements

micov's compute runs on DuckDB with the
[miint](https://github.com/the-miint/duckdb-miint) extension. miint is a DuckDB
**extension**, not a Python package, so it is not installed by `pip` or
`conda`. micov fetches it from <https://ftp.microbio.me/pub/miint> the first
time it opens a connection, which **requires outbound network access on first
use**; afterwards it is cached in `~/.duckdb/extensions/` and no network is
needed.

miint is currently published **unsigned**, so micov enables DuckDB's
`allow_unsigned_extensions` on every connection. This is a development
posture and will be tightened once signed builds are available.

DuckDB is pinned to **1.5.4**. Extensions are built per DuckDB version, and
the repository above carries a `v1.5.4` tree only; a newer DuckDB has no miint
build to load.

Supported platforms are **Linux** (x86_64, aarch64) and **macOS on Apple
silicon**. No miint build is published for Windows or for Intel macOS, so
micov does not run there.

To install without network access, or to use a local miint build, put the
extension somewhere on disk and point micov at it:

```bash
export MICOV_MIINT_EXTENSION_PATH=/path/to/miint.duckdb_extension
```

Alternatively, pre-seed the cache on a machine that does have network access
and copy `~/.duckdb/extensions/` across.

## Installation

We recommend creating a separate conda environment, and installing
into that.

```bash
$ pip install micov
```

## Installation From Source

To install the most up-to-date version of micov

```bash
$ git clone https://github.com/biocore/micov.git
$ cd micov
$ conda create -n micov -c conda-forge python=3.12
$ conda install -q --yes -n micov -c conda-forge --file ci/conda_requirements.txt
$ conda activate micov
$ pip install -e ".[test]"
```

The `[test]` extra installs `pytest`, which `make test` needs. Omit it for a
plain runtime install.

## Example Usages

See below for examples of running `micov` on SAM files.

### 1. Set Up Environment
First, activate the **Conda environment** where `micov` is installed:

```bash
conda activate micov
```

### 2. Process SAM Files into Coverage Parquet

Next, process SAM files into coverage data. `micov` accepts **headerless**
SAM/BAM.

If your input files contain headers, remove them using `samtools` before running micov:

```bash
samtools view -S input.sam > output.sam
```

Similarly, if your input files are in BAM format, convert them to SAM format using `samtools`:

```bash
samtools view input.bam > output.sam
```

`micov compress` writes two Parquet files per sample,
`{output}.coverage.parquet` and `{output}.covered_positions.parquet`.

**`--lengths` is required.** It supplies the coverage denominators, and it also
serves as the reference map: headerless SAM carries no header for htslib to
resolve reference names against. An example length file is at
`./example/metadata/length.tsv`; see step 3 for how to build one.

```bash
mkdir -p "./example/parquet"

for file in ./example/samfiles/*.sam.xz; do
    sample_id=$(basename "$file" .sam.xz)

    echo "Processing $file..."

    micov compress \
        --data "$file" \
        --lengths ./example/metadata/length.tsv \
        --output "./example/parquet/${sample_id}"
done
```

`micov` also reads from a pipe, which is the better option for large inputs
since nothing is staged on disk. Give it `--sample-id`, as there is no filename
to take one from:

```bash
xzcat foo.sam.xz | micov compress \
    --lengths length.tsv --sample-id foo --output foo
```

Reads whose reference is absent from `--lengths` cannot be attributed to a
genome and are dropped; `micov` reports how many. Note it cannot distinguish
those from reads that simply did not align, because htslib reports both the
same way — so if that count is surprising, check that `--lengths` covers every
reference the data was aligned against.

### 3. Consolidate Coverage Files
`micov cov-to-parquet` builds the same Parquet representation from BED3
`.cov`/`.cov.gz` files. `micov compress` no longer writes `.cov`, so this is
for **existing** coverage files — including aggregating one sample across
several runs, which `compress` does not do:

```bash
micov cov-to-parquet \
    --pattern "run*/sample1.cov.gz" \
    --output combined/sample1 \
    --lengths length.tsv
```

It requires a **length mapping file (`length.tsv`)**, which
maps genome IDs to their corresponding genome lengths. An example length file
can be found in `./example/metadata/length.tsv`. If this file is not available,
it can for example be generated using `seqkit`:

```bash
seqkit fx2tab --length --name --header-line foo.fasta > length.tsv
```

Now, consolidate the coverage files. On read, `micov` will interpret the non-extension
portion of a filename as the sample ID. For example, given `foo/bar/baz.cov.gz`, the
sample ID will be `baz`.

```bash

micov cov-to-parquet \
    --pattern "example/coverages/*.cov.gz" \
    --output example/parquet/example \
    --lengths example/metadata/length.tsv
```

This command was named `nonqiita-to-parquet` before micov dropped its Qiita
support. The old name still works but is hidden from `--help`; prefer
`cov-to-parquet`.

### 4. Generate Per-Sample-Group Plots
A series of plots can be constructed guided by metadata. Specifically, `micov` produces the following:

* **Non-cumulative coverage curves** for each genome in the feature metadata.
* **Cumulative coverage curves** for each genome in the feature metadata. These accumulation data are supported by K-S tests written to the output directory.
* **Scaled and unscaled position plots** for each genome in the feature metadata.

Categorical metadata can be used to group samples; `sample-metadata` is
required. The genomes to examine can optionally be constrained using
`features-to-keep`. Specific start and stop regions of genomes can also be
specified within the `features-to-keep` but limited to a single region per
genome currently.

Both files **must have a header line**. The first column of a sample metadata
file is the sample ID, under the header `sample_id` or `sample_name`; the first
column of a feature metadata file (and of a `--target-names` file) is the
genome ID, under the header `genome_id`. `micov` stops with an error naming the
file otherwise. A file without a header -- a taxonomy `lineages.txt`, say --
would have its first row taken as column names, and that genome or sample
silently dropped.

The `--output` parameter specified a prefix for the output files.

Optionally, Monte Carlo curves can be produced for the cumulative plots by
specifying `--monte`. There are two Monte Carlo options: `unfocused` and
`focused`. The `unfocused` option will select samples at random with _any_
coverage data, while the `focused` option will randomly select samples with
nonzero coverage of the current genome. Both options select independent of
sample metadata, and will select the max number of samples observed in a sample
group.

Additionally, users can specify `--percentile` to display plots with the x-axis
representing percentile of samples instead of absolute sample counts. 

Pairwise Kolmogorov-Smirnov (KS) tests between all sample groups' cumulative coverage curves are automatically conducted and results saved in `cumulative.ks.csv`. The KS test quantifies whether two sample groups differ in the distribution of their cumulative genome coverages, with the KS statistic measuring the maximal difference between the two cumulative distributions, and the KS p-value assessing the statistical significance of the difference.

The file is comma-separated, one row per pair of curves, with columns `label_A`, `label_B`, `ks-statistic`, `ks-pvalue` and `ks-pvalue-bonferroni`. `ks-pvalue` is **uncorrected**. `ks-pvalue-bonferroni` is `min(1, p × m)`, where the family `m` is the number of group-vs-group comparisons in that file -- that is, for that genome. Comparisons against a `--monte` curve are a null-model check rather than a hypothesis: they are not counted in `m` and their corrected value is left empty, so adding `--monte` never changes a group pair's corrected p-value. Correcting across genomes, or by another method, is left to the analyst.


```bash
mkdir -p "./example/plots/per_sample_groups"

micov per-sample \
 --parquet-coverage "./example/parquet/example" \
 --sample-metadata "./example/metadata/sample_metadata.txt" \
 --sample-metadata-column "dog" \
 --features-to-keep "./example/metadata/feature_metadata.txt" \
 --output "./example/plots/per_sample_groups/example" \
 --plot
```

### 5. Binning and Ranking

The `binning` command allows you to divide genome positions into fix-sized bins and compute summary statistics across samples, based on sample metadata. This is useful for identifying regions of interest (e.g. high variability across samples).

```bash
mkdir -p "./example/binning"

micov binning \
    --parquet-coverage ./example/parquet/example \
    --sample-metadata ./example/metadata/sample_metadata.txt \
    --features-to-keep ./example/metadata/feature_metadata.txt \
    --metadata-variable "dog" \
    --outdir ./example/binning \
    --rank
```

Each bin is ranked based on the standard deviation of sample hits across groups assoicated with the chosen metadata category, with bins exhibiting higher variability ranked at the top. 

The rankings are saved in the output `stats_by_variance_of_sample_hits.tsv` whereas binning statistics (start and end positions of each bin, number of sample hits per bin, number of read hits per bin.etc) are saved in `stats_bins.tsv`.

### 6. Additional Usage (optional)

Per-genome coverage percentages are a column of `{output}.coverage.parquet`:

```bash
$ duckdb -c "SELECT genome_id, percent_covered FROM 'foo.coverage.parquet'"
```

Multiple coverage files for the same sample are aggregated with
`cov-to-parquet`, which takes a glob (see step 3). `micov compress` takes
SAM/BAM only.
