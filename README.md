## micov: aggregate MIcrobiome COVerage

We introduce aggregate MIcrobiome COVerage (micov), a bioinformatic tool that efficiently computes precise, optionally-aggregated, genomic coverage positions across numerous metagenomes and arbitrary sample types. Micov offers three key advantages over conventional tools: rapid sample type-specific cumulative coverage calculations, identification of mobile or polymorphic genetic elements, and detection of strain heterogeneity through coverage variations.

## Design

The primary input mapping structure for micov is per-sample SAM/BAM or BED
(3-column). These data are then consolidated into Parquet files to utilize
pushdown filters.

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
$ conda create -n micov python=3.12
$ conda install -q --yes -n micov -c conda-forge --file ci/conda_requirements.txt
$ conda activate micov
$ pip install -e .
```

## Example Usages

See below for examples of running `micov` on SAM files.

### 1. Set Up Environment
First, activate the **Conda environment** where `micov` is installed:

```bash
conda activate micov
```

### 2. Process SAM Files to Extract Covered Positions
Next, we will process SAM files to extract covered positions. Note: If you have
`coverages.tgz` coverage files from Qitta, please go to step 4. `micov` accepts
**headerless** SAM/BAM files, and writes out BED-like files which describe the
observed start and stop positions on the references in the SAM data.

If your input files contain headers, remove them using `samtools` before running micov:

```bash
samtools view -S input.sam > output.sam
```

Similarly, if your input files are in BAM format, convert them to SAM format using `samtools`:

```bash
samtools view input.bam > output.sam
```

Next, compress the SAM data into BED coverge files. The `samtools` command above
can be piped into `micov` to compress the SAM data into BED-like files if
desired, but for simplicity, we will demonstrate use from SAM. In writing, we
asssume the name of the SAM file corresponds to a sample name. The subsequent
code expects the BED files to have either a `.cov` or `.cov.gz` extension.

```bash
mkdir -p "./example/coverages"

for file in ./example/samfiles/*.sam.xz; do
    sample_id=$(basename "$file" .sam.xz)

    echo "Processing $file..."

    # Run micov compress
    xzcat $file | micov compress | gzip > "./example/coverages/${sample_id}.cov.gz"
done
```


### 3. Consolidate Coverage Files
After extracting coverage data, consolidate the `.cov` files into Parquet
representations. This requires a **length mapping file (`length.tsv`)**, which
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

micov nonqiita-to-parquet \
    --pattern "example/coverages/*.cov.gz" \
    --output example/parquet/example \
    --lengths example/metadata/length.tsv
```

### 4. Convert Coverage Data to Parquet Format
`micov` provides functionality to convert **Qiita-formatted coverage data** into **Parquet format** as well.

```bash
mkdir -p "./example/parquet"

# note: multiple coverage files can be specified by repeating the --qiita-coverages argument
micov qiita-to-parquet \
 --qiita-coverages  "./example/consolidate/consolidated.tgz" \
 --output "./example/parquet/example" \
 --lengths "./example/metadata/length.tsv"
```

### 5. Generate Per-Sample-Group Plots
A series of plots can be constructed guided by metadata. Specifically, `micov` produces the following:

* **Non-cumulative coverage curves** for each genome in the feature metadata.
* **Cumulative coverage curves** for each genome in the feature metadata. These accumulation data are supported by K-S tests written to the output directory.
* **Scaled and unscaled position plots** for each genome in the feature metadata.

Categorical metadata can be used to group samples; `sample-metadata` is
required. The genomes to examine can optionally be constrained using
`features-to-keep`. Specific start and stop regions of genomes can also be
specified within the `features-to-keep` but limited to a single region per
genome currently.

`micov` expects the first column of a sample metadata file to be the sample ID
under the header `sample_id`. Similarly, the first column of a feature metadata
file should be the feature ID under the header `genome_id`.

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

Pairwise Kolmogorov-Smirnov (KS) tests between all sample groups' cumulative coverage curves are automatically conducted and results saved in `cumulative.ks.tsv`. The KS test quantifies whether two sample groups differ in the distribution of their cumulative genome coverages, with the KS statistic measuring the maximal difference between the two cumulative distributions, and the KS p-value assessing the statistical significance of the difference.


```bash
mkdir -p "./example/plots/per_sample_groups"

micov per-sample-group \
 --parquet-coverage "./example/parquet/example" \
 --sample-metadata "./example/metadata/sample_metadata.txt" \
 --sample-metadata-column "dog" \
 --features-to-keep "./example/metadata/feature_metadata.txt" \
 --output "./example/plots/per_sample_groups/example" \
 --plot
```

### 6. Binning and Ranking

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

### 7. Additional Usage (optional)

Existing .SAM/.BAM can be converted into coverage percentages by specifying length data at compression:

```bash
$ xzcat some_data.sam.xz | micov compress --length length.tsv > coverages.tsv
```

Multiple coverage files for the same sample can be aggregated into a single file:

```bash
$ zcat run1/sample1.cov.gz run2/sample1.cov.gz | micov compress | gzip > combined/sample1.cov.gz
```
