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
into that. It is likely the dependency pins can be relaxed but we haven't 
verified that just yet.

```bash
$ conda env create -n micov -c conda-forge polars matplotlib scipy click tqdm numba
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

The above command can be constrained to particular features as well.

If instead, the desire is to produce non-cumulative, cumulative and coverage
maps, the command is slightly restructured. This command as well can be limited
to specific features.

```bash
$ micov per-sample-group \
    --qiita-coverages /qmounts/qiita_data/BIOM/191463/coverages.tgz \
    --qiita-coverages /qmounts/qiita_data/BIOM/191556/coverages.tgz \
    --qiita-coverages /qmounts/qiita_data/BIOM/191575/coverages.tgz \
    --qiita-coverages /qmounts/qiita_data/BIOM/191879/coverages.tgz \
    --lengths genome-lengths-in-reference.map \
    --sample-metadata metadata-with-samples-of-interest.tsv \
    --sample-metadata-column cool_categorical_variable \
    --output plots-example 
```

Exising .SAM/.BAM can be compressed into a BED-like format by file or pipe. A
pipe example is shown below:

```bash
$ xzcat some_data.sam.xz | micov compress | gzip > compressed.tsv.gz
```

Compressed BED-like representations can be aggregated into Qiita-like coverage
files as well:

```bash
$ micov consolidate \
    --lengths genome-lengths.tsv \
    --paths a-file-with-a-list-of-paths \
    --output consolidated.tgz
```
