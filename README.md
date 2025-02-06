## micov: aggregate MIcrobiome COVerage

We introduce aggregate MIcrobiome COVerage (micov), a bioinformatic tool that efficiently computes precise, optionally-aggregated, genomic coverage positions across numerous metagenomes and arbitrary sample types. Micov offers three key advantages over conventional tools: rapid sample type-specific cumulative coverage calculations, identification of mobile or polymorphic genetic elements, and detection of strain heterogeneity through coverage variations.

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
into that

```bash
$ conda env create -f micov.yml
$ conda activate micov
$ pip install micov
```

## Installation From Source

To install the most up-to-date version of micov
```bash
$ conda create -n micov -c conda-forge polars matplotlib scipy click tqdm numba duckdb pyarrow
$ conda activate micov
$ git clone https://github.com/biocore/micov.git
$ cd micov
$ pip install -e .
```

## Example Usages

See below for examples of running micov on SAM files. 

### 1. Set Up Environment
First, activate the **Conda environment** where **micov** is installed:  
```bash
conda activate micov
```

### 2. Process SAM Files to Extract Covered Positions
If you already have tgz format coverage files from Qitta, go to step 4. 

micov currently only processes **headerless** SAM/BAM files. If your input files contain headers, remove them using `samtools` before running micov:  
```bash
samtools view -S input.sam > output.sam
```

Now, create an output directory and run micov on each SAM file:  
```bash
mkdir -p "./example/coverages"

for file in ./example/samfiles/*.sam.xz; do
    filename=$(basename "$file" .sam.xz)

    echo "Processing $file..."
    
    # Run micov compress
    xzcat $file | micov compress | gzip > "./example/coverages/${filename}.cov.gz"
done
```

### 3. Consolidate Coverage Files
After extracting coverage data, consolidate the `.cov` files into a compressed `.tgz` archive. This requires a **length mapping file (`length.tsv`)**, which maps genome IDs to their corresponding genome lengths. An example length file can be found in `./example/metadata/length.tsv`. If this file is not available, it can be generated using `seqkit`:
```bash
seqkit fx2tab --length --name --header-line foo.fasta > length.tsv
```

Now, consolidate the coverage files:
```bash
mkdir -p "./example/consolidate"

find "./example/coverages" -type f -name '*.cov.gz' > "./example/consolidate/paths.txt"

micov consolidate \
    --lengths "./example/metadata/length.tsv" \
    --paths "./example/consolidate/paths.txt" \
    --output "./example/consolidate/consolidated.tgz"
```

### 4. Convert Coverage Data to Parquet Format
micov provides functionality to convert **TGZ-formatted coverage data** into **Parquet format** for efficient querying and processing:
```bash
mkdir -p "./example/parquet"

micov qiita-to-parquet \
 --qiita-coverages  "./example/consolidate/consolidated.tgz" \
 --output "./example/parquet/example" \
 --lengths "./example/metadata/length.tsv"
```

### 5. Generate Per-Sample-Group Plots
micov can generate **per-sample-group plots** based on **sample metadata** and a **feature metadata file**. Ensure that `feature_metadata.txt` has a header like `feature_id` as the first line and no blank lines at the end. This produces non-cumulative, cumulative, scaled, and unscaled position plots for each genome in feature metadata.

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

### 6. Generate Monte Carlo-Based Plots
For Monte Carlo-based sample group plots, run:
```bash
mkdir -p "./example/plots/per_sample_groups_monte"

micov per-sample-group \
 --parquet-coverage "./example/parquet/example" \
 --sample-metadata "./example/metadata/sample_metadata.txt" \
 --sample-metadata-column "dog" \
 --features-to-keep "./example/metadata/feature_metadata.txt" \
 --output "./example/plots/per_sample_groups_monte/example" \
 --plot \
 --monte "unfocused" \
 --monte-iters 100
```


### 7. Additional usage (optional)

Exising .SAM/.BAM can be compressed into a BED-like format by file or pipe. A
pipe example is shown below:

```bash
$ xzcat some_data.sam.xz | micov compress > compressed.tsv
```

Aggregate genome coverages can be calculated using 
```bash
$ xzcat some_data.sam.xz | micov compress --length length.tsv > coverages.tsv
```
or 
```bash
$ micov compress \
    --data input.sam \
    --output compressed_output.tsv \
    --lengths genome-lengths.tsv \
    --taxonomy taxonomy.tsv
```
