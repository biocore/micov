# Data formats

**Read this when** you touch a parser or a writer, or before you change
anything that lands in a user's output directory. Every format below is part
of the compatibility contract unless it is marked otherwise.

## Coordinates

These conventions hold everywhere in micov:

- **Intervals are 1-based and half-open, `[start, stop)`.** `start` is SAM
  `POS`, and `stop = POS + reference span`. The span is computed by htslib
  inside miint's `read_alignments`: `M`, `=` and `X` advance both read and
  reference; `D` and `N` advance the reference only, so deletions and skipped
  regions count as covered.
- **Breadth is `sum(stop - start)`** over merged intervals, with no `+1`.
- **Unmapped reads cover nothing, even when placed.** An aligner places an
  unmapped mate at its partner's RNAME and POS (flag 4, CIGAR `*`), and
  `read_alignments` reports it with `stop_position` 0. `compress` drops every
  row with `stop_position <= position` before merging; see
  [traps.md](traps.md).
- **Touching intervals merge.** When `stop1 == start2` the two become one
  interval. This is miint's behaviour in both `compress_intervals` and
  `cumulative_coverage`, and `test_cov.IntervalMergeTests` and
  `test_alignments` pin it.
- **Regions are half-open too.** An interval that starts exactly at a
  region's `stop` is outside it (`View`'s predicate is `pos.start < fc.stop`).
- `.cov` files are called "BED-like" but carry 1-based coordinates, whereas
  real BED is 0-based. Breadth is unaffected; a join against a genuinely
  0-based source is not.
- Primary and secondary alignments are both kept, on purpose: CNVs, HGT
  elements and repeats stay represented.

## Inputs

### Genome lengths (`--lengths`)

`_io.load_genome_lengths` reads two tab-separated columns, taken by position:
`genome_id`, then `length`. A header is optional. `_test_has_header` treats
the first line as a header when any of these holds:

- it starts with `#`
- its first field **equals** `genome_id`
- its second field is not an integer

Validation:

- the length column must be an integer type
- genome ids must be unique
- every length must be greater than 0

For `compress`, this file is also htslib's **reference map**. Headerless SAM
has no `@SQ` lines, so `read_alignments(..., reference_lengths :=
genome_lengths)` is how RNAME resolves. A read whose reference is missing from
the map comes back as reference `*` with FLAG 4, which is indistinguishable
from an unaligned read; `_io._report_unattributed` warns about these and does
not raise.

### SAM/BAM (`compress --data`, or stdin)

Formats:

- **Headerless SAM is the expected input.** Plain SAM, BGZF `.gz`, BAM and
  CRAM go straight to htslib.
- **`.xz` and `.bz2`**, which htslib cannot read, are decompressed to a temp
  file first by `_io._htslib_readable`.
- **A directory** is passed to htslib as one glob, `*.sam*`, so it cannot
  contain `.xz` or `.bz2` files: `cli.compress` rejects them with a usage
  error.

Reading:

- **The input is read exactly once.** That is a requirement, not an
  optimisation: the documented idiom
  `xzcat f.sam.xz | micov compress --sample-id f ...` streams stdin, which
  cannot be rewound.
- **`--sample-id`** defaults to the filename with its compression and format
  extensions stripped (`cli._sample_id_from_path`). It is required for stdin
  and for directories.
- **An input that yields no attributed alignments** raises `ValueError`
  rather than writing empty Parquet. One example is BED3 passed to `compress`
  by mistake.

### BED3 `.cov` / `.cov.gz` (`cov-to-parquet`, `position-plot`)

The columns are `genome_id`, `start`, `stop`, read with `delim='\t'`.

- **`cov-to-parquet`** reads a glob with `header=true` and fixed column
  types. The **sample id is the filename stem**, taken by the regex
  `^(.*/)?(.+).cov(.gz)?$`.
- **`position-plot`** uses `_io.load_bed_cov`. It takes the first three
  columns by position and lets DuckDB's sniffer decide whether there is a
  header, which also covers `#`-prefixed headers. Its stdin is spooled to a
  temp file first (`_io.positions_path`): the sniffer consumes a pipe and then
  returns zero rows without raising.

`micov compress` no longer writes `.cov`. Existing files stay valid input,
and `example/coverages/*.cov.gz` are the committed record of what the
pre-miint implementation produced.

### Sample metadata (`--sample-metadata`)

A tab-separated file with a **required header**. The first column must be
named `sample_id` or `sample_name` (`_io.SAMPLE_ID_COLUMNS`), and is renamed
to `sample_id`. Every column is read as VARCHAR. Rows whose sample has no
coverage at all are dropped (`SEMI JOIN` against `coverage.parquet`).

### Features (`--features-to-keep`, `--target-names`)

Both files need a header. **The first column must be named `genome_id`**
(`_io.FEATURE_ID_COLUMNS`). `_io.read_tsv_with_header` enforces this, so that a
headerless file is rejected instead of losing its first row.

- **Genome mode:** `genome_id` alone, with any extra columns ignored.
- **Region mode:** the file also has columns named `start` and `stop`, which
  must appear together, and every row needs `stop > start`. A genome may
  appear in several rows.
- **`--target-names`:** the columns are `genome_id`, then a name. A
  lineage-style name keeps only the text after its last `"; "`. Spaces and
  square brackets become `_`. Genomes without a name fall back to their id.
  `_io.target_names_query` does this for `per-sample` and `depth-plot` alike;
  `test_io.TargetNamesTests` pins it.

### `depth-plot` inputs

These are `depth-plot`'s readers in `_io`, which `_depth.intersect_layers`
then reconciles.

- **Alignments (`--depth`, and `--breadth`, which defaults to it):** Parquet
  holding `read_alignments`' columns plus a `sample_id` column, which
  `read_alignments` does not produce. `load_alignment_layer` checks for
  `sample_id`, `reference`, `position`, `stop_position` and `cigar`, and says
  how to add `sample_id` if it is missing.
- **Features (`--features-to-keep`):** the header rule as above, then a
  **required `length`** (the alignments carry no lengths), an optional
  `is_circular` (missing or empty is linear), and optional `start`/`stop`.
  A row with a region is a detail panel, and one without only names the
  genome. `load_depth_features` reads every column as text and converts it
  explicitly. It rejects a non-positive length, an `is_circular` that is not
  true or false, half a region, an empty region, a region outside
  `[1, length + 1)`, a genome with two lengths, and a repeated region. A
  genome listed more than once without a region is still one genome.
- **Sample metadata:** as above. The stratifying column is read as text, so
  `Yes` stays `Yes`. A sample with no value in it is left out and reported.
  A sample listed more than once is an error naming it: its reads would
  count twice in its group, or once in each of two.
- **ORFs (`--orfs`):** `read_gff` output saved as Parquet. Only `CDS`,
  `rRNA`, `tRNA`, `tmRNA` and `ncRNA` are kept (`_io.ORF_TYPES`). `gene`
  repeats its CDS, and `region` is the whole sequence. Coordinates stay as
  `read_gff` wrote them, already half-open. Every ORF on a plotted genome
  needs an `ID`; its label is `gene`, else `locus_tag`, else `ID`; a missing
  strand becomes `.`, and so does an unknown one (`?`). Each GFF line is its
  own ORF, so a feature on several lines sharing an `ID` (NCBI's
  frameshifted CDSs) is several. An ORF must span at least one base of its
  genome -- from base 1, and ending after it starts -- and lie within it,
  except that a circular genome's may run past the end: GFF3 writes an ORF
  across the origin with an end beyond the length, and it is split there.
- **Which samples and genomes are used:** a sample needs metadata and an
  aligned read in both layers. A genome needs a features row and an aligned
  read from a used sample in both layers. An aligned read is one with
  `stop_position > position` (`_io.ALIGNED_ROWS`). Everything the metadata
  or features name but this leaves out is reported by name. No sample or no
  genome left, more than 10 groups, a read starting beyond its genome's
  length, ORFs on none of the genomes, or an ORF without an `ID` or outside
  its genome is an error. ORFs on genomes not plotted are ignored, unchecked.

## Outputs

### The Parquet pair (frozen)

`_io.write_coverage_parquet` is the single writer. Both files are written
with `FORMAT PARQUET, PARQUET_VERSION V2, COMPRESSION zstd`
(`_io.PARQUET_OPTIONS`).

| File | Columns, in order (type) |
|---|---|
| `{base}.covered_positions.parquet` | `genome_id` VARCHAR, `start` UINTEGER, `stop` UINTEGER, `sample_id` VARCHAR |
| `{base}.coverage.parquet` | `sample_id` VARCHAR, `genome_id` VARCHAR, `covered` UINTEGER, `length` BIGINT, `percent_covered` DOUBLE |

- `length` is BIGINT because it comes from the lengths file as read.
  Older micov wrote some columns as int64, and `View.coverages()` and
  `View.positions()` cast BIGINT `covered`/`length`/`start`/`stop` back to
  UINTEGER.
- `percent_covered` is `(covered / length) * 100`, **in that order**.
  `covered * 100 / length` is a different double, and the published values
  were computed the first way. `test_alignments` pins a case where the two
  differ.
- Column order is part of the contract: `assert_parquet_equal` checks the
  ordered schema.
- Row order is **not** part of the contract. DuckDB writes in parallel.
- `compress` writes one pair **per sample**. No command merges per-sample
  pairs; `cov-to-parquet` over `.cov` files is the multi-sample path.

### `per-sample` outputs

Per genome, under the `--output` prefix:

```
{output}.{target_name}.{genome}.{variable}.{tag}.png
{output}.{target_name}.{genome}.{variable}.{tag}.ks.csv          (cumulative only)
{output}.{target_name}.{genome}.{variable}.position-plot.png
{output}.{target_name}.{genome}.{variable}.position-plot-scaled.png
{output}.{target_name}.{genome}.{variable}.position-plot-scaled.tsv.gz
```

- **`tag`** is `cumulative` or `non-cumulative`, with `-monte-{focused|unfocused}`
  appended under `--monte`.
- **`target_name`** is the genome id unless `--target-names` maps it.
- **A genome with no metadata group of at least 10 samples gets no curve PNG
  and no `.ks.csv`.** Position plots are written for every genome regardless.

**`.ks.csv`** is comma-separated, despite being misnamed `.ks.tsv` before
M11a. Its header is `_plot.KS_HEADER`:

```
label_A,label_B,ks-statistic,ks-pvalue,ks-pvalue-bonferroni
```

- **The first four columns are frozen** in name, order and value.
- **Labels** are the bare metadata value (`No`, `Yes`), or `add_monte`'s
  `Monte Carlo {type} (n={k})`. The `(n=...)` suffix on group names appears
  only in the plot legend.
- **One row per unordered pair**, in curve order.
- **`ks-pvalue` is raw.** Never correct it in place.
- **`ks-pvalue-bonferroni`** is `min(1, p × m)`, where `m` is the number of
  group-vs-group rows in **this file**. Rows involving the Monte Carlo curve
  are not counted, and their corrected field is empty.
- See [curves-and-ks.md](curves-and-ks.md) for how the values are computed.

**Position `.tsv.gz`** is tab-separated with the header `group	x	y`:

- `group`: the metadata value.
- `x`: the sample's rank within the whole plot.
- `y`: the left edge of each bucket the sample has any coverage in. A
  genome of 1Mb or more has 10,000 buckets, with `np.histogram`'s fractional
  edges; a shorter genome or region has 100bp buckets from its start, the
  last holding whatever remains. See [curves-and-ks.md](curves-and-ks.md).
- Until 0.0.1-dev these files were `position-plot-1_10000th-scale.*`, with
  10,000 buckets on every genome and only the buckets holding an interval's
  ends marked.

The bytes differ every run (the gzip mtime), so compare the decompressed
content.

Both text writers go through `_plot._write_delimited`, which relies on
`csv.writer` rendering floats with `str()`. See [traps.md](traps.md).

### `binning` outputs (`--outdir`)

`stats_bins.tsv` has one row per genome × metadata value × bin that has at
least one hit:

```
genome_id  <variable>  bin_idx  read_hits  sample_hits  samples  bin_start  bin_stop
```

- **`bin_idx`** is 1-based.
- **`samples`** is rendered as `[a,b,c]`, sorted.
- **Row order** is `genome_id, bin_idx, <variable>`.

`stats_by_variance_of_sample_hits.tsv` has one row per genome × bin:

```
genome_id  bin_idx  bin_start  bin_stop  sample_hits_std
```

- **`sample_hits_std`** is the sample standard deviation of `sample_hits`
  across metadata values, or 0 when undefined.
- **Rows are ordered by `sample_hits_std DESC` alone**, so ties come out in
  arbitrary order.

### `extract-sample-presence` output

A wide TSV with one row per sample and one column per region. The column
names are `region_id` = `{genome_id}_{start}_{stop}`. Each cell holds one of
the `_constants` states: `present`, `absent` or `not applicable`. The columns
come from `feature_metadata`, not from the requested regions. A requested
region gets no column if its genome has no row in the region-constrained
coverage.

### `position-plot` output

One PNG per genome, `{output}.{genome}.position-plot.png`. There is no data
file, so `test_plot.PositionPlotSegmentTests` asserts the values instead.

### `depth-plot` per-ORF table

With `--orfs`, one Parquet per run, `{output}.{variable}.depth-plot-orfs.parquet`
(`variable` is the metadata column), collected a genome at a time by
`_io.add_orf_table` and written by `_io.write_orf_table` with the pair's
options (`_io.PARQUET_OPTIONS`). Its columns are
`_io.ORF_STATISTICS_COLUMNS`, frozen from the release that adds the
command. `golden/dp.orfs.parquet` is the fixture's.

| Column | Type | Meaning |
|---|---|---|
| `genome_id` | VARCHAR | |
| `orf_id`, `label`, `type` | VARCHAR | `ID`; `gene`, else `locus_tag`, else `ID`; the GFF type |
| `start`, `stop` | BIGINT | half-open, as `read_gff` gives them; `stop` may pass a circular genome's length |
| `strand` | VARCHAR | `+`, `-` or `.` |
| `group` | VARCHAR | the metadata value |
| `n_samples` | BIGINT | the group's samples |
| `depth_q1`, `depth_median`, `depth_q3` | DOUBLE | quantiles, across the group's samples, of each sample's mean depth over the ORF |
| `depth_mean` | DOUBLE | the group's mean depth over the ORF's bases |
| `prevalence` | DOUBLE | the share of (sample, base) pairs the breadth layer covers |
| `union_breadth` | DOUBLE | the share of the ORF's bases any of the group's samples covers |
| `contrast` | DOUBLE | the same on both of an ORF's rows; NULL unless there are exactly two groups (see [depth-plot.md](depth-plot.md)) |

A row per ORF and group, ORF by ORF in genome order, groups sorted within
each. An ORF is a GFF line, so the key is `genome_id`, `orf_id`, `start`
and `group`: a feature on several lines sharing an `ID` has rows for each. `test_io.WriteOrfTableTests` pins the columns and types literally.
