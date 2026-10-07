micov ChangeLog
=====================

micov 0.0.1-dev
---------------

New features:

* **`micov depth-plot` draws per-base depth and breadth along each genome,
  by sample group.** Depth and breadth may come from different alignment
  files -- metatranscriptomic depth over metagenomic breadth, say -- each
  `read_alignments` output saved as Parquet with a `sample_id` column. For
  each genome in `--features-to-keep`, which gives its `length` and
  optionally `is_circular` and detail regions, it writes an overview
  (`{output}.{target_name}.{genome}.{variable}.depth-plot.png`), a ring for
  a circular genome (`...depth-plot-circular.png`, three groups at most),
  and a panel per region (`...depth-plot-detail-{start}-{stop}.png`). Depth
  is each group's per-base median, IQR and mean, on a symlog axis; breadth
  is where any sample covers and the share of samples that do. With
  `--orfs`, a `read_gff` Parquet, it draws the ORFs, which `--highlight`,
  `--orf-color-by` and `--orf-contrast` can mark or colour, and writes a
  per-ORF table, `{output}.{variable}.depth-plot-orfs.parquet`. This is an
  addition to the frozen command set, approved by the maintainer; its
  options and the per-ORF table's columns are frozen from this release.

Backward incompatible changes:

* **The scaled position plot is now `position-plot-scaled.{png,tsv.gz}`,** not
  `position-plot-1_10000th-scale.*`, for every genome. Anything globbing the
  old name needs updating. Genomes and regions of 1Mb or more keep their
  10,000 buckets and their bucket edges. Shorter ones now have 100bp buckets
  from their start, the last holding whatever remains: 10,000 buckets on a
  145kb chloroplast were 15bp, narrower than the intervals being plotted, and
  a genome or region under 10kb had buckets of less than a base. A
  zero-length genome now stops with an error naming it, where it previously
  plotted nothing.
* **The scaled position plot marks every bucket an interval overlaps.** It
  marked only the buckets holding an interval's start and stop, so a bucket in
  the middle of a long interval was left blank. On a 145kb genome with 15bp
  buckets that was half the covered buckets. It also marked the bucket
  beginning exactly at an interval's end, which the interval does not cover;
  that bucket is no longer marked. This removes rows only where a bucket edge
  falls on a whole position, which depends on the genome length, so the
  `example/` genomes gained 836 and 130 rows (0.48% and 0.10%) and lost none.
  On a 100-sample study of 9,388 genomes the `.tsv.gz` rows grew 1.35x
  overall: the median genome is unchanged, the 99th percentile doubles, and
  no genome can exceed samples x 10,000 rows.
* **Samples with equal breadth are ranked by `sample_id`.** They were ranked in
  whatever order the coverage rows arrived, which changes between runs because
  the Parquet is scanned in parallel. Running `per-sample` twice on the same
  input could therefore write different cumulative curves, position files and
  KS statistics for any genome where two samples in one group had identical
  breadth -- about one genome in ten on a real 100-sample study. Outputs are
  now reproducible, and genomes without such ties are unaffected.
* **Sample metadata and feature files must have a header.** The first column of
  `--sample-metadata` must be named `sample_id` or `sample_name`, and the first
  column of `--features-to-keep` and `--target-names` must be named
  `genome_id`; micov stops with an error naming the file otherwise. These files
  were read as having a header whether or not they did, so a headerless one --
  a taxonomy `lineages.txt`, for example -- silently lost its first genome or
  sample, which then appeared in no output. Files whose first column has some
  other name need renaming.
* **KS results are now `.ks.csv`, not `.ks.tsv`.** The files were always
  comma-separated; only the name was wrong. Anything globbing `*.ks.tsv` needs
  updating. The content of the first four columns is unchanged.
* **KS results carry a Bonferroni-corrected p-value** in a new last column,
  `ks-pvalue-bonferroni`. The paper describes a Bonferroni correction that
  micov previously left to the analyst. It is `min(1, p × m)`, where
  `m` is the number of group-vs-group comparisons in the file (one file per
  genome). Comparisons against a `--monte` curve are not counted and are left
  blank, so `--monte` never changes a group pair's corrected value.
  `ks-pvalue` stays uncorrected. Readers that take columns by position are
  unaffected; readers that check the header will see five names.
* **`micov position-plot` names its output files correctly.** They were written
  as `{output}.('G000000001',).position-plot.png` -- a literal Python tuple
  repr, because the genome key came from a polars `group_by` and went straight
  into the filename. They are now `{output}.{genome_id}.position-plot.png`.
  Anything globbing the old names will not match.
* **`micov position-plot` can read stdin.** `--positions` was already optional,
  but omitting it raised `io.UnsupportedOperation: underlying stream is not
  seekable`, so the documented-by-the-interface path never worked.
* **A region's end position is now exclusive when deciding overlap.** Regions
  are half-open `[start, stop)`, but an interval beginning at exactly `stop`
  was counted as overlapping. It contributed zero bases, so it never changed a
  breadth value -- but it did make `micov extract-sample-presence` report the
  sample *present* in a region it covers nothing of, and it put a zero-width
  interval into the position output. Presence calls at an exact region
  boundary flip from `present` to `absent`. No region in `example/` is
  affected; the nearest interval start is 4 bases from a region edge.
* **Regions covering more than one part of the same genome are scored
  separately.** Coverage was summed across *all* regions on a genome and then
  divided by *each* region's length, so a sample could be reported as covering
  21 bases of a 20bp region -- `percent_covered` above 100. Each region now has
  its own numerator. Feature files with one region per genome, which is what
  `example/` and the published analyses use, are unaffected.
* **`micov extract-sample-presence` honours the sample metadata.** Presence was
  computed from the unfiltered coverage, so a metadata file naming a subset of
  samples still produced a row for every sample in the Parquet.

Other changes:

* **Unmapped mates add no breadth.** Aligners place an unmapped mate at its
  partner's position. After the move to miint, `micov compress` read such a
  mate as an interval running back to the start of the genome, so a single
  one made the sample cover everything up to its partner, and
  `--disable-compression` failed on it outright. Unmapped reads now cover
  nothing, and a file whose reads are all unmapped is reported as having no
  alignments. Released micov was not affected, and neither is `example/`,
  which has no unmapped reads.
* **Plots use a colour-blind-safe palette.** Groups were coloured with
  matplotlib's default cycle, whose orange and green -- the second and third
  groups -- are the same colour to a reader with protanopia. Groups are now
  drawn in the first five Okabe–Ito colours (blue, orange, sky blue,
  vermillion, bluish green), which stay distinct in every pair under
  protanopia and deuteranopia. The first two groups look much as before. In
  the coverage curves, groups six to ten reuse those five colours as dashed
  lines; position plots already name each group under its block. Only PNGs
  change: no data file, coverage value or KS statistic moves.
* **`micov per-sample` names the metadata groups its coverage curves leave
  out.** A curve plots at most ten groups, the first ten values in sorted
  order, and an eleventh was dropped from both the plot and the `.ks.csv`
  without a word. It is still not plotted, but any such group with enough
  samples to plot (10) is now named in a warning on stderr.
* **`micov per-sample` no longer slows down with the number of genomes.** Every
  plotting step filtered the whole positions table for its one genome, four
  times per genome. On a 100-sample study with 9,388 genomes and 29.7M
  intervals that filtering alone was about 2 hours, and `per-sample` took 6.1 h
  against 2.5 h for the previous release. Positions now stay in DuckDB and are
  fetched one genome at a time (~8 s of filtering for the same study). A
  plotting figure was also left open for every genome with no group of at
  least 10 samples, so memory grew with the number of genomes; it is now
  closed.
* **Paths and sample IDs containing `'` work.** Every command failed with a
  DuckDB `Parser Error` when a path -- or `compress --sample-id` -- contained a
  single quote, which is common in home directories (`/Users/o'brien`). They
  are now escaped everywhere micov builds SQL.
* **A lengths file no longer loses its first genome to header detection.** A
  headerless file whose first genome ID was a substring of `genome_id` -- such
  as `id` or `genome` -- was read as having a header, and that genome silently
  had no length. Headers are unaffected.
* `micov binning --rank` is documented as having no effect. The variance
  ranking has always been written unconditionally; the flag is still accepted
  so existing invocations keep working.
* **`scipy` is no longer a dependency.** The pairwise KS tests written to
  `.ks.tsv` now use the `ks_2samp` function of the miint DuckDB extension.
  Runtime dependencies are now `click`, `matplotlib` and `duckdb`.

  The **KS statistics are unchanged**, byte for byte, in every frozen output
  under `example/plots/`. The **p-values agree to within a few units in the
  last place** -- at most 3.8e-16 relative on `example/`, e.g.
  `0.24244968766417713` is now written `0.24244968766417715` -- because miint
  computes the exact p-value independently rather than transcribing scipy's.
  No p-value moves by an amount that matters to any test of significance, but
  the printed digits differ in 20 of the 24 deterministic rows.

  miint raises for a metadata group of more than 10000 samples, where scipy
  switched to an approximate p-value. micov has not been run on groups that
  large.
* **Refresh a cached miint extension when upgrading.** DuckDB caches extensions
  in `~/.duckdb/extensions/` and never replaces a cached build on its own, so
  anyone who ran micov before 2026-09-11 has a miint whose `ks_2samp` returns
  the KS statistic unrounded -- `0.30000000000000004` where the published value
  is `0.3`. micov cannot detect this. Run
  `FORCE INSTALL miint FROM 'https://ftp.microbio.me/pub/miint'` in DuckDB, or
  delete the cached `miint.duckdb_extension`, before producing `.ks.tsv`
  output.
* **`micov per-sample` is substantially faster on large sample groups.** The
  cumulative curve accumulated by re-merging a growing interval set once per
  sample, which is quadratic in group size; it is now a single aggregate.
  Measured on the 49-sample `example/` dataset the change is modest -- 5.1s to
  4.5s, and 9.1s to 6.5s with `--monte` -- because that is where a quadratic
  cost is still small. On synthetic groups it is 4.7x at 100 samples, 16.7x at
  500 and 28.1x at 1000. Curve values, KS statistics and p-values are
  unchanged, byte for byte.
* **A malformed region is rejected by name.** A feature file whose `start` and
  `stop` are transposed failed with `Out of Range Error: Overflow in
  subtraction of UINT32 (40 - 60)`, and a zero-width region with `No positions
  left after filtering.` -- neither of which names the offending row. Both now
  raise naming the genome and the interval.

* **`polars` is no longer a dependency.** Every use moved to DuckDB (parsing,
  joins, aggregation) or to the standard library (writing `.ks.tsv` and the
  position-plot `.tsv.gz`, which was formatting rather than computation). All
  sixteen frozen plot outputs under `example/plots/` are byte-identical, and
  the full golden suite passes with polars uninstalled. Runtime dependencies
  are now `click`, `scipy`, `matplotlib` and `duckdb`.
* `micov position-plot` now accepts a `.cov` file with a `#`-prefixed header,
  which it previously read as data.
* **`micov per-sample --sort-by-metadata-value`** lays position plot groups out
  by metadata value: finite numbers numerically, then other text (`nan` and
  `inf` included), then blanks. Groups are otherwise laid out smallest first, with ties in text
  order, so equal-sized depth groups read 270, 30, 5. Without the flag nothing
  changes, and the coverage curves are unaffected either way.

Backward incompatible changes (earlier in this release):

* **Qiita support has been removed.** `micov qiita-coverage`,
  `micov qiita-to-parquet` and `micov consolidate` are gone, along with the
  reader and writer for the Qiita `coverages.tgz` layout. micov no longer
  reads or writes tar archives, and `example/consolidate/` was removed with
  them. This may be revisited; nothing about the Parquet or `.cov` formats
  changed, so a future reader would be additive.
* **`micov nonqiita-to-parquet` is now `micov cov-to-parquet`.** The old name
  only ever meant "not the Qiita one". It stays registered as a **hidden
  alias** -- existing scripts keep working -- but it no longer appears in
  `micov --help`. Both names reach the same command object, so they cannot
  diverge.

  `example/parquet/` was regenerated with `cov-to-parquet`, since
  `qiita-to-parquet` produced it and is gone. The rows are unchanged;
  `sample_id` moved from the last column of `example.coverage.parquet` to the
  first. Every downstream artifact -- the KS statistics, the position data,
  the binning tables -- reproduces from the regenerated corpus.
* **`micov compress` writes Parquet, not `.cov`.** It now produces
  `{output}.coverage.parquet` and `{output}.covered_positions.parquet`
  directly from SAM/BAM, collapsing the intermediate BED3 hop. Consequences:
  - `--lengths` and `--output` are now **required**. `--lengths` supplies the
    coverage denominators *and* htslib's reference map, since headerless SAM
    carries no header to resolve reference names against.
  - `--sample-id` is new. It defaults to the `--data` filename with its
    extensions stripped, and is required when reading stdin or a directory,
    because `coverage.parquet` is keyed by sample.
  - The two TSV summary output modes are gone, and `--taxonomy` with them.
    `{output}.coverage.parquet` already carries `genome_id`, `covered`,
    `length` and `percent_covered`.
  - **BED3 input is no longer accepted.** Aggregating `.cov`/`.cov.gz` --
    including one sample across several runs -- is `micov cov-to-parquet`,
    which takes a glob.

  `.cov` remains fully *readable* through `cov-to-parquet`, and existing
  `.cov` artifacts stay valid input. Only the writer went away. Verified
  against all 49 committed `example/samfiles/`: every sample's intervals come
  back identical to the `example/coverages/*.cov.gz` produced by the previous
  implementation.
* micov's alignment ingest now runs through miint's `read_alignments` and
  `compress_intervals` rather than its own CIGAR walker and numba interval
  merge. **`numba` is no longer a dependency.**
* DuckDB is now pinned to `>=1.5.4,<=1.5.5`. miint is published per DuckDB
  version and the repository carries trees up to `v1.5.5`, so a newer DuckDB
  has no extension build to load. The ceiling rises as builds are published.
* The miint extension is now installed from `https://ftp.microbio.me/pub/miint`
  rather than the DuckDB community repository, and micov enables
  `allow_unsigned_extensions` because those builds are currently unsigned. A
  cache holding a community-origin build is upgraded automatically.

* micov now requires the [miint](https://github.com/the-miint/duckdb-miint)
  DuckDB extension. miint is a DuckDB *community extension* rather than a
  Python package, so it cannot be declared as a dependency; micov installs and
  loads it itself, and raises if it cannot. Two consequences for deployment:
  the first run needs **outbound network access** to fetch the extension
  (afterwards it is cached in `~/.duckdb/extensions/`), and
  `MICOV_MIINT_EXTENSION_PATH` can point at a build on disk for air-gapped
  installs or local miint development.
* **Windows and Intel macOS are no longer supported.** miint is published for
  Linux (x86_64, aarch64) and macOS on Apple silicon only; it wraps htslib and
  minimap2, so Windows was never realistically reachable. Carrying micov's
  previous compute path as a fallback on those platforms would have meant two
  implementations that must agree numerically, and the coverage values and KS
  statistics are a frozen contract. The CI matrix drops `macos-13` and
  `windows-latest`; `macos-latest` is arm64, so macOS remains covered.
* DuckDB now needs to be `>=1.5.4`. The previous `>=1.2.0,<1.3` pin held
  because jemalloc failed to compile on very old Linux; duckdb-miint requires
  1.5.4, so the ceiling was unholdable. Existing `.coverage.parquet` and
  `.covered_positions.parquet` artifacts are unaffected in both directions:
  files written under 1.5.4 were verified to carry an identical schema and row
  set to files written under 1.2.2, to be consumable end to end by micov on
  1.2.2, and 1.5.4 reads 1.2.2-written files.
* `pyarrow` is no longer a dependency. micov never imported it; it was pulled
  in as the polars-to-DuckDB handoff, and no such handoff remains. Nothing
  about the CLI or the output formats changes.

* The `micov per-sample-group` subcommand is now `micov per-sample`. Click 8.2
  merged https://github.com/pallets/click/pull/2604, which implicitly strips
  `_group` suffixes when deriving a command name from its function name, so the
  `per_sample_group` callback registers as `per-sample`. micov previously pinned
  `click<8.2` to hold the old name; that pin has been removed and the shorter
  name is now canonical. `micov per-sample-group` no longer resolves.
