# Commands

**Read this when** you change a command, or need to trace one from its click
callback down to SQL.

## The surface is frozen

The registered command set is exactly:

- `binning`
- `compress`
- `cov-to-parquet`
- `extract-sample-presence`
- `per-sample`
- `position-plot`
- the hidden alias `nonqiita-to-parquet`

`test_equivalence.TestCliSurface` pins this set, and it also pins that
`per-sample-group`, `qiita-coverage`, `qiita-to-parquet` and `consolidate` do
**not** resolve. Changing options or names needs explicit approval, and the
change goes in `ChangeLog.md`.

- **`per-sample`** comes from the callback `per_sample_group`: click ≥ 8.2
  strips the `_group` suffix (pallets/click#2604).
- **`--memory` and `--threads`** go to `_miint.connection`. The defaults are
  `16gb` and 4 on every command that has them, and `View`'s own defaults are
  `8gb` and 1.

## `compress`

`cli.compress` → `_io.load_genome_lengths` → `_io.compress_alignments` →
`_io.write_coverage_parquet`.

- **Input** is `--data`, which is a file or a directory, or stdin when
  `--data` is omitted. See [data-formats.md](data-formats.md) for what each
  form requires.
- **Ingest** is
  `read_alignments(path, reference_lengths := genome_lengths)`, grouped by
  reference, with `compress_intervals(position, stop_position)` per group.
  `--disable-compression` uses `list({start, stop})` instead, so overlapping
  intervals are kept. Both aggregates take only rows with
  `stop_position > position`, which leaves out unmapped reads.
- **Unattributed reads** have reference `*`. They are counted and warned
  about, then dropped. If no genome is left with an interval, the command
  raises. This includes a file whose reads are all unmapped.
- **Output** is `{output}.covered_positions.parquet` and
  `{output}.coverage.parquet` for this one sample.
- `cli_test.sh` checks that stdin and `--data` produce the same intervals.

## `cov-to-parquet` (alias `nonqiita-to-parquet`)

`cli.cov_to_parquet`:

- Reads `--pattern`, a glob of `.cov`/`.cov.gz` files, with
  `read_csv(..., filename=true)`. The sample id is the filename stem.
- Writes the Parquet pair through `write_coverage_parquet`.
- **Reads no SAM, so intervals are not re-merged.** They are written as the
  `.cov` files hold them.

The alias is `copy.copy(cov_to_parquet)` with `hidden = True`, so the two
names cannot drift into separate implementations.

## `per-sample`

`cli.per_sample_group` → `View(...)` → `_plot.per_sample_plots`.

- **`--plot` is accepted and ignored.** Plots are always written, and this
  was already so before the migration. Making it gate the plots would change
  what a plain run produces.
- **`--monte focused|unfocused`, `--monte-iters N`** add a Monte Carlo curve
  to every curve plot (see [curves-and-ks.md](curves-and-ks.md)).
- **`--percentile`** puts percentiles on the x-axis. It changes PNGs only:
  `test_percentile_does_not_alter_data_outputs` pins that the data files are
  unchanged.
- **`--target-names`** maps genome ids to display names, which appear in
  filenames and titles.

### Per-genome loop

`per_sample_plots`:

1. Materializes `View.positions()` into the temp table `plot_positions`,
   `ORDER BY genome_id`.
2. Fetches all coverage (one row per sample × genome) into numpy, and groups
   it by genome with an argsort.
3. Computes `sample_universe`, every sample with any coverage, which
   unfocused Monte Carlo draws from.
4. **For each genome,** fetches only its positions
   (`WHERE genome_id = ?`), then calls:
   - `coverage_curve` twice: non-cumulative, then cumulative. The cumulative
     call writes the `.ks.csv`.
   - `position_plot` twice: unscaled, then `scale=10000`. The scaled call
     writes the `.tsv.gz`.

   Plotting functions get **only their genome's rows**. That is a
   performance invariant; see [traps.md](traps.md).

In region mode, `per_sample_plots` raises if any genome has more than one
region: plotting multiple regions per genome is not supported.

## `binning`

`cli.binning` → `View` → `_quant.pos_to_bins` (SQL) → two TSVs.

- **Bins.** Each genome is split into `--bin-num` bins (default 1000) by
  `_quant.bin_list_sql`.
  - Breakpoints are `floor(i × length / n + 0.5)`, which rounds half away
    from zero, as polars `hist` did. numpy's round-half-to-even would move
    a bin edge.
  - The first bin starts at 0, and the last stops at `length + 1`.
- **Hits.** An interval counts in every bin it overlaps
  (`bin_stop > start AND bin_start < stop`), so `read_hits` summed over bins
  exceeds the read count.
- **Materialized once.** The positions-to-bins join is stored as `bin_stats`,
  and both output files read from it.
- **`--rank` has no effect**: the variance ranking is always written. The
  flag stays because the surface is frozen, and its help text says it has
  no effect.

## `extract-sample-presence`

`cli.extract_sample_presence` → `View(...)` →
`View.sample_presence_absence()` → TSV.

- **Requires region mode.** `--features-to-keep` must have `start`/`stop`
  columns; without them the command raises
  `Cannot calculate presence/absence without positions.`
- **The SQL** is miint's `region_presence(selected_positions,
  presence_regions, metadata)` followed by `PIVOT ... ON region_id USING
  FIRST(state)`. See [view.md](view.md).

## `position-plot`

`cli.position_plot`:

- Reads one BED3 file, `--positions` or stdin, with `load_bed_cov`, and
  lengths with `load_genome_lengths`.
- `_plot.single_sample_position_plot` draws one PNG per genome. The values
  come from `position_plot_segments`, which unit-normalizes each interval by
  its own genome's length.
- Genomes missing from `--lengths` are dropped silently, by an inner join.
