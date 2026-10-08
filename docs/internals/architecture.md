# Architecture

**Read this when** you need to know where something lives, how data moves
between modules, or what the dependency surface is.

## What micov computes

**Breadth of coverage** is the fraction of a reference genome covered by at
least one read. micov computes it per sample per genome, and compares it
across sample groups defined by metadata. Its core contribution is the
**cumulative coverage curve**:

1. Rank a group's samples by their own breadth, lowest first.
2. Merge their covered intervals in that order. The value at rank *k* is the
   breadth of the union of ranks 0..k.

Weak individual signals stack into a detectable one. Supporting analyses are
Monte Carlo null curves, pairwise KS tests between curves, bin-and-rank region
discovery (`binning`), and per-region presence calls.

Paper: Weng Y, Guccione C, McDonald D, et al., *Communications Biology* 2025,
[PMC12635244](https://pmc.ncbi.nlm.nih.gov/articles/PMC12635244/).

## Runtime dependencies

The package dependencies are `click`, `matplotlib` and `duckdb`. The
`duckdb` pin is `>=1.5.4,<=1.5.5`; its upper bound exists only because the
miint repository has no build newer than v1.5.5. The **miint DuckDB extension**
is also required at runtime, but it is not a Python package and cannot be
declared. `_miint.connection` installs and loads it (see [miint.md](miint.md)).
`numpy` comes in through matplotlib and is used directly.

`pyproject.toml` and `ci/conda_requirements.txt` must list the same
dependencies. `test_dependencies.DependencySurfaceTests` asserts that nothing
imports `polars`, `numba`, `pyarrow` or `scipy`.

## Data flow

```
SAM/BAM (headerless)                       .cov / .cov.gz (BED3, read-only)
   │ micov compress                           │ micov cov-to-parquet
   │ _io.compress_alignments                  │ (hidden alias: nonqiita-to-parquet)
   │   read_alignments + compress_intervals   │ DuckDB read_csv glob;
   │                                          │ sample_id from the filename
   └──────────────┬───────────────────────────┘
                  ▼  _io.write_coverage_parquet (the only producer)
{base}.covered_positions.parquet  genome_id, start, stop, sample_id      (large)
{base}.coverage.parquet           sample_id, genome_id, covered, length,
                                  percent_covered                        (small)
                  │
                  ▼  _view.View: in-memory DuckDB, three filter modes
     ┌────────────┼───────────────────────┬───────────────────────────┐
     ▼            ▼                       ▼                           ▼
per-sample    binning                 extract-sample-presence     (position-plot
_plot.        _quant.pos_to_bins      View.sample_presence_absence reads one .cov,
per_sample_                                                        not the Parquet)
plots
```

`depth-plot` reads none of these:

```
read_alignments Parquet + sample_id (depth layer, breadth layer), read_gff
Parquet, features (with lengths), metadata
   │ micov depth-plot: _io readers → _depth.intersect_layers
   ▼ _depth.stage_breadth, stage_depth_reads (once, by genome)
per genome: _depth.genome_statistics → _depth_plot.linear_plot,
            circular_plot, detail_plot → PNGs
            _io.add_orf_table ──→ _io.write_orf_table (once) → per-ORF Parquet
```

The two-file Parquet split is load-bearing. `coverage.parquet` has one row
per sample × genome, and drives ranking and filtering.
`covered_positions.parquet` is large, and is only read through views with
join and filter predicates.

## Modules

| Module | Owns |
|---|---|
| `cli.py` | The click command surface: argument parsing and glue only. `compress` input-source logic, `depth-plot`'s usage errors, plus the hidden `nonqiita-to-parquet` alias (a `copy.copy` of the command object) |
| `_miint.py` | `connection()`, **the only place micov opens a DuckDB connection** and the only place the extension's install source (`MIINT_REPOSITORY`) is named. `REQUIRED_MIINT_FUNCTIONS` is checked on every connection |
| `_io.py` | Input parsers (`load_genome_lengths`, `load_bed_cov`, `_test_has_header`, and the header rule `read_tsv_with_header`), the SAM/BAM ingest (`compress_alignments`), `write_coverage_parquet`, `target_names_query` (the `--target-names` transform, shared by `View` and `depth-plot`), `depth-plot`'s readers (`load_alignment_layer`, `load_depth_features`, `load_sample_groups`, `load_orfs`), and its per-ORF table (`start_orf_table` once a run, `add_orf_table` per genome, `write_orf_table` once; the frozen `ORF_STATISTICS_COLUMNS`); `PARQUET_OPTIONS`, how every Parquet output is written |
| `_view.py` | `View`: loads the Parquet pair, metadata and feature constraints into one connection, and exposes `coverages()`, `positions()`, `metadata()`, `feature_metadata()`, `feature_names()`, `sample_presence_absence()` as relations |
| `_cov.py` | Ranking (`ordered_coverage`) and accumulation (`cumulative_covered`, `cumulative_curves`, `compute_cumulative`). Operates on dict-of-numpy tables; the accumulation itself is miint SQL |
| `_plot.py` | `per_sample_plots` (the per-genome loop), `coverage_curve`, `add_monte`, `position_plot`, `ks_2samp`/`ks_table`, the single-sample `position-plot`, and `_write_delimited` |
| `_quant.py` | `binning`'s SQL: `bin_list_sql`, `pos_to_bins`, `create_bin_list` |
| `_depth.py` | `depth-plot`'s computation ([depth-plot.md](depth-plot.md)): `intersect_layers` (which samples and genomes are used, and the report of those left out), then per genome `genome_statistics`: binned group statistics from windowed per-base depth (`stage_depth_reads` once, `stage_depth`, `window_depth`) and merged breadth (`stage_breadth`, `coverage_counts`), and per-ORF statistics (`genome_orfs`, `orf_segments`, `orf_contrast`) from the same pass |
| `_depth_plot.py` | `depth-plot`'s drawing ([depth-plot.md](depth-plot.md)): `linear_plot` (the overview, in rows of up to 2 Mb), `detail_plot` (one region) and `circular_plot` (a circular genome's ring), and the pure helpers they use: ticks, spans, highlights (`parse_highlight`, `highlight_masks`), ORF colours (`orf_track`, `orf_categories`, `contrast_colors`), shapes, label lanes, and the ring's geometry and label placement; and `depth_plots`, the per-genome run the command calls |
| `_constants.py` | Frozen column names (`COLUMN_*`) and the three presence states |
| `_utils.py` | The `micov` logger, and `sql_string`, the one way a value enters a SQL string literal |

## Two table representations

- **DuckDB relations and tables** are used for everything set-shaped: parsing,
  joins, filtering, aggregation, binning and presence.
- **dict of numpy arrays**, the shape `DuckDBPyRelation.fetchnumpy()` returns,
  is used on the curve and plot path (`_cov`, `_plot`) and for `depth-plot`'s
  bins (`_depth`). String columns arrive
  as `object` arrays. `_cov.mask_table(table, keep)` applies a boolean mask or
  an index array to every column at once.

When moving logic between the two, watch numpy promotion: `uint64 + int`
becomes `float64`. The rank columns (`x_unscaled`) are `uint64`, and the code
adds `np.uint64(...)` or calls `int(...)` deliberately.

## Connection catalog

`View` names its tables and views in one connection: `metadata`,
`feature_constraint`, `unconstrained_positions`, `coverage`, `positions`,
`feature_metadata`, `genome_lengths`. In region mode it adds
`selected_positions`, `regions`, `clipped_positions`,
`recompressed_positions` and `recomputed_coverage`. Other modules that share
that connection choose names that cannot collide:

- `_cov.CURVE_INPUT_RELATION` (`micov_curve_input`) is registered and
  unregistered around each accumulation.
- `_plot.PLOT_POSITIONS_TABLE` (`plot_positions`) is a temp table.
- `binning` creates the views `binning_positions`, `binning_metadata` and
  `binning_features`, and the table `bin_stats`.
- `_io` uses `genome_lengths`, `alignment_groups`, `alignment_positions` and
  `bed_positions`, but on its own connection (`compress`, `cov-to-parquet`,
  `position-plot`), never on a `View`'s.
- `depth-plot` never builds a `View`, because its inputs are not the Parquet
  pair. On its own connection it uses the views `depth_layer` and
  `breadth_layer`, the tables `depth_features`, `sample_groups`, `orfs`,
  `depth_roster` and `depth_genomes`, the temp tables `depth_reads` (every
  plotted genome's reads, by genome), `depth_alignments` (one genome at a
  time), `breadth_intervals` and `depth_orf_statistics` (the per-ORF table,
  a genome at a time), and the relation `depth_orf_genome`, registered while
  one genome's per-ORF table is copied in.

## Platform

Linux (x86_64, aarch64) and macOS on Apple silicon. There are no miint builds
for Windows or Intel macOS. micov reaches the network on its first run to
fetch the extension; [miint.md](miint.md) covers offline use.
