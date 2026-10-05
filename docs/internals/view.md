# `View`

**Read this when** you touch `micov/_view.py`, feature filtering, regions,
region breadth, or presence/absence.

`View(dbbase, sample_metadata, features_to_keep, feature_names=None,
threads=1, memory="8gb")` opens one connection (`_miint.connection`) and
builds a catalog of tables and views over `{dbbase}.coverage.parquet` and
`{dbbase}.covered_positions.parquet`. Callers read results through accessor
methods that return relations; nothing is fetched until a caller asks.
`View.close()` closes the connection, and `__del__` tolerates a connection
that never opened.

## Construction order (`_load_db`)

1. **Check that both Parquet files exist.** If either is missing, raise
   `OSError`.
2. **Load `metadata`** with `_io.read_tsv_with_header(..., SAMPLE_ID_COLUMNS,
   all_varchar=True)`, then `SEMI JOIN` it to `coverage.parquet` on
   `sample_id`. This happens **before** feature filtering on purpose:
   unfocused Monte Carlo needs every sample with any coverage, not only the
   samples covering the selected genomes.
3. **Run `_feature_filters()`.** It builds `feature_constraint` and picks the
   mode, as described in the next section.
4. **Create `unconstrained_positions`**, a view straight over the positions
   Parquet.
5. **Build the mode-specific `coverage`, `positions` and `feature_metadata`.**
6. **Run `_integrity_checks()`.** `region_id` must be unique within
   `feature_metadata`.

## The three filter modes

Two flags select the mode:

- **`constrain_positions`** is set when the feature file has `start` and
  `stop` columns. It is region mode.
- **`constrain_features`** is set when `feature_constraint` is non-empty.

What each mode produces:

| Mode | When | `coverage` | `positions` | `feature_metadata` |
|---|---|---|---|---|
| None | `features_to_keep is None`; `feature_constraint` is then every genome in coverage, with NULL bounds | Parquet ⋈ `metadata` | Parquet ⋈ `metadata` | per genome: `start=0`, `stop=length`, `region_id={genome}_0_{length}` |
| Genome | feature file with `genome_id` only | Parquet ⋈ `feature_constraint` ⋈ `metadata` | the same joins | as above |
| Region | feature file with `start`/`stop` | `recomputed_coverage`, from miint `region_coverage` | `recompressed_positions` | the feature rows plus `length = stop - start` and `region_id = {genome}_{start}_{stop}`, restricted to genomes present in `coverage` |

Notes on the table:

- **No-feature mode** builds `feature_constraint` from every genome in
  coverage, then returns early, so `constrain_features` stays False and the
  `else` branch runs. The genome and no-feature branches run the same
  `feature_metadata` SQL, copied between them; a `TODO` in the code marks it.
  A feature file with a header and no rows also leaves `constrain_features`
  False, and so falls into the no-feature branch.
- **Genome length** in the two non-region modes is
  `FIRST(length) GROUP BY genome_id` over `coverage`, as the view
  `genome_lengths`.

### Region mode in detail

Region mode is the only mode that computes anything new:

- **`selected_positions`** = unconstrained positions ⋈ `metadata`.
- **`regions`** has the columns `genome_id`, `region_start`, `region_stop`
  and `region_id`, which is the shape miint's table macros expect.
- **`clipped_positions`** clips each interval to its region with
  `GREATEST(start, fc.start)` and `LEAST(stop, fc.stop)`. It keeps only the
  intervals that overlap, `pos.start < fc.stop AND pos.stop > fc.start`. The
  comparison is strictly `<`, because the region is half-open:
  `test_view.test_abutting_interval_is_outside_the_region` and
  `test_no_zero_width_intervals_reach_positions` pin it.
- **`recompressed_positions`**: `compress_intervals(start, stop)` per
  `(sample_id, genome_id)`, because clipping can create overlaps. This is the
  same merge primitive `compress` uses. If the table is empty, the
  constructor raises `No positions left after filtering.`
- **`recomputed_coverage`**: `region_coverage(selected_positions, regions)`,
  with the unclipped positions as input, since the macro clips and merges
  internally.
  - `covered` and `length` are the region's, cast to UINTEGER.
  - `percent_covered = proportion_covered * 100`, with the multiplication
    applied **after** miint's division. The order changes the double, and
    `test_view` pins a literal.
- **Validation** happens in `_feature_filters`, so the errors are in the
  user's terms:
  - a region with `stop <= start` raises `ValueError` naming the row
  - `start` without `stop`, or `stop` without `start`, raises `KeyError`

## Accessors

- **`metadata()`** returns the filtered sample metadata, all VARCHAR, with
  first column `sample_id`.
- **`coverages()` and `positions()`** return the mode's views. If the
  Parquet stored `covered`/`length` or `start`/`stop` as BIGINT (older micov
  files), they cast them to UINTEGER, so callers see one dtype whichever
  micov wrote the file.
- **`feature_metadata()`** returns `genome_id, start, stop, length,
  region_id`, plus any extra feature columns in region mode.
- **`feature_names()`** returns `genome_id, name`. Without `--target-names`,
  the name is the genome id. With it, the names are cleaned as described in
  [data-formats.md](data-formats.md) and left-joined, falling back to the
  genome id.
- **`sample_presence_absence()`** works in region mode only.
  1. It creates `presence_regions` from `feature_metadata`.
  2. It runs `region_presence(selected_positions, presence_regions,
     metadata)`, which emits `present`, `absent` or `not applicable` for
     **every** sample × region pair. A sample that is both present and absent
     in one region resolves to present.
  3. It `PIVOT`s the result into a materialized table. A pivot cannot be a
     view when its columns are not known in advance.

## Header rule (`_io.read_tsv_with_header`)

`read_tsv_with_header(con, path, rename, first_column, all_varchar=False)`
returns SQL rather than a relation, so callers can embed it in a larger
statement. It lives in `_io` rather than on `View` so that readers of other
inputs, such as `depth-plot`'s, enforce the same rule.

- **It runs `DESCRIBE` on `read_csv(path, delim='\t', header=true)`** and
  raises `ValueError` unless the first column is in `first_column`.
- **Without this check, a headerless file loses its first row silently**:
  DuckDB reads that row as the column names, and the genome or sample is
  dropped from every output.
- **Only the leading columns are renamed.** Any others pass through,
  double-quoted by name.

The rule is pinned by `test_view.test_headerless_*`, the
`*_must_name_genome_id_first` tests, and `test_io.ReadTsvWithHeaderTests`.

## Table macros need named relations

`region_coverage` and `region_presence` are miint **table macros** built on
`query_table()`. Every argument must be the name of a table or view; a
subquery is a Binder Error. That is why `selected_positions`, `regions` and
`presence_regions` exist as views instead of being inlined.
