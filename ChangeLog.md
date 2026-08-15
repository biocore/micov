micov ChangeLog
=====================

micov 0.0.1-dev
---------------

Backward incompatible changes:

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
