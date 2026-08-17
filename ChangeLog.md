micov ChangeLog
=====================

micov 0.0.1-dev
---------------

Backward incompatible changes:

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
