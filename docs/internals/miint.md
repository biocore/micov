# The miint extension

**Read this when** you add or change a call into miint, debug an extension
that won't load, or see published numbers move with no code change.

[duckdb-miint](https://github.com/the-miint/duckdb-miint) is a DuckDB
**extension**, not a Python package. It is required at runtime, with no
fallback: micov's frozen coverage and KS numbers can't depend on two compute
paths agreeing. The same maintainer owns micov and miint, so put each change
in the project that should own it. A missing primitive belongs in miint,
not in a hand-written workaround in micov.

## Loading: `_miint.connection(memory="8gb", threads=1)`

The only place micov opens a DuckDB connection. Do not call `duckdb.connect`
anywhere else.

1. **`duckdb.connect(":memory:")`**, with the config `threads`,
   `memory_limit` and `allow_unsigned_extensions: True`. Unsigned is required
   because miint's published builds are not signed; DuckDB refuses them
   otherwise. The code comment says to narrow this once miint is signed.
2. **The extension is found one of two ways:**
   - **`MICOV_MIINT_EXTENSION_PATH` set:** `LOAD '<path>'`, for local miint
     builds or offline hosts.
   - **Otherwise:** `INSTALL miint FROM '<repository>'`, then `LOAD miint`.
     The repository is `MIINT_REPOSITORY`
     (`https://ftp.microbio.me/pub/miint`), or `MICOV_MIINT_REPOSITORY` if
     set. If `INSTALL` fails for any reason, it retries once with
     `FORCE INSTALL`. That covers a cache from a different origin (micov
     used to install from the community repository) and a corrupt or
     partial cache.
3. **If any of that fails,** the DuckDB error is replaced by
   `RuntimeError(_unavailable_message(...))`, which names the platform, the
   DuckDB version (from `SELECT version()`, because `duckdb.__version__` is
   `None` on the 1.5.4 wheel) and both override variables.
4. **`_assert_capabilities(con)`** checks that every name in
   `REQUIRED_MIINT_FUNCTIONS` is in `duckdb_functions()`. If any is missing,
   it raises `RuntimeError` listing only the missing ones.

**Offline use.** `INSTALL` is a no-op when the extension is already cached
in `~/.duckdb/extensions/`, so only the first run needs the network. Hosts
without network access can seed that cache, or set
`MICOV_MIINT_EXTENSION_PATH`.

**Versions.** There is no version floor. `miint_version()` returns a git
short hash (for example `96630a5`), which cannot be ordered, so the
capability check stands in for a minimum version.

## Functions micov calls

Keep `REQUIRED_MIINT_FUNCTIONS` in step with this table. A name listed there
but never called rejects working installs; a call added without listing it
fails deep inside a query.

| Function | Kind | Signature (build `96630a5`) | Used by |
|---|---|---|---|
| `read_alignments` | table function | `(path, reference_lengths := <table>, include_filepath, include_seq_qual)`; micov reads `reference`, `position`, `stop_position` | `_io.compress_alignments` |
| `compress_intervals` | aggregate | `(BIGINT start, BIGINT stop) → STRUCT(start, stop)[]`; touching intervals merge | `_io.compress_alignments`, `View` region mode, `_depth.stage_breadth` |
| `compute_coverage_depth` | aggregate | `(BIGINT position, BIGINT stop_position, VARCHAR cigar, BIGINT ref_len, VARCHAR mode) → UINTEGER[]`; element i is base i + 1; `'include_deletions'` counts D, never N | `_depth.window_depth` |
| `cumulative_coverage` | aggregate | `(INTEGER rank, BIGINT start, BIGINT stop) → STRUCT(rank, covered)[]`; ranks must be contiguous `0..N-1` | `_cov.cumulative_covered` |
| `region_coverage` | table macro | `(positions, regions)`; emits `covered`, `region_length`, `proportion_covered` (0..1) per sample × region | `View` region mode |
| `region_presence` | table macro | `(positions, regions, samples)`; emits `sample_id`, `region_id`, `state` | `View.sample_presence_absence` |
| `ks_2samp` | scalar | `(DOUBLE[], DOUBLE[]) → STRUCT(statistic, pvalue)`; exact method only | `_plot.ks_2samp` |

How to call them:

- **Table macros need named relations.** Their arguments must be table or
  view names, never subqueries.
- **The `regions` relation** must have the columns `genome_id`,
  `region_start`, `region_stop` and `region_id`.
- **`compute_coverage_depth` returns NULL, not zeros, for a group whose rows
  it all skips** (a NULL, `*` or empty CIGAR, or a NULL position). Filter `IS
  NOT NULL` and back-fill with zeros, as `_depth.window_depth` does.
- **It windows exactly without padding.** A position below 1 still walks the
  CIGAR and drops the bases before 1, and bases past `ref_len` are dropped,
  so shifting positions by `w0 - 1` and passing the window's width gives that
  window of the genome's depth. For a CIGAR without N, `'include_deletions'`
  fills `[position, stop_position)` without walking the CIGAR, so the stop
  must be htslib's, as `read_alignments` reports it.

miint also offers `cumulative_coverage_curve`, a macro that ranks samples
itself. micov does not use it: it ranks in `_cov.ordered_coverage`. Both
break breadth ties by `sample_id`. Nobody has checked whether the macro could
replace micov's ranking, including its back-fill of zero-coverage samples.

## A stale cached build changes published numbers

`INSTALL` never replaces a cached build, and the capability check compares
names, not behaviour. A build older than the-miint/duckdb-miint#257 has
`ks_2samp`, but returns the un-snapped KS statistic: `0.30000000000000004`
where `0.3` was published. This happened on the maintainer's machine.

- **The test that catches it** is `test_plot.KsTwoSampleTests`.
- **The fix** is
  `FORCE INSTALL miint FROM 'https://ftp.microbio.me/pub/miint'`.
- **To see which build is loaded,** run `SELECT miint_version()`.

## Tests

- **`test_miint.py`** covers loading, the settings, both override variables,
  the error messages, the capability check, and that `View` goes through
  `connection()`.
- **`test_quoting.ApostropheExtensionPathTests`** loads the extension from a
  path that contains `'`.
- **`requires_miint_build`** skips tests that need a locally installed build
  file. This is the only legitimate kind of skip.
