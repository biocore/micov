# Known traps

**Read this before any non-trivial change.** Every entry here has already
caused a real bug or a near miss. Each one names the code involved and the
test that guards it.

## Outputs and numbers

- **Never format float outputs with `repr()` or an f-string.**
  `_plot._write_delimited` relies on `csv.writer` rendering with `str()`,
  which gives the shortest round-trip form. The position values are numpy
  scalars (from `np.histogram`), and under numpy 2, `repr(np.float64(0.3))`
  is the string `np.float64(0.3)`. That would corrupt every `.ks.csv` and
  `.tsv.gz`.
- **`(covered / length) * 100`, never `covered * 100 / length`.** The two
  give different doubles, and the published values used the first. The order
  is the same in `_io.write_coverage_parquet`,
  `_cov.cumulative_curves` and `View`'s `proportion_covered * 100`.
- **`ks-pvalue` is raw.** Bonferroni lives only in `ks-pvalue-bonferroni`.
  Never correct the published column in place.
- **Do not widen the golden tolerance onto `ks-statistic`.** It matches
  exactly. Only the p-values, from miint rather than scipy, drift by a few
  ULP.
- **Identify the Monte Carlo curve by the label `add_monte` returns**, never
  by the `Monte Carlo ` text prefix. A metadata group can be named that way.
  (The golden comparator does use the prefix, which is fine for the
  committed fixtures.)
- **Breadth ties are broken by `sample_id`.** Do not reintroduce a
  dependence on row order anywhere on the curve path. `View` relations come
  from parallel scans, and their order changes between runs.

## Performance

- **Never hand a plotting function the whole positions table.**
  `per_sample_plots` keeps positions in the genome-sorted temp table
  `plot_positions` and fetches one genome at a time. `coverage_curve`,
  `position_plot` and `add_monte` receive only that genome's rows.
  - Filtering the full table per genome was four numpy string scans per
    genome. On 9,388 genomes and 29.7M intervals it made `per-sample` 2.5×
    slower than the polars release.
  - `example/` has 2 genomes and cannot show this.
    `test_plot.PerSamplePlotsPerGenomeTests` pins it.
- **Close every figure.** `coverage_curve` returns early when no group
  reaches 10 samples, which is most genomes in a real study. An unclosed
  figure lives until the process exits.
  `test_a_genome_with_no_large_group_leaves_no_figure_open` pins this.
- **Unfocused Monte Carlo needs `sample_universe`.** It is computed once,
  before the loop, because one genome's rows cannot tell it.

## Inputs

- **Metadata and feature files must have a header.** `View._read_tsv` raises
  unless the first column is `genome_id` (features, regions,
  `--target-names`) or `sample_id`/`sample_name` (sample metadata).
  Otherwise a headerless file, such as a taxonomy `lineages.txt`, would have
  its first row read as column names, and that genome or sample would vanish
  from every output with no error. Released micov has this bug.
- **`_test_has_header` uses `==`, not `in`.** `COLUMN_GENOME_ID` is a plain
  string, so `x in COLUMN_GENOME_ID` is a substring test. A headerless
  lengths file whose first genome was `id` or `genome` silently lost that
  row. `test_io` pins this.
- **Every value interpolated into a SQL string literal goes through
  `_utils.sql_string`.**
  - Paths and `--sample-id` are user input, and `/Users/o'brien` is an
    ordinary home directory.
  - Bound parameters are not an option at most sites. The literals sit in
    SQL fragments spliced into larger statements, several of them
    `CREATE VIEW`, which DuckDB will not prepare. Bound parameters do work
    in plain queries, for example `WHERE genome_id = ?`, so use them there.
  - `test_quoting.py` drives every literal site with an apostrophe path.
- **Identifiers built from file headers are not escaped.** `"{column}"` in
  `_read_tsv` and `load_genome_lengths` breaks on a header containing `"`.
  This is known, and has not been fixed.
- **`compress` reads its input once.** Anything needing a second pass breaks
  the stdin idiom.
- **The DuckDB CSV sniffer cannot read a pipe.** It consumes the stream and
  then returns zero rows without raising. That is why `position-plot` spools
  stdin through `_io.positions_path`.

## Environment and packaging

- **A stale cached miint passes the capability guard and changes published
  numbers.** See [miint.md](miint.md). The fix is `FORCE INSTALL miint FROM
  'https://ftp.microbio.me/pub/miint'`. `test_plot.KsTwoSampleTests` is what
  notices.
- **`pyproject.toml` and `ci/conda_requirements.txt` must declare the same
  dependencies, bounds included.** CI's conda path installs with
  `pip install . --no-deps`, so when the two disagreed, the conda and pypi
  paths silently tested different duckdb majors.
- **`numpy` is imported directly but not declared.** It arrives through
  matplotlib. Declare it before relying on a numpy floor.
- **`MANIFEST.in` has `graft micov`**, so any stray file under `micov/`,
  tracked or not, goes into the sdist and breaks `check-manifest`. Keep
  scratch work in `localdocs/`, which is gitignored and pruned.
- **Only `make lint-fix` edits files.** ruff runs with `fix = false`, and
  `make lint` only reports.
- **Python 3.13 is not claimed.** Nothing blocks it any more, but nothing
  has been run on it either. Add it to the CI matrix before claiming it.

## Long-standing behaviour that looks like a bug

These are frozen, or awaiting a decision. Do not "fix" one in passing:
changing it changes outputs or the CLI, so raise it with the maintainer
first.

- **`per-sample --plot` does nothing.** Plots are always written, on `main`
  as well.
- **`binning --rank` does nothing.** The ranking is always written, and the
  help text says so.
- **`coverage_curve` considers only the first 10 metadata values** in sorted
  order: `zip(..., range(10), strict=False)`.
- **The scaled position plot bins interval endpoints, not covered bases.** A
  bucket that a long interval spans, but which contains neither of its ends,
  is not marked.
- **Monte Carlo is unseeded**, so its rows differ between runs.
- **`stats_by_variance_of_sample_hits.tsv` is ordered by
  `sample_hits_std DESC` alone**, so ties come out in arbitrary order.
- **`cov-to-parquet` does not re-merge intervals.** Overlapping intervals in
  a `.cov` are written as given.
