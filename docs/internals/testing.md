# Testing

**Read this when** you add a test, a golden fails, or you need to know what
a tier covers before claiming a change is verified.

## Running

```bash
make test                          # pytest micov, then bash cli_test.sh
MICOV_GOLDEN_FULL=1 pytest micov   # adds the full example/ corpus (~1-2 min)
make lint                          # ruff check micov + check-manifest; reports only
make lint-fix                      # the only target that edits files
```

- **Lint tools.** `make lint` needs `pip install -r
  ci/requirements.lint.txt` (`ruff==0.16.9`, `check-manifest==0.51`).
  Otherwise it runs whatever `ruff` is on `PATH`.
- **Test dependencies.** `pytest` comes from the `[test]` extra:
  `pip install -e ".[test]"`.
- **Which micov the CLI tests run.** The CLI tests run the `micov` console
  script found **next to `sys.executable`**, and fall back to `PATH`. A stale
  non-editable install, or an editable install of a different checkout, would
  shadow the working tree. The unit tests would still pass against your
  tree, because pytest puts the cwd first on `sys.path`, while the CLI goldens
  tested other code. Check with `cd /tmp && python -c "import micov;
  print(micov.__file__)"`, and if it is wrong, export `PYTHONPATH=$PWD`.

At the time of writing:

| Tier | Result |
|---|---|
| Fast | 290 passed, 10 skipped |
| Full | 300 passed, 0 skipped |

The 10 fast-tier skips are the `requires_full_tier` tests. Any other skip
needs a reason: `requires_miint_build` is legitimate, and a skip you caused
is not.

## Two tiers

- **Fast** uses only `micov/tests/test_data/`. It has to run from an
  unpacked sdist, and `MANIFEST.in` prunes `example/` from sdists.
- **Full** (`MICOV_GOLDEN_FULL=1`) adds `example/`:
  - 49 SAM files compressed and compared against
    `example/coverages/*.cov.gz`
  - the Parquet pair
  - binning
  - presence
  - the four `per-sample` variants (`plain`, `percentile`, `monte`,
    `monte_percentile`) compared against `example/plots/per_sample_groups*`

  It also checks that `--percentile` leaves the data files unchanged, and
  that `--monte` leaves group-vs-group KS rows unchanged.

`example/` has **2 genomes**. It cannot reveal per-genome costs, or
behaviour that only shows when most genomes have no group of at least 10
samples. Test those with synthetic data spanning thousands of genomes; see
`test_plot.PerSamplePlotsPerGenomeTests` for the pattern.

## Where tests live

| File | Covers |
|---|---|
| `test_equivalence.py` | End-to-end CLI runs in subprocesses, compared against goldens. Also the frozen command surface and the `nonqiita-to-parquet` alias |
| `test_alignments.py` | The miint ingest: coordinates, merging, the reference-map guard, and the Parquet pair written by `compress` |
| `test_cov.py` | Ranking, accumulation and the tie-break; `IntervalMergeTests` checks merge cases against `compress_intervals` |
| `test_view.py` | `View` modes, region clipping and breadth, presence, feature names, the header rule |
| `test_plot.py` | `position_plot_segments`, `ks_2samp`, `ks_table`, the per-genome loop's slicing, Monte Carlo pool and figure closing, and group colours: their colour-blind separation, dashes past five groups, the warning past ten |
| `test_quant.py` | Bin edges and hit counts |
| `test_io.py` | Lengths parsing and header detection; BED3 loading; the shared header rule; `depth-plot`'s readers |
| `test_depth.py` | `depth-plot`'s computation: which samples and genomes are used and what is reported, windowed per-base depth, merged breadth, and the binned group statistics, hand-computed on the `dp_*` fixtures and on literal reads, plus window-size invariance and a CIGAR-walking oracle |
| `test_miint.py` | Connection, overrides, error messages, capability check |
| `test_quoting.py` | `sql_string`, and every SQL literal site driven with a path containing `'` |
| `test_dependencies.py` | No module imports `polars`, `numba`, `pyarrow` or `scipy` |
| `test_golden_selftest.py` | Each `_golden` comparator passes what it must tolerate and fails what it must catch |

## Goldens

- **`example/`** holds the full-tier goldens:
  - `coverages/` is the pre-miint `compress` output
  - `parquet/`
  - `binning/`
  - `plots/per_sample_groups*/`
- **`micov/tests/test_data/golden/`** holds the fast-tier goldens.
  `micov/tests/test_data/README.md` explains each fixture and how to
  regenerate it.

**Never refresh a golden to make a test pass**, and never change an expected
value without the maintainer's permission. A failing equivalence test means
the code changed, until there is evidence otherwise. Goldens are only
regenerated in an approved, recorded change. For example, M11a appended the
Bonferroni column to the `.ks.csv` goldens and kept columns 1–4 byte for
byte.

Some goldens came from the old stack:

- The KS goldens were made with **scipy 1.17.1**.
- The `sample_hits_std` values come from **polars**.

micov uses neither now, which is why two tolerances exist (see below).

## Comparators (`micov/tests/_golden.py`)

Every comparator is `(observed, expected)`, and raises `AssertionError`
naming both paths and the first difference.

| Comparator | Exact | Tolerated |
|---|---|---|
| `assert_ks_equal` | header (5 names), labels, `ks-statistic`, row set | `ks-pvalue` and `ks-pvalue-bonferroni` within `TSV_FLOAT_REL_TOL = 1e-14`. Monte Carlo rows (label prefix `Monte Carlo `) are checked only for presence, values in [0, 1], and an **empty** Bonferroni field |
| `assert_gzip_text_equal` | decompressed content | gzip header bytes (mtime) |
| `assert_parquet_equal` | ordered schema, row multiset (bidirectional `EXCEPT`) | row order |
| `assert_tsv_equal_unordered` | everything except named float columns | row order within the sort keys; `float_columns` within tolerance (`sample_hits_std`) |
| `assert_png_plausible` | — | only existence, non-zero size and PNG magic, because PNG bytes vary with matplotlib, backend and fonts |
| `assert_file_set` | the set of output filenames | — |

**Never widen a tolerance onto `ks-statistic`.** It is an exact lattice value
and matches the published one bit for bit.

The module docstring of `_golden.py` numbers the sources of
nondeterminism. Other code cites them by number, and #3 and #8 are retired.
Before changing a comparator, read that list and add a negative control to
`test_golden_selftest.py`.

## Writing tests here

- **Red first, and red for the intended reason.** If a test passes before the
  change, it is broken.
- **Encode why, not only what.** A test that cannot fail when the business
  logic changes is wrong. Docstrings in this suite say what would break for
  a user.
- **Mutation checks.** For a key assertion, break the code deliberately and
  watch the test fail. Restore the code from an in-memory copy in a
  `finally`, never with `git checkout`.
- **Tables in unit tests** are dict-of-numpy. String columns are `object`
  arrays and integer columns use the Parquet dtypes; `test_cov.py`'s
  `COVERAGE_COLUMNS`/`POSITION_COLUMNS` helpers show the shapes.
- **New miint calls** need their function name added to
  `REQUIRED_MIINT_FUNCTIONS`.
  `test_miint.test_every_required_function_is_present` then checks it on
  the installed build.
