# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ABSOLUTE HARD REQUIREMENTS
- NEVER use `rm` without permission

## Project Overview

**micov** (aggregate MIcrobiome COVerage) computes **breadth of coverage** — the fraction of a reference genome covered by at least one read — per sample per genome, and compares it across metadata-defined sample groups.

The distinguishing capability is the **cumulative coverage curve**: rank samples within a group by their own breadth, then accumulate covered intervals from lowest to highest. Individually weak signals stack into a detectable one (the authors' analogy is long-exposure astrophotography), which surfaces genomic regions present in one sample group and absent in another even when no single sample has good coverage. Supporting analyses are Monte Carlo null curves, pairwise KS tests between curves, and bin-and-rank region discovery.

Reference: Weng Y, Guccione C, McDonald D, et al. "Calculating fast differential genome coverages among metagenomic sources using micov." *Communications Biology* 2025. [PMC12635244](https://pmc.ncbi.nlm.nih.gov/articles/PMC12635244/)

micov is published, on pip and conda. It was integrated with Qiita; **that support was removed in M4** and may be revisited. Existing `.cov` and Parquet artifacts and the CLI surface are **contracts** — see Compatibility below.

## Priorities

1. Red/green/refactor Test Driven Development (TDD)
2. Verifiably correct code
3. Maintainable code, using Don't Repeat Yourself (DRY) and Keep It Simple Stupid (KISS)
4. Performance

## Rules

These rules apply to every task in this project unless explicitly overridden.

### Rule 1 — Think Before Coding
Bias: caution over speed on non-trivial work.
State assumptions explicitly. Ask rather than guess.
Push back when a simpler approach exists. Stop when confused.

### Rule 2 — Simplicity First
Minimum code that solves the problem. Nothing speculative.
No abstractions for single-use code.

### Rule 3 — Surgical Changes
Touch only what you must. Don't improve adjacent code.
Match existing style. Don't refactor what isn't broken.

### Rule 4 — Goal-Driven Execution
Define success criteria. Loop until verified.

### Rule 5 — Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.

### Rule 6 — Read before you write
Before adding code, read exports, immediate callers, shared utilities.
If unsure why existing code is structured a certain way, ask.

### Rule 7 — Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when business logic changes is wrong.

### Rule 8 — Checkpoint after every significant step
Summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back.

### Rule 9 — Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase.
If you think a convention is harmful, surface it. Don't fork silently.

### Rule 10 — Fail loud
"Completed" is wrong if anything was skipped silently.
Don't silently skip tests you caused to be skipped (commented out, xfail'd, missed).
Default to surfacing uncertainty, not hiding it.

## Build and Test

```bash
conda create -n micov -c conda-forge python=3.12
conda install -q --yes -n micov -c conda-forge --file ci/conda_requirements.txt
conda activate micov
pip install -e ".[test]"
```

`-c conda-forge` on `conda create` is required on hosts with no configured
channels. `[test]` installs `pytest`, which `make test` needs — it is an extra,
not a runtime dependency. Lint tools come from `ci/requirements.lint.txt` and
are **not** installed by the above; `make lint` otherwise silently uses
whatever `ruff` is on `PATH`.

```bash
make test     # pytest micov + bash cli_test.sh (stdin-vs-file compress equivalence)
make lint     # ruff check micov + check-manifest
```

Ruff is configured in `pyproject.toml` with `fix = true`, line length 88, numpy docstring convention. `micov/tests/*` is exempt from `D` and `PT` rules.

If a test produces an **incorrect expected value**: DO NOT change the expected value without permission.

## Compatibility contract

micov is published and its outputs are in the wild. Unless a task explicitly overrides this:

- **`.cov` is now read-only, and `cov-to-parquet` is its only reader.** `micov compress` stopped writing BED3 in M3; M4 removed the Qiita readers. Existing `.cov` artifacts stay valid. `example/coverages/*.cov.gz` are kept as committed fixtures and are the only independent record of what the pre-miint implementation produced.

- **CLI surface is frozen**, and the migration has taken four approved exceptions. The registered set is now exactly: `binning`, `compress`, `cov-to-parquet`, `extract-sample-presence`, `per-sample`, `position-plot`, plus the hidden alias `nonqiita-to-parquet`. Every exception is recorded in `ChangeLog.md`.
  1. **`per-sample`** is the canonical name; the `click<8.2` pin was removed. pallets/click#2604 strips the `_group` suffix when deriving a command name, so the `per_sample_group` callback registers as `per-sample`; `per-sample-group` no longer resolves.
  2. **Windows and Intel macOS were dropped** (see Platform support below).
  3. **`compress` takes SAM/BAM only** and writes `{output}.coverage.parquet` + `{output}.covered_positions.parquet`; `--lengths` and `--output` became required, `--sample-id` was added, and the BED3 input path, the two TSV summary modes and `--taxonomy` were removed.
  4. **Qiita support was removed and `nonqiita-to-parquet` renamed.** `qiita-coverage`, `qiita-to-parquet` and `consolidate` are gone; `nonqiita-to-parquet` became `cov-to-parquet` and survives only as a hidden alias. `micov/tests/test_equivalence.py` pins all of this — the frozen set, that the three Qiita names do not resolve, and that the alias is hidden but works.
- **Output file formats are frozen.** `.cov` / `.cov.gz` BED-like TSVs, and `{base}.coverage.parquet` / `{base}.covered_positions.parquet` column names and types, must stay readable by released micov versions. The Qiita `coverages.tgz` layout is no longer produced or read; no specimen of it survives in the repo, so reviving it means reconstructing the layout from git history (`example/consolidate/consolidated.tgz`, removed in M4).
- **Published numbers must reproduce.** Coverage values, KS statistics, and p-values are cited in the paper. A change that moves them is a regression, not an improvement.
- **Platform support was narrowed, deliberately.** Windows and Intel macOS were dropped because miint publishes no build for them; supported platforms are Linux (x86_64, aarch64) and macOS on Apple silicon. micov also now needs network access on its **first** run to fetch the extension — a deployment-surface change that matters for HPC compute nodes without egress. `MICOV_MIINT_EXTENSION_PATH` and a pre-seeded `~/.duckdb/extensions/` are the two ways around it.

## Coordinate and coverage conventions

- Intervals are **1-based half-open** `[start, stop)`. `start` is SAM `POS`; `stop = POS + reference span from CIGAR` (`M/=/X` advance alignment, `D/N` advance reference, so deletions and gaps count as covered).
- Breadth is `sum(stop - start)` over merged intervals — **no `+1`**.
- Interval merging **collapses touching intervals** (`stop1 == start2` → one interval). This is intentional and is now miint's behaviour throughout: `compress_intervals` for the ingest and the region clip, `cumulative_coverage` for the curves. `_cov.merge_intervals`, micov's own numpy merge, was deleted in M7 once the aggregate took over the accumulation; its cases live on in `micov/tests/test_cov.py` pointed at `compress_intervals`, alongside `micov/tests/test_alignments.py`. The `compress` docstring's stale case 3, which claimed the opposite, was corrected in M3.
- `.cov` files are described as "BED-like" but carry 1-based coordinates, whereas real BED is 0-based. Breadth math is unaffected; joins against genuinely 0-based sources are not.
- **Regions are half-open too**, so an interval starting exactly at a region's `stop` is outside it. micov's overlap predicate was `pos.start <= fc.stop`, which admitted that interval, clipped it to a zero-width `[stop, stop)`, and reported the sample present in a region it covered no bases of. M6 adopted miint's `<`. Regions with `stop <= start` are now rejected in `_feature_filters` rather than reaching the SQL, where they surfaced as a UINT32 subtraction overflow.
- Primary *and* secondary alignments are retained deliberately, so CNVs, horizontally transferred elements, and repeats are represented rather than silently dropped.

## Architecture

```
SAM/BAM (headerless)                          .cov / .cov.gz  (read-only now)
   │  micov compress                              │  BED3; stem is the sample_id
   │  read_alignments + compress_intervals        │
   │  --lengths is the reference map              │  micov cov-to-parquet
   │                                              │  (was nonqiita-to-parquet)
   ▼                                              ▼
{base}.covered_positions.parquet   (genome_id, start, stop, sample_id)   — large
{base}.coverage.parquet            (genome_id, sample_id, covered, length, percent_covered) — small
   │  View (in-memory DuckDB, pushdown-filtered)
   ▼
per-sample-group  →  curves, position plots, KS tests
binning           →  per-bin stats, variance ranking
extract-sample-presence → per-sample present/absent/NA per region
```

The two-file Parquet split is load-bearing: `coverage.parquet` is one row per sample×genome and drives ordering and filtering; `covered_positions.parquet` is large and only ever scanned with pushdown predicates.

| Module | Role |
|---|---|
| `cli.py` | click command surface; the only user-facing contract |
| `_view.py` | `View` — DuckDB session with three filter modes: none, genome-level (`constrain_features`), sub-genome region (`constrain_positions`). Only the third does real work, and since M6 it is miint's: the clipped intervals are re-merged by `compress_intervals`, region-relative breadth comes from `region_coverage`, and `sample_presence_absence` is `region_presence` plus a `PIVOT` |
| `_cov.py` | rank ordering (`ordered_coverage`) and cumulative accumulation, all on the curve path. **Polars-free since M4**; since M7 `compute_cumulative`/`cumulative_curves` accumulate with miint's `cumulative_coverage` aggregate, so this module now needs a DuckDB connection and `merge_intervals` is gone |
| `_io.py` | parsers/writers, all DuckDB since M5. `load_bed_cov` and `load_genome_lengths` read input into tables; `compress_alignments` is the miint ingest; `write_coverage_parquet` is the single producer of the frozen Parquet pair |
| `_plot.py` | matplotlib curves and position plots, plus the KS tests. Largest module; `position_plot_segments` is the only part with unit tests, added in M5 because `position-plot` writes no data file and so had no golden |
| `_quant.py` | binning |
| `_constants.py` | column names and the three presence/absence values. The polars dtypes and seven `_SCHEMA` objects went in M5 |
| `_miint.py` | `connection()` — the only place micov opens a DuckDB connection, **and the only place the extension's install source is named**. `cli.py` was moved behind it so that stays true. `REQUIRED_MIINT_FUNCTIONS` lists every miint function micov calls and is checked on each connection (names only, not signatures) |

`micov/_convert.py` is **gone** — htslib computes the reference span now. `micov/_rank.py` is **dead code** — nothing imports it, and it pulls in an undeclared pandas dependency. The `--rank` flag on `micov binning` is a **no-op**; the variance ranking is written unconditionally.

## Known traps

- `.ks.tsv` outputs are comma-separated despite the extension. Frozen — released micov wrote them that way, so `_plot._write_delimited` is called with the default comma.
- **Never format micov's float outputs with `repr()` or an f-string.** `_plot._write_delimited` relies on `csv.writer` rendering via `str()`. The values are numpy scalars — `scipy.stats.ks_2samp` returns `np.float64`, `np.histogram` returns float64 — and under numpy 2 `repr(np.float64(0.3))` is the string `np.float64(0.3)`. That would corrupt every float in every `.ks.tsv` and `.tsv.gz`.
- Bonferroni correction is described in the paper but not implemented; `_plot.py` writes raw KS p-values.
- `ruff` runs with `fix = true`, so **`make lint` edits your files** rather than reporting. Check `git status` after linting.
- `_test_has_header` (`_io.py`) tests *substrings*, not membership: its column-name constants are plain strings, so a column named `genome` is accepted as a header. The plural names make them read as collections. It now has exactly one caller, `load_genome_lengths` — the BED3 path moved to DuckDB's CSV sniffer in M5, which also handles a `#`-prefixed header that this only accepts by accident.
- `MANIFEST.in` has `graft micov`, so **any** stray file under `micov/` is packaged into the sdist — including untracked ones, which then breaks `check-manifest`. Keep scratch work in `localdocs/` (gitignored, pruned from the sdist).
- `pyproject.toml` and `ci/conda_requirements.txt` **must declare the same dependencies**, duckdb floor included. They previously disagreed (`<1.3` in one, no ceiling in the other), and because CI's conda path installs with `pip install . --no-deps`, the conda and pypi paths silently tested different duckdb majors. Both now say `duckdb>=1.5.4`; change them together.
- Python 3.13 is unclaimed but no longer blocked. The blocker was `pyarrow<16.0.0`, which capped at 15.0.2 and has no cp313 wheels; pyarrow is gone. Nothing has been run on 3.13, so add it to the CI matrix before claiming it.

## In-flight work

An in-progress migration replaces micov's compute internals with [duckdb-miint](https://github.com/the-miint/duckdb-miint). **Done so far:** the DuckDB floor is `>=1.5.4`, **miint is loaded on every connection** via `_miint.connection()` and does the alignment ingest and interval merge, Qiita support has been dropped, and **`pyarrow`, `numba` and `polars` are all gone** — runtime dependencies are now `click`, `scipy`, `matplotlib` and `duckdb`. M6 moved `View`'s region operations onto `region_coverage`, `region_presence` and `compress_intervals`, and M7 moved the cumulative curves and Monte Carlo onto the `cumulative_coverage` aggregate — retiring micov's last hand-written interval merge and the O(n²) accumulation loop with it. **Still to do:** `ks_2samp`, and removing scipy with it. Plan and milestone gating live in **`MIGRATE-TO-MIINT.md`** (local, uncommitted). Blocking upstream capabilities: the-miint/duckdb-miint#214, #215, #216, #217, #218.

miint is a DuckDB **community extension**, not a Python package — it cannot be declared in `pyproject.toml`, and `pip index versions duckdb-miint` finds nothing. It is required at runtime instead, with no fallback: two compute paths that must agree numerically would put the frozen coverage and KS numbers at risk. `miint_version()` returns a **git short hash**, not a semantic version, so there is no orderable floor to pin; the guard that will replace it is a capability check, added with the first micov code that calls a miint primitive.

`_plot.py` and `_cov.py`'s curve functions now carry **`dict`s of numpy arrays** — the shape `DuckDBPyRelation.fetchnumpy()` returns — rather than polars frames. `_cov.mask_table(table, keep)` applies a boolean mask or index array to every column. Watch for numpy's `uint64 + int` promoting to `float64`; the rank columns cast deliberately to avoid it.
