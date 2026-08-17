# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ABSOLUTE HARD REQUIREMENTS
- NEVER use `rm` without permission

## Project Overview

**micov** (aggregate MIcrobiome COVerage) computes **breadth of coverage** — the fraction of a reference genome covered by at least one read — per sample per genome, and compares it across metadata-defined sample groups.

The distinguishing capability is the **cumulative coverage curve**: rank samples within a group by their own breadth, then accumulate covered intervals from lowest to highest. Individually weak signals stack into a detectable one (the authors' analogy is long-exposure astrophotography), which surfaces genomic regions present in one sample group and absent in another even when no single sample has good coverage. Supporting analyses are Monte Carlo null curves, pairwise KS tests between curves, and bin-and-rank region discovery.

Reference: Weng Y, Guccione C, McDonald D, et al. "Calculating fast differential genome coverages among metagenomic sources using micov." *Communications Biology* 2025. [PMC12635244](https://pmc.ncbi.nlm.nih.gov/articles/PMC12635244/)

micov is published, on pip and conda, and integrated with Qiita. Existing `.cov` and Parquet artifacts and the CLI surface are **contracts** — see Compatibility below.

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

- **CLI surface is frozen**, with one approved exception already taken: the `click<8.2` pin has been **removed** and `micov per-sample` is now the canonical name. pallets/click#2604 strips the `_group` suffix when deriving a command name, so the `per_sample_group` callback registers as `per-sample`; `per-sample-group` no longer resolves. Recorded in `ChangeLog.md`. The registered set is exactly: `binning`, `compress`, `consolidate`, `extract-sample-presence`, `nonqiita-to-parquet`, `per-sample`, `position-plot`, `qiita-coverage`, `qiita-to-parquet`.
- **Output file formats are frozen.** `.cov` / `.cov.gz` BED-like TSVs, `{base}.coverage.parquet` and `{base}.covered_positions.parquet` column names and types, and the Qiita `coverages.tgz` layout must stay readable by released micov versions and by Qiita.
- **Published numbers must reproduce.** Coverage values, KS statistics, and p-values are cited in the paper. A change that moves them is a regression, not an improvement.
- **Platform support was narrowed, deliberately.** Windows and Intel macOS were dropped because miint publishes no build for them; supported platforms are Linux (x86_64, aarch64) and macOS on Apple silicon. micov also now needs network access on its **first** run to fetch the extension — a deployment-surface change that matters for Qiita and for HPC compute nodes without egress. `MICOV_MIINT_EXTENSION_PATH` and a pre-seeded `~/.duckdb/extensions/` are the two ways around it.

## Coordinate and coverage conventions

- Intervals are **1-based half-open** `[start, stop)`. `start` is SAM `POS`; `stop = POS + reference span from CIGAR` (`M/=/X` advance alignment, `D/N` advance reference, so deletions and gaps count as covered).
- Breadth is `sum(stop - start)` over merged intervals — **no `+1`**.
- `_compress` **merges touching intervals** (`stop1 == start2` → one interval). This is intentional and covered by `micov/tests/test_cov.py`. The `compress` docstring's case 3 claims the opposite and is stale.
- `.cov` files are described as "BED-like" but carry 1-based coordinates, whereas real BED is 0-based. Breadth math is unaffected; joins against genuinely 0-based sources are not.
- Primary *and* secondary alignments are retained deliberately, so CNVs, horizontally transferred elements, and repeats are represented rather than silently dropped.

## Architecture

```
SAM/BAM (headerless) or BED3
   │  micov compress               merge overlapping intervals per genome
   ▼
.cov / .cov.gz                     BED3; filename stem is the sample_id
   │  micov nonqiita-to-parquet (DuckDB)  |  micov qiita-to-parquet (Qiita .tgz)
   ▼
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
| `_view.py` | `View` — DuckDB session with three filter modes: none, genome-level (`constrain_features`), sub-genome region (`constrain_positions`). Only the third does real work: clips intervals to region bounds, re-compresses per sample, recomputes breadth against *region* length |
| `_cov.py` | two interval-merge implementations: `compress` (numba + polars, still used by `_io.py`, `cli.py`, `_per_sample.py`) and `merge_intervals` (numpy, used by the curve path). Plus breadth, rank ordering, cumulative accumulation |
| `_io.py` | parsers/writers; `compress_from_stream` flushes every 100 MB so memory is bounded on arbitrarily large SAM streams |
| `_plot.py` | matplotlib curves and position plots, plus the KS tests (largest module, no unit tests) |
| `_quant.py` | binning |
| `_convert.py` | CIGAR → reference span, LRU-cached (`150M` dominates real data) |
| `_constants.py` | column names and dtypes |
| `_miint.py` | `connection()` — the only place micov opens a DuckDB connection, and the only place the miint extension is installed and loaded |

`micov/_rank.py` is **dead code** — nothing imports it, and it pulls in an undeclared pandas dependency. The `--rank` flag on `micov binning` is a **no-op**; the variance ranking is written unconditionally.

## Known traps

- `_cov.compress_per_sample(df)` and `_per_sample.compress_per_sample(coverage, lengths)` are different functions with the same name and different signatures. Each module resolves its own; do not cross-import.
- `.ks.tsv` outputs are comma-separated despite the extension (`_plot.py` calls `write_csv` without `separator`).
- Bonferroni correction is described in the paper but not implemented; `_plot.py` writes raw KS p-values.
- `ruff` runs with `fix = true`, so **`make lint` edits your files** rather than reporting. Check `git status` after linting.
- `_test_has_header_taxonomy` (`_io.py`) tests *substrings*, not membership: `genome_id_columns` is the plain string `"genome_id"`, so a column named `genome` is accepted as a header. The plural names make it read as a collection. No test covers this function.
- `MANIFEST.in` has `graft micov`, so **any** stray file under `micov/` is packaged into the sdist — including untracked ones, which then breaks `check-manifest`. Keep scratch work in `localdocs/` (gitignored, pruned from the sdist).
- `pyproject.toml` and `ci/conda_requirements.txt` **must declare the same duckdb floor**. They previously disagreed (`<1.3` in one, no ceiling in the other), and because CI's conda path installs with `pip install . --no-deps`, the conda and pypi paths silently tested different duckdb majors. Both now say `duckdb>=1.5.4`; change them together.
- Python 3.13 is unclaimed but no longer blocked. The blocker was `pyarrow<16.0.0`, which capped at 15.0.2 and has no cp313 wheels; pyarrow is gone. Nothing has been run on 3.13, so add it to the CI matrix before claiming it.

## In-flight work

An in-progress migration replaces micov's compute internals with [duckdb-miint](https://github.com/the-miint/duckdb-miint). **Done so far:** every polars↔DuckDB crossing has been deleted by moving its computation into plain DuckDB SQL — `_view.py`, `_plot.py`, `_quant.py` and `cli.py` no longer cross that boundary — `pyarrow` has been dropped, the DuckDB floor is now `>=1.5.4`, and **miint is loaded on every connection** via `_miint.connection()`. **Still to do:** replace the hand-written SQL with miint's primitives, then remove polars, numba, and scipy. Plan and milestone gating live in **`MIGRATE-TO-MIINT.md`** (local, uncommitted). Blocking upstream capabilities: the-miint/duckdb-miint#214, #215, #216, #217, #218.

miint is a DuckDB **community extension**, not a Python package — it cannot be declared in `pyproject.toml`, and `pip index versions duckdb-miint` finds nothing. It is required at runtime instead, with no fallback: two compute paths that must agree numerically would put the frozen coverage and KS numbers at risk. `miint_version()` returns a **git short hash**, not a semantic version, so there is no orderable floor to pin; the guard that will replace it is a capability check, added with the first micov code that calls a miint primitive.

`_plot.py` and `_cov.py`'s curve functions now carry **`dict`s of numpy arrays** — the shape `DuckDBPyRelation.fetchnumpy()` returns — rather than polars frames. `_cov.mask_table(table, keep)` applies a boolean mask or index array to every column. Watch for numpy's `uint64 + int` promoting to `float64`; the rank columns cast deliberately to avoid it.
