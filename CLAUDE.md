# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ABSOLUTE HARD REQUIREMENTS
- NEVER use `rm` without permission

## Project Overview

**micov** computes **breadth of coverage**, the fraction of a reference genome covered by at least one read, for each sample and genome, and compares it across sample groups defined by metadata. Its core feature is the **cumulative coverage curve**: a group's samples are ranked by their own breadth, then their covered intervals are merged from lowest to highest, so weak individual signals stack into a detectable one. Supporting analyses are Monte Carlo null curves, pairwise KS tests, bin-and-rank region discovery, and per-region presence.

Paper: Weng Y, Guccione C, McDonald D, et al., *Communications Biology* 2025, [PMC12635244](https://pmc.ncbi.nlm.nih.gov/articles/PMC12635244/). micov is published on pip and conda, and its outputs are cited, so its outputs and CLI are contracts (see below).

The compute runs on DuckDB plus the **miint** DuckDB extension. Runtime dependencies are `click`, `matplotlib` and `duckdb`; polars, scipy, numba and pyarrow are gone.

## Internals documentation

**`docs/internals/` describes how micov works on this branch.** Read the relevant document before changing code. Start at `docs/internals/README.md`, and when the code changes, update the document in the same change.

| Document | Covers |
|---|---|
| `architecture.md` | Data flow, module ownership, dependencies, the DuckDB catalog |
| `data-formats.md` | Coordinates, every input and output format, the frozen columns |
| `commands.md` | Each CLI command, from click down to SQL |
| `view.md` | `View`'s three filter modes, regions, presence, the header rule |
| `curves-and-ks.md` | Ranking, the tie-break, accumulation, Monte Carlo, position plots, KS and Bonferroni |
| `depth-plot.md` | `depth-plot`'s layers, windowed per-base depth, breadth, bins, per-ORF statistics, the plots, and the per-genome run |
| `miint.md` | Loading the extension, the functions called, the stale-cache trap |
| `testing.md` | Tiers, goldens, comparators, tolerances |
| `traps.md` | **Read before any non-trivial change.** Each entry is a mistake already made once |

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
pip install -r ci/requirements.lint.txt   # pinned ruff + check-manifest
```

```bash
make test                          # pytest micov + bash cli_test.sh (fast tier)
MICOV_GOLDEN_FULL=1 pytest micov   # adds the full example/ golden corpus
make lint                          # ruff check micov + check-manifest; reports only
make lint-fix                      # the only target that edits files
```

On hosts with no configured channels, `conda create` needs `-c conda-forge`. Without the pinned lint tools, `make lint` runs whatever `ruff` is on `PATH`. Ruff uses line length 88 and the numpy docstring convention; `micov/tests/*` is exempt from the `D` and `PT` rules.

If a test produces an **incorrect expected value**: DO NOT change the expected value without permission. The same goes for goldens: never refresh one to make a test pass.

## Compatibility contract

Unless a task explicitly overrides this:

- **The CLI surface is frozen.** The commands are `binning`, `compress`, `cov-to-parquet`, `depth-plot`, `extract-sample-presence`, `per-sample` and `position-plot`, plus the hidden alias `nonqiita-to-parquet`. `test_equivalence.TestCliSurface` pins the set, and `depth-plot`'s options.
- **Output formats are frozen**: the `{base}.coverage.parquet` / `{base}.covered_positions.parquet` column names, order and types; `.ks.csv`; the position `.tsv.gz`; the binning and presence TSVs; `depth-plot`'s per-ORF Parquet (`_io.ORF_STATISTICS_COLUMNS`). Released micov must still read them. `.cov` is read-only input now, and `cov-to-parquet` is its reader.
- **Published numbers must reproduce.** A change that moves coverage values, KS statistics or p-values is a regression, not an improvement.
- **Every approved break is recorded in `ChangeLog.md`.** These include the `per-sample` rename, SAM/BAM-only `compress`, Qiita removal, `.ks.csv` with its Bonferroni column, the header requirement and the `sample_id` tie-break. Record any new one there, with the maintainer's approval.
- **Supported platforms** are Linux (x86_64, aarch64) and macOS on Apple silicon; miint has no other builds. The first run needs network access to fetch the extension (`docs/internals/miint.md` covers offline use).

## Traps you will hit first

`docs/internals/traps.md` has the full list, with the reasons. In short:

- Every value interpolated into a SQL string literal goes through `_utils.sql_string`.
- `_miint.connection()` is the only place a DuckDB connection is opened, and every new miint function name goes in `REQUIRED_MIINT_FUNCTIONS`.
- Never format float outputs with `repr()` or f-strings; `_plot._write_delimited` relies on `str()`.
- Plotting functions receive one genome's rows, never the whole positions table. `example/` has 2 genomes, so test per-genome cost with thousands.
- Breadth ties are broken by `sample_id`; nothing on the curve path may depend on row order.
- `pyproject.toml` and `ci/conda_requirements.txt` declare the same dependencies, bounds included.
- `MANIFEST.in` grafts `micov/`, so scratch files go in `localdocs/` (gitignored), never under `micov/`.

## History

The move from polars/scipy/numba to DuckDB and miint happened in milestones M0–M11b on the `migrate-to-miint` branch. `MIGRATE-TO-MIINT.md` is the maintainer's local running record of it: it is not committed, and may not exist in your checkout.
