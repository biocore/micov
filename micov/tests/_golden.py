"""Golden-artifact comparators for the micov equivalence suite.

Every comparator in this module exists because some micov output is *not*
byte-stable. Comparing bytes would give flaky tests; comparing too loosely
would let a real regression through. This docstring records which is which,
so the next person does not have to rediscover it.

Why this matters for the miint migration: the whole compute layer is being
replaced, and the only thing standing between that and a silent change to
published numbers is this suite. A comparator that is too loose is worse
than no comparator, because it reads as coverage.

Provenance of the classification
--------------------------------

Verified empirically on 2026-08-06 against the committed ``example/`` corpus,
on macOS-15.7.7-arm64 with python 3.12.13, micov 2025.3.dev39+g7a6629dd4,
click 8.4.2, duckdb 1.2.2, polars-u64-idx 1.22.0, numba 0.66.0, scipy 1.17.1,
pyarrow 13.0.0, matplotlib 3.11.1.

Each source below was confirmed by *observing it vary across runs*, not by
reading the source alone. That distinction is load-bearing -- see the warning
at the end.

Sources of nondeterminism
-------------------------

1. Unseeded ``pl.col(...).shuffle()`` -- ``_plot.py:218``
   Randomizes the Monte Carlo sample draw, so the Monte Carlo curve and any
   ``.ks.tsv`` row whose label begins ``Monte Carlo `` change every run.
   The non-Monte rows in the same file do not.
   => Partition ``.ks.tsv`` rows on that label prefix. Exact-compare the
      deterministic rows; assert only presence and range on the rest.

2. ``gzip.open`` stores the current mtime -- ``_plot.py:672``
   ``.tsv.gz`` bytes differ run to run even when the data is identical.
   => Decompress first, then compare content.

3. ``ti.mtime = int(time.time())`` -- the Qiita ``coverages.tgz`` writer
   Archive bytes differed run to run. **Retired in M4**: micov no longer
   writes tar archives, and ``assert_tgz_equal`` went with the format.

4. matplotlib version, backend, and font availability
   PNG bytes are not reproducible across environments, and sizes drift
   noticeably between matplotlib releases.
   => Existence, non-zero size, and PNG magic bytes only. PNG content is
      deliberately unguarded; the plotted values are guarded instead via the
      ``.tsv.gz`` position data and the ``.ks.tsv`` statistics.

5. Ties in an ``ORDER BY`` are broken arbitrarily
   Row order within a tie block of ``stats_by_variance_of_sample_hits.tsv``
   varies. Originally attributed to polars' unstable sort; it survived the
   move to DuckDB, and **the file differs run to run on byte-identical
   input** -- measured in M4, where three rows share one ``sample_hits_std``.
   => Sort-normalize with explicit keys before comparing.

6. DuckDB parallel write
   Parquet row order varies between runs.
   => Compare as a row set (bidirectional ``EXCEPT``), plus an ordered
      schema check, since column order is itself part of the contract.

7. ``set(length_map)`` iteration order -- ``cli.py:516``
   Genome block order in the binning outputs follows set iteration, which
   varies with ``PYTHONHASHSEED``.
   => Covered by the same explicit-key normalization as #5.

8. Genome block order in ``.cov`` is not stable
   ``compress()`` accumulated one frame per genome in whatever order polars
   yielded groups. That function was deleted in M4 and micov no longer writes
   ``.cov`` at all, but ``example/coverages/*.cov.gz`` were frozen under it
   and are still read as fixtures, so the comparator is still needed.
   => Order-insensitive ``.cov`` comparison.
   Corroboration that this was known but never written down: ``cli_test.sh``
   already pipes both sides through ``sort``.

9. Summation order inside a standard deviation -- ``sample_hits_std`` in
   ``stats_by_variance_of_sample_hits.tsv``
   Added 2026-08-14 while moving binning into SQL. polars' ``std()`` cannot be
   reproduced bit-for-bit outside polars: over 18 hand-checked cases it matches
   neither ``numpy.std(ddof=1)``, DuckDB's ``stddev_samp``, sum-of-squares,
   mean-corrected sum-of-squares, two-pass, nor Welford -- and on ``[2, 3, 3]``
   *none* of the six agree with it. It looks like a chunked SIMD reduction,
   whose summation order no SQL expression fixes.
   Measured against the committed golden: 103 of 1999 rows differ, by at most
   **3 ULP** (5.8e-16 relative), in *both* directions -- the same golden value
   drifts up in one row and down in another, which is the signature of float
   noise rather than a systematic change. Every ranking swap it causes is
   between bins whose golden values differ by <= 8.9e-16, in a column whose
   largest value is 6.8.
   => Compare that one column with a tight relative tolerance
      (``TSV_FLOAT_REL_TOL``, ~20x looser than the observed drift and far
      tighter than any real defect) and every other column exactly.

Two traps: outputs that matched by luck
---------------------------------------

During the audit, two artifacts compared byte-identical to their committed
goldens on the first run *despite being genuinely unstable*:

- ``stats_bins.tsv`` matched exactly, but its genome block order flips with
  ``PYTHONHASHSEED`` (seeds 1 and 12345 put ``G000154205`` first; the golden
  has ``G000436435`` first). Source #7.
- The non-percentile Monte Carlo ``.ks.tsv`` matched exactly, which flatly
  contradicts source #1. Three further runs showed the Monte Carlo rows do
  vary. The KS statistic over ~20 samples takes few distinct values, so
  collisions are common and a single match proves nothing.

The lesson, and the reason ``test_golden_selftest.py`` exists: a comparator
calibrated by observing one run of the code it is meant to guard will be
wrong. Each comparator must be shown to fail on a perturbed input, not just
to pass on a real one.

Stable outputs
--------------

For contrast -- these were verified stable and are compared exactly:

- ``.ks.tsv`` rows that are not Monte Carlo rows, including the statistic
  and p-value to full double precision. These are the published numbers.
- ``.tsv.gz`` position-plot content, once decompressed.
- ``.cov`` content as a multiset of intervals.
- Parquet row sets and their ordered schemas.
- Output filename sets. The
  ``{output}.{target_name}.{target}.{variable}.{tag}.png`` scheme is
  contractual, and ``--monte`` alters the tag, so the fileset is a real
  assertion rather than a formality.

Note that ``.ks.tsv`` files are comma-separated despite the extension
(``_plot.py`` calls ``write_csv`` without ``separator``). That is a frozen
output format, not a bug to fix here.

Conventions
-----------

Every comparator takes ``(observed, expected)`` in that order and raises
``AssertionError`` on mismatch, so they compose with ``unittest.TestCase``
without needing a ``self``. Messages name both paths and the first concrete
difference, because a failure here will most often be read by someone
bisecting a migration milestone.
"""

import csv
import gzip
import math
from collections import Counter
from pathlib import Path

import duckdb

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Rows whose label begins with this are Monte Carlo comparisons, whose
#: values are unreproducible by construction (source #1).
MONTE_LABEL_PREFIX = "Monte Carlo "

KS_COLUMNS = ("label_A", "label_B", "ks-statistic", "ks-pvalue")

#: Relative tolerance for TSV columns whose summation order is not
#: reproducible (source #9). The measured drift is 5.8e-16; this is ~20x
#: looser, and still orders of magnitude tighter than any real defect.
TSV_FLOAT_REL_TOL = 1e-14


def _require_file(path, role):
    """Resolve `path` to an existing regular file or raise."""
    path = Path(path)
    if not path.exists():
        raise AssertionError(f"{role} does not exist: {path}")
    if not path.is_file():
        raise AssertionError(f"{role} is not a regular file: {path}")
    return path


def _first_difference(observed_lines, expected_lines):
    """Return (index, observed, expected) of the first differing line.

    Indexes explicitly rather than using ``zip``: the two may legitimately
    have different lengths (the caller reports counts separately), and
    ``zip(strict=...)`` is 3.10+ while this project supports 3.9.
    """
    for i in range(min(len(observed_lines), len(expected_lines))):
        if observed_lines[i] != expected_lines[i]:
            return i, observed_lines[i], expected_lines[i]
    return None


def _content_message(observed, expected, observed_text, expected_text):
    got = observed_text.splitlines()
    want = expected_text.splitlines()
    lines = [
        f"content differs\n  observed: {observed}\n  expected: {expected}",
        f"  line counts: {len(got)} observed, {len(want)} expected",
    ]
    diff = _first_difference(got, want)
    if diff is not None:
        i, g, w = diff
        lines.append(
            f"  first difference at line {i + 1}:\n"
            f"    observed: {g!r}\n"
            f"    expected: {w!r}"
        )
    return "\n".join(lines)


def _assert_text_content_equal(observed, expected, observed_text, expected_text):
    if observed_text != expected_text:
        raise AssertionError(
            _content_message(observed, expected, observed_text, expected_text)
        )


def assert_gzip_text_equal(observed, expected):
    """Assert two gzip files hold identical text.

    Bytes are expected to differ: ``gzip.open`` records the current mtime
    (source #2). Only the decompressed content is compared.
    """
    observed = _require_file(observed, "observed")
    expected = _require_file(expected, "expected golden")
    with gzip.open(observed, "rt") as fp:
        observed_text = fp.read()
    with gzip.open(expected, "rt") as fp:
        expected_text = fp.read()
    _assert_text_content_equal(observed, expected, observed_text, expected_text)


def _cov_mismatch(got, want):
    """Compare two lists of ``.cov`` lines.

    Returns None if they agree, else a message. Header is compared exactly;
    the body is compared as a **multiset**, because ``compress()`` emits
    genome blocks in nondeterministic order (source #8). A multiset rather
    than a set, so a duplicated interval is still caught.

    Used by :func:`assert_cov_equal`. It also backed ``assert_tgz_equal``,
    whose ``.cov`` archive members inherited the same instability, until M4
    removed micov's Qiita support and that helper with it.
    """
    if not got or not want:
        return "empty .cov content, expected a header"

    if got[0] != want[0]:
        return (
            f"header differs\n"
            f"    observed: {got[0]!r}\n    expected: {want[0]!r}"
        )

    got_body, want_body = Counter(got[1:]), Counter(want[1:])
    if got_body == want_body:
        return None

    only_observed = got_body - want_body
    only_expected = want_body - got_body
    return (
        f"interval sets differ (order-insensitive)\n"
        f"  {len(got) - 1} observed rows, {len(want) - 1} expected rows\n"
        f"  {sum(only_observed.values())} rows only in observed, "
        f"{sum(only_expected.values())} only in expected\n"
        f"  sample observed-only: {sorted(only_observed)[:3]}\n"
        f"  sample expected-only: {sorted(only_expected)[:3]}"
    )


def assert_cov_equal(observed, expected):
    """Assert two ``.cov`` files describe the same intervals.

    Order-insensitive in the body -- see :func:`_cov_mismatch`. Accepts
    plain or gzipped input, since micov writes both.
    """
    observed = _require_file(observed, "observed")
    expected = _require_file(expected, "expected golden")

    def read_lines(path):
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as fp:
                return fp.read().splitlines()
        return path.read_text().splitlines()

    problem = _cov_mismatch(read_lines(observed), read_lines(expected))
    if problem is not None:
        raise AssertionError(
            f"{problem}\n  observed: {observed}\n  expected: {expected}"
        )


def assert_parquet_equal(observed, expected):
    """Assert two parquet files have the same ordered schema and row multiset.

    Column order **is** part of the contract: released micov versions read
    these files, and the two Parquet producers micov used to have differed
    only in where ``coverage.parquet`` put ``sample_id`` -- a difference
    invisible to a row-only comparison. So the schema is compared as an
    ordered sequence of ``(name, type)``.

    Row order is not part of the contract (source #6), so rows are compared
    with a bidirectional ``EXCEPT ALL``, which is multiset-exact.
    """
    observed = _require_file(observed, "observed")
    expected = _require_file(expected, "expected golden")

    con = duckdb.connect()
    try:

        def schema_of(path):
            rows = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{path}')"
            ).fetchall()
            return [(r[0], r[1]) for r in rows]

        def count_of(path):
            return con.execute(
                f"SELECT count(*) FROM read_parquet('{path}')"
            ).fetchone()[0]

        got_schema, want_schema = schema_of(observed), schema_of(expected)
        if got_schema != want_schema:
            raise AssertionError(
                f"parquet schema differs (name, type, and order all matter)\n"
                f"  observed: {observed}\n  expected: {expected}\n"
                f"    observed: {got_schema}\n    expected: {want_schema}"
            )

        # Diagnostic only: the EXCEPT ALL below already catches any count
        # difference, since it is multiset-exact. This runs first purely
        # because "1000 rows vs 999" reads better than a multiset diff.
        # Mutation-tested: removing it loses no coverage.
        got_count, want_count = count_of(observed), count_of(expected)
        if got_count != want_count:
            raise AssertionError(
                f"parquet row count differs\n"
                f"  observed: {observed} ({got_count} rows)\n"
                f"  expected: {expected} ({want_count} rows)"
            )

        def excess(left, right):
            return con.execute(
                f"SELECT count(*) FROM ("
                f"SELECT * FROM read_parquet('{left}') EXCEPT ALL "
                f"SELECT * FROM read_parquet('{right}'))"
            ).fetchone()[0]

        only_observed = excess(observed, expected)
        only_expected = excess(expected, observed)
        if only_observed or only_expected:
            raise AssertionError(
                f"parquet rows differ (order-insensitive)\n"
                f"  observed: {observed}\n  expected: {expected}\n"
                f"  {only_observed} rows only in observed, "
                f"{only_expected} only in expected (of {got_count})"
            )
    finally:
        con.close()


def _read_tsv(path):
    """Return (header, body_lines) for a tab-separated file with a header."""
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise AssertionError(f"empty TSV, expected a header: {path}")
    return lines[0], lines[1:]


def assert_tsv_equal_unordered(observed, expected, sort_keys, float_columns=()):
    """Assert two TSVs match after normalizing row order.

    Binning outputs order tie blocks arbitrarily (source #5) and genome
    blocks by set iteration (source #7), so rows are sorted by
    `sort_keys` -- then by their remaining fields, to make the order total --
    before comparing. Values are then compared exactly.

    `float_columns` names columns to compare with `TSV_FLOAT_REL_TOL` instead,
    for values whose summation order is not reproducible (source #9). Those
    columns are excluded from the sort tie-break too, so a drifted value can
    never pair two different rows against each other. Every column not named
    stays exact.

    An unknown name in `sort_keys` or `float_columns` raises rather than
    silently degrading to an unsorted or unchecked comparison, which would
    reintroduce the flakiness this exists to prevent.
    """
    observed = _require_file(observed, "observed")
    expected = _require_file(expected, "expected golden")

    got_header, got_body = _read_tsv(observed)
    want_header, want_body = _read_tsv(expected)

    if got_header != want_header:
        raise AssertionError(
            f"TSV header differs\n  observed: {observed}\n  expected: {expected}\n"
            f"    observed: {got_header!r}\n    expected: {want_header!r}"
        )

    columns = got_header.split("\t")
    missing = [
        key for key in (*sort_keys, *float_columns) if key not in columns
    ]
    if missing:
        raise AssertionError(
            f"column names not present in header: {missing}\n  columns: {columns}"
        )
    indices = [columns.index(key) for key in sort_keys]
    approximate = {columns.index(key) for key in float_columns}
    exact = [i for i in range(len(columns)) if i not in approximate]

    def canonical(line):
        fields = line.split("\t")
        return (
            tuple(fields[i] for i in indices),
            tuple(fields[i] for i in exact),
        )

    if len(got_body) != len(want_body):
        raise AssertionError(
            f"TSV row count differs\n"
            f"  observed: {observed} ({len(got_body)} rows)\n"
            f"  expected: {expected} ({len(want_body)} rows)"
        )

    got_sorted = sorted(got_body, key=canonical)
    want_sorted = sorted(want_body, key=canonical)

    for row, (got_line, want_line) in enumerate(
        zip(got_sorted, want_sorted, strict=True)
    ):
        if got_line == want_line:
            continue

        got_fields = got_line.split("\t")
        want_fields = want_line.split("\t")
        reason = None
        if len(got_fields) != len(want_fields):
            reason = "field count differs"
        else:
            for i, (got_field, want_field) in enumerate(
                zip(got_fields, want_fields, strict=True)
            ):
                if got_field == want_field:
                    continue
                if i not in approximate:
                    reason = f"{columns[i]!r} differs"
                    break
                try:
                    close = math.isclose(
                        float(got_field), float(want_field),
                        rel_tol=TSV_FLOAT_REL_TOL, abs_tol=TSV_FLOAT_REL_TOL,
                    )
                except ValueError:
                    reason = f"{columns[i]!r} is not numeric"
                    break
                if not close:
                    reason = (
                        f"{columns[i]!r} differs by more than "
                        f"{TSV_FLOAT_REL_TOL:g} relative"
                    )
                    break

        if reason is not None:
            raise AssertionError(
                f"TSV rows differ after sorting by {list(sort_keys)}: {reason}\n"
                f"  observed: {observed}\n  expected: {expected}\n"
                f"  first difference at sorted row {row + 1}:\n"
                f"    observed: {got_line!r}\n    expected: {want_line!r}"
            )


def _is_monte(row):
    return any(field.startswith(MONTE_LABEL_PREFIX) for field in row[:2])


def _parse_ks(path):
    """Parse a ``.ks.tsv``, which is comma-separated despite the extension."""
    with open(path, newline="") as fp:
        rows = list(csv.reader(fp))
    rows = [r for r in rows if r]
    if not rows:
        raise AssertionError(f"empty .ks.tsv, expected a header: {path}")
    header = tuple(rows[0])
    if header != KS_COLUMNS:
        raise AssertionError(
            f"unexpected .ks.tsv header in {path}\n"
            f"    observed: {header}\n    expected: {KS_COLUMNS}"
        )
    body = rows[1:]
    for row in body:
        if len(row) != len(KS_COLUMNS):
            raise AssertionError(
                f"malformed .ks.tsv row in {path}: expected "
                f"{len(KS_COLUMNS)} fields, got {len(row)}: {row}"
            )
    return header, body


def assert_ks_equal(observed, expected):
    """Assert two ``.ks.tsv`` files agree, tolerating Monte Carlo variance.

    Rows are partitioned on the ``Monte Carlo `` label prefix:

    - **Deterministic rows** are compared exactly, statistic and p-value
      included, to full double precision. These are the published numbers.
    - **Monte Carlo rows** are unseeded (source #1), so only their label
      pairs must match -- catching a dropped or renamed comparison -- and
      their statistic and p-value must lie in ``[0, 1]``.
    """
    observed = _require_file(observed, "observed")
    expected = _require_file(expected, "expected golden")

    _, got_rows = _parse_ks(observed)
    _, want_rows = _parse_ks(expected)

    got_fixed = sorted(",".join(r) for r in got_rows if not _is_monte(r))
    want_fixed = sorted(",".join(r) for r in want_rows if not _is_monte(r))
    if got_fixed != want_fixed:
        raise AssertionError(
            f"deterministic KS rows differ -- these are published numbers\n"
            f"  observed: {observed}\n  expected: {expected}\n"
            f"  only in observed: {sorted(set(got_fixed) - set(want_fixed))}\n"
            f"  only in expected: {sorted(set(want_fixed) - set(got_fixed))}"
        )

    got_labels = sorted(tuple(r[:2]) for r in got_rows if _is_monte(r))
    want_labels = sorted(tuple(r[:2]) for r in want_rows if _is_monte(r))
    if got_labels != want_labels:
        raise AssertionError(
            f"Monte Carlo KS comparisons differ\n"
            f"  observed: {observed}\n  expected: {expected}\n"
            f"    observed labels: {got_labels}\n"
            f"    expected labels: {want_labels}"
        )

    for row in (r for r in got_rows if _is_monte(r)):
        # index rather than zip: _parse_ks already validated field counts,
        # and zip(strict=...) is 3.10+ while this project supports 3.9
        for offset, name in enumerate(KS_COLUMNS[2:], start=2):
            raw = row[offset]
            try:
                value = float(raw)
            except ValueError:
                raise AssertionError(
                    f"non-numeric {name} in {observed}: {raw!r} (row {row})"
                ) from None
            if not 0.0 <= value <= 1.0:
                raise AssertionError(
                    f"Monte Carlo {name} outside [0, 1] in {observed}: "
                    f"{value} (row {row})"
                )


def assert_png_plausible(observed):
    """Assert a PNG exists and is structurally a PNG.

    PNG bytes are not reproducible across matplotlib versions, backends, or
    font sets (source #4), so content is deliberately unguarded -- the
    plotted values are covered by the ``.tsv.gz`` and ``.ks.tsv`` goldens
    instead. This checks only that a plot was actually written: the magic
    bytes plus a non-empty payload, so a truncated or empty write fails.
    """
    observed = _require_file(observed, "observed PNG")
    data = observed.read_bytes()
    if not data:
        raise AssertionError(f"PNG is empty: {observed}")
    if not data.startswith(PNG_MAGIC):
        raise AssertionError(
            f"not a PNG (bad magic bytes): {observed}\n"
            f"  first bytes: {data[:8]!r}"
        )
    if len(data) <= len(PNG_MAGIC):
        raise AssertionError(
            f"PNG is truncated to its magic bytes: {observed} ({len(data)} bytes)"
        )


def assert_file_set(directory, expected_names):
    """Assert `directory` contains exactly `expected_names` (files only).

    The output filename scheme
    ``{output}.{target_name}.{target}.{variable}.{tag}.png`` is contractual,
    and ``--monte`` changes the tag, so this is a real assertion: it catches
    a plot silently not being produced as well as an unexpected extra file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise AssertionError(f"not a directory: {directory}")

    found = {p.name for p in directory.iterdir() if p.is_file()}
    expected_names = set(expected_names)
    if found != expected_names:
        raise AssertionError(
            f"output file set differs in {directory}\n"
            f"  unexpected: {sorted(found - expected_names)}\n"
            f"  missing:    {sorted(expected_names - found)}"
        )
