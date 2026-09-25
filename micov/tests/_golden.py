"""Golden-artifact comparators for the micov equivalence suite.

micov's outputs are frozen contracts -- the coverage values and KS statistics
are cited in the paper -- and `test_equivalence.py` checks them against
committed goldens. Not every output is byte-stable, so each comparator here
encodes exactly how much variation one artifact is allowed: too strict and
the suite flakes, too loose and a real regression reads as coverage.
`test_golden_selftest.py` proves each one both passes what it must tolerate
and fails what it must catch.

Sources of nondeterminism
-------------------------

Each was confirmed by *observing it vary across runs*, not by reading code
(the classification dates from 2026-08-06; its environment is in this file's
git history). Numbers are stable because other modules cite them; #3 and #8
are retired.

1. Unseeded ``rng.permutation`` in ``_plot.add_monte``. The Monte Carlo curve,
   and every ``.ks.csv`` row whose label begins ``Monte Carlo ``, change
   every run; the other rows do not.
   => Partition on that prefix; compare deterministic rows, and only check
      presence, range and an empty Bonferroni field on the Monte Carlo ones.

2. ``gzip.open`` stores the current mtime, so ``.tsv.gz`` bytes differ.
   => Decompress, then compare content.

4. matplotlib version, backend and fonts make PNG bytes non-reproducible.
   => Existence, non-zero size and PNG magic only. The plotted values are
      guarded through the ``.tsv.gz`` and ``.ks.csv`` data instead.

5. Ties in an ``ORDER BY`` are broken arbitrarily -- row order within a tie
   block of ``stats_by_variance_of_sample_hits.tsv`` differs run to run on
   identical input.
   => Sort-normalize on explicit keys.

6. DuckDB writes Parquet in parallel, so row order varies.
   => Compare as a row set (bidirectional ``EXCEPT``), plus the ordered
      schema, since column order is itself part of the contract.

7. ``set(length_map)`` iteration order puts binning genome blocks in an
   order that follows ``PYTHONHASHSEED``.
   => The same explicit-key normalization as #5.

9. ``sample_hits_std`` cannot be reproduced bit for bit outside polars, which
   produced the golden: its ``std()`` matches no summation order SQL can
   express. 103 of 1999 rows differ by at most 3 ULP (5.8e-16 relative), in
   both directions -- float noise, not drift.
   => That column within ``TSV_FLOAT_REL_TOL``; every other column exactly.

10. KS p-values come from miint, the goldens from scipy 1.17.1. miint derives
    the exact p-value independently, so they disagree by 1-4 ULP (worst
    3.8e-16 on the goldens). The statistic is an exact lattice value in both
    and matches bit for bit.
    => ``ks-pvalue`` and ``ks-pvalue-bonferroni`` (``p * m``) within
       ``TSV_FLOAT_REL_TOL``; labels and ``ks-statistic`` exactly.

Outputs that matched by luck
----------------------------

Two artifacts compared byte-identical on the first run despite being
unstable: ``stats_bins.tsv`` (its genome order flips with
``PYTHONHASHSEED``, #7) and a Monte Carlo ``.ks.csv`` (a KS statistic over
~20 samples takes few values, so collisions are common, #1). A comparator
calibrated on one observed run will be wrong; that is why every one here has
negative controls.

Compared exactly
----------------

- ``.ks.csv`` deterministic rows: labels and statistic.
- ``.tsv.gz`` position-plot content, once decompressed.
- Parquet row sets and ordered schemas.
- Output filename sets. The
  ``{output}.{target_name}.{target}.{variable}.{tag}.png`` scheme is
  contractual, and ``--monte`` changes the tag.

Conventions
-----------

Every comparator takes ``(observed, expected)`` in that order and raises
``AssertionError`` naming both paths and the first concrete difference, so
they compose with ``unittest.TestCase`` without needing a ``self``.
"""

import csv
import gzip
import math
from pathlib import Path

import duckdb

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Rows whose label begins with this are Monte Carlo comparisons, whose
#: values are unreproducible by construction (source #1).
MONTE_LABEL_PREFIX = "Monte Carlo "

KS_COLUMNS = (
    "label_A", "label_B", "ks-statistic", "ks-pvalue", "ks-pvalue-bonferroni",
)

#: Relative tolerance for floats that are not reproducible bit for bit:
#: ``sample_hits_std`` (source #9, measured drift 5.8e-16, so ~20x margin) and
#: the deterministic ``ks-pvalue`` (source #10, 3.8e-16 on the goldens and
#: at most ~1.7e-15 over randomized and heavily tied trials, so ~6x margin at
#: worst). Either way, orders of magnitude tighter than any real defect.
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
    have different lengths (the caller reports counts separately).
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
    """Parse a ``.ks.csv``."""
    with open(path, newline="") as fp:
        rows = list(csv.reader(fp))
    rows = [r for r in rows if r]
    if not rows:
        raise AssertionError(f"empty .ks.csv, expected a header: {path}")
    header = tuple(rows[0])
    if header != KS_COLUMNS:
        raise AssertionError(
            f"unexpected .ks.csv header in {path}\n"
            f"    observed: {header}\n    expected: {KS_COLUMNS}"
        )
    body = rows[1:]
    for row in body:
        if len(row) != len(KS_COLUMNS):
            raise AssertionError(
                f"malformed .ks.csv row in {path}: expected "
                f"{len(KS_COLUMNS)} fields, got {len(row)}: {row}"
            )
    return header, body


def assert_ks_equal(observed, expected):
    """Assert two ``.ks.csv`` files agree, tolerating Monte Carlo variance.

    Rows are partitioned on the ``Monte Carlo `` label prefix:

    - **Deterministic rows** are the published numbers. Label pair and
      statistic are compared exactly; the p-value within
      ``TSV_FLOAT_REL_TOL``, because miint derives it independently of the
      scipy that produced the goldens and agrees only to a few ULP
      (source #10). The statistic is an exact lattice value in both, so it
      gets no tolerance at all.
      ``ks-pvalue-bonferroni`` is ``p * m`` capped at 1, so it inherits the
      p-value's tolerance.
    - **Monte Carlo rows** are unseeded (source #1), so only their label
      pairs must match -- catching a dropped or renamed comparison -- and
      their statistic and p-value must lie in ``[0, 1]``. They are outside
      the Bonferroni family, so their corrected field must be **empty**.
    """
    observed = _require_file(observed, "observed")
    expected = _require_file(expected, "expected golden")

    _, got_rows = _parse_ks(observed)
    _, want_rows = _parse_ks(expected)

    # everything but the p-value is exact, so it is also the sort key that
    # pairs rows up for the p-value comparison
    got_fixed = sorted(r for r in got_rows if not _is_monte(r))
    want_fixed = sorted(r for r in want_rows if not _is_monte(r))
    got_exact = [r[:3] for r in got_fixed]
    want_exact = [r[:3] for r in want_fixed]
    if got_exact != want_exact:
        raise AssertionError(
            f"deterministic KS rows differ -- these are published numbers\n"
            f"  observed: {observed}\n  expected: {expected}\n"
            f"    observed (labels, statistic): {got_exact}\n"
            f"    expected (labels, statistic): {want_exact}"
        )

    for got, want in zip(got_fixed, want_fixed, strict=True):
        for offset in (3, 4):
            try:
                close = math.isclose(
                    float(got[offset]), float(want[offset]),
                    rel_tol=TSV_FLOAT_REL_TOL, abs_tol=TSV_FLOAT_REL_TOL,
                )
            except ValueError:
                close = False
            if not close:
                raise AssertionError(
                    f"deterministic KS {KS_COLUMNS[offset]!r} differs by more "
                    f"than {TSV_FLOAT_REL_TOL:g} relative -- a published number\n"
                    f"  observed: {observed}\n  expected: {expected}\n"
                    f"    observed: {got}\n    expected: {want}"
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
        if row[4] != "":
            raise AssertionError(
                f"Monte Carlo row carries a {KS_COLUMNS[4]} value in {observed}; "
                f"Monte Carlo comparisons are outside the family (row {row})"
            )
        for offset, name in enumerate(KS_COLUMNS[2:4], start=2):
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
    plotted values are covered by the ``.tsv.gz`` and ``.ks.csv`` goldens
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
