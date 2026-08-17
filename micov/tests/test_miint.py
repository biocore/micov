"""Tests for the miint connection helper.

micov's compute internals are moving onto duckdb-miint, which ships as a
**DuckDB community extension** rather than a Python package -- there is no
`pip install` for it and no way to express the requirement in
`pyproject.toml`. It is therefore required at runtime, by the one helper every
DuckDB connection in micov is opened through, and these tests pin that
contract.

The miint-availability skip is deliberately narrow. It fires only on platforms
for which upstream publishes no build, so a load failure on a *supported*
platform stays a test failure rather than a green skip.
"""

import os
import platform
import shutil
import unittest
from tempfile import mkdtemp
from unittest import mock

import duckdb

from micov._constants import (
    COLUMN_COVERED,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_PERCENT_COVERED,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)
from micov._miint import MIINT_EXTENSION_PATH_VARIABLE, connection
from micov._view import View

# (system, machine) pairs upstream publishes a miint build for. Windows and
# Intel macOS are absent from the community repository -- probed directly, with
# the `h3` extension as a control to confirm the probe itself was sound -- and
# micov dropped both rather than carry a fallback path.
SUPPORTED_PLATFORMS = frozenset(
    {
        ("Linux", "x86_64"),
        ("Linux", "aarch64"),
        ("Linux", "arm64"),
        ("Darwin", "arm64"),
    }
)
PLATFORM = (platform.system(), platform.machine())

requires_miint_build = unittest.skipUnless(
    PLATFORM in SUPPORTED_PLATFORMS,
    f"miint publishes no community build for {PLATFORM[0]}/{PLATFORM[1]}; "
    "micov does not support this platform",
)


def miint_is_loaded(con):
    """Report whether the miint extension is loaded on `con`."""
    return (
        con.sql(
            "SELECT loaded FROM duckdb_extensions() WHERE extension_name = 'miint'"
        ).fetchall()
        == [(True,)]
    )


def installed_extension_path():
    """Return the on-disk path of the installed miint build, or None."""
    con = duckdb.connect(":memory:")
    try:
        con.sql("INSTALL miint FROM community")
        rows = con.sql(
            "SELECT install_path FROM duckdb_extensions() "
            "WHERE extension_name = 'miint'"
        ).fetchall()
    except duckdb.Error:
        return None
    finally:
        con.close()
    return rows[0][0] if rows and rows[0][0] else None


@requires_miint_build
class MiintConnectionTests(unittest.TestCase):
    def test_extension_is_loaded_and_answers(self):
        """Every micov connection must carry miint, ready to be called.

        Later milestones replace micov's hand-written SQL with miint
        primitives, so "loaded" is not enough -- the extension has to actually
        answer, which catches a build that loads but is ABI-mismatched.
        """
        con = connection()
        self.addCleanup(con.close)

        self.assertTrue(miint_is_loaded(con))

        version = con.sql("SELECT miint_version()").fetchall()
        # Only presence is asserted, not an ordering. miint reports a git short
        # hash (e.g. 'c2e8d97'), not a semantic version, so there is no
        # orderable floor to compare against -- see the note in `_miint.py`.
        self.assertEqual(len(version), 1)
        self.assertTrue(version[0][0])

    def test_settings_reach_duckdb(self):
        """`threads` and `memory` must not be dropped on the way through.

        micov runs on shared HPC nodes where an unbounded connection is a
        problem for everyone else on the box, so the caller's limits have to
        take effect, not merely be accepted.
        """
        con = connection(memory="2gb", threads=3)
        self.addCleanup(con.close)

        threads, memory_limit = con.sql(
            "SELECT current_setting('threads'), current_setting('memory_limit')"
        ).fetchall()[0]

        self.assertEqual(threads, 3)
        # compared against a bare connection given the same value rather than
        # against a literal: DuckDB renders the limit in its own units, and the
        # claim here is that the value arrives unchanged, not how it prints
        reference = duckdb.connect(":memory:", config={"memory_limit": "2gb"})
        self.addCleanup(reference.close)
        self.assertEqual(
            memory_limit,
            reference.sql("SELECT current_setting('memory_limit')").fetchall()[0][0],
        )

    def test_default_settings_match_the_previous_view_defaults(self):
        """The helper inherits View's old defaults verbatim.

        `View` opened its own connection with these values before the helper
        existed; changing them here would silently change every command's
        memory ceiling.
        """
        con = connection()
        self.addCleanup(con.close)

        threads, memory_limit = con.sql(
            "SELECT current_setting('threads'), current_setting('memory_limit')"
        ).fetchall()[0]

        reference = duckdb.connect(
            ":memory:", config={"threads": 1, "memory_limit": "8gb"}
        )
        self.addCleanup(reference.close)
        self.assertEqual(
            (threads, memory_limit),
            reference.sql(
                "SELECT current_setting('threads'), current_setting('memory_limit')"
            ).fetchall()[0],
        )

    def test_override_path_loads_a_local_build(self):
        """`MICOV_MIINT_EXTENSION_PATH` loads a build from disk.

        This is the escape hatch for an air-gapped deployment and for local
        miint development, both of which are on the critical path for the rest
        of the migration.
        """
        path = installed_extension_path()
        if path is None or not os.path.exists(path):
            self.skipTest("no installed miint build to point the override at")

        with mock.patch.dict(os.environ, {MIINT_EXTENSION_PATH_VARIABLE: path}):
            con = connection()
        self.addCleanup(con.close)

        self.assertTrue(miint_is_loaded(con))


class MiintFailureTests(unittest.TestCase):
    """Failures must be micov's own, and legible.

    A raw DuckDB `IOException` naming an extension the user has never heard of
    is not an actionable message; micov owns the requirement, so it owns the
    error.
    """

    def test_unusable_override_path_is_reported_by_micov(self):
        missing = "/nonexistent/miint.duckdb_extension"

        with mock.patch.dict(os.environ, {MIINT_EXTENSION_PATH_VARIABLE: missing}):
            with self.assertRaises(RuntimeError) as ctx:
                connection()

        # not a duckdb.Error: the point is that micov replaced the raw
        # exception rather than letting it through
        self.assertNotIsInstance(ctx.exception, duckdb.Error)

        message = str(ctx.exception)
        self.assertIn(missing, message)
        self.assertIn(MIINT_EXTENSION_PATH_VARIABLE, message)

    def test_failure_message_names_the_platform_and_the_override(self):
        """The message has to tell an unsupported-platform user what happened.

        Windows and Intel macOS users get a load failure and nothing else to go
        on unless the error says so.
        """
        with mock.patch.dict(
            os.environ, {MIINT_EXTENSION_PATH_VARIABLE: "/nonexistent/miint.ext"}
        ):
            with self.assertRaises(RuntimeError) as ctx:
                connection()

        message = str(ctx.exception)
        self.assertIn(platform.system(), message)
        self.assertIn(platform.machine(), message)
        self.assertIn("miint", message)

        # the real DuckDB version, asked of the connection. `duckdb.__version__`
        # is `None` on the 1.5.4 PyPI wheel, whose metadata is broken, and
        # "DuckDB None" in the message would strand exactly the user who most
        # needs it -- someone whose DuckDB has no matching miint build.
        version = duckdb.connect(":memory:").sql("SELECT version()").fetchall()[0][0]
        self.assertIn(version, message)
        self.assertNotIn("DuckDB None", message)

    def test_view_surfaces_a_connection_failure_cleanly(self):
        """A failed connection must not also spew from `View.__del__`.

        `View` closes its connection in `__del__`; if the connection never
        existed, the interpreter prints an ignored `AttributeError` on top of
        micov's authored message, which buries the actionable part.
        """
        with mock.patch.dict(
            os.environ, {MIINT_EXTENSION_PATH_VARIABLE: "/nonexistent/miint.ext"}
        ):
            with self.assertRaises(RuntimeError):
                View("/nonexistent/base", None, None)

        # the object was allocated, so __del__ will run; it must tolerate a
        # View whose connection was never opened
        view = View.__new__(View)
        view.__del__()


@requires_miint_build
class ViewUsesTheHelperTests(unittest.TestCase):
    """`View` must open through the helper, not `duckdb.connect` directly."""

    def setUp(self):
        self.d = mkdtemp()
        con = duckdb.connect(":memory:")
        con.sql(
            f"""COPY (
                    SELECT * FROM (VALUES ('G1', 'S1', 8::UINTEGER,
                                           100::UINTEGER, 8.0::DOUBLE))
                    t({COLUMN_GENOME_ID}, {COLUMN_SAMPLE_ID}, {COLUMN_COVERED},
                      {COLUMN_LENGTH}, {COLUMN_PERCENT_COVERED})
                ) TO '{self.d}/testdata.coverage.parquet'"""
        )
        con.sql(
            f"""COPY (
                    SELECT * FROM (VALUES ('G1', 'S1', 1::UINTEGER, 9::UINTEGER))
                    t({COLUMN_GENOME_ID}, {COLUMN_SAMPLE_ID}, {COLUMN_START},
                      {COLUMN_STOP})
                ) TO '{self.d}/testdata.covered_positions.parquet'"""
        )
        con.close()

        self.metadata = f"{self.d}/sample_metadata.tsv"
        with open(self.metadata, "w") as fp:
            fp.write(f"{COLUMN_SAMPLE_ID}\tfoo\nS1\ta\n")

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_view_connection_carries_miint(self):
        view = View(f"{self.d}/testdata", self.metadata, None)
        self.addCleanup(view.close)

        self.assertTrue(miint_is_loaded(view.con))
        # and the View still works, so loading miint did not disturb it
        self.assertEqual(view.coverages().fetchall(), [("G1", "S1", 8, 100, 8.0)])


if __name__ == "__main__":
    unittest.main()
