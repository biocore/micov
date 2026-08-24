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

from micov import _miint
from micov._constants import (
    COLUMN_COVERED,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_PERCENT_COVERED,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)
from micov._miint import (
    MIINT_EXTENSION_PATH_VARIABLE,
    MIINT_REPOSITORY,
    MIINT_REPOSITORY_VARIABLE,
    REQUIRED_MIINT_FUNCTIONS,
    connection,
)
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
    """Return the on-disk path of the installed miint build, or None.

    Goes through `connection()` rather than installing independently. An
    earlier version named the repository itself, which silently rotted the
    moment micov changed where it installs from -- the install failed, this
    returned None, and the override test skipped instead of failing. Asking the
    helper keeps there being exactly one place that knows the source.
    """
    con = connection()
    try:
        rows = con.sql(
            "SELECT install_path FROM duckdb_extensions() "
            "WHERE extension_name = 'miint'"
        ).fetchall()
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


@requires_miint_build
class MiintRepositoryTests(unittest.TestCase):
    """Where the extension is installed from is named in exactly one place.

    micov installs miint from its own repository rather than the DuckDB
    community one, and that will move again when miint is signed and published.
    The point of these tests is not the current URL -- it is that there is a
    single constant to change, and that nothing else in micov names a source.
    """

    def test_repository_is_the_default_source(self):
        con = connection()
        self.addCleanup(con.close)

        installed_from = con.sql(
            "SELECT installed_from FROM duckdb_extensions() "
            "WHERE extension_name = 'miint'"
        ).fetchall()

        # 'community' would mean the constant is not being used
        self.assertEqual(len(installed_from), 1)
        self.assertNotEqual(installed_from[0][0], "community")

    def test_repository_is_overridable_without_a_code_change(self):
        """The override exists so a redeploy is not a release.

        A bad URL must fail as micov's own error naming the variable, the same
        way a bad `MICOV_MIINT_EXTENSION_PATH` does -- not as a bare DuckDB
        HTTP error the user cannot connect to micov.
        """
        with mock.patch.dict(
            os.environ, {MIINT_REPOSITORY_VARIABLE: "https://nonexistent.invalid/miint"}
        ):
            with self.assertRaises(RuntimeError) as ctx:
                connection()

        message = str(ctx.exception)
        self.assertNotIsInstance(ctx.exception, duckdb.Error)
        self.assertIn("https://nonexistent.invalid/miint", message)
        self.assertIn(MIINT_REPOSITORY_VARIABLE, message)

    def test_default_repository_is_named_in_the_message(self):
        """An install failure has to say where micov was looking."""
        self.assertTrue(MIINT_REPOSITORY.startswith("http"))

        con = connection()
        self.addCleanup(con.close)
        self.assertTrue(miint_is_loaded(con))


@requires_miint_build
class MiintCapabilityTests(unittest.TestCase):
    """micov names the miint functions it calls, and checks for them.

    There is deliberately no *version* floor -- `miint_version()` reports a git
    short hash, so there is nothing to order against. The check that earns its
    keep is this one, and it only became possible to write once micov depended
    on a known set of primitives rather than none.

    Without it, an miint too old for one of these fails deep inside a query as
    `Catalog Error: Table Function with name region_coverage does not exist`,
    which names neither micov nor the extension the user has to update.
    """

    def test_every_required_function_is_present(self):
        """The real assertion: the deployed build satisfies micov.

        Written against the live connection rather than a stub so that an
        upstream removal is caught here, in one obvious place, instead of as a
        scattered set of query failures.
        """
        con = connection()
        self.addCleanup(con.close)

        for name in REQUIRED_MIINT_FUNCTIONS:
            with self.subTest(function=name):
                self.assertTrue(
                    con.execute(
                        "SELECT COUNT(*) FROM duckdb_functions() "
                        "WHERE function_name = ?",
                        [name],
                    ).fetchone()[0]
                )

    def test_a_missing_function_is_micovs_own_error(self):
        """An miint without one of them must fail at connect, naming it.

        The required set is patched rather than the extension downgraded --
        micov cannot install an older miint on demand, and the branch under
        test is "a name micov needs is absent from `duckdb_functions()`",
        which a fictional name reproduces exactly.
        """
        # deliberately not a superstring of a real name -- the assertion below
        # tests substrings, and "region_coverage_that_is_absent" would contain
        # "region_coverage" and fail against a correct message
        absent = "no_such_miint_function"
        with mock.patch.object(
            _miint, "REQUIRED_MIINT_FUNCTIONS", (*REQUIRED_MIINT_FUNCTIONS, absent)
        ):
            with self.assertRaises(RuntimeError) as ctx:
                connection()

        message = str(ctx.exception)
        self.assertNotIsInstance(ctx.exception, duckdb.Error)
        self.assertIn(absent, message)
        self.assertIn("miint", message)
        # naming only what is missing -- a message listing everything micov
        # needs buries the one line the user has to act on
        for present in REQUIRED_MIINT_FUNCTIONS:
            self.assertNotIn(present, message)


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
