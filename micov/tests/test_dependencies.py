"""micov's runtime dependency surface.

The duckdb-miint migration exists to shrink this. Each dependency it removes is
removed from `pyproject.toml`, and a stale import left behind anywhere in the
package turns that into a broken install for anyone who does not happen to have
the old library sitting in their environment -- which, during the migration,
is everyone working on it.

So the claim these tests make is not "polars is unused". It is "micov runs
without polars installed at all", which is what removing it from
`pyproject.toml` promises on the user's behalf.
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent

#: Removed by the migration. `numba` went in M3 with the CIGAR walker, `pyarrow`
#: in M1 with the last polars<->DuckDB handoff, `polars` in M5, and `scipy` in
#: M9 when the KS tests moved to miint's `ks_2samp`.
#:
#: These are *import* names. micov depended on the distribution
#: `polars-u64-idx`, which installs a module called `polars` -- there is no
#: `polars_u64_idx` to import, so checking for that name would look like
#: coverage while testing nothing.
REMOVED_DEPENDENCIES = ("polars", "numba", "pyarrow", "scipy")


def imported_modules(path):
    """Return the top-level module names `path` imports."""
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


class DependencySurfaceTests(unittest.TestCase):
    def test_no_module_imports_a_removed_dependency(self):
        """Source-level check, so it names the file and the import.

        The subprocess check below is the stronger claim, but it reports only
        that *something* pulled the module in. This one says where.

        Covers `micov/tests/` as well as the package. A test module importing
        a dropped library does not break an install, but it does mean the
        suite cannot be run against one -- and the suite is the only evidence
        that the install works.
        """
        offenders = {}
        sources = sorted(PACKAGE.glob("*.py")) + sorted(PACKAGE.glob("tests/*.py"))
        for module in sources:
            used = imported_modules(module) & set(REMOVED_DEPENDENCIES)
            if used:
                offenders[str(module.relative_to(PACKAGE))] = sorted(used)

        self.assertEqual(
            offenders,
            {},
            f"modules still import dependencies micov has dropped: {offenders}",
        )

    def test_importing_micov_does_not_load_a_removed_dependency(self):
        """The real claim: a fresh interpreter never touches them.

        Run out-of-process on purpose. Checking `sys.modules` inside the test
        run proves nothing -- the rest of the suite has already imported these
        libraries by then, and the assertion would pass or fail on test
        ordering rather than on micov.
        """
        probe = (
            "import sys, micov.cli, micov._io, micov._plot, micov._view, "
            "micov._quant, micov._cov, micov._constants; "
            f"print(','.join(m for m in {REMOVED_DEPENDENCIES!r} "
            "if m in sys.modules))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, check=False
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"importing micov failed:\n{proc.stderr.decode(errors='replace')}",
        )
        self.assertEqual(
            proc.stdout.decode().strip(),
            "",
            "importing micov loaded a dependency it no longer declares",
        )


if __name__ == "__main__":
    unittest.main()
