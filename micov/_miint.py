"""The single DuckDB connection micov opens, with miint loaded.

`duckdb-miint <https://github.com/the-miint/duckdb-miint>`_ ships as a DuckDB
**community extension**, not as a Python package: there is no distribution to
depend on, so the requirement cannot be expressed in ``pyproject.toml`` and is
enforced here instead, at the one place micov opens a connection.

The extension is required, not optional. micov does not fall back to its
previous hand-written SQL when miint is missing -- a silent fallback would mean
two compute paths that have to agree numerically, and micov's published
coverage values and KS statistics are a frozen contract.

Community extensions are built per DuckDB version and per platform. micov
supports Linux (x86_64 and aarch64) and macOS on Apple silicon; upstream
publishes no build for Windows or Intel macOS.
"""

import os
import platform

import duckdb

#: Points at a miint build on disk, bypassing the community repository. This is
#: the escape hatch for local miint development and for deployments with no
#: outbound network -- micov reaches the network on first use otherwise.
MIINT_EXTENSION_PATH_VARIABLE = "MICOV_MIINT_EXTENSION_PATH"

# There is deliberately no minimum-version check. `miint_version()` reports a
# git short hash (e.g. 'c2e8d97'), not a semantic version, so there is nothing
# to order a floor against, and pinning an exact hash would break micov on
# every upstream commit. The check that will earn its keep is a capability one
# -- assert the specific functions micov calls exist -- and it belongs with the
# first micov code that calls a miint primitive, where the required set is
# known rather than empty.


def connection(memory="8gb", threads=1):
    """Open an in-memory DuckDB connection with the miint extension loaded.

    Parameters
    ----------
    memory : str, optional
        DuckDB ``memory_limit`` for the connection.
    threads : int, optional
        DuckDB ``threads`` for the connection.

    Returns
    -------
    duckdb.DuckDBPyConnection
        A connection on which miint's functions are callable.

    Raises
    ------
    RuntimeError
        If the extension cannot be installed or loaded. The message names the
        platform, the DuckDB version and the override variable; micov replaces
        DuckDB's own error rather than letting it through, because the
        requirement is micov's and DuckDB's message does not mention micov.
    """
    config = {"threads": threads, "memory_limit": memory}

    override = os.environ.get(MIINT_EXTENSION_PATH_VARIABLE)
    if override:
        # only when a build from disk was explicitly asked for: local and
        # development builds are unsigned, but relaxing this unconditionally
        # would let any unsigned extension load on every micov connection
        config["allow_unsigned_extensions"] = True

    con = duckdb.connect(":memory:", config=config)
    try:
        if override:
            con.sql(f"LOAD '{override}'")
        else:
            # INSTALL is a no-op when the extension is already present, so the
            # network is only reached on first use; seeding
            # ~/.duckdb/extensions/ ahead of time makes this work offline
            con.sql("INSTALL miint FROM community")
            con.sql("LOAD miint")
    except duckdb.Error as exc:
        # asked of the connection rather than read from `duckdb.__version__`:
        # duckdb 1.5.4's PyPI wheel ships broken metadata and reports `None`
        # there, which would put "DuckDB None" in the one message the user has
        # to orient themselves by. `version()` is right on that wheel.
        version = con.sql("SELECT version()").fetchall()[0][0]
        con.close()
        raise RuntimeError(_unavailable_message(override, version, exc)) from exc

    return con


def _unavailable_message(override, version, exc):
    """Compose the error micov reports when miint cannot be loaded."""
    system = platform.system()
    machine = platform.machine()
    context = f"DuckDB {version} on {system}/{machine}"

    if override:
        problem = (
            f"micov could not load the miint extension from '{override}', "
            f"named by {MIINT_EXTENSION_PATH_VARIABLE}."
        )
        remedy = (
            f"Check that the path is a miint build for {context}. Unset "
            f"{MIINT_EXTENSION_PATH_VARIABLE} to install from the DuckDB "
            "community repository instead."
        )
    else:
        problem = (
            "micov requires the miint DuckDB extension, and could not install "
            f"or load it for {context}."
        )
        remedy = (
            "miint is published for Linux (x86_64, aarch64) and macOS on "
            "Apple silicon; there is no build for Windows or Intel macOS, and "
            "a build has to exist for the DuckDB version in use. Installing "
            "reaches the network on first use. To load a build from disk "
            f"instead, set {MIINT_EXTENSION_PATH_VARIABLE}."
        )

    return f"{problem}\n\n{remedy}\n\nDuckDB reported: {exc}"
