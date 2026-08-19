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

#: Points at a miint build on disk, bypassing the repository entirely. This is
#: the escape hatch for local miint development and for deployments with no
#: outbound network -- micov reaches the network on first use otherwise.
MIINT_EXTENSION_PATH_VARIABLE = "MICOV_MIINT_EXTENSION_PATH"

#: Overrides `MIINT_REPOSITORY` so the source can move without a release.
MIINT_REPOSITORY_VARIABLE = "MICOV_MIINT_REPOSITORY"

#: **The only place micov names where the extension comes from.** miint is not
#: in the DuckDB community repository; it is published here. Changing where
#: micov installs from is this one line, which is why `connection()` is also the
#: only place micov opens a DuckDB connection -- see `cli.py`, which was moved
#: behind it so that stayed true.
MIINT_REPOSITORY = "https://ftp.microbio.me/pub/miint"

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
    # Unconditional, where M2 scoped this to the override branch alone. miint's
    # published builds are **not signed**, so DuckDB refuses them outright
    # ("Attempting to install an extension file that doesn't have a valid
    # signature") and micov cannot run at all without this. It is a development
    # posture, not a settled one: when miint is signed, delete this line and
    # restore the narrow version that only relaxed signatures for a build the
    # caller explicitly pointed at from disk.
    config = {
        "threads": threads,
        "memory_limit": memory,
        "allow_unsigned_extensions": True,
    }

    override = os.environ.get(MIINT_EXTENSION_PATH_VARIABLE)
    repository = os.environ.get(MIINT_REPOSITORY_VARIABLE, MIINT_REPOSITORY)

    con = duckdb.connect(":memory:", config=config)
    try:
        if override:
            con.sql(f"LOAD '{override}'")
        else:
            # INSTALL is a no-op when the extension is already present, so the
            # network is only reached on first use; seeding
            # ~/.duckdb/extensions/ ahead of time makes this work offline
            try:
                con.sql(f"INSTALL miint FROM '{repository}'")
            except duckdb.Error:
                # A cache holding a build from a *different* origin makes
                # INSTALL fail outright ("the origin is different ... rerun
                # with FORCE INSTALL") rather than fall through. Every user who
                # ran an earlier micov has exactly that cache, since micov used
                # to install from the DuckDB community repository, so without
                # this retry the repository change would strand all of them.
                #
                # Retried on any install error rather than by matching the
                # message: a corrupt or partial cache needs the same fix, and
                # if FORCE INSTALL fails too the error is raised below anyway.
                # The re-download happens once, not per connection.
                con.sql(f"FORCE INSTALL miint FROM '{repository}'")
            con.sql("LOAD miint")
    except duckdb.Error as exc:
        # asked of the connection rather than read from `duckdb.__version__`:
        # duckdb 1.5.4's PyPI wheel ships broken metadata and reports `None`
        # there, which would put "DuckDB None" in the one message the user has
        # to orient themselves by. `version()` is right on that wheel.
        version = con.sql("SELECT version()").fetchall()[0][0]
        con.close()
        raise RuntimeError(
            _unavailable_message(override, repository, version, exc)
        ) from exc

    return con


def _unavailable_message(override, repository, version, exc):
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
            f"or load it from '{repository}' for {context}."
        )
        remedy = (
            "miint is published for Linux (x86_64, aarch64) and macOS on "
            "Apple silicon; there is no build for Windows or Intel macOS, and "
            "the repository has to carry a build for the DuckDB version in "
            f"use. Set {MIINT_REPOSITORY_VARIABLE} to install from elsewhere, "
            f"or {MIINT_EXTENSION_PATH_VARIABLE} to load a build from disk. "
            "Installing reaches the network on first use."
        )

    return f"{problem}\n\n{remedy}\n\nDuckDB reported: {exc}"
