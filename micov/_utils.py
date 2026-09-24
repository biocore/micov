import logging


def configure_logger():
    logger = logging.getLogger("micov")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = configure_logger()


def sql_string(value):
    """Render `value` as a SQL string literal, quotes escaped.

    Every path and user-supplied ID micov interpolates into SQL goes through
    this. Without it a single quote -- ``/Users/o'brien`` is an ordinary macOS
    home -- ends the literal early and DuckDB fails with a parse error naming
    neither the path nor micov.

    Escaping rather than bound parameters, deliberately. Most of these
    literals sit in SQL *fragments* -- `View._read_tsv` and the `source`
    strings in `_io` -- that are spliced into larger statements, several of
    them ``CREATE VIEW``, where DuckDB rejects prepared parameters outright
    ("This type of statement can't be prepared"). Escaping is the one
    mechanism that works at every site, so every site uses it.
    """
    return "'" + str(value).replace("'", "''") + "'"
