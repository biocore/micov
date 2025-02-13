"""micov: microbiome coverage."""

from . import _version

__version__ = _version.get_versions()["version"]
# note: currently for use with duckdb. we cannot easily enforce threads for polars
# as a specific environment variable must be set prior to the first import. it's
# doable but will need some engineeering to do it correctly.'And, polars does not
# currently have a way to limit memory use.
THREADS = 1
MEMORY = 8  # gb
