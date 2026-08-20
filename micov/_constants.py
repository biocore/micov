"""Column names, and the values micov writes into them.

These names are a **frozen output contract**: they are the column headers of
`{base}.coverage.parquet` and `{base}.covered_positions.parquet`, and released
micov versions read files written under them.

This module used to also carry polars dtypes and seven `_SCHEMA` objects
describing frames micov passed around. Those went in M5 -- the schemas belong
to whatever produces the data, which is now SQL, and every one of them had
been left without a caller by an earlier milestone. `micov/tests/test_view.py`
states the same types in DuckDB's terms, which is what they are on disk.
"""

COLUMN_GENOME_ID = "genome_id"
COLUMN_SAMPLE_ID = "sample_id"
COLUMN_START = "start"
COLUMN_STOP = "stop"
COLUMN_LENGTH = "length"
COLUMN_COVERED = "covered"
COLUMN_PERCENT_COVERED = "percent_covered"
COLUMN_NAME = "name"
COLUMN_REGION_ID = "region_id"

PRESENT = "present"
ABSENT = "absent"
NOT_APPLICABLE = "not applicable"
