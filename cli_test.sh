#!/bin/bash
# Outputs go to a temp directory: writing them into the repo root left two
# untracked files behind on every `make test`.
out=$(mktemp -d)
lengths=micov/tests/test_data/lengths.tsv

# The pipe idiom is the reason this script exists. Headerless SAM on stdin has
# no reference map and htslib needs one, so this only works because --lengths
# supplies it up front and the stream is read exactly once, never rewound.
xzcat micov/tests/test_data/test.sam.xz \
    | micov compress --lengths "$lengths" --sample-id test \
                     --output "$out/from_stdin"
micov compress --data micov/tests/test_data/test.sam.xz \
               --lengths "$lengths" --output "$out/from_args"

# Compared as an ordered row set rather than byte-for-byte: `compress` groups
# by genome without imposing an order, so only the content is comparable.
python - "$out" <<'PY'
import sys

import duckdb

out = sys.argv[1]


def intervals(base):
    return duckdb.sql(
        f"SELECT genome_id, start, stop "
        f"FROM '{out}/{base}.covered_positions.parquet' ORDER BY 1, 2, 3"
    ).fetchall()


if intervals("from_stdin") != intervals("from_args"):
    print("Files are different")
    sys.exit(1)
PY
