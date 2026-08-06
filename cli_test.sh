#!/bin/bash
# Outputs go to a temp directory: writing them into the repo root left two
# untracked files behind on every `make test`.
out=$(mktemp -d)
xzcat micov/tests/test_data/test.sam.xz | micov compress > "$out/from_stdin"
micov compress --data micov/tests/test_data/test.sam.xz > "$out/from_args"
# Both sides are sorted because `compress` row order is not stable: it groups
# by genome_id without maintain_order, so only the content is comparable.
cmp --silent <(sort "$out/from_stdin") <(sort "$out/from_args")
if [[ $? -ne 0 ]]; then
    echo "Files are different"
    exit 1
fi
