#!/bin/bash
xzcat micov/tests/test_data/test.sam.xz | micov compress > from_stdin
micov compress --data micov/tests/test_data/test.sam.xz > from_args
cmp --silent <(sort from_stdin) <(sort from_args)
if [[ $? -ne 0 ]]; then
    echo "Files are different"
    exit 1
fi
