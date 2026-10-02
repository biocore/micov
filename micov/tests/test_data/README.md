# Test fixtures

Inputs for the test suite. Frozen expected outputs live in `golden/`.

`example/` at the repository root is the large golden corpus, but
`MANIFEST.in` prunes it from the sdist, so anything that must run in the fast
tier gets its inputs from here instead.

## Real-data fixtures

- **`test.sam.xz`** — pre-existing. A headerless SAM covering **232 genomes**,
  915 compressed intervals. Also used by `cli_test.sh`.

- **`lengths.tsv`** — genome lengths for all 232 genomes in `test.sam.xz`,
  needed at full width rather than as a subset, since a genome with no length
  cannot be given a coverage denominator.

  It is **synthetic**: lengths are the maximum observed `stop` per genome
  rounded up to the next 10 kb, which keeps every `percent_covered` in range
  (the maximum is ~1.5%). These are not the real lengths for these accessions
  and must not be used as reference data — the file exists to exercise
  `--lengths` deterministically.

  It carries a header, which also exercises the header-detection branch in
  `_test_has_header`.

- **`feature_metadata_regions.tsv`** — sub-genome regions for the two genomes
  in `example/parquet/`, for `extract-sample-presence` (which raises
  `Cannot calculate presence/absence without positions.` without them). The
  windows were chosen for a mixed result rather than arbitrarily:
  `G000154205 [2000000, 2005000)` splits 36 present / 13 absent and
  `G000436435 [1000000, 1005000)` splits 22 / 27.

## The `mini_*` corpus

A three-sample synthetic corpus. It exists because of one specific gap: the
`example/` corpus **cannot** produce the third presence state. All 49 of its
samples have coverage of both genomes, so every result is `present` or
`absent`, and `not applicable` is unreachable.

`not applicable` is produced for a sample that appears in neither the
`present` nor the `absent` pivot, i.e. one with no coverage of that genome at
all — a branch that is easy to lose while rewriting
`_view.sample_presence_absence`, which is why the state is pinned explicitly.

The corpus is built so one sample lands in each state for region
`G000000001 [4000, 5000)`:

| Sample | `G000000001` | Result |
|---|---|---|
| `mini_sampleA` | `[4200, 4500)` — inside the region | `present` |
| `mini_sampleB` | `[100, 500)` — genome yes, region no | `absent` |
| `mini_sampleC` | none at all | `not applicable` |

All three cover `G000000002`, so `mini_sampleC` still appears in the Parquet
and the sample metadata — otherwise it would drop out before the pivot and
prove nothing.

Files: `mini_sample{A,B,C}.cov` (BED3 with header, as `micov compress`
writes), `mini_lengths.tsv`, `mini_regions.tsv`, `mini_sample_metadata.tsv`
(a `group` column of `case`/`control`).

This corpus also gives the fast tier its only coverage of
**`cov-to-parquet`**. That command was `nonqiita-to-parquet` until M4, when
micov dropped Qiita support; `qiita-to-parquet`, which had produced the
committed `example/parquet/` files, was removed and that corpus regenerated
with `cov-to-parquet`. It is now the only producer of the Parquet pair from
`.cov` input, so these goldens are the whole of its coverage.

## `golden/`

Frozen expected outputs. **Do not refresh one to make a test pass.** These
encode published behavior; a failure means the code changed, until proven
otherwise. Which comparator each requires is decided by the determinism
classification in the module docstring of `micov/tests/_golden.py` — read that
before touching anything here.

Regenerate with, from the repository root:

```bash
D=micov/tests/test_data; G=$D/golden

# writes both mini.coverage.parquet and mini.covered_positions.parquet
micov cov-to-parquet --pattern "$D/mini_sample*.cov" \
    --output $G/mini --lengths $D/mini_lengths.tsv

micov extract-sample-presence --parquet-coverage $G/mini \
    --sample-metadata $D/mini_sample_metadata.tsv \
    --features-to-keep $D/mini_regions.tsv --output $G/mini_presence.tsv

micov extract-sample-presence --parquet-coverage example/parquet/example \
    --sample-metadata example/metadata/sample_metadata.txt \
    --features-to-keep $D/feature_metadata_regions.tsv \
    --output $G/example_presence.tsv
```

`example_presence.tsv` needs `example/`, which `MANIFEST.in` prunes from
sdists, so it can only be regenerated from a git checkout.

`lengths.tsv` is derived from `test.sam.xz` rather than produced by a micov
command; see the note above on how its values are computed. (`taxonomy.tsv`,
its orphaned sibling, was removed in M11b.)
