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

## The `dp_*` corpus (`depth-plot`)

A **synthetic** corpus for `depth-plot`, small enough to compute every
expected value by hand. The sources are the readable `dp.sam` and `dp.gff`.
The Parquet inputs are derived from them through miint's own
`read_alignments` and `read_gff` (block below), so their schemas are exactly
what a user's files would have. The layer each read belongs to is in its
QNAME, `sample:layer:id`, and `sample_id` is the part before the first `:`.

| File | What |
|---|---|
| `dp_depth.parquet` | reads whose layer is `both` or `depth` |
| `dp_breadth.parquet` | reads whose layer is `both` or `breadth` |
| `dp_orfs.parquet` | every `dp.gff` line, through `read_gff` |
| `dp_features.tsv` | `genome_id`, `length`, `is_circular` |
| `dp_regions.tsv` | the same plus `start`/`stop`: two detail regions on GC, none on the others |
| `dp_metadata.tsv` | `sample_id` and four columns giving 2, 3 and 4 groups, one with a `'` in a value |
| `dp_target_names.tsv` | lineage-style names for GC and GL |

**Genomes**

| Genome | Length | Why |
|---|---|---|
| GC | 3,000, circular | the circular path; regions [1001, 1501) and [2501, 2901) |
| GL | 2,000, linear | the linear path |
| GX | 1,000 | depth layer only: left out and reported |
| GB | 1,000 | breadth layer only: left out and reported |

**Samples** (`group` / `trio` / `quad` / `site`)

| Sample | Values | Layers | Why |
|---|---|---|---|
| S1–S3 | case / a,b,c / a,b,c / o'hare | both | case, n = 3 |
| S4, S5 | control / a,b / d,a / midway | both | control, n = 2 |
| S6 | case / c / b / midway | breadth only | left out and reported |
| S7 | control / a / c / midway | depth only | left out and reported |
| S8 | control / b / d / midway | neither | in the metadata with no reads: reported |
| S9 | — | both | not in the metadata: left out by the user, so not reported |

**Reads worth knowing about** (GC unless said)

| Read | Span | Why |
|---|---|---|
| `S1:both:a1`, flag 256 | [1201, 1301) | a secondary alignment, which counts |
| `S1:both:a2`, flag 16 | [951, 1051) | crosses 1001, a bin and region edge |
| `S1:both:a3` | [2951, 3001) | ends exactly at the genome's end |
| `S2:both:b1` `50M10D40M` | [1001, 1101) | the deletion counts toward depth |
| `S3:both:c2` `30M100N30M` | [2601, 2761) | the skip counts toward breadth, not depth |
| `S4:both:d2`, `d3` | [1221, 1321) twice | depth 2 |
| `S4:both:d4`, flag 4 at 1500 | stop 0 | a placed unmapped read, which covers nothing |
| `S2:depth:b4`, flag 4, RNAME `*` | — | unplaced: never a genome |
| `S1:breadth:a6` (GL) | [1801, 1851) | breadth over gl_5, which has no depth |

**ORFs.** Besides the plain ones: a `gene` and a `region` line, which are
dropped; gc_2 (−, crosses 1001, label from `locus_tag`, `product` with a
`'`); gc_4 (strand `.`, label from `ID`); gc_5 (under the `N` skip); gc_7
(2951–3050, crossing the origin of circular GC); gl_5 (breadth but no depth);
gx_1 on the left-out GX; gq_1 on GQ, which is not a feature.

Regenerate the Parquet inputs, from the repository root:

```bash
D=micov/tests/test_data
python - "$D" <<'PY'
import sys
from micov._miint import connection
from micov._utils import sql_string
D = sys.argv[1]
con = connection()
con.sql("""CREATE TABLE genome_lengths AS SELECT * FROM (VALUES
               ('GC', 3000::BIGINT), ('GL', 2000::BIGINT),
               ('GX', 1000::BIGINT), ('GB', 1000::BIGINT)) t(genome_id, length)""")
con.sql(f"""CREATE TABLE a AS
            SELECT split_part(read_id, ':', 1) AS sample_id,
                   split_part(read_id, ':', 2) AS layer, *
            FROM read_alignments({sql_string(D + '/dp.sam')},
                                 reference_lengths := genome_lengths)""")
for layer in ("depth", "breadth"):
    con.sql(f"""COPY (SELECT * EXCLUDE (layer) FROM a
                      WHERE layer IN ('both', {sql_string(layer)})
                      ORDER BY sample_id, read_id, flags)
                TO {sql_string(f'{D}/dp_{layer}.parquet')}
                (FORMAT PARQUET, COMPRESSION zstd)""")
con.sql(f"""COPY (SELECT * FROM read_gff({sql_string(D + '/dp.gff')})
                  ORDER BY seqid, position, type)
            TO {sql_string(D + '/dp_orfs.parquet')} (FORMAT PARQUET, COMPRESSION zstd)""")
PY
```

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
