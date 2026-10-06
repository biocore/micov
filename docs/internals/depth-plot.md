# depth-plot

**Read this when** you touch `_depth.py`, or `depth-plot`'s readers in
`_io.py`. The command is not wired into the CLI yet; this describes the
pieces built so far. [data-formats.md](data-formats.md) has the input formats.

`depth-plot` draws, along each genome, per-base **depth** from one alignment
layer and **breadth** from another (say metatranscriptomic over
metagenomic), for each group of a metadata column. Both layers are
`read_alignments` output with a `sample_id` column, and may be the same file.

## Which samples and genomes: `_depth.intersect_layers`

- A sample is used if it is in the metadata and has an aligned read
  (`_io.ALIGNED_ROWS`) in **both** layers. A genome is used if it is in the
  features and has an aligned read in both layers.
- Whatever the metadata or features name but this leaves out is warned about
  by name. A sample with no metadata, or a genome not in the features, was
  left out by the user, and is not.
- No sample or no genome left, more than `_plot.MAX_GROUPS` groups, an
  aligned read starting beyond its genome's `length`, or (with ORFs) no ORF
  on a used genome or an ORF outside its genome: `ValueError`. Only a
  circular genome's ORF may run past its end. All of this is checked before
  any genome is computed.
- It creates `depth_roster` (`sample_id`, a dense `sample_idx` in `sample_id`
  order, `group_name`) and `depth_genomes`.

`test_depth.IntersectLayersTests` pins these on the `dp_*` fixtures.

## Per-base depth, in windows

A genome's depth is computed in windows of `window_size(n, L) = max(1,
min(L, WINDOW_CELLS // n))` bases, so memory is bounded by the number of
samples, not the genome's length.

- **`stage_depth`** copies one genome's aligned reads of the used samples into
  the temp table `depth_alignments`: `sample_idx`, UINTEGER `position` and
  `stop_position`, `cigar`, sorted by position.
- **`window_depth(con, n, w0, w1)`** calls miint's `compute_coverage_depth`
  per sample, with positions shifted by `w0 - 1` and the window's width as the
  reference length. miint clips reads that start before `w0` or run past
  `w1`, so windows need no padding. It returns an `n` x (`w1 - w0`) uint32
  array; a sample without reads in the window is a row of zeros.
- **What counts as depth:** M, = and X, and **D** (the read spans the deleted
  bases). **N never counts**: nothing was sequenced there. S, H, I and P
  consume no reference. A read overhanging the genome's end is clipped.

## Breadth

- **`stage_breadth`** merges each used sample's aligned breadth-layer reads per
  genome with `compress_intervals`, into the temp table `breadth_intervals`.
  Merged, so a sample counts once however many of its reads overlap. A read's
  whole span counts, **N included**: breadth is where reads align, depth where
  their bases are.
- **`coverage_counts`** gives how many of a group's intervals cover each base
  of a window, by binary search over the group's sorted starts and sorted
  stops (count at x = #{start ≤ x} − #{stop ≤ x}). `genome_statistics`
  sorts them once per genome.

## Group statistics and bins: `_depth.genome_statistics`

`genome_statistics(con, depth_view, genome_id, length, edge_sets, orfs=None)`
makes one pass of windows and fills every bin set at once, returning
`(bins, orf_table)`: the overview,
`display_bin_edges(1, L + 1, OVERVIEW_BIN_BP)`, and each detail region,
`display_bin_edges(start, stop, detail_bin_bp(start, stop))`, which has at
most `DETAIL_MAX_BINS` bins (single bases for a region up to 1,500 bp).

- **At each base, per group:** depth Q1, median and Q3 across the group's
  samples (numpy's default linear method), with a sample without reads counted
  as 0; the mean depth; **prevalence**, the share of the group's samples whose
  breadth covers the base; and the **union**, whether any does.
- **A bin is the mean of its bases' values**, and is in the union if any base
  is. So the median drawn is a per-base median, never the median of
  per-sample averages: with depth 3 in one of three samples at base 1 and in
  another at base 2, the bin's median is 0, not 1.5
  (`GenomeBinsTests.test_a_bin_averages_per_base_quantiles`).
- **Exact for any window size.** On integer depth the linear quantiles are
  multiples of 0.25, so `quartiles_x4` is an exact integer. All six series (Q1,
  median and Q3 times four, depth, covering samples, covered bases) are
  integer bin sums, carried across windows by prefix sums and divided once at
  the end. `WindowInvarianceTests` checks bit-identical bins for windows of 1,
  2, 3, 7 and L bases, and agreement with a Python CIGAR-walking oracle. Do
  not divide inside the window loop.
- **The table**, one per edge set: `group`, `bin_start`, `bin_stop` (1-based,
  half-open), `q1`, `median`, `q3`, `mean`, `prevalence` (float64), `union`
  (bool), a row per group (sorted) and bin.

Every span set -- the bins and the ORFs' segments -- is summed the same way,
by `_add_span_sums`, which touches only the spans a window overlaps.

## Per-ORF statistics

`genome_orfs(con, genome_id)` gives a genome's ORFs by position. With
`orfs`, `genome_statistics` also returns the per-ORF table that
`_io.write_orf_table` writes ([data-formats.md](data-formats.md)).

- **Segments.** `orf_segments` splits an ORF running past the genome's end,
  which only a circular genome's may, into `[start, L + 1)` and
  `[1, stop − L)`. An ORF's length is `stop − start`.
- **Per sample, then per group.** Each sample's depth is summed over the
  ORF's bases, in the same window pass, and divided by the length: that
  sample's mean depth. The group's `depth_q1`, `depth_median` and
  `depth_q3` are quantiles of those, so they describe the typical *sample*,
  where the bins' describe the typical *base*. With depth 2 then 0, 0 then 2
  and 1 then 0 over two bases, the median is 1, not 0.5
  (`OrfStatisticsTests.test_the_median_is_of_each_samples_mean_depth`).
- `depth_mean` is the group's depth summed, over (samples x length);
  `prevalence` the covered (sample, base) pairs over the same; and
  `union_breadth` the bases any sample covers, over the length. An ORF with
  no reads is 0, never NaN.
- **Contrast**, by `orf_contrast`, only with exactly two groups, A and B in
  sorted order: `log2((B / norm B + 0.05) / (A / norm A + 0.05))` on the
  per-ORF medians, where a group's norm is the median of its per-ORF medians
  over the genome's ORFs. Positive where B is higher. Scaling by the norm
  makes a group sequenced twice as deep throughout no different: the
  contrast is then exactly 0. The 0.05 (`CONTRAST_PSEUDOCOUNT`) keeps an ORF
  with no depth in one group finite.
- **A norm of 0** -- at least half the genome's ORFs have median depth 0 in
  that group -- leaves nothing to scale by: the genome's contrasts are NaN
  (NULL in the file). Only a run that asked for `--orf-contrast`
  (`warn_contrast`) is warned, naming the genome and the group; the column
  is written either way. On the `dp_*` fixture this is every genome, so
  contrast is tested on literal reads. On `example/` too: the median `Yes`
  sample has 2,701 alignments on G000154205 against 6,870 for `No`, and 94%
  of its ORFs have median depth 0.
- `OrfWindowInvarianceTests` checks the ORF table bit-identical for windows
  of 1, 2, 3, 7 and L bases, ORFs across window edges and the origin
  included, and against a Python oracle.

## Cost

Measured 2026-10-05 on an Apple-silicon laptop, `--threads 4`
(`localdocs/depth-plot/bench/bench_phase3.py`, which is not committed):

| Case | Time | Peak RSS | With an ORF per kb |
|---|---|---|---|
| Synthetic: 10 Mb, 300 samples, 1M reads | 23 s | 0.64 GiB | 31 s, 0.78 GiB |
| Specimen: 2 genomes (4.7 and 5.3 Mb), 49 samples, 430k reads | 5.8 s | 0.53 GiB | 7.3 s, 0.53 GiB |

About half the time is `np.quantile`'s partition, which is inherent to
per-base quantiles. ORFs add a per-sample prefix sum of every window. **miint's aggregation peaks at six to seven times the
window's array** (1.6 GiB for a 256 MiB window, whatever the thread count or
the way it is fetched), so `WINDOW_CELLS` is 2**22: 16 MiB of depth, about
110 MiB to compute. At 2**26 the synthetic case took the same time and peaked at
3.15 GiB.
