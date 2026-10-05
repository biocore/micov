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
  on a used genome: `ValueError`.
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
  stops (count at x = #{start ≤ x} − #{stop ≤ x}). `genome_bins` sorts them
  once per genome.

## Group statistics and bins: `_depth.genome_bins`

`genome_bins(con, depth_view, genome_id, length, edge_sets)` makes one pass
of windows and fills every bin set at once: the overview,
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

## Cost

Measured 2026-10-05 on an Apple-silicon laptop, `--threads 4`
(`localdocs/depth-plot/bench/bench_phase3.py`, which is not committed):

| Case | Time | Peak RSS |
|---|---|---|
| Synthetic: 10 Mb, 300 samples, 1M reads | 22.5 s | 0.61 GiB |
| Specimen: 2 genomes (4.7 and 5.3 Mb), 49 samples, 430k reads | 5.6 s | 0.50 GiB |

About half the time is `np.quantile`'s partition, which is inherent to
per-base quantiles. **miint's aggregation peaks at six to seven times the
window's array** (1.6 GiB for a 256 MiB window, whatever the thread count or
the way it is fetched), so `WINDOW_CELLS` is 2**22: 16 MiB of depth, about
110 MiB to compute. At 2**26 the synthetic case took the same time and peaked at
3.15 GiB.
