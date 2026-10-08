# depth-plot

**Read this when** you touch `_depth.py` or `_depth_plot.py`, or
`depth-plot`'s readers and writer in `_io.py`. [commands.md](commands.md) has
the command and its errors, [data-formats.md](data-formats.md) the input
formats and the per-ORF table.

`depth-plot` draws, along each genome, per-base **depth** from one alignment
layer and **breadth** from another (say metatranscriptomic over
metagenomic), for each group of a metadata column. Both layers are
`read_alignments` output with a `sample_id` column, and may be the same file.

## Which samples and genomes: `_depth.intersect_layers`

- A sample is used if it is in the metadata and has an aligned read
  (`_io.ALIGNED_ROWS`) in **both** layers. A genome is used if it is in the
  features and **a used sample** has an aligned read on it in both layers:
  reads from samples without metadata do not count, or a genome only they
  reach would be drawn as flat zeros.
- Whatever the metadata or features name but this leaves out is warned about
  by name, with the layer it lacks (`_in_both`). A sample with no metadata,
  or a genome not in the features, was left out by the user, and is not.
- No sample or no genome left, more than `_plot.MAX_GROUPS` groups, an
  aligned read starting beyond its genome's `length`, or, with ORFs, no ORF
  on any used genome, an ORF on one without an `ID`, or an ORF on one that
  does not span a base of it (a start before 1, a stop not past its start,
  or beyond the genome): `ValueError`. Only a circular genome's ORF may run
  past its end. ORFs on genomes not used are not checked, so a
  database-wide GFF is fine. All of this is checked before any genome is
  computed.
- It creates `depth_roster` (`sample_id`, a dense `sample_idx` in `sample_id`
  order, `group_name`) and `depth_genomes`.

`test_depth.IntersectLayersTests` pins these on the `dp_*` fixtures.

## Per-base depth, in windows

A genome's depth is computed in windows of `window_size(n, L) = max(1,
min(L, WINDOW_CELLS // n))` bases, so memory is bounded by the number of
samples, not the genome's length.

- **`stage_depth_reads`**, once a run, copies what can add depth -- the used
  samples' aligned reads on the used genomes -- into the temp table
  `depth_reads`: `sample_idx`, `reference`, UINTEGER `position` and
  `stop_position`, `cigar`, ordered by genome and position; see
  [the run](#the-run-_depth_plotdepth_plots).
- **`stage_depth`** copies one genome's rows of it into `depth_alignments`,
  sorted by position, for the window queries.
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

`genome_statistics(con, genome_id, length, edge_sets, orfs=None)` makes one
pass of windows over `depth_reads` and fills every bin set at once,
returning `(bins, orf_table, contrast)`. The bins are the overview,
`display_bin_edges(1, L + 1, overview_bin_bp(L))`, and each detail region,
`display_bin_edges(start, stop, detail_bin_bp(start, stop))`, which has at
most `DETAIL_MAX_BINS` bins (single bases for a region up to 1,500 bp).
`overview_bin_bp` keeps each overview row to `OVERVIEW_ROW_BINS` (2,000)
bins: 1 kb on any genome of 2 Mb or more, 9 bp on a 16.6 kb mitochondrion,
25 bp on a 50 kb phage.

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
  integer bin sums, carried across windows and divided once at the end. `WindowInvarianceTests` checks bit-identical bins for windows of 1,
  2, 3, 7 and L bases, and agreement with a Python CIGAR-walking oracle. Do
  not divide inside the window loop.
- **The table**, one per edge set: `group`, `bin_start`, `bin_stop` (1-based,
  half-open), `q1`, `median`, `q3`, `mean`, `prevalence` (float64), `union`
  (bool), a row per group (sorted) and bin.

Every span set -- the bins and the ORFs' segments -- is summed the same way,
by `_add_span_sums`, which touches only the spans a window overlaps: it sums
the bases between the spans' ends (`np.add.reduceat`), then accumulates
those. A prefix sum of every base, for every sample, was most of the time
ORFs added.

## Per-ORF statistics

`genome_orfs(con, genome_id, attributes=False)` gives a genome's ORFs by
position, and with `attributes` each one's GFF attributes as a dict, which
only `--highlight` and `--orf-color-by` read: the dicts cost about 75 ms a
genome of 10,000 ORFs. With `orfs`, `genome_statistics` also returns the
per-ORF table, and each ORF's contrast once, for its colour. The run starts
the table with `_io.start_orf_table`; `_io.add_orf_table` copies each
genome's into DuckDB as it is done, as numpy strings rather than the
objects `fetchnumpy` gives, which took DuckDB about a second a genome; and
`_io.write_orf_table` writes it once ([data-formats.md](data-formats.md)).

Each GFF line is an ORF of its own. A feature written on several lines
that share an `ID` -- NCBI writes a frameshifted CDS that way -- gives a set
of rows per line, so the table's key is `genome_id`, `orf_id`, `start` and
`group`, not the `ID` alone.

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

## Drawing: `_depth_plot`

`linear_plot` draws one genome's overview, `detail_plot` one region, and
`circular_plot` a circular genome's ring, each from that genome's tables
only. x is `position − 1`, so base p spans [p − 1, p) and a bin's edges are
its coordinates less one.

- **Layout.** Each row has depth on top, the ORFs on the genome line, and
  breadth below. Up to `OVERLAY_MAX_GROUPS` (3) groups share one set of
  axes; more get a lane each, every lane on the same depth scale. The
  overview's rows are `overview_row_bp(L)` wide: the whole genome, up to
  `ROW_BP` (2 Mb). A larger genome wraps into rows of exactly 2 Mb, so all
  of them share one resolution, and the last row is masked past the
  genome's end. A shorter one, such as a mitochondrion, is one row of its
  own length rather than a sliver, at its own finer bins; up to 150 kb its
  ORFs are arrows.
- **Depth.** Per group, the IQR as a filled `stairs` from Q1 to Q3, the
  median as a line in `_plot.group_style`'s colour and dash, and the mean
  dotted. The axis is symlog: linear below `DEPTH_LINTHRESH` (2), with
  `symlog_ticks`, and shared by every row. The label says "alignments",
  because secondary alignments count.
- **Breadth.** The union as segments in a strip above 0 (`true_runs` of the
  bins' `union`), and prevalence hanging down from 0 to 1.
- **ORFs.** Boxes, or arrows when the panel spans at most `ARROW_MAX_BP`
  (150 kb). + above the line, − below, `.` across it. An ORF across a
  circular genome's origin is drawn as its two parts, both boxes.
  `orf_track` decides each ORF's fill:
  - neutral grey, highlighted ORFs in ink;
  - `--orf-color-by`: the top three values (ties to the value that sorts
    first) in `ORF_CATEGORY_COLORS` plus a grey Other, with a legend;
  - `--orf-contrast`: `contrast_colors`, from the first group's colour
    through grey to the second's, clipped at ±3 (8-fold), with a scale.
  In the two colour modes a highlighted ORF keeps its fill and gains an ink
  outline.
- **Highlights** (`parse_highlight`): `KEY=VALUE` matches exactly and
  `KEY~REGEX` searches; the first operator wins. KEY is `type` or `strand`,
  else a GFF attribute. `highlight_masks` gives a mask per expression: an
  ORF any matches is highlighted, and an expression that matches none is
  told of. A highlighted ORF gets a faint band through depth and breadth
  (merged within a point; under both parts of one across the origin), and a
  label in one of two lanes per side (`assign_label_lanes`); labels that fit
  nowhere are counted as "+N unlabelled". The bands are one artist per
  axes: one each was 0.7 s for 800 of them.
- **Regions** are shaded on the overview where they fall; each gets its own
  `detail_plot`, at its own bins and depth scale.
- **Files** (`plot_path`): `{output}.{target_name}.{genome}.{variable}.depth-plot.png`,
  `...depth-plot-circular.png` and `...depth-plot-detail-{start}-{stop}.png`,
  like micov's other plots.
- **Guards.** Masked values, or bins that do not tile the same span for
  every group, raise before a figure opens. Drawing happens inside
  `rc_context(STYLE)` on a `Figure` that pyplot never tracks.

### The ring: `circular_plot`

A circular genome also gets a ring, from the overview's bins: the linear
plot's anatomy bent round, the mirror layout approved from the mockups.

- **Outside in** (`circular_layout`): the highlighted ORFs' labels, the
  coordinates (labelled along the ring, inside it, by `tangential_text`),
  depth, the ORFs either side of the backbone, a union arc per group, then
  prevalence from 0 hanging inward to 1. Polar axes run clockwise from the
  top (`theta`); an ORF across the origin is one shape, past 2π, which the
  axes wrap.
- **Depth** is on the linear plot's scale: `radial_symlog` puts a depth the
  same fraction of the way up the ring's depth band as up the linear depth
  axis, headroom (`DEPTH_HEADROOM`) included.
- **Chords.** Polar axes join points with straight lines, so every arc is
  traced in steps of at most `MAX_ARC` (half a degree) by `densify`, and
  a step function by `polar_steps`. A ring given by its two ends alone is
  not drawn at all.
- **ORFs** keep the linear plot's colours, outlines and shapes
  (`orf_polygons`), arrows included up to `ARROW_MAX_BP`, bent round all at
  once; only an ORF longer than `MAX_ARC` is traced. Highlights get a wedge
  from prevalence 1 out to the top of depth, merged within a point.
- **Labels** (`place_ring_labels`) are spread round the ring by least
  squares (`spread_angles`), the ring cut at its widest gap so labels
  either side of the top stay neighbours. A label moved more than
  `RING_LABEL_SHIFT` label widths is dropped, farthest first, and counted
  as "+N unlabelled"; more labels than the ring holds are thinned first, so
  thousands cost milliseconds. `radial_text` reads them outward, never
  upside down.
- **At most three groups.** A ring can only overlay, so more raise
  `ValueError` before drawing; the linear plot gives them lanes. (The
  command will skip the ring with a warning.)

`test_depth_plot` reads each drawing back by artist gid (`depth:{row}`,
`median:{group}`, `orfs:+`, `past-end`, `region:{i}` and so on) with
`Figure.savefig` patched.

## The run: `_depth_plot.depth_plots`

`cli.depth_plot` loads the inputs through the `_io` readers and hands the
connection to `depth_plots`, which:

1. **Refuses what it can, first:** `check_orf_options` (ORF options without
   `--orfs`, or both colourings; the command refuses the same as usage
   errors), `intersect_layers`, then `check_orf_mode` on the number of
   groups. Nothing is computed or written before all pass.
2. **Warns once:** no rings with four or more groups (naming the circular
   genomes); plotted genomes with no ORF, a bare track, most likely a
   seqid that is not the genome_id.
3. **Stages once:** `stage_breadth`, and `stage_depth_reads`, which copies the
   plotted samples' aligned reads on the plotted genomes into `depth_reads`,
   ordered by genome and position. `_io.load_orfs` likewise stores the ORFs
   by genome. `_io.start_orf_table` makes the per-ORF table afresh, so a
   second run on one connection writes only its own rows.
4. **Per genome**, in `genome_id` order: bins for the overview
   (`overview_bin_bp`) and each region (`detail_bin_bp`), one
   `genome_statistics` pass over `depth_reads`, the ORF track
   (`highlight_masks`, `orf_track`, and the contrast `genome_statistics`
   gives per ORF), then
   `linear_plot`, `circular_plot` if circular and at most three groups, and
   a `detail_plot` per region. Titles are `{name} ({genome}) · {length} bp ·
   {topology} · {variable}`, names from `_io.target_names_query`.
5. **Afterwards:** warns of a `--highlight` that matched no ORF and an
   `--orf-color-by` attribute no ORF has, and writes the per-ORF table.

**Why stage by genome.** Each genome's queries filter on `genome_id`. On
the layer itself, or an unordered ORF table, every genome scans the whole
input, so each one costs more the larger the input is; on a table ordered by
genome, DuckDB's row-group min/max skip the others. With 500 genomes
(`localdocs/depth-plot/many/bench.py`, not committed):

| Reads in the layer | Per genome, from the layer | Per genome, from `depth_reads` |
|---|---|---|
| 0.5M | 3.5 ms | 2.5 ms |
| 5M | 12.0 ms | 3.1 ms |
| 20M | 41.9 ms | 4.0 ms |

Staging 20M reads takes 0.6 s once. For the same reason each genome's
per-ORF table goes into DuckDB as it is computed, which can spill to disk,
rather than into a list held until the end.
`test_depth_plot.ManyGenomesTests` runs 300 genomes and checks that each
plot gets its own genome's bins; `DepthPlotsTests` checks each plot's bins
and ORFs on the fixture.

## Cost

Measured on an Apple-silicon laptop, `--threads 4`
(`localdocs/depth-plot/bench/bench_phase3.py`, which is not committed); the
synthetic case 2026-10-07, the specimen 2026-10-05, before the ORF sums
were made cheaper:

| Case | Time | Peak RSS | With an ORF per kb |
|---|---|---|---|
| Synthetic: 10 Mb, 300 samples, 1M reads | 23 s | 0.6-0.75 GiB | 24 s, 0.84 GiB (31 s before) |
| Specimen: 2 genomes (4.7 and 5.3 Mb), 49 samples, 430k reads | 5.8 s | 0.53 GiB | 7.3 s, 0.53 GiB |

The command on the specimen (both genomes, 49 samples, synthetic ORFs on
one, highlights, a ring and a detail panel) took 9.4 s and peaked at
0.66 GiB, 2026-10-08; with two groups and `--orf-color-by` or
`--orf-contrast`, 7.3 s and 0.68-0.71 GiB.

About half the time is `np.quantile`'s partition, which is inherent to
per-base quantiles. **miint's aggregation peaks at six to seven times the
window's array** (1.6 GiB for a 256 MiB window, whatever the thread count or
the way it is fetched), so `WINDOW_CELLS` is 2**22: 16 MiB of depth, about
110 MiB to compute. At 2**26 the synthetic case took the same time and peaked at
3.15 GiB.
