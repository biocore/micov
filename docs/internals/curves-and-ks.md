# Curves, Monte Carlo, position plots and KS

**Read this when** you touch `_cov.py`, or `coverage_curve`, `add_monte`,
`position_plot` or `ks_table` in `_plot.py`. These produce the published
numbers. Read [traps.md](traps.md) as well.

All functions here take **dict-of-numpy tables** (see
[architecture.md](architecture.md)). Inside the per-genome loop they receive
only one genome's rows.

## Ranking: `_cov.ordered_coverage(coverage, grp, target, length)`

1. Keep `coverage` rows for `target` whose sample is in `grp`.
2. **Back-fill** every `grp` sample that has no row, with `covered=0` and
   `percent_covered=0.0`. The group's other metadata columns are carried
   along for both kinds of sample.
3. Sort by `percent_covered` ascending, **ties broken by `sample_id`**
   (`np.lexsort`).
4. Add `x_unscaled = 0..n-1` (uint64) and `x = rank / n`.

The tie-break was input row order until `e326f91`. Row order is not stable:
`View.coverages()` is a parallel Parquet scan. On a real 100-sample study,
995 of 9,337 position files and 26 KS files changed between identical runs
because of it. `test_cov.test_compute_cumulative_breaks_breadth_ties_by_sample_id`
pins the new behaviour over every input permutation.

## Accumulation

The functions nest: `compute_cumulative`, for one group, calls
`cumulative_curves`, for several groups, which calls `cumulative_covered`,
the SQL.

- **`cumulative_covered(con, n_iterations, n_ranks, iterations, ranks,
  starts, stops)`** registers the intervals as `micov_curve_input`. It
  generates the full roster `range(n_iterations) × range(n_ranks)` in SQL and
  LEFT JOINs the intervals onto it, then runs
  `UNNEST(cumulative_coverage(rank::INTEGER, start, stop))` grouped by
  iteration.
  - The roster matters because miint needs contiguous ranks `0..N-1`. A
    sample with no intervals would otherwise leave a gap and **silently
    shorten the curve**.
  - Results are sorted by `(iteration, rank)` and reshaped positionally to
    `(n_iterations, n_ranks)`.
- **`cumulative_curves(con, coverage, groups, target, target_positions,
  lengths)`** ranks each group with `ordered_coverage` and maps intervals to
  ranks (`_rank_intervals`). It runs **every group in one aggregate call**,
  which is what makes Monte Carlo cheap.
  - All groups must be the same size, or it raises `ValueError`, because the
    roster has one width.
  - Returns `(ordered_tables, percents)`, with
    `percents = (covered / length) * 100` in that order. The goldens carry
    that double.
- **`compute_cumulative(...)`** returns `(x_unscaled, list_of_y)` for one
  group, or `(None, None)` if the group is empty.

## Curves: `_plot.coverage_curve`

`coverage_curve` is called twice per genome: `accumulate=False`, then
`accumulate=True`.

- **The metadata is restricted to samples with a coverage row for this
  genome.** A group is plotted only if it has **at least `min_group_size`
  (10)** such samples.
- **Only the first 10 metadata values (sorted) are considered.** The loop is
  `zip(np.unique(values), range(10), strict=False)`, so an 11th group is
  silently skipped. This is long-standing behaviour.
- **y values:** non-cumulative uses each sample's own `percent_covered` in
  rank order; cumulative uses `compute_cumulative`.
- **`--percentile`** plots x as `x × 100 / (n - 1)`. It changes PNGs only.
- **If no group qualifies**, the figure is closed and nothing is written.
  Most genomes in a real study end here.
- **On the cumulative call**, the `.ks.csv` is written from the curves dict:
  group value → y values, plus the Monte Carlo median when there is one.

## Monte Carlo: `_plot.add_monte`

- **Draw size:** `max_x + 1` samples, where `max_x` is the largest rank of
  any plotted group. That is the size of the largest plotted group.
- **Sample pool:**
  - `focused` draws from the samples with coverage of this genome.
  - `unfocused` draws from `sample_universe`, every sample with coverage of
    any genome. It is computed once in `per_sample_plots`, because one
    genome's rows cannot tell it.
- **Each iteration** takes `rng.permutation(pool)[:k]` to pick *which*
  samples take part. `np.isin` then keeps them in pool order, and
  `ordered_coverage` ranks them by breadth.
  - `rng = np.random.default_rng()` is **unseeded**, so Monte Carlo output
    does not reproduce between runs, and the golden suite treats those rows
    as nondeterministic.
  - **Keep the selection in numpy.** Moving it to SQL would change the RNG
    stream.
- **Curves:** cumulative runs all iterations through one `cumulative_curves`
  call; non-cumulative is a per-iteration breadth lookup.
- **Plotted:** the median ± std envelope.
- **Returns** the label `Monte Carlo {type} (n={k})` and the median curve.
  The median is what the KS tests compare against.

## KS: `_plot.ks_2samp` and `_plot.ks_table`

- **`ks_2samp(con, a, b)`** is miint's
  `ks_2samp(?::DOUBLE[], ?::DOUBLE[])`. Its inputs are the two curves' **y
  values**, treated as samples. It returns `(statistic, pvalue)`.
  - The statistic is the exact lattice value `h / lcm(n1, n2)`, and matches
    the scipy 1.17.1 goldens **bit for bit**.
  - The p-value is the exact two-sided one, derived independently, and
    agrees to 1–4 ULP.
  - miint implements only the exact method and **raises above 10,000
    observations per sample**, i.e. a metadata group of more than 10,000
    samples. scipy used to fall back to an approximation there.
- **`ks_table(con, curves, monte_label=None)`** compares every unordered
  pair in curve order: groups in sorted value order, then Monte Carlo last.
  - `m` is the number of pairs that do not involve `monte_label`.
  - Group pairs get `min(1.0, p × m)`; Monte Carlo pairs get `""`.
  - **The Monte Carlo curve is identified by the label `add_monte` returned,
    not by text prefix**, so a metadata value that starts "Monte Carlo" is
    still corrected.
  - `test_plot.KsTableTests` pins all of this.

## Position plots: `_plot.position_plot`

Called twice per genome: `scale=None`, then `scale=10000`.

**Layout:**

1. Metadata is restricted to samples with positions on this genome.
2. Groups are the sorted unique values, and a group's sorted index is its
   colour (`C{index}`). Values are text, so `"30" < "5"`. A blank value is
   masked, and `np.unique` sorts a mask as `"?"`: after the digits, before
   the letters.
3. Groups are laid out **smallest first**, with ties kept in value order.
   With `sort_by_value` (`--sort-by-metadata-value`) they are laid out by
   value instead: values that parse as numbers, numerically, then other text
   in sorted order, then blanks. Colours do not change, so a group keeps the
   colour it has in the curve plots.
4. Within a group, samples are ranked by `ordered_coverage`. With **exactly
   two groups** the second is drawn in reverse, so the two mirror each other.
5. `ymin` and `ymax` are `feature_metadata`'s `start` and `stop`. In genome
   mode those are 0 and the genome length; in region mode, the region.

**What the scaled plot measures.** The scaled plot and its `.tsv.gz` record,
per sample, the left edge of every bucket that one of its intervals
overlaps.

- **Bucket width** is `(ymax - ymin) / scale`, but never less than
  `_plot.MIN_BUCKET_WIDTH` (100bp). With `scale=10000`, a genome or region of
  1Mb or more gets 10,000 buckets with `np.linspace` edges, which are the
  edges `np.histogram` produced, so the published `y` values hold. Anything
  shorter gets buckets exactly 100bp wide from `ymin`, and the last bucket
  holds the remainder.
- **Marking** is by overlap: `[start, stop)` marks every bucket from the one
  holding `start` to the last one whose left edge is before `stop`. Fractional
  edges can fall inside an interval's last base, so asking which bucket holds
  `stop - 1` would drop a bucket the interval overlaps.
- **An interval past `ymax`** is plotted up to `ymax`. `example/` has one,
  1bp past the end of G000436435.
- Before 0.0.1-dev, buckets were always 1/10,000 of the genome and only the
  buckets holding an interval's `start` and `stop` were marked. On a 145kb
  genome that is 15bp buckets, and half the covered buckets went unmarked.
  The `example/` goldens gained 836 and 130 rows, and lost none.
- `test_plot.ScaledPositionPlotTests` pins all of this.

**Group size.** There is no minimum, so every genome gets position plots.
