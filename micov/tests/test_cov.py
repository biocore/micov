import itertools
import unittest
from unittest import mock

import numpy as np

from micov import _cov
from micov._constants import (
    COLUMN_COVERED,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_PERCENT_COVERED,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_STOP,
)
from micov._cov import (
    compute_cumulative,
    get_covered,
    ordered_coverage,
    slice_positions,
)
from micov._miint import connection

# the curve functions take the dict-of-numpy-arrays that
# DuckDBPyRelation.fetchnumpy() returns, so strings arrive as object arrays
COVERAGE_COLUMNS = [
    (COLUMN_SAMPLE_ID, object),
    (COLUMN_GENOME_ID, object),
    (COLUMN_COVERED, np.uint32),
    (COLUMN_LENGTH, np.uint32),
    (COLUMN_PERCENT_COVERED, np.float64),
]
POSITION_COLUMNS = [
    (COLUMN_SAMPLE_ID, object),
    (COLUMN_GENOME_ID, object),
    (COLUMN_START, np.uint32),
    (COLUMN_STOP, np.uint32),
]


def table(columns, rows):
    """Build the dict-of-arrays a relation's fetchnumpy() would hand back."""
    by_column = list(zip(*rows, strict=True))
    return {
        name: np.array(values, dtype=dtype)
        for (name, dtype), values in zip(columns, by_column, strict=True)
    }


class CovTests(unittest.TestCase):
    def setUp(self):
        # `compute_cumulative` accumulates with miint's `cumulative_coverage`
        # aggregate since M7, so the curve functions need a live connection.
        # `ordered_coverage`, `slice_positions` and `get_covered` remain pure
        # numpy and do not use it.
        self.con = connection()
        self.addCleanup(self.con.close)

    def test_slice_positions(self):
        df = table(POSITION_COLUMNS,
                   [['S1', 'G1', 1, 10],
                    ['S1', 'G1', 10, 20],
                    ['S1', 'G1', 30, 40],
                    ['S1', 'G2', 100, 200],
                    ['S1', 'G2', 200, 300],
                    ['S1', 'G2', 300, 400],
                    ['S2', 'G1', 39, 49],
                    ['S2', 'G2', 109, 209]])
        obs = slice_positions(df, 'S1')

        # sample_id is projected away: the caller already knows the sample, and
        # the accumulation only ever needs the intervals
        self.assertEqual(list(obs), [COLUMN_GENOME_ID, COLUMN_START, COLUMN_STOP])
        self.assertEqual(obs[COLUMN_GENOME_ID].tolist(),
                         ['G1', 'G1', 'G1', 'G2', 'G2', 'G2'])
        self.assertEqual(obs[COLUMN_START].tolist(), [1, 10, 30, 100, 200, 300])
        self.assertEqual(obs[COLUMN_STOP].tolist(), [10, 20, 40, 200, 300, 400])

    def test_ordered_coverage(self):
        df = table(COVERAGE_COLUMNS,
                   [['S1', 'G1', 1, 100, 1.],
                    ['S1', 'G2', 15, 100, 15.],
                    ['S1', 'G3', 101, 1000, 10.1],
                    ['S2', 'G2', 6, 100, 6.],
                    ['S3', 'G3', 7, 1000, .7],
                    ['S3', 'G2', 8, 100, 8.],
                    ['S3', 'G1', 9, 100, 9.]])
        grp = table([(COLUMN_SAMPLE_ID, object), ('blah', object)],
                    [['S1', 'foo'],
                     ['S2', 'foo'],
                     ['S3', 'foo'],
                     ['S4', 'foo'],
                     ['S5', 'foo']])

        obs = ordered_coverage(df, grp, 'G2', 100)

        # S4 and S5 have no coverage of G2 at all, yet still occupy ranks 0 and
        # 1 -- that back-fill is what makes a curve comparable across groups of
        # different sizes, so it is the point of this test
        self.assertEqual(obs[COLUMN_SAMPLE_ID].tolist(),
                         ['S4', 'S5', 'S2', 'S3', 'S1'])
        self.assertEqual(obs[COLUMN_GENOME_ID].tolist(), ['G2'] * 5)
        self.assertEqual(obs[COLUMN_COVERED].tolist(), [0, 0, 6, 8, 15])
        self.assertEqual(obs[COLUMN_LENGTH].tolist(), [100] * 5)
        self.assertEqual(obs[COLUMN_PERCENT_COVERED].tolist(),
                         [0., 0., 6., 8., 15.])
        self.assertEqual(obs['x'].tolist(), [0, 1 / 5, 2 / 5, 3 / 5, 4 / 5])
        self.assertEqual(obs['x_unscaled'].tolist(), [0, 1, 2, 3, 4])

        # the metadata column rides along, for both covered and back-filled rows
        self.assertEqual(obs['blah'].tolist(), ['foo'] * 5)

        # ranks index a plot axis and are offset by group; they must stay
        # integral rather than being promoted to float
        self.assertEqual(obs['x_unscaled'].dtype, np.uint64)
        self.assertEqual(obs[COLUMN_COVERED].dtype, np.uint32)
        self.assertEqual(obs[COLUMN_LENGTH].dtype, np.uint32)

    def test_ordered_coverage_ignores_samples_outside_the_group(self):
        df = table(COVERAGE_COLUMNS,
                   [['S1', 'G2', 15, 100, 15.],
                    ['S2', 'G2', 6, 100, 6.]])
        grp = table([(COLUMN_SAMPLE_ID, object)], [['S2']])

        obs = ordered_coverage(df, grp, 'G2', 100)
        self.assertEqual(obs[COLUMN_SAMPLE_ID].tolist(), ['S2'])

    def test_compute_cumulative(self):
        df = table(COVERAGE_COLUMNS,
                   [['S1', 'G1', 1, 100, 1.],
                    ['S1', 'G2', 15, 100, 15.],
                    ['S1', 'G3', 101, 1000, 10.1],
                    ['S2', 'G2', 6, 100, 6.],
                    ['S3', 'G3', 7, 1000, .7],
                    ['S3', 'G2', 8, 100, 8.],
                    ['S3', 'G1', 9, 100, 9.]])
        pos = table(POSITION_COLUMNS,
                    [['S1', 'G2', 1, 16],
                     ['S2', 'G2', 14, 20],
                     ['S3', 'G2', 18, 26]])
        grp = table([(COLUMN_SAMPLE_ID, object), ('blah', object)],
                    [['S1', 'foo'],
                     ['S2', 'foo'],
                     ['S3', 'foo'],
                     ['S4', 'foo'],
                     ['S5', 'foo']])
        # G1's length is deliberately NOT G2's. `compute_cumulative` used to
        # read `lengths[COLUMN_LENGTH][0]` -- row 0, whatever genome that is --
        # regardless of which genome `target` named, and this fixture had 100
        # in both rows so it could not tell the two readings apart (R22).
        lengths = table([(COLUMN_GENOME_ID, object), (COLUMN_LENGTH, np.uint32)],
                        [['G1', 500],
                         ['G2', 100],
                         ['G3', 1000]])
        exp_x = [0, 1, 2, 3, 4]

        # S2 contributes [14, 20); S3 then adds [18, 26), which *overlaps* it,
        # so the pair accumulates to 12 rather than 14. Accumulating merged
        # breadth rather than summing per sample breadth is the whole point of
        # the curve.
        exp_y = [0., 0., 6., 12., 25.]

        obs_x, obs_y = compute_cumulative(self.con, df, grp, 'G2', pos, lengths)
        self.assertEqual(obs_x.tolist(), exp_x)
        self.assertEqual(obs_y, exp_y)

    def test_compute_cumulative_uses_the_targets_own_length(self):
        """R22: the denominator must follow `target`, not row 0 of `lengths`.

        Asserted on the call rather than on the curve because the value is
        *inert* in the returned data: it only fills the `length` column that
        `ordered_coverage` back-fills for zero-coverage samples, and nothing
        reads that. There is therefore no black-box route to this, and a
        fixture alone cannot catch it -- which is exactly why the wrong
        reading survived. `coverage_curve` only ever passes a single-row
        `lengths`, so the two readings coincide in production today; this
        pins the caller so they still agree if that ever stops being true.
        """
        df = table(COVERAGE_COLUMNS, [['S1', 'G2', 15, 100, 15.]])
        pos = table(POSITION_COLUMNS, [['S1', 'G2', 1, 16]])
        grp = table([(COLUMN_SAMPLE_ID, object)], [['S1']])
        lengths = table([(COLUMN_GENOME_ID, object), (COLUMN_LENGTH, np.uint32)],
                        [['G1', 500],
                         ['G2', 100]])

        with mock.patch.object(_cov, 'ordered_coverage',
                               wraps=_cov.ordered_coverage) as spy:
            compute_cumulative(self.con, df, grp, 'G2', pos, lengths)

        self.assertEqual(int(spy.call_args.args[3]), 100)

    def test_compute_cumulative_keeps_zero_coverage_samples_in_rank(self):
        """A sample with no coverage still occupies a rank position.

        This is what `ordered_coverage`'s back-fill exists for: group size has
        to stay honest, because the x axis is "within group sample rank" and
        the KS test compares curves of that length. Dropping the uncovered
        samples would silently shorten the curve and shift every other sample
        left.

        The accumulation is an aggregate over interval rows, and a sample with
        no coverage contributes no rows -- so this only holds if the ranks are
        supplied from a roster rather than from the intervals themselves.
        """
        df = table(COVERAGE_COLUMNS, [['S3', 'G1', 10, 100, 10.]])
        pos = table(POSITION_COLUMNS, [['S3', 'G1', 0, 10]])
        grp = table([(COLUMN_SAMPLE_ID, object)],
                    [['S1'], ['S2'], ['S3'], ['S4']])
        lengths = table([(COLUMN_GENOME_ID, object), (COLUMN_LENGTH, np.uint32)],
                        [['G1', 100]])

        obs_x, obs_y = compute_cumulative(self.con, df, grp, 'G1', pos, lengths)

        # four samples in, four points out -- three of them flat at zero
        self.assertEqual(obs_x.tolist(), [0, 1, 2, 3])
        self.assertEqual(obs_y, [0., 0., 0., 10.])

    def test_compute_cumulative_breaks_breadth_ties_by_sample_id(self):
        """Ties are broken by `sample_id`, so the curve ignores input order.

        Row order is not stable on real data: `View.coverages()` comes from a
        parallel Parquet scan and arrives in a different order run to run, so a
        row-order tie-break made identical invocations disagree (R20a). On a
        100-sample study, 995 of 9,337 position files and 26 KS files moved
        between runs for this reason alone.

        Three samples tied at 10%, of which S1 [0,10) and S2 [5,15) overlap
        and S3 [50,60) does not. Rank order therefore changes the *middle* of
        the curve: adjacent S1/S2 accumulate to 15, while S3 between them
        gives 20. Ranked S1,S2,S3 the curve is [10, 15, 25] for every input
        order.
        """
        spans = {'S1': (0, 10), 'S2': (5, 15), 'S3': (50, 60)}
        lengths = table([(COLUMN_GENOME_ID, object), (COLUMN_LENGTH, np.uint32)],
                        [['G1', 100]])
        for order in itertools.permutations(spans):
            with self.subTest(order=order):
                df = table(COVERAGE_COLUMNS,
                           [[s, 'G1', 10, 100, 10.] for s in order])
                pos = table(POSITION_COLUMNS,
                            [[s, 'G1', *spans[s]] for s in order])
                grp = table([(COLUMN_SAMPLE_ID, object)], [[s] for s in order])

                _, obs_y = compute_cumulative(self.con, df, grp, 'G1', pos,
                                              lengths)
                self.assertEqual(obs_y, [10., 15., 25.])

    def test_get_covered(self):
        test = np.array([(1, 2, 3), (10, 20, 30)])
        exp = [[(1, 2), (1, 3)], [(10, 20), (10, 30)]]
        obs = get_covered(test)
        self.assertEqual(obs, exp)


class IntervalMergeTests(unittest.TestCase):
    """micov's interval merge, which is now entirely miint's.

    These cases were written against `merge_intervals`, the numpy merge on the
    accumulation path, which was itself checked case-for-case against
    `compress` -- the numba+polars implementation the published coverage
    values came from -- until M4 deleted that path with Qiita support. M7
    deleted `merge_intervals` too: `cumulative_coverage` accumulates now, so
    it had no callers left.

    The cases moved onto `compress_intervals` rather than being deleted with
    it. That is the primitive `_io.compress_alignments` and `_view`'s region
    clip both use, so it *is* micov's merge, and it returns intervals -- which
    matters, because `cumulative_coverage` returns only covered counts and a
    count cannot distinguish a touching merge from no merge at all
    (`[400,500)` + `[500,505)` is 105 bases either way). The property these
    defend is structural, so they need the structural entry point.
    """

    def _reference(self, rows):
        """Merge by sort-and-sweep, the obvious way, with no numpy.

        Touching intervals collapse (`stop == next start` -> one interval),
        which is micov's convention and the one `compress` implemented.
        """
        merged = []
        for start, stop in sorted(rows):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
            else:
                merged.append((start, stop))
        return merged

    def _merge(self, rows):
        if rows:
            values = ", ".join(
                f"({start}::UINTEGER, {stop}::UINTEGER)" for start, stop in rows
            )
            source = f"SELECT * FROM (VALUES {values}) t(start, stop)"
        else:
            source = ("SELECT 0::UINTEGER AS start, 0::UINTEGER AS stop "
                      "WHERE false")
        merged = self.con.sql(
            f"SELECT compress_intervals(start, stop) FROM ({source})"
        ).fetchone()[0]

        # the aggregate returns NULL rather than an empty list for no input
        if merged is None:
            return []
        return [(row["start"], row["stop"]) for row in merged]

    def setUp(self):
        self.con = connection()
        self.addCleanup(self.con.close)

    def test_touching_intervals_merge(self):
        # the case the docstring gets wrong, and the one breadth depends on
        self.assertEqual(self._merge([(400, 500), (500, 505)]), [(400, 505)])

    def test_overlapping_and_nested_merge(self):
        self.assertEqual(self._merge([(600, 700), (650, 800)]), [(600, 800)])
        self.assertEqual(self._merge([(900, 1000), (920, 950)]), [(900, 1000)])

    def test_disjoint_intervals_are_kept(self):
        self.assertEqual(self._merge([(10, 20), (50, 60)]), [(10, 20), (50, 60)])

    def test_unordered_input(self):
        self.assertEqual(self._merge([(50, 60), (10, 20), (15, 30)]),
                         [(10, 30), (50, 60)])

    def test_empty(self):
        self.assertEqual(self._merge([]), [])

    def test_breadth_excludes_no_plus_one(self):
        # breadth is sum(stop - start) over merged intervals, with no +1 --
        # the convention the published coverage values are computed under
        merged = self._merge([(0, 5), (10, 20)])
        self.assertEqual(sum(stop - start for start, stop in merged), 15)

    def test_matches_the_reference_on_many_random_cases(self):
        rng = np.random.default_rng(42)
        for case in range(200):
            n = int(rng.integers(1, 12))
            starts = rng.integers(0, 60, size=n)
            widths = rng.integers(1, 15, size=n)
            rows = list(zip(starts.tolist(),
                            (starts + widths).tolist(), strict=True))
            with self.subTest(case=case, rows=rows):
                self.assertEqual(self._merge(rows), self._reference(rows))


if __name__ == '__main__':
    unittest.main()
