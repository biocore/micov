import unittest

import numpy as np

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
    merge_intervals,
    ordered_coverage,
    slice_positions,
)

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
        lengths = table([(COLUMN_GENOME_ID, object), (COLUMN_LENGTH, np.uint32)],
                        [['G1', 100],
                         ['G2', 100],
                         ['G3', 1000]])
        exp_x = [0, 1, 2, 3, 4]

        # S2 contributes [14, 20); S3 then adds [18, 26), which *overlaps* it,
        # so the pair accumulates to 12 rather than 14. Accumulating merged
        # breadth rather than summing per sample breadth is the whole point of
        # the curve.
        exp_y = [0., 0., 6., 12., 25.]

        obs_x, obs_y = compute_cumulative(df, grp, 'G2', pos, lengths)
        self.assertEqual(obs_x.tolist(), exp_x)
        self.assertEqual(obs_y, exp_y)

    def test_get_covered(self):
        test = np.array([(1, 2, 3), (10, 20, 30)])
        exp = [[(1, 2), (1, 3)], [(10, 20), (10, 30)]]
        obs = get_covered(test)
        self.assertEqual(obs, exp)


class MergeIntervalsTests(unittest.TestCase):
    """merge_intervals is now the only interval merge micov has.

    It was written against `compress`, the numba+polars implementation the
    published coverage values came from, and was checked case-for-case against
    it until M4 deleted that path along with Qiita support. The randomized
    check below now runs against a deliberately naive reference instead: the
    property it defends -- that touching intervals collapse, and that breadth
    is therefore what the paper reports -- outlived the implementation it was
    originally cross-checked against.
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
        starts = np.array([s for s, _ in rows], dtype=np.uint32)
        stops = np.array([e for _, e in rows], dtype=np.uint32)
        merged_starts, merged_stops = merge_intervals(starts, stops)
        return list(zip(merged_starts.tolist(), merged_stops.tolist(), strict=True))

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
        starts, stops = merge_intervals(np.array([], dtype=np.uint32),
                                        np.array([], dtype=np.uint32))
        self.assertEqual(len(starts), 0)
        self.assertEqual(len(stops), 0)

    def test_breadth_excludes_no_plus_one(self):
        # breadth is sum(stop - start) over merged intervals, with no +1 --
        # the convention the published coverage values are computed under
        starts, stops = merge_intervals(np.array([0, 10], dtype=np.uint32),
                                        np.array([5, 20], dtype=np.uint32))
        self.assertEqual(int((stops - starts).sum()), 15)

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
