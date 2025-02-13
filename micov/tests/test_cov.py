import unittest

import numpy as np
import polars as pl
import polars.testing as plt

from micov._constants import (
    BED_COV_SCHEMA,
    COLUMN_COVERED,
    COLUMN_COVERED_DTYPE,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_LENGTH_DTYPE,
    COLUMN_PERCENT_COVERED,
    COLUMN_SAMPLE_ID,
    COLUMN_START,
    COLUMN_START_DTYPE,
    COLUMN_STOP,
    COLUMN_STOP_DTYPE,
    GENOME_COVERAGE_SCHEMA,
    GENOME_LENGTH_SCHEMA,
)
from micov._cov import (
    compress,
    compute_cumulative,
    coverage_percent,
    get_covered,
    ordered_coverage,
    slice_positions,
)


class CovTests(unittest.TestCase):
    def test_compress(self):
        exp = pl.DataFrame([['G123', 10, 50],
                            ['G123', 51, 89],
                            ['G123', 90, 100],
                            ['G123', 101, 110],
                            ['G456', 200, 300],
                            ['G456', 400, 505]],
                           orient='row',
                           schema=BED_COV_SCHEMA.dtypes_flat)
        data = pl.DataFrame([['G123', 11, 50],
                             ['G123', 20, 30],
                             ['G456', 200, 299],
                             ['G123', 10, 12],
                             ['G456', 201, 300],
                             ['G123', 90, 100],
                             ['G123', 51, 89],
                             ['G123', 101, 110],
                             ['G456', 400, 500],
                             ['G456', 500, 505]],
                            orient='row',
                            schema=BED_COV_SCHEMA.dtypes_flat)
        obs = compress(data).sort(COLUMN_GENOME_ID).sort(COLUMN_START)
        plt.assert_frame_equal(obs, exp)

    def test_coverage_percent(self):
        data = pl.DataFrame([['G123', 11, 50],
                             ['G456', 200, 299],
                             ['G123', 90, 100],
                             ['G456', 400, 500]],
                            orient='row',
                            schema=BED_COV_SCHEMA.dtypes_flat)
        lengths = pl.DataFrame([['G123', 100],
                                ['G456', 1000],
                                ['G789', 500]],
                               orient='row',
                               schema=GENOME_LENGTH_SCHEMA.dtypes_flat)

        g123_covered = (50 - 11) + (100 - 90)
        g456_covered = (299 - 200) + (500 - 400)
        exp = pl.DataFrame([['G123', g123_covered, 100, (g123_covered / 100) * 100],
                            ['G456', g456_covered, 1000, (g456_covered / 1000) * 100]],
                           orient='row',
                           schema=GENOME_COVERAGE_SCHEMA.dtypes_flat)

        obs = coverage_percent(data, lengths).sort(COLUMN_GENOME_ID).collect()
        plt.assert_frame_equal(obs, exp)

    def test_slice_positions(self):
        df = pl.DataFrame([['S1', 'G1', 1, 10],
                           ['S1', 'G1', 10, 20],
                           ['S1', 'G1', 30, 40],
                           ['S1', 'G2', 100, 200],
                           ['S1', 'G2', 200, 300],
                           ['S1', 'G2', 300, 400],
                           ['S2', 'G1', 39, 49],
                           ['S2', 'G2', 109, 209]],
                          orient='row', schema=[(COLUMN_SAMPLE_ID, str),
                                                (COLUMN_GENOME_ID, str),
                                                (COLUMN_START, int),
                                                (COLUMN_STOP, int)])
        exp = pl.LazyFrame([['G1', 1, 10],
                            ['G1', 10, 20],
                            ['G1', 30, 40],
                            ['G2', 100, 200],
                            ['G2', 200, 300],
                            ['G2', 300, 400]],
                           orient='row', schema=[(COLUMN_GENOME_ID, str),
                                                 (COLUMN_START, int),
                                                 (COLUMN_STOP, int)])
        obs = slice_positions(df, 'S1')
        plt.assert_frame_equal(obs, exp)

    def test_ordered_coverage(self):
        df = pl.DataFrame([['S1', 'G1', 1, 100, 1.],
                           ['S1', 'G2', 15, 100, 15.],
                           ['S1', 'G3', 101, 1000, 10.1],
                           ['S2', 'G2', 6, 100, 6.],
                           ['S3', 'G3', 7, 1000, .7],
                           ['S3', 'G2', 8, 100, 8.],
                           ['S3', 'G1', 9, 100, 9.]],
                          orient='row', schema=[(COLUMN_SAMPLE_ID, str),
                                                (COLUMN_GENOME_ID, str),
                                                (COLUMN_COVERED, COLUMN_COVERED_DTYPE),
                                                (COLUMN_LENGTH, COLUMN_LENGTH_DTYPE),
                                                (COLUMN_PERCENT_COVERED, float)])
        grp = pl.DataFrame([['S1', 'foo'],
                            ['S2', 'foo'],
                            ['S3', 'foo'],
                            ['S4', 'foo'],
                            ['S5', 'foo']],
                           orient='row',
                           schema=[(COLUMN_SAMPLE_ID, str),
                                   ('blah', str)])
        exp = pl.DataFrame([['S4', 'G2', 0, 100, 0., 0, 0, 'foo'],
                            ['S5', 'G2', 0, 100, 0., 1 / 5, 1, 'foo'],
                            ['S2', 'G2', 6, 100, 6., 2 / 5, 2, 'foo'],
                            ['S3', 'G2', 8, 100, 8., 3 / 5, 3, 'foo'],
                            ['S1', 'G2', 15, 100, 15., 4 / 5, 4, 'foo']],
                           orient='row', schema=[(COLUMN_SAMPLE_ID, str),
                                                 (COLUMN_GENOME_ID, str),
                                                 (COLUMN_COVERED, COLUMN_COVERED_DTYPE),
                                                 (COLUMN_LENGTH, COLUMN_LENGTH_DTYPE),
                                                 (COLUMN_PERCENT_COVERED, float),
                                                 ('x', float),
                                                 ('x_unscaled', pl.UInt64),
                                                 ('blah', str)])
        obs = ordered_coverage(df, grp, 'G2', 100)
        plt.assert_frame_equal(obs, exp, check_column_order=False)


    def test_compute_cumulative(self):
        df = pl.DataFrame([['S1', 'G1', 1, 100, 1.],
                           ['S1', 'G2', 15, 100, 15.],
                           ['S1', 'G3', 101, 1000, 10.1],
                           ['S2', 'G2', 6, 100, 6.],
                           ['S3', 'G3', 7, 1000, .7],
                           ['S3', 'G2', 8, 100, 8.],
                           ['S3', 'G1', 9, 100, 9.]],
                          orient='row', schema=[(COLUMN_SAMPLE_ID, str),
                                                (COLUMN_GENOME_ID, str),
                                                (COLUMN_COVERED, COLUMN_COVERED_DTYPE),
                                                (COLUMN_LENGTH, COLUMN_LENGTH_DTYPE),
                                                (COLUMN_PERCENT_COVERED, float)])
        pos = pl.DataFrame([['S1', 'G2', 1, 16],
                            ['S2', 'G2', 14, 20],
                            ['S3', 'G2', 18, 26]],
                           orient='row', schema=[(COLUMN_SAMPLE_ID, str),
                                                 (COLUMN_GENOME_ID, str),
                                                 (COLUMN_START, COLUMN_START_DTYPE),
                                                 (COLUMN_STOP, COLUMN_STOP_DTYPE)])
        grp = pl.DataFrame([['S1', 'foo'],
                            ['S2', 'foo'],
                            ['S3', 'foo'],
                            ['S4', 'foo'],
                            ['S5', 'foo']],
                           orient='row',
                           schema=[(COLUMN_SAMPLE_ID, str),
                                   ('blah', str)])
        lengths = pl.DataFrame([['G1', 100],
                                ['G2', 100],
                                ['G3', 1000]],
                               orient='row',
                               schema=[(COLUMN_GENOME_ID, str),
                                       (COLUMN_LENGTH, COLUMN_LENGTH_DTYPE)])
        exp_x = [0, 1, 2, 3, 4]
        exp_y = [0., 0., 6., 12., 25.]

        obs_x, obs_y = compute_cumulative(df, grp, 'G2', pos, lengths)
        self.assertEqual(obs_x.to_list(), exp_x)
        self.assertEqual(obs_y, exp_y)


    def test_get_covered(self):
        test = np.array([(1, 2, 3), (10, 20, 30)])
        exp = [[(1, 2), (1, 3)], [(10, 20), (10, 30)]]
        obs = get_covered(test)
        self.assertEqual(obs, exp)


if __name__ == '__main__':
    unittest.main()
