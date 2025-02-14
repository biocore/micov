import unittest

import polars as pl
import polars.testing as plt

from micov._constants import (
    BED_COV_SAMPLEID_SCHEMA,
    COLUMN_GENOME_ID,
    COLUMN_SAMPLE_ID,
    GENOME_COVERAGE_WITH_SAMPLEID_SCHEMA,
    GENOME_LENGTH_SCHEMA,
)
from micov._per_sample import compress_per_sample


class Tests(unittest.TestCase):
    def test_compress_per_sample(self):
        lengths = pl.DataFrame([['A', 400],
                                ['B', 500]],
                               orient='row',
                               schema=GENOME_LENGTH_SCHEMA.dtypes_flat)
        df = pl.DataFrame([['A', 10, 100, 'S1'],
                           ['A', 10, 20, 'S1'],
                           ['A', 90, 110, 'S1'],
                           ['B', 50, 150, 'S1'],
                           ['B', 200, 250, 'S1'],
                           ['A', 90, 95, 'S2'],
                           ['A', 50, 150, 'S2'],
                           ['A', 200, 300, 'S1'],
                           ['A', 201, 299, 'S1']],
                          orient='row',
                          schema=BED_COV_SAMPLEID_SCHEMA.dtypes_flat)
        s1_a = ((110 - 10) + (300 - 200))
        s1_b = ((150 - 50) + (250 - 200))
        s2_a = (150 - 50)
        exp = pl.DataFrame([['A', s1_a, 400, (s1_a / 400) * 100, 'S1'],
                            ['A', s2_a, 400, (s2_a / 400) * 100, 'S2'],
                            ['B', s1_b, 500, (s1_b / 500) * 100, 'S1']],
                           orient='row',
                           schema=GENOME_COVERAGE_WITH_SAMPLEID_SCHEMA.dtypes_flat)
        obs = compress_per_sample(df, lengths).sort([COLUMN_GENOME_ID,
                                                     COLUMN_SAMPLE_ID]).collect()
        plt.assert_frame_equal(obs, exp)


if __name__ == '__main__':
    unittest.main()
