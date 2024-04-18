import unittest
from micov._cov import compress, coverage_percent
from micov._constants import (BED_COV_SCHEMA, COLUMN_GENOME_ID,
                              GENOME_LENGTH_SCHEMA,
                              GENOME_COVERAGE_SCHEMA)
import polars as pl
import polars.testing as plt


class CovTests(unittest.TestCase):
    def test_compress(self):
        exp = pl.DataFrame([['G123', 10, 50],
                            ['G123', 51, 89],
                            ['G123', 90, 100],
                            ['G123', 101, 110],
                            ['G456', 200, 300],
                            ['G456', 400, 500]],
                           schema=BED_COV_SCHEMA.dtypes_flat)
        data = pl.DataFrame([['G123', 11, 50],
                             ['G123', 20, 30],
                             ['G456', 200, 299],
                             ['G123', 10, 12],
                             ['G456', 201, 300],
                             ['G123', 90, 100],
                             ['G123', 51, 89],
                             ['G123', 101, 110],
                             ['G456', 400, 500]],
                            schema=BED_COV_SCHEMA.dtypes_flat)
        obs = compress(data).sort(COLUMN_GENOME_ID)
        plt.assert_frame_equal(obs, exp)

    def test_coverage_percent(self):
        data = pl.DataFrame([['G123', 11, 50],
                             ['G456', 200, 299],
                             ['G123', 90, 100],
                             ['G456', 400, 500]],
                            schema=BED_COV_SCHEMA.dtypes_flat)
        lengths = pl.DataFrame([['G123', 100],
                                ['G456', 1000],
                                ['G789', 500]],
                               schema=GENOME_LENGTH_SCHEMA.dtypes_flat)

        g123_covered = (50 - 11) + (100 - 90)
        g456_covered = (299 - 200) + (500 - 400)
        exp = pl.DataFrame([['G123', g123_covered, 100, (g123_covered / 100) * 100],
                            ['G456', g456_covered, 1000, (g456_covered / 1000) * 100]],
                           schema=GENOME_COVERAGE_SCHEMA.dtypes_flat)

        obs = coverage_percent(data, lengths).sort(COLUMN_GENOME_ID).collect()
        plt.assert_frame_equal(obs, exp)


if __name__ == '__main__':
    unittest.main()
