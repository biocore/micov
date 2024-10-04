import unittest
import polars as pl
import polars.testing as plt
from micov._quant import pos_to_bins


class CovTests(unittest.TestCase):
    def test_compress(self):
        # TBD
        exp = pl.DataFrame(
            [
                ["G123", 10, 50],
                ["G123", 51, 89],
                ["G123", 90, 100],
                ["G123", 101, 110],
                ["G456", 200, 300],
                ["G456", 400, 500],
            ],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        data = pl.DataFrame(
            [
                ["G123", 11, 50],
                ["G123", 20, 30],
                ["G456", 200, 299],
                ["G123", 10, 12],
                ["G456", 201, 300],
                ["G123", 90, 100],
                ["G123", 51, 89],
                ["G123", 101, 110],
                ["G456", 400, 500],
            ],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        obs = compress(data).sort(COLUMN_GENOME_ID)
        plt.assert_frame_equal(obs, exp)


if __name__ == "__main__":
    unittest.main()
