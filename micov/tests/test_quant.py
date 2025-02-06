import unittest
import polars as pl
import polars.testing as plt
from micov._quant import create_bin_list, pos_to_bins
from micov._constants import (BED_COV_SAMPLEID_SCHEMA, COLUMN_STOP_DTYPE,
                              COLUMN_START_DTYPE)


class CovTests(unittest.TestCase):
    def test_create_bin_list_case_1(self):
        # genome length is a multiple of bin_num
        genome_length = 100
        bin_num = 10

        obs = create_bin_list(genome_length, bin_num)

        exp = pl.DataFrame(
            [
                [1, 0.0, 10.0],
                [2, 10.0, 20.0],
                [3, 20.0, 30.0],
                [4, 30.0, 40.0],
                [5, 40.0, 50.0],
                [6, 50.0, 60.0],
                [7, 60.0, 70.0],
                [8, 70.0, 80.0],
                [9, 80.0, 90.0],
                [10, 90.0, 100.0],
            ],
            orient="row",
            schema=[
                ("bin_idx", int),
                ("bin_start", float),
                ("bin_stop", float),
            ],
        ).lazy()
        plt.assert_frame_equal(obs, exp)

    def test_create_bin_list_case_2(self):
        # genome length is not a multiple of bin_num
        genome_length = 100
        bin_num = 6

        obs = create_bin_list(genome_length, bin_num)

        exp = pl.DataFrame(
            [
                [1, 0.0, 16.666667],
                [2, 16.666667, 33.333333],
                [3, 33.333333, 50.0],
                [4, 50.0, 66.666667],
                [5, 66.666667, 83.333333],
                [6, 83.333333, 100.0],
            ],
            orient="row",
            schema=[
                ("bin_idx", int),
                ("bin_start", float),
                ("bin_stop", float),
            ],
        ).lazy()
        plt.assert_frame_equal(obs, exp)

    def test_pos_to_bins_case_1(self):
        # no cross-bin reads, no edge cases
        pos = pl.DataFrame(
            [
                ["G000006605", 5, 17, "A"],
                ["G000006605", 10, 17, "A"],
                ["G000006605", 11, 15, "A"],
                ["G000006605", 54, 59, "A"],
                ["G000006605", 71, 76, "B"],
                ["G000006605", 95, 99, "B"],
            ],
            orient="row",
            schema=BED_COV_SAMPLEID_SCHEMA.dtypes_flat,
        )
        genome_length = 100
        bin_num = 5

        obs_bin_df, obs_pos_updated = pos_to_bins(pos, genome_length, bin_num)

        exp_bin_df = pl.DataFrame(
            [
                [1, 3, 1, ["A"], 0.0, 20.0],
                [3, 1, 1, ["A"], 40.0, 60.0],
                [4, 1, 1, ["B"], 60.0, 80.0],
                [5, 1, 1, ["B"], 80.0, 100.0],
            ],
            orient="row",
            schema=[
                ("bin_idx", int),
                ("read_hits", pl.UInt32),
                ("sample_hits", pl.UInt32),
                ("samples", pl.List(str)),
                ("bin_start", float),
                ("bin_stop", float),
            ],
        )

        exp_pos_updated = pl.DataFrame(
            [
                ["G000006605", 5, 17, "A", 1, 1, [1]],
                ["G000006605", 10, 17, "A", 1, 1, [1]],
                ["G000006605", 11, 15, "A", 1, 1, [1]],
                ["G000006605", 54, 59, "A", 3, 3, [3]],
                ["G000006605", 71, 76, "B", 4, 4, [4]],
                ["G000006605", 95, 99, "B", 5, 5, [5]],
            ],
            orient="row",
            schema=[
                ("genome_id", str),
                ("start", COLUMN_START_DTYPE),
                ("stop", COLUMN_STOP_DTYPE),
                ("sample_id", str),
                ("start_bin_idx", int),
                ("stop_bin_idx", int),
                ("bin_idx", pl.List(int)),
            ],
        )

        # if using polars, we expect 32-bit indices. If using polars-u64-idx
        # then we expect 64-bit
        plt.assert_frame_equal(obs_bin_df, exp_bin_df, check_dtypes=False)
        plt.assert_frame_equal(obs_pos_updated, exp_pos_updated)

    def test_pos_to_bins_case_2(self):
        # cross-bin reads
        pos = pl.DataFrame(
            [
                ["G000006605", 5, 39, "A"],
                ["G000006605", 25, 45, "A"],
                ["G000006605", 11, 15, "A"],
                ["G000006605", 45, 65, "A"],
                ["G000006605", 71, 76, "B"],
                ["G000006605", 65, 99, "B"],
            ],
            orient="row",
            schema=BED_COV_SAMPLEID_SCHEMA.dtypes_flat,
        )
        genome_length = 100
        bin_num = 5

        obs_bin_df, obs_pos_updated = pos_to_bins(pos, genome_length, bin_num)

        exp_bin_df = pl.DataFrame(
            [
                [1, 2, 1, ["A"], 0.0, 20.0],
                [2, 2, 1, ["A"], 20.0, 40.0],
                [3, 2, 1, ["A"], 40.0, 60.0],
                [4, 3, 2, ["A", "B"], 60.0, 80.0],
                [5, 1, 1, ["B"], 80.0, 100.0],
            ],
            orient="row",
            schema=[
                ("bin_idx", int),
                ("read_hits", pl.UInt32),
                ("sample_hits", pl.UInt32),
                ("samples", pl.List(str)),
                ("bin_start", float),
                ("bin_stop", float),
            ],
        )

        exp_pos_updated = pl.DataFrame(
            [
                ["G000006605", 5, 39, "A", 1, 2, [1, 2]],
                ["G000006605", 25, 45, "A", 2, 3, [2, 3]],
                ["G000006605", 11, 15, "A", 1, 1, [1]],
                ["G000006605", 45, 65, "A", 3, 4, [3, 4]],
                ["G000006605", 71, 76, "B", 4, 4, [4]],
                ["G000006605", 65, 99, "B", 4, 5, [4, 5]],
            ],
            orient="row",
            schema=[
                ("genome_id", str),
                ("start", COLUMN_START_DTYPE),
                ("stop", COLUMN_STOP_DTYPE),
                ("sample_id", str),
                ("start_bin_idx", int),
                ("stop_bin_idx", int),
                ("bin_idx", pl.List(int)),
            ],
        )

        # if using polars, we expect 32-bit indices. If using polars-u64-idx
        # then we expect 64-bit
        plt.assert_frame_equal(obs_bin_df, exp_bin_df, check_dtypes=False)
        plt.assert_frame_equal(obs_pos_updated, exp_pos_updated)

    def test_pos_to_bins_case_3(self):
        # edge cases
        pos = pl.DataFrame(
            [
                ["G000006605", 0, 20, "A"],
                ["G000006605", 20, 40, "A"],
                ["G000006605", 60, 80, "B"],
                ["G000006605", 80, 100, "B"],
            ],
            orient="row",
            schema=BED_COV_SAMPLEID_SCHEMA.dtypes_flat,
        )
        genome_length = 100
        bin_num = 5

        obs_bin_df, obs_pos_updated = pos_to_bins(pos, genome_length, bin_num)

        exp_bin_df = pl.DataFrame(
            [
                [1, 1, 1, ["A"], 0.0, 20.0],
                [2, 1, 1, ["A"], 20.0, 40.0],
                [4, 1, 1, ["B"], 60.0, 80.0],
                [5, 1, 1, ["B"], 80.0, 100.0],
            ],
            orient="row",
            schema=[
                ("bin_idx", int),
                ("read_hits", pl.UInt32),
                ("sample_hits", pl.UInt32),
                ("samples", pl.List(str)),
                ("bin_start", float),
                ("bin_stop", float),
            ],
        )

        exp_pos_updated = pl.DataFrame(
            [
                ["G000006605", 0, 20, "A", 1, 1, [1]],
                ["G000006605", 20, 40, "A", 2, 2, [2]],
                ["G000006605", 60, 80, "B", 4, 4, [4]],
                ["G000006605", 80, 100, "B", 5, 5, [5]],
            ],
            orient="row",
            schema=[
                ("genome_id", str),
                ("start", COLUMN_START_DTYPE),
                ("stop", COLUMN_STOP_DTYPE),
                ("sample_id", str),
                ("start_bin_idx", int),
                ("stop_bin_idx", int),
                ("bin_idx", pl.List(int)),
            ],
        )

        # if using polars, we expect 32-bit indices. If using polars-u64-idx
        # then we expect 64-bit
        plt.assert_frame_equal(obs_bin_df, exp_bin_df, check_dtypes=False)
        plt.assert_frame_equal(obs_pos_updated, exp_pos_updated)

    def test_pos_to_bins_case_4(self):
        # edge cases
        pos = pl.DataFrame(
            [
                ["G000006605", 0, 40, "A"],
                ["G000006605", 15, 80, "A"],
                ["G000006605", 59, 81, "B"],
                ["G000006605", 80, 100, "B"],
            ],
            orient="row",
            schema=BED_COV_SAMPLEID_SCHEMA.dtypes_flat,
        )
        genome_length = 100
        bin_num = 5

        obs_bin_df, obs_pos_updated = pos_to_bins(pos, genome_length, bin_num)

        exp_bin_df = pl.DataFrame(
            [
                [1, 2, 1, ["A"], 0.0, 20.0],
                [2, 2, 1, ["A"], 20.0, 40.0],
                [3, 2, 2, ["A", "B"], 40.0, 60.0],
                [4, 2, 2, ["A", "B"], 60.0, 80.0],
                [5, 2, 1, ["B"], 80.0, 100.0],
            ],
            orient="row",
            schema=[
                ("bin_idx", int),
                ("read_hits", pl.UInt32),
                ("sample_hits", pl.UInt32),
                ("samples", pl.List(str)),
                ("bin_start", float),
                ("bin_stop", float),
            ],
        )

        exp_pos_updated = pl.DataFrame(
            [
                ["G000006605", 0, 40, "A", 1, 2, [1, 2]],
                ["G000006605", 15, 80, "A", 1, 4, [1, 2, 3, 4]],
                ["G000006605", 59, 81, "B", 3, 5, [3, 4, 5]],
                ["G000006605", 80, 100, "B", 5, 5, [5]],
            ],
            orient="row",
            schema=[
                ("genome_id", str),
                ("start", COLUMN_START_DTYPE),
                ("stop", COLUMN_STOP_DTYPE),
                ("sample_id", str),
                ("start_bin_idx", int),
                ("stop_bin_idx", int),
                ("bin_idx", pl.List(int)),
            ],
        )

        # if using polars, we expect 32-bit indices. If using polars-u64-idx
        # then we expect 64-bit
        plt.assert_frame_equal(obs_bin_df, exp_bin_df, check_dtypes=False)
        plt.assert_frame_equal(obs_pos_updated, exp_pos_updated)


if __name__ == "__main__":
    unittest.main()
