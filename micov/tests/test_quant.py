import unittest

import polars as pl
import polars.testing as plt

from micov._constants import (
    COLUMN_GENOME_ID,
    COLUMN_GENOME_ID_DTYPE,
    COLUMN_SAMPLE_ID,
    COLUMN_SAMPLE_ID_DTYPE,
    COLUMN_START,
    COLUMN_START_DTYPE,
    COLUMN_STOP,
    COLUMN_STOP_DTYPE,
)
from micov._quant import create_bin_list, pos_to_bins


class Tests(unittest.TestCase):
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
                [10, 90.0, 101.0],
            ],
            orient="row",
            schema=[
                ("bin_idx", COLUMN_START_DTYPE),
                ("bin_start", COLUMN_START_DTYPE),
                ("bin_stop", COLUMN_STOP_DTYPE),
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
                [1, 0, 17],
                [2, 17, 33],
                [3, 33, 50],
                [4, 50, 67],
                [5, 67, 83],
                [6, 83, 101],
            ],
            orient="row",
            schema=[
                ("bin_idx", COLUMN_START_DTYPE),
                ("bin_start", COLUMN_START_DTYPE),
                ("bin_stop", COLUMN_STOP_DTYPE),
            ],
        ).lazy()
        plt.assert_frame_equal(obs, exp)

    def test_pos_to_bins_case_1(self):
        # no cross-bin reads, no edge cases
        pos = pl.DataFrame(
            [
                ["G000006605", 5, 17, "s1", "A", 100],
                ["G000006605", 10, 17, "s1", "A", 100],
                ["G000006605", 11, 15, "s1", "A", 100],
                ["G000006605", 54, 59, "s1", "A", 100],
                ["G000006605", 71, 76, "s2", "B", 100],
                ["G000006605", 95, 99, "s2", "B", 100],
            ],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                (COLUMN_START, COLUMN_START_DTYPE),
                (COLUMN_STOP, COLUMN_STOP_DTYPE),
                (COLUMN_SAMPLE_ID, COLUMN_SAMPLE_ID_DTYPE),
                ("variable", str),
                ("length", int),
            ],
        ).lazy()
        bin_num = 5

        obs_bin_df = pos_to_bins(pos, "variable", bin_num, 100)
        exp_bin_df = pl.DataFrame(
            [
                ["G000006605", "A", 1, 3, 1, ["s1"], 0.0, 20.0],
                ["G000006605", "A", 3, 1, 1, ["s1"], 40.0, 60.0],
                ["G000006605", "B", 4, 1, 1, ["s2"], 60.0, 80.0],
                ["G000006605", "B", 5, 1, 1, ["s2"], 80.0, 101.0],
            ],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                ("variable", str),
                ("bin_idx", int),
                ("read_hits", int),
                ("sample_hits", int),
                ("samples", pl.List(pl.Utf8)),
                ("bin_start", COLUMN_START_DTYPE),
                ("bin_stop", COLUMN_STOP_DTYPE),
            ],
        ).lazy()

        # if using polars, we expect 32-bit indices. If using polars-u64-idx
        # then we expect 64-bit
        plt.assert_frame_equal(obs_bin_df, exp_bin_df, check_dtypes=False)

    def test_pos_to_bins_case_2(self):
        # cross-bin reads
        pos = pl.DataFrame(
            [
                ["G000006605", 5, 39, "s1", "A", 100],
                ["G000006605", 25, 45, "s1", "A", 100],
                ["G000006605", 11, 15, "s1", "A", 100],
                ["G000006605", 45, 65, "s1", "A", 100],
                ["G000006605", 71, 76, "s2", "B", 100],
                ["G000006605", 65, 99, "s2", "B", 100],
            ],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                (COLUMN_START, COLUMN_START_DTYPE),
                (COLUMN_STOP, COLUMN_STOP_DTYPE),
                (COLUMN_SAMPLE_ID, COLUMN_SAMPLE_ID_DTYPE),
                ("variable", str),
                ("length", int),
            ],
        ).lazy()
        bin_num = 5

        obs_bin_df = pos_to_bins(pos, "variable", bin_num, 100)
        exp_bin_df = pl.DataFrame(
            [
                ["G000006605", "A", 1, 2, 1, ["s1"], 0.0, 20.0],
                ["G000006605", "A", 2, 2, 1, ["s1"], 20.0, 40.0],
                ["G000006605", "A", 3, 2, 1, ["s1"], 40.0, 60.0],
                ["G000006605", "A", 4, 1, 1, ["s1"], 60.0, 80.0],
                ["G000006605", "B", 4, 2, 1, ["s2"], 60.0, 80.0],
                ["G000006605", "B", 5, 1, 1, ["s2"], 80.0, 101.0],
            ],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                ("variable", str),
                ("bin_idx", int),
                ("read_hits", int),
                ("sample_hits", int),
                ("samples", pl.List(pl.Utf8)),
                ("bin_start", COLUMN_START_DTYPE),
                ("bin_stop", COLUMN_STOP_DTYPE),
            ],
        ).lazy()

        # if using polars, we expect 32-bit indices. If using polars-u64-idx
        # then we expect 64-bit
        plt.assert_frame_equal(obs_bin_df, exp_bin_df, check_dtypes=False)

    def test_pos_to_bins_case_3(self):
        # edge cases
        pos = pl.DataFrame(
            [
                ["G000006605", 0, 20, "s1", "A", 100],
                ["G000006605", 20, 40, "s1", "A", 100],
                ["G000006605", 60, 80, "s2", "B", 100],
                ["G000006605", 80, 100, "s2", "B", 100],
            ],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                (COLUMN_START, COLUMN_START_DTYPE),
                (COLUMN_STOP, COLUMN_STOP_DTYPE),
                (COLUMN_SAMPLE_ID, COLUMN_SAMPLE_ID_DTYPE),
                ("variable", str),
                ("length", int),
            ],
        ).lazy()
        bin_num = 5
        obs_bin_df = pos_to_bins(pos, "variable", bin_num, 100)
        exp_bin_df = pl.DataFrame(
            [
                ["G000006605", "A", 1, 1, 1, ["s1"], 0.0, 20.0],
                ["G000006605", "A", 2, 1, 1, ["s1"], 20.0, 40.0],
                ["G000006605", "B", 4, 1, 1, ["s2"], 60.0, 80.0],
                ["G000006605", "B", 5, 1, 1, ["s2"], 80.0, 101.0],
            ],
            orient="row",
            schema=[
                (COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                ("variable", str),
                ("bin_idx", int),
                ("read_hits", int),
                ("sample_hits", int),
                ("samples", pl.List(pl.Utf8)),
                ("bin_start", COLUMN_START_DTYPE),
                ("bin_stop", COLUMN_STOP_DTYPE),
            ],
        ).lazy()

        # if using polars, we expect 32-bit indices. If using polars-u64-idx
        # then we expect 64-bit
        plt.assert_frame_equal(obs_bin_df, exp_bin_df, check_dtypes=False)


if __name__ == "__main__":
    unittest.main()
