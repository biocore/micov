import itertools
import unittest

import duckdb

from micov._quant import create_bin_list, pos_to_bins


class Tests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect(":memory:")

    def tearDown(self):
        self.con.close()

    def binned(self, rows, bin_num, lengths):
        """Bin covered positions that are already joined to sample metadata.

        That join is what `binning` hands `pos_to_bins`: one row per covered
        interval, carrying the sample's stratification value.
        """
        values = ", ".join(
            f"('{genome}', {start}::UINTEGER, {stop}::UINTEGER, "
            f"'{sample}', '{value}')"
            for genome, start, stop, sample, value in rows
        )
        positions = (
            f"SELECT * FROM (VALUES {values}) "
            f"t(genome_id, start, stop, sample_id, variable)"
        )
        lengths = " UNION ALL ".join(
            f"SELECT '{genome}' AS genome_id, {length} AS length"
            for genome, length in lengths.items()
        )
        return self.con.sql(
            pos_to_bins(positions, lengths, "variable", bin_num)
        ).fetchall()

    def test_create_bin_list_case_1(self):
        # genome length is a multiple of bin_num
        obs = create_bin_list(self.con, 100, 10).fetchall()
        exp = [
            (1, 0, 10),
            (2, 10, 20),
            (3, 20, 30),
            (4, 30, 40),
            (5, 40, 50),
            (6, 50, 60),
            (7, 60, 70),
            (8, 70, 80),
            (9, 80, 90),
            # the last bin stops one past the genome so a read ending on the
            # final base still lands in a bin
            (10, 90, 101),
        ]
        self.assertEqual(obs, exp)

    def test_create_bin_list_case_2(self):
        # genome length is not a multiple of bin_num, so the breakpoints are
        # rounded -- half away from zero, which is what polars' hist() did
        obs = create_bin_list(self.con, 100, 6).fetchall()
        exp = [
            (1, 0, 17),
            (2, 17, 33),
            (3, 33, 50),
            (4, 50, 67),
            (5, 67, 83),
            (6, 83, 101),
        ]
        self.assertEqual(obs, exp)

    def test_create_bin_list_bins_are_contiguous(self):
        """Every base of the genome falls in exactly one bin.

        A gap or an overlap would silently lose or double-count reads, which
        no single expected-value test would reveal.
        """
        for genome_length, bin_num in [(100, 6), (3212345, 1000), (997, 13)]:
            bins = create_bin_list(self.con, genome_length, bin_num).fetchall()
            with self.subTest(length=genome_length, bins=bin_num):
                self.assertEqual(len(bins), bin_num)
                self.assertEqual(bins[0][1], 0)
                self.assertEqual(bins[-1][2], genome_length + 1)
                for (_, _, stop), (_, start, _) in itertools.pairwise(bins):
                    self.assertEqual(start, stop)

    def test_pos_to_bins_case_1(self):
        # no cross-bin reads, no edge cases
        obs = self.binned(
            [
                ["G000006605", 5, 17, "s1", "A"],
                ["G000006605", 10, 17, "s1", "A"],
                ["G000006605", 11, 15, "s1", "A"],
                ["G000006605", 54, 59, "s1", "A"],
                ["G000006605", 71, 76, "s2", "B"],
                ["G000006605", 95, 99, "s2", "B"],
            ],
            bin_num=5,
            lengths={"G000006605": 100},
        )
        exp = [
            ("G000006605", "A", 1, 3, 1, "[s1]", 0, 20),
            ("G000006605", "A", 3, 1, 1, "[s1]", 40, 60),
            ("G000006605", "B", 4, 1, 1, "[s2]", 60, 80),
            ("G000006605", "B", 5, 1, 1, "[s2]", 80, 101),
        ]
        self.assertEqual(obs, exp)

    def test_pos_to_bins_case_2(self):
        # cross-bin reads: a read is counted once in every bin it spans, so
        # read_hits sums to more than the number of reads
        obs = self.binned(
            [
                ["G000006605", 5, 39, "s1", "A"],
                ["G000006605", 25, 45, "s1", "A"],
                ["G000006605", 11, 15, "s1", "A"],
                ["G000006605", 45, 65, "s1", "A"],
                ["G000006605", 71, 76, "s2", "B"],
                ["G000006605", 65, 99, "s2", "B"],
            ],
            bin_num=5,
            lengths={"G000006605": 100},
        )
        exp = [
            ("G000006605", "A", 1, 2, 1, "[s1]", 0, 20),
            ("G000006605", "A", 2, 2, 1, "[s1]", 20, 40),
            ("G000006605", "A", 3, 2, 1, "[s1]", 40, 60),
            ("G000006605", "A", 4, 1, 1, "[s1]", 60, 80),
            ("G000006605", "B", 4, 2, 1, "[s2]", 60, 80),
            ("G000006605", "B", 5, 1, 1, "[s2]", 80, 101),
        ]
        self.assertEqual(obs, exp)

    def test_pos_to_bins_case_3(self):
        # reads flush against bin boundaries. Intervals are half open, so a
        # read ending exactly on a boundary must not reach into the next bin
        obs = self.binned(
            [
                ["G000006605", 0, 20, "s1", "A"],
                ["G000006605", 20, 40, "s1", "A"],
                ["G000006605", 60, 80, "s2", "B"],
                ["G000006605", 80, 100, "s2", "B"],
            ],
            bin_num=5,
            lengths={"G000006605": 100},
        )
        exp = [
            ("G000006605", "A", 1, 1, 1, "[s1]", 0, 20),
            ("G000006605", "A", 2, 1, 1, "[s1]", 20, 40),
            ("G000006605", "B", 4, 1, 1, "[s2]", 60, 80),
            ("G000006605", "B", 5, 1, 1, "[s2]", 80, 101),
        ]
        self.assertEqual(obs, exp)

    def test_pos_to_bins_uses_each_genomes_own_length(self):
        """Bin bounds are per genome, not per run.

        Every genome is binned in one statement now, so a bug here shows up as
        one genome wearing another's bin bounds -- silently, since the counts
        still look plausible.
        """
        obs = self.binned(
            [
                ["G1", 5, 17, "s1", "A"],
                ["G2", 5, 17, "s1", "A"],
                ["G2", 50, 60, "s1", "A"],
            ],
            bin_num=5,
            lengths={"G1": 100, "G2": 200},
        )
        exp = [
            ("G1", "A", 1, 1, 1, "[s1]", 0, 20),
            ("G2", "A", 1, 1, 1, "[s1]", 0, 40),
            ("G2", "A", 2, 1, 1, "[s1]", 40, 80),
        ]
        self.assertEqual(obs, exp)

    def test_pos_to_bins_counts_distinct_samples(self):
        """read_hits counts intervals, sample_hits counts samples.

        They differ whenever one sample contributes several reads to a bin,
        which is the case the variance ranking is built on.
        """
        obs = self.binned(
            [
                ["G000006605", 1, 5, "s1", "A"],
                ["G000006605", 6, 9, "s1", "A"],
                ["G000006605", 2, 8, "s2", "A"],
            ],
            bin_num=5,
            lengths={"G000006605": 100},
        )
        self.assertEqual(obs, [("G000006605", "A", 1, 3, 2, "[s1,s2]", 0, 20)])


if __name__ == "__main__":
    unittest.main()
