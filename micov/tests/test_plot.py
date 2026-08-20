"""Tests for the data `_plot.py` plots.

`_plot.py` is the largest module in micov and has almost no unit tests; the
`.tsv.gz` and `.ks.tsv` goldens guard the `per-sample` path, and that is the
only reason changing it has been safe. **The `position-plot` command is not
covered by either** -- it writes PNGs and no data file, and its only test
asserts that two PNGs exist with valid magic bytes. PNG bytes are not
comparable across matplotlib versions, so the plotted values were, in effect,
unguarded.

`position_plot_segments` exists to close that. It is the data half of
`single_sample_position_plot`, split out so the numbers can be asserted without
rendering anything. The values below were captured from the polars
implementation before M5 replaced it -- this is Milestone 0's shape, freezing a
baseline and then changing the code underneath it.
"""

import shutil
import unittest
from tempfile import mkdtemp

from micov._io import load_bed_cov, load_genome_lengths
from micov._miint import connection
from micov._plot import position_plot_segments

#: Deliberately more awkward than `mini_sampleA.cov`: several intervals per
#: genome, rows interleaved and out of order, two different genome lengths so a
#: swapped denominator shows, and a genome with no length at all.
POSITIONS = (
    "genome_id\tstart\tstop\n"
    "G2\t500\t600\n"
    "G1\t300\t400\n"
    "G2\t100\t200\n"
    "G1\t0\t50\n"
    "G1\t950\t1000\n"
    "GX\t10\t20\n"
)
LENGTHS = "genome_id\tlength\nG1\t1000\nG2\t2000\n"

#: Captured from the pre-M5 polars implementation. x is the constant 0.5 the
#: plot draws every segment at; the other two are start/length and stop/length.
EXPECTED = {
    "G1": [[0.5, 0.0, 0.05], [0.5, 0.3, 0.4], [0.5, 0.95, 1.0]],
    "G2": [[0.5, 0.05, 0.1], [0.5, 0.25, 0.3]],
}


class PositionPlotSegmentTests(unittest.TestCase):
    def setUp(self):
        self.d = mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.con = connection()
        self.addCleanup(self.con.close)

    def load(self, positions=POSITIONS, lengths=LENGTHS):
        pos_path = f"{self.d}/s.cov"
        len_path = f"{self.d}/l.tsv"
        with open(pos_path, "w") as fp:
            fp.write(positions)
        with open(len_path, "w") as fp:
            fp.write(lengths)
        load_genome_lengths(self.con, len_path)
        load_bed_cov(self.con, pos_path)
        return position_plot_segments(self.con)

    def as_lists(self, segments):
        return {genome: array.tolist() for genome, array in segments.items()}

    def test_positions_are_normalized_by_their_own_genome_length(self):
        """The denominator has to be per-genome, not the first row's.

        G1 and G2 have different lengths precisely so that a single shared
        denominator -- the mistake `compute_cumulative` actually makes
        elsewhere, recorded as R22 -- produces different numbers here.
        """
        self.assertEqual(self.as_lists(self.load()), EXPECTED)

    def test_segments_are_ordered_by_start(self):
        """Input row order is not plot order.

        The rows are interleaved and unsorted on purpose. polars sorted
        stably; DuckDB does not, so the ordering has to be asked for
        explicitly or this comes back shuffled.
        """
        for genome, segments in self.as_lists(self.load()).items():
            with self.subTest(genome=genome):
                starts = [row[1] for row in segments]
                self.assertEqual(starts, sorted(starts))

    def test_genomes_without_a_length_are_dropped(self):
        """`GX` has coverage but no length, so it cannot be normalized.

        Dropping it is what micov has always done -- an inner join -- and it
        is preserved deliberately rather than promoted to an error, because
        this is a plotting command and a length file that covers only the
        genomes of interest is a normal way to use it.
        """
        self.assertNotIn("GX", self.load())

    def test_every_genome_with_a_length_is_plotted(self):
        self.assertEqual(sorted(self.load()), ["G1", "G2"])


if __name__ == "__main__":
    unittest.main()
