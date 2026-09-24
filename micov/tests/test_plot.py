"""Tests for the data `_plot.py` plots.

`_plot.py` is the largest module in micov and has almost no unit tests; the
`.tsv.gz` and `.ks.csv` goldens guard the `per-sample` path, and that is the
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

import math
import shutil
import unittest
from tempfile import mkdtemp

import numpy as np

from micov._io import load_bed_cov, load_genome_lengths
from micov._miint import connection
from micov._plot import KS_HEADER, ks_2samp, ks_table, position_plot_segments
from micov.tests._golden import TSV_FLOAT_REL_TOL

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


#: Two frozen curve pairs from `example/`, captured by recording what
#: `coverage_curve` handed to the KS test when regenerating the committed
#: `.ks.csv` goldens. Each entry is (curve A, curve B, statistic, p-value), the
#: last two as `example/plots/per_sample_groups/*.cumulative.ks.csv` records
#: them -- the published numbers, computed by scipy 1.17.1.
KS_CASES = {
    # G000154205, No vs Yes: n=20 and n=19, and D is exactly 114/380. A
    # merge-walk computes 19/19 - 14/20 there, which is 0.30000000000000004
    # in floating point; the published statistic is 0.3.
    "G000154205 No vs Yes": (
        (0.10697629973873544, 0.6112416857125725, 1.5797914163437496,
         2.42930909073959, 6.278379494450643, 13.667604783910628,
         20.869171311876062, 31.62347817261852, 41.09682382725987,
         49.6634876053475, 57.773113205248514, 64.49255541145618,
         70.08464666569346, 75.03284610985739, 79.1538172571904,
         81.94274384356586, 84.38605795195791, 86.62325888073849,
         87.99439460291961, 89.2058392236686),
        (0.8525686918571945, 1.4834724900984948, 1.9273107802405092,
         2.7446444579433136, 3.045445116963085, 3.4468657893437706,
         4.485758422556172, 4.679264967518317, 9.22977276064323,
         15.490460591342273, 21.310678963679543, 27.179989054474856,
         36.677594535458226, 46.55780608114392, 56.320850081265114,
         63.519980032785725, 69.61735367881727, 74.42726151902109,
         78.50231909108494),
        0.3,
        0.24244968766417713,
    ),
    # G000436435, No vs Yes: the pair whose p-value drifts furthest from
    # scipy's, 3.8e-16 relative.
    "G000436435 No vs Yes": (
        (0.008376906961733242, 0.2256342328286496, 0.766019525672602,
         1.3843960661446557, 2.2951603167966708, 3.297079526016654,
         4.604456664091266, 6.185279979416743, 8.45637538715147,
         11.698705842668225, 14.770375517292702, 18.81451433759982,
         22.72398689911586, 27.321469040223363, 33.4039075279224,
         39.310543160143276, 44.969854354009584, 49.983694948949484,
         55.89853920205473, 62.76064708614527),
        (0.2888537025554802, 2.3882599144807553, 4.9497796948262875,
         8.247681952776682, 11.691263858358472, 15.271924123173442,
         20.24962060838783, 26.840413938874008, 33.705775353793435,
         42.52089926096234, 50.518227625992054, 57.55544652279827,
         62.515529065249375, 68.98936357197296, 73.58448970799748,
         76.97577204042754, 79.42743840916553, 81.06252463521187,
         82.18882595405117),
        0.3736842105263158,
        0.10846042853587466,
    ),
}


class KsTwoSampleTests(unittest.TestCase):
    """The KS statistics and p-values are cited in the paper.

    M9 moved them from `scipy.stats.ks_2samp` to miint's `ks_2samp`. The
    statistic has to come out **exactly** as published; the p-value may drift
    by a few ULP, because miint derives it independently of scipy (Hodges
    lattice paths, computed as the mass escaping the band rather than
    ``1 - P(inside)``), and that is the one place a tolerance is accepted.
    """

    def setUp(self):
        self.con = connection()
        self.addCleanup(self.con.close)

    def test_statistic_is_the_published_value_exactly(self):
        """A stale pre-#257 miint returns the raw sweep value, 1 ULP off.

        The capability guard in `_miint` checks names only, so an extension
        cached in ``~/.duckdb/extensions/`` before the fix passes it and then
        reports different published statistics. This is the check that
        notices.
        """
        for name, (a, b, statistic, _) in KS_CASES.items():
            with self.subTest(pair=name):
                observed, _ = ks_2samp(self.con, a, b)
                self.assertEqual(
                    observed,
                    statistic,
                    "the KS statistic is not the published value. A miint "
                    "build predating the-miint/duckdb-miint#257 returns the "
                    "un-snapped sweep value; run FORCE INSTALL miint to "
                    "replace a stale cached extension.",
                )

    def test_statistic_is_the_exact_lattice_value(self):
        """D is a multiple of 1/lcm(n1, n2): 114/380, not 19/19 - 14/20."""
        a, b, _, _ = KS_CASES["G000154205 No vs Yes"]
        raw_sweep = 19 / 19 - 14 / 20
        # the premise: the two really are different doubles
        self.assertNotEqual(raw_sweep, 114 / 380)
        observed, _ = ks_2samp(self.con, a, b)
        self.assertEqual(observed, 114 / 380)

    def test_pvalue_is_the_published_value_within_tolerance(self):
        for name, (a, b, _, pvalue) in KS_CASES.items():
            with self.subTest(pair=name):
                _, observed = ks_2samp(self.con, a, b)
                self.assertTrue(
                    math.isclose(
                        observed, pvalue,
                        rel_tol=TSV_FLOAT_REL_TOL, abs_tol=TSV_FLOAT_REL_TOL,
                    ),
                    f"p-value {observed!r} differs from the published "
                    f"{pvalue!r} by more than {TSV_FLOAT_REL_TOL:g} relative",
                )

    def test_accepts_numpy_curves(self):
        """`coverage_curve` hands over numpy arrays, not tuples."""
        a, b, statistic, _ = KS_CASES["G000154205 No vs Yes"]
        observed, _ = ks_2samp(self.con, np.asarray(a), np.asarray(b))
        self.assertEqual(observed, statistic)


#: Synthetic curves for the table tests. Their KS values are irrelevant; what
#: matters is how many comparisons each row is corrected for. `SAME` is `LOW`
#: again, so that pair's p-value is 1 and its correction has to be capped.
LOW = [float(x) for x in range(10)]
MID = [x + 0.5 for x in LOW]
HIGH = [x + 5.0 for x in LOW]
MONTE = [x + 2.5 for x in LOW]


class KsTableTests(unittest.TestCase):
    """The `.ks.csv` rows, including the Bonferroni column added in M11a.

    The family is **per file, excluding Monte Carlo rows**: a Monte Carlo
    curve is a null-model check, not a hypothesis, so comparing against it
    neither gets corrected nor inflates the correction of the real
    comparisons. Were it counted, the same pair of groups would report a
    different corrected p-value depending on whether ``--monte`` was passed.
    """

    def setUp(self):
        self.con = connection()
        self.addCleanup(self.con.close)

    def table(self, curves, monte_label=None):
        return {
            (row[0], row[1]): row[2:]
            for row in ks_table(self.con, curves, monte_label)
        }

    def test_header_appends_the_corrected_column(self):
        """The first four names are frozen; the new one goes last."""
        self.assertEqual(
            KS_HEADER,
            ("label_A", "label_B", "ks-statistic", "ks-pvalue",
             "ks-pvalue-bonferroni"),
        )

    def test_every_pair_is_compared_once_in_curve_order(self):
        rows = ks_table(self.con, {"a": LOW, "b": MID, "c": HIGH})
        self.assertEqual(
            [tuple(r[:2]) for r in rows], [("a", "b"), ("a", "c"), ("b", "c")]
        )

    def test_statistic_and_pvalue_are_ks_2samp(self):
        _, (statistic, pvalue, _) = next(
            iter(self.table({"a": LOW, "c": HIGH}).items())
        )
        self.assertEqual((statistic, pvalue), ks_2samp(self.con, LOW, HIGH))

    def test_bonferroni_multiplies_by_the_number_of_comparisons(self):
        table = self.table({"a": LOW, "b": MID, "c": HIGH})
        for pair, (_, pvalue, corrected) in table.items():
            with self.subTest(pair=pair):
                self.assertEqual(corrected, min(1.0, pvalue * 3))
        # the premise: at least one row is actually scaled, not capped
        self.assertTrue(any(p * 3 < 1 for _, p, _ in table.values()))

    def test_bonferroni_is_capped_at_one(self):
        """p = 1 for identical curves; p * m would be 3, not a probability."""
        _, pvalue, corrected = self.table(
            {"a": LOW, "same": list(LOW), "c": HIGH}
        )[("a", "same")]
        self.assertEqual(pvalue, 1.0)
        self.assertEqual(corrected, 1.0)

    def test_monte_carlo_rows_are_neither_corrected_nor_counted(self):
        """Three groups plus Monte Carlo is m = 3, not 6."""
        monte = "Monte Carlo unfocused (n=10)"
        curves = {"a": LOW, "b": MID, "c": HIGH, monte: MONTE}
        table = self.table(curves, monte_label=monte)

        for pair, (_, pvalue, corrected) in table.items():
            with self.subTest(pair=pair):
                if monte in pair:
                    self.assertEqual(corrected, "")
                else:
                    self.assertEqual(corrected, min(1.0, pvalue * 3))

    def test_a_group_named_like_monte_carlo_is_still_a_group(self):
        """Identified by being the Monte Carlo curve, not by its label."""
        curves = {"Monte Carlo lookalike": LOW, "b": MID, "c": HIGH}
        table = self.table(curves)
        for pair, (_, pvalue, corrected) in table.items():
            with self.subTest(pair=pair):
                self.assertEqual(corrected, min(1.0, pvalue * 3))

    def test_one_group_and_monte_carlo_has_nothing_to_correct(self):
        monte = "Monte Carlo unfocused (n=10)"
        rows = ks_table(self.con, {"a": LOW, monte: MONTE}, monte)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][4], "")


if __name__ == "__main__":
    unittest.main()
