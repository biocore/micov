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

import csv
import gzip
import math
import os
import shutil
import unittest
from itertools import combinations
from tempfile import mkdtemp
from unittest import mock

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from micov import _plot
from micov._io import load_bed_cov, load_genome_lengths
from micov._miint import connection
from micov._plot import (
    GROUP_COLORS,
    KS_HEADER,
    ks_2samp,
    ks_table,
    position_plot_segments,
)
from micov._view import View
from micov.tests._golden import TSV_FLOAT_REL_TOL
from micov.tests.test_cov import COVERAGE_COLUMNS, POSITION_COLUMNS, table

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

    def test_the_plot_is_drawn_in_the_first_group_colour(self):
        self.load()
        colours = []

        def capture(*args, **kwargs):
            colours.extend(to_hex(c.get_color()[0]) for c in plt.gca().collections)

        with mock.patch.object(_plot.plt, "savefig", capture):
            _plot.single_sample_position_plot(self.con, f"{self.d}/out")
        self.assertEqual(colours, [GROUP_COLORS[0].lower()] * 2, "one per genome")


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


class PositionPlotGroupOrderTests(unittest.TestCase):
    """Where each metadata group sits along the position plot's x-axis.

    Metadata is read as text and groups are laid out smallest first, ties in
    text order, so a depth column came out as 10, 20, 24, 55, 56, 71, 30, 5,
    270 -- no order a reader can follow. `sort_by_value` lays groups out by
    value instead, numbers numerically. The default must not move: the scaled
    plot's `x` is a frozen output.
    """

    #: Sizes chosen so that smallest first, text order and numeric order all
    #: disagree. `None` is a blank metadata value, which `View` hands over
    #: masked with `None` beneath; there are 146 `not applicable` and 28 blank
    #: depths in the study this flag was written for.
    GROUPS = ("5", "5", "5", "30", "270", "270", "not applicable", None, None)

    def setUp(self):
        self.d = mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.build(self.GROUPS)

    def build(self, groups):
        n = len(groups)
        samples = np.array([f"S{i}" for i in range(n)], dtype=object)
        self.metadata = {
            "sample_id": samples,
            "grp": np.ma.array(
                groups, mask=[g is None for g in groups], dtype=object
            ),
        }
        self.coverage = {
            "sample_id": samples,
            "genome_id": np.full(n, "G1", dtype=object),
            "covered": np.full(n, 490, dtype=np.uint32),
            "length": np.full(n, 1000, dtype=np.uint32),
            "percent_covered": np.full(n, 49.0),
        }
        self.positions = {
            "genome_id": np.full(n, "G1", dtype=object),
            "start": np.full(n, 10, dtype=np.uint32),
            "stop": np.full(n, 500, dtype=np.uint32),
            "sample_id": samples,
        }

    def groups_left_to_right(self, **kwargs):
        _plot.position_plot(self.metadata, self.coverage, self.positions, "G1",
                            "grp", f"{self.d}/out", "G1", 0, 1000, scale=10000,
                            **kwargs)
        path = f"{self.d}/out.G1.G1.grp.position-plot-scaled.tsv.gz"
        with gzip.open(path, "rt") as fp:
            rows = sorted(csv.DictReader(fp, delimiter="\t"),
                          key=lambda row: int(row["x"]))
        return list(dict.fromkeys(row["group"] for row in rows))

    def test_default_is_smallest_group_first(self):
        # "--" is how the writer renders the blank group, which np.unique
        # sorts as "?": 270 precedes it on the tie at two samples
        self.assertEqual(self.groups_left_to_right(),
                         ["30", "not applicable", "270", "--", "5"])

    def test_sort_by_value_puts_numbers_in_numeric_order_then_text(self):
        self.assertEqual(self.groups_left_to_right(sort_by_value=True),
                         ["5", "30", "270", "not applicable", "--"])

    def test_non_finite_values_sort_as_text(self):
        """`float()` accepts "-inf" and "NaN", but neither is a depth.

        As numbers, "-inf" led the axis, and NaN -- which compares false both
        ways -- had no defined place among the numbers at all.
        """
        self.build(("5", "-inf", "NaN", "missing"))
        self.assertEqual(self.groups_left_to_right(sort_by_value=True),
                         ["5", "-inf", "NaN", "missing"])


class ScaledPositionPlotTests(unittest.TestCase):
    """Which buckets the scaled position plot marks, and how wide they are.

    Buckets were 1/10000 of the genome whatever the genome: 15bp on a 145kb
    chloroplast, narrower than the peptides being plotted. Only the buckets
    holding an interval's two ends were marked, which there dropped 49% of the
    covered buckets. Buckets are now never narrower than 100bp, and every
    bucket holding a covered base is marked.
    """

    def plot(self, intervals, ymin, ymax):
        d = mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        n = len(intervals)
        sample = np.array(["S0"], dtype=object)
        _plot.position_plot(
            {"sample_id": sample, "grp": np.array(["g"], dtype=object)},
            {
                "sample_id": sample,
                "genome_id": np.array(["G1"], dtype=object),
                "covered": np.array([1], dtype=np.uint32),
                "length": np.array([ymax - ymin], dtype=np.uint32),
                "percent_covered": np.array([1.0]),
            },
            {
                "genome_id": np.full(n, "G1", dtype=object),
                "start": np.array([s for s, _ in intervals], dtype=np.uint32),
                "stop": np.array([e for _, e in intervals], dtype=np.uint32),
                "sample_id": np.full(n, "S0", dtype=object),
            },
            "G1", "grp", f"{d}/out", "G1", ymin, ymax, scale=10000,
        )
        return f"{d}/out.G1.G1.grp.position-plot"

    def marked(self, intervals, ymin=0, ymax=1000):
        with gzip.open(f"{self.plot(intervals, ymin, ymax)}-scaled.tsv.gz",
                       "rt") as fp:
            return [float(row["y"]) for row in csv.DictReader(fp, delimiter="\t")]

    def test_files_are_named_scaled_whatever_the_bucket_count(self):
        prefix = self.plot([(150, 151)], 0, 1000)
        self.assertTrue(os.path.exists(f"{prefix}-scaled.png"))
        self.assertTrue(os.path.exists(f"{prefix}-scaled.tsv.gz"))

    def test_a_short_genome_gets_100bp_buckets(self):
        self.assertEqual(self.marked([(150, 151)]), [100.0])

    def test_every_bucket_an_interval_covers_is_marked(self):
        self.assertEqual(self.marked([(150, 420)]),
                         [100.0, 200.0, 300.0, 400.0])

    def test_stop_is_exclusive(self):
        # [100, 200) covers nothing in the bucket that starts at 200
        self.assertEqual(self.marked([(100, 200)]), [100.0])

    def test_the_last_bucket_holds_the_remainder(self):
        self.assertEqual(self.marked([(1020, 1050)], ymax=1050), [1000.0])

    def test_an_interval_running_off_the_end_stops_at_the_last_bucket(self):
        """`example/` has one: G000436435 is 5,348,036bp, an interval ends 5,348,037."""
        self.assertEqual(self.marked([(850, 1001)]), [800.0, 900.0])

    def test_an_interval_starting_before_the_region_marks_from_its_start(self):
        """`View` clips to the region, so the CLI cannot reach this today.

        It was a silent wipe-out all the same: a start below `ymin` fell
        outside the buckets and cancelled every other mark for the sample.
        """
        self.assertEqual(
            self.marked([(10, 40), (50, 150), (300, 450)], 100, 1000),
            [100.0, 300.0, 400.0],
        )

    def test_a_zero_length_genome_is_rejected_by_name(self):
        """It has no buckets; this died on an IndexError in the axis label."""
        with self.assertRaisesRegex(ValueError, "G1"):
            self.plot([(0, 1)], 0, 0)

    def test_region_buckets_start_at_the_region(self):
        self.assertEqual(self.marked([(1060, 1100), (1150, 1151)], 1050, 1500),
                         [1050.0, 1150.0])

    def test_a_long_genome_keeps_its_published_bucket_edges(self):
        """Above 1Mb the buckets are still np.histogram's 10,000.

        The `example/` genomes are 4.7 and 5.3Mb, and the published plots were
        drawn from them, so their `y` values must not move.
        """
        length = 4_719_737
        _, edges = np.histogram([], bins=10000, range=(0, length))
        self.assertEqual(self.marked([(1000, 1001)], ymax=length), [edges[2]])

    def test_a_bucket_edge_inside_the_last_base_still_marks_its_bucket(self):
        """Edges above 1Mb are fractional; 943.9474 falls inside base 943.

        [900, 944) overlaps the bucket starting there, and the published plots
        marked it. Asking which bucket holds `stop - 1` would not.
        """
        length = 4_719_737
        _, edges = np.histogram([], bins=10000, range=(0, length))
        self.assertEqual(self.marked([(900, 944)], ymax=length),
                         [edges[1], edges[2]])

    def test_a_bucket_starting_at_stop_is_not_marked_on_a_long_genome(self):
        """[0, 200) covers nothing at 200.

        On a 2Mb genome the edges are whole 200bp steps, and the old rule,
        which binned `stop` itself, marked the bucket starting at 200. Such
        genomes lose those rows; `example/`'s edges are never whole, so its
        goldens only gained rows.
        """
        self.assertEqual(self.marked([(0, 200)], ymax=2_000_000), [0.0])


class PerSamplePlotsPerGenomeTests(unittest.TestCase):
    """Each genome's plots see that genome's rows, and only those.

    `per_sample_plots` used to hand every plotting call the *whole* positions
    table, and each call then filtered it for its one genome with a numpy
    string comparison. On a real study -- 9,388 genomes, 29.7M intervals --
    that was four full scans per genome, and `per-sample` ran 2.5x slower than
    the polars release it replaced (6.1 h against 2.5 h). `example/` has two
    genomes, so nothing in the golden suite could see it.
    """

    #: G2 is covered by every sample but the last; G1 by all of them. Twelve
    #: samples in one group clears `coverage_curve`'s minimum of ten, so the
    #: Monte Carlo path actually runs.
    N = 12

    def setUp(self):
        self.d = mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        rows = []
        for i in range(self.N):
            sid = f"S{i:02d}"
            rows.append(("G1", 10 + i, 500 + i, sid))
            if i < self.N - 1:
                rows.append(("G2", 100 + i, 900 + i, sid))
        con = duckdb.connect()
        con.sql("CREATE TABLE p (genome_id VARCHAR, start UINTEGER, "
                "stop UINTEGER, sample_id VARCHAR)")
        con.executemany("INSERT INTO p VALUES (?, ?, ?, ?)", rows)
        base = f"{self.d}/db"
        con.sql(f"COPY p TO '{base}.covered_positions.parquet' (FORMAT PARQUET)")
        con.sql(f"""COPY (SELECT sample_id, genome_id,
                                 SUM(stop - start)::UINTEGER AS covered,
                                 1000::BIGINT AS length,
                                 (SUM(stop - start)::UINTEGER / 1000::BIGINT) * 100
                                     AS percent_covered
                          FROM p GROUP BY ALL)
                    TO '{base}.coverage.parquet' (FORMAT PARQUET)""")
        con.close()
        with open(f"{self.d}/md.tsv", "w") as fp:
            fp.write("sample_id\tgrp\n")
            fp.writelines(f"S{i:02d}\tx\n" for i in range(self.N))
        with open(f"{self.d}/features.tsv", "w") as fp:
            fp.write("genome_id\nG1\nG2\n")
        self.view = View(base, f"{self.d}/md.tsv", f"{self.d}/features.tsv")
        self.addCleanup(self.view.close)

    def run_plots(self, monte=None):
        _plot.per_sample_plots(self.view, "grp", f"{self.d}/out", monte, 5, False)

    def test_each_plot_receives_only_its_genomes_positions(self):
        seen = []

        def record(name):
            def spy(*args, **kwargs):
                positions, target = args[3], args[4]
                if name == "position_plot":
                    positions, target = args[2], args[3]
                seen.append((name, target, set(positions["genome_id"])))
            return spy

        with mock.patch.object(_plot, "coverage_curve", record("coverage_curve")), \
             mock.patch.object(_plot, "position_plot", record("position_plot")):
            self.run_plots()

        self.assertEqual(len(seen), 8, "2 genomes x (2 curves + 2 position plots)")
        for name, target, genomes in seen:
            with self.subTest(call=name, target=target):
                self.assertEqual(genomes, {target})

    def test_sort_by_value_reaches_both_position_plots(self):
        """The PNG and the scaled `.tsv.gz` must lay groups out alike.

        Only the scaled plot writes data, so a flag dropped on the way to the
        unscaled one would leave the PNG disagreeing with its own `.tsv.gz`.
        """
        seen = []

        def spy(*args, **kwargs):
            seen.append(kwargs.get("sort_by_value"))

        with mock.patch.object(_plot, "coverage_curve"), \
             mock.patch.object(_plot, "position_plot", spy):
            _plot.per_sample_plots(self.view, "grp", f"{self.d}/out", None, 5,
                                   False, sort_by_value=True)

        self.assertEqual(seen, [True] * 4, "2 genomes x 2 position plots")

    def test_unfocused_monte_carlo_draws_from_samples_of_any_genome(self):
        """S11 covers only G1, and must still be in G2's unfocused pool.

        "unfocused" means *any* sample with coverage of *any* genome. Handing
        each genome only its own rows must not narrow that to the samples
        covering the target -- which is what "focused" means -- and the
        envelope is unseeded, so nothing else would notice.
        """
        pools = []
        real_rng = np.random.default_rng

        class Recording:
            def __init__(self):
                self.rng = real_rng(0)

            def permutation(self, values):
                pools.append(set(values))
                return self.rng.permutation(values)

        with mock.patch.object(_plot.np.random, "default_rng", Recording):
            self.run_plots(monte="unfocused")

        everyone = {f"S{i:02d}" for i in range(self.N)}
        self.assertTrue(pools, "the Monte Carlo path never ran")
        for pool in pools:
            self.assertEqual(pool, everyone)

    def test_a_genome_with_no_large_group_leaves_no_figure_open(self):
        """Most genomes in a real study have no group of ten samples.

        `coverage_curve` opened a figure and then returned early for those
        without closing it. On 9,337 genomes, 7,056 took that path, twice
        each, and every figure stayed in memory until the process ended.
        """
        import matplotlib.pyplot as plt

        plt.close("all")
        coverage = self.view.coverages().fetchnumpy()
        positions = self.view.positions().fetchnumpy()
        _plot.coverage_curve(
            self.view.con,
            self.view.metadata().fetchnumpy(),
            coverage,
            positions,
            "G1",
            "grp",
            f"{self.d}/out",
            "G1",
            False,
            accumulate=True,
            min_group_size=self.N + 1,
            sample_universe=np.unique(coverage["sample_id"]),
        )
        self.assertEqual(plt.get_fignums(), [])


#: Machado, Oliveira & Fernandes (2009) dichromacy at severity 1.0, applied to
#: linear RGB. This is the simulation the dataviz skill's `validate_palette.js`
#: uses, and the thresholds below are calibrated to it.
MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
}

#: OKLab Delta E x 100 that any two groups must keep: as a protanope or
#: deuteranope sees them, and under normal vision.
CVD_MIN_DELTA_E = 8.0
NORMAL_MIN_DELTA_E = 15.0


def _linear_rgb(hex_color):
    srgb = np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _oklab(rgb):
    lms = np.cbrt(np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                            [0.2119034982, 0.6806995451, 0.1073969566],
                            [0.0883024619, 0.2817188376, 0.6299787005]]) @ rgb)
    return np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                     [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]]) @ lms


def delta_e(a, b, deficiency=None):
    """OKLab Delta E x 100 between two hex colours, optionally as a dichromat."""
    a, b = _linear_rgb(a), _linear_rgb(b)
    if deficiency is not None:
        a = np.clip(MACHADO[deficiency] @ a, 0, 1)
        b = np.clip(MACHADO[deficiency] @ b, 0, 1)
    return 100 * np.linalg.norm(_oklab(a) - _oklab(b))


class GroupPaletteTests(unittest.TestCase):
    """Any two group colours stay distinct to a colour-blind reader.

    micov overlays its groups, so any two of them can end up side by side. The
    matplotlib default micov used before this fails exactly that: its orange
    (C1) and green (C2) are the same colour under protanopia, and roughly one
    man in twelve has a red-green deficiency.
    """

    def test_the_check_catches_the_palette_it_replaced(self):
        # the validator measures 0.7: indistinguishable
        self.assertLess(delta_e("#ff7f0e", "#2ca02c", "protan"), 1.0)

    def test_the_check_agrees_with_the_palette_validator(self):
        """Reproduce `validate_palette.js --pairs all` on Okabe-Ito's first five.

        It reported 11.0 worst-case colour-blind and 15.6 worst-case normal
        vision. If this port disagreed, the thresholds would not mean what
        they say.
        """
        okabe_ito = ("#0072B2", "#E69F00", "#56B4E9", "#D55E00", "#009E73")
        pairs = list(combinations(okabe_ito, 2))
        worst_cvd = min(delta_e(a, b, d) for a, b in pairs for d in MACHADO)
        worst_normal = min(delta_e(a, b) for a, b in pairs)
        self.assertEqual(round(worst_cvd, 1), 11.0)
        self.assertEqual(round(worst_normal, 1), 15.6)

    def test_every_pair_of_group_colours_is_distinct(self):
        for a, b in combinations(GROUP_COLORS, 2):
            for deficiency in MACHADO:
                with self.subTest(pair=(a, b), vision=deficiency):
                    self.assertGreaterEqual(
                        delta_e(a, b, deficiency), CVD_MIN_DELTA_E
                    )
            with self.subTest(pair=(a, b), vision="normal"):
                self.assertGreaterEqual(delta_e(a, b), NORMAL_MIN_DELTA_E)


class CoverageCurveGroupStyleTests(unittest.TestCase):
    """How `coverage_curve` keeps up to ten overlaid groups apart.

    Only five colours stay distinct on every pair, so groups six to ten reuse
    them as dashed lines. Groups past the tenth have never been plotted and
    still are not, because adding one would change the published `.ks.csv`
    rows. What changes is that they are no longer dropped silently: a study
    could otherwise lose a group from its figure and its KS table without
    anyone noticing.
    """

    def setUp(self):
        self.d = mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)

    def curve(self, sizes, min_group_size=1):
        """Draw a non-cumulative curve for one group of each size in `sizes`.

        Groups are named g00, g01, ... so that sorted order is index order.
        Returns the (colour, linestyle) of each line and the legend text.
        """
        metadata, coverage, positions = [], [], []
        for g, size in enumerate(sizes):
            for i in range(size):
                sid = f"S{g:02d}_{i}"
                metadata.append((sid, f"g{g:02d}"))
                coverage.append([sid, "G1", 10 + i, 1000, (10 + i) / 1000 * 100])
                positions.append([sid, "G1", 0, 10 + i])
        metadata = {
            "sample_id": np.array([m[0] for m in metadata], dtype=object),
            "grp": np.array([m[1] for m in metadata], dtype=object),
        }
        coverage = table(COVERAGE_COLUMNS, coverage)

        drawn = {}

        def capture(*args, **kwargs):
            ax = plt.gca()
            drawn["lines"] = [(to_hex(ln.get_color()), ln.get_linestyle())
                              for ln in ax.get_lines()]
            drawn["legend"] = [t.get_text() for t in ax.get_legend().get_texts()]

        with mock.patch.object(_plot.plt, "savefig", capture):
            _plot.coverage_curve(
                None,
                metadata,
                coverage,
                table(POSITION_COLUMNS, positions),
                "G1",
                "grp",
                f"{self.d}/out",
                "name1",
                False,
                min_group_size=min_group_size,
                sample_universe=np.unique(coverage["sample_id"]),
            )
        return drawn.get("lines", []), drawn.get("legend", [])

    def test_groups_six_to_ten_reuse_the_colours_dashed(self):
        lines, _ = self.curve([2] * 10)
        expected = [(GROUP_COLORS[i % 5].lower(), "-" if i < 5 else "--")
                    for i in range(10)]
        self.assertEqual(lines, expected)

    def test_an_eleventh_group_is_named_in_a_warning(self):
        with self.assertLogs("micov", level="WARNING") as logged:
            lines, legend = self.curve([2] * 12)
        self.assertEqual(len(lines), 10, "groups past the tenth stay unplotted")
        self.assertNotIn("g10 (n=2)", legend)
        message = "\n".join(logged.output)
        for expected in ("G1", "name1", "non-cumulative", "grp", "g10", "g11"):
            self.assertIn(expected, message)

    def test_ten_groups_report_nothing(self):
        with self.assertNoLogs("micov", level="WARNING"):
            self.curve([2] * 10)

    def test_a_group_too_small_to_plot_is_not_reported(self):
        """g10 has two samples against a minimum of three: the cap loses nothing.

        Without this, a metadata column with many rare values would print a
        line for nearly every genome of a large study.
        """
        with self.assertNoLogs("micov", level="WARNING"):
            self.curve([3] * 10 + [2], min_group_size=3)


class PositionPlotGroupColourTests(unittest.TestCase):
    """Position plots colour groups like the curves, and name every block.

    Groups six and up reuse colours here as well, and a block of samples
    cannot be dashed. What tells two same-coloured groups apart is the group
    name under each block, so that is pinned too.
    """

    def setUp(self):
        self.d = mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)

    def test_blocks_take_the_group_colours_and_are_named(self):
        # g00 is the largest, so blocks are laid out g05 (1 sample) .. g00 (6)
        sizes = [6, 5, 4, 3, 2, 1]
        metadata, coverage, positions = [], [], []
        for g, size in enumerate(sizes):
            for i in range(size):
                sid = f"S{g:02d}_{i}"
                metadata.append((sid, f"g{g:02d}"))
                coverage.append([sid, "G1", 10, 1000, 1.0])
                positions.append([sid, "G1", 0, 10])
        metadata = {
            "sample_id": np.array([m[0] for m in metadata], dtype=object),
            "grp": np.array([m[1] for m in metadata], dtype=object),
        }

        drawn = {}

        def capture(*args, **kwargs):
            ax = plt.gca()
            drawn["colours"] = [to_hex(c.get_color()[0]) for c in ax.collections]
            drawn["names"] = [t.get_text() for t in ax.get_xticklabels()]

        with mock.patch.object(_plot.plt, "savefig", capture):
            _plot.position_plot(
                metadata,
                table(COVERAGE_COLUMNS, coverage),
                table(POSITION_COLUMNS, positions),
                "G1",
                "grp",
                f"{self.d}/out",
                "name1",
                0,
                1000,
            )

        layout = range(len(sizes) - 1, -1, -1)
        expected = [GROUP_COLORS[g % 5].lower()
                    for g in layout for _ in range(sizes[g])]
        self.assertEqual(drawn["colours"], expected)
        self.assertEqual(drawn["names"], [f"g{g:02d}" for g in layout])


if __name__ == "__main__":
    unittest.main()
