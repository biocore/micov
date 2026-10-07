"""End-to-end equivalence tests: micov's outputs must not move.

This is the suite the duckdb-miint migration is gated on. Every milestone's
definition of done is "this is still green", so it has to fail when an output
changes and *not* fail when a documented nondeterminism kicks in. The
tolerances live in ``_golden``; ``test_golden_selftest`` proves they are
neither too strict nor too loose.

Why subprocesses rather than click's ``CliRunner``
-------------------------------------------------

Subprocesses exercise the installed console script and its exit codes, which
are themselves part of the frozen CLI surface; an in-process call checks
neither. They were originally forced on this suite -- ``nonqiita_to_parquet``
created ``genome_lengths`` on DuckDB's module-level default connection, so a
second in-process invocation died on a duplicate table -- and M3 fixed that,
but the reason above is the one that still holds.

Two tiers
---------

**Fast** (runs in ``make test``) uses only fixtures under ``test_data/``, so it
works from an unpacked sdist, where ``MANIFEST.in`` has pruned ``example/``.

**Full** (``MICOV_GOLDEN_FULL=1``) adds the whole ``example/`` corpus: 49
samfiles, ``cov-to-parquet``, all four ``per-sample`` variants, and binning.
Roughly two minutes, dominated almost entirely by ``compress``.

Tests that need ``example/`` skip with a reason naming the missing directory
rather than silently passing.
"""

import csv
import gzip
import lzma
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import duckdb

from micov.cli import cli
from micov.tests._golden import (
    assert_file_set,
    assert_gzip_text_equal,
    assert_ks_equal,
    assert_parquet_equal,
    assert_png_plausible,
    assert_tsv_equal_unordered,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "test_data"
GOLDEN = DATA / "golden"
EXAMPLE = HERE.parents[1] / "example"

FULL_TIER = os.environ.get("MICOV_GOLDEN_FULL") == "1"

#: The frozen CLI surface. `per-sample-group` is deliberately absent: click 8.2
#: strips the `_group` suffix from the callback name, and micov accepted that
#: rename rather than keeping the `click<8.2` pin (see ChangeLog.md).
#:
#: `nonqiita-to-parquet` is registered but hidden -- it is the compatibility
#: alias for `cov-to-parquet`, kept so existing scripts keep working. It is in
#: this set because it resolves; `TestCovToParquetAlias` pins that it does not
#: appear in `--help`.
EXPECTED_COMMANDS = frozenset(
    {
        "binning",
        "compress",
        "cov-to-parquet",
        "depth-plot",
        "extract-sample-presence",
        "nonqiita-to-parquet",
        "per-sample",
        "position-plot",
    }
)

#: Removed in M4 along with micov's Qiita support. Named individually rather
#: than left to the set comparison above so a regression says *which* command
#: came back and why that matters.
REMOVED_QIITA_COMMANDS = ("qiita-coverage", "qiita-to-parquet", "consolidate")

BIN_STATS_KEYS = ("genome_id", "dog", "bin_idx")
BIN_VARIANCE_KEYS = ("genome_id", "bin_idx")


def _find_micov():
    """Locate the console script in the *same* environment as this interpreter.

    Resolving next to ``sys.executable`` first, rather than trusting ``PATH``,
    is deliberate: a stale non-editable micov earlier in ``PATH`` silently
    shadowed the working tree during this migration and made a whole audit
    pass measure the wrong code.
    """
    candidate = Path(sys.executable).parent / "micov"
    if candidate.exists():
        return str(candidate)
    return shutil.which("micov")


MICOV = _find_micov()

requires_micov = unittest.skipUnless(
    MICOV is not None,
    f"micov console script not found next to {sys.executable} nor on PATH; "
    'run `pip install -e ".[test]"`',
)
requires_example = unittest.skipUnless(
    EXAMPLE.is_dir(),
    f"golden corpus {EXAMPLE} is absent (MANIFEST.in prunes it from sdists)",
)
requires_full_tier = unittest.skipUnless(
    FULL_TIER, "set MICOV_GOLDEN_FULL=1 to run the full-corpus tier"
)


class MicovCliTestCase(unittest.TestCase):
    """Base class providing a tmpdir and a checked subprocess runner."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def micov(self, *args, stdin_bytes=None, expect_success=True):
        proc = subprocess.run(
            [MICOV, *[str(a) for a in args]],
            input=stdin_bytes,
            capture_output=True,
            check=False,
        )
        if expect_success and proc.returncode != 0:
            rendered = " ".join(str(a) for a in args)
            self.fail(
                f"`micov {rendered}` exited {proc.returncode}\n"
                f"--- stderr ---\n{proc.stderr.decode(errors='replace')}"
            )
        return proc

    def micov_stdout_to(self, path, *args, stdin_bytes=None):
        """Run micov and capture stdout to `path`; `compress` writes there."""
        proc = self.micov(*args, stdin_bytes=stdin_bytes)
        path = Path(path)
        path.write_bytes(proc.stdout)
        return path


class TestCliSurface(unittest.TestCase):
    """The canary. A drift here silently broke the documented command name.

    click 8.2 merged pallets/click#2604, which strips `_group` suffixes when
    deriving a command name. micov had pinned `click<8.2` to hold
    `per-sample-group`, but 8.2 was installed anyway, so the README's command
    had stopped resolving with nothing failing. These assertions make any
    future rename of the frozen surface a test failure.
    """

    def test_registered_command_names_are_exactly_the_frozen_set(self):
        self.assertEqual(set(cli.commands), set(EXPECTED_COMMANDS))

    def test_per_sample_group_no_longer_resolves(self):
        self.assertNotIn("per-sample-group", cli.commands)

    def test_qiita_commands_no_longer_resolve(self):
        """Qiita support was removed in M4, and may be revisited.

        Asserted by name rather than by the set comparison above so that a
        partial revival -- one command back, its module still deleted -- fails
        here instead of somewhere downstream.
        """
        for name in REMOVED_QIITA_COMMANDS:
            with self.subTest(command=name):
                self.assertNotIn(name, cli.commands)

    def test_per_sample_module_is_gone(self):
        """`_per_sample.py` was the last polars breadth path.

        Its only callers were the Qiita commands. If it comes back, `_cov.py`
        is no longer polars-free and the module-name collision with
        `_cov.compress_per_sample` is back with it.
        """
        import importlib

        with self.assertRaises(ImportError):
            importlib.import_module("micov._per_sample")

    def test_per_sample_still_declares_percentile(self):
        """A stale shadowing install lacked this flag; that cost an audit pass."""
        params = {p.name for p in cli.commands["per-sample"].params}
        self.assertIn("percentile", params)

    def test_per_sample_sort_by_metadata_value_is_off_by_default(self):
        """Without the flag, position plots keep their published layout."""
        (flag,) = [p for p in cli.commands["per-sample"].params
                   if p.name == "sort_by_metadata_value"]
        self.assertTrue(flag.is_flag)
        self.assertFalse(flag.default)

    def test_depth_plot_options(self):
        """`depth-plot`'s options are part of the frozen surface from its first
        release (ChangeLog.md)."""
        self.assertEqual(
            {p.name for p in cli.commands["depth-plot"].params},
            {"depth", "breadth", "orfs", "sample_metadata",
             "sample_metadata_column", "features_to_keep", "target_names",
             "output", "highlight", "orf_color_by", "orf_contrast", "memory",
             "threads"},
        )

    def test_binning_still_declares_rank(self):
        """`--rank` is a documented no-op, but removing it is a CLI change."""
        params = {p.name for p in cli.commands["binning"].params}
        self.assertIn("rank", params)

    def test_binning_rank_help_says_it_has_no_effect(self):
        """Kept for compatibility, so the help must not promise anything."""
        (rank,) = [p for p in cli.commands["binning"].params if p.name == "rank"]
        self.assertIn("no effect", rank.help)

    @requires_micov
    def test_console_script_runs_every_command(self):
        for name in sorted(EXPECTED_COMMANDS):
            with self.subTest(command=name):
                proc = subprocess.run(
                    [MICOV, name, "--help"], capture_output=True, check=False
                )
                self.assertEqual(
                    proc.returncode,
                    0,
                    f"`micov {name} --help` failed:\n"
                    f"{proc.stderr.decode(errors='replace')}",
                )


@requires_micov
class TestCovToParquetAlias(MicovCliTestCase):
    """`nonqiita-to-parquet` keeps working under its new name.

    The old name only ever meant "not the Qiita one", and with Qiita support
    gone it names nothing. It stays registered as a hidden alias rather than
    being deleted outright because it is the command the README documented for
    two releases and is what existing pipelines call; hidden, because it should
    not be what anyone reaches for now.
    """

    def test_old_name_still_runs(self):
        """The alias is a working command, not just a registered name."""
        prefix = self.tmp / "alias"
        self.micov(
            "nonqiita-to-parquet",
            "--pattern",
            f"{DATA}/mini_sample*.cov",
            "--output",
            prefix,
            "--lengths",
            DATA / "mini_lengths.tsv",
        )
        assert_parquet_equal(
            Path(f"{prefix}.coverage.parquet"),
            GOLDEN / "mini.coverage.parquet",
        )

    def test_old_name_is_hidden_and_new_name_is_not(self):
        listing = self.micov("--help").stdout.decode()
        self.assertIn("cov-to-parquet", listing)
        self.assertNotIn("nonqiita-to-parquet", listing)

    def test_both_names_reach_the_same_callback(self):
        """An alias that drifted into a second implementation is worse than none."""
        self.assertIs(
            cli.commands["nonqiita-to-parquet"].callback,
            cli.commands["cov-to-parquet"].callback,
        )


@requires_micov
class TestCompressFastTier(MicovCliTestCase):
    """`compress` writes the parquet pair from SAM/BAM.

    The BED3 `.cov` output mode and the two TSV summary modes are gone: micov
    now goes SAM -> parquet in one step. `.cov` remains *readable* via
    `cov-to-parquet` -- only `compress` stopped writing it.
    """

    LENGTHS = DATA / "lengths.tsv"

    def compress(self, output, *args, **kwargs):
        self.micov(
            "compress", "--lengths", self.LENGTHS, "--output", output,
            *args, **kwargs,
        )
        return Path(f"{output}.coverage.parquet")

    def intervals(self, base):
        """The covered positions, ordered, for comparison across two runs."""
        import duckdb

        return duckdb.sql(
            f"SELECT genome_id, start, stop "
            f"FROM '{base}.covered_positions.parquet' ORDER BY 1, 2, 3"
        ).fetchall()

    def test_stdin_and_data_flag_agree(self):
        """The pipe idiom and --data must produce identical intervals.

        This is the R1 case: headerless SAM on stdin has no reference map, and
        htslib needs one. It works only because --lengths supplies the map up
        front, so the stream is read once and never rewound.
        """
        sam = DATA / "test.sam.xz"
        self.compress(self.tmp / "from_data", "--data", sam)
        self.compress(
            self.tmp / "from_stdin",
            "--sample-id", "test",
            stdin_bytes=lzma.open(sam).read(),
        )

        self.assertEqual(
            self.intervals(self.tmp / "from_stdin"),
            self.intervals(self.tmp / "from_data"),
        )

    def test_writes_both_parquet_files(self):
        self.compress(self.tmp / "out", "--data", DATA / "test.sam.xz")

        self.assertTrue((self.tmp / "out.coverage.parquet").exists())
        self.assertTrue((self.tmp / "out.covered_positions.parquet").exists())

    def test_sample_id_defaults_to_the_filename(self):
        """`coverage.parquet` is keyed by sample_id, so it cannot be blank.

        The README's loop names outputs after the SAM stem, so taking it from
        the filename is what a caller means by omitting the flag.
        """
        import duckdb

        self.compress(self.tmp / "out", "--data", DATA / "test.sam.xz")
        observed = duckdb.sql(
            f"SELECT DISTINCT sample_id FROM '{self.tmp}/out.coverage.parquet'"
        ).fetchall()

        self.assertEqual(observed, [("test",)])

    def test_sample_id_is_required_when_reading_stdin(self):
        """There is no filename to fall back on, so this must not guess."""
        proc = self.micov(
            "compress", "--lengths", self.LENGTHS,
            "--output", self.tmp / "out",
            stdin_bytes=lzma.open(DATA / "test.sam.xz").read(),
            expect_success=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"--sample-id", proc.stderr)

    def test_lengths_is_required(self):
        """Required for SAM because it doubles as htslib's reference map.

        A second approved exception to the frozen CLI surface, recorded in
        ChangeLog.md alongside the `per-sample` rename.
        """
        proc = self.micov(
            "compress", "--data", DATA / "test.sam.xz",
            "--output", self.tmp / "out",
            expect_success=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"--lengths", proc.stderr)

    def test_bed3_input_is_no_longer_accepted(self):
        """`.cov` aggregation moved to cov-to-parquet.

        Keeping BED3 here would have meant two commands writing the same
        frozen format from the same input.

        The message is asserted, not just the exit code. htslib reads a BED3
        file as SAM without complaining and simply yields nothing, so the
        failure micov raises here is the *only* thing that tells the user
        their input went to the wrong command -- and it names that command, so
        a rename that misses the string strands them. M4 renamed it once
        already.
        """
        cov = self.tmp / "in.cov"
        cov.write_text("genome_id\tstart\tstop\nG1\t1\t10\n")
        proc = self.micov(
            "compress", "--data", cov, "--lengths", self.LENGTHS,
            "--sample-id", "s", "--output", self.tmp / "out",
            expect_success=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        message = proc.stderr.decode(errors="replace")
        self.assertIn("cov-to-parquet", message)
        self.assertIn("cov-to-parquet", set(cli.commands))


@requires_micov
class TestParquetFastTier(MicovCliTestCase):
    """`cov-to-parquet` over the mini corpus.

    Since M4 this is the *only* producer of the parquet pair from `.cov`
    input -- `qiita-to-parquet`, which wrote the committed `example/parquet/`
    files, is gone and that corpus was regenerated here. So these goldens are
    now the whole of micov's coverage for the `.cov` aggregation path.
    """

    def build(self):
        prefix = self.tmp / "mini"
        self.micov(
            "cov-to-parquet",
            "--pattern",
            f"{DATA}/mini_sample*.cov",
            "--output",
            prefix,
            "--lengths",
            DATA / "mini_lengths.tsv",
        )
        return prefix

    def test_matches_goldens(self):
        prefix = self.build()
        for part in ("coverage", "covered_positions"):
            with self.subTest(part=part):
                assert_parquet_equal(
                    Path(f"{prefix}.{part}.parquet"),
                    GOLDEN / f"mini.{part}.parquet",
                )

    def test_can_be_invoked_twice_in_one_process(self):
        """Two in-process invocations must both succeed.

        This asserted the opposite until M3. The command created
        `genome_lengths` on DuckDB's module-level default connection -- a
        process-global catalog -- so a second call died with "Table with name
        genome_lengths already exists". The old test pinned that as though it
        were a contract; it is a defect, and enshrining it in the interface was
        the wrong call. Going through `_miint.connection()` gives each
        invocation its own catalog and fixes it.

        The suite still uses subprocesses everywhere else, deliberately: they
        exercise the console script and its exit codes, which in-process calls
        cannot.
        """
        from micov.cli import cov_to_parquet

        args = [
            "--pattern",
            f"{DATA}/mini_sample*.cov",
            "--lengths",
            str(DATA / "mini_lengths.tsv"),
        ]
        for output in ("one", "two"):
            result = cov_to_parquet.main(
                [*args, "--output", str(self.tmp / output)],
                standalone_mode=False,
            )
            self.assertIsNone(result)

        # and both runs actually produced their files, so "succeeded" is not
        # just "did not raise"
        for output in ("one", "two"):
            self.assertTrue((self.tmp / f"{output}.coverage.parquet").exists())


@requires_micov
class TestSamplePresenceFastTier(MicovCliTestCase):
    """The three-state presence output.

    `not applicable` is produced by a Polars `fill_null` in
    `_view.sample_presence_absence` that **M10 deletes**. The `example/`
    corpus cannot reach that state -- all 49 of its samples cover both
    genomes -- so this mini corpus is the only thing standing between M10 and
    a silently removed behavior.
    """

    def run_presence(self):
        prefix = self.tmp / "mini"
        self.micov(
            "cov-to-parquet",
            "--pattern",
            f"{DATA}/mini_sample*.cov",
            "--output",
            prefix,
            "--lengths",
            DATA / "mini_lengths.tsv",
        )
        out = self.tmp / "presence.tsv"
        self.micov(
            "extract-sample-presence",
            "--parquet-coverage",
            prefix,
            "--sample-metadata",
            DATA / "mini_sample_metadata.tsv",
            "--features-to-keep",
            DATA / "mini_regions.tsv",
            "--output",
            out,
        )
        return out

    def test_matches_golden(self):
        assert_tsv_equal_unordered(
            self.run_presence(),
            GOLDEN / "mini_presence.tsv",
            sort_keys=("sample_id",),
        )

    def test_all_three_states_are_produced(self):
        """Asserted independently of the golden, so refreshing one cannot
        quietly drop a state."""
        rows = self.run_presence().read_text().splitlines()
        header, body = rows[0].split("\t"), [r.split("\t") for r in rows[1:]]
        self.assertEqual(header, ["sample_id", "G000000001_4000_5000"])
        states = dict(body)
        self.assertEqual(
            states,
            {
                "mini_sampleA": "present",
                "mini_sampleB": "absent",
                "mini_sampleC": "not applicable",
            },
        )

    def test_regions_are_required(self):
        """Without positions the command must fail loudly, not emit nothing."""
        prefix = self.tmp / "mini"
        self.micov(
            "cov-to-parquet",
            "--pattern",
            f"{DATA}/mini_sample*.cov",
            "--output",
            prefix,
            "--lengths",
            DATA / "mini_lengths.tsv",
        )
        no_regions = self.tmp / "no_regions.tsv"
        no_regions.write_text("genome_id\nG000000001\n")
        proc = self.micov(
            "extract-sample-presence",
            "--parquet-coverage",
            prefix,
            "--sample-metadata",
            DATA / "mini_sample_metadata.tsv",
            "--features-to-keep",
            no_regions,
            "--output",
            self.tmp / "unused.tsv",
            expect_success=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"without positions", proc.stderr)


@requires_micov
class TestBinningAndPlotsFastTier(MicovCliTestCase):
    def test_binning_emits_expected_files_and_headers(self):
        prefix = self.tmp / "mini"
        self.micov(
            "cov-to-parquet",
            "--pattern",
            f"{DATA}/mini_sample*.cov",
            "--output",
            prefix,
            "--lengths",
            DATA / "mini_lengths.tsv",
        )
        outdir = self.tmp / "binning"
        outdir.mkdir()
        self.micov(
            "binning",
            "--parquet-coverage",
            prefix,
            "--sample-metadata",
            DATA / "mini_sample_metadata.tsv",
            "--metadata-variable",
            "group",
            "--outdir",
            outdir,
            "--bin-num",
            50,
        )
        assert_file_set(
            outdir, {"stats_bins.tsv", "stats_by_variance_of_sample_hits.tsv"}
        )
        bins = (outdir / "stats_bins.tsv").read_text().splitlines()
        self.assertEqual(
            bins[0],
            "genome_id\tgroup\tbin_idx\tread_hits\tsample_hits\tsamples"
            "\tbin_start\tbin_stop",
        )
        variance = (
            (outdir / "stats_by_variance_of_sample_hits.tsv").read_text().splitlines()
        )
        self.assertEqual(
            variance[0], "genome_id\tbin_idx\tbin_start\tbin_stop\tsample_hits_std"
        )
        self.assertGreater(len(bins), 1)
        self.assertGreater(len(variance), 1)

    def test_per_sample_sort_by_metadata_value_orders_groups_numerically(self):
        """A depth column has to read left to right as 5, 30, 270.

        Metadata is text, so without the flag these one-sample groups sit in
        text order: 270, 30, 5. G000000002 is the genome all three cover.
        """
        prefix = self.tmp / "mini"
        self.micov(
            "cov-to-parquet",
            "--pattern",
            f"{DATA}/mini_sample*.cov",
            "--output",
            prefix,
            "--lengths",
            DATA / "mini_lengths.tsv",
        )
        metadata = self.tmp / "depth.tsv"
        metadata.write_text(
            "sample_id\tdepth\n"
            "mini_sampleA\t30\n"
            "mini_sampleB\t5\n"
            "mini_sampleC\t270\n"
        )
        features = self.tmp / "features.tsv"
        features.write_text("genome_id\nG000000002\n")
        self.micov(
            "per-sample",
            "--parquet-coverage",
            prefix,
            "--sample-metadata",
            metadata,
            "--sample-metadata-column",
            "depth",
            "--features-to-keep",
            features,
            "--output",
            self.tmp / "out",
            "--sort-by-metadata-value",
        )
        scaled = (self.tmp / "out.G000000002.G000000002.depth"
                  ".position-plot-scaled.tsv.gz")
        with gzip.open(scaled, "rt") as fp:
            rows = sorted(csv.DictReader(fp, delimiter="\t"),
                          key=lambda row: int(row["x"]))
        self.assertEqual(list(dict.fromkeys(row["group"] for row in rows)),
                         ["5", "30", "270"])

    #: `mini_sampleA.cov` covers exactly these two genomes.
    POSITION_PLOT_GENOMES: ClassVar[tuple] = ("G000000001", "G000000002")

    def run_position_plot(self, outdir, from_stdin=False):
        args = ["position-plot", "--lengths", DATA / "mini_lengths.tsv",
                "--output", outdir / "plot"]
        cov = DATA / "mini_sampleA.cov"
        if from_stdin:
            return self.micov(*args, stdin_bytes=cov.read_bytes())
        return self.micov(*args, "--positions", cov)

    def test_position_plot_writes_a_png_per_genome(self):
        outdir = self.tmp / "pp"
        outdir.mkdir()
        self.run_position_plot(outdir)
        pngs = sorted(outdir.glob("*.png"))
        self.assertEqual(len(pngs), 2, f"expected one PNG per genome, got {pngs}")
        for png in pngs:
            with self.subTest(png=png.name):
                assert_png_plausible(png)

    def test_position_plot_names_files_after_the_genome(self):
        """The genome ID has to appear in the filename, and only the ID.

        These were written as `plot.('G000000001',).position-plot.png` -- a
        literal Python tuple repr, because polars `group_by` yields tuple keys
        and the name went straight into an f-string. The old test globbed
        `*.png` and counted two, so it never saw this. Whatever writes these
        files, the name is what a user greps for.
        """
        outdir = self.tmp / "pp"
        outdir.mkdir()
        self.run_position_plot(outdir)

        self.assertEqual(
            sorted(p.name for p in outdir.glob("*.png")),
            [f"plot.{genome}.position-plot.png"
             for genome in self.POSITION_PLOT_GENOMES],
        )

    def test_position_plot_reads_stdin(self):
        """`--positions` is optional, so the stdin path is advertised.

        It did not work: micov peeked at the first line to sniff a header and
        then rewound, which a pipe cannot do, so this died with
        `io.UnsupportedOperation: underlying stream is not seekable`. A CLI
        that offers a path has to have one.
        """
        outdir = self.tmp / "pp"
        outdir.mkdir()
        self.run_position_plot(outdir, from_stdin=True)

        self.assertEqual(
            sorted(p.name for p in outdir.glob("*.png")),
            [f"plot.{genome}.position-plot.png"
             for genome in self.POSITION_PLOT_GENOMES],
        )


def depth_plot_args(output, column="group", breadth=True, orfs=True,
                    features="dp_regions.tsv", names=True, metadata=None):
    """A `depth-plot` command line over the dp fixture (test_data/README.md)."""
    args = ["depth-plot", "--depth", DATA / "dp_depth.parquet",
            "--sample-metadata", metadata or DATA / "dp_metadata.tsv",
            "--sample-metadata-column", column,
            "--features-to-keep", DATA / features, "--output", output]
    if breadth:
        args += ["--breadth", DATA / "dp_breadth.parquet"]
    if orfs:
        args += ["--orfs", DATA / "dp_orfs.parquet"]
    if names:
        args += ["--target-names", DATA / "dp_target_names.tsv"]
    return args


@requires_micov
class TestDepthPlotFastTier(MicovCliTestCase):
    """`depth-plot` end to end on the dp fixture. Run A uses everything:
    separate layers, regions, target names, ORFs, two highlights and the
    contrast."""

    GC = "run.s__Circulus_testii.GC.group"
    GL = "run.s__Linearia_testii.GL.group"

    @classmethod
    def setUpClass(cls):
        cls._run_a = tempfile.TemporaryDirectory()
        cls.out_a = Path(cls._run_a.name)
        cls.proc_a = subprocess.run(
            [MICOV, *map(str, depth_plot_args(cls.out_a / "run")),
             "--highlight", "product~phage", "--highlight", "type=rRNA",
             "--orf-contrast"],
            capture_output=True, check=False,
        )
        cls.stderr_a = cls.proc_a.stderr.decode(errors="replace")

    @classmethod
    def tearDownClass(cls):
        cls._run_a.cleanup()

    def orf_rows(self, where):
        return duckdb.execute(
            f"""SELECT depth_q1, depth_median, depth_q3, depth_mean, prevalence,
                       union_breadth
                FROM read_parquet(?) WHERE {where}""",
            [str(self.out_a / "run.group.depth-plot-orfs.parquet")],
        ).fetchall()

    def test_run_a_succeeds(self):
        self.assertEqual(self.proc_a.returncode, 0, self.stderr_a)

    def test_run_a_writes_every_plot_and_one_orf_table(self):
        assert_file_set(self.out_a, {
            f"{self.GC}.depth-plot.png", f"{self.GC}.depth-plot-circular.png",
            f"{self.GC}.depth-plot-detail-1001-1501.png",
            f"{self.GC}.depth-plot-detail-2501-2901.png",
            f"{self.GL}.depth-plot.png", "run.group.depth-plot-orfs.parquet",
        })
        for png in self.out_a.glob("*.png"):
            with self.subTest(png=png.name):
                assert_png_plausible(png)

    def test_run_a_orf_table_matches_the_golden(self):
        assert_parquet_equal(self.out_a / "run.group.depth-plot-orfs.parquet",
                             GOLDEN / "dp.orfs.parquet")

    def test_run_a_orf_table_holds_the_hand_computed_values(self):
        """Read independently of the golden: test_depth works these out."""
        self.assertEqual(self.orf_rows("orf_id = 'gc_3' AND \"group\" = 'case'"),
                         [(0.5, 0.5, 0.5, 0.5, 0.5, 0.75)])
        self.assertEqual(self.orf_rows("orf_id = 'gc_4' AND \"group\" = 'control'"),
                         [(0.25, 0.5, 0.75, 0.5, 0.5, 1.0)])
        self.assertEqual(self.orf_rows("orf_id = 'gl_2' AND \"group\" = 'control'"),
                         [(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)])

    def test_run_a_reports_what_it_left_out_and_why_contrast_is_empty(self):
        for name in ("S6", "S7", "S8", "GX", "GB", "No ORF contrast for GC"):
            with self.subTest(name=name):
                self.assertIn(name, self.stderr_a)
        self.assertNotIn("S9", self.stderr_a)
        genomes = [line for line in self.stderr_a.splitlines() if "genome(s)" in line]
        self.assertFalse([line for line in genomes if "*" in line])

    def test_breadth_defaults_to_depth_and_no_orfs_no_table(self):
        """With one layer, S7 and GX (depth only before) are in both."""
        self.micov(*depth_plot_args(self.tmp / "run", breadth=False, orfs=False,
                                    features="dp_features.tsv", names=False))
        assert_file_set(self.tmp, {
            "run.GC.GC.group.depth-plot.png",
            "run.GC.GC.group.depth-plot-circular.png",
            "run.GL.GL.group.depth-plot.png", "run.GX.GX.group.depth-plot.png",
        })

    def test_four_groups_get_lanes_and_no_ring(self):
        proc = self.micov(*depth_plot_args(self.tmp / "run", column="quad",
                                           orfs=False, names=False))
        assert_file_set(self.tmp, {
            "run.GC.GC.quad.depth-plot.png",
            "run.GC.GC.quad.depth-plot-detail-1001-1501.png",
            "run.GC.GC.quad.depth-plot-detail-2501-2901.png",
            "run.GL.GL.quad.depth-plot.png",
        })
        self.assertIn("No circular plots", proc.stderr.decode())

    def refused(self, args, code, message):
        out = self.tmp / "out"
        out.mkdir()
        proc = self.micov(*args, expect_success=False)
        stderr = proc.stderr.decode(errors="replace")
        self.assertEqual(proc.returncode, code, stderr)
        self.assertIn(message, stderr)
        self.assertEqual(list(out.iterdir()), [])

    def test_refusals_leave_nothing_behind(self):
        """Mistakes on the command line are usage errors (2); mistakes in the
        files are found before any genome is computed (1)."""
        out = self.tmp / "out" / "run"
        nobody = self.tmp / "nobody.tsv"
        nobody.write_text("sample_id\tgroup\nZ1\tcase\n")
        no_sample = self.tmp / "no_sample.parquet"
        duckdb.execute(f"""COPY (SELECT * EXCLUDE (sample_id)
                                 FROM read_parquet('{DATA}/dp_depth.parquet'))
                           TO '{no_sample}' (FORMAT PARQUET)""")
        for why, args, code, message in (
            ("highlight without ORFs",
             [*depth_plot_args(out, orfs=False), "--highlight", "type=CDS"], 2,
             "--orfs"),
            ("a highlight that is not KEY=VALUE",
             [*depth_plot_args(out), "--highlight", "phage"], 2, "KEY=VALUE"),
            ("both colourings",
             [*depth_plot_args(out), "--orf-color-by", "product",
              "--orf-contrast"], 2, "choose one"),
            ("an output directory that does not exist",
             depth_plot_args(self.tmp / "out" / "missing" / "run"), 2,
             "does not exist"),
            ("no sample in common",
             depth_plot_args(out, metadata=nobody), 1, "No sample"),
            ("colour-by with three groups",
             [*depth_plot_args(out, column="trio"), "--orf-color-by", "product"],
             1, "at most two groups"),
            ("contrast with three groups",
             [*depth_plot_args(out, column="trio"), "--orf-contrast"], 1,
             "exactly two groups"),
            ("a layer without sample_id",
             [*depth_plot_args(out)[:2], no_sample, *depth_plot_args(out)[3:]], 1,
             "sample column"),
        ):
            with self.subTest(why):
                self.refused(args, code, message)
                (self.tmp / "out").rmdir()


@requires_micov
@requires_example
@requires_full_tier
class TestCompressFullCorpus(MicovCliTestCase):
    """The whole point of M3: miint ingest must reproduce micov's own output.

    `example/coverages/*.cov.gz` were produced by micov's hand-written CIGAR
    walker and numba interval merge, both of which are now gone. Every one of
    the 49 samples must come back byte-identical through `read_alignments` and
    `compress_intervals`, or the published coverage values move.

    The `.cov` files stay committed as fixtures even though `compress` no
    longer writes that format -- they are still valid input to
    `cov-to-parquet`, and they are the only independent record of what the
    old implementation produced.
    """

    def test_all_samfiles_match_committed_coverages(self):
        import duckdb

        samfiles = sorted((EXAMPLE / "samfiles").glob("*.sam.xz"))
        self.assertEqual(len(samfiles), 49)

        for sam in samfiles:
            sample_id = sam.name[: -len(".sam.xz")]
            with self.subTest(sample=sample_id):
                base = self.tmp / sample_id
                self.micov(
                    "compress",
                    "--data", sam,
                    "--lengths", EXAMPLE / "metadata" / "length.tsv",
                    "--output", base,
                )
                produced = duckdb.sql(
                    f"SELECT genome_id, start, stop "
                    f"FROM '{base}.covered_positions.parquet' ORDER BY 1, 2, 3"
                ).fetchall()

                frozen_path = EXAMPLE / "coverages" / f"{sample_id}.cov.gz"
                frozen = duckdb.sql(
                    f"SELECT genome_id, start, stop FROM read_csv("
                    f"'{frozen_path}', delim='\t', header=true, columns="
                    "{'genome_id': 'VARCHAR', 'start': 'UINTEGER', "
                    "'stop': 'UINTEGER'}) ORDER BY 1, 2, 3"
                ).fetchall()

                self.assertEqual(produced, frozen)


@requires_micov
@requires_example
@requires_full_tier
class TestParquetFullCorpus(MicovCliTestCase):
    """`cov-to-parquet` is now the provenance of `example/parquet/`.

    It was `qiita-to-parquet` until M4, and that command is gone, so the
    corpus was regenerated from `example/coverages/*.cov.gz`. The rows are
    unchanged; `sample_id` moved from the trailing column position to the
    leading one, which is the whole of the difference between the two
    producers.

    This matters more than one test usually does: every downstream golden --
    binning, sample presence, all four `per-sample` variants -- is computed
    from this corpus, so without a producer to check it against, a corrupted
    regeneration would look like a consistent set of "new correct" answers.
    """

    def test_matches_committed(self):
        prefix = self.tmp / "example"
        self.micov(
            "cov-to-parquet",
            "--pattern",
            f"{EXAMPLE}/coverages/*.cov.gz",
            "--output",
            prefix,
            "--lengths",
            EXAMPLE / "metadata" / "length.tsv",
        )
        for part in ("coverage", "covered_positions"):
            with self.subTest(part=part):
                assert_parquet_equal(
                    Path(f"{prefix}.{part}.parquet"),
                    EXAMPLE / "parquet" / f"example.{part}.parquet",
                )


@requires_micov
@requires_example
@requires_full_tier
class TestBinningFullCorpus(MicovCliTestCase):
    def test_matches_committed_binning(self):
        outdir = self.tmp / "binning"
        outdir.mkdir()
        self.micov(
            "binning",
            "--parquet-coverage",
            EXAMPLE / "parquet" / "example",
            "--sample-metadata",
            EXAMPLE / "metadata" / "sample_metadata.txt",
            "--features-to-keep",
            EXAMPLE / "metadata" / "feature_metadata.txt",
            "--metadata-variable",
            "dog",
            "--outdir",
            outdir,
            "--rank",
        )
        assert_tsv_equal_unordered(
            outdir / "stats_bins.tsv",
            EXAMPLE / "binning" / "stats_bins.tsv",
            sort_keys=BIN_STATS_KEYS,
        )
        # sample_hits_std alone is approximate: a standard deviation's
        # summation order is not reproducible outside polars (source #9).
        # Every other column, including the bin bounds, stays exact.
        assert_tsv_equal_unordered(
            outdir / "stats_by_variance_of_sample_hits.tsv",
            EXAMPLE / "binning" / "stats_by_variance_of_sample_hits.tsv",
            sort_keys=BIN_VARIANCE_KEYS,
            float_columns=("sample_hits_std",),
        )


@requires_micov
@requires_example
@requires_full_tier
class TestSamplePresenceFullCorpus(MicovCliTestCase):
    def test_matches_golden(self):
        out = self.tmp / "presence.tsv"
        self.micov(
            "extract-sample-presence",
            "--parquet-coverage",
            EXAMPLE / "parquet" / "example",
            "--sample-metadata",
            EXAMPLE / "metadata" / "sample_metadata.txt",
            "--features-to-keep",
            DATA / "feature_metadata_regions.tsv",
            "--output",
            out,
        )
        assert_tsv_equal_unordered(
            out, GOLDEN / "example_presence.tsv", sort_keys=("sample_id",)
        )


@requires_micov
@requires_example
@requires_full_tier
class TestPerSampleFullCorpus(MicovCliTestCase):
    """All four `per-sample` variants against the four committed plot dirs.

    `_plot.py` is the largest module and has no unit tests, so these data
    goldens are its only guard. PNG *content* is not comparable across
    matplotlib versions, so the plotted values are covered indirectly: the
    `.tsv.gz` position data and the `.ks.csv` statistics.
    """

    VARIANTS: ClassVar[dict] = {
        "plain": ("per_sample_groups", "example", ()),
        "percentile": (
            "per_sample_groups_percentile",
            "example_percentile",
            ("--percentile",),
        ),
        "monte": (
            "per_sample_groups_monte",
            "example",
            ("--monte", "unfocused"),
        ),
        "monte_percentile": (
            "per_sample_groups_monte_percentile",
            "example_percentile",
            ("--monte", "unfocused", "--percentile"),
        ),
    }

    def run_variant(self, name):
        golden_dir, prefix, flags = self.VARIANTS[name]
        outdir = self.tmp / name
        outdir.mkdir()
        self.micov(
            "per-sample",
            "--parquet-coverage",
            EXAMPLE / "parquet" / "example",
            "--sample-metadata",
            EXAMPLE / "metadata" / "sample_metadata.txt",
            "--sample-metadata-column",
            "dog",
            "--features-to-keep",
            EXAMPLE / "metadata" / "feature_metadata.txt",
            "--output",
            outdir / prefix,
            "--plot",
            *flags,
        )
        return outdir, EXAMPLE / "plots" / golden_dir

    def check_variant(self, name):
        outdir, golden = self.run_variant(name)
        assert_file_set(outdir, {p.name for p in golden.iterdir() if p.is_file()})
        for produced in sorted(outdir.iterdir()):
            with self.subTest(artifact=produced.name):
                expected = golden / produced.name
                if produced.name.endswith(".ks.csv"):
                    assert_ks_equal(produced, expected)
                elif produced.name.endswith(".tsv.gz"):
                    assert_gzip_text_equal(produced, expected)
                elif produced.name.endswith(".png"):
                    assert_png_plausible(produced)
                else:
                    self.fail(f"unclassified output artifact: {produced.name}")

    def test_plain(self):
        self.check_variant("plain")

    def test_percentile(self):
        self.check_variant("percentile")

    def test_monte_unfocused(self):
        self.check_variant("monte")

    def test_monte_unfocused_percentile(self):
        self.check_variant("monte_percentile")

    def test_percentile_does_not_alter_data_outputs(self):
        """`--percentile` is a plot-axis change only.

        A mechanical migration could easily couple the axis choice to the
        curve computation; this catches that without needing a golden.
        """
        plain_dir, _ = self.run_variant("plain")
        pct_dir, _ = self.run_variant("percentile")
        pairs = 0
        for plain in sorted(plain_dir.iterdir()):
            if not (plain.name.endswith(".ks.csv") or plain.name.endswith(".tsv.gz")):
                continue
            counterpart = pct_dir / plain.name.replace(
                "example.", "example_percentile.", 1
            )
            with self.subTest(artifact=plain.name):
                self.assertTrue(counterpart.exists(), f"missing {counterpart.name}")
                if plain.name.endswith(".tsv.gz"):
                    assert_gzip_text_equal(counterpart, plain)
                else:
                    assert_ks_equal(counterpart, plain)
                pairs += 1
        self.assertEqual(pairs, 4, "expected 2 .ks.csv and 2 .tsv.gz comparisons")

    def test_monte_leaves_non_monte_ks_rows_unchanged(self):
        """Adding Monte Carlo curves must not perturb the published rows."""
        plain_dir, _ = self.run_variant("plain")
        monte_dir, _ = self.run_variant("monte")

        def deterministic_rows(path):
            rows = path.read_text().splitlines()
            return [r for r in rows[1:] if "Monte Carlo " not in r]

        compared = 0
        for plain in sorted(plain_dir.glob("*.cumulative.ks.csv")):
            monte = monte_dir / plain.name.replace(
                ".cumulative.ks.csv", ".cumulative-monte-unfocused.ks.csv"
            )
            with self.subTest(artifact=plain.name):
                self.assertTrue(monte.exists(), f"missing {monte.name}")
                self.assertEqual(
                    sorted(deterministic_rows(monte)),
                    sorted(deterministic_rows(plain)),
                )
                compared += 1
        self.assertEqual(compared, 2)


if __name__ == "__main__":
    unittest.main()
