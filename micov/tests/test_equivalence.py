"""End-to-end equivalence tests: micov's outputs must not move.

This is the suite the duckdb-miint migration is gated on. Every milestone's
definition of done is "this is still green", so it has to fail when an output
changes and *not* fail when a documented nondeterminism kicks in. The
tolerances live in ``_golden``; ``test_golden_selftest`` proves they are
neither too strict nor too loose.

Why subprocesses rather than click's ``CliRunner``
-------------------------------------------------

``nonqiita_to_parquet`` issues ``duckdb.sql("CREATE TABLE genome_lengths …")``
against the module-level default connection (``cli.py:307``), so a second
in-process invocation fails with a duplicate-table error. Subprocesses also
exercise the installed console script, which is itself part of the frozen CLI
surface.

Two tiers
---------

**Fast** (runs in ``make test``) uses only fixtures under ``test_data/``, so it
works from an unpacked sdist, where ``MANIFEST.in`` has pruned ``example/``.

**Full** (``MICOV_GOLDEN_FULL=1``) adds the whole ``example/`` corpus: 49
samfiles, both Parquet producers, all four ``per-sample`` variants, and
binning. Roughly two minutes, dominated almost entirely by ``compress``.

Tests that need ``example/`` skip with a reason naming the missing directory
rather than silently passing.
"""

import lzma
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

from micov.cli import cli
from micov.tests._golden import (
    assert_cov_equal,
    assert_file_set,
    assert_gzip_text_equal,
    assert_ks_equal,
    assert_parquet_equal,
    assert_png_plausible,
    assert_tgz_equal,
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
EXPECTED_COMMANDS = frozenset(
    {
        "binning",
        "compress",
        "consolidate",
        "extract-sample-presence",
        "nonqiita-to-parquet",
        "per-sample",
        "position-plot",
        "qiita-coverage",
        "qiita-to-parquet",
    }
)

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

    def write_paths_file(self, name, paths):
        """`consolidate --paths` takes a file listing one path per line."""
        target = self.tmp / name
        target.write_text("".join(f"{p}\n" for p in paths))
        return target


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

    def test_per_sample_still_declares_percentile(self):
        """A stale shadowing install lacked this flag; that cost an audit pass."""
        params = {p.name for p in cli.commands["per-sample"].params}
        self.assertIn("percentile", params)

    def test_binning_still_declares_rank(self):
        """`--rank` is a documented no-op, but removing it is a CLI change."""
        params = {p.name for p in cli.commands["binning"].params}
        self.assertIn("rank", params)

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
class TestCompressFastTier(MicovCliTestCase):
    """`compress` against goldens built from test_data/test.sam.xz."""

    def test_stdin_and_data_flag_agree(self):
        """The stdin and --data paths must produce identical intervals.

        Compared order-insensitively: `compress` groups by genome without
        maintain_order, so row order is not stable between runs of the *same*
        input, let alone between two input paths.
        """
        sam = DATA / "test.sam.xz"
        from_data = self.micov_stdout_to(
            self.tmp / "from_data.tsv", "compress", "--data", sam
        )
        from_stdin = self.micov_stdout_to(
            self.tmp / "from_stdin.tsv",
            "compress",
            stdin_bytes=lzma.open(sam).read(),
        )
        assert_cov_equal(from_stdin, from_data)

    def test_plain_matches_golden(self):
        produced = self.micov_stdout_to(
            self.tmp / "out.tsv", "compress", "--data", DATA / "test.sam.xz"
        )
        assert_cov_equal(produced, GOLDEN / "compress_plain.tsv")

    def test_with_lengths_matches_golden(self):
        produced = self.micov_stdout_to(
            self.tmp / "out.tsv",
            "compress",
            "--data",
            DATA / "test.sam.xz",
            "--lengths",
            DATA / "lengths.tsv",
        )
        assert_tsv_equal_unordered(
            produced, GOLDEN / "compress_lengths.tsv", sort_keys=("genome_id",)
        )

    def test_with_lengths_and_taxonomy_matches_golden(self):
        produced = self.micov_stdout_to(
            self.tmp / "out.tsv",
            "compress",
            "--data",
            DATA / "test.sam.xz",
            "--lengths",
            DATA / "lengths.tsv",
            "--taxonomy",
            DATA / "taxonomy.tsv",
        )
        assert_tsv_equal_unordered(
            produced,
            GOLDEN / "compress_lengths_taxonomy.tsv",
            sort_keys=("genome_id",),
        )

    def test_taxonomy_requires_every_genome(self):
        """Documents a real constraint: the taxonomy cannot be a subset.

        `set_taxonomy_as_id` raises when any covered genome lacks a lineage,
        which is why the taxonomy fixture has 232 rows and not two.
        """
        partial = self.tmp / "partial_taxonomy.tsv"
        partial.write_text(
            "genome_id\ttaxonomy\nG000011065\tk__Bacteria; s__Only one\n"
        )
        proc = self.micov(
            "compress",
            "--data",
            DATA / "test.sam.xz",
            "--lengths",
            DATA / "lengths.tsv",
            "--taxonomy",
            partial,
            expect_success=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(b"unrepresented in", proc.stderr)

    def test_multiple_cov_files_aggregate_when_headers_are_stripped(self):
        """Concatenated `.cov` input is merged per genome across samples.

        Only the first file may carry its header -- see the companion test
        below for why.
        """
        first = (DATA / "mini_sampleA.cov").read_bytes()
        body = (DATA / "mini_sampleB.cov").read_bytes().splitlines(keepends=True)[1:]
        produced = self.micov_stdout_to(
            self.tmp / "merged.tsv", "compress", stdin_bytes=first + b"".join(body)
        )
        rows = produced.read_text().splitlines()
        self.assertEqual(rows[0], "genome_id\tstart\tstop")
        self.assertEqual(
            set(rows[1:]),
            {
                "G000000001\t100\t500",
                "G000000001\t4200\t4500",
                "G000000002\t100\t500",
                "G000000002\t600\t900",
            },
        )

    def test_concatenating_cov_files_with_headers_currently_fails(self):
        """**Pins a bug, not desired behavior.**

        `README.md` documents::

            zcat run1/sample1.cov.gz run2/sample1.cov.gz | micov compress

        as the way to aggregate a sample's coverage across runs. But `micov
        compress` writes a `genome_id\\tstart\\tstop` header, and the BED
        reader does not skip repeated headers mid-stream, so the second file's
        header reaches the parser as data and polars fails to cast `start` to
        u32. Reproduced with the committed `example/coverages/*.cov.gz`, so it
        affects the documented workflow and not just this fixture.

        This test asserts the *current* failure so the migration cannot change
        it unnoticed -- in either direction. If a milestone makes this pass,
        that is an intentional fix and this test plus the README should be
        updated together. Logged in MIGRATE-TO-MIINT.md §8.
        """
        payload = b"".join(
            (DATA / f"mini_sample{s}.cov").read_bytes() for s in ("A", "B")
        )
        proc = self.micov("compress", stdin_bytes=payload, expect_success=False)
        self.assertNotEqual(
            proc.returncode, 0, "concatenated headers now parse -- see docstring"
        )
        self.assertIn(b"could not parse", proc.stderr)


@requires_micov
class TestParquetFastTier(MicovCliTestCase):
    """`nonqiita-to-parquet` over the mini corpus.

    This is the only golden coverage `nonqiita-to-parquet` has. The committed
    `example/parquet/` files came from `qiita-to-parquet`, and the two commands
    disagree on `coverage.parquet` column order, so they are not
    interchangeable despite how README steps 3 and 4 read.
    """

    def build(self):
        prefix = self.tmp / "mini"
        self.micov(
            "nonqiita-to-parquet",
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

    def test_cannot_be_invoked_twice_in_one_process(self):
        """Guards the reason this suite uses subprocesses at all.

        `nonqiita_to_parquet` creates `genome_lengths` on the module-level
        DuckDB connection, so a second in-process call fails. If that is ever
        fixed, this test fails and the CliRunner option reopens.
        """
        import duckdb

        from micov.cli import nonqiita_to_parquet

        args = [
            "--pattern",
            f"{DATA}/mini_sample*.cov",
            "--lengths",
            str(DATA / "mini_lengths.tsv"),
        ]
        first = nonqiita_to_parquet.main(
            [*args, "--output", str(self.tmp / "one")],
            standalone_mode=False,
        )
        self.assertIsNone(first)
        # duckdb.CatalogException: Table with name "genome_lengths"
        # already exists!
        with self.assertRaises(duckdb.CatalogException):
            nonqiita_to_parquet.main(
                [*args, "--output", str(self.tmp / "two")],
                standalone_mode=False,
            )


@requires_micov
class TestQiitaRoundTripFastTier(MicovCliTestCase):
    """`consolidate` then `qiita-coverage`, neither of which had a golden."""

    def test_consolidate_then_qiita_coverage_match_goldens(self):
        covs = [DATA / f"mini_sample{s}.cov" for s in ("A", "B", "C")]
        paths = self.write_paths_file("paths.txt", covs)
        tgz = self.tmp / "mini_consolidated.tgz"
        self.micov(
            "consolidate",
            "--paths",
            paths,
            "--output",
            tgz,
            "--lengths",
            DATA / "mini_lengths.tsv",
        )
        self.micov(
            "qiita-coverage",
            "--qiita-coverages",
            tgz,
            "--output",
            self.tmp / "mini_qiita",
            "--lengths",
            DATA / "mini_lengths.tsv",
        )
        assert_tsv_equal_unordered(
            self.tmp / "mini_qiita.coverage.tsv",
            GOLDEN / "mini_qiita.coverage.tsv",
            sort_keys=("genome_id",),
        )
        assert_cov_equal(
            self.tmp / "mini_qiita.covered-positions.tsv",
            GOLDEN / "mini_qiita.covered-positions.tsv",
        )

    def test_consolidated_archive_layout(self):
        """The Qiita `.tgz` layout is a frozen format Qiita itself reads."""
        import tarfile

        covs = [DATA / f"mini_sample{s}.cov" for s in ("A", "B", "C")]
        paths = self.write_paths_file("paths.txt", covs)
        tgz = self.tmp / "out.tgz"
        self.micov(
            "consolidate",
            "--paths",
            paths,
            "--output",
            tgz,
            "--lengths",
            DATA / "mini_lengths.tsv",
        )
        with tarfile.open(tgz) as tar:
            names = {m.name for m in tar.getmembers() if m.isfile()}
        self.assertEqual(
            names,
            {
                "coverages/mini_sampleA.cov",
                "coverages/mini_sampleB.cov",
                "coverages/mini_sampleC.cov",
                "artifact.cov",
                "coverage_percentage.txt",
            },
        )


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
            "nonqiita-to-parquet",
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
            "nonqiita-to-parquet",
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
            "nonqiita-to-parquet",
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

    def test_position_plot_writes_a_png_per_genome(self):
        outdir = self.tmp / "pp"
        outdir.mkdir()
        self.micov(
            "position-plot",
            "--positions",
            DATA / "mini_sampleA.cov",
            "--lengths",
            DATA / "mini_lengths.tsv",
            "--output",
            outdir / "plot",
        )
        pngs = sorted(outdir.glob("*.png"))
        self.assertEqual(len(pngs), 2, f"expected one PNG per genome, got {pngs}")
        for png in pngs:
            with self.subTest(png=png.name):
                assert_png_plausible(png)


@requires_micov
@requires_example
@requires_full_tier
class TestCompressFullCorpus(MicovCliTestCase):
    def test_all_samfiles_match_committed_coverages(self):
        samfiles = sorted((EXAMPLE / "samfiles").glob("*.sam.xz"))
        self.assertEqual(len(samfiles), 49)
        for sam in samfiles:
            sample_id = sam.name[: -len(".sam.xz")]
            with self.subTest(sample=sample_id):
                produced = self.micov_stdout_to(
                    self.tmp / f"{sample_id}.cov", "compress", "--data", sam
                )
                assert_cov_equal(
                    produced, EXAMPLE / "coverages" / f"{sample_id}.cov.gz"
                )


@requires_micov
@requires_example
@requires_full_tier
class TestParquetFullCorpus(MicovCliTestCase):
    def test_qiita_to_parquet_matches_committed(self):
        """`qiita-to-parquet` is the provenance of `example/parquet/`."""
        prefix = self.tmp / "example"
        self.micov(
            "qiita-to-parquet",
            "--qiita-coverages",
            EXAMPLE / "consolidate" / "consolidated.tgz",
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

    def test_nonqiita_to_parquet_agrees_except_on_column_order(self):
        """Pins a real incompatibility between the two producers.

        README steps 3 and 4 write the same prefix and read as
        interchangeable, but `nonqiita-to-parquet` puts `sample_id` first
        while `qiita-to-parquet` puts it last. The rows are identical. If the
        migration ever aligns them, this test fails and the README and the
        compatibility contract both need updating.
        """
        import duckdb

        prefix = self.tmp / "nonqiita"
        self.micov(
            "nonqiita-to-parquet",
            "--pattern",
            f"{EXAMPLE}/coverages/*.cov.gz",
            "--output",
            prefix,
            "--lengths",
            EXAMPLE / "metadata" / "length.tsv",
        )
        committed = EXAMPLE / "parquet" / "example.coverage.parquet"
        produced = Path(f"{prefix}.coverage.parquet")

        con = duckdb.connect()
        try:

            def columns(path):
                return [
                    r[0]
                    for r in con.execute(
                        f"DESCRIBE SELECT * FROM read_parquet('{path}')"
                    ).fetchall()
                ]

            produced_cols, committed_cols = columns(produced), columns(committed)
            self.assertNotEqual(produced_cols, committed_cols)
            self.assertEqual(set(produced_cols), set(committed_cols))
            self.assertEqual(produced_cols[0], "sample_id")
            self.assertEqual(committed_cols[-1], "sample_id")

            ordered = ", ".join(committed_cols)
            for left, right in ((produced, committed), (committed, produced)):
                extra = con.execute(
                    f"SELECT count(*) FROM ("
                    f"SELECT {ordered} FROM read_parquet('{left}') EXCEPT ALL "
                    f"SELECT {ordered} FROM read_parquet('{right}'))"
                ).fetchone()[0]
                self.assertEqual(extra, 0)
        finally:
            con.close()

        assert_parquet_equal(
            Path(f"{prefix}.covered_positions.parquet"),
            EXAMPLE / "parquet" / "example.covered_positions.parquet",
        )


@requires_micov
@requires_example
@requires_full_tier
class TestConsolidateFullCorpus(MicovCliTestCase):
    def test_matches_committed_archive(self):
        covs = sorted((EXAMPLE / "coverages").glob("*.cov.gz"))
        self.assertEqual(len(covs), 49)
        paths = self.write_paths_file("paths.txt", covs)
        produced = self.tmp / "consolidated.tgz"
        self.micov(
            "consolidate",
            "--paths",
            paths,
            "--output",
            produced,
            "--lengths",
            EXAMPLE / "metadata" / "length.tsv",
        )
        assert_tgz_equal(produced, EXAMPLE / "consolidate" / "consolidated.tgz")


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
    `.tsv.gz` position data and the `.ks.tsv` statistics.
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
                if produced.name.endswith(".ks.tsv"):
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
            if not (plain.name.endswith(".ks.tsv") or plain.name.endswith(".tsv.gz")):
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
        self.assertEqual(pairs, 4, "expected 2 .ks.tsv and 2 .tsv.gz comparisons")

    def test_monte_leaves_non_monte_ks_rows_unchanged(self):
        """Adding Monte Carlo curves must not perturb the published rows."""
        plain_dir, _ = self.run_variant("plain")
        monte_dir, _ = self.run_variant("monte")

        def deterministic_rows(path):
            rows = path.read_text().splitlines()
            return [r for r in rows[1:] if "Monte Carlo " not in r]

        compared = 0
        for plain in sorted(plain_dir.glob("*.cumulative.ks.tsv")):
            monte = monte_dir / plain.name.replace(
                ".cumulative.ks.tsv", ".cumulative-monte-unfocused.ks.tsv"
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
