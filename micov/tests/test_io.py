import io
import sys
import tarfile
import tempfile
import time
import unittest

import polars as pl
import polars.testing as plt

from micov._constants import (
    BED_COV_SCHEMA,
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_NAME,
    COLUMN_START,
    COLUMN_TAXONOMY,
    GENOME_COVERAGE_SCHEMA,
    GENOME_LENGTH_SCHEMA,
    SAM_SUBSET_SCHEMA_PARSED,
)
from micov._io import (
    compress_from_stream,
    parse_feature_names,
    parse_genome_lengths,
    parse_qiita_coverages,
    parse_sam_to_df,
    parse_taxonomy,
    set_taxonomy_as_id,
    write_qiita_cov,
)


def _add_file(tf, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = int(time.time())
    tf.addfile(ti, io.BytesIO(data.encode("ascii")))


def _create_qiita_cov(name):
    tf = tarfile.open(name, "w:gz")

    covdataname = "coverage_percentage.txt"
    covdata = "foobar"

    sample_a_name = "coverages/sample_a.cov"
    sample_a_data = "G123\t1\t10\n" "G123\t100\t200\n" "G456\t5\t20\n" "G789\t2\t40\n"

    sample_b_name = "coverages/sample_b.cov"
    sample_b_data = "G123\t8\t15\n" "G123\t300\t400\n" "G789\t1\t100\n"

    sample_c_name = "coverages/sample_c.cov"
    sample_c_data = "G123\t1000\t10000\n"

    _add_file(tf, covdataname, covdata)
    _add_file(tf, sample_a_name, sample_a_data)
    _add_file(tf, sample_b_name, sample_b_data)
    _add_file(tf, sample_c_name, sample_c_data)

    tf.close()


class QiitaCovTests(unittest.TestCase):
    def setUp(self):
        kwargs = {}
        if sys.version_info.minor >= 10:
            kwargs.update({"ignore_cleanup_errors": True})

        self.temp_dir = tempfile.TemporaryDirectory(**kwargs)
        self.name = self.temp_dir.name + "/coverages.tgz"
        _create_qiita_cov(self.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_qiita_cov(self):
        covs = [
            (
                "coverage_1.cov",
                (
                    "genome_id\tstart\tstop\n"
                    "GXXX\t200\t300\n"
                    "GXXX\t500\t600\n"
                    "GXXX\t100\t200\n"
                    "GYYY\t100\t200\n"
                ),
            ),
            (
                "coverage_2.cov",
                (
                    "genome_id\tstart\tstop\n"
                    "GYYY\t500\t1000\n"
                    "GYYY\t200\t400\n"
                    "GXXX\t300\t400\n"
                ),
            ),
            (
                "coverage_3.cov",
                (
                    "genome_id\tstart\tstop\n"
                    "GYYY\t500\t1000\n"
                    "GXXX\t100\t400\n"
                    "GZZZ\t200\t400\n"
                ),
            ),
        ]
        paths = []
        for fname, data in covs:
            path = self.temp_dir.name + f"/{fname}"
            with open(path, "w") as fp:
                fp.write(data)
            paths.append(path)

        lengths = pl.DataFrame(
            [["GXXX", 600], ["GYYY", 1100], ["GZZZ", 2000]],
            orient="row",
            schema=GENOME_LENGTH_SCHEMA.dtypes_flat,
        )

        write_qiita_cov(self.name, paths, lengths)

        tgz = tarfile.open(self.name)
        obs_artifact_cov = pl.read_csv(
            tgz.extractfile("artifact.cov").read(), separator="\t"
        )
        obs_cov_percent = pl.read_csv(
            tgz.extractfile("coverage_percentage.txt").read(), separator="\t"
        )

        exp_artifact_cov = pl.DataFrame(
            [
                ["GXXX", 100, 400],
                ["GXXX", 500, 600],
                ["GYYY", 100, 400],
                ["GYYY", 500, 1000],
                ["GZZZ", 200, 400],
            ],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )

        exp_cov_percent = pl.DataFrame(
            [
                ["GXXX", 400, 600, (400 / 600) * 100],
                ["GYYY", 800, 1100, (800 / 1100) * 100],
                ["GZZZ", 200, 2000, (200 / 2000) * 100],
            ],
            orient="row",
            schema=GENOME_COVERAGE_SCHEMA.dtypes_flat,
        )

        obs_artifact_cov = obs_artifact_cov.sort([COLUMN_GENOME_ID, COLUMN_START])
        obs_cov_percent = obs_cov_percent.sort(
            [
                COLUMN_GENOME_ID,
            ]
        )

        # we disable type assertion as these are read from CSV, and the type
        # is inferred
        plt.assert_frame_equal(obs_artifact_cov, exp_artifact_cov, check_dtypes=False)
        plt.assert_frame_equal(obs_cov_percent, exp_cov_percent, check_dtypes=False)

        for name, exp in covs:
            obs = tgz.extractfile(f"coverages/{name}")
            self.assertEqual(obs.read().decode("utf-8").replace("\r\n", "\n"), exp)

    def test_parse_qiita_coverages(self):
        exp = pl.DataFrame(
            [
                ["G123", 1, 15],
                ["G123", 100, 200],
                ["G123", 300, 400],
                ["G123", 1000, 10000],
                ["G456", 5, 20],
                ["G789", 1, 100],
            ],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        # always compress
        obs = parse_qiita_coverages(self.name)
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_always_compress(self):
        exp = pl.DataFrame(
            [
                ["G123", 1, 15],
                ["G123", 100, 200],
                ["G123", 300, 400],
                ["G123", 1000, 10000],
                ["G456", 5, 20],
                ["G789", 1, 100],
            ],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        # always compress
        obs = parse_qiita_coverages(self.name, compress_size=0)
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_never_compress(self):
        exp = pl.DataFrame(
            [
                ["G123", 1, 10],
                ["G123", 8, 15],
                ["G123", 100, 200],
                ["G123", 300, 400],
                ["G123", 1000, 10000],
                ["G456", 5, 20],
                ["G789", 1, 100],
                ["G789", 2, 40],
            ],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        obs = parse_qiita_coverages(self.name, compress_size=None)
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_keep(self):
        exp = pl.DataFrame(
            [
                ["G123", 1, 15],
                ["G123", 100, 200],
                ["G123", 300, 400],
                ["G456", 5, 20],
                ["G789", 1, 100],
            ],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        obs = parse_qiita_coverages(self.name, sample_keep={"sample_a", "sample_b"})
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_drop(self):
        exp = pl.DataFrame(
            [
                ["G123", 1, 15],
                ["G123", 100, 200],
                ["G123", 300, 400],
                ["G456", 5, 20],
                ["G789", 1, 100],
            ],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        obs = parse_qiita_coverages(
            self.name,
            sample_drop={
                "sample_c",
            },
        )
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_keep_drop(self):
        exp = pl.DataFrame(
            [["G123", 1, 10], ["G123", 100, 200], ["G456", 5, 20], ["G789", 2, 40]],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        obs = parse_qiita_coverages(
            self.name,
            sample_drop={
                "sample_c",
            },
            sample_keep={
                "sample_a",
            },
        )
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_keep_feature(self):
        exp = pl.DataFrame(
            [
                ["G123", 1, 15],
                ["G123", 100, 200],
                ["G123", 300, 400],
                ["G123", 1000, 10000],
            ],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        obs = parse_qiita_coverages(
            self.name,
            feature_keep={
                "G123",
            },
        )
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_drop_feature(self):
        exp = pl.DataFrame(
            [["G456", 5, 20], ["G789", 2, 40]],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        obs = parse_qiita_coverages(
            self.name,
            sample_drop={
                "sample_c",
            },
            sample_keep={
                "sample_a",
            },
            feature_drop={
                "G123",
            },
        )
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)


class IOTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.name = self.temp_dir.name + "/foo.tsv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_genome_lengths_good(self):
        data = "foo\tbar\tbaz\n" "a\t10\txyz\n" "b\t20\txyz\n" "c\t30\txyz\n"

        with open(self.name, "w") as fp:
            fp.write(data)

        exp = pl.DataFrame(
            [["a", 10], ["b", 20], ["c", 30]],
            orient="row",
            schema=[COLUMN_GENOME_ID, COLUMN_LENGTH],
        )
        obs = parse_genome_lengths(self.name)
        plt.assert_frame_equal(obs, exp)

    def test_parse_genome_lengths_noheader(self):
        data = "a\t10\txyz\n" "b\t20\txyz\n" "c\t30\txyz\n"

        with open(self.name, "w") as fp:
            fp.write(data)

        exp = pl.DataFrame(
            [["a", 10], ["b", 20], ["c", 30]],
            orient="row",
            schema=[COLUMN_GENOME_ID, COLUMN_LENGTH],
        )
        obs = parse_genome_lengths(self.name)
        plt.assert_frame_equal(obs, exp)

    def test_parse_genome_lengths_not_numeric(self):
        data = "foo\tbar\tbaz\n" "a\t10\txyz\n" "b\tXXX\txyz\n" "c\t30\txyz\n"

        with open(self.name, "w") as fp:
            fp.write(data)

        with self.assertRaisesRegex(ValueError, "'bar' is not integer"):
            parse_genome_lengths(self.name)

    def test_parse_genome_lengths_not_unique(self):
        data = "foo\tbar\tbaz\n" "a\t10\txyz\n" "b\t20\txyz\n" "b\t30\txyz\n"

        with open(self.name, "w") as fp:
            fp.write(data)

        with self.assertRaisesRegex(ValueError, "'foo' is not unique"):
            parse_genome_lengths(self.name)

    def test_parse_genome_lengths_bad_sizes(self):
        data = "foo\tbar\tbaz\n" "a\t10\txyz\n" "b\t-5\txyz\n" "c\t30\txyz\n"

        with open(self.name, "w") as fp:
            fp.write(data)

        with self.assertRaisesRegex(ValueError, "Lengths of zero or less"):
            parse_genome_lengths(self.name)

    def test_parse_feature_names(self):
        data = (
            "some_id\tsomecolumn\tanothercolumn\n"
            "abc\tthings and stuff\temtpy\n"
            "x\tfoo; bar; baz thing\tdsf\n"
        )
        with open(self.name, "w") as fp:
            fp.write(data)

        exp = pl.DataFrame(
            [["abc", "things_and_stuff"], ["x", "baz_thing"]],
            orient="row",
            schema=[COLUMN_GENOME_ID, COLUMN_NAME],
        )
        obs = parse_feature_names(self.name)
        plt.assert_frame_equal(obs, exp)

    def test_parse_taxonomy_good(self):
        data = (
            "genome_id\ttaxonomy\tbaz\n"
            "a\tspecies1\txyz\n"
            "b\tspecies2\txyz\n"
            "c\tspecies3\txyz\n"
        )

        with open(self.name, "w") as fp:
            fp.write(data)

        exp = pl.DataFrame(
            [["a", "species1"], ["b", "species2"], ["c", "species3"]],
            orient="row",
            schema=[COLUMN_GENOME_ID, COLUMN_TAXONOMY],
        )
        obs = parse_taxonomy(self.name)
        plt.assert_frame_equal(obs, exp)

    def test_parse_taxonomy_noheader(self):
        data = "a\tspecies1\txyz\n" "b\tspecies2\txyz\n" "c\tspecies3\txyz\n"

        with open(self.name, "w") as fp:
            fp.write(data)

        exp = pl.DataFrame(
            [["a", "species1"], ["b", "species2"], ["c", "species3"]],
            orient="row",
            schema=[COLUMN_GENOME_ID, COLUMN_TAXONOMY],
        )
        obs = parse_taxonomy(self.name)
        plt.assert_frame_equal(obs, exp)

    def test_set_taxonomy_as_id(self):
        cov = pl.DataFrame(
            {
                "genome_id": ["G000006925", "G000007525", "G000008865"],
                "covered": [2501356, 4378, 2582128],
                "length": [4828820, 2260266, 5594477],
                "percent_covered": [
                    51.800564112971706,
                    0.19369401654495533,
                    46.154948889771106,
                ],
            }
        )

        tax = pl.DataFrame(
            {
                "genome_id": [
                    "G000009925",
                    "G000010425",
                    "G000006925",
                    "G000007525",
                    "G000008865",
                ],
                "taxonomy": [
                    "species1",
                    "species2",
                    "species3",
                    "species4",
                    "species5",
                ],
            }
        )

        exp = pl.DataFrame(
            {
                "taxonomy": ["species3", "species4", "species5"],
                "genome_id": ["G000006925", "G000007525", "G000008865"],
                "covered": [2501356, 4378, 2582128],
                "length": [4828820, 2260266, 5594477],
                "percent_covered": [
                    51.800564112971706,
                    0.19369401654495533,
                    46.154948889771106,
                ],
            }
        )

        obs = set_taxonomy_as_id(cov, tax)
        plt.assert_frame_equal(obs, exp)

    def test_taxonomy_as_id_missing_taxonomy(self):
        cov = pl.DataFrame(
            {
                "genome_id": ["G000006925", "G000007525", "G000008865"],
                "covered": [2501356, 4378, 2582128],
                "length": [4828820, 2260266, 5594477],
                "percent_covered": [
                    51.800564112971706,
                    0.19369401654495533,
                    46.154948889771106,
                ],
            }
        )

        tax = pl.DataFrame(
            {
                "genome_id": ["G000009925", "G000010425", "G000006925", "G000007525"],
                "taxonomy": ["species1", "species2", "species3", "species4"],
            }
        )

        with self.assertRaisesRegex(ValueError, "[G000008865]"):
            set_taxonomy_as_id(cov, tax)

    def test_compress_from_stream(self):
        data = io.BytesIO(
            b"A\t0\tX\t1\t1\t50M\t*\t0\t0\t*\t*\n"
            b"B\t0\tY\t10\t1\t50M\t*\t0\t0\t*\t*\n"
            b"C\t0\tX\t100\t1\t50M\t*\t0\t0\t*\t*\n"
            b"D\t0\tX\t90\t1\t50M\t*\t0\t0\t*\t*\n"
            b"E\t0\tY\t100\t1\t50M\t*\t0\t0\t*\t*\n"
        )
        exp = pl.DataFrame(
            [["X", 1, 51], ["X", 90, 150], ["Y", 10, 60], ["Y", 100, 150]],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        obs = compress_from_stream(data, bufsize=2)
        plt.assert_frame_equal(
            obs.sort(
                [
                    COLUMN_GENOME_ID,
                ]
            ),
            exp,
        )

        obs = compress_from_stream(io.BytesIO())
        self.assertEqual(obs, None)

    def test_compress_from_stream_disable_compression(self):
        data = io.BytesIO(
            b"A\t0\tX\t1\t1\t50M\t*\t0\t0\t*\t*\n"
            b"B\t0\tY\t10\t1\t50M\t*\t0\t0\t*\t*\n"
            b"C\t0\tX\t100\t1\t50M\t*\t0\t0\t*\t*\n"
            b"D\t0\tX\t90\t1\t50M\t*\t0\t0\t*\t*\n"
            b"E\t0\tY\t100\t1\t50M\t*\t0\t0\t*\t*\n"
        )
        exp = pl.DataFrame(
            [
                ["X", 1, 51],
                ["X", 90, 140],
                ["X", 100, 150],
                ["Y", 10, 60],
                ["Y", 100, 150],
            ],
            orient="row",
            schema=BED_COV_SCHEMA.dtypes_flat,
        )
        obs = compress_from_stream(data, bufsize=2, disable_compression=True)
        plt.assert_frame_equal(obs.sort([COLUMN_GENOME_ID, COLUMN_START]), exp)

        obs = compress_from_stream(io.BytesIO())
        self.assertEqual(obs, None)

    def test_parse_sam_to_df(self):
        data = io.BytesIO(
            b"A\t0\tX\t1\t1\t50M\t*\t0\t0\t*\t*\n"
            b"B\t0\tY\t10\t1\t50M\t*\t0\t0\t*\t*\n"
            b"C\t0\tX\t100\t1\t50M\t*\t0\t0\t*\t*\n"
        )
        exp = pl.DataFrame(
            [
                ["A", 0, "X", 1, "50M", 51],
                ["B", 0, "Y", 10, "50M", 60],
                ["C", 0, "X", 100, "50M", 150],
            ],
            orient="row",
            schema=SAM_SUBSET_SCHEMA_PARSED.dtypes_flat,
        )
        obs = parse_sam_to_df(data)
        plt.assert_frame_equal(obs, exp)


if __name__ == "__main__":
    unittest.main()
