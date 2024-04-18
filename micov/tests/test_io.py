import unittest
import io
from micov._io import (parse_genome_lengths, parse_qiita_coverages,
                       _single_df, _check_and_compress)
from micov._constants import (BED_COV_SCHEMA, COLUMN_GENOME_ID, COLUMN_START,
                              COLUMN_LENGTH)
import tempfile
import tarfile
import time
import polars as pl
import polars.testing as plt


def _add_file(tf, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = int(time.time())
    tf.addfile(ti, io.BytesIO(data.encode('ascii')))


def _create_qiita_cov(name):
    tf = tarfile.open(name, "w:gz")

    covdataname = 'coverage_percentage.txt'
    covdata = 'foobar'

    sample_a_name = 'coverages/sample_a.cov'
    sample_a_data = ("G123\t1\t10\n"
                     "G123\t100\t200\n"
                     "G456\t5\t20\n"
                     "G789\t2\t40\n")

    sample_b_name = 'coverages/sample_b.cov'
    sample_b_data = ("G123\t8\t15\n"
                     "G123\t300\t400\n"
                     "G789\t1\t100\n")

    sample_c_name = 'coverages/sample_c.cov'
    sample_c_data = "G123\t1000\t10000\n"

    _add_file(tf, covdataname, covdata)
    _add_file(tf, sample_a_name, sample_a_data)
    _add_file(tf, sample_b_name, sample_b_data)
    _add_file(tf, sample_c_name, sample_c_data)

    tf.close()


class QiitaCovTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.name = self.temp_dir.name + '/coverages.tgz'
        _create_qiita_cov(self.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_qiita_coverages(self):
        exp = pl.DataFrame([['G123', 1, 15],
                            ['G123', 100, 200],
                            ['G123', 300, 400],
                            ['G123', 1000, 10000],
                            ['G456', 5, 20],
                            ['G789', 1, 100]],
                           schema=BED_COV_SCHEMA.dtypes_flat)
        # always compress
        obs = parse_qiita_coverages(self.name)
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_always_compress(self):
        exp = pl.DataFrame([['G123', 1, 15],
                            ['G123', 100, 200],
                            ['G123', 300, 400],
                            ['G123', 1000, 10000],
                            ['G456', 5, 20],
                            ['G789', 1, 100]],
                           schema=BED_COV_SCHEMA.dtypes_flat)
        # always compress
        obs = parse_qiita_coverages(self.name, compress_size=0)
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_never_compress(self):
        exp = pl.DataFrame([['G123', 1, 10],
                            ['G123', 8, 15],
                            ['G123', 100, 200],
                            ['G123', 300, 400],
                            ['G123', 1000, 10000],
                            ['G456', 5, 20],
                            ['G789', 1, 100],
                            ['G789', 2, 40]],
                           schema=BED_COV_SCHEMA.dtypes_flat)
        obs = parse_qiita_coverages(self.name, compress_size=None)
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_keep(self):
        exp = pl.DataFrame([['G123', 1, 15],
                            ['G123', 100, 200],
                            ['G123', 300, 400],
                            ['G456', 5, 20],
                            ['G789', 1, 100]],
                           schema=BED_COV_SCHEMA.dtypes_flat)
        obs = parse_qiita_coverages(self.name,
                                    sample_keep={'sample_a', 'sample_b'})
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_drop(self):
        exp = pl.DataFrame([['G123', 1, 15],
                            ['G123', 100, 200],
                            ['G123', 300, 400],
                            ['G456', 5, 20],
                            ['G789', 1, 100]],
                           schema=BED_COV_SCHEMA.dtypes_flat)
        obs = parse_qiita_coverages(self.name,
                                    sample_drop={'sample_c', })
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_keep_drop(self):
        exp = pl.DataFrame([['G123', 1, 10],
                            ['G123', 100, 200],
                            ['G456', 5, 20],
                            ['G789', 2, 40]],
                           schema=BED_COV_SCHEMA.dtypes_flat)
        obs = parse_qiita_coverages(self.name,
                                    sample_drop={'sample_c', },
                                    sample_keep={'sample_a', })
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_keep_feature(self):
        exp = pl.DataFrame([['G123', 1, 15],
                            ['G123', 100, 200],
                            ['G123', 300, 400],
                            ['G123', 1000, 10000]],
                           schema=BED_COV_SCHEMA.dtypes_flat)
        obs = parse_qiita_coverages(self.name,
                                    feature_keep={'G123', })
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)

    def test_parse_qiita_coverages_drop_feature(self):
        exp = pl.DataFrame([['G456', 5, 20],
                            ['G789', 2, 40]],
                           schema=BED_COV_SCHEMA.dtypes_flat)
        obs = parse_qiita_coverages(self.name,
                                    sample_drop={'sample_c', },
                                    sample_keep={'sample_a', },
                                    feature_drop={'G123', })
        obs = obs.sort([COLUMN_GENOME_ID, COLUMN_START])
        plt.assert_frame_equal(obs, exp)


class IOTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.name = self.temp_dir.name + '/foo.tsv'

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_genome_lengths_good(self):
        data = ("foo\tbar\tbaz\n"
                "a\t10\txyz\n"
                "b\t20\txyz\n"
                "c\t30\txyz\n")

        with open(self.name, 'w') as fp:
            fp.write(data)

        exp = pl.DataFrame([['a', 10],
                            ['b', 20],
                            ['c', 30]],
                           schema=[COLUMN_GENOME_ID, COLUMN_LENGTH])
        obs = parse_genome_lengths(self.name)
        plt.assert_frame_equal(obs, exp)

    def test_parse_genome_lengths_noheader(self):
        data = ("a\t10\txyz\n"
                "b\t20\txyz\n"
                "c\t30\txyz\n")

        with open(self.name, 'w') as fp:
            fp.write(data)

        exp = pl.DataFrame([['a', 10],
                            ['b', 20],
                            ['c', 30]],
                           schema=[COLUMN_GENOME_ID, COLUMN_LENGTH])
        obs = parse_genome_lengths(self.name)
        plt.assert_frame_equal(obs, exp)

    def test_parse_genome_lengths_not_numeric(self):
        data = ("foo\tbar\tbaz\n"
                "a\t10\txyz\n"
                "b\tXXX\txyz\n"
                "c\t30\txyz\n")

        with open(self.name, 'w') as fp:
            fp.write(data)

        with self.assertRaisesRegex(ValueError, "'bar' is not integer"):
            parse_genome_lengths(self.name)

    def test_parse_genome_lengths_not_unique(self):
        data = ("foo\tbar\tbaz\n"
                "a\t10\txyz\n"
                "b\t20\txyz\n"
                "b\t30\txyz\n")

        with open(self.name, 'w') as fp:
            fp.write(data)

        with self.assertRaisesRegex(ValueError, "'foo' is not unique"):
            parse_genome_lengths(self.name)

    def test_parse_genome_lengths_bad_sizes(self):
        data = ("foo\tbar\tbaz\n"
                "a\t10\txyz\n"
                "b\t-5\txyz\n"
                "c\t30\txyz\n")

        with open(self.name, 'w') as fp:
            fp.write(data)

        with self.assertRaisesRegex(ValueError, "Lengths of zero or less"):
            parse_genome_lengths(self.name)


if __name__ == '__main__':
    unittest.main()
