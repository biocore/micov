import tempfile
import unittest

import polars as pl
import polars.testing as plt

from micov._constants import (
    COLUMN_GENOME_ID,
    COLUMN_LENGTH,
    COLUMN_NAME,
)
from micov._io import (
    parse_feature_names,
    parse_genome_lengths,
)


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


if __name__ == "__main__":
    unittest.main()
