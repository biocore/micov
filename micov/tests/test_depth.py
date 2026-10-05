"""Tests for `depth-plot`'s computation.

Depth and breadth may come from different files -- metatranscriptomic depth
over metagenomic breadth, say -- so before anything is computed the two layers,
the metadata and the features have to be reconciled. Anything left out is
reported by name: a sample quietly missing from one layer would change a
group's median with nothing to show why.

The `dp_*` fixtures are described in `test_data/README.md`.
"""

import shutil
import unittest
from pathlib import Path
from tempfile import mkdtemp

from micov._depth import GENOMES_TABLE, ROSTER_TABLE, intersect_layers
from micov._io import (
    load_alignment_layer,
    load_depth_features,
    load_orfs,
    load_sample_groups,
)
from micov._miint import connection

DATA = Path(__file__).parent / "test_data"


class IntersectLayersTests(unittest.TestCase):
    def setUp(self):
        self.d = mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.con = connection()
        self.addCleanup(self.con.close)

    def write(self, name, text):
        path = f"{self.d}/{name}"
        with open(path, "w") as fp:
            fp.write(text)
        return path

    def layer(self, name, select):
        """Write a layer Parquet derived from `dp_depth.parquet` by `select`."""
        path = f"{self.d}/{name}.parquet"
        self.con.sql(f"""COPY ({select.format(depth=f"'{DATA}/dp_depth.parquet'")})
                         TO '{path}' (FORMAT PARQUET)""")
        return path

    def resolve(self, depth=None, breadth=None, metadata=None, column="group",
                features=None, orfs=None):
        load_alignment_layer(
            self.con, str(depth or DATA / "dp_depth.parquet"), "depth_layer"
        )
        load_alignment_layer(
            self.con, str(breadth or DATA / "dp_breadth.parquet"), "breadth_layer"
        )
        load_sample_groups(
            self.con, str(metadata or DATA / "dp_metadata.tsv"), column
        )
        load_depth_features(self.con, str(features or DATA / "dp_regions.tsv"))
        if orfs is not None:
            load_orfs(self.con, str(orfs))
        intersect_layers(
            self.con, "depth_layer", "breadth_layer", orfs=orfs is not None
        )

    def roster(self):
        return self.con.sql(f"FROM {ROSTER_TABLE} ORDER BY sample_idx").fetchall()

    def genomes(self):
        return self.con.sql(f"FROM {GENOMES_TABLE} ORDER BY 1").fetchall()

    def test_samples_are_those_in_the_metadata_and_both_layers(self):
        """S6 is breadth only, S7 depth only, S8 neither; S9 has no metadata."""
        with self.assertLogs("micov", level="WARNING"):
            self.resolve()
        self.assertEqual(
            self.roster(),
            [("S1", 0, "case"), ("S2", 1, "case"), ("S3", 2, "case"),
             ("S4", 3, "control"), ("S5", 4, "control")],
        )

    def test_genomes_are_those_in_the_features_and_both_layers(self):
        with self.assertLogs("micov", level="WARNING"):
            self.resolve()
        self.assertEqual(self.genomes(), [("GC", 3000, True), ("GL", 2000, False)])

    def test_every_sample_and_genome_left_out_is_named(self):
        with self.assertLogs("micov", level="WARNING") as logged:
            self.resolve()
        message = "\n".join(logged.output)
        for name in ("S6", "S7", "S8", "GX", "GB"):
            with self.subTest(left_out=name):
                self.assertIn(name, message)

    def test_unaligned_reads_and_unlisted_samples_are_not_reported(self):
        """`*` is no genome, and S9 was left out by the metadata, not by micov."""
        with self.assertLogs("micov", level="WARNING") as logged:
            self.resolve()
        message = "\n".join(logged.output)
        self.assertNotIn("*", message)
        self.assertNotIn("S9", message)

    def test_a_placed_unmapped_read_does_not_put_a_genome_in_a_layer(self):
        """S4's unmapped read sits on GC, but aligned nowhere."""
        depth = self.layer(
            "depth", "SELECT * FROM {depth} WHERE read_id = 'S4:both:d4'"
        )
        with self.assertRaisesRegex(ValueError, "sample"):
            self.resolve(depth=depth)

    def test_the_same_file_for_both_layers(self):
        """`--breadth` defaults to `--depth`: then no layer can lack anything.

        S7 and GX, depth-only in the two-file run, are now in both; S6 is in
        neither, which is still reported.
        """
        path = DATA / "dp_depth.parquet"
        with self.assertLogs("micov", level="WARNING") as logged:
            self.resolve(depth=path, breadth=path)
        self.assertEqual(
            [row[0] for row in self.roster()], ["S1", "S2", "S3", "S4", "S5", "S7"]
        )
        self.assertEqual([row[0] for row in self.genomes()], ["GC", "GL", "GX"])
        message = "\n".join(logged.output)
        self.assertNotIn("layer only", message)
        self.assertIn("S6", message)

    def test_no_sample_in_common_is_an_error(self):
        breadth = self.layer(
            "breadth", "SELECT * REPLACE ('Z' || sample_id AS sample_id) FROM {depth}"
        )
        with self.assertRaisesRegex(ValueError, "sample"):
            self.resolve(breadth=breadth)

    def test_no_genome_in_common_is_an_error(self):
        features = self.write("f.tsv", "genome_id\tlength\nGX\t1000\nGB\t1000\n")
        with self.assertRaisesRegex(ValueError, "genome"):
            self.resolve(features=features)

    def test_more_than_ten_groups_is_an_error_naming_them(self):
        """Ten is as many as can be drawn distinctly (see `_plot.group_style`)."""
        depth = self.layer(
            "eleven",
            "SELECT * REPLACE ('T' || k AS sample_id) "
            "FROM {depth}, range(11) t(k) WHERE read_id = 'S1:both:a1'",
        )
        metadata = self.write(
            "m.tsv", "sample_id\tgroup\n"
            + "".join(f"T{k}\tg{k:02d}\n" for k in range(11))
        )
        with self.assertRaises(ValueError) as raised:
            self.resolve(depth=depth, breadth=depth, metadata=metadata)
        for k in range(11):
            self.assertIn(f"g{k:02d}", str(raised.exception))

    def test_an_alignment_beyond_the_length_is_an_error(self):
        """A read at 2,951 on a 2,000 bp genome means the length is wrong."""
        features = self.write("f.tsv", "genome_id\tlength\nGC\t2000\nGL\t2000\n")
        with self.assertRaisesRegex(ValueError, "GC.*2000"):
            self.resolve(features=features)

    def test_orfs_on_no_plotted_genome_are_an_error(self):
        """An empty ORF track would look like a genome with no genes."""
        gff = self.write(
            "x.gff", "##gff-version 3\nGQ\tt\tCDS\t1\t30\t.\t+\t0\tID=q1\n"
        )
        orfs = f"{self.d}/orfs.parquet"
        self.con.sql(f"COPY (FROM read_gff('{gff}')) TO '{orfs}' (FORMAT PARQUET)")
        with self.assertRaisesRegex(ValueError, "GQ"):
            self.resolve(orfs=orfs)

    def test_orfs_on_other_genomes_are_ignored(self):
        """dp_orfs also has ORFs on GX (left out) and GQ (not a feature)."""
        with self.assertLogs("micov", level="WARNING"):
            self.resolve(orfs=DATA / "dp_orfs.parquet")
        self.assertEqual(len(self.roster()), 5)


if __name__ == "__main__":
    unittest.main()
