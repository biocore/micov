import unittest

from micov._convert import cigar_to_lens


class ConvertTests(unittest.TestCase):
    def test_cigar_to_lens(self):
        self.assertEqual(cigar_to_lens('150M'), 150)
        self.assertEqual(cigar_to_lens('3M1I3M1D5M'), 12)

    def test_cigar_to_lens_sequence_match_and_mismatch(self):
        # '=' and 'X' advance the alignment exactly like 'M'. Nothing else in
        # the suite exercised them, and no SAM fixture contains them, so this
        # pins the behavior before it is reimplemented on htslib's bam_endpos.
        self.assertEqual(cigar_to_lens('150='), 150)
        self.assertEqual(cigar_to_lens('150X'), 150)
        self.assertEqual(cigar_to_lens('10=1X10='), 21)
        self.assertEqual(cigar_to_lens('3=1I3X1D5='), 12)

    def test_cigar_to_lens_reference_consuming_ops(self):
        # 'D' and 'N' advance the reference, so both count toward the covered
        # span -- deletions and gaps are deliberately treated as covered.
        self.assertEqual(cigar_to_lens('5M10D5M'), 20)
        self.assertEqual(cigar_to_lens('5M10N5M'), 20)

    def test_cigar_to_lens_query_only_ops_do_not_advance(self):
        # 'I', 'S', 'H' and 'P' consume query or nothing, never reference.
        self.assertEqual(cigar_to_lens('10S150M10S'), 150)
        self.assertEqual(cigar_to_lens('10H150M10H'), 150)
        self.assertEqual(cigar_to_lens('150M10I'), 150)
        self.assertEqual(cigar_to_lens('10P150M'), 150)


if __name__ == '__main__':
    unittest.main()
