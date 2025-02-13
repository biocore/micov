import unittest

from micov._convert import cigar_to_lens


class ConvertTests(unittest.TestCase):
    def test_cigar_to_lens(self):
        self.assertEqual(cigar_to_lens('150M'), 150)
        self.assertEqual(cigar_to_lens('3M1I3M1D5M'), 12)


if __name__ == '__main__':
    unittest.main()
