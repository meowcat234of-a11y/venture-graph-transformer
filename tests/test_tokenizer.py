import unittest

from model.tokenizer import ByteTokenizer


class TestByteTokenizer(unittest.TestCase):
    def test_utf8_round_trip(self):
        tokenizer = ByteTokenizer()
        text = "Venture Δ"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_rejects_insufficient_vocabulary(self):
        with self.assertRaises(ValueError):
            ByteTokenizer(vocab_size=255)


if __name__ == "__main__":
    unittest.main()
