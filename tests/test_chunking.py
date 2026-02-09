import unittest

from lsi.chunking import chunk_text


class TestChunking(unittest.TestCase):
    def test_chunk_text_overlap(self) -> None:
        text = "a" * 120
        chunks = list(chunk_text(text, chunk_chars=50, overlap_chars=10))
        self.assertEqual(chunks[0].text, "a" * 50)
        self.assertEqual(chunks[1].start, 40)
        self.assertEqual(chunks[1].text, "a" * 50)
        self.assertEqual(chunks[-1].end, 120)
