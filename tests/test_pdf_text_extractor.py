"""Unit tests for pdf_text_extractor helpers."""
# pylint: disable=import-error, wrong-import-position
# Keep tests runnable directly by ensuring the project root is importable.

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import pdf_text_extractor as ext  # noqa: E402


def make_cfg(**overrides):
    """Create a minimal ExtractConfig for tests."""
    base = {
        "pdf_path": Path("dummy.pdf"),
        "detect_tables": True,
        "use_ocr": False,
        "force_ocr": False,
    }
    return ext.ExtractConfig(**{**base, **overrides})


class TestPdfTextExtractor(unittest.TestCase):
    """Tests for parsing, bullet detection, tables, and hierarchy."""

    def test_parse_page_spec(self):
        """Parse page ranges and lists."""
        pages = ext.parse_page_spec("1-3,5")
        self.assertEqual(pages, {1, 2, 3, 5})

    def test_detect_bullet_symbol_and_number(self):
        """Detect bullets from symbols and numbered patterns."""
        cfg = make_cfg()
        block_sym = {"text": "- Item", "x0": 12}
        is_bullet, indent, text = ext.detect_bullet(
            cfg, block_sym, page_min_x0=0
        )
        self.assertTrue(is_bullet)
        self.assertEqual(indent, 0)
        self.assertEqual(text, "Item")

        block_num = {"text": "1) First", "x0": 0}
        is_bullet, indent, text = ext.detect_bullet(
            cfg, block_num, page_min_x0=0
        )
        self.assertTrue(is_bullet)
        self.assertEqual(indent, 0)
        self.assertEqual(text, "First")

    def test_detect_tables_simple(self):
        """Detect a simple 2x2 table from aligned blocks."""
        cfg = make_cfg(detect_tables=True)
        blocks = [
            {"text": "A", "x0": 0, "y0": 0, "bbox": [0, 0, 10, 10]},
            {"text": "B", "x0": 50, "y0": 0, "bbox": [50, 0, 60, 10]},
            {"text": "C", "x0": 0, "y0": 20, "bbox": [0, 20, 10, 30]},
            {"text": "D", "x0": 50, "y0": 20, "bbox": [50, 20, 60, 30]},
        ]
        page = {"page": 1, "blocks": blocks}
        detected = ext.detect_tables(cfg, [page])
        tables = detected[0].get("tables", [])
        self.assertTrue(tables)
        self.assertEqual(tables[0]["rows"], [["A", "B"], ["C", "D"]])

    def test_build_bullet_hierarchy(self):
        """Nest bullet items by indentation."""
        blocks = [
            {"type": "bullet", "text": "Top", "indent_level": 0},
            {"type": "bullet", "text": "Child", "indent_level": 1},
        ]
        hier = ext.build_bullet_hierarchy(blocks)
        self.assertEqual(len(hier), 1)
        self.assertEqual(hier[0]["text"], "Top")
        self.assertEqual(hier[0]["children"][0]["text"], "Child")


if __name__ == "__main__":
    unittest.main()
