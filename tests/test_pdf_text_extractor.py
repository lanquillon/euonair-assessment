import unittest

import pdf_text_extractor as ext


class TestPdfTextExtractor(unittest.TestCase):
    # Page selection parsing
    def test_parse_page_spec(self):
        pages = ext.parse_page_spec("1-3,5")
        self.assertEqual(pages, {1, 2, 3, 5})

    # Bullet detection (symbol and numbered)
    def test_detect_bullet_type_symbol_and_number(self):
        block_sym = {"text": "- Item", "x0": 12}
        is_bullet, indent = ext.detect_bullet_type(block_sym, page_min_x0=0)
        self.assertTrue(is_bullet)
        self.assertEqual(block_sym["text"], "Item")
        block_num = {"text": "1) First", "x0": 0}
        is_bullet, indent = ext.detect_bullet_type(block_num, page_min_x0=0)
        self.assertTrue(is_bullet)
        self.assertEqual(block_num["text"], "First")

    # Heuristic table detection on aligned blocks
    def test_detect_tables_simple(self):
        blocks = [
            {"text": "A", "x0": 0, "y0": 0, "bbox": [0, 0, 10, 10]},
            {"text": "B", "x0": 50, "y0": 0, "bbox": [50, 0, 60, 10]},
            {"text": "C", "x0": 0, "y0": 20, "bbox": [0, 20, 10, 30]},
            {"text": "D", "x0": 50, "y0": 20, "bbox": [50, 20, 60, 30]},
        ]
        page = {"page": 1, "blocks": blocks}
        detected = ext.detect_tables([page])
        tables = detected[0].get("tables", [])
        self.assertTrue(tables)
        self.assertEqual(tables[0]["rows"], [["A", "B"], ["C", "D"]])

    # Bullet hierarchy building
    def test_build_hierarchical_blocks(self):
        blocks = [
            {"type": "bullet", "text": "Top", "indent_level": 0},
            {"type": "bullet", "text": "Child", "indent_level": 1},
        ]
        hier = ext.build_hierarchical_blocks(blocks)
        self.assertEqual(len(hier), 1)
        self.assertEqual(hier[0]["text"], "Top")
        self.assertEqual(hier[0]["children"][0]["text"], "Child")


if __name__ == "__main__":
    unittest.main()
