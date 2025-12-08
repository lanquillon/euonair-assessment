"""Unit tests for H5P export helpers."""
# pylint: disable=import-error, wrong-import-position
# Keep tests runnable directly by ensuring the project root is importable.

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from h5p_export import (  # noqa: E402
    build_multichoice_content,
    slugify,
    validate_question,
)


class TestH5PExport(unittest.TestCase):
    """Tests for slug creation, content building, and validation."""

    def test_slugify_basic(self):
        """Slugify should lowercase, strip spaces/specials, and hyphenate."""
        slug = slugify("What is Machine Learning?")
        self.assertEqual(slug, slug.lower())
        self.assertNotIn(" ", slug)
        self.assertNotIn("?", slug)
        self.assertTrue(slug.startswith("what"))
        self.assertIn("-", slug)

    def test_build_multichoice_content(self):
        """Content builder should map correct answer and keep explanation."""
        question_data = {
            "question": "What is 2 + 2?",
            "options": ["3", "4", "5", "22"],
            "correct_answer_index": 1,
            "explanation": "Because 2 + 2 = 4.",
        }
        content = build_multichoice_content(question_data)
        answers = content["answers"]
        flags = [a["correct"] for a in answers]
        self.assertEqual(len(answers), 4)
        self.assertEqual(sum(flags), 1)
        self.assertTrue(flags[1])
        feedback = content["overallFeedback"][0]["feedback"]
        self.assertIn("Because 2 + 2 = 4.", feedback)

    def test_validate_question_out_of_range(self):
        """Validation should flag out-of-range correct_answer_index."""
        question = {
            "question": "Sample?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_index": 5,
        }
        errors = validate_question(question, question_num=1)
        self.assertTrue(any("out of range" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
