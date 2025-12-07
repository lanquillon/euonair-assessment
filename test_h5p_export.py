"""Tests for helper functions in h5p_export.py (slugify, build_multichoice_content)."""

import unittest

from h5p_export import slugify, build_multichoice_content


def test_slugify_basic():
    """Slugify a normal string: lowercase, no spaces/specials, hyphen separation."""
    text = "What is Machine Learning?"
    slug = slugify(text)

    # Basic expectations
    assert slug == slug.lower()
    assert " " not in slug
    assert "?" not in slug
    assert slug.startswith("what")
    # Example: "what-is-machine-learning" or similar
    assert "-" in slug


def test_build_multichoice_content_correct_mapping():
    """One question yields 4 answers with exactly one correct flag at the right index."""
    question_data = {
        "question": "What is 2 + 2?",
        "options": ["3", "4", "5", "22"],
        "correct_answer_index": 1,  # index 1 = "4"
        "explanation": "Because 2 + 2 = 4."
    }

    content = build_multichoice_content(question_data)

    # Check basic keys
    assert "question" in content
    assert "answers" in content

    answers = content["answers"]

    # Expect 4 answers
    assert len(answers) == 4

    correct_flags = [a["correct"] for a in answers]

    # Count how many are True
    num_correct = sum(1 for flag in correct_flags if flag)

    assert num_correct == 1
    assert correct_flags[1] is True  # index 1 should be correct


def test_build_multichoice_content_includes_explanation():
    """Explanation from question_data appears in the final H5P question text."""
    question_data = {
        "question": "Test question?",
        "options": ["A", "B", "C", "D"],
        "correct_answer_index": 0,
        "explanation": "This is a test explanation."
    }

    content = build_multichoice_content(question_data)
    question_text = content["question"]

    # We expect the explanation text to be included somewhere
    assert "This is a test explanation." in question_text


# -----------------------------------------------------------------------------
# unittest variants (pytest will collect these too)
# -----------------------------------------------------------------------------


class TestH5PExportUnit(unittest.TestCase):
    def test_slugify_basic(self):
        slug = slugify("What is Machine Learning?")
        self.assertEqual(slug, slug.lower())
        self.assertNotIn(" ", slug)
        self.assertNotIn("?", slug)
        self.assertTrue(slug.startswith("what"))
        self.assertIn("-", slug)

    def test_build_multichoice_content_correct_mapping(self):
        question_data = {
            "question": "What is 2 + 2?",
            "options": ["3", "4", "5", "22"],
            "correct_answer_index": 1,
            "explanation": "Because 2 + 2 = 4.",
        }
        answers = build_multichoice_content(question_data)["answers"]
        flags = [a["correct"] for a in answers]
        self.assertEqual(sum(flags), 1)
        self.assertTrue(flags[1])

    def test_build_multichoice_content_includes_explanation(self):
        question_data = {
            "question": "Test question?",
            "options": ["A", "B", "C", "D"],
            "correct_answer_index": 0,
            "explanation": "This is a test explanation.",
        }
        question_text = build_multichoice_content(question_data)["question"]
        self.assertIn("This is a test explanation.", question_text)
