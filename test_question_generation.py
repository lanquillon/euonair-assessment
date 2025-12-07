"""Tests for helper functions in question_generation.py (pure functions, no network)."""

import json
import unittest

# Import the functions we want to test
from question_generation import extract_json_from_text, normalize_questions, format_questions


def test_extract_json_from_text_simple():
    """Clean JSON string parses to a dict with questions."""
    payload = {
        "questions": [
            {
                "question": "What is AI?",
                "options": ["A", "B", "C", "D"],
                "correct_answer_index": 0,
                "explanation": "Test explanation"
            }
        ]
    }

    text = json.dumps(payload)

    parsed = extract_json_from_text(text)

    assert parsed is not None
    assert "questions" in parsed
    assert isinstance(parsed["questions"], list)
    assert len(parsed["questions"]) == 1
    assert parsed["questions"][0]["question"] == "What is AI?"


def test_extract_json_from_text_with_noise_around():
    """JSON surrounded by noise still parses correctly."""
    payload = {
        "questions": [
            {
                "question": "What is Machine Learning?",
                "options": ["A", "B", "C", "D"],
                "correct_answer_index": 1,
                "explanation": "Test explanation"
            }
        ]
    }

    json_str = json.dumps(payload)

    noisy_text = (
        "Here is your result:\n\n"
        + json_str
        + "\n\nHope this helps!"
    )

    parsed = extract_json_from_text(noisy_text)

    assert parsed is not None
    assert "questions" in parsed
    assert len(parsed["questions"]) == 1
    assert parsed["questions"][0]["question"] == "What is Machine Learning?"


def test_extract_json_from_text_invalid_input():
    """Invalid JSON returns None without crashing."""
    invalid_text = "this is not json at all } { ???"

    parsed = extract_json_from_text(invalid_text)

    assert parsed is None


def test_extract_json_from_text_code_fence():
    """JSON wrapped in code fences still parses."""
    payload = {"questions": [{"question": "Q1", "options": ["A", "B", "C", "D"], "correct_answer_index": 2}]}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    parsed = extract_json_from_text(fenced)
    assert parsed and parsed["questions"][0]["correct_answer_index"] == 2


def test_normalize_questions_accepts_correct_index():
    """Legacy correct_index maps to correct_answer_index."""
    parsed = {"questions": [{"question": "Q1", "options": ["A", "B", "C", "D"], "correct_index": 3}]}
    normalized = normalize_questions(parsed)
    assert normalized
    assert normalized[0]["correct_answer_index"] == 3


def test_format_questions_letters_and_numbers():
    """Formatted questions prefix options with letters and correct letter."""
    questions = [
        {"question": "Q1", "options": ["a", "b", "c", "d"], "correct_answer_index": 1, "bloom_level": "remember", "explanation": "because"},
        {"question": "Q2", "options": ["w", "x", "y", "z"], "correct_answer_index": 0, "bloom_level": "understand", "explanation": "because"},
    ]
    formatted = format_questions(questions)
    assert len(formatted) == 2
    assert formatted[0]["question_number"] == 1
    assert formatted[0]["options"][0].startswith("A)")
    assert formatted[0]["correct_answer_letter"] == "B"
    assert formatted[1]["correct_answer_letter"] == "A"


# -----------------------------------------------------------------------------
# unittest variants (pytest will collect these too)
# -----------------------------------------------------------------------------


class TestQuestionGenerationUnit(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "questions": [
                {
                    "question": "What is AI?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer_index": 0,
                    "explanation": "Test explanation",
                }
            ]
        }

    def test_extract_json_simple(self):
        """Plain JSON loads correctly."""
        parsed = extract_json_from_text(json.dumps(self.payload))
        self.assertIsNotNone(parsed)
        self.assertIn("questions", parsed)
        self.assertEqual(parsed["questions"][0]["question"], "What is AI?")

    def test_extract_json_code_fence(self):
        """Code-fenced JSON loads correctly."""
        fenced = "```json\n" + json.dumps(self.payload) + "\n```"
        parsed = extract_json_from_text(fenced)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["questions"][0]["question"], "What is AI?")

    def test_normalize_correct_index(self):
        """Legacy correct_index is normalized."""
        parsed = {"questions": [{"question": "Q1", "options": ["A", "B", "C", "D"], "correct_index": 2}]}
        normalized = normalize_questions(parsed)
        self.assertTrue(normalized)
        self.assertEqual(normalized[0]["correct_answer_index"], 2)

    def test_format_questions_letters(self):
        """Options get letter prefixes and correct letter."""
        questions = [
            {"question": "Q1", "options": ["a", "b", "c", "d"], "correct_answer_index": 3, "bloom_level": "remember", "explanation": "because"}
        ]
        formatted = format_questions(questions)
        self.assertEqual(formatted[0]["options"][0][:2], "A)")
        self.assertEqual(formatted[0]["correct_answer_letter"], "D")

