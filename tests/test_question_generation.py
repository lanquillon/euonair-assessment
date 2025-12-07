import json
import unittest

from question_generation import extract_json_from_text, normalize_questions, format_questions


class TestQuestionGeneration(unittest.TestCase):
    # JSON parsing should accept clean JSON
    def test_extract_json_simple(self):
        payload = {
            "questions": [
                {
                    "question": "What is AI?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer_index": 0,
                    "explanation": "Test explanation",
                }
            ]
        }
        parsed = extract_json_from_text(json.dumps(payload))
        self.assertIsNotNone(parsed)
        self.assertIn("questions", parsed)
        self.assertEqual(parsed["questions"][0]["question"], "What is AI?")

    # JSON parsing should tolerate code fences
    def test_extract_json_code_fence(self):
        payload = {"questions": [{"question": "Q1", "options": ["A", "B", "C", "D"], "correct_answer_index": 2}]}
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        parsed = extract_json_from_text(fenced)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["questions"][0]["correct_answer_index"], 2)

    # Normalization should map correct_index to correct_answer_index
    def test_normalize_correct_index(self):
        parsed = {"questions": [{"question": "Q1", "options": ["A", "B", "C", "D"], "correct_index": 2}]}
        normalized = normalize_questions(parsed)
        self.assertTrue(normalized)
        self.assertEqual(normalized[0]["correct_answer_index"], 2)

    # Formatting should add letter prefixes and correct letter
    def test_format_questions_letters(self):
        questions = [
            {
                "question": "Q1",
                "options": ["a", "b", "c", "d"],
                "correct_answer_index": 3,
                "bloom_level": "remember",
                "explanation": "because",
            }
        ]
        formatted = format_questions(questions)
        self.assertEqual(formatted[0]["options"][0][:2], "A)")
        self.assertEqual(formatted[0]["correct_answer_letter"], "D")


if __name__ == "__main__":
    unittest.main()
