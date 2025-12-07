import unittest

from h5p_export import build_multichoice_content, slugify


class TestH5PExport(unittest.TestCase):
    # Slugify should lowercase, strip spaces/specials, and hyphenate
    def test_slugify_basic(self):
        slug = slugify("What is Machine Learning?")
        self.assertEqual(slug, slug.lower())
        self.assertNotIn(" ", slug)
        self.assertNotIn("?", slug)
        self.assertTrue(slug.startswith("what"))
        self.assertIn("-", slug)

    # Content builder should map correct answer and keep explanation in feedback
    def test_build_multichoice_content(self):
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


if __name__ == "__main__":
    unittest.main()
