# H5P Export: create one H5P.MultiChoice .h5p file per question.
# Assumes questions.json from question_generation.py and that the target system
# already has H5P.MultiChoice installed (libraries are not bundled).

import json
import re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

# CONFIG
# ==============================================================================

# Input file from the question generation step
QUESTIONS_JSON = "questions.json"

# Output directory for generated .h5p files
OUTPUT_DIR = Path("h5p_output")

# H5P MultiChoice version (major/minor)
# This should match a version installed on your target system.
H5P_MC_MAJOR = 1
H5P_MC_MINOR = 16  # common version as of recent releases

# Default language code for your content
H5P_LANGUAGE = "en"


# Helper: sanitize filename
# ==============================================================================

def slugify(text: str, max_length: int = 40) -> str:
    """
    Make a simple, safe filename from the question text.

    Steps:
      - Lowercase
      - Replace non-letter/digit with '-'
      - Collapse multiple '-' into one
      - Strip leading/trailing '-'
      - Cut to max_length characters

    Example:
      "What is Machine Learning?" -> "what-is-machine-learning"
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        text = "question"
    return text[:max_length]


# Step 1: Load questions.json
# ==============================================================================

def load_questions(path: str) -> list[dict]:
    """
    Read questions.json and return the list of question dicts.

    Expected structure (from question_generation.py):
      top-level dict with "questions": [ {question, options, correct_answer_index, ...}, ... ]
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{path} not found. Run question_generation.py first.")

    data = json.loads(file_path.read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("questions.json does not contain a 'questions' list.")
    return questions


# ==============================================================================
# Step 2: Build content.json for H5P.MultiChoice
# ==============================================================================

def build_multichoice_content(question_data: dict) -> dict:
    """
    Convert one question dict from questions.json into an H5P.MultiChoice
    content.json structure.

    Keep it minimal:
      - question text -> "question"
      - options -> "answers" (with "text" + "correct")
      - basic behaviour & UI defaults.
    """

    question_text = question_data.get("question", "").strip()
    options = question_data.get("options", [])
    correct_index = question_data.get("correct_answer_index", 0)
    if not isinstance(correct_index, int) or correct_index < 0 or correct_index >= len(options):
        correct_index = 0
    explanation = question_data.get("explanation", "").strip()

    if not question_text or not options:
        raise ValueError("Question or options missing in question_data")

    # Build the "answers" list expected by H5P.MultiChoice
    answers = []
    for idx, opt_text in enumerate(options):
        opt_text = str(opt_text).strip()
        if not opt_text:
            continue

        answers.append({
            "text": opt_text,
            "correct": (idx == correct_index),
            # "tipsAndFeedback" could be used to attach more feedback per option.
            # For now we keep it minimal (optional field).
        })

    if not answers:
        raise ValueError("No non-empty answers for question")

    # Basic behaviour config (simplified)
    behaviour = {
        # Allow multiple attempts
        "enableRetry": True,
        # Show "Show solution" button
        "enableSolutionsButton": True,
        # Show "Check" button instead of auto-checking
        "autoCheck": False,
        # Randomize the order of the answer options
        "randomAnswers": True,
        # Single or multiple points: False = sum of correct options
        "singlePoint": False,
        # Require the learner to answer before showing solution
        "showSolutionsRequiresInput": True,
        # Pass percentage (100 means all correct)
        "passPercentage": 100
    }

    # Basic UI labels
    ui = {
        "checkAnswer": "Check",
        "showSolutionButton": "Show solution",
        "tryAgain": "Retry",
        "correctText": "Correct!",
        "incorrectText": "Incorrect.",
        "showSolution": "Show solution",
        "yourResult": "Your result:",
    }

    # Overall feedback per score range
    overall_feedback = [
        {
            "from": 0,
            "to": 100,
            "feedback": "You scored @score out of @maxScore."
        }
    ]

    # Append explanation to the question text
    if explanation:
        question_with_expl = (
            question_text
            + "\n\n"
            + "<em>Explanation:</em> " + explanation
        )
    else:
        question_with_expl = question_text

    content = {
        # Main question text (HTML allowed, but we keep it simple)
        "question": question_with_expl,

        # The answer options
        "answers": answers,

        # Basic behaviour + UI
        "behaviour": behaviour,
        "UI": ui,

        # Simple feedback
        "overallFeedback": overall_feedback,

        # Optional fields we leave out:
        # - "media" (image/video above the question)
        # - "l10n" for language overrides
        # - advanced behaviour options ...
    }

    return content


# ==============================================================================
# Step 3: Build h5p.json metadata
# ==============================================================================

def build_h5p_metadata(title: str) -> dict:
    """
    Build the minimal h5p.json metadata required by the H5P specification.

    Mandatory fields (according to h5p.org):
      - title
      - mainLibrary
      - language
      - embedTypes
      - preloadedDependencies
    """
    return {
        "title": title,
        "language": H5P_LANGUAGE,
        "mainLibrary": "H5P.MultiChoice",
        "embedTypes": ["div"],
        "preloadedDependencies": [
            {
                "machineName": "H5P.MultiChoice",
                "majorVersion": H5P_MC_MAJOR,
                "minorVersion": H5P_MC_MINOR,
            }
        ]
        # Optional fields like "license", "authors", ... could be added here.
    }


# ==============================================================================
# Step 4: Write .h5p ZIP file
# ==============================================================================

def write_h5p_package(output_path: Path, h5p_meta: dict, content_data: dict) -> None:
    """
    Create a .h5p file (ZIP):
      - h5p.json at root
      - content/content.json inside "content"
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dicts to JSON strings
    h5p_json_str = json.dumps(h5p_meta, ensure_ascii=False, indent=2)
    content_json_str = json.dumps(content_data, ensure_ascii=False, indent=2)

    # Create ZIP (.h5p) and write the two files
    with ZipFile(output_path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("h5p.json", h5p_json_str)
        zf.writestr("content/content.json", content_json_str)


# ==============================================================================
# Step 5: High-level export function
# ==============================================================================

def export_all_questions_to_h5p(
    questions_file: str = QUESTIONS_JSON,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Main pipeline:
      1) Load all questions.
      2) For each question build content.json + h5p.json and write one .h5p.
    """
    questions = load_questions(questions_file)
    if not questions:
        print("No questions found in questions.json")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, q in enumerate(questions, start=1):
        # Build title and file name from question text
        question_text = q.get("question", "").strip()
        short_slug = slugify(question_text)
        title = f"MCQ {idx}: {question_text[:60]}"

        h5p_meta = build_h5p_metadata(title)
        content_data = build_multichoice_content(q)

        # e.g. mcq_001_what-is-machine-learning.h5p
        filename = f"mcq_{idx:03d}_{short_slug}.h5p"
        output_path = output_dir / filename

        write_h5p_package(output_path, h5p_meta, content_data)
        print(f"Created {output_path}")

    print(f"\nDone. {len(questions)} H5P files written to: {output_dir}")


# ==============================================================================
# Script entry point
# ==============================================================================

if __name__ == "__main__":
    export_all_questions_to_h5p()
