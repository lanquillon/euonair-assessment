"""
Question generation using local LLM (Ollama).

Generates exam-style multiple-choice questions from extracted PDF content,
aligned with Bloom's taxonomy and evidence-based MCQ guidelines.
"""

import json
import logging
import textwrap
from pathlib import Path

import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

EXTRACTED_JSON = Path("output") / "extracted_text.json"
OUTPUT_JSON = Path("output") / "questions.json"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"
OLLAMA_TIMEOUT = 60  # seconds

# Generation parameters
MAX_BLOCKS = 8  # Maximum text blocks to process
TARGET_QUESTIONS = 5  # Target number of questions to generate

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT ENGINEERING
# =============================================================================

def build_prompt(text_block: str) -> str:
    """Build structured prompt for MCQ generation with quality guidelines."""
    prompt_template = """
    You are an experienced university lecturer creating exam questions.

    TASK:
    Design {question_count} HIGH-QUALITY, EXAM-STYLE MULTIPLE-CHOICE QUESTIONS in ENGLISH
    for undergraduate/graduate students based ONLY on the source text below.

    REQUIREMENTS:

    1. LANGUAGE & CONTENT
       - Write EVERYTHING in ENGLISH
       - Do NOT ask about word meanings, translations, or quoted phrases
       - Focus on concepts, principles, and application

    2. BLOOM'S TAXONOMY
       - At least 2 questions at "Understand" level or higher
       - At least 1 question at "Apply" or "Analyze" level
       - Avoid pure memorization questions

    3. QUESTION QUALITY
       - Clear, self-contained stem that can be answered without seeing options
       - Paraphrase concepts; don't copy sentences verbatim
       - Focus on important ideas, not trivial details
       - Stems must reference specific concepts/techniques from the source (e.g., modularity, table detection); do NOT refer to "the text" or "the author"
       - BAN phrases like "according to the text/author"; if such wording appears, rewrite the stem to name the specific concept directly
       - Avoid vague summary stems like "What is the main challenge..."; rewrite to target a concrete concept or brief scenario
       - At least 2 stems should include a short scenario/application that tests applying a concept, not just recalling it
       - Cover different concepts across the text; do not repeat the same idea twice
       - Vary stem styles (definition/contrast, scenario/application, diagnosis/consequence) to keep questions diverse

    4. ANSWER OPTIONS
       - Provide EXACTLY 4 options: A, B, C, D
       - ONE clearly correct answer supported by the text
       - 3 plausible distractors based on common misconceptions
       - All options similar in length and style
       - NO "all of the above" or "none of the above"

    5. EXPLANATION
       - 2-4 sentences justifying the correct answer
       - Briefly explain why other options are incorrect

    OUTPUT FORMAT (JSON):
    Return a JSON array with {question_count} objects, each containing:
    - "question": question stem (string)
    - "options": array of 4 answer options [A, B, C, D]
    - "correct_answer_index": index 0-3 of correct option
    - "bloom_level": one of ["remember", "understand", "apply", "analyze", "evaluate", "create"]
    - "explanation": justification for correct answer

    EXAMPLE:
    [
      {{
        "question": "Which statement best describes the role of feedback in an interactive system?",
        "options": [
          "It stores user data permanently without any user interaction.",
          "It informs the user about the current system state and consequences of actions.",
          "It prevents all possible user errors automatically.",
          "It replaces the need for user training and documentation."
        ],
        "correct_answer_index": 1,
        "bloom_level": "understand",
        "explanation": "Good feedback keeps users informed about system state and action consequences. It doesn't prevent all errors or eliminate the need for training, but helps users build a correct mental model."
      }}
    ]

    SOURCE TEXT:
    <<<SOURCE_TEXT_START>>>
    {source_text}
    <<<SOURCE_TEXT_END>>>

    Generate {question_count} questions now. Return ONLY valid JSON, no additional text.
    """

    return textwrap.dedent(prompt_template).format(
        source_text=text_block,
        question_count=TARGET_QUESTIONS,
    )


# =============================================================================
# LLM INTERACTION
# =============================================================================

def call_ollama(prompt: str) -> str:
    """Send prompt to Ollama and return the response."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    except requests.exceptions.Timeout:
        logger.error(f"Ollama request timed out after {OLLAMA_TIMEOUT}s")
        raise
    except requests.exceptions.RequestException as exc:
        logger.error(f"Ollama request failed: {exc}")
        raise


# =============================================================================
# JSON PARSING
# =============================================================================

def extract_json_from_text(text: str) -> dict | None:
    """Extract and parse JSON from LLM output (handles code fences, noise)."""
    cleaned = text

    # Remove code fences if present
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
            # Remove language identifier (e.g., "json")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]

    # Try direct parse first
    try:
        parsed = json.loads(cleaned)
        return {"questions": parsed} if isinstance(parsed, list) else parsed
    except json.JSONDecodeError:
        pass

    # Try extracting array or object segments
    candidates = []

    # Try array segment
    a_start = cleaned.find("[")
    a_end = cleaned.rfind("]") + 1
    if a_start != -1 and a_end > a_start:
        candidates.append(cleaned[a_start:a_end])

    # Try object segment
    o_start = cleaned.find("{")
    o_end = cleaned.rfind("}") + 1
    if o_start != -1 and o_end > o_start:
        candidates.append(cleaned[o_start:o_end])

    # Parse candidates
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return {"questions": parsed} if isinstance(parsed, list) else parsed
        except json.JSONDecodeError:
            continue

    logger.warning("Could not parse JSON from LLM output")
    return None


def normalize_questions(parsed: dict | list) -> list[dict]:
    """
    Normalize question format and handle variations in field names.
    Accepts either a list of dicts or a dict with a 'questions' key.
    """
    questions = None

    if isinstance(parsed, list):
        questions = parsed
    elif isinstance(parsed, dict):
        questions = parsed.get("questions")

    if not questions or not isinstance(questions, list):
        return []

    normalized = []
    for q in questions:
        if not isinstance(q, dict):
            continue

        # Normalize field names (handle variations)
        if "correct_answer_index" not in q and "correct_index" in q:
            q["correct_answer_index"] = q["correct_index"]

        normalized.append(q)

    return normalized


def format_questions(questions: list[dict]) -> list[dict]:
    """Format questions with numbered prefixes and letter labels for options."""
    formatted = []
    letters = ["A", "B", "C", "D"]

    for idx, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            continue

        # Add letter prefixes to options
        options = q.get("options", [])
        prefixed = []
        for i, opt in enumerate(options):
            letter = letters[i] if i < len(letters) else chr(65 + i)
            prefixed.append(f"{letter}) {opt}")

        correct_idx = q.get("correct_answer_index")
        correct_letter = None
        if isinstance(correct_idx, int) and 0 <= correct_idx < len(letters):
            correct_letter = letters[correct_idx]

        formatted.append({
            "question_number": idx,
            "question": q.get("question", ""),
            "options": prefixed,
            "correct_answer_index": correct_idx,
            "correct_answer_letter": correct_letter,
            "bloom_level": q.get("bloom_level", ""),
            "explanation": q.get("explanation", ""),
        })

    return formatted


# =============================================================================
# TEXT BLOCK LOADING
# =============================================================================

def load_text_blocks(json_file: str) -> list[str]:
    """Load and filter text blocks from extracted PDF data."""
    path = Path(json_file)
    if not path.exists():
        logger.error(f"File not found: {json_file}")
        return []

    with path.open("r", encoding="utf-8") as f:
        pages = json.load(f)

    blocks = []

    # Calculate font size threshold for heading detection
    font_sizes = [
        float(block.get("font_size", 0))
        for page in pages
        for block in page.get("blocks", [])
        if block.get("font_size", 0) > 0
    ]

    header_threshold = None
    if font_sizes:
        sorted_sizes = sorted(font_sizes)
        mid = len(sorted_sizes) // 2
        median_size = (
            sorted_sizes[mid]
            if len(sorted_sizes) % 2
            else (sorted_sizes[mid - 1] + sorted_sizes[mid]) / 2
        )
        header_threshold = median_size * 1.2

    # Extract meaningful text blocks
    for page in pages:
        for block in page.get("blocks", []):
            text = (block.get("text") or "").strip()
            btype = block.get("type", "text")
            fsize = block.get("font_size") or 0

            if not text:
                continue

            # Include longer paragraphs and bullets
            if btype in ("text", "bullet") and len(text) > 40:
                blocks.append(text)
                continue

            # Include headings (by size or type)
            is_heading = (
                (header_threshold and fsize >= header_threshold) or
                btype.startswith("header_")
            )
            if is_heading and len(text) > 5:
                blocks.append(text)

    if not blocks:
        logger.warning("No suitable text blocks found in extracted data")
        return []

    # Limit to MAX_BLOCKS
    if len(blocks) > MAX_BLOCKS:
        logger.info(f"Using first {MAX_BLOCKS} of {len(blocks)} blocks")
        blocks = blocks[:MAX_BLOCKS]

    logger.info(f"Loaded {len(blocks)} text blocks for processing")
    return blocks


# =============================================================================
# QUESTION GENERATION
# =============================================================================

def generate_questions_from_blocks(blocks: list[str]) -> list[dict]:
    """Generate MCQ questions from text blocks using Ollama."""
    all_questions = []

    for i, block in enumerate(blocks, start=1):
        logger.info(f"Processing block {i}/{len(blocks)}")
        logger.debug(f"Block preview: {block[:200]}...")

        prompt = build_prompt(block)

        try:
            raw_answer = call_ollama(prompt)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is 'ollama serve' running?")
            break
        except requests.exceptions.Timeout:
            logger.error(f"Timeout on block {i}")
            continue
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            continue

        # Parse response
        parsed = extract_json_from_text(raw_answer)
        if not parsed:
            logger.warning(f"Could not parse JSON from block {i}")
            continue

        questions = normalize_questions(parsed)
        if questions:
            all_questions.extend(questions)
            logger.info(f"Generated {len(questions)} questions")

            # Stop early if target reached
            if len(all_questions) >= TARGET_QUESTIONS:
                logger.info(f"Reached target of {TARGET_QUESTIONS} questions")
                break
        else:
            logger.warning(f"No valid questions from block {i}")

    # Trim to target count
    if len(all_questions) > TARGET_QUESTIONS:
        all_questions = all_questions[:TARGET_QUESTIONS]

    return all_questions


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main question generation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Starting question generation")

    # Load text blocks
    blocks = load_text_blocks(EXTRACTED_JSON)
    if not blocks:
        logger.error("No text blocks loaded. Exiting.")
        return

    # Generate questions
    logger.info("Generating questions from blocks...")
    questions = generate_questions_from_blocks(blocks)
    formatted_questions = format_questions(questions)
    logger.info(f"Total questions generated: {len(formatted_questions)}")

    # Build output structure
    result = {
        "llm_provider": "ollama",
        "model": OLLAMA_MODEL,
        "question_type": "multiple_choice",
        "total_blocks_used": len(blocks),
        "questions": formatted_questions,
    }

    # Save to file
    out_path = Path(OUTPUT_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Questions saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
