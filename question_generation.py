# flake8: noqa: E501
"""Generate multiple-choice questions from extracted PDF content using
Ollama."""

import argparse
import json
import logging
import textwrap
from pathlib import Path
from typing import Optional

import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

EXTRACTED_JSON = Path("all_output") / "extracted_text_output" / "extracted_text.json"
OUTPUT_JSON = Path("all_output") / "generated_questions_output" / "questions.json"

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

def build_prompt(text_block: str, topic_hint: str | None = None) -> str:
    """Build structured prompt for MCQ generation with quality guidelines."""
    # Template encodes quality rules (Bloom levels, scenarios, 4 options).
    prompt_template = """
    You can be either an experienced university lecturer creating exam questions or a super
    curious student trying to test your understanding of complex material.
    
    Before writing the final output, THINK STEP BY STEP (internally):
    1. Read the source text carefully and identify the most important concepts, principles, 
    and methods related to "{topic_hint}".
    2. For each Bloom level (remember, understand, apply), decide which concepts are most suitable.
    3. Draft candidate question stems and options for each level.
    4. Check that each question matches its intended Bloom level, is answerable from the source text, 
    and follows all quality rules below.
    5. Only then write the final questions in the required JSON format.
    Do NOT include this plan or your reasoning in the output; return ONLY the JSON.
    Then, follow the detailed instructions below successively to generate high-quality multiple-choice questions.

    -->Design {question_count} HIGH-QUALITY, EXAM-STYLE MULTIPLE-CHOICE QUESTIONS in ENGLISH
    for graduate or early graduate students (that means higher education/academia is the target group)
    based on those requirements:

    1. LANGUAGE & CONTENT
       - Write EVERYTHING in ENGLISH
       - Base all questions ONLY on the source text. Do NOT invent external facts.
       - Do NOT ask about word meanings, figures, translations, or quoted phrases.
       - Focus on concepts, principles, design choices, trade-offs, and applications within the topic "{topic_hint}".

    2. BLOOM LEVELS (ONLY 3 LEVELS ARE ALLOWED)
    Use exactly these Bloom levels (1, 2 and 3) and definitions (revised Bloom’s taxonomy):
    - "remember": the student recalls or recognizes a key fact, term, or definition.
    - "understand": the student explains, interprets, summarizes, or classifies a concept in their own words.
    - "apply": the student uses a concept, rule, or principle in a concrete situation or short scenario.

    Across all {question_count} questions:
    - At least 1 question must be "remember".
    - At least 1 question must be "understand".
    - At least 1 question must be "apply" (e.g. scenario, choosing the best method, detecting a violation).

    You are free to choose which specific question uses which Bloom level, but each question must have
    exactly ONE of: "remember", "understand", "apply".

    3. QUESTION QUALITY & TOPIC COVERAGE
    For EACH question:
    - Write a clear, self-contained QUESTION STEM that can be read and understood without seeing the options.
    - Paraphrase ideas; do NOT copy sentences verbatim from the source text.
    - Target important ideas, not trivial details (no page numbers, no isolated buzzwords).
    - Stems must reference specific concepts or techniques from the source text
    (e.g. feedback, interaction principles, visual encoding, modularity, evaluation methods);
    do NOT refer to "the text", "the slides" or "the author".
    - BAN phrases like "according to the text/author" or "according to the slides".
    If such wording would appear, rewrite the stem to name the specific concept directly.
    - Avoid vague stems like "What is the main challenge ... ?".
    Instead, ask about a concrete concept or a brief scenario / use case.
    - At least 2 stems should include a short scenario or example that tests applying a concept,
    not just recalling it (this is ideal for "apply" or strong "understand" questions).
    - Cover different concepts across the questions; do NOT repeat the same idea twice.
    - Vary stem styles (definition/contrast, scenario/application, diagnosis/consequence).

    4. ANSWER OPTIONS
    For EACH question:
    - Provide EXACTLY 4 answer options: A, B, C, D.
    - ONLY ONE option is fully correct.
    - The correct option must be clearly supported by the source text and sound reasoning.
    - The 3 distractors must be:
        * conceptually related to the topic of the stem,
        * plausible for a partially informed student,
        * clearly incorrect for an expert who has understood the material.
    - All options should be similar in length, style, and level of detail.
    - Make all options mutually exclusive (no overlapping meanings).
    - Do NOT use "all of the above" or "none of the above".
    
    5. EXPLANATION
    For EACH question:
    - Write 2-4 sentences that:
        * justify why the correct option is correct, explicitly using concepts from the source text, and
        * briefly explain why each distractor is not the best answer.


    OUTPUT FORMAT (JSON):
    
    - Return a single JSON array with {question_count} objects.
    - Do NOT wrap the JSON in Markdown code fences.
    - Do NOT include any additional text before or after the JSON.

    Each object MUST have exactly these fields:
    - "question": the question stem (string)
    - "options": an array of 4 answer options in order [A, B, C, D]
    - "correct_answer_index": the index (0-3) of the correct option
    - "bloom_level": one of ["remember", "understand", "apply"]
    - "explanation": explanation text (string)

    EXAMPLE:
    [
      {{
        "question": "A user drags a document onto a trash bin icon and an animation shows 
        the document shrinking into the bin. Which principle of interaction does this best illustrate?",
        "options": [
          "Error Prevention",
          "Visibility of system status",
          "Flexibility and efficiency of use",
          "Consistency"
        ],
        "correct_answer_index": 1,
        "bloom_level": "understand",
        "explanation": "Animation makes the system state visible by showing what happens to the document.
        It does not mainly prevent errors or enforce consistency; its primary role is to keep the user informed about the ongoing action."
      }}, 
      {{"question": "The remarkable principle of Mobile 2.0 is:",
        "options": [
          "Recognising that we are not only the consumers.",
          "Recognising that we are the Lords of the Mobile market.",
          "Recognising that we are in a new age of consumerization.",
          "Recognising that we are not recognised at all."
        ],
        "correct_answer_index": 0,
        "bloom_level": "understand",
        "explanation": "Mobile 2.0 emphasizes that users are co-creators, not just consumers, aligning with option A. The other options overstate control, misstate the concept, or suggest a lack of recognition."
      }},
      {{"question": "Which of the following is the correct color association?",
        "options": [
          "Yellow — Go, OK, clear, vegetation, safety.",
          "Red — Stop, fire, hot, danger.",
          "Green — Cold, water, calm, sky, neutrality.",
          "Blue — Caution, slow, test."
        ],
        "correct_answer_index": 1,
        "bloom_level": "remember",
        "explanation": "Red conventionally signals stop/danger/hot, so option B is correct. Yellow usually signals caution/go depending on context, green signals go/safety, and blue does not conventionally mean caution."
      }},
      {{"question": "A special type of overlapping window that has the windows automatically arranged in a regular progression is:",
        "options": [
          "Tiled window.",
          "Cascading windows.",
          "Primary window.",
          "Secondary window."
        ],
        "correct_answer_index": 1,
        "bloom_level": "remember",
        "explanation": "Cascading windows overlap with a regular offset. Tiled windows do not overlap; primary/secondary are roles, not arrangement styles."
      }},
      {{"question": "A checkout form repeatedly shows errors after submission because users miss mandatory fields. Which design change best applies error prevention?",
        "options": [
          "Add field-level inline validation as soon as a required field loses focus.",
          "Hide all optional fields behind an accordion to reduce clutter.",
          "Move the error messages to the bottom of the page.",
          "Increase the form font size on every field."
        ],
        "correct_answer_index": 0,
        "bloom_level": "apply",
        "explanation": "Applying error prevention means stopping mistakes early. Inline validation on blur gives immediate feedback before submission. The other options do not directly prevent the error condition."
      }},
      {{"question": "You need to ensure users verify a destructive action (delete all records). Which UI pattern best applies confirmation while keeping flow efficient?",
        "options": [
          "Replace the delete action with an undo after the fact without warning.",
          "Trigger the delete immediately to avoid extra clicks.",
          "Hide the delete action behind a submenu with no confirmation.",
          "Add a modal dialog asking for typed confirmation before executing."
        ],
        "correct_answer_index": 3,
        "bloom_level": "apply",
        "explanation": "A typed confirmation modal for destructive, irreversible actions applies error prevention and explicit consent. The other options either remove safeguards or defer them until after damage is done."
      }}
    ]

    SOURCE TEXT:
    <<<SOURCE_TEXT_START>>>
    {source_text}
    <<<SOURCE_TEXT_END>>>

    Generate {question_count} questions now. Return ONLY valid JSON, no additional text.
    """

    # Optionally prepend a topic hint to the source text without changing the prompt body.
    if topic_hint:
        text_block = f"[Topic hint: {topic_hint}]\n\n{text_block}"

    hint_value = topic_hint or "your topic"
    return textwrap.dedent(prompt_template).format(
        source_text=text_block,
        question_count=TARGET_QUESTIONS,
        topic_hint=hint_value,
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
        logger.error(
            "Ollama request timed out after %ss",
            OLLAMA_TIMEOUT,
        )
        raise
    except requests.exceptions.RequestException as exc:
        logger.error("Ollama request failed: %s", exc)
        raise


# =============================================================================
# JSON PARSING
# =============================================================================

def extract_json_from_text(text: str) -> dict | None:  # pylint: disable=too-many-branches
    """Extract JSON from LLM output (handles code fences/noise)."""
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

    # Heuristic: wrap concatenated objects in a list if no outer brackets
    stripped = cleaned.strip()
    if stripped.startswith("{") and not stripped.startswith("["):
        try:
            parsed = json.loads(f"[{stripped}]")
            return {"questions": parsed}
        except json.JSONDecodeError:
            # Sometimes the model appends a trailing ']' without a leading '['
            if stripped.endswith("]"):
                try:
                    parsed = json.loads(f"[{stripped}")
                    return {"questions": parsed}
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
            if isinstance(parsed, list):
                return {"questions": parsed}
            return parsed
        except json.JSONDecodeError:
            continue

    logger.warning("Could not parse JSON from LLM output")
    return None


def normalize_questions(parsed: dict | list) -> list[dict]:
    """Normalize question format; accept list or dict with 'questions'."""
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
    """Format questions with numbered prefixes and letter labels."""
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

# pylint: disable=too-many-locals
def derive_topic_hint(pages: list) -> Optional[str]:
    """Pick a default topic hint from the first page header/title."""
    if not pages or not isinstance(pages[0], dict):
        return None
    first = pages[0]
    blocks = first.get("blocks_hier") or first.get("blocks", [])
    # Prefer explicit headers on the first page.
    for block in blocks:
        if str(block.get("type", "")).startswith("header_"):
            title = (block.get("text") or "").strip()
            if title:
                return title
    # Fallback: first non-empty block text.
    for block in blocks:
        title = (block.get("text") or "").strip()
        if title:
            return title[:120]
    return None


def load_text_blocks(json_file: str) -> tuple[list[str], Optional[str]]:
    """Load and filter text blocks from extracted PDF data."""
    path = Path(json_file)
    if not path.exists():
        logger.error("File not found: %s", json_file)
        return [], None

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    # Support both legacy (list of pages) and new structured JSON with "pages".
    pages = data.get("pages") if isinstance(data, dict) else data
    if not isinstance(pages, list):
        logger.error("Unexpected JSON structure: expected list or dict['pages']")
        return [], None

    topic_hint = derive_topic_hint(pages)
    blocks = []

    # Calculate font size threshold for heading detection
    font_sizes = [
        float(block.get("font_size", 0))
        for page in pages
        for block in (
            page.get("blocks_hier")
            or page.get("blocks", [])
            if isinstance(page, dict)
            else []
        )
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
        if not isinstance(page, dict):
            continue
        page_blocks = page.get("blocks_hier") or page.get("blocks", [])
        for block in page_blocks:
            text = (block.get("text") or "").strip()
            btype = block.get("type", "text")
            fsize = block.get("font_size") or 0

            if not text:
                continue

            # Prefer longer paragraphs/bullets to avoid noisy fragments.
            if btype in ("text", "bullet") and len(text) > 40:
                blocks.append(text)
                continue

            # Treat large text or explicit headers as section anchors.
            is_heading = (
                (header_threshold and fsize >= header_threshold) or
                btype.startswith("header_")
            )
            if is_heading and len(text) > 5:
                blocks.append(text)

    if not blocks:
        logger.warning("No suitable text blocks found in extracted data")
        return [], topic_hint

    # Limit to MAX_BLOCKS to keep the prompt concise.
    if len(blocks) > MAX_BLOCKS:
        logger.info("Using first %s of %s blocks", MAX_BLOCKS, len(blocks))
        blocks = blocks[:MAX_BLOCKS]

    logger.info("Loaded %s text blocks for processing", len(blocks))
    return blocks, topic_hint


# =============================================================================
# QUESTION GENERATION
# =============================================================================

def generate_questions_from_blocks(
    blocks: list[str],
    topic_hint: str | None = None,
) -> list[dict]:
    """Generate MCQ questions from text blocks using Ollama."""
    all_questions = []

    for i, block in enumerate(blocks, start=1):
        logger.info("Processing block %s/%s", i, len(blocks))
        logger.debug("Block preview: %s...", block[:200])

        prompt = build_prompt(block, topic_hint=topic_hint)

        try:
            raw_answer = call_ollama(prompt)
        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot connect to Ollama. Is 'ollama serve' running?"
            )
            break
        except requests.exceptions.Timeout:
            logger.error("Timeout on block %s", i)
            continue
        except requests.exceptions.RequestException as exc:
            logger.error("Error calling Ollama: %s", exc)
            continue

        # Parse and normalize LLM output; skip block on parse failure.
        parsed = extract_json_from_text(raw_answer)
        if not parsed:
            # Persist raw response for debugging when parsing fails.
            debug_dir = Path("all_output") / "generated_questions_output" / "raw_responses"
            debug_dir.mkdir(parents=True, exist_ok=True)
            raw_path = debug_dir / f"block_{i}.txt"
            raw_path.write_text(raw_answer, encoding="utf-8")
            logger.warning("Could not parse JSON from block %s (saved to %s)", i, raw_path)
            continue

        questions = normalize_questions(parsed)
        if questions:
            all_questions.extend(questions)
            logger.info("Generated %s questions", len(questions))

            # Stop early once we hit the target count to keep runtime short.
            if len(all_questions) >= TARGET_QUESTIONS:
                logger.info("Reached target of %s questions", TARGET_QUESTIONS)
                break
        else:
            logger.warning("No valid questions from block %s", i)

    # Trim to target count
    if len(all_questions) > TARGET_QUESTIONS:
        all_questions = all_questions[:TARGET_QUESTIONS]

    return all_questions


# =============================================================================
# MAIN
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    """CLI to allow custom input/output paths."""
    parser = argparse.ArgumentParser(
        description="Generate MCQs from extracted text JSON."
    )
    parser.add_argument(
        "--input-json",
        default=str(EXTRACTED_JSON),
        help=f"Path to extracted text JSON (default: {EXTRACTED_JSON})",
    )
    parser.add_argument(
        "--output-json",
        default=str(OUTPUT_JSON),
        help=f"Path to write generated questions (default: {OUTPUT_JSON})",
    )
    parser.add_argument(
        "--topic-hint",
        help="Optional topic hint to prepend to the source text",
    )
    parser.add_argument(
        "--auto-version",
        action="store_true",
        help="Write to incrementing files (e.g., questions_v1.json, v2...)",
    )
    return parser.parse_args(argv)


def next_versioned_path(base: Path) -> Path:
    """Return the next available _vN filename (starting at v1)."""
    stem, suffix = base.stem, base.suffix
    parent = base.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_v{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def main(argv=None):
    """Main question generation pipeline."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Starting question generation")

    # Load text blocks
    in_path = Path(args.input_json)
    out_path = Path(args.output_json)
    if args.auto_version:
        # Always write to a new numbered file to keep prior runs.
        out_path = next_versioned_path(out_path)

    blocks, derived_topic = load_text_blocks(in_path)
    if not blocks:
        logger.error("No text blocks loaded. Exiting.")
        return

    topic_hint = args.topic_hint or derived_topic

    # Generate questions
    logger.info("Generating questions from blocks...")
    questions = generate_questions_from_blocks(
        blocks, topic_hint=topic_hint
    )
    formatted_questions = format_questions(questions)
    logger.info(
        "Total questions generated: %s",
        len(formatted_questions),
    )

    # Build output structure
    result = {
        "llm_provider": "ollama",
        "model": OLLAMA_MODEL,
        "question_type": "multiple_choice",
        "total_blocks_used": len(blocks),
        "topic_hint": topic_hint,
        "questions": formatted_questions,
    }

    # Save to file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info("Questions saved to %s", out_path)


if __name__ == "__main__":
    main()
