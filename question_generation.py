import json
import logging
import textwrap
from pathlib import Path

import requests  # HTTP calls to Ollama

# CONFIGURATION
# ============================================================================

# Read extracted text/blocks from this file
EXTRACTED_JSON = "extracted_text.json"

# Write generated questions here
OUTPUT_JSON = "questions.json"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"

# Limit how many text blocks to send
MAX_BLOCKS = 8
# Target number of questions
TARGET_QUESTIONS = 5
OLLAMA_TIMEOUT = 60  # seconds

# Logging
logger = logging.getLogger(__name__)


# ============================================================================
# Helper: build prompt
# ============================================================================

def build_prompt(text_block: str) -> str:
    """
    Build the prompt with strict MCQ requirements (Bloom levels, quality rules, JSON shape).
    """
    prompt_template = """
    You are an experienced university lecturer.

    Your goal:
    Design HIGH-QUALITY, EXAM-STYLE MULTIPLE-CHOICE QUESTIONS in ENGLISH
    for undergraduate / early graduate students, based ONLY on the source text
    provided below.

    Theoretical background:
    - Align each question with a clear learning objective and Bloom's taxonomy level
      (at least "Understand", preferably "Apply" or "Analyze").
    - Follow evidence-based MCQ item-writing guidelines:
      * clear, focused stem that can be answered before seeing the options;
      * one best correct answer;
      * 3 distractors that are plausible, homogeneous, and based on common
        misconceptions or typical confusions;
      * no "all of the above" or "none of the above";
      * avoid trivial recall of isolated phrases or word meanings.

    STRICT RULES:
    1. LANGUAGE
       - Write EVERYTHING in ENGLISH.
       - Do NOT ask about the meaning or translation of German phrases or any
         quoted sentence from the text.
       - Do NOT create questions that only test vocabulary or wording such as:
         "What does the phrase 'Systemkomponenten Informationen austauschen' mean?"

    2. NUMBER AND LEVEL OF QUESTIONS
       - Create EXACTLY {question_count} multiple-choice questions.
       - Target: undergraduate / early graduate students in a technical / design /
         computer science-related field.
       - At least 2 questions must target Bloom level "Understand" or higher.
       - At least 1 question should require "Apply" or "Analyze" level reasoning
         (e.g., choosing the best example, detecting a violation, or selecting the
         most appropriate concept for a scenario).

    3. QUESTION QUALITY
       For EACH question:
       - Write a clear, self-contained QUESTION STEM that focuses on an important
         concept, definition, principle, or design decision from the text.
       - Avoid copying whole sentences from the text; paraphrase and generalize.
       - Avoid purely factual lookup questions (e.g., "On which page is X defined?").

    4. ANSWER OPTIONS
       For EACH question:
       - Provide EXACTLY 4 answer options: A, B, C, D.
       - ONLY ONE option is fully correct.
       - The correct option must be clearly supported by the source text and by
         sound reasoning.
       - The 3 distractors must be:
           * conceptually related to the topic,
           * plausible for a partially informed student,
           * clearly incorrect for an expert.
       - Make all 4 options similar in style, length, and level of detail.
       - Do NOT use "all of the above" or "none of the above".

    5. FEEDBACK / EXPLANATION
       For EACH question:
       - Provide a short explanation (2-4 sentences) that:
           * justifies why the correct option is correct, and
           * briefly explains why the other options are not the best answer.

    OUTPUT FORMAT:
    Return the result as a JSON array with 3 elements.
    Each element must have the following fields:
    - "question": the question stem (string)
    - "options": an array of 4 answer options in order [A, B, C, D]
    - "correct_answer_index": the index (0-3) of the correct option
    - "bloom_level": one of ["remember", "understand", "apply", "analyze",
                             "evaluate", "create"]
    - "explanation": a short explanation for the correct answer

    Example of the required JSON structure (the content here is ONLY an example):

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
        "explanation": "Good feedback keeps users informed about what the system is doing and what will happen next. It does not magically prevent all errors or remove the need for training; instead, it helps users build a correct mental model of the system."
      }}
    ]

    Now generate the JSON output for the following source text:

    <<<SOURCE_TEXT_START>>>
    {source_text}
    <<<SOURCE_TEXT_END>>>
    """
    return textwrap.dedent(prompt_template).format(
        source_text=text_block,
        question_count=TARGET_QUESTIONS,
    )


# ============================================================================
# Helper: call Ollama
# ============================================================================

def call_ollama(prompt: str) -> str:
    """
    Send the prompt to Ollama and return the raw response text.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out after %s seconds", OLLAMA_TIMEOUT)
        raise
    except requests.exceptions.RequestException as exc:
        logger.error("Ollama request failed: %s", exc)
        raise

    data = response.json()

    # Ollama puts the generated text in the "response" field
    return data.get("response", "")


# ============================================================================
# Helper: extract JSON from LLM output
# ============================================================================

def extract_json_from_text(text: str):
    """
    Parse JSON from LLM output that may include code fences or noise.
    """
    cleaned = text
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]

    # First try a direct json.loads
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return {"questions": parsed}
        return parsed
    except json.JSONDecodeError:
        pass

    candidates = []

    # Second try: array segment
    a_start = cleaned.find("[")
    a_end = cleaned.rfind("]") + 1
    if a_start != -1 and a_end > a_start:
        candidates.append(cleaned[a_start:a_end])

    # Third try: object segment
    o_start = cleaned.find("{")
    o_end = cleaned.rfind("}") + 1
    if o_start != -1 and o_end > o_start:
        candidates.append(cleaned[o_start:o_end])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return {"questions": parsed}
            return parsed
        except json.JSONDecodeError as e:
            logger.debug("JSON parse error on candidate: %s", e)

    logger.warning("JSON parse error: could not decode any candidate segment.")
    return None


def normalize_questions(parsed):
    """
    I accept either a list of question dicts or a dict with a 'questions' list,
    and I normalize correct_index -> correct_answer_index.
    """
    questions = None
    if isinstance(parsed, list):
        questions = parsed
    elif isinstance(parsed, dict):
        maybe = parsed.get("questions")
        if isinstance(maybe, list):
            questions = maybe

    if not questions:
        return []

    normalized = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        if "correct_answer_index" not in q and "correct_index" in q:
            q["correct_answer_index"] = q["correct_index"]
        normalized.append(q)
    return normalized


def format_questions(questions):
    """
    I reshape questions and prefix options with letters for clarity.
    """
    formatted = []
    letters = ["A", "B", "C", "D"]

    for idx, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            continue

        options = q.get("options", [])
        prefixed = []
        for i, opt in enumerate(options):
            letter = letters[i] if i < len(letters) else chr(65 + i)
            prefixed.append(f"{letter}) {opt}")

        correct_idx = q.get("correct_answer_index")

        correct_letter = letters[correct_idx] if isinstance(correct_idx, int) and 0 <= correct_idx < len(letters) else None

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


# ============================================================================
# SCHRITT 1: Textbloecke aus extracted.json lesen
# ============================================================================

def load_text_blocks(json_file: str):
    """
    Read extracted_text.json and collect text blocks (headings, bullets, longer text).
    """
    path = Path(json_file)
    if not path.exists():
        logger.error("Error: %s not found.", json_file)
        return []

    with path.open("r", encoding="utf-8") as f:
        pages = json.load(f)

    blocks = []

    # I derive a rough font-size reference to spot headings by size
    font_sizes = []
    for page in pages:
        for block in page.get("blocks", []):
            size = block.get("font_size")
            if isinstance(size, (int, float)) and size > 0:
                font_sizes.append(float(size))

    header_threshold = None
    if font_sizes:
        sorted_sizes = sorted(font_sizes)
        mid = len(sorted_sizes) // 2
        median_size = sorted_sizes[mid] if len(sorted_sizes) % 2 else (sorted_sizes[mid - 1] + sorted_sizes[mid]) / 2
        header_threshold = median_size * 1.2  # treat clearly larger text as heading

    for page in pages:
        for block in page.get("blocks", []):
            text = (block.get("text") or "").strip()
            btype = block.get("type", "text")
            fsize = block.get("font_size") or 0

            if not text:
                continue

            # Keep longer paragraphs or bullets
            if btype in ("text", "bullet") and len(text) > 40:
                blocks.append(text)
                continue

            # Treat larger text as heading or keep explicit header_* types
            if (header_threshold and fsize >= header_threshold) or btype.startswith("header_"):
                if len(text) > 5:
                    blocks.append(text)

    if not blocks:
        logger.warning("No suitable text blocks found in extracted_text.json")
        return []

    # Cap the number of blocks
    if len(blocks) > MAX_BLOCKS:
        logger.info("Using only first %s of %s text blocks.", MAX_BLOCKS, len(blocks))
        blocks = blocks[:MAX_BLOCKS]

    logger.info("Loaded %s text blocks for question generation.", len(blocks))
    return blocks


# ============================================================================
# STEP 2: Generate questions for all blocks
# ============================================================================

def generate_questions_from_blocks(blocks):
    """
    Take text blocks and generate MC questions via Ollama + Llama2.
    """
    all_questions = []

    for i, block in enumerate(blocks, start=1):
        logger.info("Processing block %s/%s", i, len(blocks))
        logger.debug("Block preview (first 200 chars): %r", block[:200])

        prompt = build_prompt(block)

        try:
            raw_answer = call_ollama(prompt)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is 'ollama serve' running?")
            break
        except requests.exceptions.Timeout:
            logger.error("Timeout while calling Ollama for block %s", i)
            continue
        except Exception as e:
            logger.error("Error while calling Ollama: %s", e)
            continue

        parsed = extract_json_from_text(raw_answer)
        if not parsed:
            logger.warning("Could not parse JSON from model output (block %s).", i)
            continue

        questions = normalize_questions(parsed)
        if questions:
            all_questions.extend(questions)
            logger.info("Added %s questions.", len(questions))
            if len(all_questions) >= TARGET_QUESTIONS:
                logger.info("Reached target of %s questions; stopping early.", TARGET_QUESTIONS)
                break
        else:
            logger.warning("No questions found in parsed output (block %s).", i)

    if len(all_questions) > TARGET_QUESTIONS:
        all_questions = all_questions[:TARGET_QUESTIONS]
    return all_questions


# ============================================================================
# STEP 3: Merge and save
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 1. Load text blocks
    blocks = load_text_blocks(EXTRACTED_JSON)
    if not blocks:
        return

    # 2. Generate questions
    questions = generate_questions_from_blocks(blocks)
    formatted_questions = format_questions(questions)
    logger.info("Total questions generated: %s", len(formatted_questions))

    # 3. Build result object
    result = {
        "llm_provider": "ollama",
        "model": OLLAMA_MODEL,
        "question_type": "multiple_choice",
        "total_blocks_used": len(blocks),
        "questions": formatted_questions,
    }

    # 4. Write JSON file
    out_path = Path(OUTPUT_JSON)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Questions saved to %s", OUTPUT_JSON)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Execute main when run as a script
    main()
