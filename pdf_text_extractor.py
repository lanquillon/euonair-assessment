"""
PDF text extractor with improved structure recognition.

Robust handling of different PDF layouts: presentations, documents, scanned pages.
Detects headings, bullets, tables through multiple heuristics.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

# Optional OCR dependencies
try:
    import pytesseract
    from PIL import Image, ImageEnhance
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Optional table extraction dependency
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_OUTPUT_JSON = "extracted_text.json"
DEFAULT_OUTPUT_MD = "extracted_text.md"

# Header/footer detection regions (in points; 1 pt = 1/72 inch)
HEADER_HEIGHT = 80
FOOTER_HEIGHT = 80

# OCR settings
USE_OCR_FOR_IMAGE_PAGES = True
OCR_MIN_TEXT_LENGTH = 30
OCR_LANG = "eng+deu"
FORCE_OCR = False
DETECT_TABLES = False  # Often over-detects; better to rely on formatting

# Tesseract path (optional)
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

# Content detection heuristics
HEADING_REL_THRESHOLD = 1.15  # Font size multiplier
INDENT_STEP = 12.0  # Default points per indentation level (adjusted per median font size)
Y_TOLERANCE = 5  # Pixels for row grouping
X_TOLERANCE = 30  # Pixels for column alignment

# Aggressive structure detection for presentation slides
AGGRESSIVE_HEADING_DETECTION = True  # Treat bold/large text as headings
SHORT_LINE_IS_HEADING = True  # Lines <60 chars + large font = heading

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class PdfOpenError(RuntimeError):
    """Raised when the PDF cannot be opened."""
    pass

class PasswordRequiredError(RuntimeError):
    """Raised when the PDF is encrypted."""
    pass


# =============================================================================
# EXTRACTION
# =============================================================================

def parse_page_spec(page_spec: str) -> set[int]:
    """Parse a comma-separated page spec like '1-3,5' into a set of 1-based page numbers."""
    pages: set[int] = set()
    if not page_spec:
        return pages

    for part in page_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))

    return {p for p in pages if p > 0}


def _requires_password(doc) -> bool:
    """Check if PDF needs password."""
    for attr in ("needs_pass", "needs_passwd", "is_encrypted"):
        if hasattr(doc, attr):
            val = getattr(doc, attr)
            try:
                return val() if callable(val) else bool(val)
            except Exception:
                return False
    return False


def extract_pages(pdf_path: str, password: Optional[str] = None, page_filter: Optional[set[int]] = None) -> list[dict]:
    """Extract text blocks with metadata from PDF pages."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise PdfOpenError(f"Cannot open PDF {pdf_path}: {exc}") from exc

    try:
        if _requires_password(doc):
            if not password:
                raise PasswordRequiredError(f"PDF {pdf_path} is password-protected")
            auth_method = getattr(doc, "authenticate", None)
            if callable(auth_method):
                authed = auth_method(password)
                if not authed and _requires_password(doc):
                    raise PasswordRequiredError(f"Invalid password for PDF {pdf_path}")
    except PasswordRequiredError:
        raise
    except Exception as exc:
        raise PasswordRequiredError(f"Password check failed: {exc}") from exc

    pages_text = []

    for i, page in enumerate(doc, start=1):
        if page_filter and i not in page_filter:
            continue

        page_dict = page.get_text("dict")
        page_blocks = []

        # Extract text blocks with positioning metadata
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                text = clean_text("".join(s.get("text", "") for s in spans))
                if not text:
                    continue

                x0 = min(s["bbox"][0] for s in spans)
                y0 = min(s["bbox"][1] for s in spans)
                x1 = max(s["bbox"][2] for s in spans)
                y1 = max(s["bbox"][3] for s in spans)

                font_size = max((s.get("size", 0) for s in spans), default=0)
                font_name = spans[0].get("font", "") if spans else ""
                is_bold = any(
                    "bold" in s.get("font", "").lower() or "bf" in s.get("font", "").lower()
                    for s in spans
                )

                page_blocks.append({
                    "text": text,
                    "bbox": [x0, y0, x1, y1],
                    "x0": x0,
                    "y0": y0,
                    "font_size": float(font_size),
                    "font_name": font_name,
                    "is_bold": is_bold,
                })

        if USE_OCR_FOR_IMAGE_PAGES:
            total_text_len = sum(len(b["text"]) for b in page_blocks)
            should_ocr = FORCE_OCR or total_text_len < OCR_MIN_TEXT_LENGTH

            if should_ocr:
                logger.info(f"Page {i}: Low text ({total_text_len} chars) - trying OCR")
                ocr_blocks = ocr_page_simple(page)
                if ocr_blocks:
                    if total_text_len > 0 and not FORCE_OCR:
                        page_blocks.extend(ocr_blocks)
                    else:
                        page_blocks = ocr_blocks
                else:
                    logger.info(f"Page {i}: OCR returned no text")

        pages_text.append({"page": i, "blocks": page_blocks})

    return pages_text


def ocr_page_simple(page) -> list[dict]:
    """OCR a single PDF page."""
    if not OCR_AVAILABLE:
        logger.warning("pytesseract/Pillow not installed - skipping OCR")
        return []

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    try:
        pix = page.get_pixmap(dpi=400)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = ImageEnhance.Contrast(img.convert("L")).enhance(1.8)
        raw_text = pytesseract.image_to_string(img, lang=OCR_LANG).strip()
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract not found. Install and add to PATH or set TESSERACT_CMD")
        return []
    except Exception as exc:
        logger.error(f"OCR error: {exc}")
        return []

    if not raw_text:
        return []

    ocr_blocks = []
    for chunk in raw_text.split("\n\n"):
        chunk = clean_text(chunk)
        if chunk:
            ocr_blocks.append({
                "text": chunk,
                "bbox": [0, 0, page.rect.width, page.rect.height],
                "x0": 0,
                "y0": 0,
                "font_size": 12.0,
                "font_name": "ocr",
                "is_bold": False,
            })

    logger.info(f"OCR recognized {len(ocr_blocks)} block(s)")
    return ocr_blocks


# =============================================================================
# TABLE EXTRACTION (pdfplumber)
# =============================================================================

def extract_tables_pdfplumber(pdf_path: str, page_filter: Optional[set[int]] = None) -> dict[int, list[list[list[str]]]]:
    """Extract tables using pdfplumber; returns page -> list of tables (rows of cells)."""
    if not PDFPLUMBER_AVAILABLE:
        return {}

    tables_by_page: dict[int, list[list[list[str]]]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                if page_filter and idx not in page_filter:
                    continue
                try:
                    tables = page.extract_tables() or []
                except Exception as exc:
                    logger.debug(f"pdfplumber table extraction failed on page {idx}: {exc}")
                    continue
                clean_tables = []
                for tbl in tables:
                    if not tbl:
                        continue
                    normalized = []
                    for row in tbl:
                        normalized.append([clean_text(cell or "") for cell in row])
                    # drop empty tables
                    non_empty_rows = [row for row in normalized if any(cell for cell in row)]
                    if not non_empty_rows:
                        continue
                    # simple guard: need at least 2 rows with data and 2 columns with some data
                    col_counts = len(normalized[0]) if normalized else 0
                    if col_counts < 2:
                        continue
                    rows_with_data = sum(1 for row in normalized if any(cell for cell in row))
                    cols_with_data = sum(
                        1 for col_idx in range(col_counts)
                        if any(row[col_idx] for row in normalized if len(row) > col_idx)
                    )
                    if rows_with_data < 2 or cols_with_data < 2:
                        continue
                    clean_tables.append(normalized)
                if clean_tables:
                    tables_by_page[idx] = clean_tables
    except Exception as exc:
        logger.debug(f"pdfplumber could not open {pdf_path}: {exc}")

    return tables_by_page


def inject_tables(pages: list[dict], tables_by_page: dict[int, list[list[list[str]]]]) -> list[dict]:
    """Inject pdfplumber tables into page blocks as 'table' entries."""
    if not tables_by_page:
        return pages

    new_pages = []
    for page in pages:
        blocks = list(page.get("blocks", []))
        page_tables = tables_by_page.get(page.get("page"), [])
        for tbl in page_tables:
            blocks.append({
                "type": "table",
                "rows": tbl,
                "bbox": [0, 0, 0, 0],
                "num_cols": len(tbl[0]) if tbl and tbl[0] else 0,
                "text": "",
            })
        new_pages.append({**page, "blocks": blocks})
    return new_pages


# =============================================================================
# HEADER/FOOTER DETECTION
# =============================================================================

def detect_headers_footers(pages_text: list[dict]) -> set[str]:
    """Identify repetitive text in header/footer regions."""
    header_texts = defaultdict(int)
    footer_texts = defaultdict(int)

    for page_data in pages_text:
        blocks = page_data["blocks"]
        if not blocks:
            continue

        max_y = max(b["bbox"][3] for b in blocks)

        for block in blocks:
            bbox = block["bbox"]
            y_top, y_bottom = bbox[1], bbox[3]
            normalized = block["text"].lower().strip()

            if y_top < HEADER_HEIGHT:
                header_texts[normalized] += 1
            if y_bottom > (max_y - FOOTER_HEIGHT):
                footer_texts[normalized] += 1

    total_pages = len(pages_text)
    threshold = total_pages * 0.5
    header_footer_set = set()

    for text, count in header_texts.items():
        if count > threshold:
            header_footer_set.add(text)
            logger.info(f"Header: '{text}' ({count}/{total_pages} pages)")

    for text, count in footer_texts.items():
        if count > threshold:
            header_footer_set.add(text)
            logger.info(f"Footer: '{text}' ({count}/{total_pages} pages)")

    return header_footer_set


def is_page_number(text: str) -> bool:
    """Check if text is a page number."""
    normalized = text.lower().strip()
    patterns = [
        r"^-?\s*\d+\s*-?$",
        r"^(seite|page|p\.?)\s+\d+$",
        r"^\d+\s+(von|of|/)\s+\d+$",
        r"^ese\s+\d+\.\d+$",
    ]
    return any(re.match(pattern, normalized) for pattern in patterns)


# =============================================================================
# IMPROVED CONTENT CLASSIFICATION
# =============================================================================

def get_font_size_ranks(pages_text: list[dict]) -> dict[float, int]:
    """Map font sizes to ranks (0=largest)."""
    sizes = {
        float(b.get("font_size", 0))
        for p in pages_text
        for b in p.get("blocks", [])
        if b.get("font_size", 0) > 0
    }
    sorted_sizes = sorted(sizes, reverse=True)
    return {size: idx for idx, size in enumerate(sorted_sizes)}


def get_median_font_size(pages_text: list[dict]) -> float:
    """Compute median font size across all blocks."""
    all_sizes = sorted(
        b.get("font_size", 0)
        for p in pages_text
        for b in p.get("blocks", [])
        if b.get("font_size", 0)
    )
    return all_sizes[len(all_sizes) // 2] if all_sizes else 0.0


def clean_text(text: str) -> str:
    """Normalize common bad glyphs and stray symbols."""
    replacements = {
        "ƒ?": "",
        "ƒ?T": "",
        "¶¸": "",
        "ƒ\"": "",
        "ƒ?½": "",
        "ƒ?o": "",
        "ƒ?": "",
        "�": "",
        "": "",
        "•": "•",
        "·": "·",
        "–": "-",
        "—": "-",
        "\u00ad": "",  # soft hyphen
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.strip()


def is_slide_like_page(page: dict) -> bool:
    """Heuristic: slide-like pages are bullet heavy with short lines."""
    blocks = page.get("blocks", [])
    if not blocks:
        return False
    bullet_count = sum(1 for b in blocks if b.get("type") == "bullet")
    short_lines = sum(1 for b in blocks if len(b.get("text", "")) < 80)
    total = len(blocks)
    return (total <= 40 and bullet_count >= 3 and (bullet_count / max(total, 1)) >= 0.3 and (short_lines / max(total, 1)) >= 0.5)


def detect_header_type(
    block: dict,
    size_rank_map: dict[float, int],
    median_size: float,
    page_height: float | None = None
) -> str | None:
    """Determine if block is a heading (h1/h2/h3)."""
    size = float(block.get("font_size", 0))
    if size == 0:
        return None

    text = block.get("text", "")
    is_bold = block.get("is_bold", False)

    y_top = float(block.get("bbox", [0, 0, 0, 0])[1])
    near_top = y_top < (max(HEADER_HEIGHT, page_height * 0.12) if page_height else HEADER_HEIGHT)

    rank = size_rank_map.get(size)
    if rank == 0:
        return "h1"
    if rank == 1:
        return "h2" if near_top else "h3"
    if rank == 2:
        return "h3"

    if AGGRESSIVE_HEADING_DETECTION:
        if SHORT_LINE_IS_HEADING and len(text) < 60:
            if size >= median_size * HEADING_REL_THRESHOLD and (is_bold or near_top):
                return "h2"
        if is_bold and near_top:
            return "h2"
        if size >= median_size * HEADING_REL_THRESHOLD:
            if is_bold:
                return "h2"
            if near_top:
                return "h2"
            return "h3"

    if median_size and size >= median_size * HEADING_REL_THRESHOLD:
        if is_bold or near_top:
            return "h2"
        return "h3"

    return None


def detect_bullet_type(block: dict, page_min_x0: float | None, indent_step: float) -> tuple[bool, int]:
    """Detect bullet/list items (unordered or ordered)."""
    text = block.get("text", "").strip()
    x0 = float(block.get("x0", 0))

    # Unordered bullets: dash, asterisk, arrow, dot styles
    if m := re.match(r"^\s*([-*>\u2022·•▪◦‣–—])\s+(.*)$", text):
        block["text"] = m.group(2).strip()
        return True, 0

    # Ordered bullets: 1., 1), a., a), i., i)
    if m := re.match(r"^\s*((?:\d+|[A-Za-z]+)[\.\)])\s+(.*)$", text):
        block["text"] = m.group(2).strip()
        block["ordered_marker"] = m.group(1)
        return True, 0

    if page_min_x0 is None:
        return False, 0

    rel = max(0.0, x0 - page_min_x0)
    indent_level = int(rel // indent_step)
    word_count = len(text.split())

    if indent_level > 0 and word_count <= 25:
        return True, indent_level

    return False, 0


def classify_blocks(pages_text: list[dict]) -> list[dict]:
    """Classify blocks as headings, bullets, tables, or text."""
    size_rank_map = get_font_size_ranks(pages_text)

    median_size = get_median_font_size(pages_text)
    indent_step = max(INDENT_STEP, median_size * 1.2) if median_size else INDENT_STEP

    classified_pages = []

    for page_data in pages_text:
        blocks = page_data.get("blocks", [])
        page_min_x0 = min((b.get("x0", 0) for b in blocks), default=None)

        classified_blocks = []
        idx = 0

        while idx < len(blocks):
            block = blocks[idx]
            text = block.get("text", "")
            block_type = block.get("type", "text")
            meta: dict = {}

            if not DETECT_TABLES and block_type == "table":
                block_type = "text"

            if block_type == "table":
                classified_blocks.append(block)
                idx += 1
                continue

            if re.match(r"^\s*[-*>\u2022·•▪◦‣–—]\s*$", text):
                if idx + 1 < len(blocks):
                    next_b = blocks[idx + 1]
                    combined_text = next_b.get("text", "").strip()

                    bbox_a = block.get("bbox", [0, 0, 0, 0])
                    bbox_b = next_b.get("bbox", [0, 0, 0, 0])
                    combined_bbox = [
                        min(bbox_a[0], bbox_b[0]),
                        min(bbox_a[1], bbox_b[1]),
                        max(bbox_a[2], bbox_b[2]),
                        max(bbox_a[3], bbox_b[3]),
                    ]

                    indent = int(round(float(next_b.get("x0", 0)) // INDENT_STEP))
                    entry = {
                        "text": combined_text,
                        "bbox": combined_bbox,
                        "type": "bullet",
                        "indent_level": indent,
                    }
                    classified_blocks.append(entry)
                    idx += 2
                    continue
                else:
                    indent = int(round(float(block.get("x0", 0)) // INDENT_STEP))
                    block_type = "bullet"
                    meta["indent_level"] = indent

            else:
                is_bullet, indent = detect_bullet_type(block, page_min_x0, indent_step)
                if is_bullet:
                    block_type = "bullet"
                    meta["indent_level"] = indent
                    if "ordered_marker" in block:
                        meta["ordered"] = True
                        meta["marker"] = block.pop("ordered_marker", "")
                else:
                    if hdr := detect_header_type(block, size_rank_map, median_size):
                        block_type = f"header_{hdr}"
                        meta["heading_level"] = hdr

            entry = {"text": text, "bbox": block.get("bbox"), "type": block_type}
            entry.update(meta)
            classified_blocks.append(entry)
            idx += 1

        classified_pages.append({"page": page_data.get("page"), "blocks": classified_blocks})

    return classified_pages


# =============================================================================
# FILTERING
# =============================================================================

def filter_irrelevant_content(pages_text: list[dict], header_footer_set: set[str]) -> list[dict]:
    """Remove headers/footers and page numbers."""
    filtered_pages = []

    for page_data in pages_text:
        original_count = len(page_data["blocks"])

        filtered_blocks = []
        last_text = None
        for block in page_data["blocks"]:
            text = block["text"]
            normalized = text.lower().strip()

            if normalized in header_footer_set or is_page_number(text):
                continue

            # Drop exact consecutive duplicates (common on slides)
            if last_text is not None and normalized == last_text:
                continue

            filtered_blocks.append(block)
            last_text = normalized

        removed_count = original_count - len(filtered_blocks)
        if removed_count > 0:
            logger.info(f"  Page {page_data['page']}: removed {removed_count} blocks")

        filtered_pages.append({
            "page": page_data["page"],
            "blocks": filtered_blocks
        })

    return filtered_pages


# =============================================================================
# TABLE DETECTION
# =============================================================================

def detect_tables(pages_text: list[dict]) -> list[dict]:
    """Detect tables by Y/X alignment."""
    new_pages = []

    for page in pages_text:
        blocks = page.get("blocks", [])
        used = set()
        tables = []

        sorted_blocks = sorted(enumerate(blocks), key=lambda x: x[1].get("y0", 0))

        rows = []
        current_row = []
        current_y = None

        for idx, b in sorted_blocks:
            y = b.get("y0", 0)
            if current_y is None or abs(y - current_y) <= Y_TOLERANCE:
                current_row.append((idx, b))
                current_y = y if current_y is None else current_y
            else:
                if len(current_row) >= 2:
                    rows.append(current_row)
                current_row = [(idx, b)]
                current_y = y

        if len(current_row) >= 2:
            rows.append(current_row)

        i = 0
        while i < len(rows):
            table_rows = [rows[i]]
            j = i + 1
            first_x = sorted(b[1].get("x0", 0) for b in rows[i])

            while j < len(rows):
                curr_x = sorted(b[1].get("x0", 0) for b in rows[j])
                if len(curr_x) == len(first_x):
                    aligned = all(abs(a - b) <= X_TOLERANCE for a, b in zip(first_x, curr_x))
                    if aligned:
                        table_rows.append(rows[j])
                        j += 1
                        continue
                break

            if len(table_rows) >= 2:
                table_data = []
                all_indices = set()

                for row in table_rows:
                    sorted_cells = sorted(row, key=lambda x: x[1].get("x0", 0))
                    table_data.append([cell[1].get("text", "").strip() for cell in sorted_cells])
                    all_indices.update(cell[0] for cell in sorted_cells)

                all_blocks = [cell[1] for row in table_rows for cell in row]
                x0 = min(b.get("bbox", [0, 0, 0, 0])[0] for b in all_blocks)
                y0 = min(b.get("bbox", [0, 0, 0, 0])[1] for b in all_blocks)
                x1 = max(b.get("bbox", [0, 0, 0, 0])[2] for b in all_blocks)
                y1 = max(b.get("bbox", [0, 0, 0, 0])[3] for b in all_blocks)

                first_idx = min(all_indices)
                tables.append({
                    "rows": table_data,
                    "bbox": [x0, y0, x1, y1],
                    "start_idx": first_idx,
                    "num_cols": len(table_data[0]) if table_data else 0,
                })
                used.update(all_indices)

            i = j

        new_blocks = []
        for idx, b in enumerate(blocks):
            if idx in used:
                matching = [t for t in tables if t["start_idx"] == idx]
                if matching:
                    new_blocks.append({
                        "type": "table",
                        "rows": matching[0]["rows"],
                        "bbox": matching[0]["bbox"],
                        "num_cols": matching[0]["num_cols"],
                        "text": ""
                    })
            else:
                new_blocks.append(b)

        new_pages.append({
            "page": page.get("page"),
            "blocks": new_blocks,
            "tables": tables
        })

    return new_pages


# =============================================================================
# OUTPUT
# =============================================================================

def build_hierarchical_blocks(blocks: list[dict]) -> list[dict]:
    """Group bullets into hierarchy, preserving ordered markers."""
    result = []
    stack = []

    for b in blocks:
        node = dict(b)
        node["children"] = [c for c in b.get("children", [])]

        if b.get("type") != "bullet":
            stack.clear()
            result.append(node)
            continue

        indent = int(b.get("indent_level", 0))
        while stack and indent <= stack[-1][0]:
            stack.pop()

        if stack:
            stack[-1][1]["children"].append(node)
        else:
            result.append(node)

        stack.append((indent, node))

    return result


def save_json(pages_text: list[dict], path: str, encoding: str = "utf-8") -> None:
    """Save as JSON with hierarchy."""
    enriched = []
    for page in pages_text:
        hier = build_hierarchical_blocks(page.get("blocks", []))
        enriched.append({**page, "blocks_hier": hier})

    json_str = json.dumps(enriched, ensure_ascii=False, indent=2)
    Path(path).write_text(json_str, encoding=encoding)
    logger.info(f"JSON saved: {path}")


def format_markdown_table(rows: list[list[str]]) -> str:
    """Format as Markdown table."""
    if not rows or not rows[0]:
        return ""

    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]

    col_widths = [0] * max_cols
    for row in normalized:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    col_widths = [max(w, 3) for w in col_widths]

    lines = []
    header = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(normalized[0])) + " |"
    lines.append(header)
    separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    lines.append(separator)
    for row in normalized[1:]:
        data_row = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        lines.append(data_row)
    return "\n".join(lines)


def save_markdown(pages_text: list[dict], path: str, encoding: str = "utf-8") -> None:
    """Convert to Markdown with tables."""
    lines = []

    def merge_bullet_fragments(blocks: list[dict]) -> list[dict]:
        """Merge wrapped bullets."""
        merged = []
        for b in blocks:
            b = dict(b)
            children = b.get("children", [])
            if children:
                b["children"] = merge_bullet_fragments(children)

            if (
                merged
                and b.get("type") == "bullet"
                and merged[-1].get("type") == "bullet"
                and not merged[-1].get("children")
                and not b.get("children")
            ):
                prev_indent = int(merged[-1].get("indent_level", 0))
                curr_indent = int(b.get("indent_level", 0))
                if abs(prev_indent - curr_indent) <= 1:
                    merged[-1]["text"] = (merged[-1].get("text", "").rstrip() + " " + b.get("text", "").lstrip()).strip()
                    continue

            merged.append(b)
        return merged

    def render_block(block: dict, depth: int = 0):
        btype = block.get("type", "text")
        text = (block.get("text") or "").rstrip()

        if btype == "table":
            rows = block.get("rows", [])
            if rows:
                lines.append("\n" + format_markdown_table(rows) + "\n")
            return

        if btype.startswith("header_"):
            lvl = btype.split("_")[1]
            marker = "#" * (1 if lvl == "h1" else 2 if lvl == "h2" else 3)
            lines.append(f"\n{marker} {text}\n")
            return

        if btype == "bullet":
            indent = min(depth, 3)
            ordered = block.get("ordered", False)
            marker = block.get("marker", "")
            bullet_prefix = f"{marker} " if ordered and marker else "- "
            prefix = "  " * indent + bullet_prefix
            lines.append(f"{prefix}{text}\n")
            for child in block.get("children", []):
                render_block(child, depth + 1)
            return

        lines.append(f"{text}\n")

    for p in pages_text:
        lines.append(f"## Page {p['page']}\n")
        blocks = p.get("blocks_hier") or p.get("blocks", [])
        blocks = merge_bullet_fragments(blocks)
        for b in blocks:
            render_block(b, depth=0)
        lines.append("\n")

    md_content = "\n".join(lines)
    Path(path).write_text(md_content, encoding=encoding)
    logger.info(f"Markdown saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description="PDF extractor with structure recognition")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output-json", "-j", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", "-m", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-md", action="store_true")
    parser.add_argument("--pages", help="Page selection (e.g. '1-3,5')")
    parser.add_argument("--password", help="PDF password")
    parser.add_argument("--detect-tables", action="store_true")
    parser.add_argument("--ocr", dest="use_ocr", action="store_true", default=True)
    parser.add_argument("--no-ocr", dest="use_ocr", action="store_false")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main pipeline."""
    args = parse_args(argv)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return 1

    try:
        page_filter = parse_page_spec(args.pages) if args.pages else None
    except ValueError as exc:
        logger.error(f"Invalid page selection '{args.pages}': {exc}")
        return 3

    global USE_OCR_FOR_IMAGE_PAGES, FORCE_OCR, DETECT_TABLES
    USE_OCR_FOR_IMAGE_PAGES = args.use_ocr
    FORCE_OCR = args.force_ocr
    DETECT_TABLES = args.detect_tables

    print("=" * 70)
    print("PDF Extractor with Tables & Hierarchical Structure")
    print("=" * 70)
    logger.info(f"Extracting PDF: {pdf_path}")

    try:
        logger.info("[1/5] Extracting text blocks...")
        pages_text = extract_pages(str(pdf_path), password=args.password, page_filter=page_filter)
    except PasswordRequiredError as exc:
        logger.error(str(exc))
        return 2
    except PdfOpenError as exc:
        logger.error(str(exc))
        return 3

    total_blocks = sum(len(p["blocks"]) for p in pages_text)
    logger.info(f"Found {len(pages_text)} pages, {total_blocks} blocks")

    logger.info("[2/5] Detecting headers and footers...")
    header_footer_set = detect_headers_footers(pages_text)
    logger.info(f"Detected {len(header_footer_set)} header/footer texts")

    logger.info("[3/5] Filtering irrelevant content...")
    filtered_pages = filter_irrelevant_content(pages_text, header_footer_set)
    filtered_blocks = sum(len(p["blocks"]) for p in filtered_pages)
    removed = total_blocks - filtered_blocks
    logger.info(f"Removed {removed} blocks, {filtered_blocks} remain")

    tables_by_page = {}
    table_pages = filtered_pages

    if DETECT_TABLES:
        slide_like_pages = {p.get("page") for p in filtered_pages if is_slide_like_page(p)}

        # Always try pdfplumber first (even on slide-like pages) to catch clean tables
        if PDFPLUMBER_AVAILABLE:
            logger.info("[4/5] Extracting tables with pdfplumber...")
            tables_by_page = extract_tables_pdfplumber(str(pdf_path), page_filter=page_filter)
            extracted_tables = sum(len(tables_by_page.get(p.get("page"), [])) for p in filtered_pages)
            logger.info(f"pdfplumber extracted {extracted_tables} table(s)")
            if extracted_tables:
                table_pages = inject_tables(filtered_pages, tables_by_page)
        else:
            logger.info("[4/5] pdfplumber not available; skipping structured table extraction")

        # Heuristic detection only on non-slide pages and only if pdfplumber found nothing there
        if not tables_by_page:
            logger.info("[4/5] Detecting tables with column alignment on non-slide pages...")
            table_pages = []
            for p in filtered_pages:
                if p.get("page") in slide_like_pages:
                    table_pages.append(p)
                    continue
                tp = detect_tables([p])[0]
                table_pages.append(tp)
            table_count = sum(len(p.get("tables", [])) for p in table_pages)
            logger.info(f"Detected {table_count} table(s) heuristically")
    else:
        logger.info("[4/5] Skipping table detection (DETECT_TABLES=False)")
        table_pages = filtered_pages

    logger.info("[5/5] Classifying blocks...")
    classified_pages = classify_blocks(table_pages)
    header_count = sum(1 for p in classified_pages for b in p["blocks"] if str(b.get("type", "")).startswith("header_"))
    bullet_count = sum(1 for p in classified_pages for b in p["blocks"] if b.get("type") == "bullet")
    text_count = sum(1 for p in classified_pages for b in p["blocks"] if b.get("type") == "text")
    final_table_count = sum(1 for p in classified_pages for b in p["blocks"] if b.get("type") == "table")
    logger.info(f"Classified: {header_count} headings, {bullet_count} bullets, {text_count} text, {final_table_count} tables")

    if args.no_json and args.no_md:
        logger.info("No outputs requested (--no-json and --no-md set); skipping file writes")
    else:
        logger.info("Saving outputs...")
        if not args.no_json:
            json_path = args.output_json or DEFAULT_OUTPUT_JSON
            save_json(classified_pages, json_path, encoding=args.encoding)
        if not args.no_md:
            md_path = args.output_md or DEFAULT_OUTPUT_MD
            save_markdown(classified_pages, md_path, encoding=args.encoding)

    print("\n" + "=" * 70)
    print("Extraction complete!")
    if not args.no_json:
        print(f"  JSON: {args.output_json or DEFAULT_OUTPUT_JSON}")
    if not args.no_md:
        print(f"  Markdown: {args.output_md or DEFAULT_OUTPUT_MD}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
