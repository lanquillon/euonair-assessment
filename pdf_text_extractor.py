"""
PDF text extractor with header/footer filtering, OCR fallback, tables, and hierarchical bullets.

Extracts structured content (headings, body text, bullets, tables) from PDFs,
removes repeating headers/footers, and OCRs image-heavy pages when needed.
"""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF

# Optional OCR dependencies
try:
    import pytesseract
    from PIL import Image, ImageEnhance
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

PDF_FILE = "sample2.pdf"
OUT_JSON = "extracted_text.json"
OUT_MD = "extracted_text.md"

# Header/footer detection regions (in points; 1 pt ≈ 1/72 inch)
HEADER_HEIGHT = 80  # ~2.8 cm from top
FOOTER_HEIGHT = 80  # ~2.8 cm from bottom

# OCR settings
USE_OCR_FOR_IMAGE_PAGES = True
OCR_MIN_TEXT_LENGTH = 30  # Minimum chars before triggering OCR
OCR_LANG = "eng+deu"  # Tesseract language packs
FORCE_OCR = False  # Prefer native text; set True to force OCR on all pages
DETECT_TABLES = False  # Alignment-based table detection (can over-detect)

# Tesseract path (optional, set via environment variable)
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

# Content detection heuristics
HEADING_REL_THRESHOLD = 1.15  # Font size multiplier for heading detection
INDENT_STEP = 12.0  # Points per indentation level
Y_TOLERANCE = 5  # Pixels for row grouping (tables)
X_TOLERANCE = 30  # Pixels for column alignment (tables)

# Logging
logger = logging.getLogger(__name__)


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text blocks with metadata from all PDF pages."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.error(f"Cannot open PDF {pdf_path}: {exc}")
        return []

    try:
        needs_password = False
        for attr in ("needs_pass", "needs_passwd", "is_encrypted"):
            if hasattr(doc, attr):
                val = getattr(doc, attr)
                needs_password = val() if callable(val) else bool(val)
                break
        if needs_password:
            logger.error(f"PDF {pdf_path} is password-protected")
            return []
    except Exception as exc:
        logger.error(f"Password check failed for {pdf_path}: {exc}")
        return []

    pages_text = []

    for i, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        page_blocks = []

        # Extract text blocks with positioning metadata
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue

                # Calculate bounding box
                x0 = min(s["bbox"][0] for s in spans)
                y0 = min(s["bbox"][1] for s in spans)
                x1 = max(s["bbox"][2] for s in spans)
                y1 = max(s["bbox"][3] for s in spans)

                font_size = max((s.get("size", 0) for s in spans), default=0)
                font_name = spans[0].get("font", "") if spans else ""

                page_blocks.append({
                    "text": text,
                    "bbox": [x0, y0, x1, y1],
                    "x0": x0,
                    "y0": y0,
                    "font_size": float(font_size),
                    "font_name": font_name,
                })

        # OCR fallback for image-heavy pages
        if USE_OCR_FOR_IMAGE_PAGES:
            total_text_len = sum(len(b["text"]) for b in page_blocks)
            should_ocr = FORCE_OCR or total_text_len < OCR_MIN_TEXT_LENGTH

            if should_ocr:
                logger.info(f"Page {i}: Low text content ({total_text_len} chars) - trying OCR")
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
    """OCR a single PDF page and return text blocks."""
    if not OCR_AVAILABLE:
        logger.warning("pytesseract/Pillow not installed - skipping OCR")
        return []

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    try:
        # Render page to image at higher DPI for better OCR accuracy
        pix = page.get_pixmap(dpi=400)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Grayscale and boost contrast to help OCR on diagrams/screenshots
        img = ImageEnhance.Contrast(img.convert("L")).enhance(1.8)
        raw_text = pytesseract.image_to_string(img, lang=OCR_LANG).strip()

    except pytesseract.TesseractNotFoundError:
        logger.error(
            "Tesseract not found. Install it and add to PATH or set TESSERACT_CMD env variable"
        )
        return []
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return []

    if not raw_text:
        return []

    # Split OCR output into pseudo-blocks on blank lines
    ocr_blocks = []
    for chunk in raw_text.split("\n\n"):
        chunk = chunk.strip()
        if chunk:
            ocr_blocks.append({
                "text": chunk,
                "bbox": [0, 0, page.rect.width, page.rect.height],
                "x0": 0,
                "y0": 0,
                "font_size": 12.0,  # Dummy value
                "font_name": "ocr",  # Marker for OCR content
            })

    logger.info(f"OCR recognized {len(ocr_blocks)} text block(s)")
    return ocr_blocks


# =============================================================================
# HEADER/FOOTER DETECTION
# =============================================================================

def detect_headers_footers(pages_text: list[dict]) -> set[str]:
    """Identify repetitive text in header/footer regions across pages."""
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
            logger.info(f"Header detected: '{text}' ({count}/{total_pages} pages)")

    for text, count in footer_texts.items():
        if count > threshold:
            header_footer_set.add(text)
            logger.info(f"Footer detected: '{text}' ({count}/{total_pages} pages)")

    return header_footer_set


def is_page_number(text: str) -> bool:
    """Check if text matches common page number patterns."""
    normalized = text.lower().strip()
    patterns = [
        r"^-?\s*\d+\s*-?$",
        r"^(seite|page|p\.?)\s+\d+$",
        r"^\d+\s+(von|of|/)\s+\d+$",
    ]
    return any(re.match(pattern, normalized) for pattern in patterns)


# =============================================================================
# CONTENT CLASSIFICATION
# =============================================================================

def get_font_size_ranks(pages_text: list[dict]) -> dict[float, int]:
    """Map font sizes to ranks (0=largest, 1=second largest, etc.)."""
    sizes = {
        float(b.get("font_size", 0))
        for p in pages_text
        for b in p.get("blocks", [])
        if b.get("font_size", 0) > 0
    }
    sorted_sizes = sorted(sizes, reverse=True)
    return {size: idx for idx, size in enumerate(sorted_sizes)}


def detect_header_type(
    block: dict,
    size_rank_map: dict[float, int],
    median_size: float,
    page_height: float = None
) -> str | None:
    """Determine if block is a heading (h1/h2/h3) based on font size and position."""
    size = float(block.get("font_size", 0))
    if size == 0:
        return None

    y_top = float(block.get("bbox", [0, 0, 0, 0])[1])
    near_top = False
    if page_height:
        near_top = y_top < max(HEADER_HEIGHT, page_height * 0.12)
    else:
        near_top = y_top < HEADER_HEIGHT

    rank = size_rank_map.get(size)
    if rank == 0:
        return "h1"
    if rank == 1:
        return "h2" if near_top else "h3"
    if rank == 2:
        return "h3"

    fname = (block.get("font_name") or "").lower()
    if median_size and size >= median_size * HEADING_REL_THRESHOLD:
        if "bold" in fname or "bf" in fname:
            return "h2"
        if near_top:
            return "h2"
        return "h3"
    return None


def detect_bullet_type(block: dict, page_min_x0: float | None):
    """Detect bullet/list items based on markers or indentation."""
    text = block.get("text", "").strip()
    x0 = float(block.get("x0", 0))

    # Pattern 1: Symbol bullets
    if m := re.match(r"^\s*([-*•◦o◦▪▸·\u2022>])\s+(.*)$", text):
        block["text"] = m.group(2).strip()
        return True, 0

    # Pattern 2: Numbered bullets
    if m := re.match(r"^\s*(\d+[.)])\s+(.*)$", text):
        block["text"] = m.group(2).strip()
        return True, 0

    # Pattern 3: Lettered bullets
    if m := re.match(r"^\s*([a-zA-Z][.)])\s+(.*)$", text):
        block["text"] = m.group(2).strip()
        return True, 0

    if page_min_x0 is None:
        return False, 0

    rel = max(0.0, x0 - page_min_x0)
    indent_level = int(rel // INDENT_STEP)
    word_count = len(text.split())

    if indent_level > 0 and word_count <= 25:
        return True, indent_level

    return False, 0


def classify_blocks(pages_text: list[dict]) -> list[dict]:
    """Classify blocks as headings, bullets, tables, or text."""
    size_rank_map = get_font_size_ranks(pages_text)

    all_sizes = sorted(
        b.get("font_size", 0)
        for p in pages_text
        for b in p.get("blocks", [])
        if b.get("font_size", 0)
    )
    median_size = all_sizes[len(all_sizes) // 2] if all_sizes else 0

    classified_pages = []
    for page_data in pages_text:
        blocks = page_data.get("blocks", [])
        page_min_x0 = min((b.get("x0", 0) for b in blocks), default=None)

        classified_blocks = []
        idx = 0
        while idx < len(blocks):
            block = blocks[idx]
            text = block.get("text", "")
            block_type = block.get("type", "text")  # Preserve existing type

            # If table detection is disabled, treat tables as plain text
            if not DETECT_TABLES and block_type == "table":
                block_type = "text"

            # Preserve detected tables
            if block_type == "table":
                classified_blocks.append(block)
                idx += 1
                continue

            meta = {}

            # Standalone bullet marker with next-line text
            if re.match(r"^\s*[-*•◦o◦▪▸·\u2022>]\s*$", text):
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
                    block_type = "bullet"
                    meta["indent_level"] = indent
                    entry = {"text": combined_text, "bbox": combined_bbox, "type": block_type}
                    entry.update(meta)
                    classified_blocks.append(entry)
                    idx += 2
                    continue
                else:
                    indent = int(round(float(block.get("x0", 0)) // INDENT_STEP))
                    block_type = "bullet"
                    meta["indent_level"] = indent

            else:
                is_bullet, indent = detect_bullet_type(block, page_min_x0)
                if is_bullet:
                    block_type = "bullet"
                    meta["indent_level"] = indent
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
        for block in page_data["blocks"]:
            text = block["text"]
            normalized = text.lower().strip()

            if normalized in header_footer_set or is_page_number(text):
                continue

            filtered_blocks.append(block)

        removed_count = original_count - len(filtered_blocks)
        if removed_count > 0:
            logger.info(f"  Page {page_data['page']}: removed {removed_count} irrelevant blocks")

        filtered_pages.append({
            "page": page_data["page"],
            "blocks": filtered_blocks
        })

    return filtered_pages


# =============================================================================
# IMPROVED TABLE DETECTION
# =============================================================================

def detect_tables(pages_text: list[dict]) -> list[dict]:
    """
    Detect tables by grouping rows (Y proximity) and aligning columns (X proximity).
    """
    new_pages = []
    for page in pages_text:
        blocks = page.get("blocks", [])
        used = set()
        tables = []

        # Sort blocks by Y position
        sorted_blocks = sorted(enumerate(blocks), key=lambda x: x[1].get("y0", 0))

        # Group into potential rows based on Y proximity
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
            # Compare column alignment between consecutive rows
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

        # Rebuild blocks with tables inserted
        new_blocks = []
        for idx, b in enumerate(blocks):
            if idx in used:
                matching_tables = [t for t in tables if t["start_idx"] == idx]
                if matching_tables:
                    new_blocks.append({
                        "type": "table",
                        "rows": matching_tables[0]["rows"],
                        "bbox": matching_tables[0]["bbox"],
                        "num_cols": matching_tables[0]["num_cols"],
                        "text": ""
                    })
            else:
                new_blocks.append(b)

        new_pages.append({"page": page.get("page"), "blocks": new_blocks, "tables": tables})

    return new_pages


# =============================================================================
# OUTPUT WITH TABLES & HIERARCHY
# =============================================================================

def build_hierarchical_blocks(blocks: list[dict]) -> list[dict]:
    """Group bullets into hierarchy based on indent_level."""
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


def save_json(pages_text: list[dict], path: str) -> None:
    """Save extracted data as JSON with hierarchical bullets."""
    enriched_pages = []
    for page in pages_text:
        hier = build_hierarchical_blocks(page.get("blocks", []))
        enriched_pages.append({
            **page,
            "blocks_hier": hier,
        })

    json_str = json.dumps(enriched_pages, ensure_ascii=False, indent=2)
    Path(path).write_text(json_str, encoding="utf-8")
    logger.info(f"JSON saved: {path}")


def format_markdown_table(rows: list[list[str]]) -> str:
    """Format rows as a Markdown table with column alignment."""
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


def save_markdown(pages_text: list[dict], path: str) -> None:
    """Convert extracted data to Markdown with proper table formatting."""
    lines = []

    def merge_bullet_fragments(blocks: list[dict]) -> list[dict]:
        """Merge consecutive bullet fragments that likely belong to the same line."""
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
                # If indents are close, treat as wrapped line
                if abs(prev_indent - curr_indent) <= 6:
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
            lines.append(f"{marker} {text}\n")
            return

        if btype == "bullet":
            indent = min(depth, 3)
            prefix = "  " * indent + "- "
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
    Path(path).write_text(md_content, encoding="utf-8")
    logger.info(f"Markdown saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main extraction pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("=" * 70)
    print("PDF Extractor with Tables & Hierarchical Structure")
    print("=" * 70)

    if not Path(PDF_FILE).exists():
        logger.error(f"PDF not found: {PDF_FILE}")
        return

    logger.info(f"Extracting PDF: {PDF_FILE}")

    # Step 1: Extract raw text blocks
    logger.info("[1/5] Extracting text blocks...")
    pages_text = extract_pages(PDF_FILE)
    total_blocks = sum(len(p["blocks"]) for p in pages_text)
    logger.info(f"Found {len(pages_text)} pages, {total_blocks} blocks")

    # Step 2: Detect headers/footers
    logger.info("[2/5] Detecting headers and footers...")
    header_footer_set = detect_headers_footers(pages_text)
    logger.info(f"Detected {len(header_footer_set)} header/footer texts")

    # Step 3: Filter irrelevant content
    logger.info("[3/5] Filtering irrelevant content...")
    filtered_pages = filter_irrelevant_content(pages_text, header_footer_set)
    filtered_blocks = sum(len(p["blocks"]) for p in filtered_pages)
    removed = total_blocks - filtered_blocks
    logger.info(f"Removed {removed} blocks, {filtered_blocks} remain")

    # Step 4: Detect tables
    if DETECT_TABLES:
        logger.info("[4/5] Detecting tables with column alignment...")
        table_pages = detect_tables(filtered_pages)
        table_count = sum(len(p.get("tables", [])) for p in table_pages)
        logger.info(f"Detected {table_count} table(s)")
    else:
        logger.info("[4/5] Skipping table detection (DETECT_TABLES=False)")
        table_pages = filtered_pages
        table_count = 0

    # Step 5: Classify content
    logger.info("[5/5] Classifying blocks...")
    classified_pages = classify_blocks(table_pages)
    header_count = sum(
        1 for p in classified_pages
        for b in p["blocks"]
        if str(b.get("type", "")).startswith("header_")
    )
    bullet_count = sum(
        1 for p in classified_pages
        for b in p["blocks"]
        if b.get("type") == "bullet"
    )
    text_count = sum(
        1 for p in classified_pages
        for b in p["blocks"]
        if b.get("type") == "text"
    )
    final_table_count = sum(
        1 for p in classified_pages
        for b in p["blocks"]
        if b.get("type") == "table"
    )
    logger.info(f"Classified: {header_count} headings, {bullet_count} bullets, {text_count} text, {final_table_count} tables")

    # Save outputs
    logger.info("Saving outputs...")
    save_json(classified_pages, OUT_JSON)
    save_markdown(classified_pages, OUT_MD)

    print("\n" + "=" * 70)
    print("Extraction complete!")
    print(f"  JSON: {OUT_JSON}")
    print(f"  Markdown: {OUT_MD}")
    print("=" * 70)


if __name__ == "__main__":
    main()
