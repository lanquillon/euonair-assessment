"""
PDF extractor with header/footer filtering and optional OCR fallback.
Extract headings, body text, bullets, simple tables; drop repeating headers/footers and page numbers; optionally OCR image-heavy pages.
"""

import os
import fitz  # PyMuPDF — PDF library
import json
import re
from pathlib import Path
from collections import defaultdict

# Optional: OCR support (pytesseract + Pillow)
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Optional: explicit path to tesseract executable via environment variable:
#   - Either Tesseract is on PATH (so "tesseract" works in a terminal), OR
#   - Set TESSERACT_CMD to the full path.
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PDF_FILE = "sample2.pdf"
OUT_JSON = "extracted_text.json"
OUT_MD = "extracted_text.md"

# How much of the top/bottom of each page do we treat as header/footer region?
# Values are in points (1 pt ≈ 1/72 inch). 80 pt ≈ 2.8 cm.
HEADER_HEIGHT = 80
FOOTER_HEIGHT = 80

# Optional: OCR fallback for pages with almost no extractable text
USE_OCR_FOR_IMAGE_PAGES = True

# Minimum text length to avoid OCR on image-heavy pages
OCR_MIN_TEXT_LENGTH = 30

# OCR configuration
OCR_LANG = "eng+deu"  # language packs installed in Tesseract
FORCE_OCR = False      # set True to OCR every page regardless of text found

# Heuristic parameters for heading/bullet detection
HEADING_REL_THRESHOLD = 1.15  # how much larger than median to treat as heading
INDENT_STEP = 12.0            # pt per indentation level for nested lists

# ==============================================================================
# FUNCTION 1: Extract raw blocks from a PDF
# ==============================================================================

def extract_pages(pdf_path: str):
    """
    Open the PDF, read each page, collect textual blocks with their positions.

    Returns:
        List of pages, each page is a dict:
        {
          "page": <page_number>,
          "blocks": [
             {
               "text": ...,
               "bbox": [x0, y0, x1, y1],
               "x0": x0,
               "font_size": ...,
               "font_name": ...
             }, ...
          ]
        }
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for i, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        page_blocks = []

        for b in page_dict.get("blocks", []):
            for line in b.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue

                x0 = min(s["bbox"][0] for s in spans)
                y0 = min(s["bbox"][1] for s in spans)
                x1 = max(s["bbox"][2] for s in spans)
                y1 = max(s["bbox"][3] for s in spans)
                bbox = [x0, y0, x1, y1]

                span_sizes = [s.get("size", 0) for s in spans]
                font_size = max(span_sizes) if span_sizes else 0

                font_names = [s.get("font", "") for s in spans]
                font_name = font_names[0] if font_names else ""

                page_blocks.append({
                    "text": text,
                    "bbox": bbox,
                    "x0": x0,
                    "font_size": float(font_size),
                    "font_name": font_name,
                })

        # OCR fallback for pages with very little text (or forced)
        if USE_OCR_FOR_IMAGE_PAGES:
            total_text_len = sum(len(b["text"]) for b in page_blocks)
            should_ocr = FORCE_OCR or total_text_len == 0 or total_text_len < OCR_MIN_TEXT_LENGTH
            if should_ocr:
                reason = "forced" if FORCE_OCR else f"{total_text_len} chars"
                print(f"  [INFO] Page {i}: very little extracted text ({reason}) -> trying OCR...")
                ocr_blocks = ocr_page_simple(page)
                if ocr_blocks:
                    page_blocks = ocr_blocks
                else:
                    print(f"  [INFO] Page {i}: OCR did not return text, keeping original blocks.")

        pages_text.append({"page": i, "blocks": page_blocks})

    return pages_text


def ocr_page_simple(page):
    """
    Simple OCR fallback for a single page.

    Steps:
      - render the PDF page to an image,
      - run Tesseract OCR on the image,
      - return a single text block with the recognized text (or [] on failure).
    """
    if not OCR_AVAILABLE:
        print("  [OCR] pytesseract/Pillow not installed - skipping OCR.")
        return []

    # Configure pytesseract.tesseract_cmd in a robust way,
    # following the typical StackOverflow approach:
    #
    # 1. If TESSERACT_CMD environment variable is set, use that.
    # 2. Otherwise, rely on Tesseract being available on PATH.
    #    (i.e., "tesseract" works in a normal terminal).
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    try:
        # 1) Render page to an image (300 dpi improves OCR accuracy)
        pix = page.get_pixmap(dpi=300)

        # 2) Convert PyMuPDF pixmap to Pillow Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 3) Run OCR (language from config)
        lang = OCR_LANG or "eng"
        raw_text = pytesseract.image_to_string(img, lang=lang).strip()

    except pytesseract.TesseractNotFoundError:
        # This is the exact error discussed in the StackOverflow article:
        # "Tesseract is not installed or it's not in your PATH".
        print(
            "  [OCR] Tesseract not found. Install Tesseract and either add it "
            "to PATH or set the TESSERACT_CMD environment variable to the full "
            "path of the tesseract executable."
        )
        return []
    except Exception as e:
        print(f"  [OCR] Unexpected error during OCR: {e}")
        return []

    if not raw_text:
        print("  [OCR] No text recognized.")
        return []

    # 4) Split OCR output on blank lines to create pseudo-blocks
    ocr_blocks = []
    for chunk in [t.strip() for t in raw_text.split("\n\n") if t.strip()]:
        ocr_blocks.append({
            "text": chunk,
            "bbox": [0, 0, page.rect.width, page.rect.height],
            "x0": 0,
            "font_size": 12.0,    # dummy value; we do not know the real size
            "font_name": "ocr",   # marker: this text came from OCR
        })

    print(f"  [OCR] Text successfully recognized ({len(ocr_blocks)} block(s)).")
    return ocr_blocks


# ==============================================================================
# FUNCTION 2: Detect headers and footers
# ==============================================================================

def detect_headers_footers(pages_text):
    """
    Find text blocks that appear repeatedly near top/bottom positions (likely headers/footers).

    Returns:
        Set of normalized texts that are considered header/footer.
    """
    header_texts = defaultdict(int)
    footer_texts = defaultdict(int)

    for page_data in pages_text:
        blocks = page_data["blocks"]
        if not blocks:
            continue

        max_y = max(b["bbox"][3] for b in blocks)  # bottom of page

        for block in blocks:
            bbox = block["bbox"]
            y_top = bbox[1]
            y_bottom = bbox[3]
            text = block["text"]
            normalized = text.lower().strip()

    # Header region: upper HEADER_HEIGHT pt
            if y_top < HEADER_HEIGHT:
                header_texts[normalized] += 1

    # Footer region: last FOOTER_HEIGHT pt at bottom of the page
            if y_bottom > (max_y - FOOTER_HEIGHT):
                footer_texts[normalized] += 1

    total_pages = len(pages_text)
    threshold = total_pages * 0.5  # appears on > 50 % of pages

    header_footer_set = set()

    for text, count in header_texts.items():
        if count > threshold:
            header_footer_set.add(text)
            print(f"  [HEADER] '{text}' ({count}/{total_pages} pages)")

    for text, count in footer_texts.items():
        if count > threshold:
            header_footer_set.add(text)
            print(f"  [FOOTER] '{text}' ({count}/{total_pages} pages)")

    return header_footer_set


# ==============================================================================
# FUNCTION 3: Detect page numbers (e.g. "1", "Seite 3", "- 5 -")
# ==============================================================================

def is_page_number(text):
    """
    Return True if text likely represents a page number (common patterns).
    """
    normalized = text.lower().strip()

    if re.match(r"^-?\s*\d+\s*-?$", normalized):
        return True

    if re.match(r"^(seite|page|p\.?)\s+\d+$", normalized):
        return True

    if re.match(r"^\d+\s+(von|of|\/)\s+\d+$", normalized):
        return True

    return False


# ==============================================================================
# Helpers for font sizes, headings, bullets
# ==============================================================================

def get_font_size_ranks(pages_text):
    """
    Collect all font sizes and rank them (0 = largest size, 1 = second largest, ...).
    """
    sizes = set()
    for p in pages_text:
        for b in p.get("blocks", []):
            sizes.add(float(b.get("font_size", 0)))
    sizes.discard(0)
    sorted_sizes = sorted(sizes, reverse=True)
    return {s: idx for idx, s in enumerate(sorted_sizes)}


def detect_header_type(block, size_rank_map, median_size, page_height=None):
    """
    Decide if a block is a heading based on font size and position.

    Returns:
        "h1", "h2", "h3" or None
    """
    size = float(block.get("font_size", 0))
    fname = (block.get("font_name") or "").lower()
    if size == 0:
        return None

    # Position: if near the top of the page, more likely a heading
    y_top = float(block.get("bbox", [0, 0, 0, 0])[1])
    near_top = False
    try:
        if page_height:
            if y_top < max(HEADER_HEIGHT, page_height * 0.12):
                near_top = True
        else:
            if y_top < HEADER_HEIGHT:
                near_top = True
    except Exception:
        near_top = False

    rank = size_rank_map.get(size, None)
    if rank == 0:
        return "h1"
    if rank == 1:
        return "h2" if near_top else "h3"
    if rank == 2:
        return "h3"

    # Fallback: compare to median font size
    try:
        if median_size and size >= median_size * HEADING_REL_THRESHOLD:
            if "bold" in fname or "bf" in fname:
                return "h2"
            if near_top:
                return "h2"
            return "h3"
    except Exception:
        pass

    return None


def detect_bullet_type(block, page_min_x0):
    """
    Detect if a block is likely a bullet/list item.

    Strategy:
      - Look for bullet markers at the start ( -, *, •, number., a), ... )
      - Fallback: if the block is indented compared to the left margin and
        relatively short, treat it as a bullet.

    Returns:
      (is_bullet: bool, indent_level: int)
    """
    text = block.get("text", "").strip()
    x0 = float(block.get("x0", 0))

    # Pattern 1: symbol bullets
    m = re.match(r"^\s*([\-\*•○o◦▪▸·\u2022])\s+(.*)$", text)
    if m:
        block["text"] = m.group(2).strip()
        return True, 0

    # Pattern 2: numbered bullets (1. 2. 3.) etc.
    m = re.match(r"^\s*(\d+[\.\)])\s+(.*)$", text)
    if m:
        block["text"] = m.group(2).strip()
        return True, 0

    # Pattern 3: lettered bullets (a) b) ...)
    m = re.match(r"^\s*([a-zA-Z][\.\)])\s+(.*)$", text)
    if m:
        block["text"] = m.group(2).strip()
        return True, 0

    # Fallback: indentation relative to page_min_x0
    if page_min_x0 is None:
        return False, 0

    rel = max(0.0, x0 - page_min_x0)
    indent_level = int(rel // INDENT_STEP)

    word_count = len(text.split())
    if indent_level > 0 and word_count <= 25:
        return True, indent_level

    return False, 0


def classify_blocks(pages_text):
    """
    Main classification step:
      - compute font size ranks
      - for each block, decide if it is:
          * a heading (h1/h2/h3),
          * a bullet (with indent_level),
          * or normal text.
    """
    size_rank_map = get_font_size_ranks(pages_text)

    all_sizes = sorted(
        b.get("font_size", 0)
        for p in pages_text
        for b in p.get("blocks", [])
        if b.get("font_size", 0)
    )
    median_size = 0
    if all_sizes:
        median_size = all_sizes[len(all_sizes) // 2]

    classified_pages = []
    for page_data in pages_text:
        blocks = page_data.get("blocks", [])
        page_min_x0 = min((b.get("x0", 0) for b in blocks), default=None)

        classified_blocks = []
        idx = 0
        while idx < len(blocks):
            block = blocks[idx]
            text = block.get("text", "")
            block_type = "text"
            meta = {}

            # 1) Marker-only bullets ('•' on one line, text on next line)
            if re.match(r"^\s*[\-\*•○o◦▪▸·\u2022]\s*$", text):
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
                    hdr = detect_header_type(block, size_rank_map, median_size)
                    if hdr:
                        block_type = f"header_{hdr}"
                        meta["heading_level"] = hdr

            entry = {"text": text, "bbox": block.get("bbox"), "type": block_type}
            entry.update(meta)
            classified_blocks.append(entry)
            idx += 1

        classified_pages.append({"page": page_data.get("page"), "blocks": classified_blocks})

    return classified_pages


# ==============================================================================
# FUNCTION 4: Filter out header/footer and page numbers
# ==============================================================================

def filter_irrelevant_content(pages_text, header_footer_set):
    """
    # Remove blocks that are:
    #   1. Recognized headers/footers
    #   2. Page numbers
    """
    filtered_pages = []

    for page_data in pages_text:
        original_count = len(page_data["blocks"])

        filtered_blocks = []
        for block in page_data["blocks"]:
            text = block["text"]
            normalized = text.lower().strip()

            if normalized in header_footer_set:
                continue

            if is_page_number(text):
                continue

            filtered_blocks.append(block)

        removed_count = original_count - len(filtered_blocks)
        if removed_count > 0:
            print(f"  Page {page_data['page']}: removed {removed_count} irrelevant blocks")

        filtered_pages.append({
            "page": page_data["page"],
            "blocks": filtered_blocks
        })

    return filtered_pages


# ==============================================================================
# Simple table detection (heuristic)
# ==============================================================================

def detect_tables(pages_text):
    """
    Heuristic to detect simple text tables:
      - several consecutive lines with multiple "columns" separated by
        multiple spaces, tabs, or '|'.
    """
    new_pages = []
    for page in pages_text:
        blocks = page.get("blocks", [])
        used = set()
        tables = []

        candidates = []
        for idx, b in enumerate(blocks):
            text = b.get("text", "")
            if re.search(r"\S\s{2,}\S", text) or "|" in text or "\t" in text:
                candidates.append((idx, b))

        i = 0
        while i < len(candidates):
            start_idx, start_block = candidates[i]
            rows = [start_block]
            j = i + 1
            while j < len(candidates) and candidates[j][0] == candidates[j-1][0] + 1:
                rows.append(candidates[j][1])
                j += 1

            if len(rows) >= 2:
                table_rows = []
                for r in rows:
                    cells = re.split(r"\s{2,}|\t|\|", r.get("text", ""))
                    cells = [c.strip() for c in cells if c.strip()]
                    table_rows.append(cells)

                x0 = min(r.get("bbox")[0] for r in rows)
                y0 = min(r.get("bbox")[1] for r in rows)
                x1 = max(r.get("bbox")[2] for r in rows)
                y1 = max(r.get("bbox")[3] for r in rows)

                tables.append({"rows": table_rows, "bbox": [x0, y0, x1, y1], "start_idx": start_idx})
                for k in range(start_idx, start_idx + len(rows)):
                    used.add(k)

            i = j

        new_blocks = []
        for idx, b in enumerate(blocks):
            if idx in used:
                starts = [t for t in tables if t["start_idx"] == idx]
                if starts:
                    t = starts[0]
                    new_blocks.append({"type": "table", "rows": t["rows"], "bbox": t["bbox"], "text": ""})
                continue
            else:
                new_blocks.append(b)

        new_pages.append({"page": page.get("page"), "blocks": new_blocks, "tables": tables})

    return new_pages


# ==============================================================================
# FUNCTION 5: Save as JSON
# ==============================================================================

def save_json(pages_text, path: str):
    """
    Save the page data as JSON (machine-readable intermediate format).
    """
    json_str = json.dumps(pages_text, ensure_ascii=False, indent=2)
    Path(path).write_text(json_str, encoding="utf-8")
    print(f"JSON saved: {path}")


# ==============================================================================
# FUNCTION 6: Save as Markdown
# ==============================================================================

def save_markdown(pages_text, path: str):
    """
    Write the page data as a human-readable Markdown file.

    Idea:
      - Each page starts with "## Seite X".
      - Headings use # / ## / ###.
      - Bullet blocks:
          * Consecutive bullet blocks with the same indent_level
            are treated as ONE list item.
          * Inside that list item we insert explicit line breaks,
            so the Markdown looks like:
                - First line
                  Second line
                  Third line
      - Normal text is written as paragraphs.
    """
    lines = []

    for p in pages_text:
        page_num = p["page"]
        blocks = p["blocks"]

        # Page heading
        lines.append(f"## Seite {page_num}\n")

        current_list_indent = 0  # current nesting level for lists

        idx = 0
        while idx < len(blocks):
            b = blocks[idx]
            text = (b.get("text") or "").rstrip()
            btype = b.get("type", "text")

            # ---------------------------------------------------------
            # HEADINGS
            # ---------------------------------------------------------
            if btype.startswith("header_"):
                # close any open list with a blank line
                if current_list_indent > 0:
                    lines.append("\n")
                    current_list_indent = 0

                lvl = btype.split("_")[1]
                if lvl == "h1":
                    lines.append(f"# {text}\n")
                elif lvl == "h2":
                    lines.append(f"## {text}\n")
                elif lvl == "h3":
                    lines.append(f"### {text}\n")
                else:
                    lines.append(f"## {text}\n")

                idx += 1
                continue

            # ---------------------------------------------------------
            # BULLETS
            # ---------------------------------------------------------
            if btype == "bullet":
                indent = int(b.get("indent_level", 0))

                # Limit nesting to at most one sub-level for Markdown.
                # Deeper levels often create ugly rendering.
                if indent > 1:
                    indent = 1

                # Collect all consecutive bullet blocks with the same indent
                # and treat them as one logical list item.
                bullet_texts = [text]
                j = idx + 1
                while j < len(blocks):
                    nb = blocks[j]
                    nb_type = nb.get("type", "text")
                    if nb_type != "bullet":
                        break

                    n_indent = int(nb.get("indent_level", 0))
                    if n_indent > 1:
                        n_indent = 1

                    if n_indent != indent:
                        break

                    bullet_texts.append((nb.get("text") or "").rstrip())
                    j += 1

                # Join the bullet fragments with explicit Markdown line breaks.
                # "  \n  " means:
                #   - two spaces + newline = hard line break
                #   - next line starts with two spaces => visually under the bullet.
                inner = "  \n  ".join(t for t in bullet_texts if t)

                prefix = "  " * indent + "- "
                lines.append(f"{prefix}{inner}\n")

                current_list_indent = indent
                idx = j
                continue

            # ---------------------------------------------------------
            # NORMAL TEXT
            # ---------------------------------------------------------
            # If we were inside a list, add a blank line to visually separate.
            if current_list_indent > 0:
                lines.append("\n")
                current_list_indent = 0

            lines.append(f"{text}\n")
            idx += 1

        # Blank line between pages
        lines.append("\n")

    md_content = "\n".join(lines)
    Path(path).write_text(md_content, encoding="utf-8")
    print(f"Markdown gespeichert: {path}")

# def save_markdown(pages_text, path: str):
#     """
#     Save the page data as a human-readable Markdown file.
#     """
#     lines = []

#     for p in pages_text:
#         page_num = p["page"]
#         blocks = p["blocks"]

#         lines.append(f"## Seite {page_num}\n")

#         current_list_indent = 0

#         for b in blocks:
#             text = b.get("text", "").rstrip()
#             btype = b.get("type", "text")

#             if btype.startswith("header_"):
#                 if current_list_indent > 0:
#                     lines.append("\n")
#                     current_list_indent = 0

#                 lvl = btype.split("_")[1]
#                 if lvl == "h1":
#                     lines.append(f"# {text}\n")
#                 elif lvl == "h2":
#                     lines.append(f"## {text}\n")
#                 elif lvl == "h3":
#                     lines.append(f"### {text}\n")
#                 else:
#                     lines.append(f"## {text}\n")

#             elif btype == "bullet":
#                 indent = int(b.get("indent_level", 0))
#                 prefix = "  " * indent + "- "
#                 lines.append(f"{prefix}{text}\n")
#                 current_list_indent = indent

#             else:
#                 if current_list_indent > 0:
#                     lines.append("\n")
#                     current_list_indent = 0
#                 lines.append(f"{text}\n")

#         lines.append("\n")

#     md_content = "\n".join(lines)
#     Path(path).write_text(md_content, encoding="utf-8")
#     print(f"Markdown saved: {path}")


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PDF-Extractor with header/footer filtering and optional OCR")
    print("=" * 70)

    if not Path(PDF_FILE).exists():
        print(f"PDF not found: {PDF_FILE}")
    else:
        print(f"\nExtracting PDF: {PDF_FILE}\n")

        print("[1/4] Extracting text blocks...")
        pages_text = extract_pages(PDF_FILE)
        total_blocks = sum(len(p["blocks"]) for p in pages_text)
        print(f"      → {len(pages_text)} pages, {total_blocks} blocks found\n")

        print("[2/4] Detecting headers and footers...")
        header_footer_set = detect_headers_footers(pages_text)
        print(f"      → {len(header_footer_set)} header/footer texts detected\n")

        print("[3/4] Filtering irrelevant content (headers/footers/page numbers)...")
        filtered_pages = filter_irrelevant_content(pages_text, header_footer_set)
        filtered_blocks = sum(len(p["blocks"]) for p in filtered_pages)
        removed = total_blocks - filtered_blocks
        print(f"      → {removed} blocks removed, {filtered_blocks} remain\n")

        print("[3b] Detecting simple tables (heuristic)...")
        table_pages = detect_tables(filtered_pages)
        print(f"      → tables checked (pages: {len(table_pages)})\n")

        print("[4/4] Classifying blocks (headings, bullet points, text)...")
        classified_pages = classify_blocks(table_pages)
        header_count = sum(1 for p in classified_pages for b in p["blocks"] if str(b.get("type", "")).startswith("header_"))
        bullet_count = sum(1 for p in classified_pages for b in p["blocks"] if b.get("type") == "bullet")
        text_count = sum(1 for p in classified_pages for b in p["blocks"] if b.get("type") == "text")
        print(f"      -> {header_count} headings, {bullet_count} bullets, {text_count} text blocks\n")

        print("[SAVE] Writing JSON and Markdown outputs...")
        save_json(classified_pages, OUT_JSON)
        save_markdown(classified_pages, OUT_MD)

        print("\n" + "=" * 70)
        print("Done! Results:")
        print(f"  JSON: {OUT_JSON}")
        print(f"  Markdown: {OUT_MD}")
        print("=" * 70)
