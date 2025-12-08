"""PDF extractor with header/footer removal, optional OCR/table detection, and
structured JSON/Markdown outputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import fitz  # type: ignore

try:
    import pytesseract
    from PIL import Image, ImageEnhance

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


DEFAULT_OUTPUT_DIR = Path("all_output") / "extracted_text_output"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "extracted_text.json"
DEFAULT_OUTPUT_MD = DEFAULT_OUTPUT_DIR / "extracted_text.md"


@dataclass
class ExtractConfig:  # pylint: disable=too-many-instance-attributes
    """User-supplied extraction settings."""

    pdf_path: Path
    output_json: Path = DEFAULT_OUTPUT_JSON
    output_md: Path = DEFAULT_OUTPUT_MD
    password: Optional[str] = None
    pages: Optional[Set[int]] = None
    detect_tables: bool = False
    use_ocr: bool = True
    force_ocr: bool = False
    encoding: str = "utf-8"
    header_height: float = 80.0
    footer_height: float = 80.0
    heading_rel_threshold: float = 1.15
    indent_step: float = 12.0
    y_tolerance: float = 5.0
    x_tolerance: float = 30.0
    ocr_min_text_len: int = 30
    ocr_lang: str = "eng+deu"
    tesseract_cmd: Optional[str] = os.getenv("TESSERACT_CMD")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class PdfOpenError(RuntimeError):
    """PDF cannot be opened."""


class PasswordRequiredError(RuntimeError):
    """PDF requires a password."""


class ExtractionError(RuntimeError):
    """Extraction failed."""


# --------------------------------------------------------------------------- #
# PDF helpers
# --------------------------------------------------------------------------- #


def parse_page_spec(page_spec: Optional[str]) -> Optional[Set[int]]:
    """Parse '1-3,5' into a set of 1-based page numbers."""
    if not page_spec:
        return None
    pages: Set[int] = set()
    for part in page_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return {p for p in pages if p > 0}


def _requires_password(doc: Any) -> bool:
    """Return True if the document needs a password."""
    for attr in ("needs_pass", "needs_passwd", "is_encrypted"):
        if hasattr(doc, attr):
            val = getattr(doc, attr)
            try:
                return val() if callable(val) else bool(val)
            except (RuntimeError, TypeError, ValueError):
                return False
    return False


def open_pdf(cfg: ExtractConfig) -> Any:
    """Open a PDF, authenticating if needed."""
    try:
        doc = fitz.open(cfg.pdf_path)
    except (RuntimeError, OSError) as exc:
        msg = f"Cannot open PDF {cfg.pdf_path}: {exc}"
        raise PdfOpenError(msg) from exc

    if _requires_password(doc):
        if not cfg.password:
            msg = (
                f"PDF {cfg.pdf_path} is password-protected "
                "(password required)"
            )
            raise PasswordRequiredError(msg)
        auth = getattr(doc, "authenticate", None)
        if callable(auth):
            authed = auth(cfg.password)
            if not authed and _requires_password(doc):
                msg = f"Invalid password for PDF {cfg.pdf_path}"
                raise PasswordRequiredError(msg)
        elif _requires_password(doc):
            msg = f"Cannot unlock PDF {cfg.pdf_path} (password needed)"
            raise PasswordRequiredError(msg)
    return doc


# --------------------------------------------------------------------------- #
# Extraction (text + OCR)
# --------------------------------------------------------------------------- #


def _ocr_page(cfg: ExtractConfig, page: Any) -> List[Dict[str, Any]]:
    """OCR a single page into pseudo-blocks."""
    if not OCR_AVAILABLE:
        logging.warning("pytesseract/Pillow not installed - skipping OCR")
        return []
    if cfg.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd
    try:
        pix = page.get_pixmap(dpi=400)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = ImageEnhance.Contrast(img.convert("L")).enhance(1.8)
        raw = pytesseract.image_to_string(img, lang=cfg.ocr_lang).strip()
    except pytesseract.TesseractNotFoundError as exc:
        logging.error("Tesseract not found: %s", exc)
        return []
    except RuntimeError as exc:
        logging.error("OCR error: %s", exc)
        return []
    if not raw:
        return []

    blocks: List[Dict[str, Any]] = []
    for chunk in raw.split("\n\n"):
        text = chunk.strip()
        if text:
            blocks.append(
                {
                    "text": text,
                    "bbox": [0, 0, page.rect.width, page.rect.height],
                    "x0": 0,
                    "y0": 0,
                    "font_size": 12.0,
                    "font_name": "ocr",
                }
            )
    return blocks


# pylint: disable=too-many-locals,too-many-branches,too-many-nested-blocks
def extract_pages(
    cfg: ExtractConfig,
) -> List[Dict[str, Any]]:
    """Extract text blocks and apply OCR fallback."""
    doc = open_pdf(cfg)
    pages: List[Dict[str, Any]] = []
    try:
        for idx, page in enumerate(doc, start=1):
            if cfg.pages and idx not in cfg.pages:
                continue

            page_dict = page.get_text("dict")
            blocks: List[Dict[str, Any]] = []
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    # Merge all spans on the line to preserve font/bbox info.
                    text_parts = (span.get("text", "") for span in spans)
                    text = "".join(text_parts).strip()
                    if not text:
                        continue
                    x0 = min(span["bbox"][0] for span in spans)
                    y0 = min(span["bbox"][1] for span in spans)
                    x1 = max(span["bbox"][2] for span in spans)
                    y1 = max(span["bbox"][3] for span in spans)
                    blocks.append(
                        {
                            "text": text,
                            "bbox": [x0, y0, x1, y1],
                            "x0": x0,
                            "y0": y0,
                            "font_size": float(
                                max(
                                    (span.get("size", 0) for span in spans),
                                    default=0,
                                )
                            ),
                            "font_name": (
                                spans[0].get("font", "") if spans else ""
                            ),
                        }
                    )

            if cfg.use_ocr:
                # OCR only when text is scarce or explicitly forced.
                total_len = sum(len(block["text"]) for block in blocks)
                if cfg.force_ocr or total_len < cfg.ocr_min_text_len:
                    logging.info(
                        "Page %s low text (%s) -> OCR",
                        idx,
                        total_len,
                    )
                    ocr_blocks = _ocr_page(cfg, page)
                    if ocr_blocks:
                        if total_len > 0 and not cfg.force_ocr:
                            blocks.extend(ocr_blocks)
                        else:
                            blocks = ocr_blocks

            pages.append(
                {
                    "page": idx,
                    "page_height": float(page.rect.height),
                    "page_width": float(page.rect.width),
                    "blocks": blocks,
                }
            )
        return pages
    finally:
        with suppress(Exception):
            doc.close()


# --------------------------------------------------------------------------- #
# Header/footer detection and filtering
# --------------------------------------------------------------------------- #


def detect_headers_footers(
    cfg: ExtractConfig, pages: List[Dict[str, Any]]
) -> Set[str]:
    """Find repeating header/footer texts across pages."""
    headers: Dict[str, int] = defaultdict(int)
    footers: Dict[str, int] = defaultdict(int)
    for page in pages:
        blocks = page.get("blocks", [])
        if not blocks:
            continue
        max_y = max(block["bbox"][3] for block in blocks)
        for block in blocks:
            text_norm = block["text"].lower().strip()
            y0, y1 = block["bbox"][1], block["bbox"][3]
            if y0 < cfg.header_height:
                headers[text_norm] += 1
            if y1 > max_y - cfg.footer_height:
                footers[text_norm] += 1

    total = max(len(pages), 1)
    threshold = total * 0.5
    found = {t for t, count in headers.items() if count > threshold}
    found |= {t for t, count in footers.items() if count > threshold}
    for text in found:
        logging.info("Header/footer detected: '%s'", text)
    return found


def is_page_number(text: str) -> bool:
    """Return True if text looks like a page number."""
    norm = text.lower().strip()
    patterns = [
        r"^-?\s*\d+\s*-?$",
        r"^(seite|page|p\.?)\s+\d+$",
        r"^\d+\s+(von|of|/)\s+\d+$",
    ]
    return any(re.match(pattern, norm) for pattern in patterns)


def filter_irrelevant(
    cfg: ExtractConfig, pages: List[Dict[str, Any]], header_footer: Set[str]
) -> List[Dict[str, Any]]:
    """Remove detected headers/footers and page numbers (also by position)."""
    targets = {text.lower().strip() for text in header_footer}
    cleaned: List[Dict[str, Any]] = []
    for page in pages:
        page_height = float(page.get("page_height", 0) or 0)
        kept = []
        for block in page.get("blocks", []):
            text = block.get("text", "")
            norm = text.lower().strip()
            bbox = block.get("bbox", [0, 0, 0, 0])
            y0, y1 = bbox[1], bbox[3]
            near_header = page_height and y0 < cfg.header_height
            near_footer = page_height and y1 > (
                page_height - cfg.footer_height
            )
            if norm in targets or is_page_number(text):
                continue
            # Drop blocks located in header/footer bands even if text varies.
            if near_header or near_footer:
                continue
            kept.append(block)
        cleaned.append({**page, "blocks": kept})
    return cleaned


# --------------------------------------------------------------------------- #
# Classification (headings, bullets, tables)
# --------------------------------------------------------------------------- #


def get_font_size_ranks(pages: List[Dict[str, Any]]) -> Dict[float, int]:
    """Map font sizes to rank (0 = largest)."""
    sizes = {
        float(block.get("font_size", 0))
        for page in pages
        for block in page.get("blocks", [])
        if block.get("font_size", 0) > 0
    }
    sorted_sizes = sorted(sizes, reverse=True)
    return {size: idx for idx, size in enumerate(sorted_sizes)}


# pylint: disable=too-many-return-statements
def detect_header_type(
    cfg: ExtractConfig,
    block: Dict[str, Any],
    size_rank: Dict[float, int],
    median_size: float,
    page_height: float,
) -> Optional[str]:
    """Decide if a block is a header h1/h2/h3."""
    size = float(block.get("font_size", 0))
    if size == 0:
        return None
    y_top = float(block.get("bbox", [0, 0, 0, 0])[1])
    near_top = y_top < max(cfg.header_height, page_height * 0.12)
    rank = size_rank.get(size)
    if rank == 0:
        return "h1"
    if rank == 1:
        return "h2" if near_top else "h3"
    if rank == 2:
        return "h3"
    fname = (block.get("font_name") or "").lower()
    if median_size and size >= median_size * cfg.heading_rel_threshold:
        if "bold" in fname or "bf" in fname:
            return "h2"
        if near_top:
            return "h2"
        return "h3"
    return None


def detect_bullet(
    cfg: ExtractConfig, block: Dict[str, Any], page_min_x0: Optional[float]
) -> Tuple[bool, int, str]:
    """Detect list items by marker or indentation."""
    text = block.get("text", "").strip()
    x0 = float(block.get("x0", 0))

    bullet_patterns = [
        r"^\s*([-\u2022*])\s+(.*)$",
        r"^\s*(\d+[.)])\s+(.*)$",
        r"^\s*([a-zA-Z][.)])\s+(.*)$",
    ]
    for pattern in bullet_patterns:
        match = re.match(pattern, text)
        if match:
            return True, 0, match.group(2).strip()

    if page_min_x0 is None:
        return False, 0, text

    rel = max(0.0, x0 - page_min_x0)
    indent = int(rel // cfg.indent_step)
    if indent > 0 and len(text.split()) <= 25:
        return True, indent, text
    return False, 0, text


# pylint: disable=too-many-locals
def classify(
    cfg: ExtractConfig, pages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Classify blocks into headings, bullets, tables, or text."""
    size_rank = get_font_size_ranks(pages)
    all_sizes = [
        block.get("font_size", 0)
        for page in pages
        for block in page.get("blocks", [])
        if block.get("font_size", 0)
    ]
    median_size = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 0

    classified: List[Dict[str, Any]] = []
    for page in pages:
        blocks = page.get("blocks", [])
        page_min_x0 = min(
            (block.get("x0", 0) for block in blocks),
            default=None,
        )
        page_height = float(page.get("page_height", 0) or 0)
        new_blocks: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(blocks):
            block = blocks[idx]
            btype = block.get("type", "text")
            text = block.get("text", "")
            meta: Dict[str, Any] = {}

            if not cfg.detect_tables and btype == "table":
                btype = "text"

            if btype == "table":
                new_blocks.append(block)
                idx += 1
                continue

            if re.match(r"^\s*[-\u2022*]\s*$", text):
                if idx + 1 < len(blocks):
                    next_block = blocks[idx + 1]
                    combined_text = next_block.get("text", "").strip()
                    bbox_a = block.get("bbox", [0, 0, 0, 0])
                    bbox_b = next_block.get("bbox", [0, 0, 0, 0])
                    combined_bbox = [
                        min(bbox_a[0], bbox_b[0]),
                        min(bbox_a[1], bbox_b[1]),
                        max(bbox_a[2], bbox_b[2]),
                        max(bbox_a[3], bbox_b[3]),
                    ]
                    indent_val = int(
                        round(
                            float(next_block.get("x0", 0))
                            // cfg.indent_step
                        )
                    )
                    # Combine marker + next block into a single bullet entry.
                    new_blocks.append(
                        {
                            "text": combined_text,
                            "bbox": combined_bbox,
                            "type": "bullet",
                            "indent_level": indent_val,
                        }
                    )
                    idx += 2
                    continue
                indent_val = int(
                    round(float(block.get("x0", 0)) // cfg.indent_step)
                )
                new_blocks.append(
                    {
                        "text": text,
                        "bbox": block.get("bbox"),
                        "type": "bullet",
                        "indent_level": indent_val,
                    }
                )
                idx += 1
                continue

            is_bullet, indent, cleaned_text = detect_bullet(
                cfg, block, page_min_x0
            )
            if is_bullet:
                btype = "bullet"
                meta["indent_level"] = indent
                text = cleaned_text
            else:
                hdr = detect_header_type(
                    cfg, block, size_rank, median_size, page_height
                )
                if hdr:
                    btype = f"header_{hdr}"
                    meta["heading_level"] = hdr

            entry = {"text": text, "bbox": block.get("bbox"), "type": btype}
            entry.update(meta)
            new_blocks.append(entry)
            idx += 1

        classified.append({**page, "blocks": new_blocks})
    return classified


# --------------------------------------------------------------------------- #
# Table detection
# --------------------------------------------------------------------------- #


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def detect_tables(
    cfg: ExtractConfig, pages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Detect tables by aligned rows/columns."""
    if not cfg.detect_tables:
        return pages
    result: List[Dict[str, Any]] = []
    for page in pages:
        blocks = page.get("blocks", [])
        used: Set[int] = set()
        tables: List[Dict[str, Any]] = []
        sorted_blocks = sorted(
            enumerate(blocks), key=lambda item: item[1].get("y0", 0)
        )
        # Group blocks into horizontal rows by Y proximity.
        rows: List[List[Tuple[int, Dict[str, Any]]]] = []
        current: List[Tuple[int, Dict[str, Any]]] = []
        current_y: Optional[float] = None
        for idx, block in sorted_blocks:
            y_val = block.get("y0", 0)
            if current_y is None or abs(y_val - current_y) <= cfg.y_tolerance:
                current.append((idx, block))
                current_y = y_val if current_y is None else current_y
            else:
                if len(current) >= 2:
                    rows.append(current)
                current = [(idx, block)]
                current_y = y_val
        if len(current) >= 2:
            rows.append(current)

        i = 0
        while i < len(rows):
            base = rows[i]
            base_cols = sorted(block[1].get("x0", 0) for block in base)
            table_rows = [base]
            j = i + 1
            while j < len(rows):
                curr_cols = sorted(block[1].get("x0", 0) for block in rows[j])
                # Require same column count; keep columns within x tolerance.
                aligned = len(curr_cols) == len(base_cols) and all(
                    abs(col_a - col_b) <= cfg.x_tolerance
                    for col_a, col_b in zip(base_cols, curr_cols)
                )
                if aligned:
                    table_rows.append(rows[j])
                    j += 1
                    continue
                break
            if len(table_rows) >= 2:
                all_blocks = [cell[1] for row in table_rows for cell in row]
                table_data: List[List[str]] = []
                indices: Set[int] = set()
                for row in table_rows:
                    sorted_cells = sorted(
                        row, key=lambda item: item[1].get("x0", 0)
                    )
                    table_data.append(
                        [
                            cell[1].get("text", "").strip()
                            for cell in sorted_cells
                        ]
                    )
                    indices.update(cell[0] for cell in sorted_cells)
                # Compute bounding box that spans all table cells.
                x0 = min(
                    block.get("bbox", [0, 0, 0, 0])[0] for block in all_blocks
                )
                y0 = min(
                    block.get("bbox", [0, 0, 0, 0])[1] for block in all_blocks
                )
                x1 = max(
                    block.get("bbox", [0, 0, 0, 0])[2] for block in all_blocks
                )
                y1 = max(
                    block.get("bbox", [0, 0, 0, 0])[3] for block in all_blocks
                )
                tables.append(
                    {
                        "rows": table_data,
                        "bbox": [x0, y0, x1, y1],
                        "start_idx": min(indices),
                    }
                )
                used.update(indices)
            i = j

        new_blocks: List[Dict[str, Any]] = []
        for idx, block in enumerate(blocks):
            if idx in used:
                match = [
                    table for table in tables if table["start_idx"] == idx
                ]
                if match:
                    table_match = match[0]
                    new_blocks.append(
                        {
                            "type": "table",
                            "rows": table_match["rows"],
                            "bbox": table_match["bbox"],
                            "text": "",
                        }
                    )
            else:
                new_blocks.append(block)
        result.append({**page, "blocks": new_blocks, "tables": tables})
    return result


# --------------------------------------------------------------------------- #
# Hierarchy and outputs
# --------------------------------------------------------------------------- #


def build_bullet_hierarchy(
    blocks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Nest bullet items by indentation."""
    out: List[Dict[str, Any]] = []
    stack: List[Tuple[int, Dict[str, Any]]] = []
    for block in blocks:
        node = dict(block)
        node["children"] = []
        if block.get("type") != "bullet":
            stack.clear()
            out.append(node)
            continue
        indent = int(block.get("indent_level", 0))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            out.append(node)
        stack.append((indent, node))
    return out


def build_document_structure(
    pages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build section tree using detected headers."""
    level_map = {"header_h1": 1, "header_h2": 2, "header_h3": 3}
    root = {"title": None, "level": 0, "content": [], "children": []}
    stack: List[Dict[str, Any]] = [root]
    for page in pages:
        blocks = page.get("blocks_hier") or page.get("blocks") or []
        for block in blocks:
            block_copy = dict(block)
            block_copy["page"] = page.get("page")
            level = level_map.get(block_copy.get("type", ""))
            if level:
                while stack and stack[-1]["level"] >= level:
                    stack.pop()
                section = {
                    "title": block_copy.get("text", "").strip(),
                    "level": level,
                    "page": page.get("page"),
                    "content": [],
                    "children": [],
                }
                stack[-1]["children"].append(section)
                stack.append(section)
            else:
                stack[-1]["content"].append(block_copy)
    return root["children"]


def save_json(pages: List[Dict[str, Any]], path: Path, encoding: str) -> None:
    """Write structured JSON (pages + document sections)."""
    enriched: List[Dict[str, Any]] = []
    for page in pages:
        hier = build_bullet_hierarchy(page.get("blocks", []))
        enriched.append({**page, "blocks_hier": hier})
    doc_sections = build_document_structure(enriched)
    payload = {"pages": enriched, "document": {"sections": doc_sections}}
    path.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding=encoding)
    logging.info("JSON saved: %s", path)


def format_markdown_table(rows: List[List[str]]) -> str:
    """Render a Markdown table from a list of rows."""
    if not rows or not rows[0]:
        return ""
    cols = max(len(row) for row in rows)
    normalized = [row + [""] * (cols - len(row)) for row in rows]
    widths = [0] * cols
    for row in normalized:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    widths = [max(3, width) for width in widths]
    header_line = "| " + " | ".join(
        str(cell).ljust(widths[idx]) for idx, cell in enumerate(normalized[0])
    ) + " |"
    sep_line = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    data_lines = []
    for row in normalized[1:]:
        cells = " | ".join(
            str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)
        )
        data_lines.append(f"| {cells} |")
    return "\n".join([header_line, sep_line, *data_lines])


def save_markdown(
    pages: List[Dict[str, Any]], path: Path, encoding: str
) -> None:
    """Render Markdown and append a per-page view for page context."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    def merge_fragments(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for block in blocks:
            item = dict(block)
            if item.get("children"):
                item["children"] = merge_fragments(item.get("children", []))
            if (
                merged
                and item.get("type") == "bullet"
                and merged[-1].get("type") == "bullet"
                and not merged[-1].get("children")
                and not item.get("children")
            ):
                prev = int(merged[-1].get("indent_level", 0))
                curr = int(item.get("indent_level", 0))
                if abs(prev - curr) <= 6:
                    merged[-1]["text"] = (
                        merged[-1].get("text", "").rstrip()
                        + " "
                        + item.get("text", "").lstrip()
                    ).strip()
                    continue
            merged.append(item)
        return merged

    def render_block(block: Dict[str, Any], depth: int = 0) -> None:
        btype = block.get("type", "text")
        text = (block.get("text") or "").rstrip()
        if btype == "table":
            rows = block.get("rows", [])
            if rows:
                lines.append("\n" + format_markdown_table(rows) + "\n")
            return
        if btype == "bullet":
            prefix = "  " * min(depth, 3) + "- "
            lines.append(f"{prefix}{text}\n")
            for child in block.get("children", []):
                render_block(child, depth + 1)
            return
        lines.append(f"{text}\n")

    def render_section(section: Dict[str, Any]) -> None:
        title = section.get("title", "")
        level = section.get("level", 1)
        lines.append(f"{'#' * min(level, 6)} {title}\n")
        for item in merge_fragments(section.get("content", [])):
            render_block(item, depth=0)
        for child in section.get("children", []):
            render_section(child)

    doc_sections = build_document_structure(pages)
    if doc_sections:
        # Render structured view
        for section in doc_sections:
            render_section(section)

        # Always provide a per-page view so readers see original page breaks.
        lines.append("\n---\n## Per-page view\n")

    for page in pages:
        lines.append(f"### Page {page.get('page')}\n")
        blocks = merge_fragments(
            page.get("blocks_hier") or page.get("blocks", [])
        )
        for block in blocks:
            render_block(block, depth=0)
        lines.append("\n")

    path.write_text("\n".join(lines), encoding=encoding)
    logging.info("Markdown saved: %s", path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "PDF extractor with OCR, header/footer filtering, optional table "
            "detection, and structured outputs."
        ),
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output-json",
        "-j",
        default=DEFAULT_OUTPUT_JSON,
        help=f"JSON output path (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--output-md",
        "-m",
        default=DEFAULT_OUTPUT_MD,
        help=f"Markdown output path (default: {DEFAULT_OUTPUT_MD})",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing JSON output",
    )
    parser.add_argument(
        "--no-md",
        action="store_true",
        help="Skip writing Markdown output",
    )
    parser.add_argument(
        "--pages",
        help="Page selection, e.g. '1-3,5' (1-based)",
    )
    parser.add_argument("--password", help="Password for encrypted PDFs")
    parser.add_argument(
        "--detect-tables",
        action="store_true",
        help="Enable table detection",
    )
    parser.add_argument(
        "--ocr",
        dest="use_ocr",
        action="store_true",
        default=True,
        help="Enable OCR fallback on image-heavy pages (default)",
    )
    parser.add_argument(
        "--no-ocr",
        dest="use_ocr",
        action="store_false",
        help="Disable OCR fallback",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on all pages",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding for output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExtractConfig:
    """Convert CLI args to ExtractConfig."""
    return ExtractConfig(
        pdf_path=Path(args.pdf_path),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        password=args.password,
        pages=parse_page_spec(args.pages),
        detect_tables=args.detect_tables,
        use_ocr=args.use_ocr,
        force_ocr=args.force_ocr,
        encoding=args.encoding,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = build_config(args)
    if not cfg.pdf_path.exists():
        logging.error("PDF not found: %s", cfg.pdf_path)
        return 1

    print("=" * 70)
    print("PDF Extractor")
    print("=" * 70)

    try:
        logging.info("[1/5] Extracting text blocks...")
        pages = extract_pages(cfg)
        logging.info("[2/5] Detecting headers/footers...")
        header_footer = detect_headers_footers(cfg, pages)
        logging.info("[3/5] Filtering irrelevant content...")
        filtered = filter_irrelevant(cfg, pages, header_footer)
        if cfg.detect_tables:
            logging.info("[4/5] Detecting tables...")
            filtered = detect_tables(cfg, filtered)
        else:
            logging.info("[4/5] Skipping table detection")
        logging.info("[5/5] Classifying blocks...")
        classified = classify(cfg, filtered)
    except PasswordRequiredError as exc:
        logging.error("%s", exc)
        return 2
    except (PdfOpenError, ExtractionError) as exc:
        logging.error("%s", exc)
        return 3

    total_blocks = sum(len(page.get("blocks", [])) for page in classified)
    logging.info("Pages: %s, Blocks: %s", len(classified), total_blocks)

    if args.no_json and args.no_md:
        logging.info("No outputs requested; skipping writes.")
    else:
        if not args.no_json:
            save_json(classified, cfg.output_json, encoding=cfg.encoding)
        if not args.no_md:
            save_markdown(classified, cfg.output_md, encoding=cfg.encoding)

    print("\n" + "=" * 70)
    print("Extraction complete!")
    if not args.no_json:
        print(f"  JSON: {cfg.output_json}")
    if not args.no_md:
        print(f"  Markdown: {cfg.output_md}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
