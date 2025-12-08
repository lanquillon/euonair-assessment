# Commands and Options

## PDF extraction
- `python pdf_text_extractor.py sample2.pdf -j all_output/extracted_text_output/extracted_text.json -m all_output/extracted_text_output/extracted_text.md`  
  Extract text with header/footer removal and OCR fallback on low-text pages (default mode).
- `python pdf_text_extractor.py sample2.pdf --force-ocr -j all_output/extracted_text_output/ocr_forced.json -m all_output/extracted_text_output/ocr_forced.md`  
  Force OCR on every page (use when native text is missing or garbled).
- `python pdf_text_extractor.py sample2.pdf --no-ocr -j all_output/extracted_text_output/no_ocr.json -m all_output/extracted_text_output/no_ocr.md`  
  Disable OCR entirely (faster; use when PDF text is clean).
- `python pdf_text_extractor.py sample2.pdf --detect-tables -j all_output/extracted_text_output/with_tables.json -m all_output/extracted_text_output/with_tables.md`  
  Enable alignment-based table detection; detected tables stay structured in JSON and render as Markdown tables.
- `python pdf_text_extractor.py sample2.pdf --pages 2-4,6 -j all_output/extracted_text_output/pages_subset.json -m all_output/extracted_text_output/pages_subset.md`  
  Process only selected pages (1-based ranges or comma-separated).
- `python pdf_text_extractor.py secret.pdf --password YOUR_PASSWORD -j all_output/extracted_text_output/secret.json -m all_output/extracted_text_output/secret.md`  
  Supply a password when the PDF is encrypted.
- `python pdf_text_extractor.py sample2.pdf --encoding utf-8 -j all_output/extracted_text_output/extracted_text.json -m all_output/extracted_text_output/extracted_text.md --verbose`  
  Control output encoding and enable verbose logs for debugging.

## Question generation
- `python question_generation.py`  
  Reads `all_output/extracted_text_output/extracted_text.json` and writes to `all_output/generated_questions_output/questions.json`.
- With custom paths:  
  `python question_generation.py --input-json all_output/extracted_text_output/no_ocr.json --output-json all_output/generated_questions_output/questions_no_ocr.json`

## H5P export
- `python h5p_export.py`  
  Reads `all_output/generated_questions_output/questions.json` and writes H5P content to `all_output/h5p_output/`.

## Full pipeline (sequential)
1) `python pdf_text_extractor.py sample2.pdf` → `all_output/extracted_text_output/`
2) `python question_generation.py` (or with `--input-json/--output-json`) → `all_output/generated_questions_output/`
3) `python h5p_export.py` → `all_output/h5p_output/`

## Output layout
- All results live under `all_output/`: extracted text in `extracted_text_output/`, generated questions in `generated_questions_output/`, and H5P packages in `h5p_output/`.
- JSON is the canonical structured output for downstream steps; Markdown is for human review.
- Markdown now appends a per-page view (`### Page N`) so you can map content back to its source page.
- Headers and footers are removed from both JSON and Markdown outputs.
- OCR requires Tesseract (`pytesseract`, `Pillow`); set `TESSERACT_CMD` if Tesseract is not on PATH.
