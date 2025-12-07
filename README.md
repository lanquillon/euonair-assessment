# PDF-to-H5P Pipeline Instructions

Setup and execution guide for extracting text from PDFs, generating questions, and exporting to H5P.

## Prerequisites
- Windows 10/11 tested; macOS/Linux should work with adapted venv activation.
- Python 3.10+ and Git installed.
- Ollama installed; pull a model and run the server: `ollama pull llama2`, then `ollama serve`.
- Optional OCR: Tesseract installed and reachable via PATH or `TESSERACT_CMD` (e.g., `$env:TESSERACT_CMD="C:\\Users\\<user>\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"`).

## Installation
```powershell
# clone repo
git clone <REPO_URL>
cd <PROJECT_DIR>

# create & activate venv (PowerShell)
python -m venv .venv
./.venv/Scripts/Activate.ps1

# install dependencies
pip install -r requirements.txt
```

## Configuration
- Place your PDF in the project folder or set `PDF_FILE` in `pdf_text_extractor.py`.
- Adjust `OLLAMA_MODEL` / `OLLAMA_BASE_URL` in `question_generation.py` if you use a different model or endpoint.
- Ensure Tesseract is installed if you need OCR; otherwise OCR is skipped.
- OCR defaults (in `pdf_text_extractor.py`):
  - `FORCE_OCR = False` (keep native text; OCR only when text is scarce; set True to force OCR on all pages)
  - `OCR_MIN_TEXT_LENGTH = 30` (threshold for triggering OCR when not forced)
  - Pages are rendered at 400 DPI with grayscale/contrast boost to improve OCR.
  - Set Tesseract path in your session, e.g.  
    `PS> $env:TESSERACT_CMD = "C:\Users\<user>\AppData\Local\Tesseract-OCR\tesseract.exe"`

## Run the pipeline
```powershell
# 1) extract text from PDF
python pdf_text_extractor.py   # outputs: extracted_text.json, extracted_text.md

# 2) generate questions (requires running Ollama)
python question_generation.py  # outputs: questions.json

# 3) export to H5P MultiChoice
python h5p_export.py           # outputs: one .h5p per question in h5p_output/
```

## Tests
- Pytest: `pytest`
- Unittest: `python -m unittest discover`

## Verification & editing
- Review `extracted_text.md` for extraction quality; re-run extraction if needed.
- Edit `questions.json` before export if you want to refine wording or options.
- Import generated `.h5p` files into your LMS/Lumi with the `H5P.MultiChoice` library available.

## Troubleshooting
- Ollama connection errors: confirm `ollama serve` is running and the model name matches.
- Tesseract not found: install Tesseract and set `TESSERACT_CMD` to the executable path.
- Missing outputs: check file paths in the scripts and rerun the corresponding step.
