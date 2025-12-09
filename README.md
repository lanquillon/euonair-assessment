# PDF-to-H5P Multiple-Choice-Questions Pipeline

Guide for extracting text from PDFs, generating questions, and exporting to H5P.

## Prerequisites
- Windows 10/11 tested; macOS/Linux should work with adapted venv activation.
- Python 3.10+ and Git installed.
- Ollama installed; pull a model and run the server: `ollama pull llama2`, then `ollama serve`.
- Optional OCR: Tesseract installed and reachable via PATH or `TESSERACT_CMD` (e.g., `$env:TESSERACT_CMD="C:\\Users\\<user>\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"`).
- Optional tables: `pdfplumber` (already in requirements).
- Outputs are written under `all_output/` (extracted text, generated questions, H5P files).

## Installation
```powershell
git clone <REPO_URL>
cd <PROJECT_DIR>

python -m venv .venv
./.venv/Scripts/Activate.ps1

pip install -r requirements.txt
```

## PDF extraction (CLI)
- Basic: `python pdf_text_extractor.py sample.pdf`
- With tables + logs: `python pdf_text_extractor.py sample.pdf --detect-tables --verbose`
- Common flags:
  - `--pages 1-3,5` limit pages
  - `--password YOURPASS` for encrypted PDFs
  - `--ocr / --no-ocr` toggle OCR fallback (on by default)
  - `--force-ocr` force OCR on all pages
  - `--output-json out.json`, `--output-md out.md`, `--no-json`, `--no-md`

Outputs default to `all_output/extracted_text_output/` (`extracted_text.json`, `extracted_text.md` with per-page view).

## Question generation
- Requires Ollama running: `ollama serve`
- Default run: `python question_generation.py` -> `all_output/generated_questions_output/questions.json`
- Custom paths: `python question_generation.py --input-json all_output/extracted_text_output/no_ocr.json --output-json all_output/generated_questions_output/questions_no_ocr.json`
- Topic hint (steer focus): `python question_generation.py --topic-hint "interaction design"`
- Auto-version outputs: `python question_generation.py --auto-version`
- Adjust `OLLAMA_MODEL` / `OLLAMA_BASE_URL` in `question_generation.py` if needed.

## H5P export
- Run: `python h5p_export.py` -> `.h5p` files in `all_output/h5p_output/`
- Uses `all_output/generated_questions_output/questions.json` by default; validates basics before writing packages.
- Export all versioned questions (`questions_v*.json`) to subfolders: `python h5p_export.py --all-versions`

For additional command variants (OCR/no-OCR, tables, page subsets, custom paths), see `commands.md`.

## Tests
- Pytest: `pytest`
- Unittest: `python -m unittest discover -s tests -p "test_*.py"`

## Verification
- Review `extracted_text.md` for extraction quality before generating questions.
- Edit questions JSON if you want to refine wording or options.
- Import generated `.h5p` files into your LMS/Lumi with the `H5P.MultiChoice` library available.

## Troubleshooting
- Ollama connection errors: confirm `ollama serve` is running and the model name matches.
- Tesseract not found: install Tesseract and set `TESSERACT_CMD` to the executable path.
- Missing outputs: check file paths/flags and rerun the step.


--------------------------------------------------------------------

## Documentation
##### Kurze Beschreibung der verwendeten KI-Methode/Prompting-Strategie

Das KI-Setup umfasst drei Stufen: (1) Extraktion/Strukturierung der PDF-Inhalte (Header/Footer/Page-Filter, optionale OCR, Tabellenerkennung) in JSON/Markdown. (2) Generierung von Multiple-Choice-Fragen über ein lokales LLM (Ollama, Modell: llama2) auf Basis der strukturierten JSON-Blöcke. (3) Export im H5P-Format für LMS-Integration.

Der zentrale Prompt in question_generation.py erzwingt klare Frage-/Antwortformate (Bloom-Level, Quelle, Index + Buchstabe + Erklärung). Neben dem Prompt-Katalog (https://coda.io/@kic/prompt-katalog) flossen eigene Prompting-Erfahrungen, Online-Beispiele (https://studylib.net/doc/25824333/hmi-qb-answers) und wissenschaftliche Literatur (bspw. "*Automated Educational Question Generation at Different Bloom's Skill Levels using Large Language Models: Strategies and Evaluation*" https://arxiv.org/abs/2408.04394) ein; ChatGPT & Claude halfen beim Feinschliff. Deep-Research diente für Best Practices/Cheat-Sheets zu Prompt-Design und LLM-Evaluierung (v. a. Open-Source-Modelle) inkl. Literaturrecherche. Da ich Programmieranfängerin bin, kamen mehrere LLMs & Code Assistants (bspw. Github Copilot free) sowie Selbst- bzw. Literaturrecherche (bspw. "*Evaluating Large Language Models Trained on Code*" https://arxiv.org/abs/2107.03374) zum Einsatz, um KI-gestützt sauberen, optimierten Code zu erstellen

##### Wie könnte das System erweitert werden (z. B. weitere Fragetypen, OCR-Unterstützung, Feedback-Generierung)?

- durch bspw. Visual Analytics: wie oft werden bestimmte Fragetypen generiert? Woran könnte das liegen?, grafische Darstellung; Bild- bzw. Graph-Generierung zur Veranschaulichung des Inhalts
- Erklärung/Kurzbeschreibung/exakte Quellen-/Literaturangabe bzw. Zitat (wenn mitgegeben/darauf trainiert), warum welche Antwort korrekt ist
- Gamification-Elemente: Punktesystem zur Bewertung der Antworten (Feedback an Kandidaten, wie gut hat dieser abgeschnitten?) sowie der 
- Framework/Guidelines für PDF-Gestaltung / Templates / Einbau spezifischer Indikatoren für bspw. Wunsch-Schwierigkeitsgrad (direkt im pdf) oder code-intern
- Spracherkennung mithilfe installierter Sprachpakte oder ähnlich
- Fragen-Katalog nach Themen sortiert ausgeben lassen, dass der Dozent auswählen kann, welche er verwenden möchte/ in welche Richtung mehr generiert werden soll
- Konvertierung von pdf in html oder ähnlich für bessere Layout-Erkennung(?)
- ggf. Code-Evaluierung https://huggingface.co/collections/Vipitis/code-evaluation
- Output-Iterationen
- Abbildungen oder Platzhalter für diese hinzufügen; dazu Kurzbeschreibungen oder Nummerierung
- GUI
- Qualitätschecks (bspw. durch Reinforcement-Learning Methoden oder ähnlich); Orientierung an *Learning diverse rankings with multi-armed bandits* (https://dl.acm.org/doi/10.1145/1390156.1390255) oder *BanditRank: Learning to Rank Using Contextual Bandits* (https://arxiv.org/abs/1910.10410)
