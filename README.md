# EUonAIR Assessment

# Programmieraufgabe: KI-gestützte Erstellung von Multiple-Choice-Questions

Herzlich willkommen zu dieser Programmieraufgabe!

Ziel ist es, Ihre Fähigkeiten in Python sowie im effektiven Einsatz von KI-Technologien zu demonstrieren. Im Fokus steht dabei die automatisierte Erstellung von Multiple-Choice-Fragen aus Vorlesungsfolien, inklusive der strukturierten Textextraktion aus PDF-Dokumenten und dem Export im H5P-Format zur Weiterverwendung in Lernmanagementsystemen.

## 1. Aufgabenbeschreibung

### 1.1 PDF-Verarbeitung und Textextraktion

#### Eingabeformat

- Verarbeiten Sie PDF-Dokumente, die typische Vorlesungsfolien oder Skripte enthalten.
- Unterstützte PDF-Typen:
    - Textbasierte PDFs mit direkt extrahierbarem Text (Pflicht)
    - (Optional) Bildbasierte PDFs mittels OCR bzw. multimodalen LLMs

- **Wichtig:**
    - Behandeln Sie verschiedene Folienlayouts robust (Titel, Bullet Points, Tabellen).
    - Erkennen Sie nach Möglichkeit die logische Struktur (Kapitel, Abschnitte, Foliennummern).

#### Textextraktion
- Extrahieren Sie den Textinhalt folienweise oder abschnittsweise.
- Bewahren Sie nach Möglichkeit die hierarchische Struktur (Überschriften, Unterpunkte).
- Filtern Sie irrelevante Inhalte (z. B. Seitenzahlen, wiederkehrende Header/Footer).

#### Fehlerbehandlung
- Fangen Sie beschädigte oder passwortgeschützte PDFs kontrolliert ab.
- Loggen Sie Extraktionsprobleme nachvollziehbar.


### 1.2 KI-gestützte Fragegenerierung

#### Ziel
- Generieren Sie aus dem extrahierten Folieninhalt didaktisch sinnvolle Multiple-Choice-Fragen, bestehend aus:
    - Fragentext (Question Stem)
    - Einer korrekten Antwort
    - Mindestens zwei bis drei plausiblen Distraktoren (falsche Antworten)
    - (Optional) Erklärung/Feedback zur korrekten Antwort

#### Qualitätshinweise
- Distraktoren sollen plausibel, aber eindeutig falsch sein.
- Die korrekte Antwort soll sich klar aus dem Folieninhalt ableiten lassen.
- (Empfehlung) Berücksichtigen Sie nach Möglichkeit verschiedene Schwierigkeitsgrade (Bloom-Level).

#### Erlaubte KI-Methoden
- Open-Source-Modelle (z. B. Hugging Face Transformers, Ollama, LLaMA)
- Kommerzielle APIs (z. B. OpenAI, Anthropic), unter der Bedingung, dass Sie keine API-Keys fest im Code hinterlegen (Umgang via Umgebungsvariablen oder separater Konfigurationsdatei).
- Hybride Ansätze (Kombination aus regelbasierten Methoden und LLMs)


### 1.3 H5P-Export

#### Zielformat
- Exportieren Sie die generierten Fragen im H5P-Format, sodass sie in gängigen Lernmanagementsystemen (Moodle, ILIAS, etc.) importiert werden können.
- Unterstützen Sie den H5P-Inhaltstyp **"Multiple Choice"**.

#### Technische Umsetzung
- H5P-Pakete sind ZIP-Archive mit definierter Struktur (content.json, h5p.json).
- Generieren Sie valide H5P-Pakete, die ohne Nachbearbeitung importierbar sind.

#### Validierung
- Testen Sie die generierten H5P-Dateien auf Importierbarkeit und korrekte Darstellung, z. B. mit:
    - [Lumi](https://lumi.education/) (kostenloser Desktop-Editor und -Player)
    - [h5p.org](https://h5p.org/) (Online-Test)


### 1.4 Ergebnisaufbereitung

**Strukturiertes Ergebnis (Haupt-Deliverable)**
- Erstellen Sie eine Pipeline, die aus PDF-Eingaben H5P-Dateien mit Multiple-Choice-Fragen generiert.
- Mögliche Zwischenformate:
    - Export (Text, Markdown oder JSON) der extrahierten und aufbereiteten Inhalte
    - JSON-Struktur mit extrahiertem Text und generierten Fragen

**Empfehlung**
- Trennen Sie klar zwischen Extraktion, Generierung und Export.
- Ermöglichen Sie eine manuelle Überprüfung der Fragen vor dem finalen H5P-Export.


## 2. Technische Anforderungen

- **Programmiersprache:**
    - Python 3.10 oder höher

- **Python-Bibliotheken:**
    - PDF-Verarbeitung: frei wählbar (z. B. PyMuPDF, pdfplumber)
    - KI-Integration: frei wählbar (z. B. openai, anthropic, transformers, langchain)
    - H5P-Generierung: eigene Implementierung oder verfügbare Hilfsbibliotheken

- **Modularer Code:**
    - Trennen Sie die Hauptfunktionen klar (Extraktion, Generierung, Export)
    - Verwenden Sie Type Hints und sinnvolle Kommentare

- **Logging & Fehlerbehandlung:**
    - Loggen Sie wichtige Prozessschritte
    - Behandeln Sie API-Fehler und Timeouts

- **Requirements/Dependency Management:**
    - Stellen Sie eine `requirements.txt` bereit

- **Versionskontrolle:**
    - Nutzen Sie Git und entwickeln Sie in einem eigenen Branch
    - Committen Sie keine sensiblen Daten wie API-Keys ins Repository


## 3. Vorgehensweise & Abgabe

1. Forken Sie dieses (oder ein bereitgestelltes) GitHub-Repository.
2. Erstellen Sie einen eigenen Branch, in dem Sie Ihre Lösung implementieren.
3. Implementieren Sie folgende Schritte:
    - PDF-Extraktion (Text und Struktur)
    - KI-gestützte Fragegenerierung
    - H5P-Export
4. **Ergebnisformat:**
    - Lauffähiges Python-Skript oder CLI-Tool
    - Beispiel-H5P-Dateien (generiert aus Testdaten)
    - Zwischenergebnisse als JSON/CSV
5. **Dokumentation:**
    - `README.md` mit Installationsanleitung und Beispielaufruf
    - Kurze Beschreibung der verwendeten KI-Methode/Prompting-Strategie (kann in README integriert sein)
6. **Funktionsnachweis:**
    - Demonstrieren Sie die Funktionsfähigkeit anhand eines Beispieldurchlaufs
    - (Optional) Ergänzen Sie Unit-Tests für kritische Funktionen
7. **Abgabe:**
    - Committen Sie Ihre fertige Lösung in Ihrem Branch
    - (Optional) Stellen Sie einen Pull Request für eine direkte Code-Review


## 4. Beispielfragen zur Orientierung

- Wie gehen Sie mit unterschiedlichen PDF-Strukturen um (verschiedene Layouts, fehlende Struktur)?
- Welche Prompting-Strategien nutzen Sie, um qualitativ hochwertige Fragen zu generieren?
- Wie stellen Sie sicher, dass Distraktoren plausibel, aber eindeutig falsch sind?
- Wie stellen Sie sicher, dass ggf. vorgegebene bzw. verlange Schwierigkeitsgrade (Bloom-Level) erreicht werden?
- Wie validieren Sie die generierten H5P-Dateien auf Kompatibilität?
- Wie könnte das System erweitert werden (z. B. weitere Fragetypen, OCR-Unterstützung, Feedback-Generierung)?


## 5. Bewertungskriterien

- **Funktionalität (40%)**
    - Erfüllung der Kernanforderungen (Extraktion, Generierung, Export)
    - Robustheit (Umgang mit verschiedenen PDF-Formaten, API-Fehlern)
    - Qualität der generierten Fragen

- **Code-Qualität (30%)**
    - Struktur und Lesbarkeit (Funktionen, Kommentare, Type Hints)
    - Fehlerbehandlung und Logging

- **Dokumentation (20%)**
    - Vollständigkeit (README, Installationsanleitung, Beispielaufruf)
    - Nachvollziehbarkeit der KI-Methodik

- **Innovation (10%)**
    - Kreative Ansätze oder zusätzliche Features
    - Effizienz der Implementierung


## 6. Einschränkungen und Hinweise

- Verwenden Sie für Tests nur eigene oder frei lizenzierte PDF-Dokumente.
- API-Keys (falls nötig) nicht im Code committen – nutzen Sie Umgebungsvariablen oder lokale Konfigurationsdateien.
- Dokumentieren Sie ggf. Kosten für kommerzielle APIs, falls Sie diese einsetzen.
- Eine Test-Ausführung mit einem kleinen Foliensatz (z. B. 5–10 Folien, 3–5 generierte Fragen) reicht aus, um den Ablauf zu demonstrieren.


## 7. Abgabetermin und Kontakt

- **Abgabefrist:** Dienstag, 9. Dezember, 20:00 Uhr (MEZ)
- **Einreichung:** GitHub-Pull-Request und GitHub-/Download-Link (zur Sicherheit)
- **Kontakt für Rückfragen:** carsten.lanquillon@hs-heilbronn.de

---

Viel Erfolg bei der Bearbeitung!

Wir freuen uns auf Ihre kreative und saubere Umsetzung der Aufgabe. Bei Fragen oder Problemen stehen wir Ihnen innerhalb des vorgegebenen Rahmens gerne zur Verfügung.
