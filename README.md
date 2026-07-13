# Voice Based Concept Understanding Analyzer

An advanced, production-ready AI-powered developer environment and desktop application designed to analyze concept understanding from audio explanations. By integrating speech-to-text transcribers, semantic representation transformers, and acoustic feature extractors, the tool grades semantic alignment, flow fluency, and speaking confidence.

## Features

- **Streamlit Web Dashboard**: Slick dark-themed responsive interface.
- **Speech-to-Text Transcription**: Local OpenAI Whisper integration transcribing voice and generating word-level timestamps.
- **Semantic Understanding Scoring**: Semantic similarity evaluation comparing the transcript against the target explanation using Sentence-BERT (`all-MiniLM-L6-v2`).
- **Acoustic Feature Extraction**: Pitch contour tracking and RMS Energy analysis utilizing Librosa.
- **Fluency Metrics**: Speaking speed (WPM), hesitation rate, and silence/pause mapping.
- **Delivery Confidence Score**: Comprehensive confidence score calculated by blending vocal range, verbal fluency, and semantic coverage.
- **Concept & Keyword Checking**: Stem and phrase boundary-aware exact & fuzzy keyword coverage tracking.
- **Local SQLite Storage**: Saves analysis records, metrics, settings, and session histories.
- **PDF Report Compiler**: Generates formatted, beautiful PDF reports with ReportLab, embedding active acoustic charts.

---

## Folder Structure

```
Voice_Based_Concept_Understanding_Analyzer/
├── venv/                       # Virtual environment (ignored)
├── app.py                      # Streamlit application UI entrypoint
├── requirements.txt            # Python dependencies list
├── verify_env.py               # Pre-flight environment diagnostics script
├── database/                   # SQLite database operations
│   ├── __init__.py
│   └── db_helper.py            # Local history and settings storage
├── analyzer/                   # Concept Analysis Core
│   ├── __init__.py
│   ├── speech_to_text.py       # OpenAI Whisper Integration
│   ├── semantic_similarity.py  # Sentence-BERT Semantic Analysis
│   ├── audio_features.py       # Librosa acoustic & feature extraction
│   ├── fluency_analysis.py     # Speech flow, pauses, words per minute
│   ├── confidence_analysis.py  # AI Confidence combining acoustics & content
│   └── keyword_matching.py     # Key Concept & Concept Coverage matching
├── reports/                    # PDF Report Generator
│   ├── __init__.py
│   └── pdf_generator.py        # ReportLab beautiful PDF creator
├── utils/                      # Visualizations and helper functions
│   ├── __init__.py
│   └── visualization.py        # Plotly & Matplotlib wave, pitch, charts
└── tests/                      # Unit testing suite
    ├── __init__.py
    ├── test_analyzer.py
    └── test_db.py
```

---

## Installation & Setup

1. **Verify Environment Setup**:
   Ensure you are using Python 3.12+ (tested and verified on Python 3.14.3). Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Verify Dependencies**:
   You can run the environment verification script to check imports, CUDA status, and Whisper/SBERT model loading:
   ```bash
   python verify_env.py
   ```

3. **Install dependencies manually (if needed)**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Run the Application

Start the Streamlit dashboard:
```bash
streamlit run app.py
```

Once running, open your browser and navigate to the address shown in the terminal (typically `http://localhost:8501`).

---

## Running Unit Tests

Run the test suite using `pytest`:
```bash
pytest tests/
```
All unit tests should complete successfully in isolated environments.
