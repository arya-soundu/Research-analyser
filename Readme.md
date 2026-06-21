# 📄 Research Paper Analyzer

An AI-powered academic assistant that parses, structures, and lets you converse with dense research PDFs — built with Python, PyMuPDF, local Sentence Transformers, local BART, and Streamlit.

---

## ✨ Features

- **Anti-Scrambling PDF Parser**: Uses PyMuPDF (`fitz`) coordinate-based geometric sorting to reconstruct natural reading flow for double-column academic papers, preventing text garbling.
- **Multi-Modal Extraction**: Automatically extracts, crops, and renders embedded charts, diagrams, and figures.
- **Side-by-Side Live PDF Viewer**: Displays cited pages next to your chat panel. Context-relevant sentences used by the RAG model are highlighted on the PDF image in yellow using visual grounding.
- **Context-Aware Q&A Dialogue**: Runs semantic searches over local dense vectors (`all-MiniLM-L6-v2`), reformulates follow-up queries using Groq's low-latency `Llama-3.1-8B` cloud LPUs, and holds conversational context.
- **Page-Level Local Summarization**: Splices PDF pages into discrete, page-indexed chunks and generates abstractive summaries using `facebook/bart-large-cnn`.
- **State-Machine Structuring**: Classifies summarized points into Objective, Methodology, Findings, and Conclusion categories using keyphrase heuristics.
- **Persistent Workspace Cache**: Uses SHA-256 hashing to cache summaries and chat histories locally. Instantly restores prior analysis sessions on reload or via the sidebar dropdown.
- **Academic Export Options**: Export structured summaries as Obsidian/Notion Markdown notes, and copy formatted APA, MLA, or BibTeX references directly from the sidebar.

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/arya-soundu/Research-analyser.git
cd Research-analyser
```

### 2. Create a virtual environment

```bash
# Creates an isolated Python environment for this project
python -m venv venv

# Activate it (Windows):
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
# Downloads PyMuPDF, PyTorch, Transformers, SentenceTransformers, and Streamlit
```

### 4. Run the app

```bash
streamlit run app.py
# Opens http://localhost:8501 in your browser
```

---

## 🗂️ Project Structure

```
research-analyzer/
├── app.py                  # Streamlit dashboard & layout
├── requirements.txt        # All dependencies
├── .gitignore              # Ignored files (venv, local cache)
├── README.md               # This file
├── .env                    # Hidden API keys (contains GROQ_API_KEY)
└── src/
    ├── __init__.py         # Makes src/ a Python package
    ├── pdf_extractor.py    # PyMuPDF block parser & page renderer
    ├── preprocessor.py     # Clean text + page-by-page chunking
    ├── summarizer.py       # Local HuggingFace BART summarization model
    ├── structurer.py       # Heuristic keyphrase classifier
    ├── qa_engine.py        # Dense embeddings search & Groq dialogue
    └── cache_manager.py    # Hashed JSON/PDF workspace cache manager
```

---

## 🧠 How It Works

```
PDF Ingest ➔ Geometric Coordinate Sort ➔ Page-by-Page Clean & Chunk ➔ BART Summary ➔ State-Machine notes ➔ Local Cache (SHA-256)
```

1. **Extract**: PyMuPDF reads physical text coordinates, grouping them into left/right columns and full-width banners.
2. **Clean & Chunk**: Normalizes page text and chunks it page-by-page, mapping chunks back to their 1-indexed page source.
3. **Summarize**: Feeds page chunks to local BART for abstractive summarization.
4. **Structure**: Segments sentences into academic categories using phrase heuristic heuristics.
5. **Conversational RAG**: Matches user chat queries to vector embeddings, retrieves page numbers, reformulates context questions via Groq, and renders the cited page image side-by-side with yellow highlights.

---

## 🛣️ Roadmap

- **Phase 1**: PDF ➔ structured notes summary
- **Phase 2**: RAG Q&A chat on paper content
- **Phase 2**: Citation export (APA / MLA / BibTeX)
- **Phase 3**: Collapsible PDF page viewer with text highlights
- **Phase 3**: Hashed workspace persistence cache
- **Phase 4**: Multi-paper semantic workspace comparison
- **Phase 4**: Visual chart understanding (Vision LLM)

---

## 🧰 Tech Stack

| Tool                      | Version | Role                          |
| ------------------------- | ------- | ----------------------------- |
| **Streamlit**             | 1.58    | Web UI Dashboard              |
| **PyMuPDF**               | 1.27    | PDF parser & page renderer    |
| **transformers**          | 4.57    | BART summarization model      |
| **PyTorch**               | 2.12    | Local inference engine        |
| **sentence-transformers** | 2.7.0   | Dense vector embedding model  |
| **groq**                  | 1.4.0   | Groq Cloud Llama-3.1 API SDK  |
| **python-dotenv**         | 1.2.2   | Secret `.env` variable loader |
