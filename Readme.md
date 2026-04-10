# 📄 Research Paper Analyzer

An AI-powered tool that reads academic PDFs and returns structured summaries — built with Python, HuggingFace Transformers, and Streamlit.

---

## 🚀 Quickstart (Do This In Order!)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/research-analyzer.git
cd research-analyzer
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
# This takes a few minutes the first time
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
├── app.py                  # Streamlit UI — the entry point
├── requirements.txt        # All dependencies
├── .gitignore              # Files Git should ignore
├── README.md               # This file
└── src/
    ├── __init__.py         # Makes src/ a Python package
    ├── pdf_extractor.py    # PDF → raw text (PyPDF2)
    ├── preprocessor.py     # Clean text + chunk it
    ├── summarizer.py       # HuggingFace BART model
    └── structurer.py       # Build structured notes
```

---

## 🧠 How It Works

```
PDF Upload → Extract Text → Clean & Chunk → BART Summarize → Structure Notes → Display
```

1. **Extract**: PyPDF2 reads each page and pulls out raw text
2. **Clean**: regex removes broken line wraps, extra spaces, page numbers
3. **Chunk**: text is split into ~500-word pieces (BART's context limit)
4. **Summarize**: each chunk goes through `facebook/bart-large-cnn`
5. **Structure**: keyword heuristics classify sentences into Objective / Methodology / Findings / Conclusion

---

## ⚠️ Known Limitations

- **Scanned PDFs**: PyPDF2 can't read image-based PDFs. Fix: use `pymupdf` + OCR in a future version.
- **Multi-column layouts**: Common in IEEE/ACM papers. Text extraction order can be garbled.
- **Model size**: BART large is ~1.6 GB. First run downloads it automatically.
- **Processing time**: ~30–90 seconds depending on paper length and your hardware.

---

## 🛣️ Roadmap

- [ ] Phase 1 (current): PDF → structured summary
- [ ] Phase 2: Q&A chat on paper content
- [ ] Phase 2: Citation export (APA / BibTeX)
- [ ] Phase 3: Multi-paper semantic search
- [ ] Phase 3: Chart/figure understanding

---

## 🧰 Tech Stack

| Tool | Version | Role |
|------|---------|------|
| Streamlit | 1.35 | Web UI |
| PyPDF2 | 3.0 | PDF text extraction |
| transformers | 4.41 | BART summarization model |
| PyTorch | 2.3 | Model inference engine |