# Multi-PDF RAG App with Sources

This project builds a beginner-friendly Streamlit app for uploading multiple PDFs, indexing them in a single FAISS vector store, asking questions across all documents, and displaying source citations with page numbers.

## Files

- `rag_app.py` — main Streamlit application
- `requirements.txt` — required Python dependencies
- `.gitignore` — excludes local secrets, virtual environments, and generated files

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_openai_key_here"
streamlit run rag_app.py
```

## Notes

- The app reads the API key from the `OPENAI_API_KEY` environment variable.
- It preserves PDF metadata for source filenames and page numbers.
- It surfaces conflicting statements when multiple uploaded documents disagree.
