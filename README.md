# RAG Document Q&A App

A Retrieval-Augmented Generation system that lets you ask questions grounded
in your own PDF documents.

## Status: Stage 2 — Retrieval + local answer generation (in progress)

Runs 100% free and local — no API keys, no subscriptions. Uses Ollama for
the LLM instead of a paid API.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Also install [Ollama](https://ollama.com) (free), then pull a model:
```bash
ollama pull llama3.2
```

## Stage 1: Ingest PDFs

1. Drop 1-2 PDFs into `data/`
2. Run:
   ```bash
   python ingest.py
   ```
3. You should see chunk counts printed per file, and a `chroma_db/` folder
   appear — that's your local vector store.

## Stage 2: Ask questions

Make sure Ollama is running, then:
```bash
python query.py "What is this document about?"
```
or just run `python query.py` and it'll prompt you for a question.

You'll see the retrieved chunks (for transparency/debugging) followed by
the generated answer and which source file(s) it came from.

## Architecture

- `ingest.py` — loads PDFs, chunks text, embeds with `sentence-transformers`
  (local, free), stores in ChromaDB (local, persistent)
- `query.py` — retrieves relevant chunks for a question, builds a grounded
  prompt, and generates an answer via a local Ollama model
- `app.py` — Streamlit frontend (coming in Stage 3)
