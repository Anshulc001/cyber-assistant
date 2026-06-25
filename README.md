# personal-ai-assistant

A self-hosted, ChatGPT-style personal AI assistant that runs an open-source LLM on
Google Colab's free **T4 GPU** and is reachable from any browser through a
**Cloudflare Tunnel**. It supports document RAG and keeps all state on Google Drive.

## Architecture

```
My Laptop (code only, no model)
    │
    ▼
GitHub (source of truth)
    │
    ▼
Google Colab (GPU server)
    ├── Qwythos-9B loaded via HuggingFace Transformers
    ├── FastAPI serving the AI API
    ├── FAISS for RAG
    └── Cloudflare Tunnel exposing the API
    │
    ▼
Browser (phone or laptop)
    └── Chat UI calls the Colab API via the Cloudflare URL
```

## Tech stack

| Layer        | Choice                                            |
| ------------ | ------------------------------------------------- |
| LLM          | Qwythos-9B (HuggingFace Transformers, Colab only) |
| Backend      | FastAPI + Uvicorn (async)                         |
| Frontend     | Plain HTML + CSS + Vanilla JS (Phase 1)           |
| Embeddings   | BAAI/bge-small-en-v1.5                            |
| Vector DB    | FAISS (local files, no cloud)                     |
| Storage      | Google Drive (models, chats, vectors, PDFs)       |
| Tunnel       | Cloudflare Tunnel (`cloudflared`)                 |
| GPU runtime  | Google Colab free T4                              |

## Project layout

```
personal-ai-assistant/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── config.py            # All configurable values
│   ├── routes/              # chat, rag, settings endpoints
│   ├── services/            # model, rag, memory logic
│   └── models/
│       └── schemas.py       # Pydantic models
├── frontend/
│   └── index.html           # Phase 1 chat UI
├── colab/
│   └── AI_Server.ipynb      # Colab notebook (GPU server)
├── rag/
│   └── pipeline.py          # PDF → chunks → embeddings → FAISS
├── scripts/
│   └── setup.sh             # Environment setup
├── requirements.txt
└── README.md
```

## Storage layout (on Google Drive, created by Colab)

```
/content/drive/MyDrive/AI-Assistant/
├── Models/          # Qwythos cached here
├── ChatHistory/     # JSON files per chat
├── VectorDB/        # FAISS index files
├── Uploads/         # Uploaded PDFs
└── Settings/        # config JSON
```

## Roadmap

1. ✅ Project structure + GitHub setup
2. Colab notebook: model loads, FastAPI runs, Cloudflare tunnel works
3. HTML chat UI connects to the Colab API, streaming works
4. RAG: PDF upload → chunking → BAAI embeddings → FAISS → answers from docs
5. Persistent chat history on Google Drive
6. Multiple knowledge-base folders (Cybersecurity, Programming, College, …)
7. React frontend replaces HTML
8. Web search tool (optional)

## Status

**Milestone 1 complete** — clean foundation, no AI functionality yet.

## Local development

This repo holds source only; the model and GPU live on Colab. To work on the
backend locally:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
# then open http://127.0.0.1:8000/health
```
