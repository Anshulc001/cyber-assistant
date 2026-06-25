# personal-ai-assistant

A high-performance personal AI assistant running a GGUF quantized LLM on Google Colab's free **T4 GPU**, proxying requests through a **Cloudflare Tunnel** to a local **FastAPI + SQLite + FAISS** backend server running on your PC.

All data (chat logs, RAG documents, settings, vector indexes) lives locally on your PC (the "source of truth"), keeping your data secure and private.

---

## Architecture

```text
              YOUR LOCAL PC (Source of Truth)
   ┌──────────────────────────────────────────────────┐
   │                                                  │
   │   Frontend: HTML5 + CSS + JavaScript (UI)        │
   │      │                                           │
   │      ▼                                           │
   │   Backend (FastAPI):                             │
   │      ├── SQLite DB (chats, messages, settings)   │
   │      ├── Embeddings (BAAI/bge-small-en-v1.5)     │
   │      └── FAISS Vector DB (RAG chunk indexing)     │
   │                                                  │
   └──────────────┬───────────────────▲───────────────┘
                  │                   │
      Streaming   │                   │  Cloudflare
      Chat        │                   │  Tunnel
      Inference   │                   │  (Remote GPU Proxy)
                  ▼                   │
   ┌──────────────────────────────────┴───────────────┐
   │                                                  │
   │          GOOGLE COLAB T4 GPU RUNTIME             │
   │      llama-cpp-python Inference Server           │
   │   (Qwythos-9B-Claude-Mythos-5-1M-MTP-Q6_K GGUF)  │
   │                                                  │
   └──────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Choice | Location |
|---|---|---|
| **LLM** | Qwythos-9B Q6_K GGUF (`empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF`) | Google Colab GPU (T4 VRAM) |
| **Inference Engine** | `llama-cpp-python` (with CUDA layer offloading) | Google Colab |
| **Backend API** | FastAPI + Uvicorn | Local PC |
| **Database** | SQLite (`data/database/assistant.db`) | Local PC |
| **Embeddings** | BAAI/bge-small-en-v1.5 | Local PC (runs on CPU) |
| **Vector DB** | FAISS | Local PC |
| **Storage** | Local disk (`data/` directory) | Local PC |
| **Model Cache** | Google Drive (`Models/` folder) | Google Colab / Drive |
| **Tunnel** | Cloudflare Tunnel (`cloudflared`) | Google Colab |

---

## Project Layout

```text
personal-ai-assistant/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Local settings & directory paths
│   ├── routes/              # Chat, RAG, settings endpoints
│   ├── services/            # DB, memory, model proxy, RAG services
│   └── models/
│       └── schemas.py       # Pydantic schema validation
├── colab/
│   └── AI_Server.ipynb      # Colab notebook (model server)
├── frontend/
│   └── index.html           # Plain HTML/CSS/JS frontend UI
├── rag/
│   └── pipeline.py          # Standalone PDF chunk/vector ingest CLI
├── run_backend.py           # Programmatic backend launcher script
├── requirements.txt         # PC backend dependencies
└── README.md
```

---

## Setup & Running Guide

### 1. First-Time Setup (Once)

Before running the project for the first time, perform these steps on your **local PC**:

1. **Install Python**: Make sure Python 3.10 is installed on your machine.
2. **Setup virtual environment & dependencies**:
   Run the following commands in PowerShell from the project root:
   ```powershell
   # Recreate virtual env with Python 3.10 (to avoid 3.14 pre-release bugs)
   py -3.10 -m venv .venv --clear
   
   # Upgrade pip
   .venv\Scripts\python -m pip install --upgrade pip
   
   # Install all backend and RAG dependencies
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

---

### 2. Every-Time Run Procedure

Follow these steps **every time** you want to start and use the personal AI assistant:

#### Step 1: Start the Colab GPU Inference Server
1. Open [colab/AI_Server.ipynb](file:///c:/Users/anshu/Desktop/personal-ai-assistant/colab/AI_Server.ipynb) in Google Colab (or import it from GitHub).
2. Go to **Runtime** ➔ **Change runtime type** and ensure **T4 GPU** is selected.
3. Run all cells from top to bottom (Runtime ➔ Run all).
   * *Note: Cell 1 will mount Google Drive to cache the model file so it doesn't download it from HuggingFace on every VM recycle.*
4. Wait for the final cell (**Cell 5**) to start. It will print a banner containing your public Cloudflare Tunnel URL:
   ```text
   ==============================================================
   🌍  PUBLIC URL : https://xxxxxxxxxxxx.trycloudflare.com
   ==============================================================
   ```
5. Copy this URL.

#### Step 2: Start the Local Backend Server (on your PC)
1. Open PowerShell in the project root directory.
2. Run the programmatic launcher script (this automatically forces the correct `asyncio` event loop policy for Windows):
   ```powershell
   .venv\Scripts\python run_backend.py
   ```
3. The console will log:
   ```text
   Set WindowsSelectorEventLoopPolicy successfully.
   INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
   ```

#### Step 3: Open the Frontend Chat UI
1. Double-click or open [frontend/index.html](file:///c:/Users/anshu/Desktop/personal-ai-assistant/frontend/index.html) in any web browser (Chrome, Edge, Firefox, Safari).

#### Step 4: Configure & Connect
1. In the top-right corner of the Chat UI, click the **Settings (⚙)** gear icon.
2. In the settings panel:
   * **Backend API URL**: Keep it as `http://localhost:8000` (this connects to your local PC backend).
   * **Colab Tunnel URL**: Paste the `https://xxxxxxxxxxxx.trycloudflare.com` URL you copied from Step 1.
3. Click **Connect**.
4. The status indicator dot at the top of the UI will turn **Green (Connected)**.

#### Step 5: Start Chatting!
* Type messages in the input bar.
* Upload PDFs to index documents locally for RAG queries.
* Switch RAG on or off using the toggle below the input box.
* Use the sidebar to create new chats, switch between previous chats (loaded from your local SQLite DB), or delete/rename conversations.
