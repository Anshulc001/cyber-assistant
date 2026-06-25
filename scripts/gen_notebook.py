#!/usr/bin/env python3
"""
Generate colab/AI_Server.ipynb from cell source strings.
Run from the repo root: python scripts/gen_notebook.py
"""
import json
import uuid
from pathlib import Path


# ── Notebook cell helpers ─────────────────────────────────────────────────

def _id():
    return uuid.uuid4().hex[:8]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _id(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": source,
    }


# ── Cell sources ──────────────────────────────────────────────────────────

HEADER_MD = """\
# 🤖 Personal AI Assistant — Colab GPU Server

> **Before running:** Runtime → Change runtime type → **T4 GPU**

Run all cells **top to bottom**. The final cell prints your public Cloudflare URL.

| # | Cell | Purpose |
|---|------|---------|
| 1 | Mount Drive | Mount Google Drive + create `AI-Assistant/` folders |
| 2 | Install deps | ML libs, FastAPI, FAISS, cloudflared |
| 3 | Load model | Download Qwythos-9B → Drive cache, 4-bit NF4 quant |
| 4 | FastAPI app | All API routes (chat, RAG, history) defined inline |
| 5 | Start server | uvicorn on :8000 + Cloudflare tunnel → prints URL |
| 6 | Keep-alive | (Optional) prevent idle-timeout disconnect |
"""

# ─────────────────────────────────────────────────────────────────────────
CELL1 = """\
# ── Cell 1: Mount Google Drive + Create Folder Structure ──────────────────
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/AI-Assistant')
_DIRS = ['Models', 'ChatHistory', 'VectorDB', 'Uploads', 'Settings']
for d in _DIRS:
    (DRIVE_ROOT / d).mkdir(parents=True, exist_ok=True)

print('✅  Google Drive mounted.')
print(f'   Root    : {DRIVE_ROOT}')
print(f'   Folders : {", ".join(_DIRS)}')
"""

# ─────────────────────────────────────────────────────────────────────────
CELL2 = """\
# ── Cell 2: Install All Dependencies ──────────────────────────────────────
import os, subprocess, sys

def _pip(*pkgs, extra_flags=()):
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', *pkgs, *extra_flags],
        stdout=subprocess.DEVNULL,
    )

print('⏳  Installing llama-cpp-python with CUDA (compiles ~3 min) …')
os.environ['CMAKE_ARGS']     = '-DGGML_CUDA=on'
os.environ['FORCE_CMAKE']    = '1'
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install',
    'llama-cpp-python', '-q',
    '--force-reinstall', '--no-cache-dir',
])

print('⏳  Installing HuggingFace hub (for GGUF download) …')
_pip('huggingface_hub>=0.23', 'transformers>=4.40', 'sentencepiece', 'safetensors')

print('⏳  Installing web server + RAG packages …')
_pip('fastapi>=0.110', 'uvicorn[standard]>=0.29', 'python-multipart',
     'sentence-transformers>=3.0', 'faiss-cpu>=1.8',
     'pypdf>=4.0', 'httpx>=0.27', 'pydantic>=2.5')

print('⏳  Installing cloudflared …')
subprocess.run([
    'wget', '-q', '-O', '/tmp/cloudflared.deb',
    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb',
], check=True)
subprocess.run(['dpkg', '-i', '/tmp/cloudflared.deb'], check=True, capture_output=True)

print('\\n✅  All dependencies installed.')
r = subprocess.run(['cloudflared', '--version'], capture_output=True, text=True)
print('   cloudflared:', (r.stdout or r.stderr).strip())
"""

# ─────────────────────────────────────────────────────────────────────────
CELL3 = """\
# ── Cell 3: Download & Load Qwythos-9B Q6_K GGUF ─────────────────────────
# Uses llama-cpp-python — all layers offloaded to T4 GPU.
# GGUF file is cached to Google Drive; subsequent runs skip the download.
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

DRIVE_ROOT   = Path('/content/drive/MyDrive/AI-Assistant')
GGUF_REPO    = 'empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF'
GGUF_FILE    = 'Qwythos-9B-Claude-Mythos-5-1M-MTP-Q6_K.gguf'
MODEL_LABEL  = 'Qwythos-9B-Claude-Mythos-5-1M  (Q6_K GGUF, 7.09 GiB)'
MODEL_PATH   = DRIVE_ROOT / 'Models' / GGUF_FILE

if 'llm' in globals() and globals()['llm'] is not None:
    print('Model already in memory — skipping reload.')
    print(f'  {MODEL_LABEL}')
else:
    # ── 1. Download (skipped if cached on Drive) ──────────────────────────
    if MODEL_PATH.exists():
        print(f'Using cached GGUF: {MODEL_PATH}')
    else:
        print(f'Downloading {GGUF_FILE} from HuggingFace …')
        print(f'  Repo : {GGUF_REPO}')
        print(f'  Size : ~7.09 GiB  (be patient on first run)')
        hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            local_dir=str(DRIVE_ROOT / 'Models'),
            local_dir_use_symlinks=False,
        )
        print(f'  Saved to: {MODEL_PATH}')

    # ── 2. Load into GPU via llama-cpp ────────────────────────────────────
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    print('Loading model into VRAM …')
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,   # offload ALL layers to T4
        n_ctx=8192,        # 8 K context window
        n_batch=512,
        verbose=False,
    )

    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated() / 1e9
        reserv = torch.cuda.memory_reserved()  / 1e9
        total  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'\\n✅  {MODEL_LABEL}')
        print(f'   GPU       : {torch.cuda.get_device_name(0)}')
        print(f'   Allocated : {alloc:.2f} GB')
        print(f'   Reserved  : {reserv:.2f} GB')
        print(f'   Total GPU : {total:.2f} GB')
        print(f'   Free      : ~{(total - reserv):.2f} GB')
    else:
        print(f'\\n✅  {MODEL_LABEL} loaded on CPU.')
"""

# ─────────────────────────────────────────────────────────────────────────
# CELL 4: Complete FastAPI app — all routes inline
# ─────────────────────────────────────────────────────────────────────────
CELL4 = """\
# ── Cell 4: FastAPI Application — All Routes ──────────────────────────────
import asyncio
import io
import json
import pickle
import threading
import uuid
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pypdf import PdfReader
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer as _EmbedTok

# ── Config ────────────────────────────────────────────────────────────────
DRIVE_ROOT       = Path('/content/drive/MyDrive/AI-Assistant')
CHAT_HISTORY_DIR = DRIVE_ROOT / 'ChatHistory'
VECTOR_DB_DIR    = DRIVE_ROOT / 'VectorDB'
UPLOADS_DIR      = DRIVE_ROOT / 'Uploads'

MODEL_LABEL      = 'Qwythos-9B-Claude-Mythos-5-1M (Q6_K GGUF)'
EMBED_MODEL      = 'BAAI/bge-small-en-v1.5'
EMBED_DIM        = 384
CHUNK_TOKENS     = 512      # max tokens per chunk
CHUNK_OVERLAP    = 50       # overlap in tokens
TOP_K            = 5

SYSTEM_DEFAULT = (
    "You are a helpful, concise, and accurate personal AI assistant. "
    "Answer the user's question directly. If you don't know, say so."
)

# ── Embedding + tokenizer for chunking ───────────────────────────────────
_embed_model  = None
_embed_tok    = None

def _get_embedder():
    global _embed_model, _embed_tok
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL, cache_folder=str(DRIVE_ROOT / 'Models'))
        _embed_tok   = _EmbedTok.from_pretrained(EMBED_MODEL, cache_dir=str(DRIVE_ROOT / 'Models'))
    return _embed_model

# ── Token-aware text chunking ─────────────────────────────────────────────
def _chunk(text: str) -> list:
    global _embed_tok
    _get_embedder()   # ensure tokenizer loaded
    token_ids = _embed_tok.encode(text, add_special_tokens=False)
    chunks, start = [], 0
    while start < len(token_ids):
        end         = start + CHUNK_TOKENS
        chunk_ids   = token_ids[start:end]
        chunk_text  = _embed_tok.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start += CHUNK_TOKENS - CHUNK_OVERLAP
    return chunks

# ── FAISS index management ────────────────────────────────────────────────
_indices: dict = {}

def _idx_dir(kb: str) -> Path:
    return VECTOR_DB_DIR / kb

def _ensure_index(kb: str) -> dict:
    if kb in _indices:
        return _indices[kb]
    idx_file  = _idx_dir(kb) / 'index.faiss'
    meta_file = _idx_dir(kb) / 'meta.pkl'
    if idx_file.exists() and meta_file.exists():
        index = faiss.read_index(str(idx_file))
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        print(f'Loaded FAISS index kb={kb!r}: {index.ntotal} chunks')
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        meta  = []
    _indices[kb] = {'index': index, 'meta': meta}
    return _indices[kb]

def _persist_index(kb: str) -> None:
    d = _idx_dir(kb)
    d.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_indices[kb]['index'], str(d / 'index.faiss'))
    with open(d / 'meta.pkl', 'wb') as f:
        pickle.dump(_indices[kb]['meta'], f)

# ── RAG retrieval ─────────────────────────────────────────────────────────
def _retrieve(query: str, kb: str, top_k: int = TOP_K) -> list:
    data = _ensure_index(kb)
    if data['index'].ntotal == 0:
        return []
    emb   = _get_embedder()
    q_vec = emb.encode([query], normalize_embeddings=True).astype('float32')
    k     = min(top_k, data['index'].ntotal)
    scores, ids = data['index'].search(q_vec, k)
    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            m = data['meta'][idx]
            out.append({'text': m['text'], 'source': m['source'], 'score': float(score)})
    return out

# ── Message list builder (llama-cpp chat completion format) ──────────────
def _build_messages(message: str, history: list, system: str, context: str = '') -> list:
    sys_content = system
    if context:
        sys_content += '\\n\\nRelevant context retrieved from your documents:\\n\\n' + context
    msgs = [{'role': 'system', 'content': sys_content}]
    for turn in history[-10:]:
        msgs.append({'role': turn.get('role', 'user'), 'content': str(turn.get('content', ''))})
    msgs.append({'role': 'user', 'content': message})
    return msgs

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title='Personal AI Assistant', version='1.0.0', docs_url='/docs')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── GET /health ───────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    gpu = {}
    if torch.cuda.is_available():
        p   = torch.cuda.get_device_properties(0)
        gpu = {
            'device'      : p.name,
            'total_gb'    : round(p.total_memory / 1e9, 2),
            'allocated_gb': round(torch.cuda.memory_allocated() / 1e9, 2),
            'reserved_gb' : round(torch.cuda.memory_reserved()  / 1e9, 2),
        }
    return {
        'status'      : 'ok',
        'model'       : MODEL_LABEL,
        'model_loaded': ('llm' in globals() and globals().get('llm') is not None),
        'gpu'         : gpu,
        'timestamp'   : datetime.utcnow().isoformat() + 'Z',
    }

# ── POST /chat — streaming SSE via llama-cpp ─────────────────────────────
import queue as _queue

async def _stream_tokens(messages: list, max_tokens: int, temperature: float, top_p: float):
    # Run llama-cpp chat completion in a daemon thread; stream tokens via a queue.
    q    = _queue.Queue()
    DONE = object()

    def _generate():
        try:
            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                stream=True,
            )
            for chunk in stream:
                delta = chunk['choices'][0].get('delta', {})
                tok   = delta.get('content', '')
                if tok:
                    q.put(tok)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(DONE)

    threading.Thread(target=_generate, daemon=True).start()

    while True:
        try:
            item = q.get(timeout=120)
        except _queue.Empty:
            break
        if item is DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield 'data: ' + json.dumps({'token': item}) + '\\n\\n'
        await asyncio.sleep(0)

    yield 'data: [DONE]\\n\\n'

@app.post('/chat')
async def chat(body: dict):
    _llm = globals().get('llm')
    if _llm is None:
        raise HTTPException(503, detail='Model not loaded. Run Cell 3 first.')

    msg = body.get('message', '').strip()
    if not msg:
        raise HTTPException(400, detail='message is required.')

    history     = body.get('history', [])
    system      = body.get('system_prompt', SYSTEM_DEFAULT)
    kb          = body.get('knowledge_base')
    max_tokens  = int(body.get('max_new_tokens', 1024))
    temperature = float(body.get('temperature', 0.7))
    top_p       = float(body.get('top_p', 0.9))

    context  = ''
    rag_used = False
    if kb:
        chunks = _retrieve(msg, kb)
        if chunks:
            rag_used = True
            context  = '\\n\\n---\\n\\n'.join(
                '[Source: ' + c['source'] + ']\\n' + c['text'] for c in chunks
            )

    messages = _build_messages(msg, history, system, context)

    return StreamingResponse(
        _stream_tokens(messages, max_tokens, temperature, top_p),
        media_type='text/event-stream',
        headers={
            'Cache-Control'    : 'no-cache',
            'X-Accel-Buffering': 'no',
            'X-RAG-Used'       : str(rag_used).lower(),
        },
    )

# ── POST /upload — PDF → chunks → FAISS ──────────────────────────────────
@app.post('/upload')
async def upload(
    file          : UploadFile = File(...),
    knowledge_base: str        = Form(default='default'),
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, detail='Only PDF files are accepted.')
    raw = await file.read()
    if not raw:
        raise HTTPException(400, detail='Uploaded file is empty.')

    save_dir = UPLOADS_DIR / knowledge_base
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / file.filename).write_bytes(raw)

    reader = PdfReader(io.BytesIO(raw))
    text   = '\\n'.join(p.extract_text() or '' for p in reader.pages)
    if not text.strip():
        raise HTTPException(422, detail='No extractable text found in PDF.')

    chunks  = _chunk(text)
    emb     = _get_embedder()
    vectors = emb.encode(chunks, normalize_embeddings=True, show_progress_bar=False).astype('float32')

    data    = _ensure_index(knowledge_base)
    records = [{'text': c, 'source': file.filename, 'chunk_index': i} for i, c in enumerate(chunks)]
    data['index'].add(np.array(vectors))
    data['meta'].extend(records)
    _persist_index(knowledge_base)

    return {
        'filename'      : file.filename,
        'knowledge_base': knowledge_base,
        'chunks_created': len(chunks),
        'bytes_received': len(raw),
        'indexed'       : True,
        'message'       : f"Indexed {len(chunks)} chunks into '{knowledge_base}'.",
    }

# ── POST /query-rag ───────────────────────────────────────────────────────
@app.post('/query-rag')
async def query_rag(body: dict):
    query = body.get('query', '').strip()
    kb    = body.get('knowledge_base', 'default')
    top_k = int(body.get('top_k', TOP_K))
    if not query:
        raise HTTPException(400, detail='query is required.')
    data = _ensure_index(kb)
    if data['index'].ntotal == 0:
        return {'chunks': [], 'knowledge_base': kb, 'message': f"'{kb}' is empty."}
    results = _retrieve(query, kb, top_k)
    return {'chunks': results, 'knowledge_base': kb, 'total_indexed': data['index'].ntotal}

# ── GET /knowledge-bases ──────────────────────────────────────────────────
@app.get('/knowledge-bases')
async def list_kbs():
    if not VECTOR_DB_DIR.exists():
        return {'knowledge_bases': []}
    kbs = []
    for d in VECTOR_DB_DIR.iterdir():
        if d.is_dir() and (d / 'index.faiss').exists():
            try:
                n = faiss.read_index(str(d / 'index.faiss')).ntotal
                kbs.append({'name': d.name, 'chunks': n})
            except Exception:
                pass
    return {'knowledge_bases': kbs}

# ── GET /chats ────────────────────────────────────────────────────────────
@app.get('/chats')
async def list_chats():
    if not CHAT_HISTORY_DIR.exists():
        return {'chats': []}
    chats = []
    for f in sorted(CHAT_HISTORY_DIR.glob('*.json'),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
            chats.append({
                'id'           : d.get('id', f.stem),
                'title'        : d.get('title', 'Untitled'),
                'created_at'   : d.get('created_at'),
                'updated_at'   : d.get('updated_at'),
                'message_count': len(d.get('messages', [])),
            })
        except Exception:
            pass
    return {'chats': chats}

# ── GET /chats/{chat_id} ──────────────────────────────────────────────────
@app.get('/chats/{chat_id}')
async def get_chat(chat_id: str):
    path = CHAT_HISTORY_DIR / f'{chat_id}.json'
    if not path.exists():
        raise HTTPException(404, detail=f"Chat '{chat_id}' not found.")
    return json.loads(path.read_text(encoding='utf-8'))

# ── POST /chats/save ──────────────────────────────────────────────────────
@app.post('/chats/save')
async def save_chat(body: dict):
    chat_id = body.get('id') or str(uuid.uuid4())
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAT_HISTORY_DIR / f'{chat_id}.json'
    now  = datetime.utcnow().isoformat() + 'Z'
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            pass
    data = {
        'id'        : chat_id,
        'title'     : body.get('title', prev.get('title', 'New Chat')),
        'created_at': prev.get('created_at', now),
        'updated_at': now,
        'messages'  : body.get('messages', prev.get('messages', [])),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'id': chat_id, 'saved': True, 'updated_at': now}

# ── DELETE /chats/{chat_id} ───────────────────────────────────────────────
@app.delete('/chats/{chat_id}')
async def delete_chat(chat_id: str):
    path = CHAT_HISTORY_DIR / f'{chat_id}.json'
    if not path.exists():
        raise HTTPException(404, detail=f"Chat '{chat_id}' not found.")
    path.unlink()
    return {'id': chat_id, 'deleted': True}

# ── PATCH /chats/{chat_id}/rename ────────────────────────────────────────
@app.patch('/chats/{chat_id}/rename')
async def rename_chat(chat_id: str, body: dict):
    path = CHAT_HISTORY_DIR / f'{chat_id}.json'
    if not path.exists():
        raise HTTPException(404, detail=f"Chat '{chat_id}' not found.")
    data = json.loads(path.read_text(encoding='utf-8'))
    data['title']      = body.get('title', data.get('title', 'Untitled'))
    data['updated_at'] = datetime.utcnow().isoformat() + 'Z'
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'id': chat_id, 'title': data['title']}

# ── Summary ───────────────────────────────────────────────────────────────
print('✅  FastAPI app ready. Endpoints:')
for route in app.routes:
    if hasattr(route, 'methods'):
        for m in sorted(route.methods):
            print(f'   {m:7} {route.path}')
"""

# ─────────────────────────────────────────────────────────────────────────
CELL5 = """\
# ── Cell 5: Start FastAPI + Cloudflare Tunnel ─────────────────────────────
import re
import subprocess
import threading
import time

import uvicorn

if 'app' not in globals():
    raise RuntimeError("Run Cell 4 first to define the FastAPI app.")

# Kill any stale processes from a previous run
subprocess.run(['pkill', '-f', 'uvicorn'],      capture_output=True)
subprocess.run(['pkill', '-f', 'cloudflared'],  capture_output=True)
time.sleep(1)

# ── Start uvicorn in a daemon thread ──────────────────────────────────────
def _run_server():
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='warning')

_srv = threading.Thread(target=_run_server, daemon=True)
_srv.start()
time.sleep(3)
print('✅  uvicorn running on http://0.0.0.0:8000')

# ── Start cloudflared tunnel ──────────────────────────────────────────────
_tunnel = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://localhost:8000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

public_url = None
deadline   = time.time() + 45
while time.time() < deadline:
    line  = _tunnel.stdout.readline().decode('utf-8', errors='replace')
    match = re.search(r'https://[\\w-]+\\.trycloudflare\\.com', line)
    if match:
        public_url = match.group(0)
        break

SEP = '=' * 62
if public_url:
    print()
    print(SEP)
    print('🌍  PUBLIC URL :', public_url)
    print(SEP)
    print()
    print('📋  Paste into the chat UI settings bar:')
    print('   ', public_url)
    print()
    print('🔍  Verify the server is up:')
    print('    curl', public_url + '/health')
    print()
    print('🧪  Quick non-streaming chat test:')
    body = '{"message": "Say hello in one sentence.", "history": []}'
    print('    curl -s -X POST', public_url + '/chat',
          "-H 'Content-Type: application/json'",
          "-d '" + body + "'", '| head -c 300')
    print()
    print('📖  Interactive docs:', public_url + '/docs')
    print(SEP)
else:
    print('⚠️   Could not detect tunnel URL. Raw cloudflared output:')
    for _ in range(30):
        line = _tunnel.stdout.readline().decode('utf-8', errors='replace').rstrip()
        if line:
            print('    |', line)
"""

# ─────────────────────────────────────────────────────────────────────────
CELL6 = """\
# ── Cell 6 (Optional): Keep-Alive Loop ───────────────────────────────────
# Prevents Colab from disconnecting due to inactivity.
# Interrupt the kernel (⏹ button or Runtime → Interrupt) to stop.
import time

print('⏱  Keep-alive running … interrupt kernel to stop.')
counter = 0
while True:
    time.sleep(60)
    counter += 1
    print(f'  ✓ {counter} min — server still running at {public_url}')
"""

# ── Assemble notebook ─────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "machine_shape": "hm",
            "provenance": [],
            "name": "AI_Server.ipynb",
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12",
        },
    },
    "cells": [
        md_cell(HEADER_MD),
        code_cell(CELL1),
        code_cell(CELL2),
        code_cell(CELL3),
        code_cell(CELL4),
        code_cell(CELL5),
        code_cell(CELL6),
    ],
}

# ── Write output ──────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent.parent / "colab" / "AI_Server.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")

print(f"✅  Generated: {out}")
print(f"   Cells     : {len(notebook['cells'])} (1 markdown + 6 code)")
print(f"   File size : {out.stat().st_size / 1024:.1f} KB")
