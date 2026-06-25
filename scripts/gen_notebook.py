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
# 🤖 Personal AI Assistant — Colab GPU Inference Server

> **Before running:** Runtime → Change runtime type → **T4 GPU**

Run all cells **top to bottom**. The final cell prints your public Cloudflare URL.

| # | Cell | Purpose |
|---|------|---------|
| 1 | Setup Cache | Create local `Models/` cache folder |
| 2 | Install deps | Install `llama-cpp-python`, `fastapi`, `uvicorn`, `pillow`, `numpy`, and `cloudflared` |
| 3 | Load model | Download GGUF and vision projector (mmproj), and load with CUDA + Custom Chat Handler |
| 4 | FastAPI app | Expose standard OpenAI `/v1/chat/completions` and `/health` endpoints |
| 5 | Start server | Start uvicorn on :8000 and Cloudflare tunnel -> prints Public URL |
| 6 | Keep-alive | (Optional) prevent idle-timeout disconnect |
"""

# ─────────────────────────────────────────────────────────────────────────
CELL1 = """\
# ── Cell 1: Create Local Models Cache Directory ───────────────────────────
from pathlib import Path

MODELS_DIR = Path('/content/Models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print('✅  Local models cache directory set up.')
print(f'   Models Cache: {MODELS_DIR}')
"""

# ─────────────────────────────────────────────────────────────────────────
CELL2 = """\
# ── Cell 2: Install All Dependencies ──────────────────────────────────────
import os, subprocess, sys
import torch
import urllib.request

def _pip(*pkgs, extra_flags=None):
    flags = ['--no-cache-dir']
    if extra_flags:
        flags.extend(extra_flags)
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', *pkgs, *flags],
        stdout=subprocess.DEVNULL,
    )

print('⏳  Detecting CUDA version …')
cuda_version = torch.version.cuda
if cuda_version:
    cuda_tag = 'cu' + cuda_version.replace('.', '')[:3]
else:
    cuda_tag = 'cpu'
print(f'   CUDA version: {cuda_version} -> Tag: {cuda_tag}')

print('⏳  Scanning for compatible pre-built wheel repositories …')
valid_urls = []
tags_to_check = []

if cuda_tag.startswith('cu12'):
    tags_to_check = [cuda_tag, 'cu125', 'cu124', 'cu123', 'cu122', 'cu121']
elif cuda_tag.startswith('cu11'):
    tags_to_check = [cuda_tag, 'cu118']
else:
    tags_to_check = [cuda_tag]

# De-duplicate while preserving order
seen = set()
tags_to_check = [t for t in tags_to_check if not (t in seen or seen.add(t))]

for tag in tags_to_check:
    url = f'https://abetlen.github.io/llama-cpp-python/whl/{tag}/'
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                valid_urls.append(f'https://abetlen.github.io/llama-cpp-python/whl/{tag}')
                print(f'   [✓] Found compatible repository for {tag}')
    except Exception:
        pass

# Assemble pip command
pip_cmd = [
    sys.executable, '-m', 'pip', 'install',
    'llama-cpp-python',
    '--no-cache-dir',
    '--force-reinstall',
]

if valid_urls:
    print('⏳  Installing pre-built llama-cpp-python wheel (should take <15s) …')
    pip_cmd.append('--only-binary=:all:')
    for url in valid_urls:
        pip_cmd.extend(['--extra-index-url', url])
else:
    print('⚠️   No pre-built wheels found. Falling back to source compilation (takes ~5-10 min) …')
    os.environ['CMAKE_ARGS']  = '-DGGML_CUDA=on'
    os.environ['FORCE_CMAKE'] = '1'

subprocess.run(pip_cmd, check=True)

print('⏳  Installing HuggingFace hub & Server libs …')
_pip('huggingface_hub>=0.23', 'fastapi>=0.110', 'uvicorn[standard]>=0.29', 'pillow', 'numpy')

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
# ── Cell 3: Download & Load Qwythos-9B GGUF Model & CLIP Projector ─────────
# Uses llama-cpp-python — all layers offloaded to T4 GPU.
# GGUF file is cached locally; subsequent runs skip the download if cached.
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

MODELS_DIR   = Path('/content/Models')
GGUF_REPO    = 'empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF'
GGUF_FILE    = 'Qwythos-9B-Claude-Mythos-5-1M-MTP-Q6_K.gguf'
MODEL_LABEL  = 'Qwythos-9B-Claude-Mythos-5-1M  (Q6_K GGUF, 7.09 GiB)'
MODEL_PATH   = MODELS_DIR / GGUF_FILE

MMPROJ_FILE  = 'mmproj-Qwythos-9B-Claude-Mythos-5-1M-F16.gguf'
MMPROJ_PATH  = MODELS_DIR / MMPROJ_FILE

if 'llm' in globals() and globals()['llm'] is not None:
    print('Model already in memory — skipping reload.')
    print(f'  {MODEL_LABEL}')
else:
    # ── 1. Download Model (skipped if cached locally) ────────────────────
    if MODEL_PATH.exists():
        print(f'Using cached GGUF: {MODEL_PATH}')
    else:
        print(f'Downloading {GGUF_FILE} from HuggingFace …')
        print(f'  Repo : {GGUF_REPO}')
        print(f'  Size : ~7.09 GiB  (be patient on first run)')
        hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print(f'  Saved to: {MODEL_PATH}')

    # ── 2. Download Vision Projector (skipped if cached locally) ─────────
    if MMPROJ_PATH.exists():
        print(f'Using cached MMPROJ: {MMPROJ_PATH}')
    else:
        print(f'Downloading {MMPROJ_FILE} from HuggingFace …')
        hf_hub_download(
            repo_id=GGUF_REPO,
            filename=MMPROJ_FILE,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print(f'  Saved to: {MMPROJ_PATH}')

    # ── 3. Load into GPU via llama-cpp with Custom Chat Handler ───────────
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    print('Loading model and CLIP vision projector into VRAM …')
    import llama_cpp
    from llama_cpp.llama_chat_format import Llava15ChatHandler

    # Define custom chat handler to format prompts as ChatML for Qwythos-9B
    class CustomLlavaChatHandler(Llava15ChatHandler):
        DEFAULT_SYSTEM_MESSAGE = None
        CHAT_FORMAT = (
            "{% for message in messages %}"
            "{% if message.role == 'system' %}"
            "<|im_start|>system\\n{{ message.content }}<|im_end|>\\n"
            "{% endif %}"
            "{% if message.role == 'user' %}"
            "<|im_start|>user\\n"
            "{% if message.content is string %}"
            "{{ message.content }}"
            "{% endif %}"
            "{% if message.content is iterable %}"
            "{% for content in message.content %}"
            "{% if content.type == 'image_url' and content.image_url is string %}"
            "{{ content.image_url }}"
            "{% endif %}"
            "{% if content.type == 'image_url' and content.image_url is mapping %}"
            "{{ content.image_url.url }}"
            "{% endif %}"
            "{% endfor %}"
            "{% for content in message.content %}"
            "{% if content.type == 'text' %}"
            "{{ content.text }}"
            "{% endif %}"
            "{% endfor %}"
            "{% endif %}"
            "<|im_end|>\\n"
            "{% endif %}"
            "{% if message.role == 'assistant' and message.content is not none %}"
            "<|im_start|>assistant\\n{{ message.content }}<|im_end|>\\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\\n"
            "{% endif %}"
        )

    chat_handler = CustomLlavaChatHandler(clip_model_path=str(MMPROJ_PATH))

    llm = Llama(
        model_path=str(MODEL_PATH),
        chat_handler=chat_handler,
        n_gpu_layers=-1,          # offload ALL layers to T4
        n_ctx=100000,             # 100k context window
        flash_attn=True,          # Enable Flash Attention
        type_k=llama_cpp.GGML_TYPE_Q8_0, # 8-bit quantized Key Cache
        type_v=llama_cpp.GGML_TYPE_Q8_0, # 8-bit quantized Value Cache
        n_batch=512,
        verbose=False,
    )

    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated() / 1e9
        reserv = torch.cuda.memory_reserved()  / 1e9
        total  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'\\n✅  {MODEL_LABEL} with CLIP vision projector loaded.')
        print(f'   GPU       : {torch.cuda.get_device_name(0)}')
        print(f'   Allocated : {alloc:.2f} GB')
        print(f'   Reserved  : {reserv:.2f} GB')
        print(f'   Total GPU : {total:.2f} GB')
        print(f'   Free      : ~{(total - reserv):.2f} GB')
    else:
        print(f'\\n✅  {MODEL_LABEL} loaded on CPU.')
"""

# ─────────────────────────────────────────────────────────────────────────
CELL4 = """\
# ── Cell 4: FastAPI Application — All Routes ─────────────────────────────
import asyncio
import json
import threading
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_cpp import Llama

# ── Config ───────────────────────────────────────────────────────────────
MODEL_LABEL = 'Qwythos-9B-Claude-Mythos-5-1M (Q6_K GGUF)'

# ── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(title='Personal AI Assistant Inference Server', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── GET /health ──────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    gpu = 'unavailable'
    import torch
    if torch.cuda.is_available():
        gpu = f'{torch.cuda.get_device_name(0)}'
    return {
        'status'      : 'ok',
        'model'       : MODEL_LABEL,
        'model_loaded': ('llm' in globals() and globals().get('llm') is not None),
        'gpu'         : gpu,
        'timestamp'   : datetime.utcnow().isoformat() + 'Z',
    }

# ── POST /v1/chat/completions — streaming SSE via llama-cpp ──────────────
import queue as _queue

async def _stream_tokens(messages: list, max_tokens: int, temperature: float, top_p: float, repeat_penalty: float):
    q    = _queue.Queue()
    DONE = object()

    def _generate():
        try:
            # We enforce stop=["<|im_end|>", "<|im_start|>"] to prevent prompt template loops
            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=["<|im_end|>", "<|im_start|>"],
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
        
        # Output choice/delta format (OpenAI spec) & token format (legacy)
        payload = {
            "choices": [
                {
                    "delta": {
                        "content": item
                    }
                }
            ],
            "token": item
        }
        yield 'data: ' + json.dumps(payload) + '\\n\\n'
        await asyncio.sleep(0)

    yield 'data: [DONE]\\n\\n'

@app.post('/v1/chat/completions')
async def chat_completions(body: dict):
    _llm = globals().get('llm')
    if _llm is None:
        raise HTTPException(503, detail='Model not loaded. Run Cell 3 first.')

    messages = body.get('messages')
    if not messages:
        raise HTTPException(400, detail='Missing messages parameter.')

    max_tokens     = int(body.get('max_tokens', 1024))
    temperature    = float(body.get('temperature', 0.7))
    top_p          = float(body.get('top_p', 0.9))
    repeat_penalty = float(body.get('repeat_penalty', 1.1))

    return StreamingResponse(
        _stream_tokens(messages, max_tokens, temperature, top_p, repeat_penalty),
        media_type='text/event-stream',
        headers={
            'Cache-Control'    : 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )

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
    print('🧪  Quick chat test:')
    body = '{"messages": [{"role": "user", "content": "Say hello in one sentence."}]}'
    print('    curl -s -X POST', public_url + '/v1/chat/completions',
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

print(f"[OK] Generated: {out}")
print(f"   Cells     : {len(notebook['cells'])} (1 markdown + 6 code)")
print(f"   File size : {out.stat().st_size / 1024:.1f} KB")
