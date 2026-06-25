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
| 2 | Install Server | Download `llama-server` binary and `cloudflared` |
| 3 | Download GGUF | Download Qwythos-9B GGUF model and vision projector |
| 4 | Start Server | Run `llama-server` on GPU + Cloudflare tunnel |
| 5 | Keep-alive | (Optional) prevent idle-timeout disconnect |
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
# ── Cell 2: Download llama-server & cloudflared dependencies ──────────────
import os
import glob
import shutil
import subprocess
import urllib.request
import zipfile

print('⏳  Installing HuggingFace hub …')
subprocess.run(['pip', 'install', '-q', 'huggingface_hub'], check=True)

print('⏳  Installing cloudflared …')
subprocess.run([
    'wget', '-q', '-O', '/tmp/cloudflared.deb',
    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb',
], check=True)
subprocess.run(['dpkg', '-i', '/tmp/cloudflared.deb'], check=True, capture_output=True)

print('⏳  Downloading pre-built llama-server binary (CUDA 12 support) …')
url = "https://github.com/ggerganov/llama.cpp/releases/download/b3152/llama-b3152-bin-cuda-cu12-x64.zip"
zip_path = "/tmp/llama-bin.zip"
urllib.request.urlretrieve(url, zip_path)

print('⏳  Extracting llama-server archive …')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/llama-bin")

# Locate llama-server executable recursively inside zip extraction directory
server_binaries = glob.glob("/content/llama-bin/**/llama-server", recursive=True)
if server_binaries:
    shutil.move(server_binaries[0], "/content/llama-server")
    os.chmod("/content/llama-server", 0o755)
    print("✓ Successfully installed: /content/llama-server")
else:
    # Print directory tree if not found for debug
    for root, dirs, files in os.walk("/content/llama-bin"):
        print(root, files)
    raise FileNotFoundError("llama-server binary was not found in the extracted zip!")

print('\\n✅  All dependencies and llama-server installed.')
r = subprocess.run(['cloudflared', '--version'], capture_output=True, text=True)
print('   cloudflared :', (r.stdout or r.stderr).strip())
r2 = subprocess.run(['/content/llama-server', '--version'], capture_output=True, text=True)
print('   llama-server:', (r2.stdout or r2.stderr).strip())
"""

# ─────────────────────────────────────────────────────────────────────────
CELL3 = """\
# ── Cell 3: Download Qwythos-9B GGUF Model & CLIP Projector ──────────────
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR   = Path('/content/Models')
GGUF_REPO    = 'empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF'
GGUF_FILE    = 'Qwythos-9B-Claude-Mythos-5-1M-MTP-Q6_K.gguf'
MODEL_LABEL  = 'Qwythos-9B-Claude-Mythos-5-1M (Q6_K GGUF, 7.09 GiB)'
MODEL_PATH   = MODELS_DIR / GGUF_FILE

MMPROJ_FILE  = 'mmproj-Qwythos-9B-Claude-Mythos-5-1M-F16.gguf'
MMPROJ_PATH  = MODELS_DIR / MMPROJ_FILE

# 1. Download Model
if MODEL_PATH.exists():
    print(f'Using cached GGUF: {MODEL_PATH}')
else:
    print(f'Downloading {GGUF_FILE} from HuggingFace …')
    hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    print(f'  Saved to: {MODEL_PATH}')

# 2. Download CLIP Vision Projector
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

print('\\n✅  All GGUF files downloaded and verified.')
"""

# ─────────────────────────────────────────────────────────────────────────
CELL4 = """\
# ── Cell 4: Start llama-server + Cloudflare Tunnel ────────────────────────
import re
import subprocess
import time

# Kill any stale server/tunnel processes from previous runs
subprocess.run(['pkill', '-f', 'llama-server'], capture_output=True)
subprocess.run(['pkill', '-f', 'cloudflared'],  capture_output=True)
time.sleep(1)

# ── Start C++ llama-server on GPU ─────────────────────────────────────────
print("Starting llama-server in background ...")
# Offloads all layers to T4 GPU, binds context window to 8K, hosts on port 8000
_llama_proc = subprocess.Popen(
    [
        '/content/llama-server',
        '-m', '/content/Models/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q6_K.gguf',
        '--mmproj', '/content/Models/mmproj-Qwythos-9B-Claude-Mythos-5-1M-F16.gguf',
        '-ngl', '-1',          # Offload all layers to T4 GPU
        '-c', '8192',          # 8k context window
        '--host', '0.0.0.0',
        '--port', '8000',
    ],
    stdout=subprocess.DEVNULL, # Keep Jupyter cell clean
    stderr=subprocess.DEVNULL,
)
time.sleep(5)
print('✅  llama-server running on http://0.0.0.0:8000')

# ── Start cloudflared tunnel ──────────────────────────────────────────────
print("Starting Cloudflare tunnel ...")
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
    print('🧪  Quick chat test (OpenAI format):')
    body = '{"model": "qwythos", "messages": [{"role": "user", "content": "Say hello in one sentence."}], "stream": false}'
    print('    curl -s -X POST', public_url + '/v1/chat/completions',
          "-H 'Content-Type: application/json'",
          "-d '" + body + "'", '| head -c 300')
    print()
    print(SEP)
else:
    print('⚠️   Could not detect tunnel URL. Raw cloudflared output:')
    for _ in range(30):
        line = _tunnel.stdout.readline().decode('utf-8', errors='replace').rstrip()
        if line:
            print('    |', line)
"""

# ─────────────────────────────────────────────────────────────────────────
CELL5 = """\
# ── Cell 5 (Optional): Keep-Alive Loop ───────────────────────────────────
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
    ],
}

# ── Write output ──────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent.parent / "colab" / "AI_Server.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")

print(f"[OK] Generated: {out}")
print(f"   Cells     : {len(notebook['cells'])} (1 markdown + {len(notebook['cells']) - 1} code)")
print(f"   File size : {out.stat().st_size / 1024:.1f} KB")
