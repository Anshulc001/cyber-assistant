"""
rag/pipeline.py — Standalone RAG pipeline (Milestone 4 Part C).

Independently runnable script that ingests PDFs into a FAISS knowledge base
using BAAI/bge-small-en-v1.5 embeddings and token-aware chunking.

Usage
-----
# Index a PDF into the "cybersecurity" knowledge base:
    python rag/pipeline.py ingest --pdf path/to/doc.pdf --kb cybersecurity

# Query a knowledge base:
    python rag/pipeline.py query --query "What is SQL injection?" --kb cybersecurity --top-k 5

# List all knowledge bases:
    python rag/pipeline.py list-kbs

# Delete a knowledge base:
    python rag/pipeline.py delete-kb --kb cybersecurity
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import shutil
import sys
from pathlib import Path

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
EMBED_MODEL  = settings.EMBEDDING_MODEL        # "BAAI/bge-small-en-v1.5"
EMBED_DIM    = settings.EMBEDDING_DIM          # 384
CHUNK_TOKENS = 512                             # max tokens per chunk
CHUNK_OVERLAP = 50                             # token overlap between chunks


# ── Token-aware PDF chunker ───────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader    = PdfReader(str(pdf_path))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return full_text


def chunk_by_tokens(text: str, max_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into token-aware chunks using the embedding model's tokeniser."""
    from transformers import AutoTokenizer

    tok_model = str(settings.models_dir / EMBED_MODEL.split("/")[-1])
    # Fall back to HF hub if not cached locally
    try:
        tok = AutoTokenizer.from_pretrained(EMBED_MODEL, cache_dir=str(settings.models_dir))
    except Exception:
        tok = AutoTokenizer.from_pretrained(EMBED_MODEL)

    token_ids = tok.encode(text, add_special_tokens=False)
    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end        = start + max_tokens
        chunk_ids  = token_ids[start:end]
        chunk_text = tok.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        start += max_tokens - overlap

    logger.info(
        "Chunked %d tokens → %d chunks (max=%d, overlap=%d)",
        len(token_ids), len(chunks), max_tokens, overlap,
    )
    return chunks


# ── FAISS index management ────────────────────────────────────────────────

def _idx_dir(kb: str) -> Path:
    return settings.vector_db_dir / kb


def load_index(kb: str):
    """Load an existing FAISS index + metadata for *kb*. Returns (index, meta)."""
    import faiss

    idx_file  = _idx_dir(kb) / "index.faiss"
    meta_file = _idx_dir(kb) / "meta.pkl"

    if idx_file.exists() and meta_file.exists():
        index = faiss.read_index(str(idx_file))
        meta  = pickle.load(open(meta_file, "rb"))
        logger.info("Loaded index for kb=%r: %d chunks.", kb, index.ntotal)
    else:
        import faiss as _faiss
        index = _faiss.IndexFlatIP(EMBED_DIM)
        meta  = []
        logger.info("Created new index for kb=%r.", kb)
    return index, meta


def save_index(kb: str, index, meta: list[dict]) -> None:
    import faiss

    d = _idx_dir(kb)
    d.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(d / "index.faiss"))
    with open(d / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    logger.info("Saved index for kb=%r: %d total chunks.", kb, index.ntotal)


# ── Embedder ──────────────────────────────────────────────────────────────

def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, cache_folder=str(settings.models_dir))


# ── Public pipeline functions ─────────────────────────────────────────────

def ingest(pdf_path: Path, kb: str = "default") -> dict:
    """
    Full ingestion pipeline:
      PDF → extract text → token-chunk → embed → FAISS → persist to Drive.

    Returns a summary dict.
    """
    import numpy as np

    settings.ensure_dirs()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Extracting text from '%s' …", pdf_path.name)
    text = extract_text(pdf_path)
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    logger.info("Chunking text …")
    chunks = chunk_by_tokens(text)
    if not chunks:
        raise ValueError("No chunks produced from text.")

    logger.info("Embedding %d chunks …", len(chunks))
    embedder = get_embedder()
    vectors  = embedder.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype("float32")

    logger.info("Updating FAISS index for kb=%r …", kb)
    index, meta = load_index(kb)
    records = [
        {"text": c, "source": pdf_path.name, "chunk_index": i}
        for i, c in enumerate(chunks)
    ]
    index.add(np.array(vectors))
    meta.extend(records)
    save_index(kb, index, meta)

    summary = {
        "filename"      : pdf_path.name,
        "knowledge_base": kb,
        "chunks_created": len(chunks),
        "total_indexed" : index.ntotal,
    }
    logger.info("Ingestion complete: %s", summary)
    return summary


def query(q: str, kb: str = "default", top_k: int = 5) -> list[dict]:
    """
    Semantic search pipeline:
      query → embed → FAISS search → ranked results.

    Returns a list of dicts with keys: text, source, score, chunk_index.
    """
    import numpy as np

    index, meta = load_index(kb)
    if index.ntotal == 0:
        logger.warning("Knowledge base '%s' is empty.", kb)
        return []

    embedder = get_embedder()
    q_vec    = embedder.encode([q], normalize_embeddings=True).astype("float32")
    k        = min(top_k, index.ntotal)
    scores, ids = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        m = meta[idx]
        results.append({
            "text"       : m["text"],
            "source"     : m["source"],
            "chunk_index": m["chunk_index"],
            "score"      : float(score),
        })
    return results


def list_knowledge_bases() -> list[dict]:
    """Return all persisted knowledge bases with chunk counts."""
    import faiss

    vdb = settings.vector_db_dir
    if not vdb.exists():
        return []
    kbs = []
    for d in sorted(vdb.iterdir()):
        idx_file = d / "index.faiss"
        if d.is_dir() and idx_file.exists():
            try:
                n = faiss.read_index(str(idx_file)).ntotal
                kbs.append({"name": d.name, "chunks": n})
            except Exception:
                pass
    return kbs


def delete_knowledge_base(kb: str) -> bool:
    d = _idx_dir(kb)
    if not d.exists():
        return False
    shutil.rmtree(d)
    logger.info("Deleted knowledge base '%s'.", kb)
    return True


# ── CLI ───────────────────────────────────────────────────────────────────

def _cmd_ingest(args: argparse.Namespace) -> None:
    result = ingest(Path(args.pdf), args.kb)
    print("\n✅  Ingestion complete:")
    for k, v in result.items():
        print(f"   {k:<20} {v}")


def _cmd_query(args: argparse.Namespace) -> None:
    results = query(args.query, args.kb, args.top_k)
    if not results:
        print(f"No results found in kb='{args.kb}'.")
        return
    print(f"\n🔍  Top {len(results)} results from kb='{args.kb}':\n")
    for i, r in enumerate(results, 1):
        print(f"── Result {i}  (score={r['score']:.4f}, source={r['source']})")
        print(r["text"][:400].strip())
        print()


def _cmd_list(args: argparse.Namespace) -> None:
    kbs = list_knowledge_bases()
    if not kbs:
        print("No knowledge bases found.")
        return
    print(f"\n📚  Knowledge bases ({settings.vector_db_dir}):\n")
    for kb in kbs:
        print(f"   {kb['name']:<30}  {kb['chunks']:>6} chunks")
    print()


def _cmd_delete(args: argparse.Namespace) -> None:
    if not delete_knowledge_base(args.kb):
        print(f"Knowledge base '{args.kb}' not found.")
    else:
        print(f"✅  Deleted '{args.kb}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone RAG pipeline — ingest PDFs and query FAISS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a PDF into a knowledge base.")
    p_ingest.add_argument("--pdf",  required=True,     help="Path to the PDF file.")
    p_ingest.add_argument("--kb",   default="default", help="Knowledge base name.")
    p_ingest.set_defaults(func=_cmd_ingest)

    # query
    p_query = sub.add_parser("query", help="Semantic search a knowledge base.")
    p_query.add_argument("--query",  required=True,     help="Search query string.")
    p_query.add_argument("--kb",     default="default", help="Knowledge base name.")
    p_query.add_argument("--top-k",  type=int, default=5, dest="top_k", help="Number of results.")
    p_query.set_defaults(func=_cmd_query)

    # list-kbs
    p_list = sub.add_parser("list-kbs", help="List all knowledge bases.")
    p_list.set_defaults(func=_cmd_list)

    # delete-kb
    p_del = sub.add_parser("delete-kb", help="Delete a knowledge base and its index.")
    p_del.add_argument("--kb", required=True, help="Knowledge base name to delete.")
    p_del.set_defaults(func=_cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
