"""Standalone RAG pipeline script.

Run from repo root:
    python rag/pipeline.py --pdf path/to/doc.pdf --kb cybersecurity

This wraps the same logic as rag_service but as a CLI tool for batch indexing
documents without starting the full FastAPI server.
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


def extract_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def embed(chunks: list[str]) -> "np.ndarray":
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        settings.EMBEDDING_MODEL,
        cache_folder=str(settings.models_dir),
    )
    return model.encode(chunks, normalize_embeddings=True, show_progress_bar=True).astype("float32")


def upsert_index(kb: str, vectors: "np.ndarray", meta: list[dict]) -> None:
    import faiss
    import numpy as np

    idx_dir = settings.vector_db_dir / kb
    idx_dir.mkdir(parents=True, exist_ok=True)
    index_file = idx_dir / "index.faiss"
    meta_file = idx_dir / "meta.pkl"

    if index_file.exists():
        index = faiss.read_index(str(index_file))
        existing_meta: list[dict] = pickle.load(open(meta_file, "rb"))
    else:
        index = faiss.IndexFlatIP(settings.EMBEDDING_DIM)
        existing_meta = []

    index.add(np.array(vectors))
    existing_meta.extend(meta)

    faiss.write_index(index, str(index_file))
    with open(meta_file, "wb") as f:
        pickle.dump(existing_meta, f)

    logger.info(
        "Index updated: %d total chunks in kb='%s'.", index.ntotal, kb
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into a FAISS knowledge base.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument("--kb", default="default", help="Knowledge base name.")
    parser.add_argument(
        "--chunk-size", type=int, default=settings.CHUNK_SIZE, help="Chars per chunk."
    )
    parser.add_argument(
        "--overlap", type=int, default=settings.CHUNK_OVERLAP, help="Char overlap."
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)

    settings.ensure_dirs()

    logger.info("Extracting text from '%s' …", pdf_path.name)
    text = extract_text(pdf_path)
    if not text.strip():
        logger.error("No text extracted from PDF.")
        sys.exit(1)

    chunks = chunk_text(text, args.chunk_size, args.overlap)
    logger.info("%d chunks created.", len(chunks))

    vectors = embed(chunks)

    meta = [
        {"text": c, "source": pdf_path.name, "chunk_index": i}
        for i, c in enumerate(chunks)
    ]
    upsert_index(args.kb, vectors, meta)
    logger.info("Done.")


if __name__ == "__main__":
    main()
