"""RAG service — PDF ingestion, embedding, FAISS indexing, and retrieval.

All FAISS index files and document metadata are persisted to Google Drive
(or the local data directory when USE_DRIVE=False).
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.models.schemas import SourceChunk, UploadResponse

logger = logging.getLogger(__name__)


def _index_path(knowledge_base: str | None) -> Path:
    kb = knowledge_base or "default"
    return settings.vector_db_dir / kb


class RAGService:
    """Singleton that manages FAISS indices keyed by knowledge-base name."""

    def __init__(self) -> None:
        # In-memory cache: kb_name -> {"index": faiss.Index, "meta": list[dict]}
        self._indices: dict[str, dict[str, Any]] = {}
        self._embed_model = None

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                cache_folder=str(settings.models_dir),
            )
            logger.info("Embedding model '%s' loaded.", settings.EMBEDDING_MODEL)
        return self._embed_model

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_index(self, kb: str) -> None:
        import faiss

        idx_dir = _index_path(kb)
        idx_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._indices[kb]["index"], str(idx_dir / "index.faiss"))
        with open(idx_dir / "meta.pkl", "wb") as f:
            pickle.dump(self._indices[kb]["meta"], f)
        logger.debug("FAISS index saved for kb='%s'.", kb)

    def _load_index(self, kb: str) -> bool:
        import faiss

        idx_dir = _index_path(kb)
        index_file = idx_dir / "index.faiss"
        meta_file = idx_dir / "meta.pkl"
        if not index_file.exists() or not meta_file.exists():
            return False
        self._indices[kb] = {
            "index": faiss.read_index(str(index_file)),
            "meta": pickle.load(open(meta_file, "rb")),
        }
        logger.info("FAISS index loaded for kb='%s' (%d chunks).",
                    kb, self._indices[kb]["index"].ntotal)
        return True

    def _ensure_index(self, kb: str) -> None:
        if kb not in self._indices:
            if not self._load_index(kb):
                import faiss

                self._indices[kb] = {
                    "index": faiss.IndexFlatIP(settings.EMBEDDING_DIM),
                    "meta": [],
                }

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end].strip())
            start += size - overlap
        return [c for c in chunks if c]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest(
        self,
        filename: str,
        content: bytes,
        knowledge_base: str | None,
    ) -> UploadResponse:
        """Parse PDF → chunk → embed → add to FAISS → persist."""
        kb = knowledge_base or "default"

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._ingest_sync(filename, content, kb),
        )
        return result

    def _ingest_sync(self, filename: str, content: bytes, kb: str) -> UploadResponse:
        import io

        import numpy as np
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        full_text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

        if not full_text.strip():
            raise ValueError(f"Could not extract text from '{filename}'.")

        chunks = self._chunk_text(full_text)
        embedder = self._get_embedder()
        vectors = embedder.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        self._ensure_index(kb)
        meta_records = [
            {"text": c, "source": filename, "chunk_index": i}
            for i, c in enumerate(chunks)
        ]

        import numpy as np
        self._indices[kb]["index"].add(np.array(vectors))
        self._indices[kb]["meta"].extend(meta_records)
        self._save_index(kb)

        logger.info("Ingested '%s' → %d chunks into kb='%s'.", filename, len(chunks), kb)
        return UploadResponse(
            filename=filename,
            knowledge_base=kb,
            chunks_created=len(chunks),
            bytes_received=len(content),
            indexed=True,
            message=f"Ingested {len(chunks)} chunks into knowledge base '{kb}'.",
        )

    async def retrieve(
        self,
        query: str,
        knowledge_base: str | None,
        top_k: int,
    ) -> list[SourceChunk]:
        """Embed the query and return the top-k nearest chunks."""
        kb = knowledge_base or "default"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._retrieve_sync(query, kb, top_k),
        )

    def _retrieve_sync(self, query: str, kb: str, top_k: int) -> list[SourceChunk]:
        import numpy as np

        self._ensure_index(kb)
        index = self._indices[kb]["index"]
        meta = self._indices[kb]["meta"]

        if index.ntotal == 0:
            return []

        embedder = self._get_embedder()
        q_vec = embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        k = min(top_k, index.ntotal)
        scores, ids = index.search(q_vec, k)

        results: list[SourceChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            m = meta[idx]
            results.append(
                SourceChunk(
                    text=m["text"],
                    source=m["source"],
                    score=float(score),
                    chunk_index=m["chunk_index"],
                )
            )
        return results

    async def list_knowledge_bases(self) -> list[dict[str, Any]]:
        """Return all persisted knowledge bases with chunk counts."""
        import faiss

        if not settings.vector_db_dir.exists():
            return []
        kbs = []
        for d in sorted(settings.vector_db_dir.iterdir()):
            idx_file = d / "index.faiss"
            if d.is_dir() and idx_file.exists():
                try:
                    n = faiss.read_index(str(idx_file)).ntotal
                    kbs.append({"name": d.name, "chunks": n})
                except Exception:
                    pass
        return kbs

    async def delete_knowledge_base(self, name: str) -> bool:
        import shutil

        idx_dir = _index_path(name)
        if not idx_dir.exists():
            return False
        shutil.rmtree(idx_dir)
        self._indices.pop(name, None)
        logger.info("Deleted knowledge base '%s'.", name)
        return True


rag_service = RAGService()
