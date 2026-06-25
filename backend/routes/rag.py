"""RAG route — PDF upload, indexing, and retrieval endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.models.schemas import UploadResponse
from backend.services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base: str | None = Form(default=None),
):
    """Upload a PDF, chunk it, embed it, and add it to the FAISS index."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = await rag_service.ingest(
        filename=file.filename,
        content=content,
        knowledge_base=knowledge_base,
    )
    return result


@router.get("/knowledge-bases")
async def list_knowledge_bases():
    """List all knowledge-base namespaces that have been indexed."""
    bases = await rag_service.list_knowledge_bases()
    return {"knowledge_bases": bases}


@router.delete("/knowledge-bases/{name}")
async def delete_knowledge_base(name: str):
    """Remove a knowledge base and its FAISS index from storage."""
    deleted = await rag_service.delete_knowledge_base(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{name}' not found.")
    return {"detail": f"Knowledge base '{name}' deleted."}
