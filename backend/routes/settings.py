"""Settings route — expose and update runtime configuration."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def get_settings():
    """Return all current (non-secret) configuration values."""
    return {
        "model_name": settings.MODEL_NAME,
        "max_new_tokens": settings.MAX_NEW_TOKENS,
        "temperature": settings.TEMPERATURE,
        "top_p": settings.TOP_P,
        "context_window": settings.CONTEXT_WINDOW,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "top_k": settings.TOP_K,
        "use_drive": settings.USE_DRIVE,
        "storage_root": str(settings.storage_root),
    }
