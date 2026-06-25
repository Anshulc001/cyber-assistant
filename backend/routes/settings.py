"""Settings route — expose and update runtime configuration including Colab Tunnel URL."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from backend.config import settings
from backend.services.db_service import db_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
async def get_settings():
    """Return all current configuration values, loading Colab Tunnel URL from DB."""
    colab_url = db_service.get_setting("colab_url", "")
    return {
        "colab_url": colab_url,
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


@router.post("")
async def update_settings(body: dict):
    """Save the Colab Tunnel URL to the SQLite database."""
    colab_url = body.get("colab_url", "").strip()
    db_service.set_setting("colab_url", colab_url)
    logger.info("Updated Colab URL to: %s", colab_url)
    return {"status": "ok", "colab_url": colab_url}
