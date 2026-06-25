"""FastAPI application entry point.

Run locally:
    uvicorn backend.main:app --reload

Run on Colab (from repo root):
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import sys

# Set SelectorEventLoop on Windows to avoid WinError 10014 proactor accept bugs
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.models.schemas import HealthResponse
from backend.routes import chat, rag, settings as settings_router

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings.ensure_dirs()
    from backend.services.db_service import db_service
    db_service.init_db()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="", tags=["chat"])
    app.include_router(rag.router, prefix="", tags=["rag"])
    app.include_router(settings_router.router, prefix="/settings", tags=["settings"])

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        """Liveness/readiness probe — always fast, no model call."""
        from backend.services.model_service import model_service

        return HealthResponse(
            status="ok",
            app=settings.APP_NAME,
            version=settings.APP_VERSION,
            model=settings.MODEL_NAME,
            model_loaded=model_service.is_loaded,
            use_drive=settings.USE_DRIVE,
        )

    logger.info("App created — %s v%s", settings.APP_NAME, settings.APP_VERSION)
    return app


app = create_app()
