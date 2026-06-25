"""Chat routes — handles inference proxying and SQLite-based history endpoints."""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.models.schemas import ChatRequest, ChatResponse, Role
from backend.services.db_service import db_service
from backend.services.model_service import model_service
from backend.services.rag_service import rag_service

logger = logging.getLogger(__name__)

# Create router to be mounted at the app root '/'
router = APIRouter()


async def _sse_generator(
    chat_id: str,
    request: ChatRequest,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted token chunks, then a final [DONE] frame."""
    context_chunks = []
    if request.use_rag:
        context_chunks = await rag_service.retrieve(
            query=request.message,
            knowledge_base=request.knowledge_base,
            top_k=settings.TOP_K,
        )

    full_reply = ""
    async for token in model_service.stream(
        message=request.message,
        history=request.history,
        context_chunks=context_chunks,
        max_new_tokens=request.max_new_tokens or settings.MAX_NEW_TOKENS,
        temperature=request.temperature or settings.TEMPERATURE,
        top_p=request.top_p or settings.TOP_P,
        repeat_penalty=request.repeat_penalty or settings.REPEAT_PENALTY,
        image=request.image,
    ):
        full_reply += token
        payload = json.dumps({"token": token, "chat_id": chat_id})
        yield f"data: {payload}\n\n"

    # Save user message and assistant message to SQLite
    from backend.services.memory_service import memory_service
    await memory_service.append(
        chat_id=chat_id,
        user_message=request.message,
        assistant_message=full_reply,
        image=request.image,
    )
    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat(request: ChatRequest):
    """Proxy inference request to remote Colab, saving history locally."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded on Colab yet. Ensure Colab server is running and Tunnel URL is correct.",
        )

    chat_id = request.chat_id or str(uuid.uuid4())
    rag_used = False

    # Check if context will be retrieved
    context_chunks = []
    if request.use_rag:
        context_chunks = await rag_service.retrieve(
            query=request.message,
            knowledge_base=request.knowledge_base,
            top_k=settings.TOP_K,
        )
        if context_chunks:
            rag_used = True

    if request.stream:
        return StreamingResponse(
            _sse_generator(chat_id, request),
            media_type="text/event-stream",
            headers={
                "X-Chat-ID": chat_id,
                "X-RAG-Used": str(rag_used).lower(),
            },
        )

    # Non-streaming path
    reply = await model_service.generate(
        message=request.message,
        history=request.history,
        context_chunks=context_chunks,
        max_new_tokens=request.max_new_tokens or settings.MAX_NEW_TOKENS,
        temperature=request.temperature or settings.TEMPERATURE,
        top_p=request.top_p or settings.TOP_P,
        repeat_penalty=request.repeat_penalty or settings.REPEAT_PENALTY,
        image=request.image,
    )

    from backend.services.memory_service import memory_service
    await memory_service.append(
        chat_id=chat_id,
        user_message=request.message,
        assistant_message=reply,
        image=request.image,
    )

    return ChatResponse(
        chat_id=chat_id,
        message=reply,
        role=Role.ASSISTANT,
        sources=context_chunks,
        model=settings.MODEL_LABEL if hasattr(settings, "MODEL_LABEL") else settings.MODEL_NAME,
    )


# ------------------------------------------------------------------
# Chat History Management (SQLite)
# ------------------------------------------------------------------

@router.get("/chats")
async def list_chats():
    """List all saved chat summaries, ordered by mtime."""
    chats = db_service.list_chats()
    return {"chats": chats}


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Retrieve full detail (messages array) of a chat."""
    chat_data = db_service.load_chat(chat_id)
    if not chat_data:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found.")
    return chat_data


@router.post("/chats/save")
async def save_chat(body: dict):
    """Save or update chat metadata and messages."""
    chat_id = body.get("id") or str(uuid.uuid4())
    title = body.get("title", "New Chat")
    messages = body.get("messages", [])
    db_service.save_chat(chat_id, title, messages)
    return {"id": chat_id, "saved": True}


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat session."""
    deleted = db_service.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found.")
    return {"id": chat_id, "deleted": True}


@router.patch("/chats/{chat_id}/rename")
async def rename_chat(chat_id: str, body: dict):
    """Rename an existing chat session."""
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    success = db_service.rename_chat(chat_id, title)
    if not success:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found.")
    return {"id": chat_id, "renamed": True, "title": title}
