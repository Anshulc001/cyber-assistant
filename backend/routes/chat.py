"""Chat route — handles streaming and non-streaming inference requests."""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.models.schemas import ChatRequest, ChatResponse, Role
from backend.services.memory_service import memory_service
from backend.services.model_service import model_service
from backend.services.rag_service import rag_service

logger = logging.getLogger(__name__)
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
    ):
        full_reply += token
        payload = json.dumps({"token": token, "chat_id": chat_id})
        yield f"data: {payload}\n\n"

    await memory_service.append(
        chat_id=chat_id,
        user_message=request.message,
        assistant_message=full_reply,
    )
    yield "data: [DONE]\n\n"


@router.post("")
async def chat(request: ChatRequest):
    """Send a message and receive a reply (streaming or full)."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    chat_id = request.chat_id or str(uuid.uuid4())

    if request.stream:
        return StreamingResponse(
            _sse_generator(chat_id, request),
            media_type="text/event-stream",
            headers={"X-Chat-ID": chat_id},
        )

    # Non-streaming path
    context_chunks = []
    if request.use_rag:
        context_chunks = await rag_service.retrieve(
            query=request.message,
            knowledge_base=request.knowledge_base,
            top_k=settings.TOP_K,
        )

    reply = await model_service.generate(
        message=request.message,
        history=request.history,
        context_chunks=context_chunks,
        max_new_tokens=request.max_new_tokens or settings.MAX_NEW_TOKENS,
        temperature=request.temperature or settings.TEMPERATURE,
        top_p=request.top_p or settings.TOP_P,
    )

    await memory_service.append(
        chat_id=chat_id,
        user_message=request.message,
        assistant_message=reply,
    )

    return ChatResponse(
        chat_id=chat_id,
        message=reply,
        role=Role.ASSISTANT,
        sources=context_chunks,
        model=settings.MODEL_NAME,
    )


@router.get("/{chat_id}/history")
async def get_history(chat_id: str):
    """Return full message history for a given chat session."""
    history = await memory_service.load(chat_id)
    if history is None:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found.")
    return {"chat_id": chat_id, "history": history}


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat session from storage."""
    deleted = await memory_service.delete(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found.")
    return {"detail": f"Chat '{chat_id}' deleted."}
