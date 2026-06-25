"""Memory service — persists chat history via db_service SQLite storage."""

from __future__ import annotations

import logging
from backend.services.db_service import db_service

logger = logging.getLogger(__name__)


class MemoryService:
    """Delegates to db_service for SQLite persistence."""

    async def load(self, chat_id: str) -> list[dict] | None:
        chat = db_service.load_chat(chat_id)
        if not chat:
            return None
        return chat["messages"]

    async def append(
        self,
        chat_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Add user+assistant turn to the database, auto-generating title if needed."""
        chat = db_service.load_chat(chat_id)
        existing = chat["messages"] if chat else []
        existing.append({"role": "user", "content": user_message})
        existing.append({"role": "assistant", "content": assistant_message})

        # Determine title
        title = chat["title"] if chat else "New Chat"
        if not chat or title == "New Chat":
            first_user = next((m for m in existing if m["role"] == "user"), None)
            if first_user:
                title = first_user["content"][:55]
                if len(first_user["content"]) > 55:
                    title += "…"

        db_service.save_chat(chat_id, title, existing)

    async def delete(self, chat_id: str) -> bool:
        return db_service.delete_chat(chat_id)

    async def list_chats(self) -> list[str]:
        """Return list of all chat IDs."""
        return [c["id"] for c in db_service.list_chats()]


memory_service = MemoryService()
