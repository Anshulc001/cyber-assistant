"""Memory service — persists chat history as JSON files on Google Drive."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from backend.config import settings
from backend.models.schemas import Message, Role

logger = logging.getLogger(__name__)


def _chat_file(chat_id: str) -> Path:
    return settings.chat_history_dir / f"{chat_id}.json"


class MemoryService:
    """Stores each conversation as a JSON array of {role, content} objects."""

    async def load(self, chat_id: str) -> list[dict] | None:
        path = _chat_file(chat_id)
        if not path.exists():
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: json.loads(path.read_text()))

    async def append(
        self,
        chat_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Add a user+assistant turn to the chat file, creating it if needed."""
        existing = await self.load(chat_id) or []
        existing.append({"role": Role.USER, "content": user_message})
        existing.append({"role": Role.ASSISTANT, "content": assistant_message})

        path = _chat_file(chat_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: path.write_text(json.dumps(existing, indent=2, default=str)),
        )
        logger.debug("Chat '%s' saved (%d turns).", chat_id, len(existing) // 2)

    async def delete(self, chat_id: str) -> bool:
        path = _chat_file(chat_id)
        if not path.exists():
            return False
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, path.unlink)
        logger.info("Chat '%s' deleted.", chat_id)
        return True

    async def list_chats(self) -> list[str]:
        """Return all chat IDs that have persisted history."""
        if not settings.chat_history_dir.exists():
            return []
        return [p.stem for p in settings.chat_history_dir.glob("*.json")]


memory_service = MemoryService()
