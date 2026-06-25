"""SQLite database service to persist chats, messages, and settings locally."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)


class DBService:
    """Manages SQLite database connections and executes CRUD statements."""

    def __init__(self) -> None:
        self.db_path = settings.database_path

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create tables if they do not exist."""
        logger.info("Initializing database at %s", self.db_path)
        with self._connection() as conn:
            # 1. Settings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

            # 2. Chats table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
            """)

            # 3. Messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    rag_used INTEGER DEFAULT 0,
                    sources TEXT,
                    image TEXT,
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );
            """)
            
            # Database Migration: add 'image' column to 'messages' if it's missing from previous runs
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN image TEXT;")
            except sqlite3.OperationalError:
                pass  # Already exists
                
            conn.commit()

    # ------------------------------------------------------------------
    # Settings CRUD
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self._connection() as conn:
            cursor = conn.execute("SELECT value FROM settings WHERE key = ?;", (key,))
            row = cursor.fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?);",
                (key, value),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Chats & Messages CRUD
    # ------------------------------------------------------------------

    def list_chats(self) -> list[dict[str, Any]]:
        """Return all chats sorted by updated_at desc."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC;"
            )
            rows = cursor.fetchall()
            chats = []
            for r in rows:
                # Count messages
                cnt_cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM messages WHERE chat_id = ?;",
                    (r["id"],),
                )
                msg_count = cnt_cursor.fetchone()["count"]
                chats.append({
                    "id": r["id"],
                    "title": r["title"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "message_count": msg_count,
                })
            return chats

    def load_chat(self, chat_id: str) -> dict[str, Any] | None:
        """Return full chat details including all messages."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?;",
                (chat_id,),
            )
            chat_row = cursor.fetchone()
            if not chat_row:
                return None

            msg_cursor = conn.execute(
                "SELECT role, content, timestamp, rag_used, sources, image FROM messages WHERE chat_id = ? ORDER BY timestamp ASC;",
                (chat_id,),
            )
            messages = []
            for mr in msg_cursor.fetchall():
                sources = []
                if mr["sources"]:
                    try:
                        sources = json.loads(mr["sources"])
                    except Exception:
                        pass
                img_val = mr["image"] if "image" in mr.keys() else None
                messages.append({
                    "role": mr["role"],
                    "content": mr["content"],
                    "timestamp": mr["timestamp"],
                    "ragUsed": bool(mr["rag_used"]),
                    "sources": sources,
                    "image": img_val,
                })

            return {
                "id": chat_row["id"],
                "title": chat_row["title"],
                "created_at": chat_row["created_at"],
                "updated_at": chat_row["updated_at"],
                "messages": messages,
            }

    def save_chat(self, chat_id: str, title: str, messages: list[dict[str, Any]]) -> None:
        """Save/update chat metadata and fully sync its messages."""
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connection() as conn:
            # Upsert chat
            cursor = conn.execute("SELECT created_at FROM chats WHERE id = ?;", (chat_id,))
            row = cursor.fetchone()
            created_at = row["created_at"] if row else now

            conn.execute(
                "INSERT OR REPLACE INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
                (chat_id, title, created_at, now),
            )

            # Clear existing messages for this chat
            conn.execute("DELETE FROM messages WHERE chat_id = ?;", (chat_id,))

            # Insert new messages
            for i, msg in enumerate(messages):
                msg_id = f"{chat_id}_{i}"
                role = msg.get("role")
                content = msg.get("content")
                ts = msg.get("timestamp") or now
                rag_used = 1 if msg.get("ragUsed") else 0
                sources_json = json.dumps(msg.get("sources") or [])
                image = msg.get("image")

                conn.execute(
                    """
                    INSERT INTO messages (id, chat_id, role, content, timestamp, rag_used, sources, image)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (msg_id, chat_id, role, content, ts, rag_used, sources_json, image),
                )
            conn.commit()

    def rename_chat(self, chat_id: str, title: str) -> bool:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with self._connection() as conn:
            cursor = conn.execute("SELECT 1 FROM chats WHERE id = ?;", (chat_id,))
            if not cursor.fetchone():
                return False
            conn.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?;",
                (title, now, chat_id),
            )
            conn.commit()
            return True

    def delete_chat(self, chat_id: str) -> bool:
        with self._connection() as conn:
            cursor = conn.execute("SELECT 1 FROM chats WHERE id = ?;", (chat_id,))
            if not cursor.fetchone():
                return False
            # SQLite ON DELETE CASCADE will automatically delete corresponding messages
            conn.execute("DELETE FROM chats WHERE id = ?;", (chat_id,))
            conn.commit()
            return True


db_service = DBService()
