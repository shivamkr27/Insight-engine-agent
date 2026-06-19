"""
Conversation metadata store for the history sidebar.

Stores lightweight per-conversation metadata (title, timestamps, message count).
The actual message state (checkpoints) lives in LangGraph's checkpoints.db.
"""

import sqlite3
import threading
from typing import List, Dict

from .logging_config import get_logger

logger = get_logger(__name__)


class ConversationStore:
    """
    Tracks per-user conversation metadata so the history sidebar
    can list and resume past sessions.

    Schema:
        thread_id     — LangGraph configurable thread_id (session-scoped UUID)
        user_id       — stable identifier from auth layer (username) or "default"
        title         — first user message, truncated to 80 chars
        created_at    — ISO-8601 UTC timestamp
        last_active   — updated on every message
        message_count — incremented on every user turn
    """

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id     TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL DEFAULT 'default',
                title         TEXT NOT NULL DEFAULT 'New conversation',
                created_at    TEXT NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                last_active   TEXT NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                message_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.commit()

    def upsert(self, thread_id: str, user_id: str) -> None:
        """Register a new thread or bump last_active on an existing one."""
        with self._lock:
            self._conn.execute("""
                INSERT INTO conversations (thread_id, user_id)
                VALUES (?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    last_active = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            """, (thread_id, user_id))
            self._conn.commit()

    def set_title(self, thread_id: str, title: str) -> None:
        """Set the title from the first user message (only if still at default)."""
        with self._lock:
            self._conn.execute("""
                UPDATE conversations
                SET title = ?
                WHERE thread_id = ? AND title = 'New conversation'
            """, (title[:80], thread_id))
            self._conn.commit()

    def touch(self, thread_id: str) -> None:
        """Bump last_active and increment message_count for each user turn."""
        with self._lock:
            self._conn.execute("""
                UPDATE conversations
                SET last_active   = strftime('%Y-%m-%dT%H:%M:%SZ', 'now'),
                    message_count = message_count + 1
                WHERE thread_id = ?
            """, (thread_id,))
            self._conn.commit()

    def list_user(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Return the most recent conversations for a user, newest first."""
        rows = self._conn.execute("""
            SELECT thread_id, title, created_at, last_active, message_count
            FROM conversations
            WHERE user_id = ?
            ORDER BY last_active DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
        return [
            {
                "thread_id":     r[0],
                "title":         r[1],
                "created_at":    r[2],
                "last_active":   r[3],
                "message_count": r[4],
            }
            for r in rows
        ]

    def delete(self, thread_id: str, user_id: str) -> bool:
        """Delete a conversation. Returns False if thread not owned by user_id."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM conversations WHERE thread_id = ? AND user_id = ?",
                (thread_id, user_id),
            )
            self._conn.commit()
            return cur.rowcount > 0
