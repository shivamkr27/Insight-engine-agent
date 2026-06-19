"""
SQLite store for study session metadata.

The actual Q&A state (topics, current question, score) lives in LangGraph's
checkpoints.db. This store tracks lightweight session metadata for listing
past study sessions.
"""

import sqlite3
import threading
from typing import List, Dict

from .logging_config import get_logger

logger = get_logger(__name__)


class StudyStore:
    """
    Tracks per-user study sessions.

    Schema:
        study_thread_id — LangGraph thread_id for this study session
        user_id         — user who owns this session
        doc_name        — document being studied (e.g. "rbi_report.pdf")
        score           — accumulated score (float stored as REAL)
        total           — total questions answered
        status          — "active" | "completed"
        created_at      — ISO-8601 UTC timestamp
        completed_at    — set when session finishes
    """

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS study_sessions (
                study_thread_id TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                doc_name        TEXT NOT NULL,
                score           REAL    DEFAULT 0.0,
                total           INTEGER DEFAULT 0,
                status          TEXT    DEFAULT 'active',
                created_at      TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                completed_at    TEXT
            )
        """)
        self._conn.commit()

    def create_session(self, user_id: str, study_thread_id: str, doc_name: str) -> None:
        """Register a new study session. Idempotent — safe to call twice."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO study_sessions (study_thread_id, user_id, doc_name) VALUES (?, ?, ?)",
                (study_thread_id, user_id, doc_name),
            )
            self._conn.commit()

    def update_score(self, study_thread_id: str, score: float, total: int) -> None:
        """Sync score + total from graph state into the metadata store."""
        with self._lock:
            self._conn.execute(
                "UPDATE study_sessions SET score = ?, total = ? WHERE study_thread_id = ?",
                (score, total, study_thread_id),
            )
            self._conn.commit()

    def complete_session(self, study_thread_id: str) -> None:
        """Mark session as completed and stamp the completion time."""
        with self._lock:
            self._conn.execute(
                "UPDATE study_sessions SET status = 'completed', "
                "completed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
                "WHERE study_thread_id = ?",
                (study_thread_id,),
            )
            self._conn.commit()

    def list_user(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Return recent study sessions for a user, newest first."""
        rows = self._conn.execute(
            "SELECT study_thread_id, doc_name, score, total, status, created_at "
            "FROM study_sessions WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            {
                "study_thread_id": r[0],
                "doc_name":        r[1],
                "score":           r[2],
                "total":           r[3],
                "status":          r[4],
                "created_at":      r[5],
            }
            for r in rows
        ]
