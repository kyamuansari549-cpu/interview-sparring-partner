"""SQLite session store for the Interview Sparring Partner.

Plain sqlite3, typed. Persists mock-interview sessions and the candidate's
recorded answers (with LLM evaluations) per session.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from .config import DB_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """SQLite-backed store for sessions and answers."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or DB_PATH
        import os

        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                questions_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                qid TEXT NOT NULL,
                transcript TEXT NOT NULL,
                evaluation_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
            """
        )
        self._conn.commit()

    def create_session(self, profile: dict, questions: list[dict]) -> str:
        """Create a session and return its id (uuid4 hex)."""
        sid = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO sessions (id, profile_json, questions_json, created_at)"
            " VALUES (?, ?, ?, ?)",
            (sid, json.dumps(profile), json.dumps(questions), _now_iso()),
        )
        self._conn.commit()
        return sid

    def get_session(self, sid: str) -> dict | None:
        """Return {"id", "profile", "questions", "created_at"} or None."""
        row = self._conn.execute(
            "SELECT id, profile_json, questions_json, created_at"
            " FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "profile": json.loads(row["profile_json"]),
            "questions": json.loads(row["questions_json"]),
            "created_at": row["created_at"],
        }

    def save_answer(
        self, sid: str, qid: str, transcript: str, evaluation: dict
    ) -> None:
        """Persist one answered question for a session."""
        self._conn.execute(
            "INSERT INTO answers (session_id, qid, transcript,"
            " evaluation_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (sid, qid, transcript, json.dumps(evaluation), _now_iso()),
        )
        self._conn.commit()

    def get_answers(self, sid: str) -> list[dict]:
        """Return answers for a session, ordered by id (oldest first)."""
        rows = self._conn.execute(
            "SELECT qid, transcript, evaluation_json, created_at"
            " FROM answers WHERE session_id = ? ORDER BY id ASC",
            (sid,),
        ).fetchall()
        return [
            {
                "qid": row["qid"],
                "transcript": row["transcript"],
                "evaluation": json.loads(row["evaluation_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()
