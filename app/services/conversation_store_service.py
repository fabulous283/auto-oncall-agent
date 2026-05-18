import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from loguru import logger

from app.config import config
from app.models.conversation import (
    ConversationMessage,
    ConversationSession,
    ConversationSummary,
)


class ConversationStoreService:
    def __init__(self):
        self.enabled = config.conversation_persistence_enabled
        self.db_path = Path(config.conversation_db_path)
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None

        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._initialize_tables()
            logger.info(f"会话持久化 SQLite 初始化完成: {self.db_path}")

    def _initialize_tables(self):
        if self._connection is None:
            return

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_session_message_index
                ON messages(session_id, message_index)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    last_message_index INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            self._connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    def _ensure_enabled(self) -> bool:
        return self.enabled and self._connection is not None

    def upsert_session(self, session_id: str, status: str = "active"):
        if not self._ensure_enabled():
            return

        now = self._now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO sessions (session_id, created_at, updated_at, status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    status=excluded.status
                """,
                (session_id, now, now, status),
            )
            self._connection.commit()

    def get_message_count(self, session_id: str) -> int:
        if not self._ensure_enabled():
            return 0

        cursor = self._connection.execute(
            "SELECT COUNT(*) AS count FROM messages WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return int(row["count"]) if row else 0

    def append_message(self, session_id: str, role: str, content: str) -> ConversationMessage | None:
        if not self._ensure_enabled() or not content:
            return None

        self.upsert_session(session_id)
        now = self._now()

        with self._lock:
            cursor = self._connection.execute(
                "SELECT COALESCE(MAX(message_index), 0) AS max_idx FROM messages WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            next_index = int(row["max_idx"]) + 1 if row else 1

            self._connection.execute(
                """
                INSERT INTO messages (session_id, role, content, message_index, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, next_index, now),
            )
            self._connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            self._connection.commit()

        return ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            message_index=next_index,
            created_at=now,
        )

    def get_recent_messages(self, session_id: str, limit: int) -> list[ConversationMessage]:
        if not self._ensure_enabled():
            return []

        cursor = self._connection.execute(
            """
            SELECT id, session_id, role, content, message_index, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY message_index DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = cursor.fetchall()
        messages = [self._row_to_message(row) for row in rows]
        return list(reversed(messages))

    def get_messages(self, session_id: str) -> list[ConversationMessage]:
        if not self._ensure_enabled():
            return []

        cursor = self._connection.execute(
            """
            SELECT id, session_id, role, content, message_index, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY message_index ASC
            """,
            (session_id,),
        )
        return [self._row_to_message(row) for row in cursor.fetchall()]

    def get_messages_between(
        self,
        session_id: str,
        start_index: int,
        end_index: int,
    ) -> list[ConversationMessage]:
        if not self._ensure_enabled() or end_index < start_index:
            return []

        cursor = self._connection.execute(
            """
            SELECT id, session_id, role, content, message_index, created_at
            FROM messages
            WHERE session_id = ? AND message_index BETWEEN ? AND ?
            ORDER BY message_index ASC
            """,
            (session_id, start_index, end_index),
        )
        return [self._row_to_message(row) for row in cursor.fetchall()]

    def get_session(self, session_id: str) -> ConversationSession | None:
        if not self._ensure_enabled():
            return None

        cursor = self._connection.execute(
            "SELECT session_id, created_at, updated_at, status FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return ConversationSession(**dict(row))

    def get_summary(self, session_id: str) -> ConversationSummary | None:
        if not self._ensure_enabled():
            return None

        cursor = self._connection.execute(
            """
            SELECT session_id, summary, last_message_index, updated_at
            FROM session_summaries
            WHERE session_id = ?
            """,
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return ConversationSummary(**dict(row))

    def upsert_summary(self, session_id: str, summary: str, last_message_index: int):
        if not self._ensure_enabled() or not summary:
            return

        now = self._now()
        self.upsert_session(session_id)
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO session_summaries (session_id, summary, last_message_index, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary=excluded.summary,
                    last_message_index=excluded.last_message_index,
                    updated_at=excluded.updated_at
                """,
                (session_id, summary, last_message_index, now),
            )
            self._connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            self._connection.commit()

    def clear_session(self, session_id: str):
        if not self._ensure_enabled():
            return

        with self._lock:
            self._connection.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
            self._connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._connection.commit()

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> ConversationMessage:
        return ConversationMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            message_index=row["message_index"],
            created_at=row["created_at"],
        )


conversation_store_service = ConversationStoreService()
