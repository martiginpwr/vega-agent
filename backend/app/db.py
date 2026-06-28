import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.config import settings


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


class Database:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    selected_model TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    archived_at TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    model TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                    ON messages(conversation_id, created_at);

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('active', 'suggested', 'rejected', 'archived')),
                    confidence REAL,
                    importance REAL,
                    rationale TEXT,
                    source_conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS memory_sources (
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                    PRIMARY KEY (memory_id, message_id)
                );

                CREATE TABLE IF NOT EXISTS memory_jobs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                    embedding_model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    user_message_id TEXT,
                    assistant_message_id TEXT,
                    model TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS trace_events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trace_events_run_created
                    ON trace_events(run_id, created_at);
                """
            )

    def create_conversation(self, *, title: str = "New chat", selected_model: str | None = None) -> dict[str, Any]:
        now = utc_now()
        conversation_id = new_id()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, title, selected_model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, title, selected_model, now, now),
            )
        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.id = ? AND c.archived_at IS NULL
                GROUP BY c.id
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return row_to_dict(row)

    def list_conversations(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.archived_at IS NULL
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        return [row_to_dict(row) for row in rows]

    def update_conversation_model(self, conversation_id: str, selected_model: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE conversations SET selected_model = ?, updated_at = ? WHERE id = ?",
                (selected_model, utc_now(), conversation_id),
            )

    def maybe_title_from_message(self, conversation_id: str, content: str) -> None:
        title = content.strip().replace("\n", " ")
        if len(title) > 46:
            title = f"{title[:43]}..."
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT c.title, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.id = ?
                GROUP BY c.id
                """,
                (conversation_id,),
            ).fetchone()
            if row and row["title"] == "New chat" and row["message_count"] <= 1:
                connection.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (title or "New chat", utc_now(), conversation_id),
                )

    def delete_conversation(self, conversation_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        message_id = new_id()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, model, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    model,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return self.get_message(message_id)

    def get_message(self, message_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        if row is None:
            raise KeyError(message_id)
        return row_to_dict(row)

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()
        messages = []
        for row in rows:
            message = row_to_dict(row)
            message["metadata"] = json.loads(message.pop("metadata_json") or "{}")
            messages.append(message)
        return messages

    def recent_messages(self, conversation_id: str, limit: int = 12) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return list(reversed([row_to_dict(row) for row in rows]))

    def list_memories(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE status IN ('active', 'suggested')
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [row_to_dict(row) for row in rows]

    def list_memories_missing_embedding(self, *, embedding_model: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT m.*
                FROM memories m
                LEFT JOIN memory_embeddings e
                    ON e.memory_id = m.id AND e.embedding_model = ?
                WHERE m.status IN ('active', 'suggested') AND e.memory_id IS NULL
                ORDER BY m.updated_at DESC
                LIMIT ?
                """,
                (embedding_model, limit),
            ).fetchall()
        return [row_to_dict(row) for row in rows]

    def mark_memories_used(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE memories SET last_used_at = ? WHERE id IN ({placeholders})",
                (utc_now(), *memory_ids),
            )

    def add_memory_embedding(self, *, memory_id: str, embedding_model: str, vector: list[float]) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO memory_embeddings
                    (memory_id, embedding_model, vector_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, embedding_model, json.dumps(vector), utc_now()),
            )

    def find_similar_memories(
        self,
        *,
        vector: list[float],
        embedding_model: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT m.*, e.vector_json
                FROM memories m
                JOIN memory_embeddings e ON e.memory_id = m.id
                WHERE m.status IN ('active', 'suggested') AND e.embedding_model = ?
                """,
                (embedding_model,),
            ).fetchall()

        scored = []
        for row in rows:
            memory = row_to_dict(row)
            other_vector = json.loads(memory.pop("vector_json"))
            score = cosine_similarity(vector, other_vector)
            memory["similarity"] = score
            scored.append(memory)

        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:limit]

    def add_memory(
        self,
        *,
        memory_type: str,
        content: str,
        status: str,
        confidence: float | None,
        importance: float | None,
        rationale: str | None,
        source_conversation_id: str,
        source_message_ids: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        memory_id = new_id()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO memories (
                    id, type, content, status, confidence, importance, rationale,
                    source_conversation_id, created_at, updated_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    memory_type,
                    content,
                    status,
                    confidence,
                    importance,
                    rationale,
                    source_conversation_id,
                    now,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
            for message_id in source_message_ids:
                connection.execute(
                    "INSERT OR IGNORE INTO memory_sources (memory_id, message_id) VALUES (?, ?)",
                    (memory_id, message_id),
                )
        return self.get_memory(memory_id)

    def get_memory(self, memory_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return row_to_dict(row)

    def create_memory_job(self, conversation_id: str) -> str:
        job_id = new_id()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_jobs (id, conversation_id, status, created_at)
                VALUES (?, ?, 'queued', ?)
                """,
                (job_id, conversation_id, utc_now()),
            )
        return job_id

    def complete_memory_job(self, job_id: str, *, status: str, error: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE memory_jobs
                SET status = ?, error = ?, completed_at = ?
                WHERE id = ?
                """,
                (status, error, utc_now(), job_id),
            )

    def create_agent_run(self, *, conversation_id: str, model: str) -> dict[str, Any]:
        run_id = new_id()
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (id, conversation_id, model, status, created_at)
                VALUES (?, ?, ?, 'started', ?)
                """,
                (run_id, conversation_id, model, now),
            )
        return self.get_agent_run(run_id)

    def update_agent_run(
        self,
        run_id: str,
        *,
        status: str,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        error: str | None = None,
    ) -> None:
        completed_at = utc_now() if status in {"completed", "failed"} else None
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET status = ?,
                    user_message_id = COALESCE(?, user_message_id),
                    assistant_message_id = COALESCE(?, assistant_message_id),
                    error = COALESCE(?, error),
                    completed_at = COALESCE(?, completed_at)
                WHERE id = ?
                """,
                (status, user_message_id, assistant_message_id, error, completed_at, run_id),
            )

    def add_trace_event(
        self,
        *,
        run_id: str,
        step: str,
        status: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_id = new_id()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO trace_events (id, run_id, step, status, message, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, run_id, step, status, message, json.dumps(metadata or {}), utc_now()),
            )
        return self.get_trace_event(event_id)

    def get_trace_event(self, event_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM trace_events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        event = row_to_dict(row)
        event["metadata"] = json.loads(event.pop("metadata_json") or "{}")
        return event

    def get_agent_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return row_to_dict(row)

    def get_trace(self, run_id: str) -> dict[str, Any]:
        run = self.get_agent_run(run_id)
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM trace_events
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
        events = []
        for row in rows:
            event = row_to_dict(row)
            event["metadata"] = json.loads(event.pop("metadata_json") or "{}")
            events.append(event)
        return {"run": run, "events": events}


database = Database(settings.vega_database_path)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
