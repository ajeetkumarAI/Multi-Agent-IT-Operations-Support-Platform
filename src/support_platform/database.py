"""SQLite-backed knowledge storage for support artifacts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import KnowledgeArtifact


class KnowledgeRepository:
    """Stores and retrieves validated knowledge artifacts by intent."""

    def __init__(self, database_path: str = ":memory:") -> None:
        self.database_path = database_path
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_artifacts (
                intent TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def save(self, intent: str, artifact: KnowledgeArtifact) -> None:
        self.connection.execute(
            """
            INSERT INTO knowledge_artifacts (intent, title, source, summary)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                intent = excluded.intent,
                title = excluded.title,
                summary = excluded.summary
            """,
            (intent, artifact.title, artifact.source, artifact.summary),
        )
        self.connection.commit()

    def search(self, intent: str) -> list[KnowledgeArtifact]:
        rows = self.connection.execute(
            """
            SELECT title, source, summary
            FROM knowledge_artifacts
            WHERE intent = ?
            ORDER BY source
            """,
            (intent,),
        ).fetchall()
        return [KnowledgeArtifact(row["title"], row["source"], row["summary"]) for row in rows]

    def close(self) -> None:
        self.connection.close()
