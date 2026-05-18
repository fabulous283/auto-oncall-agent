"""SQLite store for lightweight knowledge graph data."""

import hashlib
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from loguru import logger

from app.config import config
from app.models.knowledge_graph import KnowledgeGraphEntity, KnowledgeGraphRelation


class KnowledgeGraphStoreService:
    def __init__(self):
        self.enabled = config.knowledge_graph_enabled
        self.db_path = Path(config.knowledge_graph_db_path)
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None

        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._initialize_tables()
            logger.info(f"知识图谱 SQLite 初始化完成: {self.db_path}")

    def _initialize_tables(self):
        if self._connection is None:
            return

        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS kg_entities (
                    entity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS kg_relations (
                    relation_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(source_id, target_id, relation, source_file, chunk_index)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kg_relations_source_file
                ON kg_relations(source_file)
                """
            )
            self._connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _normalize_name(name: str) -> str:
        return "".join(name.lower().split())

    @staticmethod
    def build_relation_id(relation: KnowledgeGraphRelation) -> str:
        raw = "|".join(
            [
                relation.source,
                relation.target,
                relation.relation,
                relation.source_file,
                str(relation.chunk_index),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _ensure_enabled(self) -> bool:
        return self.enabled and self._connection is not None

    def delete_by_source(self, source_file: str):
        if not self._ensure_enabled():
            return

        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM kg_relations WHERE source_file = ?",
                (source_file,),
            )
            self._connection.commit()
        logger.info(f"已删除旧知识图谱关系: source={source_file}, count={cursor.rowcount}")

    def upsert_entities(self, entities: list[KnowledgeGraphEntity]):
        if not self._ensure_enabled() or not entities:
            return

        now = self._now()
        with self._lock:
            for entity in entities:
                self._connection.execute(
                    """
                    INSERT INTO kg_entities (
                        entity_id, name, type, description, normalized_name, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_id) DO UPDATE SET
                        name=excluded.name,
                        type=excluded.type,
                        description=excluded.description,
                        normalized_name=excluded.normalized_name,
                        updated_at=excluded.updated_at
                    """,
                    (
                        entity.id,
                        entity.name,
                        entity.type,
                        entity.description,
                        self._normalize_name(entity.name),
                        now,
                        now,
                    ),
                )
            self._connection.commit()

    def upsert_relations(self, relations: list[KnowledgeGraphRelation]):
        if not self._ensure_enabled() or not relations:
            return

        now = self._now()
        with self._lock:
            for relation in relations:
                self._connection.execute(
                    """
                    INSERT INTO kg_relations (
                        relation_id, source_id, target_id, relation, description, evidence,
                        source_file, chunk_index, chunk_id, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, relation, source_file, chunk_index)
                    DO UPDATE SET
                        description=excluded.description,
                        evidence=excluded.evidence,
                        chunk_id=excluded.chunk_id
                    """,
                    (
                        self.build_relation_id(relation),
                        relation.source,
                        relation.target,
                        relation.relation,
                        relation.description,
                        relation.evidence,
                        relation.source_file,
                        relation.chunk_index,
                        relation.chunk_id,
                        now,
                    ),
                )
            self._connection.commit()

    def search_relations(self, keyword: str, limit: int = 20) -> list[dict]:
        if not self._ensure_enabled() or not keyword:
            return []

        pattern = f"%{keyword}%"
        cursor = self._connection.execute(
            """
            SELECT r.*, s.name AS source_name, t.name AS target_name
            FROM kg_relations r
            LEFT JOIN kg_entities s ON r.source_id = s.entity_id
            LEFT JOIN kg_entities t ON r.target_id = t.entity_id
            WHERE s.name LIKE ?
               OR t.name LIKE ?
               OR r.relation LIKE ?
               OR r.description LIKE ?
               OR r.evidence LIKE ?
            LIMIT ?
            """,
            (pattern, pattern, pattern, pattern, pattern, limit),
        )
        return [dict(row) for row in cursor.fetchall()]


knowledge_graph_store_service = KnowledgeGraphStoreService()
