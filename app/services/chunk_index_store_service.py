"""SQLite FTS chunk index for keyword retrieval."""

import json
import sqlite3
import threading
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.models.retrieval import RetrievedDocument


class ChunkIndexStoreService:
    def __init__(self):
        self.db_path = Path(config.chunk_index_db_path)
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None
        if config.keyword_retrieval_enabled or config.hybrid_retrieval_enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._initialize_tables()
            logger.info(f"RAG chunk FTS 索引初始化完成: {self.db_path}")

    def _initialize_tables(self):
        if self._connection is None:
            return

        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts
                USING fts5(chunk_id UNINDEXED, content)
                """
            )
            self._connection.commit()

    def _ensure_enabled(self) -> bool:
        return self._connection is not None

    def delete_by_source(self, source_file: str):
        if not self._ensure_enabled():
            return

        with self._lock:
            rows = self._connection.execute(
                "SELECT chunk_id FROM rag_chunks WHERE source_file = ?",
                (source_file,),
            ).fetchall()
            chunk_ids = [row["chunk_id"] for row in rows]
            self._connection.execute("DELETE FROM rag_chunks WHERE source_file = ?", (source_file,))
            for chunk_id in chunk_ids:
                self._connection.execute(
                    "DELETE FROM rag_chunks_fts WHERE chunk_id = ?",
                    (chunk_id,),
                )
            self._connection.commit()
        logger.info(f"已删除旧 chunk 关键词索引: source={source_file}, count={len(chunk_ids)}")

    def upsert_documents(self, source_file: str, documents: list[Document]):
        if not self._ensure_enabled() or not documents:
            return

        with self._lock:
            for document in documents:
                metadata = dict(document.metadata)
                chunk_id = str(metadata.get("_chunk_id", ""))
                if not chunk_id:
                    continue
                chunk_index = int(metadata.get("_chunk_index") or 0)
                content = document.page_content or ""
                self._connection.execute(
                    """
                    INSERT INTO rag_chunks (chunk_id, source_file, chunk_index, content, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        source_file=excluded.source_file,
                        chunk_index=excluded.chunk_index,
                        content=excluded.content,
                        metadata=excluded.metadata
                    """,
                    (
                        chunk_id,
                        source_file,
                        chunk_index,
                        content,
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )
                self._connection.execute(
                    "DELETE FROM rag_chunks_fts WHERE chunk_id = ?",
                    (chunk_id,),
                )
                self._connection.execute(
                    "INSERT INTO rag_chunks_fts (chunk_id, content) VALUES (?, ?)",
                    (chunk_id, content),
                )
            self._connection.commit()
        logger.info(f"chunk 关键词索引写入完成: source={source_file}, count={len(documents)}")

    def search(self, keywords: list[str], top_k: int) -> list[RetrievedDocument]:
        if not self._ensure_enabled() or not keywords or top_k <= 0:
            return []

        query = " OR ".join(self._escape_fts_term(keyword) for keyword in keywords if keyword.strip())
        if not query:
            return []

        cursor = self._connection.execute(
            """
            SELECT c.chunk_id, c.source_file, c.chunk_index, c.content, c.metadata,
                   bm25(rag_chunks_fts) AS rank
            FROM rag_chunks_fts
            JOIN rag_chunks c ON c.chunk_id = rag_chunks_fts.chunk_id
            WHERE rag_chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, top_k),
        )

        results: list[RetrievedDocument] = []
        for row in cursor.fetchall():
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            score = 1.0 / (1.0 + abs(float(row["rank"] or 0.0)))
            results.append(
                RetrievedDocument(
                    content=row["content"],
                    metadata=metadata,
                    source="keyword",
                    score=score,
                    chunk_id=row["chunk_id"],
                    source_file=row["source_file"],
                    chunk_index=row["chunk_index"],
                    sources=["keyword"],
                )
            )
        return results

    @staticmethod
    def _escape_fts_term(term: str) -> str:
        cleaned = term.replace('"', " ").strip()
        return f'"{cleaned}"'


chunk_index_store_service = ChunkIndexStoreService()
