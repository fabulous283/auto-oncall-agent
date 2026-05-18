"""Hybrid retrieval orchestration for RAG."""

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.models.retrieval import QueryRewriteResult, RetrievedDocument
from app.services.chunk_index_store_service import chunk_index_store_service
from app.services.knowledge_graph_store_service import knowledge_graph_store_service
from app.services.query_rewrite_service import query_rewrite_service
from app.services.rag_query_router_service import rag_query_router_service
from app.services.rerank_service import rerank_service
from app.services.vector_store_manager import vector_store_manager


class HybridRetrievalService:
    source_weights = {
        "vector": 1.0,
        "multi_query_vector": 0.8,
        "keyword": 0.7,
        "graph": 0.6,
    }

    def retrieve(self, query: str) -> list[Document]:
        if not config.hybrid_retrieval_enabled:
            return self._legacy_vector_retrieve(query)

        rewrite = self._run_async(query_rewrite_service.rewrite(query))
        route = rag_query_router_service.route(rewrite)
        logger.info(f"RAG 内部路由: routes={route.routes}, reason={route.reason}")

        candidates: list[RetrievedDocument] = []
        if "vector" in route.routes:
            candidates.extend(self._vector_retrieve(rewrite.rewritten_query, "vector"))
        if "multi_query_vector" in route.routes:
            candidates.extend(self._multi_query_vector_retrieve(rewrite))
        if "keyword" in route.routes and config.keyword_retrieval_enabled:
            candidates.extend(self._keyword_retrieve(rewrite))
        if "graph" in route.routes and config.graph_retrieval_enabled:
            candidates.extend(self._graph_retrieve(rewrite))

        fused = self._fuse_and_deduplicate(candidates)
        limited = fused[: config.hybrid_candidate_top_k]
        docs = [self._to_document(item) for item in limited]
        if not docs:
            return []

        rerank_query = rewrite.rewritten_query or query
        reranked_docs = rerank_service.rerank(
            query=rerank_query,
            documents=docs,
            top_k=config.rag_rerank_top_k,
        )
        logger.info(
            f"混合检索完成: candidates={len(candidates)}, fused={len(fused)}, "
            f"reranked={len(reranked_docs)}"
        )
        return reranked_docs

    def _legacy_vector_retrieve(self, query: str) -> list[Document]:
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": config.rag_recall_top_k})
        recalled_docs = retriever.invoke(query)
        if not recalled_docs:
            return []
        return rerank_service.rerank(
            query=query,
            documents=recalled_docs,
            top_k=config.rag_rerank_top_k,
        )

    def _vector_retrieve(self, query: str, source: str) -> list[RetrievedDocument]:
        if not query:
            return []
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": config.rag_recall_top_k})
        docs = retriever.invoke(query)
        return [self._from_document(doc, source, 1.0) for doc in docs]

    def _multi_query_vector_retrieve(self, rewrite: QueryRewriteResult) -> list[RetrievedDocument]:
        queries = [rewrite.rewritten_query, *rewrite.sub_questions]
        results: list[RetrievedDocument] = []
        seen_queries: set[str] = set()
        for query in queries:
            normalized = query.strip()
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            results.extend(self._vector_retrieve(normalized, "multi_query_vector"))
        return results

    def _keyword_retrieve(self, rewrite: QueryRewriteResult) -> list[RetrievedDocument]:
        keywords = rewrite.keywords or [rewrite.rewritten_query, rewrite.original_query]
        return chunk_index_store_service.search(keywords, config.keyword_top_k)

    def _graph_retrieve(self, rewrite: QueryRewriteResult) -> list[RetrievedDocument]:
        queries = [rewrite.rewritten_query, *rewrite.keywords]
        rows: list[dict] = []
        seen_relation_ids: set[str] = set()
        for query in queries:
            if not query:
                continue
            for row in knowledge_graph_store_service.search_relations(query, config.graph_top_k):
                relation_id = str(row.get("relation_id") or "")
                if relation_id and relation_id in seen_relation_ids:
                    continue
                if relation_id:
                    seen_relation_ids.add(relation_id)
                rows.append(row)
                if len(rows) >= config.graph_top_k:
                    break
            if len(rows) >= config.graph_top_k:
                break

        return [self._graph_row_to_retrieved(row) for row in rows]

    def _graph_row_to_retrieved(self, row: dict) -> RetrievedDocument:
        source_name = row.get("source_name") or row.get("source_id") or "未知实体"
        target_name = row.get("target_name") or row.get("target_id") or "未知实体"
        relation = row.get("relation") or "关联"
        evidence = row.get("evidence") or ""
        description = row.get("description") or ""
        content = (
            "【知识图谱关系】\n"
            f"{source_name} --{relation}--> {target_name}\n"
            f"说明: {description}\n"
            f"依据: {evidence}"
        )
        metadata = {
            "_retrieval_source": "graph",
            "_source": row.get("source_file", ""),
            "_file_name": row.get("source_file", ""),
            "_chunk_index": row.get("chunk_index"),
            "_chunk_id": row.get("chunk_id", ""),
            "relation_id": row.get("relation_id", ""),
            "source_entity": source_name,
            "target_entity": target_name,
            "relation": relation,
        }
        return RetrievedDocument(
            content=content,
            metadata=metadata,
            source="graph",
            score=1.0,
            chunk_id=str(row.get("chunk_id") or row.get("relation_id") or ""),
            source_file=str(row.get("source_file") or ""),
            chunk_index=row.get("chunk_index"),
            sources=["graph"],
        )

    def _fuse_and_deduplicate(self, candidates: Iterable[RetrievedDocument]) -> list[RetrievedDocument]:
        fused: dict[str, RetrievedDocument] = {}
        for item in candidates:
            key = self._dedupe_key(item)
            weight = self.source_weights.get(item.source, 0.5)
            if key not in fused:
                item.sources = item.sources or [item.source]
                item.score = item.score + weight
                fused[key] = item
                continue

            existing = fused[key]
            existing.score += item.score + weight
            for source in item.sources or [item.source]:
                if source not in existing.sources:
                    existing.sources.append(source)
            existing.metadata["retrieval_sources"] = existing.sources

        return sorted(fused.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def _dedupe_key(item: RetrievedDocument) -> str:
        if item.chunk_id:
            return f"chunk:{item.chunk_id}"
        if item.source_file and item.chunk_index is not None:
            return f"source:{item.source_file}:{item.chunk_index}"
        return "hash:" + hashlib.sha256(item.content.encode("utf-8")).hexdigest()

    @staticmethod
    def _from_document(doc: Document, source: str, score: float) -> RetrievedDocument:
        metadata = dict(doc.metadata)
        return RetrievedDocument(
            content=doc.page_content,
            metadata=metadata,
            source=source,
            score=score,
            chunk_id=str(metadata.get("_chunk_id", "")),
            source_file=str(metadata.get("_source", "")),
            chunk_index=metadata.get("_chunk_index"),
            sources=[source],
        )

    @staticmethod
    def _to_document(item: RetrievedDocument) -> Document:
        metadata = dict(item.metadata)
        metadata["_retrieval_source"] = item.source
        metadata["retrieval_sources"] = item.sources or [item.source]
        metadata["hybrid_score"] = item.score
        if item.chunk_id:
            metadata["_chunk_id"] = item.chunk_id
        if item.source_file:
            metadata["_source"] = item.source_file
        if item.chunk_index is not None:
            metadata["_chunk_index"] = item.chunk_index
        return Document(page_content=item.content, metadata=metadata)

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()


hybrid_retrieval_service = HybridRetrievalService()
