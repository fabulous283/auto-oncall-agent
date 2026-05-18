"""Knowledge graph indexing orchestration."""

import hashlib

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.models.knowledge_graph import KnowledgeGraphExtractionResult
from app.services.knowledge_graph_extractor_service import knowledge_graph_extractor_service
from app.services.knowledge_graph_store_service import knowledge_graph_store_service


class KnowledgeGraphIndexService:
    def __init__(self):
        self.extractor = knowledge_graph_extractor_service
        self.store = knowledge_graph_store_service

    async def index_documents(self, source_file: str, documents: list[Document]) -> dict:
        result = {
            "enabled": config.knowledge_graph_enabled,
            "chunks": len(documents),
            "entities": 0,
            "relations": 0,
            "failed_chunks": 0,
        }
        if not config.knowledge_graph_enabled:
            logger.info("知识图谱入库已关闭，跳过")
            return result
        if not documents:
            return result

        self.store.delete_by_source(source_file)

        for index, document in enumerate(documents, 1):
            chunk_id = self.ensure_chunk_metadata(document, source_file, index)
            try:
                extraction = await self.extractor.extract_from_document(
                    document=document,
                    source_file=source_file,
                    chunk_index=index,
                    chunk_id=chunk_id,
                )
                self._save_extraction(extraction)
                result["entities"] += len(extraction.entities)
                result["relations"] += len(extraction.relations)
            except Exception as exc:
                result["failed_chunks"] += 1
                logger.warning(
                    f"知识图谱分片抽取失败，已跳过: source={source_file}, chunk={index}, error={exc}"
                )

        logger.info(
            f"知识图谱入库完成: source={source_file}, chunks={result['chunks']}, "
            f"entities={result['entities']}, relations={result['relations']}, "
            f"failed_chunks={result['failed_chunks']}"
        )
        return result

    def _save_extraction(self, extraction: KnowledgeGraphExtractionResult):
        self.store.upsert_entities(extraction.entities)
        self.store.upsert_relations(extraction.relations)

    @staticmethod
    def ensure_chunk_metadata(document: Document, source_file: str, chunk_index: int) -> str:
        chunk_id = document.metadata.get("_chunk_id")
        if not chunk_id:
            raw = f"{source_file}|{chunk_index}|{document.page_content[:200]}"
            chunk_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        document.metadata["_source"] = source_file
        document.metadata["_chunk_index"] = chunk_index
        document.metadata["_chunk_id"] = chunk_id
        return chunk_id


knowledge_graph_index_service = KnowledgeGraphIndexService()
