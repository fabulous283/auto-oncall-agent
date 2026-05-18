"""Vector store manager wrapping LangChain Milvus operations."""

from typing import List
import time
import uuid

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


COLLECTION_NAME = "biz"


class VectorStoreManager:
    """Manage the LangChain Milvus vector store used by RAG."""

    def __init__(self):
        self.vector_store: Milvus | None = None
        self.collection_name = COLLECTION_NAME

    def _ensure_initialized(self) -> None:
        """Initialize lazily so app import/startup does not hang on VectorStore creation."""
        if self.vector_store is None:
            self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        try:
            milvus_manager.connect()
            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,
                drop_old=False,
                text_field="content",
                vector_field="vector",
                primary_field="id",
                metadata_field="metadata",
            )

            logger.info(
                f"VectorStore initialized: {config.milvus_host}:{config.milvus_port}, "
                f"collection={self.collection_name}"
            )
        except Exception as e:
            logger.error(f"VectorStore initialization failed: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        self._ensure_initialized()
        assert self.vector_store is not None

        ids = [str(uuid.uuid4()) for _ in documents]
        start_time = time.time()
        result_ids = self.vector_store.add_documents(documents, ids=ids)
        elapsed = time.time() - start_time

        logger.info(
            f"Added {len(documents)} documents to VectorStore in {elapsed:.2f}s"
        )
        return result_ids

    def delete_by_source(self, file_path: str) -> int:
        try:
            self._ensure_initialized()
            collection = milvus_manager.get_collection()
            expr = f'metadata["_source"] == "{file_path}"'
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            logger.info(f"Deleted old vectors for {file_path}, count={deleted_count}")
            return deleted_count
        except Exception as e:
            logger.warning(f"Delete by source skipped for {file_path}: {e}")
            return 0

    def get_vector_store(self) -> Milvus:
        self._ensure_initialized()
        assert self.vector_store is not None
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        try:
            self._ensure_initialized()
            assert self.vector_store is not None
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"Similarity search finished for query='{query}', results={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []


vector_store_manager = VectorStoreManager()
