from langchain_core.documents import Document

from app.services.chunk_index_store_service import ChunkIndexStoreService


def test_chunk_index_upsert_search_and_delete(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.chunk_index_store_service.config.chunk_index_db_path", str(tmp_path / "chunks.db"))
    monkeypatch.setattr("app.services.chunk_index_store_service.config.keyword_retrieval_enabled", True)
    monkeypatch.setattr("app.services.chunk_index_store_service.config.hybrid_retrieval_enabled", True)
    store = ChunkIndexStoreService()

    docs = [
        Document(
            page_content="CPU 告警需要查看高占用进程",
            metadata={"_chunk_id": "chunk-1", "_chunk_index": 1, "_source": "cpu.md"},
        )
    ]

    store.upsert_documents("cpu.md", docs)
    results = store.search(["CPU 告警"], 10)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"

    store.delete_by_source("cpu.md")
    assert store.search(["CPU 告警"], 10) == []
