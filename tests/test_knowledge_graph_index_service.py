from langchain_core.documents import Document

from app.models.knowledge_graph import (
    KnowledgeGraphEntity,
    KnowledgeGraphExtractionResult,
    KnowledgeGraphRelation,
)
from app.services.knowledge_graph_index_service import KnowledgeGraphIndexService


async def test_index_documents_adds_chunk_metadata_and_saves(monkeypatch):
    monkeypatch.setattr(
        "app.services.knowledge_graph_index_service.config.knowledge_graph_enabled",
        True,
    )
    service = KnowledgeGraphIndexService()
    docs = [Document(page_content="CPU 高时查看 top")]
    deleted_sources = []
    saved_entities = []
    saved_relations = []

    class FakeStore:
        def delete_by_source(self, source_file):
            deleted_sources.append(source_file)

        def upsert_entities(self, entities):
            saved_entities.extend(entities)

        def upsert_relations(self, relations):
            saved_relations.extend(relations)

    class FakeExtractor:
        async def extract_from_document(self, document, source_file, chunk_index, chunk_id):
            return KnowledgeGraphExtractionResult(
                entities=[
                    KnowledgeGraphEntity(
                        id="cpu_high_usage",
                        name="CPU 使用率高",
                        type="故障",
                        description="CPU 高",
                    )
                ],
                relations=[
                    KnowledgeGraphRelation(
                        source="cpu_high_usage",
                        target="cpu_high_usage",
                        relation="关联",
                        description="测试",
                        evidence="CPU 高",
                        source_file=source_file,
                        chunk_index=chunk_index,
                        chunk_id=chunk_id,
                    )
                ],
            )

    service.store = FakeStore()
    service.extractor = FakeExtractor()

    result = await service.index_documents("cpu.md", docs)

    assert deleted_sources == ["cpu.md"]
    assert result["entities"] == 1
    assert result["relations"] == 1
    assert docs[0].metadata["_chunk_index"] == 1
    assert docs[0].metadata["_chunk_id"]
    assert saved_entities[0].id == "cpu_high_usage"
    assert saved_relations[0].chunk_id == docs[0].metadata["_chunk_id"]
