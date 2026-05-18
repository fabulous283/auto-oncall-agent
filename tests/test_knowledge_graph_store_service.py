from app.models.knowledge_graph import KnowledgeGraphEntity, KnowledgeGraphRelation
from app.services.knowledge_graph_store_service import KnowledgeGraphStoreService


def test_store_upserts_and_deletes_by_source(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.knowledge_graph_store_service.config.knowledge_graph_enabled", True)
    monkeypatch.setattr(
        "app.services.knowledge_graph_store_service.config.knowledge_graph_db_path",
        str(tmp_path / "kg.db"),
    )
    store = KnowledgeGraphStoreService()

    entity = KnowledgeGraphEntity(
        id="cpu_high_usage",
        name="CPU 使用率高",
        type="故障",
        description="CPU 高负载",
    )
    relation = KnowledgeGraphRelation(
        source="cpu_high_usage",
        target="cpu_high_usage",
        relation="关联",
        description="自关联测试",
        evidence="CPU 使用率高",
        source_file="cpu.md",
        chunk_index=1,
        chunk_id="chunk-1",
    )

    store.upsert_entities([entity, entity])
    store.upsert_relations([relation, relation])

    rows = store.search_relations("CPU")
    assert len(rows) == 1

    store.delete_by_source("cpu.md")
    assert store.search_relations("CPU") == []
