from app.models.retrieval import RetrievedDocument
from app.services.hybrid_retrieval_service import HybridRetrievalService


def test_fuse_and_deduplicate_merges_sources():
    service = HybridRetrievalService()
    items = [
        RetrievedDocument(
            content="CPU 告警内容",
            source="vector",
            score=1.0,
            chunk_id="chunk-1",
            sources=["vector"],
        ),
        RetrievedDocument(
            content="CPU 告警内容",
            source="keyword",
            score=1.0,
            chunk_id="chunk-1",
            sources=["keyword"],
        ),
    ]

    fused = service._fuse_and_deduplicate(items)

    assert len(fused) == 1
    assert set(fused[0].sources) == {"vector", "keyword"}
    assert fused[0].score > 2
