import httpx
import pytest
from langchain_core.documents import Document

from app.services.rerank_service import DashScopeRerankService


def test_rerank_reorders_documents_and_writes_metadata(monkeypatch):
    service = DashScopeRerankService(
        api_key="test-key",
        model="qwen3-rerank",
        base_url="https://example.com/compatible-api/v1",
    )
    docs = [
        Document(page_content="CPU 告警处理步骤", metadata={"source": "cpu.md"}),
        Document(page_content="磁盘告警处理步骤", metadata={"source": "disk.md"}),
        Document(page_content="内存告警处理步骤", metadata={"source": "memory.md"}),
    ]

    def fake_post(url, headers, json, timeout):
        assert url == "https://example.com/compatible-api/v1/reranks"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "qwen3-rerank"
        assert json["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.93},
                    {"index": 0, "relevance_score": 0.81},
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    reranked = service.rerank("内存高怎么排查", docs, top_k=2)

    assert [doc.page_content for doc in reranked] == [
        "内存告警处理步骤",
        "CPU 告警处理步骤",
    ]
    assert reranked[0].metadata["rerank_score"] == 0.93
    assert reranked[0].metadata["rerank_model"] == "qwen3-rerank"
    assert reranked[0].metadata["rerank_original_index"] == 2


def test_rerank_raises_when_api_key_missing():
    service = DashScopeRerankService(api_key="")

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        service.rerank("问题", [Document(page_content="文档")], top_k=1)


def test_rerank_does_not_fallback_when_http_fails(monkeypatch):
    service = DashScopeRerankService(api_key="test-key")

    def fake_post(url, headers, json, timeout):
        request = httpx.Request("POST", url)
        response = httpx.Response(500, request=request, text="server error")
        raise httpx.HTTPStatusError("server error", request=request, response=response)

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="百炼重排接口调用失败"):
        service.rerank("问题", [Document(page_content="文档")], top_k=1)

