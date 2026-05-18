from app.models.retrieval import QueryRewriteResult
from app.services.rag_query_router_service import RagQueryRouterService


def test_router_defaults_to_vector():
    decision = RagQueryRouterService().route(
        QueryRewriteResult(original_query="你好", rewritten_query="你好")
    )

    assert decision.routes[0] == "vector"


def test_router_enables_multi_query_graph_and_keyword():
    rewrite = QueryRewriteResult(
        original_query="CPU 告警怎么排查 RAG_TOP_K",
        rewritten_query="CPU 告警排查和 RAG_TOP_K 配置",
        keywords=["CPU 告警", "RAG_TOP_K"],
        sub_questions=["CPU 告警原因是什么？"],
    )

    decision = RagQueryRouterService().route(rewrite)

    assert "vector" in decision.routes
    assert "multi_query_vector" in decision.routes
    assert "graph" in decision.routes
    assert "keyword" in decision.routes
