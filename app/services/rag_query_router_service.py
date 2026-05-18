"""Internal RAG route decision service."""

import re

from app.config import config
from app.models.retrieval import QueryRewriteResult, RagRouteDecision


class RagQueryRouterService:
    graph_keywords = (
        "故障",
        "告警",
        "指标",
        "排查",
        "原因",
        "解决",
        "监控",
        "异常",
        "不可用",
        "高负载",
        "慢响应",
    )
    keyword_patterns = (
        r"[A-Za-z_][A-Za-z0-9_]{2,}",
        r".+\.(md|txt|pdf|docx)$",
        r"[A-Z]{2,}",
    )
    exact_terms = ("配置", "接口", "文件", "参数", "字段", "类名", "函数", "模型")

    def route(self, rewrite: QueryRewriteResult) -> RagRouteDecision:
        routes = ["vector"]
        text = " ".join(
            [
                rewrite.original_query,
                rewrite.rewritten_query,
                " ".join(rewrite.keywords),
            ]
        )
        reasons = ["default_vector"]

        if (
            config.multi_query_retrieval_enabled
            and rewrite.sub_questions
            and "multi_query_vector" not in routes
        ):
            routes.append("multi_query_vector")
            reasons.append("has_sub_questions")

        if config.graph_retrieval_enabled and self._contains_any(text, self.graph_keywords):
            routes.append("graph")
            reasons.append("matched_graph_keywords")

        if config.keyword_retrieval_enabled and self._should_use_keyword(text, rewrite):
            routes.append("keyword")
            reasons.append("matched_keyword_signal")

        return RagRouteDecision(routes=routes, reason=";".join(reasons))

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword.lower() in text.lower() for keyword in keywords)

    def _should_use_keyword(self, text: str, rewrite: QueryRewriteResult) -> bool:
        if rewrite.keywords:
            return True
        if self._contains_any(text, self.exact_terms):
            return True
        return any(re.search(pattern, text) for pattern in self.keyword_patterns)


rag_query_router_service = RagQueryRouterService()
