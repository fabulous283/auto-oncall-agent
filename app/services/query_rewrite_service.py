"""Query rewrite service for RAG retrieval."""

import json
import re
from textwrap import dedent

from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config
from app.models.retrieval import QueryRewriteResult


class QueryRewriteService:
    def __init__(self):
        self.model = None

    async def rewrite(self, query: str) -> QueryRewriteResult:
        if not config.query_rewrite_enabled:
            return self.default_result(query)

        try:
            result = await self._get_model().ainvoke(self._build_prompt(query))
            raw_content = result.content if hasattr(result, "content") else str(result)
            if isinstance(raw_content, list):
                raw_content = "".join(
                    block.get("text", "")
                    for block in raw_content
                    if isinstance(block, dict) and block.get("type") == "text"
                )

            data = json.loads(self._extract_json_text(str(raw_content)))
            rewritten = QueryRewriteResult(
                original_query=data.get("original_query") or query,
                rewritten_query=data.get("rewritten_query") or query,
                keywords=self._normalize_list(data.get("keywords")),
                sub_questions=self._normalize_list(data.get("sub_questions")),
                intent=data.get("intent") or "knowledge_qa",
                filters=data.get("filters") if isinstance(data.get("filters"), dict) else {},
            )
            logger.info(
                f"查询改写完成: keywords={len(rewritten.keywords)}, "
                f"sub_questions={len(rewritten.sub_questions)}"
            )
            return rewritten
        except Exception as exc:
            logger.warning(f"查询改写失败，回退原始问题: {exc}")
            return self.default_result(query)

    @staticmethod
    def default_result(query: str) -> QueryRewriteResult:
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=query,
            keywords=[query] if query else [],
            sub_questions=[],
            intent="knowledge_qa",
            filters={},
        )

    @staticmethod
    def _normalize_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _extract_json_text(raw_content: str) -> str:
        text = raw_content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            return text[start : end + 1]
        return text

    @staticmethod
    def _build_prompt(query: str) -> str:
        return dedent(
            f"""
            你是 RAG 查询改写助手。请把用户问题改写成更适合知识库检索的结构化 JSON。

            要求：
            1. 只输出 JSON，不要解释，不要使用 Markdown 代码块。
            2. 保留用户原始意图，不要编造额外需求。
            3. keywords 用于关键词检索和知识图谱检索。
            4. sub_questions 用于多查询向量召回，最多 3 个。
            5. filters 如果没有明确过滤条件，输出空对象。

            JSON 格式：
            {{
              "original_query": "用户原始问题",
              "rewritten_query": "更适合检索的改写问题",
              "keywords": ["关键词1", "关键词2"],
              "sub_questions": ["子问题1", "子问题2"],
              "intent": "knowledge_qa",
              "filters": {{}}
            }}

            用户问题：
            {query}
            """
        ).strip()

    def _get_model(self) -> ChatQwen:
        if self.model is None:
            self.model = ChatQwen(
                model=config.query_rewrite_model,
                api_key=config.dashscope_api_key,
                temperature=0.0,
                streaming=False,
            )
        return self.model


query_rewrite_service = QueryRewriteService()
