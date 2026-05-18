"""RAG document reranking service backed by Alibaba Cloud Bailian."""

from typing import Any, List

import httpx
from langchain_core.documents import Document
from loguru import logger

from app.config import config


class DashScopeRerankService:
    """Call DashScope/Bailian rerank API and reorder retrieved documents."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key if api_key is not None else config.dashscope_api_key
        self.model = model or config.rag_rerank_model
        self.base_url = (base_url or config.dashscope_rerank_base_url).rstrip("/")
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/reranks"

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """Rerank candidate documents and return the top_k results.

        This method intentionally does not fall back to the vector-search order. If
        Bailian reranking fails, the exception is raised to the caller.
        """
        if not documents:
            return []
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置，无法调用百炼重排模型")
        if top_k <= 0:
            raise ValueError("rerank top_k 必须大于 0")

        top_n = min(top_k, len(documents))
        payload = {
            "model": self.model,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            f"开始调用百炼重排: model={self.model}, candidates={len(documents)}, top_n={top_n}"
        )

        try:
            response = httpx.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"百炼重排接口调用失败: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"百炼重排响应不是合法 JSON: {e}") from e

        results = data.get("results")
        if not isinstance(results, list):
            raise RuntimeError("百炼重排响应缺少 results 列表")

        reranked_docs: List[Document] = []
        for item in results:
            index = self._extract_index(item)
            if index < 0 or index >= len(documents):
                raise RuntimeError(f"百炼重排返回了非法文档下标: {index}")

            original_doc = documents[index]
            metadata = dict(original_doc.metadata)
            metadata["rerank_score"] = self._extract_score(item)
            metadata["rerank_model"] = self.model
            metadata["rerank_original_index"] = index
            reranked_docs.append(
                Document(page_content=original_doc.page_content, metadata=metadata)
            )

        logger.info(f"百炼重排完成: returned={len(reranked_docs)}")
        return reranked_docs

    @staticmethod
    def _extract_index(item: Any) -> int:
        if not isinstance(item, dict) or "index" not in item:
            raise RuntimeError(f"百炼重排结果缺少 index: {item}")
        return int(item["index"])

    @staticmethod
    def _extract_score(item: Any) -> float | None:
        if not isinstance(item, dict):
            return None
        score = item.get("relevance_score", item.get("score"))
        return float(score) if score is not None else None


rerank_service = DashScopeRerankService()
