"""Models for query understanding and hybrid retrieval."""

from typing import Any

from pydantic import BaseModel, Field


class QueryRewriteResult(BaseModel):
    original_query: str
    rewritten_query: str
    keywords: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    intent: str = "knowledge_qa"
    filters: dict[str, Any] = Field(default_factory=dict)


class RagRouteDecision(BaseModel):
    routes: list[str] = Field(default_factory=lambda: ["vector"])
    reason: str = "default_vector_route"


class RetrievedDocument(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str
    score: float = 0.0
    chunk_id: str = ""
    source_file: str = ""
    chunk_index: int | None = None
    sources: list[str] = Field(default_factory=list)
