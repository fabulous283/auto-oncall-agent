"""Knowledge graph data models."""

from pydantic import BaseModel, Field


class KnowledgeGraphEntity(BaseModel):
    id: str = Field(..., description="Entity stable ID")
    name: str = Field(..., description="Entity display name")
    type: str = Field(default="其他", description="Entity type")
    description: str = Field(default="", description="Entity description")


class KnowledgeGraphRelation(BaseModel):
    source: str = Field(..., description="Source entity ID")
    target: str = Field(..., description="Target entity ID")
    relation: str = Field(..., description="Relation type")
    description: str = Field(default="", description="Relation description")
    evidence: str = Field(default="", description="Evidence from source text")
    source_file: str = Field(default="", description="Source file path")
    chunk_index: int = Field(default=0, description="Chunk index in source file")
    chunk_id: str = Field(default="", description="Stable chunk ID")


class KnowledgeGraphExtractionResult(BaseModel):
    entities: list[KnowledgeGraphEntity] = Field(default_factory=list)
    relations: list[KnowledgeGraphRelation] = Field(default_factory=list)
