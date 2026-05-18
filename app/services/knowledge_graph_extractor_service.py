"""LLM-based entity and relation extraction for knowledge graph indexing."""

import json
import re
from textwrap import dedent

from langchain_core.documents import Document
from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config
from app.models.knowledge_graph import (
    KnowledgeGraphEntity,
    KnowledgeGraphExtractionResult,
    KnowledgeGraphRelation,
)


class KnowledgeGraphExtractorService:
    def __init__(self):
        self.model = None

    async def extract_from_document(
        self,
        document: Document,
        source_file: str,
        chunk_index: int,
        chunk_id: str,
    ) -> KnowledgeGraphExtractionResult:
        content = (document.page_content or "").strip()
        if not content:
            return KnowledgeGraphExtractionResult()

        if len(content) > config.knowledge_graph_max_chunk_chars:
            content = content[: config.knowledge_graph_max_chunk_chars]

        prompt = self._build_prompt(content, source_file, chunk_index, chunk_id)
        result = await self._get_model().ainvoke(prompt)
        raw_content = result.content if hasattr(result, "content") else str(result)
        if isinstance(raw_content, list):
            raw_content = "".join(
                block.get("text", "")
                for block in raw_content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        extraction = self.parse_extraction(str(raw_content), source_file, chunk_index, chunk_id)
        logger.info(
            f"知识图谱抽取完成: source={source_file}, chunk={chunk_index}, "
            f"entities={len(extraction.entities)}, relations={len(extraction.relations)}"
        )
        return extraction

    def parse_extraction(
        self,
        raw_content: str,
        source_file: str,
        chunk_index: int,
        chunk_id: str,
    ) -> KnowledgeGraphExtractionResult:
        data = json.loads(self._extract_json_text(raw_content))
        entities = [
            KnowledgeGraphEntity(**entity)
            for entity in data.get("entities", [])
            if isinstance(entity, dict)
        ]
        entity_ids = {entity.id for entity in entities}

        relations: list[KnowledgeGraphRelation] = []
        for relation_data in data.get("relations", []):
            if not isinstance(relation_data, dict):
                continue
            relation_data["source_file"] = source_file
            relation_data["chunk_index"] = chunk_index
            relation_data["chunk_id"] = chunk_id
            relation = KnowledgeGraphRelation(**relation_data)
            if relation.source not in entity_ids or relation.target not in entity_ids:
                logger.debug(f"过滤非法知识图谱关系，实体不存在: {relation}")
                continue
            relations.append(relation)

        return KnowledgeGraphExtractionResult(entities=entities, relations=relations)

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

    def _build_prompt(
        self,
        content: str,
        source_file: str,
        chunk_index: int,
        chunk_id: str,
    ) -> str:
        return dedent(
            f"""
            你是知识图谱抽取助手。请只根据给定原文抽取实体和关系，不要编造。

            抽取要求：
            1. 输出严格 JSON，不要输出解释，不要使用 Markdown 代码块。
            2. JSON 顶层必须包含 entities 和 relations。
            3. 实体字段必须包含 id、name、type、description。
            4. 关系字段必须包含 source、target、relation、description、evidence、source_file、chunk_index、chunk_id。
            5. source 和 target 必须引用 entities 中存在的实体 id。
            6. id 使用稳定的英文或拼音 snake_case，避免空格和中文标点。
            7. 如果关系不确定，或者原文没有依据，不要抽取。
            8. evidence 必须是原文里的依据句子或短语。

            实体类型建议：
            系统、模块、服务、接口、指标、告警、故障、原因、排查动作、解决方案、文档、配置项、工具、数据库、中间件、其他。

            关系类型建议：
            包含、依赖、调用、导致、关联、排查、解决、使用、存储于、配置、监控、触发、来源于。

            固定溯源字段：
            source_file: {source_file}
            chunk_index: {chunk_index}
            chunk_id: {chunk_id}

            输出格式：
            {{
              "entities": [
                {{
                  "id": "cpu_high_usage",
                  "name": "CPU 使用率高",
                  "type": "故障",
                  "description": "服务器 CPU 使用率持续处于高位"
                }}
              ],
              "relations": [
                {{
                  "source": "cpu_high_usage",
                  "target": "check_top_process",
                  "relation": "排查",
                  "description": "CPU 使用率高时需要查看高占用进程",
                  "evidence": "原文依据",
                  "source_file": "{source_file}",
                  "chunk_index": {chunk_index},
                  "chunk_id": "{chunk_id}"
                }}
              ]
            }}

            原文：
            {content}
            """
        ).strip()

    def _get_model(self) -> ChatQwen:
        if self.model is None:
            self.model = ChatQwen(
                model=config.knowledge_graph_extract_model,
                api_key=config.dashscope_api_key,
                temperature=0.0,
                streaming=False,
            )
        return self.model


knowledge_graph_extractor_service = KnowledgeGraphExtractorService()
