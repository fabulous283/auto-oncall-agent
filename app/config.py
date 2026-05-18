"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_recall_top_k: int = 10
    rag_rerank_top_k: int = 3
    rag_rerank_model: str = "qwen3-rerank"
    dashscope_rerank_base_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1"
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 文档分块配置
    intent_llm_model: str = "qwen-turbo"
    intent_confidence_threshold: float = 0.6
    intent_rules_enabled: bool = True
    intent_llm_enabled: bool = True
    conversation_persistence_enabled: bool = True
    conversation_db_path: str = "data/conversations.db"
    conversation_recent_messages: int = 6
    conversation_summary_enabled: bool = True
    conversation_summary_trigger: int = 12
    conversation_summary_model: str = "qwen-turbo"
    conversation_context_window_tokens: int = 32768
    conversation_compress_threshold_ratio: float = 0.7
    conversation_summary_max_tokens: int = 1200
    conversation_compress_recent_messages: int = 6
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # 知识图谱配置
    knowledge_graph_enabled: bool = True
    knowledge_graph_db_path: str = "data/knowledge_graph.db"
    knowledge_graph_extract_model: str = "qwen-turbo"
    knowledge_graph_max_chunk_chars: int = 3000

    # RAG 检索增强配置
    query_rewrite_enabled: bool = True
    query_rewrite_model: str = "qwen-turbo"
    hybrid_retrieval_enabled: bool = True
    keyword_retrieval_enabled: bool = True
    graph_retrieval_enabled: bool = True
    multi_query_retrieval_enabled: bool = True
    keyword_top_k: int = 10
    graph_top_k: int = 10
    hybrid_candidate_top_k: int = 20
    chunk_index_db_path: str = "data/rag_chunks.db"

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
