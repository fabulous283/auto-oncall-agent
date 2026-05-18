from enum import Enum

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    GENERAL_CHAT = "general_chat"
    KNOWLEDGE_QA = "knowledge_qa"
    AIOPS_DIAGNOSIS = "aiops_diagnosis"
    TOOL_QUERY = "tool_query"


class IntentDecision(BaseModel):
    intent: IntentType = Field(default=IntentType.GENERAL_CHAT)
    confidence: float = Field(default=0.0)
    reason: str = Field(default="")
    should_handoff: bool = Field(default=False)
    target_flow: str = Field(default="chat")


class IntentHandoff(BaseModel):
    type: str = Field(default="aiops")
    status: str = Field(default="pending_confirmation")
    message: str = Field(
        default="检测到你可能想做告警/故障诊断，是否进入 AIOps 分析流程？"
    )
    session_id: str
