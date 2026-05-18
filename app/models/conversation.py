from pydantic import BaseModel, Field


class ConversationSession(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    status: str = Field(default="active", description="会话状态")


class ConversationMessage(BaseModel):
    id: int | None = Field(default=None, description="消息主键")
    session_id: str = Field(..., description="会话 ID")
    role: str = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    message_index: int = Field(..., description="会话内顺序")
    created_at: str = Field(..., description="创建时间")


class ConversationSummary(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    summary: str = Field(..., description="会话摘要")
    last_message_index: int = Field(..., description="摘要覆盖到的消息序号")
    updated_at: str = Field(..., description="更新时间")
