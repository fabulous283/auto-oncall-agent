from langchain_core.messages import HumanMessage, SystemMessage

from app.models.conversation import ConversationMessage, ConversationSummary
from app.services.context_memory_service import ContextMemoryService


class DummyStore:
    def __init__(self):
        self.summary = ConversationSummary(
            session_id="session-1",
            summary="用户正在了解项目记忆实现",
            last_message_index=2,
            updated_at="2026-01-01T00:00:00",
        )

    def get_summary(self, session_id: str):
        return self.summary

    def get_recent_messages(self, session_id: str, limit: int):
        from app.models.conversation import ConversationMessage

        return [
            ConversationMessage(
                session_id=session_id,
                role="user",
                content="这个项目怎么做记忆？",
                message_index=3,
                created_at="2026-01-01T00:00:01",
            ),
            ConversationMessage(
                session_id=session_id,
                role="assistant",
                content="当前是 MemorySaver 短期记忆。",
                message_index=4,
                created_at="2026-01-01T00:00:02",
            ),
        ]


def test_build_context_messages_includes_summary_and_recent_messages():
    service = ContextMemoryService()
    service.store = DummyStore()

    messages = service.build_context_messages("session-1")

    assert len(messages) == 3
    assert isinstance(messages[0], SystemMessage)
    assert "历史摘要" in messages[0].content
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "这个项目怎么做记忆？"


class CompressStore:
    def __init__(self):
        self.summary = None
        self.upserted_summary = None
        self.upserted_index = None
        self.messages = [
            ConversationMessage(
                session_id="session-1",
                role="user" if index % 2 else "assistant",
                content=f"第 {index} 条很长的历史消息",
                message_index=index,
                created_at="2026-01-01T00:00:00",
            )
            for index in range(1, 11)
        ]

    def get_summary(self, session_id: str):
        return self.summary

    def get_recent_messages(self, session_id: str, limit: int):
        return self.messages[-limit:]

    def get_message_count(self, session_id: str):
        return len(self.messages)

    def get_messages_between(self, session_id: str, start_index: int, end_index: int):
        return [
            message
            for message in self.messages
            if start_index <= message.message_index <= end_index
        ]

    def upsert_summary(self, session_id: str, summary: str, last_message_index: int):
        self.upserted_summary = summary
        self.upserted_index = last_message_index
        self.summary = ConversationSummary(
            session_id=session_id,
            summary=summary,
            last_message_index=last_message_index,
            updated_at="2026-01-01T00:00:00",
        )


async def test_context_compression_skips_when_under_threshold(monkeypatch):
    service = ContextMemoryService()
    service.store = CompressStore()

    monkeypatch.setattr("app.services.context_memory_service.config.conversation_context_window_tokens", 1000)
    monkeypatch.setattr("app.services.context_memory_service.config.conversation_compress_threshold_ratio", 0.7)
    monkeypatch.setattr(service, "estimate_message_tokens", lambda messages: 100)

    async def unexpected_summary(previous_summary, messages):
        raise AssertionError("summary should not be generated below threshold")

    monkeypatch.setattr(service, "_generate_summary", unexpected_summary)

    compressed = await service.compress_context_if_needed(
        session_id="session-1",
        system_prompt="系统提示",
        question="当前问题",
    )

    assert compressed is False
    assert service.store.upserted_summary is None


async def test_context_compression_summarizes_older_messages_when_over_threshold(monkeypatch):
    service = ContextMemoryService()
    service.store = CompressStore()

    monkeypatch.setattr("app.services.context_memory_service.config.conversation_context_window_tokens", 1000)
    monkeypatch.setattr("app.services.context_memory_service.config.conversation_compress_threshold_ratio", 0.7)
    monkeypatch.setattr("app.services.context_memory_service.config.conversation_compress_recent_messages", 4)
    monkeypatch.setattr(service, "estimate_message_tokens", lambda messages: 800)

    captured = {}

    async def fake_summary(previous_summary, messages):
        captured["previous_summary"] = previous_summary
        captured["indexes"] = [message.message_index for message in messages]
        return "压缩后的历史摘要"

    monkeypatch.setattr(service, "_generate_summary", fake_summary)

    compressed = await service.compress_context_if_needed(
        session_id="session-1",
        system_prompt="系统提示",
        question="当前问题",
    )

    assert compressed is True
    assert captured["previous_summary"] == ""
    assert captured["indexes"] == [1, 2, 3, 4, 5, 6]
    assert service.store.upserted_summary == "压缩后的历史摘要"
    assert service.store.upserted_index == 6


async def test_context_compression_does_not_repeat_existing_summary(monkeypatch):
    service = ContextMemoryService()
    store = CompressStore()
    store.summary = ConversationSummary(
        session_id="session-1",
        summary="已有摘要",
        last_message_index=6,
        updated_at="2026-01-01T00:00:00",
    )
    service.store = store

    monkeypatch.setattr("app.services.context_memory_service.config.conversation_context_window_tokens", 1000)
    monkeypatch.setattr("app.services.context_memory_service.config.conversation_compress_threshold_ratio", 0.7)
    monkeypatch.setattr("app.services.context_memory_service.config.conversation_compress_recent_messages", 4)
    monkeypatch.setattr(service, "estimate_message_tokens", lambda messages: 800)

    async def unexpected_summary(previous_summary, messages):
        raise AssertionError("already summarized messages should not be summarized again")

    monkeypatch.setattr(service, "_generate_summary", unexpected_summary)

    compressed = await service.compress_context_if_needed(
        session_id="session-1",
        system_prompt="系统提示",
        question="当前问题",
    )

    assert compressed is False
