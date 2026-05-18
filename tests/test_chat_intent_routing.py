import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import chat
from app.models.intent import IntentDecision, IntentType


def create_client() -> TestClient:
    app = FastAPI()
    app.include_router(chat.router, prefix="/api")
    return TestClient(app)


def test_chat_returns_aiops_handoff(monkeypatch):
    client = create_client()

    async def fake_recognize(question: str, session_id: str):
        return IntentDecision(
            intent=IntentType.AIOPS_DIAGNOSIS,
            confidence=0.95,
            reason="matched_aiops_keywords",
            should_handoff=True,
            target_flow="aiops",
        )

    async def unexpected_query(question: str, session_id: str):
        raise AssertionError("rag_agent_service.query should not be called for aiops handoff")

    monkeypatch.setattr(chat.intent_recognition_service, "recognize", fake_recognize)
    monkeypatch.setattr(chat.rag_agent_service, "query", unexpected_query)

    response = client.post(
        "/api/chat",
        json={"Id": "session-1", "Question": "帮我分析一下当前 CPU 告警"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["data"]["success"] is True
    assert payload["data"]["intent"] == "aiops_diagnosis"
    assert payload["data"]["handoff"]["type"] == "aiops"
    assert payload["data"]["handoff"]["status"] == "pending_confirmation"


def test_chat_non_aiops_routes_to_rag(monkeypatch):
    client = create_client()

    async def fake_recognize(question: str, session_id: str):
        return IntentDecision(
            intent=IntentType.KNOWLEDGE_QA,
            confidence=0.86,
            reason="matched_knowledge_keywords",
            should_handoff=False,
            target_flow="chat",
        )

    async def fake_query(question: str, session_id: str):
        return "这是项目里的 RAG 实现说明"

    monkeypatch.setattr(chat.intent_recognition_service, "recognize", fake_recognize)
    monkeypatch.setattr(chat.rag_agent_service, "query", fake_query)

    response = client.post(
        "/api/chat",
        json={"Id": "session-2", "Question": "这个项目的 RAG 怎么实现"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["data"]["success"] is True
    assert payload["data"]["intent"] == "knowledge_qa"
    assert payload["data"]["answer"] == "这是项目里的 RAG 实现说明"
    assert payload["data"]["handoff"] is None


def test_chat_stream_returns_handoff_event(monkeypatch):
    client = create_client()

    async def fake_recognize(question: str, session_id: str):
        return IntentDecision(
            intent=IntentType.AIOPS_DIAGNOSIS,
            confidence=0.95,
            reason="matched_aiops_keywords",
            should_handoff=True,
            target_flow="aiops",
        )

    monkeypatch.setattr(chat.intent_recognition_service, "recognize", fake_recognize)

    with client.stream(
        "POST",
        "/api/chat_stream",
        json={"Id": "session-3", "Question": "帮我分析一下当前 CPU 告警"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"type": "handoff"' in body
    assert "pending_confirmation" in body


def test_chat_stream_non_aiops_keeps_streaming(monkeypatch):
    client = create_client()

    async def fake_recognize(question: str, session_id: str):
        return IntentDecision(
            intent=IntentType.GENERAL_CHAT,
            confidence=0.8,
            reason="fallback",
            should_handoff=False,
            target_flow="chat",
        )

    async def fake_query_stream(question: str, session_id: str):
        yield {"type": "content", "data": "你好"}
        yield {"type": "complete", "data": {"answer": "你好"}}

    monkeypatch.setattr(chat.intent_recognition_service, "recognize", fake_recognize)
    monkeypatch.setattr(chat.rag_agent_service, "query_stream", fake_query_stream)

    with client.stream(
        "POST",
        "/api/chat_stream",
        json={"Id": "session-4", "Question": "你好"},
    ) as response:
        chunks = list(response.iter_text())

    assert response.status_code == 200
    joined = "".join(chunks)
    assert json.dumps({"type": "content", "data": "你好"}, ensure_ascii=False) in joined
    assert '"type": "done"' in joined
