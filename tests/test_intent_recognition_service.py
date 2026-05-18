from app.models.intent import IntentType
from app.services.intent_recognition_service import IntentRecognitionService


def test_rule_matches_aiops_keywords():
    service = IntentRecognitionService()

    decision = service._rule_based_recognition("帮我分析一下当前 CPU 告警")

    assert decision is not None
    assert decision.intent == IntentType.AIOPS_DIAGNOSIS
    assert decision.should_handoff is True
    assert decision.target_flow == "aiops"


def test_rule_matches_tool_keywords():
    service = IntentRecognitionService()

    decision = service._rule_based_recognition("现在几点了")

    assert decision is not None
    assert decision.intent == IntentType.TOOL_QUERY
    assert decision.target_flow == "chat"


def test_rule_matches_knowledge_keywords():
    service = IntentRecognitionService()

    decision = service._rule_based_recognition("这个项目里的 RAG 原理是什么")

    assert decision is not None
    assert decision.intent == IntentType.KNOWLEDGE_QA
    assert decision.target_flow == "chat"


async def test_llm_failure_falls_back_to_general_chat(monkeypatch):
    service = IntentRecognitionService()

    async def broken_llm(question: str, session_id: str):
        return None

    monkeypatch.setattr(service, "_llm_recognition", broken_llm)

    decision = await service.recognize("随便问一个没有命中规则的问题", "session-test")

    assert decision.intent == IntentType.GENERAL_CHAT
    assert decision.target_flow == "chat"
