import json
from typing import Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config
from app.models.intent import IntentDecision, IntentType


class IntentRecognitionService:
    def __init__(self):
        self._classifier_model = None
        self.aiops_keywords = (
            "告警",
            "故障",
            "排查",
            "诊断",
            "日志",
            "监控",
            "cpu",
            "内存",
            "延迟",
            "异常",
            "报错",
            "宕机",
            "指标",
            "服务异常",
        )
        self.tool_keywords = (
            "几点",
            "时间",
            "时间戳",
            "timestamp",
            "topic",
            "region",
            "时区",
        )
        self.knowledge_keywords = (
            "文档",
            "知识库",
            "项目里",
            "项目中",
            "原理",
            "实现",
            "怎么做",
            "是什么",
            "介绍",
            "解释",
        )

    async def recognize(self, question: str, session_id: str) -> IntentDecision:
        normalized_question = (question or "").strip()
        if not normalized_question:
            return self._fallback_decision("empty_question")

        if config.intent_rules_enabled:
            rule_decision = self._rule_based_recognition(normalized_question)
            if rule_decision is not None:
                return rule_decision

        if config.intent_llm_enabled:
            llm_decision = await self._llm_recognition(normalized_question, session_id)
            if llm_decision is not None:
                if llm_decision.confidence >= config.intent_confidence_threshold:
                    return llm_decision
                logger.info(
                    f"[会话 {session_id}] 意图识别置信度过低，降级为 general_chat: "
                    f"{llm_decision.confidence}"
                )

        return self._fallback_decision("fallback_to_general_chat")

    def _rule_based_recognition(self, question: str) -> IntentDecision | None:
        lowered_question = question.lower()

        if self._contains_any(lowered_question, self.aiops_keywords):
            return IntentDecision(
                intent=IntentType.AIOPS_DIAGNOSIS,
                confidence=0.95,
                reason="matched_aiops_keywords",
                should_handoff=True,
                target_flow="aiops",
            )

        if self._contains_any(lowered_question, self.tool_keywords):
            return IntentDecision(
                intent=IntentType.TOOL_QUERY,
                confidence=0.88,
                reason="matched_tool_keywords",
                should_handoff=False,
                target_flow="chat",
            )

        if self._contains_any(lowered_question, self.knowledge_keywords):
            return IntentDecision(
                intent=IntentType.KNOWLEDGE_QA,
                confidence=0.86,
                reason="matched_knowledge_keywords",
                should_handoff=False,
                target_flow="chat",
            )

        return None

    async def _llm_recognition(
        self, question: str, session_id: str
    ) -> IntentDecision | None:
        try:
            model = self._get_classifier_model()
            result = await model.ainvoke(
                [
                    SystemMessage(content=self._build_classifier_prompt()),
                    HumanMessage(content=question),
                ]
            )
            content = result.content if hasattr(result, "content") else str(result)
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )

            parsed = json.loads((content or "").strip())
            decision = IntentDecision(**parsed)
            return self._normalize_decision(decision)
        except Exception as exc:
            logger.warning(f"[会话 {session_id}] LLM 意图识别失败: {exc}")
            return None

    def _get_classifier_model(self) -> ChatQwen:
        if self._classifier_model is None:
            self._classifier_model = ChatQwen(
                model=config.intent_llm_model,
                api_key=config.dashscope_api_key,
                temperature=0.0,
                streaming=False,
            )
        return self._classifier_model

    def _build_classifier_prompt(self) -> str:
        return """
你是一个意图分类器，只负责分类，不负责回答问题。

请从下面 4 个固定意图中选 1 个：
- general_chat
- knowledge_qa
- aiops_diagnosis
- tool_query

判断规则：
- 用户想做告警、故障、监控、日志、CPU/内存等分析排查，归为 aiops_diagnosis
- 用户在问项目知识、文档、原理、实现，归为 knowledge_qa
- 用户在问时间、时间戳、topic、region 等工具型查询，归为 tool_query
- 其他归为 general_chat

输出要求：
- 只能输出严格 JSON
- 不要输出 Markdown，不要输出解释，不要输出代码块
- JSON 字段必须且只能包含：
  intent, confidence, reason, should_handoff, target_flow

字段约束：
- intent 必须是 4 个固定值之一
- confidence 是 0 到 1 的小数
- should_handoff 只有在 aiops_diagnosis 时为 true，其余为 false
- target_flow 对 aiops_diagnosis 填 aiops，其余填 chat
""".strip()

    def _normalize_decision(self, decision: IntentDecision) -> IntentDecision:
        normalized_confidence = min(max(decision.confidence, 0.0), 1.0)
        should_handoff = decision.intent == IntentType.AIOPS_DIAGNOSIS
        target_flow = "aiops" if should_handoff else "chat"
        return IntentDecision(
            intent=decision.intent,
            confidence=normalized_confidence,
            reason=decision.reason or "llm_classification",
            should_handoff=should_handoff,
            target_flow=target_flow,
        )

    def _fallback_decision(self, reason: str) -> IntentDecision:
        return IntentDecision(
            intent=IntentType.GENERAL_CHAT,
            confidence=0.0,
            reason=reason,
            should_handoff=False,
            target_flow="chat",
        )

    @staticmethod
    def _contains_any(text: str, keywords: Iterable[str]) -> bool:
        return any(keyword in text for keyword in keywords)


intent_recognition_service = IntentRecognitionService()
