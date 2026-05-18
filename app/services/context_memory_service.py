from textwrap import dedent

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
try:
    from langchain_core.messages.utils import count_tokens_approximately
except ImportError:  # pragma: no cover - depends on installed langchain-core version
    count_tokens_approximately = None
from langchain_qwq import ChatQwen
from loguru import logger

from app.config import config
from app.models.conversation import ConversationMessage
from app.services.conversation_store_service import conversation_store_service


class ContextMemoryService:
    def __init__(self):
        self.store = conversation_store_service
        self.summary_model = None

    def build_context_messages(self, session_id: str) -> list[BaseMessage]:
        if not config.conversation_persistence_enabled:
            return []

        messages: list[BaseMessage] = []
        summary = self.store.get_summary(session_id)
        if summary and summary.summary:
            messages.append(
                SystemMessage(
                    content=(
                        "以下是当前会话的历史摘要，请结合它理解上下文，不要重复输出摘要原文。\n"
                        f"{summary.summary}"
                    )
                )
            )

        recent_messages = self.store.get_recent_messages(
            session_id,
            config.conversation_recent_messages,
        )
        messages.extend(self._convert_messages(recent_messages))
        return messages

    async def compress_context_if_needed(
        self,
        session_id: str,
        system_prompt: str,
        question: str,
    ) -> bool:
        if not config.conversation_persistence_enabled or not config.conversation_summary_enabled:
            return False

        projected_messages = [
            SystemMessage(content=system_prompt),
            *self.build_context_messages(session_id),
            HumanMessage(content=question),
        ]
        projected_tokens = self.estimate_message_tokens(projected_messages)
        threshold_tokens = int(
            config.conversation_context_window_tokens
            * config.conversation_compress_threshold_ratio
        )
        logger.debug(
            f"[会话 {session_id}] 上下文 token 估算: "
            f"{projected_tokens}/{config.conversation_context_window_tokens}, "
            f"压缩阈值={threshold_tokens}"
        )

        if projected_tokens < threshold_tokens:
            return False

        compressed = await self._compress_older_messages(session_id)
        if compressed:
            logger.info(
                f"[会话 {session_id}] 上下文达到 {config.conversation_compress_threshold_ratio:.0%} "
                f"窗口阈值，已自动压缩历史消息"
            )
        return compressed

    async def persist_chat_turn(self, session_id: str, question: str, answer: str):
        if not config.conversation_persistence_enabled:
            return

        self.store.append_message(session_id, "user", question)
        self.store.append_message(session_id, "assistant", answer)
        await self._refresh_summary_if_needed(session_id)

    async def persist_assistant_message(self, session_id: str, message: str):
        if not config.conversation_persistence_enabled:
            return

        self.store.append_message(session_id, "assistant", message)
        await self._refresh_summary_if_needed(session_id)

    def get_persisted_history(self, session_id: str) -> list[dict]:
        messages = self.store.get_messages(session_id)
        return [
            {
                "role": "user" if msg.role == "user" else "assistant",
                "content": msg.content,
                "timestamp": msg.created_at,
            }
            for msg in messages
        ]

    def has_persisted_history(self, session_id: str) -> bool:
        return self.store.get_session(session_id) is not None

    def clear_session(self, session_id: str):
        self.store.clear_session(session_id)

    async def _refresh_summary_if_needed(self, session_id: str):
        if not config.conversation_summary_enabled:
            return

        total_messages = self.store.get_message_count(session_id)
        if total_messages < config.conversation_summary_trigger:
            return

        cutoff_index = total_messages - config.conversation_recent_messages
        if cutoff_index <= 0:
            return

        current_summary = self.store.get_summary(session_id)
        start_index = 1 if current_summary is None else current_summary.last_message_index + 1
        if cutoff_index < start_index:
            return

        messages_to_summarize = self.store.get_messages_between(session_id, start_index, cutoff_index)
        if not messages_to_summarize:
            return

        try:
            updated_summary = await self._generate_summary(
                previous_summary=current_summary.summary if current_summary else "",
                messages=messages_to_summarize,
            )
            if updated_summary:
                self.store.upsert_summary(session_id, updated_summary, cutoff_index)
                logger.info(f"[会话 {session_id}] 会话摘要已更新，覆盖到消息 {cutoff_index}")
        except Exception as exc:
            logger.warning(f"[会话 {session_id}] 会话摘要更新失败: {exc}")

    async def _compress_older_messages(self, session_id: str) -> bool:
        total_messages = self.store.get_message_count(session_id)
        cutoff_index = total_messages - config.conversation_compress_recent_messages
        if cutoff_index <= 0:
            logger.debug(f"[会话 {session_id}] 可压缩消息不足，跳过上下文压缩")
            return False

        current_summary = self.store.get_summary(session_id)
        start_index = 1 if current_summary is None else current_summary.last_message_index + 1
        if cutoff_index < start_index:
            logger.debug(f"[会话 {session_id}] 历史消息已压缩到最新 cutoff，跳过")
            return False

        messages_to_summarize = self.store.get_messages_between(session_id, start_index, cutoff_index)
        if not messages_to_summarize:
            return False

        try:
            updated_summary = await self._generate_summary(
                previous_summary=current_summary.summary if current_summary else "",
                messages=messages_to_summarize,
            )
            if not updated_summary:
                return False

            self.store.upsert_summary(session_id, updated_summary, cutoff_index)
            logger.info(f"[会话 {session_id}] 自动压缩摘要已更新，覆盖到消息 {cutoff_index}")
            return True
        except Exception as exc:
            logger.warning(f"[会话 {session_id}] 自动压缩失败: {exc}")
            return False

    async def _generate_summary(
        self,
        previous_summary: str,
        messages: list[ConversationMessage],
    ) -> str:
        model = self._get_summary_model()
        conversation_text = "\n".join(
            f"{'用户' if message.role == 'user' else '助手'}: {message.content}"
            for message in messages
        )

        prompt = dedent(
            f"""
            你是一个会话记忆压缩助手，请输出简洁、结构化的中文摘要。

            输出要求：
            1. 保留用户当前目标或主题
            2. 保留已确认的事实
            3. 保留已执行过的关键动作
            4. 保留仍未解决的问题
            5. 不要保留寒暄和无意义重复
            6. 摘要长度尽量控制在 {config.conversation_summary_max_tokens} token 以内
            7. 直接输出摘要正文，不要加标题，不要用代码块

            现有摘要：
            {previous_summary or '无'}

            新增会话内容：
            {conversation_text}
            """
        ).strip()

        result = await model.ainvoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return (content or "").strip()

    @classmethod
    def estimate_message_tokens(cls, messages: list[BaseMessage]) -> int:
        if count_tokens_approximately is not None:
            try:
                return int(count_tokens_approximately(messages))
            except Exception as exc:
                logger.debug(f"LangChain token 估算失败，使用本地估算: {exc}")

        return sum(cls._estimate_text_tokens(str(message.content)) + 4 for message in messages)

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        if not text:
            return 0

        ascii_chars = 0
        non_ascii_chars = 0
        for char in text:
            if char.isspace():
                continue
            if ord(char) < 128:
                ascii_chars += 1
            else:
                non_ascii_chars += 1

        return non_ascii_chars + max(1, ascii_chars // 4)

    def _get_summary_model(self) -> ChatQwen:
        if self.summary_model is None:
            self.summary_model = ChatQwen(
                model=config.conversation_summary_model,
                api_key=config.dashscope_api_key,
                temperature=0.2,
                streaming=False,
            )
        return self.summary_model

    @staticmethod
    def _convert_messages(messages: list[ConversationMessage]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []
        for message in messages:
            if message.role == "user":
                converted.append(HumanMessage(content=message.content))
            else:
                converted.append(AIMessage(content=message.content))
        return converted


context_memory_service = ContextMemoryService()
