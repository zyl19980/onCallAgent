"""多轮对话记忆与摘要压缩。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from loguru import logger

from app.config import config


@dataclass
class SessionMemory:
    """单会话记忆。"""

    summary: str = ""
    history: List[Dict[str, str]] = field(default_factory=list)


class ConversationMemoryService:
    """管理最近 10 轮窗口与历史摘要。"""

    def __init__(self):
        self.window_rounds = config.rag_window_rounds
        self.max_messages = self.window_rounds * 2
        self.sessions: Dict[str, SessionMemory] = {}

    def get_or_create(self, session_id: str) -> SessionMemory:
        return self.sessions.setdefault(session_id, SessionMemory())

    def build_messages(self, session_id: str) -> List[BaseMessage]:
        session = self.get_or_create(session_id)
        messages: List[BaseMessage] = []
        if session.summary:
            messages.append(SystemMessage(content=f"历史对话摘要:\n{session.summary}"))

        for item in session.history[-self.max_messages :]:
            if item["role"] == "user":
                messages.append(HumanMessage(content=item["content"]))
            else:
                messages.append(AIMessage(content=item["content"]))
        return messages

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self.get_or_create(session_id).history)

    def get_summary(self, session_id: str) -> str:
        return self.get_or_create(session_id).summary

    def clear(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
        return True

    async def append_exchange(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        summarizer,
    ) -> None:
        session = self.get_or_create(session_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        session.history.extend(
            [
                {"role": "user", "content": user_text, "timestamp": timestamp},
                {"role": "assistant", "content": assistant_text, "timestamp": timestamp},
            ]
        )

        if len(session.history) <= self.max_messages:
            return

        overflow = session.history[:-self.max_messages]
        session.history = session.history[-self.max_messages :]
        session.summary = await self._summarize(session.summary, overflow, summarizer)

    async def _summarize(
        self,
        existing_summary: str,
        overflow: List[Dict[str, str]],
        summarizer,
    ) -> str:
        if not overflow:
            return existing_summary

        conversation_text = "\n".join(
            f"{item['role']}: {item['content']}" for item in overflow
        )
        prompt = (
            "请压缩以下对话历史，只保留故障对象、已确认事实、用户目标、未解决问题。"
            f" 输出不超过 {config.rag_summary_max_chars} 个中文字符。\n\n"
            f"已有摘要:\n{existing_summary or '无'}\n\n"
            f"新增对话:\n{conversation_text}"
        )

        try:
            summary = await summarizer(prompt)
            summary = (summary or "").strip()
            if summary:
                return summary[: config.rag_summary_max_chars]
        except Exception as exc:
            logger.warning(f"对话摘要压缩失败，将使用本地降级摘要: {exc}")

        fallback = f"{existing_summary}\n{conversation_text}".strip()
        return fallback[-config.rag_summary_max_chars :]


conversation_memory_service = ConversationMemoryService()
