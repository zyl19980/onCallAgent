"""Runtime helpers for AIOps Plan-Execute-Replan nodes."""

from __future__ import annotations

from typing import Any

from app.config import config
from app.core.llm_factory import llm_factory


def create_aiops_chat_model(
    *,
    model_name: str | None = None,
    temperature: float = 0.0,
    streaming: bool = False,
) -> Any:
    """Create the chat model used by AIOps nodes.

    The original runtime uses ``langchain_qwq.ChatQwen``. Some experiment
    environments only have OpenAI-compatible LangChain clients installed, so
    this helper preserves the preferred provider and falls back cleanly.
    """
    model = model_name or config.rag_model

    try:
        from langchain_qwq import ChatQwen

        return ChatQwen(
            model=model,
            api_key=config.dashscope_api_key,
            temperature=temperature,
        )
    except Exception:
        return llm_factory.create_chat_model(
            model=model,
            temperature=temperature,
            streaming=streaming,
            api_key=config.dashscope_api_key,
        )
