"""知识检索工具。"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.services.hybrid_retrieval_service import hybrid_retrieval_service


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题。"""
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")
        retrieval = hybrid_retrieval_service.retrieve(query)
        if not retrieval.candidates:
            return "没有找到相关信息。", []
        return retrieval.context_text(), retrieval.documents()
    except Exception as exc:
        logger.error(f"知识检索工具调用失败: {exc}")
        return f"检索知识时发生错误: {exc}", []
