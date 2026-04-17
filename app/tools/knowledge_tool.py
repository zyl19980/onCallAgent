"""知识检索工具。"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.services.hybrid_retrieval_service import hybrid_retrieval_service


def _format_retrieval_tool_output(retrieval) -> str:
    header = [
        f"检索整体置信度: {retrieval.confidence}",
        f"重排来源: {retrieval.rerank_provider}",
    ]
    if retrieval.low_confidence_reason:
        header.append(f"置信度说明: {retrieval.low_confidence_reason}")

    debug = retrieval.confidence_debug or {}
    if debug:
        header.append(
            "置信度统计: "
            f"top1={debug.get('top1Score', 0)}, "
            f"top2={debug.get('top2Score', 0)}, "
            f"avgTop3={debug.get('avgTop3Score', 0)}, "
            f"support={debug.get('supportCount', 0)}"
        )

    body = retrieval.context_text() or "当前没有命中知识库证据。"
    return "\n".join(header) + "\n\n" + body


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题。"""
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")
        retrieval = hybrid_retrieval_service.retrieve(query)
        if not retrieval.candidates:
            return "没有找到相关信息。", []
        logger.info(
            "工具检索完成: confidence={}, rerank_provider={}, top_scores={}",
            retrieval.confidence,
            retrieval.rerank_provider,
            [round(candidate.rerank_score, 4) for candidate in retrieval.candidates[:5]],
        )
        return _format_retrieval_tool_output(retrieval), retrieval.documents()
    except Exception as exc:
        logger.error(f"知识检索工具调用失败: {exc}")
        return f"检索知识时发生错误: {exc}", []
