"""RAG Agent 服务。"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from loguru import logger

from app.agent.mcp_client import get_mcp_tools_safely
from app.config import config
from app.services.conversation_memory_service import conversation_memory_service
from app.services.hybrid_retrieval_service import RetrievalResult, hybrid_retrieval_service
from app.services.supplement_queue_service import supplement_queue_service
from app.tools import get_current_time, retrieve_knowledge


class RagAgentService:
    """RAG Agent 服务。"""

    def __init__(self, streaming: bool = True):
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()

        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.3,
            streaming=streaming,
        )

        self.tools = [retrieve_knowledge, get_current_time]
        self.mcp_tools: list = []
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成, model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        if self._agent_initialized:
            return

        self.mcp_tools = await get_mcp_tools_safely()
        all_tools = self.tools + self.mcp_tools
        self.agent = create_agent(self.model, tools=all_tools)
        self._agent_initialized = True

        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        return (
            "你是专业的运维知识助手。必须优先依据给定知识库证据回答，禁止编造。"
            " 如果证据不足，要明确说明不确定性。"
            " 如果上下文中包含“历史对话摘要”，要用它解决代词、省略和多轮问题。"
            " 回答尽量直接、结构化，适合运维场景。"
        )

    async def query(self, question: str, session_id: str) -> Dict[str, Any]:
        await self._initialize_agent()
        logger.info(f"[会话 {session_id}] 收到查询（非流式）: {question}")

        retrieval = self._retrieve(question, session_id)
        messages = self._build_messages(question, session_id, retrieval)
        result = await self.agent.ainvoke({"messages": messages})
        answer = self._extract_answer(result.get("messages", []))

        final_payload = await self._finalize_answer(session_id, question, answer, retrieval)
        logger.info(f"[会话 {session_id}] 查询完成（非流式）")
        return final_payload

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        await self._initialize_agent()
        logger.info(f"[会话 {session_id}] 收到查询（流式）: {question}")

        retrieval = self._retrieve(question, session_id)
        messages = self._build_messages(question, session_id, retrieval)
        raw_answer_parts: List[str] = []

        yield {
            "type": "search_results",
            "data": {
                "confidence": retrieval.confidence,
                "references": retrieval.references,
                "retrievalDebug": self._build_retrieval_debug(retrieval),
            },
        }

        if retrieval.confidence == "low":
            yield {
                "type": "content",
                "data": "以下内容仅供参考，当前知识库命中证据不足。\n",
            }

        async for token, metadata in self.agent.astream(
            input={"messages": messages},
            stream_mode="messages",
        ):
            if type(token).__name__ not in ("AIMessage", "AIMessageChunk"):
                continue

            content_blocks = getattr(token, "content_blocks", None)
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content = block.get("text", "")
                        if text_content:
                            raw_answer_parts.append(text_content)
                            yield {
                                "type": "content",
                                "data": text_content,
                                "node": metadata.get("langgraph_node", "unknown") if isinstance(metadata, dict) else "unknown",
                            }
                continue

            content = getattr(token, "content", "")
            if isinstance(content, str) and content:
                raw_answer_parts.append(content)
                yield {
                    "type": "content",
                    "data": content,
                    "node": metadata.get("langgraph_node", "unknown") if isinstance(metadata, dict) else "unknown",
                }

        answer = "".join(raw_answer_parts).strip()
        final_payload = await self._finalize_answer(session_id, question, answer, retrieval)

        yield {"type": "confidence", "data": final_payload["confidence"]}
        yield {"type": "references", "data": final_payload["references"]}
        yield {"type": "complete", "data": final_payload}

    def get_session_history(self, session_id: str) -> list:
        history = conversation_memory_service.get_history(session_id)
        logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
        return history

    def clear_session(self, session_id: str) -> bool:
        success = conversation_memory_service.clear(session_id)
        logger.info(f"已清除会话历史: {session_id}")
        return success

    async def cleanup(self):
        logger.info("RAG Agent 服务资源已清理")

    def _retrieve(self, question: str, session_id: str) -> RetrievalResult:
        summary = conversation_memory_service.get_summary(session_id)
        history = conversation_memory_service.get_history(session_id)
        return hybrid_retrieval_service.retrieve(question, summary=summary, recent_messages=history)

    def _build_messages(
        self,
        question: str,
        session_id: str,
        retrieval: RetrievalResult,
    ) -> List[Any]:
        memory_messages = conversation_memory_service.build_messages(session_id)
        retrieval_prompt = self._build_retrieval_prompt(retrieval)

        return [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=retrieval_prompt),
            *memory_messages,
            HumanMessage(content=question),
        ]

    def _build_retrieval_prompt(self, retrieval: RetrievalResult) -> str:
        confidence_hint = {
            "high": "当前证据充分，请给出明确答案并使用来源编号引用关键结论。",
            "medium": "当前证据有限，请说明结论边界并尽量引用来源编号。",
            "low": "当前证据不足，请先提醒回答仅供参考，再给出最稳妥建议。",
        }[retrieval.confidence]

        return (
            f"检索置信度: {retrieval.confidence}\n"
            f"{confidence_hint}\n\n"
            "以下是知识库检索到的证据，请优先依据这些证据回答：\n"
            f"{retrieval.context_text() or '当前没有命中知识库证据。'}"
        )

    def _extract_answer(self, messages: List[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                content = message.content if hasattr(message, "content") else ""
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return ""

    async def _finalize_answer(
        self,
        session_id: str,
        question: str,
        answer: str,
        retrieval: RetrievalResult,
    ) -> Dict[str, Any]:
        formatted_answer = self._format_answer(answer, retrieval)
        queued = False

        if retrieval.confidence == "low":
            queued = supplement_queue_service.enqueue(
                {
                    "session_id": session_id,
                    "question": question,
                    "query_analysis": {
                        "primary_query": retrieval.query_analysis.primary_query,
                        "keyword_query": retrieval.query_analysis.keyword_query,
                        "expanded_queries": retrieval.query_analysis.expanded_queries,
                        "keywords": retrieval.query_analysis.keywords,
                    },
                    "references": retrieval.references,
                    "reason": retrieval.low_confidence_reason,
                }
            )

        await conversation_memory_service.append_exchange(
            session_id,
            question,
            formatted_answer,
            self._summarize_with_model,
        )

        return {
            "answer": formatted_answer,
            "confidence": retrieval.confidence,
            "references": retrieval.references if retrieval.confidence in {"high", "medium"} else retrieval.references,
            "queuedForSupplement": queued,
            "retrievalDebug": self._build_retrieval_debug(retrieval),
        }

    async def _summarize_with_model(self, prompt: str) -> str:
        result = await self.model.ainvoke([HumanMessage(content=prompt)])
        content = result.content if hasattr(result, "content") else str(result)
        return content if isinstance(content, str) else str(content)

    def _format_answer(self, answer: str, retrieval: RetrievalResult) -> str:
        clean_answer = answer.strip() or "当前未能生成有效回答。"
        references_text = self._format_references(retrieval.references)

        if retrieval.confidence == "high":
            if references_text:
                return f"{clean_answer}\n\n参考来源:\n{references_text}"
            return clean_answer

        if retrieval.confidence == "medium":
            prefix = "以下结论依据有限，请结合现场信息核实。\n\n"
            suffix = f"\n\n参考来源:\n{references_text}" if references_text else ""
            return f"{prefix}{clean_answer}{suffix}"

        suffix = f"\n\n参考来源:\n{references_text}" if references_text else ""
        return f"以下内容仅供参考，当前知识库命中证据不足。\n\n{clean_answer}{suffix}"

    def _format_references(self, references: List[Dict[str, Any]]) -> str:
        lines = []
        for index, item in enumerate(references, start=1):
            parts = [str(item.get("file_name", "未知来源"))]
            if item.get("page_number"):
                parts.append(f"第{item['page_number']}页")
            if item.get("section_path"):
                parts.append(str(item["section_path"]))
            score = item.get("score")
            confidence = item.get("confidence")
            suffix = []
            if score is not None:
                suffix.append(f"score={score}")
            if confidence:
                suffix.append(f"confidence={confidence}")
            suffix_text = f" ({', '.join(suffix)})" if suffix else ""
            lines.append(f"[{index}] {' / '.join(parts)}{suffix_text}")
        return "\n".join(lines)

    def _build_retrieval_debug(self, retrieval: RetrievalResult) -> Dict[str, Any]:
        return {
            "primaryQuery": retrieval.query_analysis.primary_query,
            "expandedQueries": retrieval.query_analysis.expanded_queries,
            "keywords": retrieval.query_analysis.keywords,
            "rerankProvider": retrieval.rerank_provider,
            "overallConfidence": retrieval.confidence,
            "confidenceDetails": retrieval.confidence_debug,
            "references": retrieval.references,
        }


rag_agent_service = RagAgentService(streaming=True)
