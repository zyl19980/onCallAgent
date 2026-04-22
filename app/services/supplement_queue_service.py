"""低置信度知识补充持久化服务。"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from app.core.postgres import postgres_manager
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.low_confidence_repository import LowConfidenceRepository
from app.services.hybrid_retrieval_service import RetrievalCandidate, RetrievalResult
from app.services.query_fingerprint_service import query_fingerprint_service


class SupplementQueueService:
    """将低置信度问题写入 PostgreSQL。"""

    def __init__(self):
        self.low_confidence_repository = LowConfidenceRepository()
        self.document_chunk_repository = DocumentChunkRepository()
        self.snapshot_top_n = 5

    def enqueue(
        self,
        *,
        session_id: str,
        question: str,
        retrieval: RetrievalResult,
        user_id: str | None = None,
    ) -> bool:
        if retrieval.confidence != "low":
            return False

        fingerprint = query_fingerprint_service.build(question, retrieval.query_analysis)
        confidence_debug = retrieval.confidence_debug or {}
        references = retrieval.references[: self.snapshot_top_n]
        candidates = retrieval.candidates[: self.snapshot_top_n]

        try:
            with postgres_manager.session_scope() as session:
                event_record = self.low_confidence_repository.create_event(
                    session,
                    session_id=session_id,
                    user_id=user_id,
                    raw_query=question,
                    normalized_query=fingerprint.normalized_query,
                    query_fingerprint=fingerprint.query_fingerprint,
                    reason=retrieval.low_confidence_reason,
                    overall_confidence=retrieval.confidence,
                    top1_score=self._optional_float(confidence_debug.get("top1Score")),
                    top2_score=self._optional_float(confidence_debug.get("top2Score")),
                    avg_top3_score=self._optional_float(confidence_debug.get("avgTop3Score")),
                    query_analysis=self._build_query_analysis_payload(retrieval),
                    retrieval_debug=self._build_retrieval_debug_payload(
                        retrieval=retrieval,
                        references=references,
                        candidates=candidates,
                    ),
                )

                self._persist_event_chunks(
                    session=session,
                    event_id=int(event_record["id"]),
                    candidates=candidates,
                )

            logger.info("低置信度问题已写入 PostgreSQL")
            return True
        except Exception as exc:
            logger.error(f"写入待补充队列失败: {exc}")
            return False

    def _persist_event_chunks(
        self,
        *,
        session: Any,
        event_id: int,
        candidates: list[RetrievalCandidate],
    ) -> None:
        for rank, candidate in enumerate(candidates, start=1):
            metadata = candidate.metadata or {}
            chunk_record = self.document_chunk_repository.get_by_chunk_key(session, candidate.id)
            chunk_id = int(chunk_record["id"]) if chunk_record else None

            self.low_confidence_repository.create_event_chunk(
                session,
                event_id=event_id,
                chunk_id=chunk_id,
                chunk_key_snapshot=str(candidate.id),
                chunk_text_snapshot=candidate.content,
                file_name_snapshot=self._optional_str(metadata.get("_file_name")),
                page_number_snapshot=self._optional_int(metadata.get("page_number")),
                section_path_snapshot=self._optional_str(metadata.get("section_path")),
                rank_no=rank,
                vector_score=self._optional_float(candidate.vector_score),
                keyword_score=self._optional_float(candidate.keyword_score),
                fused_score=self._optional_float(candidate.fused_score),
                rerank_score=self._optional_float(candidate.rerank_score),
                document_confidence=candidate.document_confidence,
                matched_queries=list(candidate.matched_queries),
            )

    def _build_query_analysis_payload(self, retrieval: RetrievalResult) -> Dict[str, Any]:
        return {
            "primary_query": retrieval.query_analysis.primary_query,
            "keyword_query": retrieval.query_analysis.keyword_query,
            "expanded_queries": retrieval.query_analysis.expanded_queries,
            "keywords": retrieval.query_analysis.keywords,
        }

    def _build_retrieval_debug_payload(
        self,
        *,
        retrieval: RetrievalResult,
        references: list[Dict[str, object]],
        candidates: list[RetrievalCandidate],
    ) -> Dict[str, Any]:
        return {
            "confidence_debug": retrieval.confidence_debug,
            "references": references,
            "rerank_provider": retrieval.rerank_provider,
            "candidates": [
                {
                    "id": candidate.id,
                    "file_name": candidate.metadata.get("_file_name"),
                    "page_number": candidate.metadata.get("page_number"),
                    "section_path": candidate.metadata.get("section_path"),
                    "rerank_score": round(candidate.rerank_score, 4),
                    "document_confidence": candidate.document_confidence,
                }
                for candidate in candidates
            ],
        }

    def _optional_str(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _optional_int(self, value: object) -> int | None:
        if value is None:
            return None
        return int(value)

    def _optional_float(self, value: object) -> float | None:
        if value is None:
            return None
        return float(value)


supplement_queue_service = SupplementQueueService()
