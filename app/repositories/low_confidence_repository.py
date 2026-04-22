"""低置信度事件 repository。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session


class LowConfidenceRepository:
    """负责 low_confidence_events 及关联表的数据访问。"""

    def create_event(
        self,
        session: Session,
        *,
        session_id: str | None,
        user_id: str | None,
        raw_query: str,
        normalized_query: str,
        query_fingerprint: str,
        reason: str | None,
        overall_confidence: str,
        top1_score: float | None = None,
        top2_score: float | None = None,
        avg_top3_score: float | None = None,
        query_analysis: dict[str, Any] | None = None,
        retrieval_debug: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = session.execute(
            text(
                """
                insert into low_confidence_events (
                    session_id, user_id, raw_query, normalized_query, query_fingerprint, reason,
                    overall_confidence, top1_score, top2_score, avg_top3_score, query_analysis, retrieval_debug
                ) values (
                    :session_id, :user_id, :raw_query, :normalized_query, :query_fingerprint, :reason,
                    :overall_confidence, :top1_score, :top2_score, :avg_top3_score, :query_analysis, :retrieval_debug
                )
                returning *
                """
            ).bindparams(
                bindparam("query_analysis", type_=JSONB),
                bindparam("retrieval_debug", type_=JSONB),
            ),
            {
                "session_id": session_id,
                "user_id": user_id,
                "raw_query": raw_query,
                "normalized_query": normalized_query,
                "query_fingerprint": query_fingerprint,
                "reason": reason,
                "overall_confidence": overall_confidence,
                "top1_score": top1_score,
                "top2_score": top2_score,
                "avg_top3_score": avg_top3_score,
                "query_analysis": query_analysis or {},
                "retrieval_debug": retrieval_debug or {},
            },
        )
        return dict(result.mappings().one())

    def create_event_chunk(
        self,
        session: Session,
        *,
        event_id: int,
        chunk_id: int | None,
        chunk_key_snapshot: str,
        chunk_text_snapshot: str,
        file_name_snapshot: str | None,
        page_number_snapshot: int | None,
        section_path_snapshot: str | None,
        rank_no: int,
        vector_score: float | None,
        keyword_score: float | None,
        fused_score: float | None,
        rerank_score: float | None,
        document_confidence: str,
        matched_queries: list[str] | None = None,
    ) -> dict[str, Any]:
        result = session.execute(
            text(
                """
                insert into low_confidence_event_chunks (
                    event_id, chunk_id, chunk_key_snapshot, chunk_text_snapshot, file_name_snapshot,
                    page_number_snapshot, section_path_snapshot, rank_no, vector_score, keyword_score,
                    fused_score, rerank_score, document_confidence, matched_queries
                ) values (
                    :event_id, :chunk_id, :chunk_key_snapshot, :chunk_text_snapshot, :file_name_snapshot,
                    :page_number_snapshot, :section_path_snapshot, :rank_no, :vector_score, :keyword_score,
                    :fused_score, :rerank_score, :document_confidence, :matched_queries
                )
                returning *
                """
            ).bindparams(bindparam("matched_queries", type_=JSONB)),
            {
                "event_id": event_id,
                "chunk_id": chunk_id,
                "chunk_key_snapshot": chunk_key_snapshot,
                "chunk_text_snapshot": chunk_text_snapshot,
                "file_name_snapshot": file_name_snapshot,
                "page_number_snapshot": page_number_snapshot,
                "section_path_snapshot": section_path_snapshot,
                "rank_no": rank_no,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "fused_score": fused_score,
                "rerank_score": rerank_score,
                "document_confidence": document_confidence,
                "matched_queries": matched_queries or [],
            },
        )
        return dict(result.mappings().one())

    def list_events_by_fingerprint(
        self,
        session: Session,
        query_fingerprint: str,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        result = session.execute(
            text(
                """
                select *
                from low_confidence_events
                where query_fingerprint = :query_fingerprint
                order by created_at desc
                limit :limit
                """
            ),
            {
                "query_fingerprint": query_fingerprint,
                "limit": limit,
            },
        )
        return [dict(row) for row in result.mappings().all()]

    def list_fingerprint_groups(
        self,
        session: Session,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        result = session.execute(
            text(
                """
                select
                    query_fingerprint,
                    min(normalized_query) as normalized_query,
                    count(*) as event_count,
                    min(created_at) as first_seen_at,
                    max(created_at) as last_seen_at
                from low_confidence_events
                group by query_fingerprint
                order by event_count desc, last_seen_at desc
                limit :limit
                """
            ),
            {"limit": limit},
        )
        return [dict(row) for row in result.mappings().all()]

    def list_event_chunks(
        self,
        session: Session,
        event_id: int,
    ) -> list[dict[str, Any]]:
        result = session.execute(
            text(
                """
                select *
                from low_confidence_event_chunks
                where event_id = :event_id
                order by rank_no asc
                """
            ),
            {"event_id": event_id},
        )
        return [dict(row) for row in result.mappings().all()]
