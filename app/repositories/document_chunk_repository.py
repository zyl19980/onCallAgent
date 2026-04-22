"""chunk 主数据 repository。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session


class DocumentChunkRepository:
    """负责 document_chunks 表的数据访问。"""

    def upsert_chunk(
        self,
        session: Session,
        *,
        chunk_key: str,
        document_id: int,
        chunk_index: int,
        source_text: str,
        published_text: str,
        draft_text: str | None = None,
        page_number: int | None = None,
        section_path: str | None = None,
        chunk_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        sync_status: str = "synced",
    ) -> dict[str, Any]:
        result = session.execute(
            text(
                """
                insert into document_chunks (
                    chunk_key, document_id, chunk_index, source_text, published_text,
                    draft_text, page_number, section_path, chunk_type, metadata, sync_status
                ) values (
                    :chunk_key, :document_id, :chunk_index, :source_text, :published_text,
                    :draft_text, :page_number, :section_path, :chunk_type, :metadata, :sync_status
                )
                on conflict (chunk_key) do update set
                    document_id = excluded.document_id,
                    chunk_index = excluded.chunk_index,
                    source_text = excluded.source_text,
                    published_text = excluded.published_text,
                    draft_text = excluded.draft_text,
                    page_number = excluded.page_number,
                    section_path = excluded.section_path,
                    chunk_type = excluded.chunk_type,
                    metadata = excluded.metadata,
                    sync_status = excluded.sync_status,
                    updated_at = now()
                returning *
                """
            ).bindparams(bindparam("metadata", type_=JSONB)),
            {
                "chunk_key": chunk_key,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "source_text": source_text,
                "published_text": published_text,
                "draft_text": draft_text,
                "page_number": page_number,
                "section_path": section_path,
                "chunk_type": chunk_type,
                "metadata": metadata or {},
                "sync_status": sync_status,
            },
        )
        return dict(result.mappings().one())

    def get_by_chunk_key(self, session: Session, chunk_key: str) -> dict[str, Any] | None:
        result = session.execute(
            text(
                """
                select *
                from document_chunks
                where chunk_key = :chunk_key
                """
            ),
            {"chunk_key": chunk_key},
        )
        row = result.mappings().one_or_none()
        return dict(row) if row is not None else None

    def lock_by_chunk_key(self, session: Session, chunk_key: str) -> dict[str, Any] | None:
        result = session.execute(
            text(
                """
                select *
                from document_chunks
                where chunk_key = :chunk_key
                for update
                """
            ),
            {"chunk_key": chunk_key},
        )
        row = result.mappings().one_or_none()
        return dict(row) if row is not None else None

    def save_draft(
        self,
        session: Session,
        *,
        chunk_key: str,
        draft_text: str,
    ) -> dict[str, Any] | None:
        result = session.execute(
            text(
                """
                update document_chunks
                set draft_text = :draft_text,
                    sync_status = 'draft',
                    updated_at = now()
                where chunk_key = :chunk_key
                returning *
                """
            ),
            {
                "chunk_key": chunk_key,
                "draft_text": draft_text,
            },
        )
        row = result.mappings().one_or_none()
        return dict(row) if row is not None else None

    def mark_publish_success(
        self,
        session: Session,
        *,
        chunk_id: int,
        published_text: str,
    ) -> dict[str, Any] | None:
        result = session.execute(
            text(
                """
                update document_chunks
                set published_text = :published_text,
                    draft_text = null,
                    published_version = published_version + 1,
                    sync_status = 'synced',
                    last_publish_error = null,
                    updated_at = now()
                where id = :chunk_id
                returning *
                """
            ),
            {
                "chunk_id": chunk_id,
                "published_text": published_text,
            },
        )
        row = result.mappings().one_or_none()
        return dict(row) if row is not None else None

    def mark_publish_failed(
        self,
        session: Session,
        *,
        chunk_id: int,
        last_publish_error: str,
    ) -> dict[str, Any] | None:
        result = session.execute(
            text(
                """
                update document_chunks
                set sync_status = 'publish_failed',
                    last_publish_error = :last_publish_error,
                    updated_at = now()
                where id = :chunk_id
                returning *
                """
            ),
            {
                "chunk_id": chunk_id,
                "last_publish_error": last_publish_error,
            },
        )
        row = result.mappings().one_or_none()
        return dict(row) if row is not None else None

    def list_by_document_id(self, session: Session, document_id: int) -> list[dict[str, Any]]:
        result = session.execute(
            text(
                """
                select *
                from document_chunks
                where document_id = :document_id
                order by chunk_index asc
                """
            ),
            {"document_id": document_id},
        )
        return [dict(row) for row in result.mappings().all()]
