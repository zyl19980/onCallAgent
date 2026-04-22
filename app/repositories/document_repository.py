"""文档主数据 repository。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session


class DocumentRepository:
    """负责 documents 表的数据访问。"""

    def upsert_document(
        self,
        session: Session,
        *,
        collection_name: str,
        source_path: str,
        file_name: str,
        file_ext: str,
        file_hash: str,
        status: str = "active",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = session.execute(
            text(
                """
                insert into documents (
                    collection_name, source_path, file_name, file_ext, file_hash, status, metadata
                ) values (
                    :collection_name, :source_path, :file_name, :file_ext, :file_hash, :status, :metadata
                )
                on conflict (collection_name, source_path) do update set
                    file_name = excluded.file_name,
                    file_ext = excluded.file_ext,
                    file_hash = excluded.file_hash,
                    status = excluded.status,
                    metadata = excluded.metadata,
                    updated_at = now()
                returning *
                """
            ).bindparams(bindparam("metadata", type_=JSONB)),
            {
                "collection_name": collection_name,
                "source_path": source_path,
                "file_name": file_name,
                "file_ext": file_ext,
                "file_hash": file_hash,
                "status": status,
                "metadata": metadata or {},
            },
        )
        return dict(result.mappings().one())

    def get_by_collection_and_source(
        self,
        session: Session,
        *,
        collection_name: str,
        source_path: str,
    ) -> dict[str, Any] | None:
        result = session.execute(
            text(
                """
                select *
                from documents
                where collection_name = :collection_name
                  and source_path = :source_path
                """
            ),
            {
                "collection_name": collection_name,
                "source_path": source_path,
            },
        )
        row = result.mappings().one_or_none()
        return dict(row) if row is not None else None

    def get_by_id(self, session: Session, document_id: int) -> dict[str, Any] | None:
        result = session.execute(
            text(
                """
                select *
                from documents
                where id = :document_id
                """
            ),
            {"document_id": document_id},
        )
        row = result.mappings().one_or_none()
        return dict(row) if row is not None else None
