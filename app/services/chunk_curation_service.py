"""chunk 草稿与人工发布服务。"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.core.postgres import postgres_manager
from app.repositories.chunk_edit_history_repository import ChunkEditHistoryRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.knowledge_corpus_service import knowledge_corpus_service
from app.services.vector_embedding_service import vector_embedding_service
from app.services.vector_store_manager import VectorStoreManager


class ChunkNotFoundError(ValueError):
    """chunk 不存在。"""


class ChunkPublishError(RuntimeError):
    """chunk 发布失败。"""


class ChunkCurationService:
    """集中处理 chunk 查询、草稿保存与同步发布。"""

    def __init__(self):
        self.postgres_manager = postgres_manager
        self.document_repository = DocumentRepository()
        self.document_chunk_repository = DocumentChunkRepository()
        self.chunk_edit_history_repository = ChunkEditHistoryRepository()
        self.knowledge_corpus_service = knowledge_corpus_service
        self.vector_embedding_service = vector_embedding_service
        self.vector_store_factory = VectorStoreManager.for_collection

    def get_chunk(self, chunk_key: str) -> dict[str, Any]:
        with self.postgres_manager.session_scope() as session:
            chunk = self.document_chunk_repository.get_by_chunk_key(session, chunk_key)
            if chunk is None:
                raise ChunkNotFoundError(f"chunk not found: {chunk_key}")
            return self._enrich_chunk(session, chunk)

    def save_draft(self, chunk_key: str, draft_text: str) -> dict[str, Any]:
        normalized_text = draft_text.strip()
        if not normalized_text:
            raise ValueError("draft_text 不能为空")

        with self.postgres_manager.session_scope() as session:
            chunk = self.document_chunk_repository.save_draft(
                session,
                chunk_key=chunk_key,
                draft_text=normalized_text,
            )
            if chunk is None:
                raise ChunkNotFoundError(f"chunk not found: {chunk_key}")
            return self._enrich_chunk(session, chunk)

    def publish_chunk(
        self,
        chunk_key: str,
        *,
        editor: str = "admin",
        edit_note: str | None = None,
    ) -> dict[str, Any]:
        session = self.postgres_manager.get_session()
        locked_chunk: dict[str, Any] | None = None
        document_record: dict[str, Any] | None = None
        metadata: dict[str, Any] = {}
        milvus_updated = False
        corpus_updated = False

        try:
            locked_chunk = self.document_chunk_repository.lock_by_chunk_key(session, chunk_key)
            if locked_chunk is None:
                raise ChunkNotFoundError(f"chunk not found: {chunk_key}")

            draft_text = str(locked_chunk.get("draft_text") or "").strip()
            if not draft_text:
                raise ValueError("draft_text 为空，无法发布")

            document_record = self.document_repository.get_by_id(
                session,
                int(locked_chunk["document_id"]),
            )
            if document_record is None:
                raise RuntimeError(f"document not found: {locked_chunk['document_id']}")

            metadata = dict(locked_chunk.get("metadata") or {})
            metadata["chunk_id"] = chunk_key

            embedding = self.vector_embedding_service.embed_documents([draft_text])[0]

            vector_store_manager = self.vector_store_factory(str(document_record["collection_name"]))
            vector_store_manager.upsert_chunk(
                chunk_key=chunk_key,
                text=draft_text,
                metadata=metadata,
                embedding=embedding,
            )
            milvus_updated = True
            self.knowledge_corpus_service.upsert_chunk(
                str(document_record["collection_name"]),
                chunk_key,
                draft_text,
                metadata,
            )
            corpus_updated = True

            next_version = int(locked_chunk["published_version"]) + 1
            updated_chunk = self.document_chunk_repository.mark_publish_success(
                session,
                chunk_id=int(locked_chunk["id"]),
                published_text=draft_text,
            )
            if updated_chunk is None:
                raise RuntimeError(f"publish success update missing: {chunk_key}")

            self.chunk_edit_history_repository.create_history(
                session,
                chunk_id=int(locked_chunk["id"]),
                version_no=next_version,
                old_text=str(locked_chunk["published_text"]),
                new_text=draft_text,
                editor=editor,
                edit_note=edit_note,
                publish_status="published",
            )

            session.commit()
            logger.info(f"chunk 发布成功: chunk_key={chunk_key}, version={next_version}")
            return self._enrich_chunk(session, updated_chunk)
        except Exception as exc:
            session.rollback()
            error_message = str(exc)
            if locked_chunk is not None and document_record is not None and (milvus_updated or corpus_updated):
                rollback_error = self._restore_online_chunk(
                    collection_name=str(document_record["collection_name"]),
                    chunk_key=chunk_key,
                    published_text=str(locked_chunk["published_text"]),
                    metadata=metadata,
                    restore_milvus=milvus_updated,
                    restore_corpus=corpus_updated,
                )
                if rollback_error:
                    error_message = f"{error_message}; rollback_error={rollback_error}"
            if locked_chunk is not None:
                self._mark_publish_failed(int(locked_chunk["id"]), error_message)

            if isinstance(exc, ChunkNotFoundError):
                raise
            if isinstance(exc, ValueError):
                raise ChunkPublishError(error_message) from exc
            raise ChunkPublishError(f"chunk 发布失败: {error_message}") from exc
        finally:
            session.close()

    def list_history(self, chunk_key: str) -> list[dict[str, Any]]:
        with self.postgres_manager.session_scope() as session:
            chunk = self.document_chunk_repository.get_by_chunk_key(session, chunk_key)
            if chunk is None:
                raise ChunkNotFoundError(f"chunk not found: {chunk_key}")
            return self.chunk_edit_history_repository.list_by_chunk_id(
                session,
                int(chunk["id"]),
            )

    def _mark_publish_failed(self, chunk_id: int, error_message: str) -> None:
        failed_session = self.postgres_manager.get_session()
        try:
            self.document_chunk_repository.mark_publish_failed(
                failed_session,
                chunk_id=chunk_id,
                last_publish_error=error_message,
            )
            failed_session.commit()
        except Exception as exc:
            failed_session.rollback()
            logger.error(
                f"记录 chunk 发布失败状态失败: chunk_id={chunk_id}, error={exc}"
            )
        finally:
            failed_session.close()

    def _restore_online_chunk(
        self,
        *,
        collection_name: str,
        chunk_key: str,
        published_text: str,
        metadata: dict[str, Any],
        restore_milvus: bool,
        restore_corpus: bool,
    ) -> str | None:
        errors: list[str] = []
        if restore_milvus:
            try:
                embedding = self.vector_embedding_service.embed_documents([published_text])[0]
                self.vector_store_factory(collection_name).upsert_chunk(
                    chunk_key=chunk_key,
                    text=published_text,
                    metadata=metadata,
                    embedding=embedding,
                )
            except Exception as exc:
                errors.append(f"milvus_restore_failed={exc}")

        if restore_corpus:
            try:
                self.knowledge_corpus_service.upsert_chunk(
                    collection_name,
                    chunk_key,
                    published_text,
                    metadata,
                )
            except Exception as exc:
                errors.append(f"corpus_restore_failed={exc}")

        return "; ".join(errors) or None

    def _enrich_chunk(self, session, chunk: dict[str, Any]) -> dict[str, Any]:
        document_record = self.document_repository.get_by_id(
            session,
            int(chunk["document_id"]),
        )
        if document_record is None:
            return chunk
        return {
            **chunk,
            "collection_name": document_record["collection_name"],
            "source_path": document_record["source_path"],
            "file_name": document_record["file_name"],
            "file_ext": document_record["file_ext"],
        }


chunk_curation_service = ChunkCurationService()
