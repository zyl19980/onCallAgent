"""向量索引服务模块。"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.config import config
from app.core.postgres import postgres_manager
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.document_splitter_service import document_splitter_service
from app.services.knowledge_corpus_service import knowledge_corpus_service
from app.services.pdf_parser_service import pdf_parser_service
from app.services.vector_store_manager import VectorStoreManager


class IndexingResult:
    """索引结果类。"""

    def __init__(self):
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        self.failed_files: Dict[str, str] = {}

    def increment_success_count(self):
        self.success_count += 1

    def increment_fail_count(self):
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str):
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }


class VectorIndexService:
    """向量索引服务。"""

    SUPPORTED_EXTENSIONS = ("*.txt", "*.md", "*.pdf")

    def __init__(self):
        self.upload_path = "./uploads"
        self.document_repository = DocumentRepository()
        self.document_chunk_repository = DocumentChunkRepository()
        logger.info("向量索引服务初始化完成")

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()
            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"目录不存在或不是有效目录: {target_path}")

            result.directory_path = str(dir_path)
            files = []
            for pattern in self.SUPPORTED_EXTENSIONS:
                files.extend(dir_path.glob(pattern))

            if not files:
                logger.warning(f"目录中没有找到支持的文件: {target_path}")
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"开始索引目录: {target_path}, 找到 {len(files)} 个文件")

            for file_path in files:
                try:
                    self.index_single_file(str(file_path))
                    result.increment_success_count()
                    logger.info(f"✓ 文件索引成功: {file_path.name}")
                except Exception as exc:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(exc))
                    logger.error(f"✗ 文件索引失败: {file_path.name}, 错误: {exc}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()
            return result

        except Exception as exc:
            logger.error(f"索引目录失败: {exc}")
            result.success = False
            result.error_message = str(exc)
            result.end_time = datetime.now()
            return result

    def index_single_file(self, file_path: str, collection_name: str | None = None):
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        logger.info(f"开始索引文件: {path}")
        normalized_path = path.as_posix()
        target_collection = collection_name or config.rag_collection_name
        vector_store_manager = VectorStoreManager.for_collection(target_collection)

        try:
            raw_bytes = path.read_bytes()
            file_hash = self._build_file_hash(raw_bytes)

            if path.suffix.lower() == ".pdf":
                pages = pdf_parser_service.parse(normalized_path)
                documents = document_splitter_service.split_document(
                    "",
                    normalized_path,
                    pages=pages,
                )
            else:
                content = raw_bytes.decode("utf-8")
                logger.info(f"读取文件: {path}, 内容长度: {len(content)} 字符")
                documents = document_splitter_service.split_document(content, normalized_path)

            self._decorate_documents(path, documents)
            self._upsert_postgres_metadata(
                path=path,
                file_hash=file_hash,
                collection_name=target_collection,
                documents=documents,
            )

            vector_store_manager.delete_by_source(normalized_path)
            knowledge_corpus_service.remove_source(normalized_path, target_collection)

            if documents:
                vector_store_manager.add_documents(documents)
                knowledge_corpus_service.replace_documents_for_source(
                    normalized_path,
                    documents,
                    target_collection,
                )
                logger.info(
                    f"文件索引完成: {file_path}, collection={target_collection}, 共 {len(documents)} 个分片"
                )
            else:
                logger.warning(f"文件内容为空或无法分割: {file_path}")

        except Exception as exc:
            logger.error(f"索引文件失败: {file_path}, 错误: {exc}")
            raise RuntimeError(f"索引文件失败: {exc}") from exc

    def _upsert_postgres_metadata(
        self,
        *,
        path: Path,
        file_hash: str,
        collection_name: str,
        documents: list,
    ) -> None:
        with postgres_manager.session_scope() as session:
            document_record = self.document_repository.upsert_document(
                session,
                collection_name=collection_name,
                source_path=path.as_posix(),
                file_name=path.name,
                file_ext=path.suffix.lower(),
                file_hash=file_hash,
            )

            for doc in documents:
                metadata = dict(doc.metadata)
                self.document_chunk_repository.upsert_chunk(
                    session,
                    chunk_key=str(metadata["chunk_id"]),
                    document_id=int(document_record["id"]),
                    chunk_index=int(metadata["chunk_index"]),
                    source_text=doc.page_content,
                    published_text=doc.page_content,
                    page_number=self._optional_int(metadata.get("page_number")),
                    section_path=self._optional_str(metadata.get("section_path")),
                    chunk_type=self._optional_str(metadata.get("chunk_type")),
                    metadata=metadata,
                )

    def _decorate_documents(self, path: Path, documents: list) -> None:
        for index, doc in enumerate(documents):
            doc.metadata.setdefault("_source", path.as_posix())
            doc.metadata.setdefault("_extension", path.suffix.lower())
            doc.metadata.setdefault("_file_name", path.name)
            doc.metadata["chunk_index"] = index
            doc.metadata["chunk_id"] = self._build_chunk_key(path.as_posix(), index)

    def _build_file_hash(self, raw_bytes: bytes) -> str:
        return hashlib.sha256(raw_bytes).hexdigest()

    def _build_chunk_key(self, source_path: str, chunk_index: int) -> str:
        return f"{source_path}::{chunk_index}"

    def _optional_str(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _optional_int(self, value: object) -> int | None:
        if value is None:
            return None
        return int(value)


vector_index_service = VectorIndexService()
