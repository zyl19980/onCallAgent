"""向量存储管理器。"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from loguru import logger

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


class VectorStoreManager:
    """封装 Milvus VectorStore，多 collection 复用。"""

    _instances: Dict[str, "VectorStoreManager"] = {}

    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or config.rag_collection_name
        self.vector_store: Milvus | None = None
        self._initialize_vector_store()

    @classmethod
    def for_collection(cls, collection_name: str | None = None) -> "VectorStoreManager":
        target = collection_name or config.rag_collection_name
        if target not in cls._instances:
            cls._instances[target] = cls(target)
        return cls._instances[target]

    def _initialize_vector_store(self) -> None:
        milvus_manager.ensure_collection(self.collection_name)

        connection_args = {
            "host": config.milvus_host,
            "port": config.milvus_port,
        }

        self.vector_store = Milvus(
            embedding_function=vector_embedding_service,
            collection_name=self.collection_name,
            connection_args=connection_args,
            auto_id=False,
            drop_old=False,
            text_field="content",
            vector_field="vector",
            primary_field="id",
            metadata_field="metadata",
        )

        logger.info(
            f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, collection: {self.collection_name}"
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        if self.vector_store is None:
            raise RuntimeError("VectorStore 未初始化")

        import time
        import uuid

        start_time = time.time()
        batch_size = 50
        result_ids: List[str] = []

        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            ids = [str(uuid.uuid4()) for _ in batch]
            batch_ids = self.vector_store.add_documents(batch, ids=ids)
            result_ids.extend(batch_ids)
            logger.info(
                f"Milvus 批次写入完成: collection={self.collection_name}, batch={start // batch_size + 1}, size={len(batch)}"
            )

        milvus_manager.get_collection(self.collection_name).flush()

        elapsed = time.time() - start_time
        logger.info(
            f"批量添加 {len(documents)} 个文档到 VectorStore 完成, collection={self.collection_name}, "
            f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
        )
        return result_ids

    def delete_by_source(self, file_path: str) -> int:
        try:
            collection = milvus_manager.get_collection(self.collection_name)
            expr = f'metadata["_source"] == "{file_path}"'
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            collection.flush()
            logger.info(
                f"删除文件旧数据: {file_path}, collection={self.collection_name}, 删除数量: {deleted_count}"
            )
            return deleted_count
        except Exception as exc:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {exc}")
            return 0

    def get_vector_store(self) -> Milvus:
        if self.vector_store is None:
            raise RuntimeError("VectorStore 未初始化")
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        if self.vector_store is None:
            return []
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(
                f"相似度搜索完成: collection={self.collection_name}, query='{query}', 结果数={len(docs)}"
            )
            return docs
        except Exception as exc:
            logger.error(f"相似度搜索失败: {exc}")
            return []


vector_store_manager = VectorStoreManager.for_collection(config.rag_collection_name)
