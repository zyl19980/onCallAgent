"""初始化、迁移与运维辅助服务。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pymilvus import utility

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_index_service import vector_index_service
from app.services.vector_store_manager import VectorStoreManager


class MaintenanceService:
    """封装初始化重建类运维动作。"""

    def initialize_from_uploads(self, upload_dir: str | None = None) -> dict[str, Any]:
        target_dir = upload_dir or vector_index_service.upload_path
        result = vector_index_service.index_directory(target_dir)
        payload = result.to_dict()
        payload["mode"] = "reindex_uploads"
        return payload

    def rebuild_milvus_collection(
        self,
        *,
        collection_name: str | None = None,
        upload_dir: str | None = None,
        backup_collection_name: str | None = None,
    ) -> dict[str, Any]:
        target_collection = collection_name or config.rag_collection_name
        source_dir = Path(upload_dir or vector_index_service.upload_path).resolve()
        if not source_dir.exists() or not source_dir.is_dir():
            raise ValueError(f"目录不存在或不是有效目录: {source_dir}")

        milvus_manager.connect()
        result: dict[str, Any] = {
            "collection_name": target_collection,
            "source_directory": str(source_dir),
            "backup_collection_name": None,
            "total_files": 0,
            "success_count": 0,
            "fail_count": 0,
            "failed_files": {},
            "success": False,
        }

        if utility.has_collection(target_collection):
            backup_name = backup_collection_name or self._build_backup_collection_name(target_collection)
            if utility.has_collection(backup_name):
                raise ValueError(f"备份 collection 已存在: {backup_name}")

            utility.rename_collection(target_collection, backup_name)
            self._clear_collection_caches(target_collection)
            result["backup_collection_name"] = backup_name
            logger.info(
                f"Milvus collection 已重命名为备份: {target_collection} -> {backup_name}"
            )

        files: list[Path] = []
        for pattern in vector_index_service.SUPPORTED_EXTENSIONS:
            files.extend(source_dir.glob(pattern))
        files = sorted({item.resolve() for item in files}, key=lambda item: item.as_posix())
        result["total_files"] = len(files)

        for file_path in files:
            try:
                vector_index_service.index_single_file(
                    str(file_path),
                    collection_name=target_collection,
                )
                result["success_count"] += 1
            except Exception as exc:
                result["fail_count"] += 1
                result["failed_files"][file_path.as_posix()] = str(exc)
                logger.error(f"重建 collection 时文件重索引失败: {file_path}, error={exc}")

        result["success"] = result["fail_count"] == 0
        return result

    def _build_backup_collection_name(self, collection_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{collection_name}_legacy_{timestamp}"

    def _clear_collection_caches(self, collection_name: str) -> None:
        milvus_manager._collections.pop(collection_name, None)
        VectorStoreManager._instances.pop(collection_name, None)


maintenance_service = MaintenanceService()
