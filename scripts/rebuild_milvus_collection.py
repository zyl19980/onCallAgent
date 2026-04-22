"""重建 Milvus collection，使主键切换为稳定 chunk_key。"""

from __future__ import annotations

import argparse
import json

from app.services.maintenance_service import maintenance_service


def main() -> int:
    parser = argparse.ArgumentParser(description="重建 Milvus collection 并从 uploads 重新索引")
    parser.add_argument(
        "--collection-name",
        default=None,
        help="目标 collection 名，默认使用当前 RAG collection",
    )
    parser.add_argument(
        "--upload-dir",
        default=None,
        help="重索引目录，默认使用 ./uploads",
    )
    parser.add_argument(
        "--backup-collection-name",
        default=None,
        help="旧 collection 的备份名，不传则自动生成",
    )
    args = parser.parse_args()

    result = maintenance_service.rebuild_milvus_collection(
        collection_name=args.collection_name,
        upload_dir=args.upload_dir,
        backup_collection_name=args.backup_collection_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
