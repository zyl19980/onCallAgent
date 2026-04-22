"""通过重新索引 uploads 初始化 PostgreSQL / Milvus / rag_corpus.jsonl。"""

from __future__ import annotations

import argparse
import json

from app.services.maintenance_service import maintenance_service


def main() -> int:
    parser = argparse.ArgumentParser(description="重新索引 uploads 目录完成初始化")
    parser.add_argument(
        "--upload-dir",
        default=None,
        help="待重索引目录，默认使用 ./uploads",
    )
    args = parser.parse_args()

    result = maintenance_service.initialize_from_uploads(args.upload_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
