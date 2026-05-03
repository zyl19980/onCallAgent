"""将实验 chunk 索引到独立 Milvus collection。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from langchain_core.documents import Document

REPORT_DEFAULT = Path("aiops-docs/experiment/results/experiment_index_report.json")

FATAL_REQUIRED_FIELDS = [
    "chunk_id",
    "text",
    "source_id",
    "source_file",
    "collection",
    "page_start",
    "page_end",
]

METADATA_FIELDS = [
    "chunk_id",
    "source_id",
    "source_file",
    "collection",
    "page_start",
    "page_end",
    "section_path",
    "chunk_index",
    "chunk_type",
    "title",
    "fault_code",
    "parameter_name",
    "safety_level",
    "text_hash",
]

OPTIONAL_METADATA_DEFAULTS: dict[str, object] = {
    "section_path": "",
    "chunk_index": -1,
    "chunk_type": "other",
    "title": "",
    "fault_code": "",
    "parameter_name": "",
    "safety_level": "",
    "text_hash": "",
}


class DocumentWriter(Protocol):
    def add_documents(self, documents: list[Document]) -> list[str]:
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 experiment chunks 索引到 Milvus 实验 collection")
    parser.add_argument(
        "--chunks",
        default="aiops-docs/experiment/chunks/experiment_chunks.jsonl",
        help="experiment_chunks.jsonl 路径",
    )
    parser.add_argument(
        "--collection",
        default="experiment_manuals_all",
        help="目标 Milvus collection 名称",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="写入批次大小")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条 chunk")
    parser.add_argument("--dry-run", action="store_true", help="仅做校验和报告，不写入 Milvus")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="写入前删除同名实验 collection，仅作用于 --collection 指定 collection",
    )
    parser.add_argument(
        "--report",
        default=str(REPORT_DEFAULT),
        help="索引报告输出路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = index_experiment_chunks(
        chunks_path=Path(args.chunks),
        collection_name=args.collection,
        report_path=Path(args.report),
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
        drop_existing=args.drop_existing,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def index_experiment_chunks(
    *,
    chunks_path: Path,
    collection_name: str,
    report_path: Path,
    batch_size: int = 64,
    limit: int | None = None,
    dry_run: bool = False,
    drop_existing: bool = False,
    writer: DocumentWriter | None = None,
    dropper: Callable[[str], None] | None = None,
) -> dict[str, object]:
    chunks_path = chunks_path.resolve()
    report_path = report_path.resolve()

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须大于 0")

    rows = load_jsonl(chunks_path, limit=limit)
    documents, missing_required_fields, skipped_chunks = prepare_documents(rows)

    warnings: list[str] = []
    if limit is not None:
        warnings.append(f"limit_applied:{limit}")
    if dry_run:
        warnings.append("dry_run_no_milvus_write")
    if drop_existing:
        warnings.append("drop_existing_requested")
        if dry_run:
            warnings.append("drop_existing_ignored_in_dry_run")

    indexed_chunks = 0
    if not dry_run:
        if drop_existing:
            active_dropper = dropper or drop_milvus_collection
            active_dropper(collection_name)
        active_writer = writer or VectorStoreDocumentWriter(collection_name)
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            active_writer.add_documents(batch)
            indexed_chunks += len(batch)
    else:
        indexed_chunks = len(documents)

    report = {
        "total_chunks": len(rows),
        "indexed_chunks": indexed_chunks,
        "skipped_chunks": skipped_chunks,
        "collection_name": collection_name,
        "drop_existing": drop_existing,
        "batch_size": batch_size,
        "metadata_fields": list(METADATA_FIELDS),
        "missing_required_fields": dict(sorted(missing_required_fields.items())),
        "warnings": warnings,
    }
    write_json(report_path, report)
    return report


def prepare_documents(
    rows: list[dict[str, object]],
) -> tuple[list[Document], Counter[str], int]:
    documents: list[Document] = []
    missing_required_fields: Counter[str] = Counter()
    skipped_chunks = 0

    for row in rows:
        fatal_missing = find_missing_fields(row, FATAL_REQUIRED_FIELDS)
        if fatal_missing:
            missing_required_fields.update(fatal_missing)
            skipped_chunks += 1
            continue

        soft_missing = find_missing_fields(row, OPTIONAL_METADATA_DEFAULTS.keys(), allow_blank=True)
        if soft_missing:
            missing_required_fields.update(soft_missing)

        text = str(row.get("text") or "").strip()
        metadata = build_metadata(row)
        documents.append(Document(page_content=text, metadata=metadata))

    return documents, missing_required_fields, skipped_chunks


def build_metadata(row: dict[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {
        "chunk_id": str(row.get("chunk_id") or "").strip(),
        "source_id": str(row.get("source_id") or "").strip(),
        "source_file": str(row.get("source_file") or "").strip(),
        "collection": str(row.get("collection") or "").strip(),
        "page_start": int(row["page_start"]),
        "page_end": int(row["page_end"]),
    }

    for field, default in OPTIONAL_METADATA_DEFAULTS.items():
        value = row.get(field, default)
        if field == "chunk_index":
            metadata[field] = to_int(value, default=-1)
        else:
            metadata[field] = str(value or default).strip() if isinstance(default, str) else value

    return metadata


def find_missing_fields(
    row: dict[str, object],
    field_names: Iterable[str],
    *,
    allow_blank: bool = False,
) -> list[str]:
    missing: list[str] = []
    for field_name in field_names:
        value = row.get(field_name)
        if value is None:
            missing.append(str(field_name))
            continue
        if not allow_blank and isinstance(value, str) and not value.strip():
            missing.append(str(field_name))
    return missing


def to_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class VectorStoreDocumentWriter:
    def __init__(self, collection_name: str) -> None:
        from app.services.vector_store_manager import VectorStoreManager

        self.manager = VectorStoreManager.for_collection(collection_name)

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self.manager.add_documents(documents)


def drop_milvus_collection(collection_name: str) -> None:
    from pymilvus import Collection, utility

    from app.core.milvus_client import milvus_manager

    target = collection_name.strip()
    if not target:
        raise ValueError("collection_name 不能为空")

    milvus_manager.connect()
    try:
        if utility.has_collection(target):  # type: ignore[arg-type]
            try:
                Collection(target).release()
            except Exception:
                pass
            utility.drop_collection(target)
    finally:
        milvus_manager.close()


def load_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
