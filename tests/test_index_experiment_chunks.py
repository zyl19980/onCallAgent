import json
from pathlib import Path

from scripts.experiment.index_experiment_chunks import index_experiment_chunks


class FakeWriter:
    def __init__(self) -> None:
        self.calls = 0
        self.documents = []

    def add_documents(self, documents):
        self.calls += 1
        self.documents.extend(documents)
        return [doc.metadata["chunk_id"] for doc in documents]


def test_index_experiment_chunks_dry_run_does_not_write_and_limit_works(tmp_path: Path):
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    report_path = tmp_path / "report.json"
    writer = FakeWriter()
    write_jsonl(
        chunks_path,
        [
            make_chunk("chunk-1", "Text one"),
            make_chunk("chunk-2", "Text two"),
            make_chunk("chunk-3", "Text three"),
        ],
    )

    report = index_experiment_chunks(
        chunks_path=chunks_path,
        collection_name="experiment_manuals_all",
        report_path=report_path,
        batch_size=2,
        limit=2,
        dry_run=True,
        writer=writer,
    )

    assert report["total_chunks"] == 2
    assert report["indexed_chunks"] == 2
    assert report["skipped_chunks"] == 0
    assert writer.calls == 0
    assert "dry_run_no_milvus_write" in report["warnings"]
    assert "limit_applied:2" in report["warnings"]
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["collection_name"] == "experiment_manuals_all"


def test_index_experiment_chunks_metadata_fields_present(tmp_path: Path):
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    report_path = tmp_path / "report.json"
    writer = FakeWriter()
    write_jsonl(chunks_path, [make_chunk("chunk-1", "Text one")])

    report = index_experiment_chunks(
        chunks_path=chunks_path,
        collection_name="experiment_manuals_all",
        report_path=report_path,
        batch_size=10,
        dry_run=False,
        writer=writer,
    )

    assert report["indexed_chunks"] == 1
    assert writer.calls == 1
    metadata = writer.documents[0].metadata
    assert metadata["chunk_id"] == "chunk-1"
    assert metadata["source_id"] == "source-a"
    assert metadata["page_start"] == 3
    assert metadata["page_end"] == 4
    assert set(report["metadata_fields"]) >= {"chunk_id", "source_id", "page_start", "page_end"}


def test_index_experiment_chunks_skips_missing_text_or_chunk_id(tmp_path: Path):
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    report_path = tmp_path / "report.json"
    writer = FakeWriter()
    write_jsonl(
        chunks_path,
        [
            make_chunk("chunk-1", "Text one"),
            make_chunk("", "Text missing id"),
            make_chunk("chunk-3", ""),
        ],
    )

    report = index_experiment_chunks(
        chunks_path=chunks_path,
        collection_name="experiment_manuals_all",
        report_path=report_path,
        batch_size=10,
        dry_run=False,
        writer=writer,
    )

    assert report["total_chunks"] == 3
    assert report["indexed_chunks"] == 1
    assert report["skipped_chunks"] == 2
    assert report["missing_required_fields"]["chunk_id"] == 1
    assert report["missing_required_fields"]["text"] == 1


def test_index_experiment_chunks_drop_existing_only_targets_requested_collection(tmp_path: Path):
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    report_path = tmp_path / "report.json"
    writer = FakeWriter()
    dropped = []
    write_jsonl(chunks_path, [make_chunk("chunk-1", "Text one")])

    report = index_experiment_chunks(
        chunks_path=chunks_path,
        collection_name="experiment_manuals_all",
        report_path=report_path,
        batch_size=10,
        dry_run=False,
        drop_existing=True,
        writer=writer,
        dropper=dropped.append,
    )

    assert report["drop_existing"] is True
    assert dropped == ["experiment_manuals_all"]
    assert writer.calls == 1


def make_chunk(chunk_id: str, text: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": "source-a",
        "source_file": "source-a.pdf",
        "collection": "experiment_manuals_all",
        "page_start": 3,
        "page_end": 4,
        "section_path": "Section > Subsection",
        "chunk_index": 7,
        "chunk_type": "parameter_and_configuration",
        "title": "Sample chunk title",
        "text": text,
        "fault_code": "",
        "parameter_name": "P001",
        "safety_level": "medium",
        "text_hash": "hash-001",
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
