import json
from pathlib import Path

from scripts.experiment.validate_rag_dataset import validate_rag_dataset


def test_validate_rag_dataset_detects_invalid_rows_and_writes_valid_output(tmp_path: Path):
    dataset_path = tmp_path / "experiment_rag_dataset.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "experiment_rag_dataset.validated.jsonl"
    report_path = tmp_path / "rag_validation_report.json"

    chunk_rows = [
        make_chunk("chunk-1", "source_a", 3, 4),
        make_chunk("chunk-2", "source_b", 7, 7),
        make_chunk("chunk-3", "source_c", 9, 9),
        make_chunk("chunk-4", "source_d", 11, 12),
    ]
    dataset_rows = [
        make_sample(
            "dup",
            reference_chunk_ids=["chunk-1"],
            reference_evidence=[make_evidence("chunk-1", 3, 4, "Quote one.")],
            expected_page_numbers=[3, 4],
        ),
        make_sample(
            "dup",
            reference_chunk_ids=["chunk-2"],
            reference_evidence=[make_evidence("chunk-2", 7, 7, "Quote two.")],
            expected_page_numbers=[7],
        ),
        make_sample(
            "missing-user",
            user_input="",
            reference_chunk_ids=["chunk-1"],
            reference_evidence=[make_evidence("chunk-1", 3, 4, "Quote one.")],
            expected_page_numbers=[3, 4],
        ),
        make_sample(
            "missing-ref-ids",
            reference_chunk_ids=[],
            reference_evidence=[make_evidence("chunk-1", 3, 4, "Quote one.")],
            expected_page_numbers=[3, 4],
        ),
        make_sample(
            "missing-chunk",
            reference_chunk_ids=["nope"],
            reference_evidence=[make_evidence("nope", 1, 1, "Quote missing.")],
            expected_page_numbers=[1],
        ),
        make_sample(
            "bad-evidence-link",
            reference_chunk_ids=["chunk-1"],
            reference_evidence=[make_evidence("chunk-2", 7, 7, "Quote mismatch.")],
            expected_page_numbers=[3, 4],
        ),
        make_sample(
            "bad-qtype",
            question_type="bad_type",
            reference_chunk_ids=["chunk-1"],
            reference_evidence=[make_evidence("chunk-1", 3, 4, "Quote one.")],
            expected_page_numbers=[3, 4],
        ),
        make_sample(
            "abstain-ok",
            should_abstain=True,
            reference_answer="Insufficient evidence to answer safely.",
            reference_chunk_ids=[],
            reference_evidence=[],
            expected_source_files=[],
            expected_page_numbers=[],
            question_type="abstention_insufficient_evidence",
            expected_confidence="low",
            reasoning_hops="abstention",
        ),
        make_sample(
            "cross-ok",
            source_ids=["source_a", "source_b"],
            reference_chunk_ids=["chunk-1", "chunk-2"],
            reference_evidence=[
                make_evidence("chunk-1", 3, 4, "Quote one."),
                make_evidence("chunk-2", 7, 7, "Quote two."),
            ],
            expected_source_files=["source_a.pdf", "source_b.pdf"],
            expected_page_numbers=[3, 4, 7],
            question_type="cross_doc_multi",
            reasoning_hops="multi_doc",
        ),
        make_sample(
            "valid",
            reference_chunk_ids=["chunk-4"],
            reference_evidence=[make_evidence("chunk-4", 11, 12, "Full quote for valid row.")],
            expected_source_files=["source_d.pdf"],
            expected_page_numbers=[11, 12],
            question_type="troubleshooting_step",
            reasoning_hops=2,
            criticality="high",
        ),
    ]

    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in dataset_rows) + "\n",
        encoding="utf-8",
    )
    chunks_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in chunk_rows) + "\n",
        encoding="utf-8",
    )

    report = validate_rag_dataset(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=output_path,
        report_path=report_path,
    )

    valid_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert report["total_samples"] == 10
    assert report["valid_samples"] == 3
    assert report["invalid_samples"] == 7
    assert report["errors_by_type"]["duplicate_id"] == 2
    assert report["errors_by_type"]["missing_user_input"] == 1
    assert report["errors_by_type"]["missing_reference_chunk_ids"] == 1
    assert report["errors_by_type"]["missing_reference_chunk"] == 1
    assert report["errors_by_type"]["evidence_chunk_not_in_reference_chunk_ids"] == 2
    assert report["errors_by_type"]["invalid_question_type"] == 1
    assert report["count_by_should_abstain"] == {"false": 2, "true": 1}
    assert any(item.startswith("normalized_reasoning_hops:") for item in report["warnings"])

    assert len(valid_rows) == 3
    valid_by_id = {row["id"]: row for row in valid_rows}
    assert valid_by_id["abstain-ok"]["should_abstain"] is True
    assert valid_by_id["abstain-ok"]["question_type"] == "abstention_insufficient_evidence"
    assert valid_by_id["abstain-ok"]["reasoning_hops"] == "abstention"
    assert valid_by_id["abstain-ok"]["expected_confidence"] == "low"
    assert valid_by_id["cross-ok"]["question_type"] == "cross_doc_multi"
    assert valid_by_id["cross-ok"]["reasoning_hops"] == "multi_doc"
    assert len(valid_by_id["cross-ok"]["reference_chunk_ids"]) == 2
    assert valid_by_id["valid"]["reference_evidence"][0]["page_start"] == 11
    assert valid_by_id["valid"]["reference_evidence"][0]["page_end"] == 12
    assert valid_by_id["valid"]["reasoning_hops"] == "multi_chunk_same_doc"


def test_validate_rag_dataset_rejects_invalid_abstention_and_cross_doc(tmp_path: Path):
    dataset_path = tmp_path / "experiment_rag_dataset.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "experiment_rag_dataset.validated.jsonl"
    report_path = tmp_path / "rag_validation_report.json"

    chunk_rows = [
        make_chunk("chunk-1", "source_a", 3, 4),
        make_chunk("chunk-2", "source_b", 7, 7),
    ]
    dataset_rows = [
        make_sample(
            "bad-abstain-qtype",
            should_abstain=True,
            reference_answer="Need abstain answer.",
            reference_chunk_ids=[],
            reference_evidence=[],
            expected_source_files=[],
            expected_page_numbers=[],
            question_type="parameter_or_fault_code",
            expected_confidence="low",
            reasoning_hops="abstention",
        ),
        make_sample(
            "bad-abstain-hops",
            should_abstain=True,
            reference_answer="Need abstain answer.",
            reference_chunk_ids=[],
            reference_evidence=[],
            expected_source_files=[],
            expected_page_numbers=[],
            question_type="abstention_insufficient_evidence",
            expected_confidence="high",
            reasoning_hops="single_chunk",
        ),
        make_sample(
            "bad-cross-count",
            reference_chunk_ids=["chunk-1"],
            reference_evidence=[make_evidence("chunk-1", 3, 4, "Quote one.")],
            expected_page_numbers=[3, 4],
            question_type="cross_doc_multi",
            reasoning_hops="multi_doc",
        ),
        make_sample(
            "bad-cross-hops",
            source_ids=["source_a", "source_b"],
            reference_chunk_ids=["chunk-1", "chunk-2"],
            reference_evidence=[
                make_evidence("chunk-1", 3, 4, "Quote one."),
                make_evidence("chunk-2", 7, 7, "Quote two."),
            ],
            expected_source_files=["source_a.pdf", "source_b.pdf"],
            expected_page_numbers=[3, 4, 7],
            question_type="cross_doc_multi",
            reasoning_hops="single_chunk",
        ),
    ]

    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in dataset_rows) + "\n",
        encoding="utf-8",
    )
    chunks_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in chunk_rows) + "\n",
        encoding="utf-8",
    )

    report = validate_rag_dataset(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=output_path,
        report_path=report_path,
    )

    assert report["valid_samples"] == 0
    assert report["invalid_samples"] == 4
    assert report["errors_by_type"]["abstain_invalid_question_type"] == 1
    assert report["errors_by_type"]["abstain_invalid_reasoning_hops"] == 1
    assert report["errors_by_type"]["abstain_invalid_expected_confidence"] == 1
    assert report["errors_by_type"]["cross_doc_insufficient_reference_chunks"] == 1
    assert report["errors_by_type"]["cross_doc_invalid_reasoning_hops"] == 1


def make_sample(
    sample_id: str,
    *,
    split: str = "build",
    user_input: str = "Question?",
    reference_answer: str = "Reference answer long enough.",
    reference_chunk_ids: list[str] | None = None,
    reference_evidence: list[dict[str, object]] | None = None,
    expected_source_files: list[str] | None = None,
    expected_page_numbers: list[int] | None = None,
    question_type: str = "parameter_or_fault_code",
    reasoning_hops: object = 1,
    criticality: str = "medium",
    expected_confidence: str = "high",
    should_abstain: bool = False,
    annotation_status: str = "reviewed",
    source_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "split": split,
        "source_ids": source_ids if source_ids is not None else ["source_a"],
        "collections": ["source_a"],
        "user_input": user_input,
        "reference_answer": reference_answer,
        "reference_chunk_ids": reference_chunk_ids if reference_chunk_ids is not None else ["chunk-1"],
        "reference_evidence": reference_evidence if reference_evidence is not None else [make_evidence("chunk-1", 3, 4, "Quote one.")],
        "expected_source_files": expected_source_files if expected_source_files is not None else ["source_a.pdf"],
        "expected_page_numbers": expected_page_numbers if expected_page_numbers is not None else [3, 4],
        "question_type": question_type,
        "reasoning_hops": reasoning_hops,
        "criticality": criticality,
        "expected_confidence": expected_confidence,
        "should_abstain": should_abstain,
        "annotation_status": annotation_status,
        "annotator": "template_dry_run",
        "reviewer": "human",
        "notes": "",
    }


def make_evidence(chunk_id: str, page_start: int, page_end: int, quote: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "page_start": page_start,
        "page_end": page_end,
        "quote": quote,
    }


def make_chunk(chunk_id: str, source_id: str, page_start: int, page_end: int) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_file": f"{source_id}.pdf",
        "collection": source_id,
        "page_start": page_start,
        "page_end": page_end,
    }
