import json
from pathlib import Path

from scripts.experiment.build_rag_dataset_from_candidates import build_rag_dataset_from_candidates


def test_build_rag_dataset_from_candidates_filters_and_maps_fields(tmp_path: Path):
    candidates_path = tmp_path / "rag_candidate_questions.reviewed.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "experiment_rag_dataset.jsonl"
    report_path = tmp_path / "rag_dataset_build_report.json"

    candidate_rows = [
        make_candidate(
            "c1",
            review_status="revised",
            source_chunk_ids=["chunk-1"],
            evidence_quote="Quoted evidence one.",
            final_question="Final question one",
            final_answer="Final answer one",
            final_question_type="safety_or_constraint",
        ),
        make_candidate(
            "c2",
            review_status="approved",
            source_chunk_ids=["chunk-2"],
            evidence_quote="Quoted evidence two.",
            final_question="",
            final_answer="",
            final_question_type="",
        ),
        make_candidate(
            "c3",
            review_status="rejected",
            source_chunk_ids=["chunk-3"],
            evidence_quote="Rejected evidence.",
        ),
        make_candidate(
            "c4",
            review_status="pending_review",
            source_chunk_ids=["chunk-4"],
            evidence_quote="Pending evidence.",
        ),
        make_candidate(
            "c5",
            review_status="revised",
            source_chunk_ids=["missing-chunk"],
            evidence_quote="Missing chunk evidence.",
        ),
        make_candidate(
            "c6",
            review_status="revised",
            source_chunk_ids=["chunk-6"],
            evidence_quote="",
        ),
        make_candidate(
            "c7",
            review_status="revised",
            source_chunk_ids=["chunk-7", "chunk-8"],
            evidence_quote="Cross document evidence quote.",
            final_question="Cross doc question",
            final_answer="Cross doc answer",
            final_question_type="cross_doc_multi",
            suggested_reasoning_hops="multi_doc",
        ),
        make_candidate(
            "c8",
            review_status="revised",
            source_chunk_ids=[],
            evidence_quote="",
            final_question="Abstain question",
            final_answer="Abstain answer",
            final_question_type="abstention_insufficient_evidence",
            suggested_reasoning_hops="abstention",
            should_abstain=True,
            weak_evidence_chunk_ids=["chunk-9"],
            abstention_reason="Need live site data.",
        ),
    ]
    chunk_rows = [
        make_chunk("chunk-1", "source_a", "source_a.pdf", 3, 4),
        make_chunk("chunk-2", "source_b", "source_b.pdf", 7, 7),
        make_chunk("chunk-3", "source_c", "source_c.pdf", 9, 9),
        make_chunk("chunk-4", "source_d", "source_d.pdf", 11, 12),
        make_chunk("chunk-6", "source_f", "source_f.pdf", 15, 15),
        make_chunk("chunk-7", "source_g", "source_g.pdf", 20, 20),
        make_chunk("chunk-8", "source_h", "source_h.pdf", 33, 33),
        make_chunk("chunk-9", "source_i", "source_i.pdf", 40, 41),
    ]

    candidates_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in candidate_rows) + "\n",
        encoding="utf-8",
    )
    chunks_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in chunk_rows) + "\n",
        encoding="utf-8",
    )

    report = build_rag_dataset_from_candidates(
        candidates_path=candidates_path,
        chunks_path=chunks_path,
        output_path=output_path,
        report_path=report_path,
        include_pending=False,
    )

    dataset_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert report["converted"] == 4
    assert report["skipped_rejected"] == 1
    assert report["skipped_pending"] == 1
    assert report["invalid_missing_chunk"] == 1
    assert report["invalid_missing_evidence"] == 1
    assert report["count_by_should_abstain"] == {"false": 3, "true": 1}
    assert report["cross_doc_multi_count"] == 1
    assert report["abstention_count"] == 1

    assert len(dataset_rows) == 4
    row1 = dataset_rows[0]
    row2 = dataset_rows[1]
    row3 = dataset_rows[2]
    row4 = dataset_rows[3]

    assert row1["id"] == "rag_001"
    assert row1["split"] == "build"
    assert row1["source_ids"] == ["source_a"]
    assert row1["collections"] == ["source_a"]
    assert row1["user_input"] == "Final question one"
    assert row1["reference_answer"] == "Final answer one"
    assert row1["question_type"] == "safety_or_constraint"
    assert row1["reference_chunk_ids"] == ["chunk-1"]
    assert row1["reference_evidence"][0]["page_start"] == 3
    assert row1["reference_evidence"][0]["page_end"] == 4
    assert row1["reference_evidence"][0]["quote"] == "Quoted evidence one."
    assert row1["expected_source_files"] == ["source_a.pdf"]
    assert row1["expected_page_numbers"] == [3, 4]
    assert row1["annotation_status"] == "reviewed"
    assert row1["annotator"] == "template_dry_run"
    assert row1["reviewer"] == "human"
    assert "candidate_id=c1" in row1["notes"]

    assert row2["user_input"] == "Generated question"
    assert row2["reference_answer"] == "Generated answer"
    assert row2["question_type"] == "parameter_or_fault_code"
    assert row2["reasoning_hops"] == "multi_chunk_same_doc"

    assert row3["question_type"] == "cross_doc_multi"
    assert row3["reasoning_hops"] == "multi_doc"
    assert row3["should_abstain"] is False
    assert row3["reference_chunk_ids"] == ["chunk-7", "chunk-8"]
    assert row3["source_ids"] == ["source_g", "source_h"]
    assert row3["expected_page_numbers"] == [20, 33]

    assert row4["question_type"] == "abstention_insufficient_evidence"
    assert row4["reasoning_hops"] == "abstention"
    assert row4["should_abstain"] is True
    assert row4["expected_confidence"] == "low"
    assert row4["reference_chunk_ids"] == []
    assert row4["reference_evidence"] == []
    assert row4["expected_source_files"] == ["source.pdf"]
    assert row4["expected_page_numbers"] == [40, 41]
    assert "abstention_reason=Need live site data." in row4["notes"]


def test_build_rag_dataset_from_candidates_include_pending_adds_warning(tmp_path: Path):
    candidates_path = tmp_path / "rag_candidate_questions.reviewed.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "experiment_rag_dataset.jsonl"
    report_path = tmp_path / "rag_dataset_build_report.json"

    candidate_rows = [
        make_candidate(
            "pending-1",
            review_status="pending_review",
            source_chunk_ids=["chunk-p"],
            evidence_quote="Pending quote.",
        )
    ]
    chunk_rows = [make_chunk("chunk-p", "source_p", "source_p.pdf", 21, 21)]

    candidates_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in candidate_rows) + "\n",
        encoding="utf-8",
    )
    chunks_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in chunk_rows) + "\n",
        encoding="utf-8",
    )

    report = build_rag_dataset_from_candidates(
        candidates_path=candidates_path,
        chunks_path=chunks_path,
        output_path=output_path,
        report_path=report_path,
        include_pending=True,
    )

    dataset_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert report["converted"] == 1
    assert "include_pending_enabled_for_pilot_only" in report["warnings"]
    assert dataset_rows[0]["user_input"] == "Generated question"


def make_candidate(
    candidate_id: str,
    *,
    review_status: str,
    source_chunk_ids: list[str],
    evidence_quote: str,
    final_question: str = "",
    final_answer: str = "",
    final_question_type: str = "",
    suggested_reasoning_hops: object = 2,
    should_abstain: bool = False,
    weak_evidence_chunk_ids: list[str] | None = None,
    abstention_reason: str = "",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_chunk_ids": source_chunk_ids,
        "source_id": "source",
        "source_file": "source.pdf",
        "page_start": 1,
        "page_end": 1,
        "chunk_type": "parameter_and_configuration",
        "generated_question": "Generated question",
        "generated_answer": "Generated answer",
        "evidence_quote": evidence_quote,
        "suggested_question_type": "parameter_or_fault_code",
        "suggested_reasoning_hops": suggested_reasoning_hops,
        "suggested_criticality": "medium",
        "generator": "template_dry_run",
        "review_status": review_status,
        "final_question": final_question,
        "final_answer": final_answer,
        "final_question_type": final_question_type,
        "should_abstain": should_abstain,
        "weak_evidence_chunk_ids": weak_evidence_chunk_ids or [],
        "abstention_reason": abstention_reason,
        "reviewer_notes": "note",
    }


def make_chunk(chunk_id: str, source_id: str, source_file: str, page_start: int, page_end: int) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_file": source_file,
        "collection": source_id,
        "page_start": page_start,
        "page_end": page_end,
    }
