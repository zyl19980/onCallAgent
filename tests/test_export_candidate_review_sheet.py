import csv
import json
from pathlib import Path

from scripts.experiment.export_candidate_review_sheet import export_candidate_review_sheet


def test_export_candidate_review_sheet_outputs_csv_and_warning_jsonl(tmp_path: Path):
    input_path = tmp_path / "rag_candidate_questions.jsonl"
    csv_path = tmp_path / "candidate_review_sheet.csv"
    warnings_path = tmp_path / "warning_candidates_for_review.jsonl"

    rows = [
        {
            "candidate_id": "c1",
            "source_chunk_ids": ["chunk-1"],
            "source_id": "source_a",
            "source_file": "source_a.pdf",
            "page_start": 1,
            "page_end": 1,
            "chunk_type": "concept_and_component",
            "generated_question": "What does the encoder module refer to on the device?",
            "generated_answer": "The encoder module provides position feedback to the drive controller.",
            "evidence_quote": "The encoder module provides position feedback to the drive controller.",
            "suggested_question_type": "definition_or_component_lookup",
            "suggested_reasoning_hops": 1,
            "suggested_criticality": "low",
            "generator": "template_dry_run",
            "review_status": "pending_review",
        },
        {
            "candidate_id": "c2",
            "source_chunk_ids": ["chunk-2"],
            "source_id": "source_b",
            "source_file": "source_b.pdf",
            "page_start": 5,
            "page_end": 5,
            "chunk_type": "safety_and_constraint",
            "generated_question": "What does terminal refer to on the device?",
            "generated_answer": "voltage to the terminals.",
            "evidence_quote": "voltage to the terminals.",
            "suggested_question_type": "definition_or_component_lookup",
            "suggested_reasoning_hops": 1,
            "suggested_criticality": "low",
            "generator": "template_dry_run",
            "review_status": "pending_review",
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary = export_candidate_review_sheet(
        input_path=input_path,
        csv_output_path=csv_path,
        warnings_output_path=warnings_path,
        warnings_only=False,
    )

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    warning_rows = [
        json.loads(line)
        for line in warnings_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["input_candidates"] == 2
    assert summary["csv_rows"] == 2
    assert summary["warning_rows"] == 1
    assert len(csv_rows) == 2
    assert set(csv_rows[0].keys()) == {
        "candidate_id",
        "source_id",
        "source_file",
        "page_start",
        "page_end",
        "chunk_type",
        "suggested_question_type",
        "generated_question",
        "generated_answer",
        "evidence_quote",
        "warnings",
        "review_status",
        "final_question",
        "final_answer",
        "final_question_type",
        "reviewer_notes",
    }
    assert csv_rows[0]["review_status"] == ""
    assert csv_rows[0]["final_question"] == ""
    assert csv_rows[0]["final_answer"] == ""
    assert csv_rows[0]["final_question_type"] == ""
    assert csv_rows[0]["reviewer_notes"] == ""

    assert len(warning_rows) == 1
    assert warning_rows[0]["candidate_id"] == "c2"
    assert "definition_fallback_candidate" in warning_rows[0]["warnings"]
    assert "short_generated_answer" in warning_rows[0]["warnings"]


def test_export_candidate_review_sheet_warnings_only_filters_csv(tmp_path: Path):
    input_path = tmp_path / "rag_candidate_questions.jsonl"
    csv_path = tmp_path / "candidate_review_sheet.csv"
    warnings_path = tmp_path / "warning_candidates_for_review.jsonl"

    rows = [
        {
            "candidate_id": "clean",
            "source_chunk_ids": ["chunk-clean"],
            "source_id": "source_a",
            "source_file": "source_a.pdf",
            "page_start": 2,
            "page_end": 2,
            "chunk_type": "concept_and_component",
            "generated_question": "What does the controller module refer to on the device?",
            "generated_answer": "The controller module manages the drive logic and command handling.",
            "evidence_quote": "The controller module manages the drive logic and command handling.",
            "suggested_question_type": "definition_or_component_lookup",
            "suggested_reasoning_hops": 1,
            "suggested_criticality": "low",
            "generator": "template_dry_run",
            "review_status": "pending_review",
        },
        {
            "candidate_id": "warn",
            "source_chunk_ids": ["chunk-warn"],
            "source_id": "source_b",
            "source_file": "source_b.pdf",
            "page_start": 9,
            "page_end": 9,
            "chunk_type": "alarm_fault_code",
            "generated_question": "What safety precaution or limit should I follow for this issue on the device?",
            "generated_answer": "Alarm 5.108",
            "evidence_quote": "Alarm 5.108\nCause\nNoise",
            "suggested_question_type": "safety_or_constraint",
            "suggested_reasoning_hops": 1,
            "suggested_criticality": "high",
            "generator": "template_dry_run",
            "review_status": "pending_review",
        },
    ]
    original_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    input_path.write_text(original_text, encoding="utf-8")

    summary = export_candidate_review_sheet(
        input_path=input_path,
        csv_output_path=csv_path,
        warnings_output_path=warnings_path,
        warnings_only=True,
    )

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))

    assert summary["csv_rows"] == 1
    assert len(csv_rows) == 1
    assert csv_rows[0]["candidate_id"] == "warn"
    assert input_path.read_text(encoding="utf-8") == original_text
