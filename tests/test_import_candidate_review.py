import json
from pathlib import Path

from scripts.experiment.import_candidate_review import import_candidate_review


def test_import_candidate_review_normalizes_and_merges(tmp_path: Path):
    candidates_path = tmp_path / "rag_candidate_questions.jsonl"
    review_sheet_path = tmp_path / "candidate_review_sheet_reviewed.csv"
    manual_output = tmp_path / "manual_candidate_review.jsonl"
    reviewed_output = tmp_path / "rag_candidate_questions.reviewed.jsonl"
    report_path = tmp_path / "candidate_review_import_report.json"

    candidate_rows = [
        make_candidate("c1"),
        make_candidate("c2"),
        make_candidate("c3"),
        make_candidate("c4"),
    ]
    original_candidates_text = (
        "\n".join(json.dumps(row, ensure_ascii=False) for row in candidate_rows) + "\n"
    )
    candidates_path.write_text(original_candidates_text, encoding="utf-8")

    review_sheet_path.write_text(
        (
            "candidate_id,source_id,source_file,page_start,page_end,chunk_type,"
            "suggested_question_type,generated_question,generated_answer,evidence_quote,"
            "warnings,review_status,final_question,final_answer,final_question_type,reviewer_notes\n"
            "c1,source,source.pdf,1,1,type,param,question,answer,quote,,revise,"
            "\"Final revised question\",\"Final revised answer\",parameter_or_fault_code,needs rewrite\n"
            "c2,source,source.pdf,1,1,type,param,question,answer,quote,,reject,,,,bad candidate\n"
            "c3,source,source.pdf,1,1,type,param,question,answer,quote,,revise,"
            "\"\",\"Missing answer\",parameter_or_fault_code,incomplete\n"
        ),
        encoding="utf-8",
    )

    report = import_candidate_review(
        review_sheet_path=review_sheet_path,
        candidates_path=candidates_path,
        manual_review_output_path=manual_output,
        reviewed_output_path=reviewed_output,
        report_path=report_path,
    )

    manual_rows = [
        json.loads(line)
        for line in manual_output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reviewed_rows = [
        json.loads(line)
        for line in reviewed_output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert report["review_rows"] == 3
    assert report["imported_reviews"] == 2
    assert report["revised"] == 1
    assert report["rejected"] == 1
    assert report["approved"] == 0
    assert report["pending_after_import"] == 2
    assert len(report["invalid_review_rows"]) == 1
    assert report["invalid_review_rows"][0]["candidate_id"] == "c3"
    assert report["invalid_review_rows"][0]["reason"] == "revised_missing_final_question"

    manual_by_id = {row["candidate_id"]: row for row in manual_rows}
    assert manual_by_id["c1"]["review_status"] == "revised"
    assert manual_by_id["c2"]["review_status"] == "rejected"
    assert manual_by_id["c2"]["final_question"] == ""
    assert manual_by_id["c2"]["final_answer"] == ""

    reviewed_by_id = {row["candidate_id"]: row for row in reviewed_rows}
    assert reviewed_by_id["c1"]["review_status"] == "revised"
    assert reviewed_by_id["c1"]["final_question"] == "Final revised question"
    assert reviewed_by_id["c1"]["final_answer"] == "Final revised answer"
    assert reviewed_by_id["c1"]["final_question_type"] == "parameter_or_fault_code"
    assert reviewed_by_id["c2"]["review_status"] == "rejected"
    assert reviewed_by_id["c2"]["reviewer_notes"] == "bad candidate"
    assert reviewed_by_id["c3"]["review_status"] == "pending_review"
    assert reviewed_by_id["c4"]["review_status"] == "pending_review"

    assert candidates_path.read_text(encoding="utf-8") == original_candidates_text


def test_import_candidate_review_handles_unknown_and_duplicate_ids(tmp_path: Path):
    candidates_path = tmp_path / "rag_candidate_questions.jsonl"
    review_sheet_path = tmp_path / "candidate_review_sheet_reviewed.csv"
    manual_output = tmp_path / "manual_candidate_review.jsonl"
    reviewed_output = tmp_path / "rag_candidate_questions.reviewed.jsonl"
    report_path = tmp_path / "candidate_review_import_report.json"

    candidate_rows = [make_candidate("c1")]
    candidates_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in candidate_rows) + "\n",
        encoding="utf-8",
    )

    review_sheet_path.write_text(
        (
            "candidate_id,source_id,source_file,page_start,page_end,chunk_type,"
            "suggested_question_type,generated_question,generated_answer,evidence_quote,"
            "warnings,review_status,final_question,final_answer,final_question_type,reviewer_notes\n"
            "unknown,source,source.pdf,1,1,type,param,question,answer,quote,,reject,,,,unknown id\n"
            "c1,source,source.pdf,1,1,type,param,question,answer,quote,,reject,,,,first\n"
            "c1,source,source.pdf,1,1,type,param,question,answer,quote,,reject,,,,duplicate\n"
        ),
        encoding="utf-8",
    )

    report = import_candidate_review(
        review_sheet_path=review_sheet_path,
        candidates_path=candidates_path,
        manual_review_output_path=manual_output,
        reviewed_output_path=reviewed_output,
        report_path=report_path,
    )

    assert report["imported_reviews"] == 1
    assert report["unknown_candidate_ids"] == ["unknown"]
    assert report["duplicate_candidate_ids"] == ["c1"]


def make_candidate(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_chunk_ids": [f"chunk-{candidate_id}"],
        "source_id": "source",
        "source_file": "source.pdf",
        "page_start": 1,
        "page_end": 1,
        "chunk_type": "parameter_and_configuration",
        "generated_question": "Generated question",
        "generated_answer": "Generated answer",
        "evidence_quote": "Evidence quote",
        "suggested_question_type": "parameter_or_fault_code",
        "suggested_reasoning_hops": 1,
        "suggested_criticality": "medium",
        "generator": "template_dry_run",
        "review_status": "pending_review",
    }
