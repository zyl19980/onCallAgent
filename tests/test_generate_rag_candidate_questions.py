import json
from pathlib import Path

from scripts.experiment.generate_rag_candidate_questions import generate_rag_candidate_questions


def test_generate_rag_candidate_questions_respects_quotas(tmp_path: Path):
    input_path = tmp_path / "annotation_pool.jsonl"
    output_path = tmp_path / "rag_candidate_questions.jsonl"
    report_path = tmp_path / "rag_candidate_generation_report.json"

    rows = [
        make_pool_row(
            chunk_id="a-1",
            source_id="source_a",
            chunk_type="troubleshooting_procedure",
            page_start=10,
            question_types=["troubleshooting_step", "symptom_cause"],
            annotation_priority="high",
            text="Troubleshooting step check wiring and inspect alarm source.",
        ),
        make_pool_row(
            chunk_id="a-2",
            source_id="source_a",
            chunk_type="alarm_fault_code",
            page_start=25,
            question_types=["parameter_or_fault_code", "symptom_cause"],
            annotation_priority="high",
            text="Alarm 5.108 indicates servo overload. Check the axis path of travel.",
            fault_code="Alarm 5.108",
        ),
        make_pool_row(
            chunk_id="a-3",
            source_id="source_a",
            chunk_type="installation_or_wiring",
            page_start=40,
            question_types=["safety_or_constraint", "parameter_or_fault_code"],
            annotation_priority="medium",
            text="Disconnect power before wiring the terminal block to the controller module.",
        ),
        make_pool_row(
            chunk_id="b-1",
            source_id="source_b",
            chunk_type="parameter_and_configuration",
            page_start=15,
            question_types=["parameter_or_fault_code"],
            annotation_priority="high",
            text="P046 [Start Source 1] must be set to 3 Serial/DSI to accept commands.",
            parameter_name="P046 [Start Source 1]",
        ),
        make_pool_row(
            chunk_id="b-2",
            source_id="source_b",
            chunk_type="safety_and_constraint",
            page_start=35,
            question_types=["safety_or_constraint"],
            annotation_priority="high",
            text="WARNING disconnect electrical power before opening the terminal box.",
        ),
        make_pool_row(
            chunk_id="b-3",
            source_id="source_b",
            chunk_type="installation_or_wiring",
            page_start=52,
            question_types=["safety_or_constraint", "parameter_or_fault_code"],
            annotation_priority="medium",
            text="The encoder module provides position feedback for the drive controller.",
        ),
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = generate_rag_candidate_questions(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        dry_run_template=True,
        source_targets={"source_a": 3, "source_b": 3},
        question_type_targets={
            "troubleshooting_step": 1,
            "symptom_cause": 1,
            "parameter_or_fault_code": 2,
            "safety_or_constraint": 1,
            "definition_or_component_lookup": 1,
        },
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 6
    assert len({row["candidate_id"] for row in rows}) == len(rows)
    assert all(row["source_chunk_ids"] for row in rows)
    assert all(row["generated_question"].strip() for row in rows)
    assert all(row["generated_answer"].strip() for row in rows)
    assert all(row["evidence_quote"].strip() for row in rows)
    assert all(row["review_status"] == "pending_review" for row in rows)

    count_by_source = {}
    count_by_qtype = {}
    for row in rows:
        count_by_source[row["source_id"]] = count_by_source.get(row["source_id"], 0) + 1
        count_by_qtype[row["suggested_question_type"]] = (
            count_by_qtype.get(row["suggested_question_type"], 0) + 1
        )

    assert count_by_source == {"source_a": 3, "source_b": 3}
    assert count_by_qtype == {
        "troubleshooting_step": 1,
        "symptom_cause": 1,
        "parameter_or_fault_code": 2,
        "safety_or_constraint": 1,
        "definition_or_component_lookup": 1,
    }
    assert report["total_candidates"] == 6
    assert report["failed_generation_count"] == 0
    assert set(report) == {
        "total_candidates",
        "count_by_source",
        "count_by_question_type",
        "failed_generation_count",
        "skipped_chunks",
        "warnings",
    }


def make_pool_row(
    *,
    chunk_id: str,
    source_id: str,
    chunk_type: str,
    page_start: int,
    question_types: list[str],
    annotation_priority: str,
    text: str,
    fault_code: str = "",
    parameter_name: str = "",
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_file": f"{source_id}.pdf",
        "page_start": page_start,
        "page_end": page_start,
        "chunk_type": chunk_type,
        "title": "Section Title",
        "text": text,
        "fault_code": fault_code,
        "parameter_name": parameter_name,
        "annotation_priority": annotation_priority,
        "recommended_question_types": question_types,
        "quality_flags": [],
    }
