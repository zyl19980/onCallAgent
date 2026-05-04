import json
from collections import Counter
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
        "reused_chunk_count",
        "excluded_existing_chunk_count",
        "failed_generation_count",
        "skipped_chunks",
        "warnings",
    }
    assert report["reused_chunk_count"] == 0
    assert report["excluded_existing_chunk_count"] == 0


def test_generate_batch2_excludes_reviewed_chunks_and_builds_special_types(tmp_path: Path):
    input_path = tmp_path / "annotation_pool.jsonl"
    output_path = tmp_path / "rag_candidate_questions_batch2.jsonl"
    report_path = tmp_path / "rag_candidate_generation_report_batch2.json"
    reviewed_path = tmp_path / "rag_candidate_questions.reviewed.jsonl"

    rows = [
        make_pool_row(
            chunk_id="g-1",
            source_id="grundfos",
            chunk_type="troubleshooting_procedure",
            page_start=10,
            question_types=["troubleshooting_step", "symptom_cause"],
            annotation_priority="high",
            text="Check the pump wiring and inspect the alarm source before restarting the pump.",
        ),
        make_pool_row(
            chunk_id="g-2",
            source_id="grundfos",
            chunk_type="safety_and_constraint",
            page_start=20,
            question_types=["safety_or_constraint"],
            annotation_priority="high",
            text="WARNING disconnect power before opening the pump terminal box.",
        ),
        make_pool_row(
            chunk_id="g-3",
            source_id="grundfos",
            chunk_type="parameter_and_configuration",
            page_start=24,
            question_types=["parameter_or_fault_code"],
            annotation_priority="medium",
            text="Parameter G12 defines the pump start behavior in the controller.",
            parameter_name="Parameter G12",
        ),
        make_pool_row(
            chunk_id="a-1",
            source_id="abb",
            chunk_type="parameter_and_configuration",
            page_start=30,
            question_types=["parameter_or_fault_code"],
            annotation_priority="high",
            text="Parameter P01 defines the startup mode for the motor controller.",
            parameter_name="P01",
        ),
        make_pool_row(
            chunk_id="a-2",
            source_id="abb",
            chunk_type="concept_and_component",
            page_start=40,
            question_types=["definition_or_component_lookup"],
            annotation_priority="medium",
            text="The encoder module provides feedback to the motor control unit.",
        ),
        make_pool_row(
            chunk_id="h-1",
            source_id="haas",
            chunk_type="alarm_fault_code",
            page_start=50,
            question_types=["symptom_cause", "parameter_or_fault_code"],
            annotation_priority="high",
            text="Alarm 5.108 indicates servo overload and may be caused by axis obstruction.",
            fault_code="Alarm 5.108",
        ),
        make_pool_row(
            chunk_id="r-1",
            source_id="rockwell",
            chunk_type="safety_and_constraint",
            page_start=60,
            question_types=["safety_or_constraint"],
            annotation_priority="high",
            text="Do not service the drive until input power is isolated and verified.",
        ),
        make_pool_row(
            chunk_id="r-2",
            source_id="rockwell",
            chunk_type="troubleshooting_procedure",
            page_start=62,
            question_types=["troubleshooting_step", "symptom_cause"],
            annotation_priority="medium",
            text="Check the drive wiring and verify the control signal before replacing the module.",
        ),
        make_pool_row(
            chunk_id="s-1",
            source_id="siemens",
            chunk_type="troubleshooting_procedure",
            page_start=70,
            question_types=["troubleshooting_step", "symptom_cause"],
            annotation_priority="high",
            text="Inspect the PLC input wiring and verify field signals before replacing the module.",
        ),
        make_pool_row(
            chunk_id="s-2",
            source_id="siemens",
            chunk_type="parameter_and_configuration",
            page_start=80,
            question_types=["parameter_or_fault_code"],
            annotation_priority="medium",
            text="The address parameter defines which signal map the CPU will expose to the module.",
            parameter_name="address parameter",
        ),
    ]
    input_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    reviewed_rows = [
        {
            "candidate_id": "old-1",
            "source_chunk_ids": ["g-1"],
            "review_status": "revised",
        }
    ]
    reviewed_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in reviewed_rows) + "\n",
        encoding="utf-8",
    )

    report = generate_rag_candidate_questions(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        dry_run_template=True,
        source_targets={"grundfos": 1, "abb": 2, "haas": 1, "rockwell": 1, "siemens": 2},
        question_type_targets={
            "troubleshooting_step": 1,
            "symptom_cause": 1,
            "parameter_or_fault_code": 1,
            "safety_or_constraint": 1,
            "definition_or_component_lookup": 1,
            "cross_doc_multi": 1,
            "abstention_insufficient_evidence": 1,
        },
        reviewed_candidates_path=reviewed_path,
        exclude_reviewed_candidates=True,
        candidate_prefix="batch2",
        normal_count=5,
        cross_doc_count=1,
        abstention_count=1,
    )

    generated = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(generated) == 7
    assert len({row["candidate_id"] for row in generated}) == 7
    assert all(row["candidate_id"].startswith("batch2::") for row in generated)
    assert all("g-1" not in (row.get("source_chunk_ids") or []) for row in generated)
    assert all("g-1" not in (row.get("weak_evidence_chunk_ids") or []) for row in generated)

    cross_doc_rows = [row for row in generated if row["suggested_question_type"] == "cross_doc_multi"]
    assert len(cross_doc_rows) == 1
    assert len(cross_doc_rows[0]["source_chunk_ids"]) >= 2
    assert cross_doc_rows[0]["suggested_reasoning_hops"] == "multi_doc"

    abstention_rows = [
        row for row in generated if row["suggested_question_type"] == "abstention_insufficient_evidence"
    ]
    assert len(abstention_rows) == 1
    assert abstention_rows[0]["should_abstain"] is True
    assert abstention_rows[0]["abstention_reason"]

    counts = Counter(row["suggested_question_type"] for row in generated)
    assert counts == {
        "troubleshooting_step": 1,
        "symptom_cause": 1,
        "parameter_or_fault_code": 1,
        "safety_or_constraint": 1,
        "definition_or_component_lookup": 1,
        "cross_doc_multi": 1,
        "abstention_insufficient_evidence": 1,
    }
    assert report["excluded_existing_chunk_count"] == 1
    assert report["reused_chunk_count"] == 0


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
