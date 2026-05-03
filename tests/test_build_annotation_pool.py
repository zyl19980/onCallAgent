import json
from pathlib import Path

from scripts.experiment.build_annotation_pool import build_annotation_pool


def test_build_annotation_pool_filters_and_applies_quota(tmp_path: Path):
    input_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "experiment_annotation_pool.jsonl"
    report_path = tmp_path / "annotation_pool_report.json"

    rows = [
        make_chunk(
            chunk_id="front-1",
            source_id="source_a",
            chunk_type="front_matter",
            char_count=500,
            text="Table of contents",
            is_annotation_candidate=True,
            page_start=1,
            page_end=1,
        ),
        make_chunk(
            chunk_id="other-1",
            source_id="source_a",
            chunk_type="other",
            char_count=520,
            text="miscellaneous note",
            is_annotation_candidate=True,
            page_start=2,
            page_end=2,
        ),
        make_chunk(
            chunk_id="short-1",
            source_id="source_a",
            chunk_type="troubleshooting_procedure",
            char_count=150,
            text="too short",
            is_annotation_candidate=True,
            page_start=3,
            page_end=3,
        ),
        make_chunk(
            chunk_id="keep-1",
            source_id="source_a",
            chunk_type="troubleshooting_procedure",
            char_count=420,
            text=(
                "Troubleshooting procedure for alarm fault symptom cause with warning and safety "
                "checks. Parameter confirmation is required before corrective action and each step "
                "must be verified."
            ),
            is_annotation_candidate=True,
            page_start=10,
            page_end=10,
        ),
        make_chunk(
            chunk_id="keep-2",
            source_id="source_a",
            chunk_type="parameter_and_configuration",
            char_count=430,
            text="Parameter P046 default value and configuration limit with warning and parameter settings",
            is_annotation_candidate=True,
            page_start=35,
            page_end=35,
            parameter_name="P046 [Start Source x]",
        ),
        make_chunk(
            chunk_id="keep-3",
            source_id="source_b",
            chunk_type="concept_and_component",
            char_count=450,
            text="Component definition for controller module and power supply overview",
            is_annotation_candidate=True,
            page_start=12,
            page_end=12,
        ),
        make_chunk(
            chunk_id="drop-flag",
            source_id="source_b",
            chunk_type="alarm_fault_code",
            char_count=430,
            text="Alarm fault code content",
            is_annotation_candidate=False,
            page_start=14,
            page_end=14,
        ),
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = build_annotation_pool(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        max_candidates_per_source=None,
        source_quotas={"source_a": 1},
    )

    selected = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(selected) == 2
    assert sum(1 for row in selected if row["source_id"] == "source_a") == 1
    assert sum(1 for row in selected if row["source_id"] == "source_b") == 1

    selected_by_id = {row["chunk_id"]: row for row in selected}
    assert selected_by_id["keep-1"]["annotation_priority"] == "high"
    assert selected_by_id["keep-1"]["recommended_question_types"] == [
        "troubleshooting_step",
        "symptom_cause",
    ]
    assert selected_by_id["keep-3"]["annotation_priority"] in {"medium", "low"}

    assert report["total_chunks"] == 7
    assert report["candidate_chunks"] == 2
    assert report["excluded_chunks"] == 5
    assert "front_matter" not in report["count_by_chunk_type"]
    assert report["source_quota_applied"]["source_a"] == 1
    assert "chunk_type:front_matter" in report["exclusion_reasons"]
    assert "char_count_lt_200" in report["exclusion_reasons"]
    assert "is_annotation_candidate_false" in report["exclusion_reasons"]
    assert set(report["count_by_annotation_priority"]) == {"high", "medium", "low"}
    assert "recommended_question_type_distribution" in report
    assert "warnings" in report


def make_chunk(
    *,
    chunk_id: str,
    source_id: str,
    chunk_type: str,
    char_count: int,
    text: str,
    is_annotation_candidate: bool,
    page_start: int,
    page_end: int,
    parameter_name: str = "",
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_file": f"{source_id}.pdf",
        "collection": source_id,
        "page_start": page_start,
        "page_end": page_end,
        "section_path": "Section",
        "chunk_index": 0,
        "chunk_type": chunk_type,
        "title": "Title",
        "text": text,
        "fault_code": "",
        "parameter_name": parameter_name,
        "safety_level": "warning" if "warning" in text.lower() else "none",
        "char_count": char_count,
        "token_estimate": max(1, char_count // 4),
        "text_hash": f"hash-{chunk_id}",
        "is_annotation_candidate": is_annotation_candidate,
        "exclude_reason": "" if is_annotation_candidate else "too_short",
    }
