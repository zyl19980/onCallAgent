import json
from pathlib import Path

from scripts.experiment.split_rag_dataset import split_rag_dataset


def test_split_rag_dataset_rebalance_outputs_target_sizes(tmp_path: Path):
    dataset_path = tmp_path / "experiment_rag_dataset.validated.jsonl"
    output_dir = tmp_path / "splits"

    rows = []
    for idx in range(8):
        rows.append(
            make_row(
                sample_id=f"rag_{idx+1:03d}",
                source_id="source_a" if idx < 4 else "source_b",
                question_type="troubleshooting_step" if idx % 2 == 0 else "parameter_or_fault_code",
                page=idx + 1,
            )
        )

    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = split_rag_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        mode="rebalance",
        build_size=3,
        dev_size=2,
        test_size=1,
        reserve_rest=True,
        seed=42,
    )

    build_rows = read_jsonl(output_dir / "rag_build.jsonl")
    dev_rows = read_jsonl(output_dir / "rag_dev.jsonl")
    test_rows = read_jsonl(output_dir / "rag_test.jsonl")
    reserve_rows = read_jsonl(output_dir / "rag_reserve.jsonl")

    assert len(build_rows) == 3
    assert len(dev_rows) == 2
    assert len(test_rows) == 1
    assert len(reserve_rows) == 2
    assert all(row["split"] == "build" for row in build_rows)
    assert all(row["split"] == "dev" for row in dev_rows)
    assert all(row["split"] == "test" for row in test_rows)
    assert all(row["split"] == "reserve" for row in reserve_rows)
    assert set(report) == {
        "total_samples",
        "count_by_split",
        "count_by_source_per_split",
        "count_by_question_type_per_split",
        "count_by_should_abstain_per_split",
        "leakage_warnings",
        "seed",
    }
    assert "false" in report["count_by_should_abstain_per_split"]["build"]


def test_split_rag_dataset_respect_existing_split_and_reports_leakage(tmp_path: Path):
    dataset_path = tmp_path / "experiment_rag_dataset.validated.jsonl"
    output_dir = tmp_path / "splits"

    rows = [
        make_row(
            sample_id="rag_001",
            source_id="source_a",
            question_type="troubleshooting_step",
            page=10,
            split="build",
            reference_chunk_ids=["shared-chunk"],
        ),
        make_row(
            sample_id="rag_002",
            source_id="source_a",
            question_type="troubleshooting_step",
            page=10,
            split="dev",
            reference_chunk_ids=["shared-chunk"],
        ),
        make_row(
            sample_id="rag_003",
            source_id="source_b",
            question_type="parameter_or_fault_code",
            page=22,
            split="reserve",
            reference_chunk_ids=["chunk-3"],
        ),
    ]

    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = split_rag_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        mode="respect_existing_split",
        seed=7,
    )

    build_rows = read_jsonl(output_dir / "rag_build.jsonl")
    dev_rows = read_jsonl(output_dir / "rag_dev.jsonl")
    reserve_rows = read_jsonl(output_dir / "rag_reserve.jsonl")

    assert len(build_rows) == 1
    assert len(dev_rows) == 1
    assert len(reserve_rows) == 1
    assert any(item.startswith("reference_chunk_cross_split:shared-chunk:") for item in report["leakage_warnings"])
    assert any(item.startswith("source_page_overlap_cross_split:source_a:page_10:") for item in report["leakage_warnings"])


def test_split_rag_dataset_rebalance_tracks_should_abstain_distribution(tmp_path: Path):
    dataset_path = tmp_path / "experiment_rag_dataset.validated.jsonl"
    output_dir = tmp_path / "splits"

    rows = [
        make_row(
            sample_id="rag_001",
            source_id="source_a",
            question_type="abstention_insufficient_evidence",
            page=1,
            should_abstain=True,
            reasoning_hops="abstention",
            expected_confidence="low",
        ),
        make_row(
            sample_id="rag_002",
            source_id="source_b",
            question_type="cross_doc_multi",
            page=2,
            source_ids=["source_b", "source_c"],
            reference_chunk_ids=["c2a", "c2b"],
            reasoning_hops="multi_doc",
        ),
        make_row(
            sample_id="rag_003",
            source_id="source_a",
            question_type="troubleshooting_step",
            page=3,
        ),
        make_row(
            sample_id="rag_004",
            source_id="source_b",
            question_type="parameter_or_fault_code",
            page=4,
        ),
    ]

    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = split_rag_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        mode="rebalance",
        build_size=1,
        dev_size=1,
        test_size=1,
        reserve_rest=True,
        seed=42,
    )

    assert set(report["count_by_should_abstain_per_split"]) == {"build", "dev", "test", "reserve"}
    assert sum(
        report["count_by_should_abstain_per_split"][split].get("true", 0)
        for split in ("build", "dev", "test", "reserve")
    ) == 1


def make_row(
    *,
    sample_id: str,
    source_id: str,
    question_type: str,
    page: int,
    split: str = "build",
    reference_chunk_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
    should_abstain: bool = False,
    reasoning_hops: str = "single_chunk",
    expected_confidence: str = "high",
) -> dict[str, object]:
    chunk_ids = reference_chunk_ids if reference_chunk_ids is not None else [f"{sample_id}-chunk"]
    return {
        "id": sample_id,
        "split": split,
        "source_ids": source_ids if source_ids is not None else [source_id],
        "collections": [source_id],
        "user_input": f"Question {sample_id}",
        "reference_answer": f"Answer {sample_id}",
        "reference_chunk_ids": chunk_ids,
        "reference_evidence": []
        if should_abstain
        else [
            {
                "chunk_id": chunk_ids[0],
                "page_start": page,
                "page_end": page,
                "quote": f"Quote {sample_id}",
            }
        ],
        "expected_source_files": [f"{source_id}.pdf"],
        "expected_page_numbers": [page],
        "question_type": question_type,
        "reasoning_hops": reasoning_hops,
        "criticality": "medium",
        "expected_confidence": expected_confidence,
        "should_abstain": should_abstain,
        "annotation_status": "reviewed",
        "annotator": "template_dry_run",
        "reviewer": "human",
        "notes": "",
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
