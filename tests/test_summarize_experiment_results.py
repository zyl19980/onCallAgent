import csv
import json
from pathlib import Path

from scripts.experiment.summarize_experiment_results import summarize_experiment_results


def test_summarize_experiment_results_generates_outputs_and_reads_metrics(tmp_path: Path):
    files = make_test_files(tmp_path)
    output_dir = tmp_path / "reports"

    report = summarize_experiment_results(output_dir=output_dir, files=files)

    assert report["missing_files"]
    assert any(
        "confidence_eval_score_threshold_dense_current_rerank_build.json" in item for item in report["warnings"]
    )

    progress_path = output_dir / "experiment_progress_summary.md"
    dataset_csv = output_dir / "dataset_summary.csv"
    retrieval_csv = output_dir / "retrieval_results_summary.csv"
    confidence_csv = output_dir / "confidence_results_summary.csv"
    findings_json = output_dir / "experiment_findings.json"
    next_steps = output_dir / "next_steps.md"

    for path in [progress_path, dataset_csv, retrieval_csv, confidence_csv, findings_json, next_steps]:
        assert path.exists()

    progress_text = progress_path.read_text(encoding="utf-8")
    assert "Experiment Scope" in progress_text
    assert "Retrieval Findings" in progress_text
    assert "Confidence Findings" in progress_text

    with dataset_csv.open("r", encoding="utf-8", newline="") as fh:
        dataset_rows = list(csv.DictReader(fh))
    assert {"section", "metric", "value", "notes"} == set(dataset_rows[0].keys())
    assert any(row["metric"] == "source_count" and row["value"] == "2" for row in dataset_rows)
    assert any(row["metric"] == "split_build" and row["value"] == "30" for row in dataset_rows)

    with retrieval_csv.open("r", encoding="utf-8", newline="") as fh:
        retrieval_rows = list(csv.DictReader(fh))
    assert "mrr" in retrieval_rows[0]
    build_current = next(row for row in retrieval_rows if row["experiment_name"] == "dense_current_rerank_build")
    assert build_current["hit_at_10"] == "0.766667"
    assert build_current["delta_hit_at_10"] == "0.133334"
    assert build_current["gold_in_candidate_not_final_count"] == "4"

    with confidence_csv.open("r", encoding="utf-8", newline="") as fh:
        confidence_rows = list(csv.DictReader(fh))
    assert "high_confidence_precision" in confidence_rows[0]
    rank_build = next(
        row for row in confidence_rows if row["strategy"] == "rank_and_margin" and row["split"] == "build"
    )
    assert rank_build["high_confidence_precision"] == "1.0"
    assert rank_build["low_confidence_error_capture"] == "0.571429"

    findings = json.loads(findings_json.read_text(encoding="utf-8"))
    assert "dataset_status" in findings
    assert "retrieval_findings" in findings
    assert "confidence_findings" in findings
    assert "files_used" in findings
    assert "generated_at" in findings

    next_steps_text = next_steps.read_text(encoding="utf-8")
    assert "Priority 1" in next_steps_text
    assert "Priority 4" in next_steps_text


def make_test_files(root: Path) -> dict[str, str]:
    write_json(
        root / "aiops-docs/experiment/sources/source_manifest.json",
        {"total_sources": 2, "sources": [{}, {}]},
    )
    write_json(
        root / "aiops-docs/experiment/chunks/chunk_build_report.json",
        {
            "total_chunks": 100,
            "count_by_source": {"source_a": 60, "source_b": 40},
            "count_by_chunk_type": {"parameter_and_configuration": 70, "troubleshooting_procedure": 30},
        },
    )
    write_json(
        root / "aiops-docs/experiment/chunks/annotation_pool_report.json",
        {
            "candidate_chunks": 20,
            "count_by_annotation_priority": {"high": 12, "medium": 7, "low": 1},
        },
    )
    write_json(
        root / "aiops-docs/experiment/rag/candidate_review_import_report.json",
        {
            "review_rows": 10,
            "revised": 8,
            "rejected": 2,
        },
    )
    write_json(
        root / "aiops-docs/experiment/rag/rag_validation_report.json",
        {
            "valid_samples": 8,
            "invalid_samples": 0,
        },
    )
    write_json(
        root / "aiops-docs/experiment/rag/splits/rag_split_report.json",
        {
            "count_by_split": {"build": 30, "dev": 20, "test": 0, "reserve": 14},
        },
    )
    write_jsonl(root / "aiops-docs/experiment/rag/rag_candidate_questions.jsonl", [{"id": "c1"}, {"id": "c2"}])
    write_jsonl(root / "aiops-docs/experiment/rag/experiment_rag_dataset.validated.jsonl", [{"id": "r1"}] * 8)

    write_json(
        root / "aiops-docs/experiment/results/dense_no_rerank_build.json",
        make_retrieval_payload("dense_no_rerank_build", "none", 0.633333, 0.434444, 8),
    )
    write_json(
        root / "aiops-docs/experiment/results/dense_current_rerank_build.json",
        make_retrieval_payload("dense_current_rerank_build", "current", 0.766667, 0.537593, 4, 11, 3),
    )
    write_json(
        root / "aiops-docs/experiment/results/dense_no_rerank_dev.json",
        make_retrieval_payload("dense_no_rerank_dev", "none", 0.4, 0.20256, 9),
    )
    write_json(
        root / "aiops-docs/experiment/results/dense_current_rerank_dev.json",
        make_retrieval_payload("dense_current_rerank_dev", "current", 0.55, 0.418333, 6, 12, 1),
    )

    write_json(
        root / "aiops-docs/experiment/results/confidence_eval_dense_current_rerank_build.json",
        make_confidence_payload("dense_current_rerank_build", "rank_and_margin", "build", 1.0, 0.571429, {"high": 7, "medium": 11, "low": 12}),
    )
    write_json(
        root / "aiops-docs/experiment/results/confidence_eval_dense_current_rerank_dev.json",
        make_confidence_payload("dense_current_rerank_dev", "rank_and_margin", "dev", 1.0, 0.666667, {"high": 4, "medium": 5, "low": 11}),
    )
    write_json(
        root / "aiops-docs/experiment/results/confidence_eval_system_top3_support_dense_current_rerank_build.json",
        make_confidence_payload(
            "dense_current_rerank_build",
            "system_top3_support",
            "build",
            0.8,
            0.857143,
            {"high": 5, "medium": 0, "low": 25},
            score_direction="higher_is_better",
            strong_threshold=0.78,
            support_threshold=0.45,
            high_avg_threshold=0.55,
        ),
    )
    write_json(
        root / "aiops-docs/experiment/results/confidence_eval_system_top3_support_dense_current_rerank_dev.json",
        make_confidence_payload(
            "dense_current_rerank_dev",
            "system_top3_support",
            "dev",
            1.0,
            1.0,
            {"high": 5, "medium": 0, "low": 15},
            score_direction="higher_is_better",
            strong_threshold=0.78,
            support_threshold=0.45,
            high_avg_threshold=0.55,
        ),
    )
    write_json(
        root / "aiops-docs/experiment/results/confidence_tuning_system_top3_support_build.json",
        {
            "best_config": {
                "score_direction": "lower_is_better",
                "high_confidence_precision": 0.909091,
                "low_confidence_error_capture": 0.571429,
                "confidence_accuracy": 0.533333,
                "count_high": 11,
                "count_medium": 4,
                "count_low": 15,
                "low_ratio": 0.5,
                "strong_threshold": 0.68,
                "support_threshold": 0.57,
                "high_avg_threshold": 0.6,
            }
        },
    )
    write_json(
        root / "aiops-docs/experiment/results/confidence_tuning_system_top3_support_dev.json",
        {
            "best_config": {
                "score_direction": "higher_is_better",
                "high_confidence_precision": 1.0,
                "low_confidence_error_capture": 0.777778,
                "confidence_accuracy": 0.65,
                "count_high": 5,
                "count_medium": 3,
                "count_low": 12,
                "low_ratio": 0.6,
                "strong_threshold": 0.6,
                "support_threshold": 0.27,
                "high_avg_threshold": 0.67,
            }
        },
    )

    return {
        "source_manifest": str(root / "aiops-docs/experiment/sources/source_manifest.json"),
        "chunk_build_report": str(root / "aiops-docs/experiment/chunks/chunk_build_report.json"),
        "annotation_pool_report": str(root / "aiops-docs/experiment/chunks/annotation_pool_report.json"),
        "candidate_questions": str(root / "aiops-docs/experiment/rag/rag_candidate_questions.jsonl"),
        "reviewed_candidates": str(root / "aiops-docs/experiment/rag/rag_candidate_questions.reviewed.jsonl"),
        "candidate_review_import_report": str(root / "aiops-docs/experiment/rag/candidate_review_import_report.json"),
        "rag_dataset": str(root / "aiops-docs/experiment/rag/experiment_rag_dataset.jsonl"),
        "validated_dataset": str(root / "aiops-docs/experiment/rag/experiment_rag_dataset.validated.jsonl"),
        "rag_validation_report": str(root / "aiops-docs/experiment/rag/rag_validation_report.json"),
        "rag_split_report": str(root / "aiops-docs/experiment/rag/splits/rag_split_report.json"),
        "rag_build": str(root / "aiops-docs/experiment/rag/splits/rag_build.jsonl"),
        "rag_dev": str(root / "aiops-docs/experiment/rag/splits/rag_dev.jsonl"),
        "rag_test": str(root / "aiops-docs/experiment/rag/splits/rag_test.jsonl"),
        "rag_reserve": str(root / "aiops-docs/experiment/rag/splits/rag_reserve.jsonl"),
        "retrieval_dense_no_rerank_build": str(root / "aiops-docs/experiment/results/dense_no_rerank_build.json"),
        "retrieval_dense_no_rerank_dev": str(root / "aiops-docs/experiment/results/dense_no_rerank_dev.json"),
        "retrieval_dense_current_rerank_build": str(root / "aiops-docs/experiment/results/dense_current_rerank_build.json"),
        "retrieval_dense_current_rerank_dev": str(root / "aiops-docs/experiment/results/dense_current_rerank_dev.json"),
        "retrieval_comparison": str(root / "aiops-docs/experiment/results/retrieval_experiment_comparison.json"),
        "confidence_rank_and_margin_build": str(root / "aiops-docs/experiment/results/confidence_eval_dense_current_rerank_build.json"),
        "confidence_rank_and_margin_dev": str(root / "aiops-docs/experiment/results/confidence_eval_dense_current_rerank_dev.json"),
        "confidence_system_top3_support_build": str(root / "aiops-docs/experiment/results/confidence_eval_system_top3_support_dense_current_rerank_build.json"),
        "confidence_system_top3_support_dev": str(root / "aiops-docs/experiment/results/confidence_eval_system_top3_support_dense_current_rerank_dev.json"),
        "confidence_tuning_build": str(root / "aiops-docs/experiment/results/confidence_tuning_system_top3_support_build.json"),
        "confidence_tuning_dev": str(root / "aiops-docs/experiment/results/confidence_tuning_system_top3_support_dev.json"),
        "confidence_score_threshold_build": str(root / "aiops-docs/experiment/results/confidence_eval_score_threshold_dense_current_rerank_build.json"),
        "confidence_score_threshold_dev": str(root / "aiops-docs/experiment/results/confidence_eval_score_threshold_dense_current_rerank_dev.json"),
        "confidence_score_margin_build": str(root / "aiops-docs/experiment/results/confidence_eval_score_margin_dense_current_rerank_build.json"),
        "confidence_score_margin_dev": str(root / "aiops-docs/experiment/results/confidence_eval_score_margin_dense_current_rerank_dev.json"),
    }


def make_retrieval_payload(
    experiment_name: str,
    rerank: str,
    hit_at_10: float,
    mrr: float,
    gold_in_candidate_not_final_count: int,
    gold_promoted_by_rerank_count: int = 0,
    gold_demoted_by_rerank_count: int = 0,
) -> dict[str, object]:
    split = "build" if experiment_name.endswith("_build") else "dev"
    return {
        "experiment_name": experiment_name,
        "dataset": f"aiops-docs/experiment/rag/splits/rag_{split}.jsonl",
        "retrieval_strategy": "dense",
        "rerank": rerank,
        "evaluated_samples": 30 if split == "build" else 20,
        "candidate_top_k": 50,
        "final_top_k": 10,
        "candidate_metrics": {
            "candidate_hit_at_10": hit_at_10,
            "candidate_hit_at_20": hit_at_10 + 0.2,
            "candidate_hit_at_50": hit_at_10 + 0.3,
        },
        "final_metrics": {
            "hit_at_1": max(hit_at_10 - 0.3, 0.0),
            "hit_at_3": max(hit_at_10 - 0.15, 0.0),
            "hit_at_5": max(hit_at_10 - 0.05, 0.0),
            "hit_at_10": hit_at_10,
            "recall_at_1": max(hit_at_10 - 0.3, 0.0),
            "recall_at_3": max(hit_at_10 - 0.15, 0.0),
            "recall_at_5": max(hit_at_10 - 0.05, 0.0),
            "recall_at_10": hit_at_10,
            "mrr": mrr,
        },
        "gold_in_candidate_not_final_count": gold_in_candidate_not_final_count,
        "gold_promoted_by_rerank_count": gold_promoted_by_rerank_count,
        "gold_demoted_by_rerank_count": gold_demoted_by_rerank_count,
        "per_sample": [],
    }


def make_confidence_payload(
    experiment_name: str,
    strategy: str,
    split: str,
    high_precision: float,
    low_capture: float,
    counts: dict[str, int],
    *,
    score_direction: str | None = None,
    strong_threshold: float | None = None,
    support_threshold: float | None = None,
    high_avg_threshold: float | None = None,
) -> dict[str, object]:
    payload = {
        "experiment_name": experiment_name,
        "split": split,
        "strategy": strategy,
        "evaluated_samples": sum(counts.values()),
        "high_confidence_precision": high_precision,
        "low_confidence_error_capture": low_capture,
        "confidence_accuracy": 0.6,
        "count_by_confidence": counts,
    }
    if score_direction is not None:
        payload["score_direction"] = score_direction
    if strong_threshold is not None:
        payload["strong_threshold"] = strong_threshold
    if support_threshold is not None:
        payload["support_threshold"] = support_threshold
    if high_avg_threshold is not None:
        payload["high_avg_threshold"] = high_avg_threshold
    return payload


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
