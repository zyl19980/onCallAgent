import csv
import json
from pathlib import Path

from scripts.experiment.evaluate_retrieval_confidence import (
    diagnose_score_direction,
    evaluate_retrieval_confidence,
    predict_confidence,
    tune_build_dev_confidence,
    tune_confidence_thresholds,
)


def test_predict_confidence_baselines_respect_score_direction():
    confidence, _ = predict_confidence(
        strategy="score_threshold",
        top1_score=0.2,
        top2_score=0.3,
        top3_score=0.0,
        score_margin=0.1,
        score_direction="lower_is_better",
        high_threshold=0.25,
        low_threshold=0.4,
        margin_threshold=0.05,
    )
    assert confidence == "high"

    confidence, _ = predict_confidence(
        strategy="score_margin",
        top1_score=0.59,
        top2_score=0.585,
        top3_score=0.0,
        score_margin=0.005,
        score_direction="higher_is_better",
        high_threshold=0.62,
        low_threshold=0.58,
        margin_threshold=0.015,
    )
    assert confidence == "low"

    confidence, _ = predict_confidence(
        strategy="rank_and_margin",
        top1_score=0.6,
        top2_score=0.59,
        top3_score=0.0,
        score_margin=0.01,
        score_direction="higher_is_better",
        high_threshold=0.62,
        low_threshold=0.58,
        margin_threshold=0.015,
    )
    assert confidence == "medium"


def test_system_top3_support_higher_is_better_high_medium_low():
    high_confidence, high_debug = predict_confidence(
        strategy="system_top3_support",
        final_results=[
            make_result("a", 0.9, "source_a"),
            make_result("b", 0.7, "source_a"),
            make_result("c", 0.65, "source_a"),
        ],
        top1_score=0.9,
        top2_score=0.7,
        top3_score=0.65,
        score_margin=0.2,
        score_direction="higher_is_better",
        high_threshold=0.62,
        low_threshold=0.58,
        margin_threshold=0.015,
        strong_threshold=0.78,
        support_threshold=0.45,
        high_avg_threshold=0.55,
    )
    assert high_confidence == "high"
    assert high_debug["support_count"] == 2
    assert high_debug["top2_is_support"] is True
    assert high_debug["top3_is_support"] is True

    medium_confidence, medium_debug = predict_confidence(
        strategy="system_top3_support",
        final_results=[
            make_result("a", 0.9, "source_a"),
            make_result("b", 0.7, "source_a"),
        ],
        top1_score=0.9,
        top2_score=0.7,
        top3_score=0.0,
        score_margin=0.2,
        score_direction="higher_is_better",
        high_threshold=0.62,
        low_threshold=0.58,
        margin_threshold=0.015,
        strong_threshold=0.78,
        support_threshold=0.45,
        high_avg_threshold=0.55,
    )
    assert medium_confidence == "medium"
    assert medium_debug["top3_is_support"] is False

    low_confidence, low_debug = predict_confidence(
        strategy="system_top3_support",
        final_results=[
            make_result("a", 0.7, "source_a"),
            make_result("b", 0.69, "source_a"),
            make_result("c", 0.68, "source_a"),
        ],
        top1_score=0.7,
        top2_score=0.69,
        top3_score=0.68,
        score_margin=0.01,
        score_direction="higher_is_better",
        high_threshold=0.62,
        low_threshold=0.58,
        margin_threshold=0.015,
        strong_threshold=0.78,
        support_threshold=0.45,
        high_avg_threshold=0.55,
    )
    assert low_confidence == "low"
    assert low_debug["top1_is_strong"] is False


def test_system_top3_support_lower_is_better_and_same_source_required():
    confidence, debug = predict_confidence(
        strategy="system_top3_support",
        final_results=[
            make_result("a", 0.1, "source_a"),
            make_result("b", 0.2, "source_a"),
            make_result("c", 0.21, "source_b"),
        ],
        top1_score=0.1,
        top2_score=0.2,
        top3_score=0.21,
        score_margin=0.1,
        score_direction="lower_is_better",
        high_threshold=0.3,
        low_threshold=0.5,
        margin_threshold=0.05,
        strong_threshold=0.15,
        support_threshold=0.25,
        high_avg_threshold=0.2,
    )
    assert confidence == "medium"
    assert debug["top2_is_support"] is True
    assert debug["top3_is_support"] is False
    assert debug["support_count"] == 1


def test_system_top3_support_missing_top3_degrades_to_medium():
    confidence, debug = predict_confidence(
        strategy="system_top3_support",
        final_results=[
            make_result("a", 0.9, "source_a"),
            make_result("b", 0.7, "source_a"),
        ],
        top1_score=0.9,
        top2_score=0.7,
        top3_score=0.0,
        score_margin=0.2,
        score_direction="higher_is_better",
        high_threshold=0.62,
        low_threshold=0.58,
        margin_threshold=0.015,
        strong_threshold=0.78,
        support_threshold=0.45,
        high_avg_threshold=0.55,
    )
    assert confidence == "medium"
    assert debug["support_count"] == 1


def test_evaluate_retrieval_confidence_writes_outputs(tmp_path: Path):
    results_path = tmp_path / "dense_current_rerank_dev.json"
    output_path = tmp_path / "confidence_eval.json"
    summary_path = tmp_path / "confidence_eval.csv"

    payload = {
        "experiment_name": "dense_current_rerank_dev",
        "dataset": "aiops-docs/experiment/rag/splits/rag_dev.jsonl",
        "per_sample": [
            make_sample(
                "s1",
                [0.9, 0.8, 0.75],
                {"1": 1.0, "3": 1.0, "5": 1.0, "10": 1.0},
                source_ids=["source_a", "source_a", "source_a"],
            ),
                make_sample(
                    "s2",
                    [0.79, 0.5, 0.49],
                    {"1": 0.0, "3": 0.0, "5": 0.0, "10": 0.0},
                    source_ids=["source_a", "source_b", "source_b"],
                ),
            make_sample(
                "s3",
                [0.81, 0.5],
                {"1": 0.0, "3": 1.0, "5": 1.0, "10": 1.0},
                source_ids=["source_c", "source_c"],
            ),
        ],
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = evaluate_retrieval_confidence(
        results_path=results_path,
        output_path=output_path,
        summary_path=summary_path,
        strategy="system_top3_support",
        score_direction="higher_is_better",
        high_threshold=0.62,
        low_threshold=0.58,
        margin_threshold=0.015,
        strong_threshold=0.78,
        support_threshold=0.45,
        high_avg_threshold=0.55,
        success_k=10,
    )

    assert output_path.exists()
    assert summary_path.exists()
    assert report["evaluated_samples"] == 3
    assert report["count_by_confidence"] == {"high": 1, "medium": 1, "low": 1}
    assert report["high_confidence_precision"] == 1.0
    assert report["low_confidence_error_capture"] == 1.0
    assert report["medium_confidence_success_rate"] == 1.0
    assert report["confidence_accuracy"] == 1.0
    assert report["abstention_precision"] == 0.0
    assert report["abstention_recall"] == 0.0
    assert report["per_sample"][0]["confidence_debug"]["support_count"] == 2
    assert report["per_sample"][1]["predicted_confidence"] == "low"
    assert report["per_sample"][2]["predicted_confidence"] == "medium"
    assert report["per_sample"][2]["confidence_debug"]["top3_is_support"] is False
    assert report["per_sample"][0]["top1_top2_margin"] == 0.1
    assert report["per_sample"][0]["top3_support_features"]["support_count"] == 2

    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["strategy"] == "system_top3_support"
    assert rows[0]["high_confidence_precision"] == "1.0"
    assert rows[0]["abstention_precision"] == "0.0"
    assert rows[0]["count_high"] == "1"
    assert rows[0]["count_medium"] == "1"
    assert rows[0]["count_low"] == "1"


def test_diagnose_score_direction_outputs_both_directions(tmp_path: Path):
    results_path = tmp_path / "results.json"
    output_path = tmp_path / "diagnose.json"
    summary_path = tmp_path / "diagnose.csv"
    payload = {
        "experiment_name": "dense_current_rerank_build",
        "dataset": "aiops-docs/experiment/rag/splits/rag_build.jsonl",
        "per_sample": [
            make_sample("s1", [0.9, 0.8, 0.75], {"10": 1.0}),
            make_sample("s2", [0.6, 0.59, 0.58], {"10": 0.0}),
        ],
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = diagnose_score_direction(
        results_path=results_path,
        output_path=output_path,
        summary_path=summary_path,
        strategy="system_top3_support",
        high_threshold=0.62,
        low_threshold=0.58,
        margin_threshold=0.015,
        strong_threshold=0.78,
        support_threshold=0.45,
        high_avg_threshold=0.55,
        success_k=10,
    )

    assert report["diagnose_score_direction"] is True
    assert len(report["diagnoses"]) == 2
    assert {row["score_direction"] for row in report["diagnoses"]} == {
        "higher_is_better",
        "lower_is_better",
    }

    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2


def test_tune_thresholds_selects_best_eligible_config(tmp_path: Path):
    results_path = tmp_path / "results.json"
    output_path = tmp_path / "tune.json"
    summary_path = tmp_path / "tune.csv"
    payload = {
        "experiment_name": "dense_current_rerank_dev",
        "dataset": "aiops-docs/experiment/rag/splits/rag_dev.jsonl",
        "per_sample": [
            make_sample("s1", [0.92, 0.84, 0.8], {"10": 1.0}, source_ids=["source_a"] * 3),
            make_sample("s2", [0.88, 0.7, 0.68], {"10": 1.0}, source_ids=["source_b", "source_b", "source_z"]),
            make_sample("s3", [0.74, 0.7, 0.68], {"10": 0.0}, source_ids=["source_c"] * 3),
            make_sample("s4", [0.73, 0.51, 0.49], {"10": 0.0}, source_ids=["source_d", "source_x", "source_y"]),
        ],
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = tune_confidence_thresholds(
        results_path=results_path,
        output_path=output_path,
        summary_path=summary_path,
        strategy="system_top3_support",
        score_direction="higher_is_better",
        diagnose_score_direction=False,
        strong_threshold=0.78,
        support_threshold=0.45,
        high_avg_threshold=0.55,
        strong_threshold_grid=[0.78, 0.9],
        support_threshold_grid=[0.45],
        high_avg_threshold_grid=[0.55],
        success_k=10,
    )

    assert report["total_configs"] == 2
    assert report["eligible_configs"] >= 1
    assert report["best_config"]["passes_constraints"] is True
    assert "low_ratio" in report["best_config"]

    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert "passes_constraints" in rows[0]


def test_evaluate_retrieval_confidence_merges_should_abstain_labels(tmp_path: Path):
    results_path = tmp_path / "expanded_build.json"
    labels_path = tmp_path / "rag_build.jsonl"
    output_path = tmp_path / "confidence_eval.json"
    summary_path = tmp_path / "confidence_eval.csv"

    payload = {
        "experiment_name": "dense_current_rerank_expanded_build",
        "dataset": "aiops-docs/experiment/rag/splits/expanded/rag_build.jsonl",
        "per_sample": [
            make_sample(
                "s1",
                [0.92, 0.7, 0.69],
                {"1": 1.0, "3": 1.0, "5": 1.0, "10": 1.0},
                source_ids=["source_a", "source_a", "source_b"],
            ),
        ],
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    labels_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "s1", "split": "build", "should_abstain": False, "source_ids": ["source_a"]}, ensure_ascii=False),
                json.dumps({"id": "s2", "split": "build", "should_abstain": True, "source_ids": []}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate_retrieval_confidence(
        results_path=results_path,
        labels_path=labels_path,
        output_path=output_path,
        summary_path=summary_path,
        strategy="rank_and_margin",
        score_direction="higher_is_better",
        high_threshold=0.9,
        low_threshold=0.4,
        margin_threshold=0.1,
        success_k=10,
    )

    assert report["evaluated_samples"] == 2
    assert report["answerable_samples"] == 1
    assert report["should_abstain_samples"] == 1
    assert report["abstention_precision"] == 1.0
    assert report["abstention_recall"] == 1.0
    assert report["false_confident_count"] == 0
    assert report["coverage"] == 0.5

    abstain_row = next(row for row in report["per_sample"] if row["sample_id"] == "s2")
    assert abstain_row["should_abstain"] is True
    assert abstain_row["top1_score"] == 0.0
    assert abstain_row["candidate_hit_at_50"] == 0.0
    assert abstain_row["predicted_confidence"] == "low"
    assert abstain_row["rerank_provider_counts"] == {}


def test_tune_build_dev_confidence_recommends_global_config(tmp_path: Path):
    build_results_path = tmp_path / "build.json"
    dev_results_path = tmp_path / "dev.json"
    build_labels_path = tmp_path / "rag_build.jsonl"
    dev_labels_path = tmp_path / "rag_dev.jsonl"
    output_path = tmp_path / "tuning.json"
    summary_path = tmp_path / "tuning.csv"

    build_payload = {
        "experiment_name": "dense_current_rerank_expanded_build",
        "dataset": "aiops-docs/experiment/rag/splits/expanded/rag_build.jsonl",
        "per_sample": [
            make_sample("b1", [0.95, 0.7, 0.69], {"10": 1.0}, source_ids=["source_a", "source_a", "source_b"]),
            make_sample("b2", [0.58, 0.575, 0.57], {"10": 0.0}, source_ids=["source_b", "source_b", "source_b"]),
        ],
    }
    dev_payload = {
        "experiment_name": "dense_current_rerank_expanded_dev",
        "dataset": "aiops-docs/experiment/rag/splits/expanded/rag_dev.jsonl",
        "per_sample": [
            make_sample("d1", [0.93, 0.71, 0.7], {"10": 1.0}, source_ids=["source_a", "source_a", "source_c"]),
            make_sample("d2", [0.57, 0.568, 0.567], {"10": 0.0}, source_ids=["source_d", "source_d", "source_d"]),
        ],
    }
    build_results_path.write_text(json.dumps(build_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dev_results_path.write_text(json.dumps(dev_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_labels_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "b1", "split": "build", "should_abstain": False, "source_ids": ["source_a"]}, ensure_ascii=False),
                json.dumps({"id": "b2", "split": "build", "should_abstain": False, "source_ids": ["source_b"]}, ensure_ascii=False),
                json.dumps({"id": "b3", "split": "build", "should_abstain": True, "source_ids": []}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dev_labels_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "d1", "split": "dev", "should_abstain": False, "source_ids": ["source_a"]}, ensure_ascii=False),
                json.dumps({"id": "d2", "split": "dev", "should_abstain": False, "source_ids": ["source_d"]}, ensure_ascii=False),
                json.dumps({"id": "d3", "split": "dev", "should_abstain": True, "source_ids": []}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = tune_build_dev_confidence(
        build_results_path=build_results_path,
        dev_results_path=dev_results_path,
        build_labels_path=build_labels_path,
        dev_labels_path=dev_labels_path,
        output_path=output_path,
        summary_path=summary_path,
        strategies=["rank_and_margin", "system_top3_support"],
        score_directions=["higher_is_better"],
        high_threshold_grid=[0.8, 0.9],
        low_threshold_grid=[0.4, 0.55],
        margin_threshold_grid=[0.02],
        strong_threshold_grid=[0.8, 0.9],
        support_threshold_grid=[0.6],
        high_avg_threshold_grid=[0.7],
        success_k=10,
    )

    assert output_path.exists()
    assert summary_path.exists()
    assert report["recommended_strategy"] in {"rank_and_margin", "system_top3_support"}
    assert report["selected_build_report"]["should_abstain_samples"] == 1
    assert report["selected_dev_report"]["should_abstain_samples"] == 1
    assert len(report["rows"]) == 6
    assert "recommended_config" in report

    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    assert "avg_abstention_f1" in rows[0]


def make_sample(
    sample_id: str,
    scores: list[float],
    hit_at_k: dict[str, float],
    source_ids: list[str] | None = None,
) -> dict[str, object]:
    final_results = []
    source_ids = source_ids or ["source_a"] * len(scores)
    for index, score in enumerate(scores, start=1):
        final_results.append(make_result(f"chunk-{sample_id}-{index}", score, source_ids[index - 1], index))
    return {
        "id": sample_id,
        "question_type": "troubleshooting_step",
        "source_ids": ["source_a"],
        "final_results": final_results,
        "hit_at_k": hit_at_k,
    }


def make_result(chunk_id: str, score: float, source_id: str, rank: int = 1) -> dict[str, object]:
    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "source_id": source_id,
        "score": score,
        "rerank_score": score,
    }
