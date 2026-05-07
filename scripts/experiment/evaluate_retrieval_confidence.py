"""基于已有 retrieval 结果评估 retrieval-level confidence / abstention 策略。"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path


DEFAULT_BUILD_INPUT = "aiops-docs/experiment/results/dense_current_rerank_build.json"
DEFAULT_DEV_INPUT = "aiops-docs/experiment/results/dense_current_rerank_dev.json"
DEFAULT_SCORE_DIRECTION = "higher_is_better"
DEFAULT_STRONG_THRESHOLD = 0.78
DEFAULT_SUPPORT_THRESHOLD = 0.45
DEFAULT_HIGH_AVG_THRESHOLD = 0.55


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 retrieval-level confidence strategy")
    parser.add_argument("--results", help="已有 retrieval results JSON")
    parser.add_argument("--output", required=True, help="confidence eval JSON 输出路径")
    parser.add_argument("--summary", required=True, help="confidence eval CSV 输出路径")
    parser.add_argument("--labels", help="可选 split JSONL，用于并入 should_abstain 标签")
    parser.add_argument("--build-results", help="build retrieval results JSON")
    parser.add_argument("--dev-results", help="dev retrieval results JSON")
    parser.add_argument("--build-labels", help="build split JSONL")
    parser.add_argument("--dev-labels", help="dev split JSONL")
    parser.add_argument(
        "--strategy",
        default="rank_and_margin",
        choices=[
            "score_threshold",
            "score_margin",
            "rank_and_margin",
            "system_top3_support",
        ],
    )
    parser.add_argument(
        "--score-direction",
        default=DEFAULT_SCORE_DIRECTION,
        choices=["lower_is_better", "higher_is_better"],
    )
    parser.add_argument("--high-threshold", type=float, default=0.62)
    parser.add_argument("--low-threshold", type=float, default=0.58)
    parser.add_argument("--margin-threshold", type=float, default=0.015)
    parser.add_argument("--strong-threshold", type=float, default=DEFAULT_STRONG_THRESHOLD)
    parser.add_argument("--support-threshold", type=float, default=DEFAULT_SUPPORT_THRESHOLD)
    parser.add_argument("--high-avg-threshold", type=float, default=DEFAULT_HIGH_AVG_THRESHOLD)
    parser.add_argument("--strong-threshold-grid", default="", help="逗号分隔的 strong threshold 候选")
    parser.add_argument("--support-threshold-grid", default="", help="逗号分隔的 support threshold 候选")
    parser.add_argument("--high-avg-threshold-grid", default="", help="逗号分隔的 high avg threshold 候选")
    parser.add_argument("--high-threshold-grid", default="", help="逗号分隔的 high threshold 候选")
    parser.add_argument("--low-threshold-grid", default="", help="逗号分隔的 low threshold 候选")
    parser.add_argument("--margin-threshold-grid", default="", help="逗号分隔的 margin threshold 候选")
    parser.add_argument("--diagnose-score-direction", action="store_true")
    parser.add_argument("--tune-thresholds", action="store_true")
    parser.add_argument("--tune-build-dev", action="store_true")
    parser.add_argument(
        "--strategies",
        default="rank_and_margin,system_top3_support,score_threshold,score_margin",
        help="build/dev tuning 时要评估的策略列表",
    )
    parser.add_argument("--success-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tune_build_dev:
        report = tune_build_dev_confidence(
            build_results_path=Path(args.build_results or ""),
            dev_results_path=Path(args.dev_results or ""),
            build_labels_path=Path(args.build_labels or ""),
            dev_labels_path=Path(args.dev_labels or ""),
            output_path=Path(args.output),
            summary_path=Path(args.summary),
            strategies=parse_strategies(args.strategies),
            score_directions=["higher_is_better", "lower_is_better"],
            high_threshold_grid=parse_grid_with_default(
                args.high_threshold_grid,
                [0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.7],
            ),
            low_threshold_grid=parse_grid_with_default(
                args.low_threshold_grid,
                [0.52, 0.54, 0.56, 0.58, 0.6],
            ),
            margin_threshold_grid=parse_grid_with_default(
                args.margin_threshold_grid,
                [0.003, 0.005, 0.01, 0.015, 0.02, 0.03],
            ),
            strong_threshold_grid=parse_grid_with_default(
                args.strong_threshold_grid,
                [0.58, 0.62, 0.66, 0.7, 0.74, 0.78],
            ),
            support_threshold_grid=parse_grid_with_default(
                args.support_threshold_grid,
                [0.45, 0.5, 0.55, 0.6, 0.65],
            ),
            high_avg_threshold_grid=parse_grid_with_default(
                args.high_avg_threshold_grid,
                [0.55, 0.58, 0.61, 0.64, 0.67],
            ),
            success_k=args.success_k,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if not args.results:
        raise ValueError("--results 或 --tune-build-dev 所需的 build/dev inputs 至少应提供一种")

    if args.tune_thresholds:
        report = tune_confidence_thresholds(
            results_path=Path(args.results),
            output_path=Path(args.output),
            summary_path=Path(args.summary),
            strategy=args.strategy,
            score_direction=args.score_direction,
            diagnose_score_direction=args.diagnose_score_direction,
            strong_threshold=args.strong_threshold,
            support_threshold=args.support_threshold,
            high_avg_threshold=args.high_avg_threshold,
            strong_threshold_grid=parse_threshold_grid(args.strong_threshold_grid, args.strong_threshold),
            support_threshold_grid=parse_threshold_grid(args.support_threshold_grid, args.support_threshold),
            high_avg_threshold_grid=parse_threshold_grid(args.high_avg_threshold_grid, args.high_avg_threshold),
            success_k=args.success_k,
            labels_path=Path(args.labels) if args.labels else None,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.diagnose_score_direction:
        report = diagnose_score_direction(
            results_path=Path(args.results),
            output_path=Path(args.output),
            summary_path=Path(args.summary),
            strategy=args.strategy,
            high_threshold=args.high_threshold,
            low_threshold=args.low_threshold,
            margin_threshold=args.margin_threshold,
            strong_threshold=args.strong_threshold,
            support_threshold=args.support_threshold,
            high_avg_threshold=args.high_avg_threshold,
            success_k=args.success_k,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    report = evaluate_retrieval_confidence(
        results_path=Path(args.results),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        strategy=args.strategy,
        score_direction=args.score_direction,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        margin_threshold=args.margin_threshold,
        strong_threshold=args.strong_threshold,
        support_threshold=args.support_threshold,
        high_avg_threshold=args.high_avg_threshold,
        success_k=args.success_k,
        labels_path=Path(args.labels) if args.labels else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def evaluate_retrieval_confidence(
    *,
    results_path: Path,
    output_path: Path,
    summary_path: Path,
    strategy: str = "rank_and_margin",
    score_direction: str = DEFAULT_SCORE_DIRECTION,
    high_threshold: float = 0.62,
    low_threshold: float = 0.58,
    margin_threshold: float = 0.015,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    high_avg_threshold: float = DEFAULT_HIGH_AVG_THRESHOLD,
    success_k: int = 10,
    labels_path: Path | None = None,
) -> dict[str, object]:
    payload = load_results_payload(results_path)
    label_rows = load_label_rows(labels_path) if labels_path else None
    report = evaluate_payload(
        payload=payload,
        results_path=results_path,
        strategy=strategy,
        score_direction=score_direction,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        margin_threshold=margin_threshold,
        strong_threshold=strong_threshold,
        support_threshold=support_threshold,
        high_avg_threshold=high_avg_threshold,
        success_k=success_k,
        label_rows=label_rows,
        labels_path=labels_path,
    )
    write_json(output_path.resolve(), report)
    write_summary_csv(summary_path.resolve(), report)
    return report


def evaluate_payload(
    *,
    payload: dict[str, object],
    results_path: Path,
    strategy: str,
    score_direction: str,
    high_threshold: float,
    low_threshold: float,
    margin_threshold: float,
    strong_threshold: float,
    support_threshold: float,
    high_avg_threshold: float,
    success_k: int,
    label_rows: list[dict[str, object]] | None = None,
    labels_path: Path | None = None,
) -> dict[str, object]:
    samples = merge_samples_with_labels(payload=payload, label_rows=label_rows)
    sample_rows = []
    for entry in samples:
        sample = dict(entry.get("sample") or {})
        label = dict(entry.get("label") or {})
        final_results = list(sample.get("final_results") or sample.get("retrieved") or [])
        top1_score = extract_score(final_results, index=0)
        top2_score = extract_score(final_results, index=1)
        top3_score = extract_score(final_results, index=2)
        score_margin = round(compute_score_margin(top1_score, top2_score, score_direction), 6)
        predicted_confidence, confidence_debug = predict_confidence(
            strategy=strategy,
            final_results=final_results,
            top1_score=top1_score,
            top2_score=top2_score,
            top3_score=top3_score,
            score_margin=score_margin,
            score_direction=score_direction,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            margin_threshold=margin_threshold,
            strong_threshold=strong_threshold,
            support_threshold=support_threshold,
            high_avg_threshold=high_avg_threshold,
        )
        hit_at_k = dict(sample.get("hit_at_k") or {})
        candidate_hit_at_k = dict(sample.get("candidate_hit_at_k") or {})
        should_abstain = bool(label.get("should_abstain", False))
        retrieval_success = (not should_abstain) and bool(float(hit_at_k.get(str(success_k), 0.0)) >= 1.0)
        top1_rerank_score = extract_rerank_score(final_results, index=0)
        provider_counts = count_rerank_providers(final_results)
        sample_rows.append(
            {
                "sample_id": str(sample.get("id") or ""),
                "split": str(label.get("split") or infer_split(payload)),
                "question_type": str(sample.get("question_type") or ""),
                "source_ids": list(sample.get("source_ids") or []),
                "should_abstain": should_abstain,
                "top1_score": round(top1_score, 6),
                "top1_rerank_score": round(top1_rerank_score, 6),
                "top2_score": round(top2_score, 6),
                "top1_top2_margin": score_margin,
                "predicted_confidence": predicted_confidence,
                "retrieval_success": retrieval_success,
                "first_relevant_rank": int(sample.get("first_relevant_rank") or 0),
                "hit_at_1": to_float(hit_at_k.get("1")),
                "hit_at_3": to_float(hit_at_k.get("3")),
                "hit_at_5": to_float(hit_at_k.get("5")),
                "hit_at_10": to_float(hit_at_k.get("10")),
                "candidate_hit_at_50": to_float(candidate_hit_at_k.get("50")),
                "gold_in_candidate_not_final": bool(sample.get("gold_in_candidate_not_final", False)),
                "rerank_provider_top1": str((final_results[0].get("rerank_provider") if final_results else "") or ""),
                "rerank_provider_counts": provider_counts,
                "rerank_used_local_fallback": bool(provider_counts.get("local", 0) > 0),
                "confidence_debug": confidence_debug,
                "top3_support_features": {
                    "avg_top3_score": confidence_debug.get("avg_top3_score", 0.0),
                    "strong_count": int(confidence_debug.get("strong_count", 0) or 0),
                    "support_count": int(confidence_debug.get("support_count", 0) or 0),
                    "top1_is_strong": bool(confidence_debug.get("top1_is_strong", False)),
                    "top2_is_support": bool(confidence_debug.get("top2_is_support", False)),
                    "top3_is_support": bool(confidence_debug.get("top3_is_support", False)),
                },
            }
        )

    metrics = summarize_confidence(
        sample_rows,
        strategy=strategy,
        score_direction=score_direction,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        margin_threshold=margin_threshold,
        strong_threshold=strong_threshold,
        support_threshold=support_threshold,
        high_avg_threshold=high_avg_threshold,
        success_k=success_k,
    )
    return {
        "results": to_repo_relative_path(results_path.resolve()),
        "experiment_name": str(payload.get("experiment_name") or ""),
        "split": infer_split(payload),
        "strategy": strategy,
        "score_direction": score_direction,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "margin_threshold": margin_threshold,
        "strong_threshold": strong_threshold,
        "support_threshold": support_threshold,
        "high_avg_threshold": high_avg_threshold,
        "success_k": success_k,
        "labels": to_repo_relative_path(labels_path.resolve()) if labels_path else "",
        "evaluated_samples": len(sample_rows),
        "per_sample": sample_rows,
        **metrics,
    }


def diagnose_score_direction(
    *,
    results_path: Path,
    output_path: Path,
    summary_path: Path,
    strategy: str,
    high_threshold: float,
    low_threshold: float,
    margin_threshold: float,
    strong_threshold: float,
    support_threshold: float,
    high_avg_threshold: float,
    success_k: int,
    labels_path: Path | None = None,
) -> dict[str, object]:
    payload = load_results_payload(results_path)
    label_rows = load_label_rows(labels_path) if labels_path else None
    diagnoses = []
    for score_direction in ("higher_is_better", "lower_is_better"):
        report = evaluate_payload(
            payload=payload,
            results_path=results_path,
            strategy=strategy,
            score_direction=score_direction,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            margin_threshold=margin_threshold,
            strong_threshold=strong_threshold,
            support_threshold=support_threshold,
            high_avg_threshold=high_avg_threshold,
            success_k=success_k,
            label_rows=label_rows,
            labels_path=labels_path,
        )
        diagnoses.append(compact_confidence_row(report))

    best_direction = max(
        diagnoses,
        key=lambda item: (
            float(item["high_confidence_precision"]),
            float(item["low_confidence_error_capture"]),
            float(item["confidence_accuracy"]),
        ),
    )["score_direction"]
    report = {
        "results": to_repo_relative_path(results_path.resolve()),
        "experiment_name": str(payload.get("experiment_name") or ""),
        "split": infer_split(payload),
        "strategy": strategy,
        "diagnose_score_direction": True,
        "recommended_score_direction": best_direction,
        "diagnoses": diagnoses,
    }
    write_json(output_path.resolve(), report)
    write_rows_csv(summary_path.resolve(), diagnoses)
    return report


def tune_confidence_thresholds(
    *,
    results_path: Path,
    output_path: Path,
    summary_path: Path,
    strategy: str,
    score_direction: str,
    diagnose_score_direction: bool,
    strong_threshold: float,
    support_threshold: float,
    high_avg_threshold: float,
    strong_threshold_grid: list[float],
    support_threshold_grid: list[float],
    high_avg_threshold_grid: list[float],
    success_k: int,
    labels_path: Path | None = None,
) -> dict[str, object]:
    if strategy != "system_top3_support":
        raise ValueError("--tune-thresholds 目前仅支持 system_top3_support")

    payload = load_results_payload(results_path)
    label_rows = load_label_rows(labels_path) if labels_path else None
    directions = ["higher_is_better", "lower_is_better"] if diagnose_score_direction else [score_direction]
    rows = []
    for direction in directions:
        for current_strong, current_support, current_avg in itertools.product(
            strong_threshold_grid,
            support_threshold_grid,
            high_avg_threshold_grid,
        ):
            report = evaluate_payload(
                payload=payload,
                results_path=results_path,
                strategy=strategy,
                score_direction=direction,
                high_threshold=0.0,
                low_threshold=0.0,
                margin_threshold=0.0,
                strong_threshold=current_strong,
                support_threshold=current_support,
                high_avg_threshold=current_avg,
                success_k=success_k,
                label_rows=label_rows,
                labels_path=labels_path,
            )
            row = compact_confidence_row(report)
            row["strong_threshold"] = current_strong
            row["support_threshold"] = current_support
            row["high_avg_threshold"] = current_avg
            row["low_ratio"] = safe_rate(int(row["count_low"]), int(row["evaluated_samples"]))
            row["passes_constraints"] = bool(
                float(row["high_confidence_precision"]) >= 0.9
                and int(row["count_medium"]) >= 1
                and float(row["low_ratio"]) <= 0.75
            )
            rows.append(row)

    eligible = [row for row in rows if bool(row["passes_constraints"])]
    warnings = []
    if not eligible:
        warnings.append("no_config_satisfies_constraints")
        best_row = max(
            rows,
            key=lambda item: (
                int(int(item["count_medium"]) >= 1),
                float(item["high_confidence_precision"]),
                -max(0.0, float(item["low_ratio"]) - 0.75),
                float(item["low_confidence_error_capture"]),
                int(item["count_medium"]),
                float(item["confidence_accuracy"]),
            ),
        )
    else:
        best_row = max(
            eligible,
            key=lambda item: (
                float(item["low_confidence_error_capture"]),
                float(item["high_confidence_precision"]),
                int(item["count_medium"]),
                -float(item["low_ratio"]),
            ),
        )
    report = {
        "results": to_repo_relative_path(results_path.resolve()),
        "experiment_name": str(payload.get("experiment_name") or ""),
        "split": infer_split(payload),
        "strategy": strategy,
        "diagnose_score_direction": diagnose_score_direction,
        "searched_score_directions": directions,
        "grid": {
            "strong_threshold_grid": strong_threshold_grid,
            "support_threshold_grid": support_threshold_grid,
            "high_avg_threshold_grid": high_avg_threshold_grid,
        },
        "selection_constraints": {
            "high_confidence_precision_min": 0.9,
            "count_medium_min": 1,
            "low_ratio_max": 0.75,
        },
        "best_config": best_row,
        "total_configs": len(rows),
        "eligible_configs": len([row for row in rows if bool(row["passes_constraints"])]),
        "rows": rows,
        "warnings": warnings,
    }
    write_json(output_path.resolve(), report)
    write_rows_csv(summary_path.resolve(), rows)
    return report


def tune_build_dev_confidence(
    *,
    build_results_path: Path,
    dev_results_path: Path,
    build_labels_path: Path,
    dev_labels_path: Path,
    output_path: Path,
    summary_path: Path,
    strategies: list[str],
    score_directions: list[str],
    high_threshold_grid: list[float],
    low_threshold_grid: list[float],
    margin_threshold_grid: list[float],
    strong_threshold_grid: list[float],
    support_threshold_grid: list[float],
    high_avg_threshold_grid: list[float],
    success_k: int,
) -> dict[str, object]:
    build_payload = load_results_payload(build_results_path)
    dev_payload = load_results_payload(dev_results_path)
    build_labels = load_label_rows(build_labels_path)
    dev_labels = load_label_rows(dev_labels_path)

    rows = []
    strategy_reports: dict[str, dict[str, object]] = {}

    for strategy in strategies:
        strategy_rows = []
        for config in iter_strategy_configs(
            strategy=strategy,
            score_directions=score_directions,
            high_threshold_grid=high_threshold_grid,
            low_threshold_grid=low_threshold_grid,
            margin_threshold_grid=margin_threshold_grid,
            strong_threshold_grid=strong_threshold_grid,
            support_threshold_grid=support_threshold_grid,
            high_avg_threshold_grid=high_avg_threshold_grid,
        ):
            build_report = evaluate_payload(
                payload=build_payload,
                results_path=build_results_path,
                strategy=strategy,
                score_direction=str(config["score_direction"]),
                high_threshold=float(config.get("high_threshold", 0.0) or 0.0),
                low_threshold=float(config.get("low_threshold", 0.0) or 0.0),
                margin_threshold=float(config.get("margin_threshold", 0.0) or 0.0),
                strong_threshold=float(config.get("strong_threshold", DEFAULT_STRONG_THRESHOLD) or DEFAULT_STRONG_THRESHOLD),
                support_threshold=float(config.get("support_threshold", DEFAULT_SUPPORT_THRESHOLD) or DEFAULT_SUPPORT_THRESHOLD),
                high_avg_threshold=float(config.get("high_avg_threshold", DEFAULT_HIGH_AVG_THRESHOLD) or DEFAULT_HIGH_AVG_THRESHOLD),
                success_k=success_k,
                label_rows=build_labels,
                labels_path=build_labels_path,
            )
            dev_report = evaluate_payload(
                payload=dev_payload,
                results_path=dev_results_path,
                strategy=strategy,
                score_direction=str(config["score_direction"]),
                high_threshold=float(config.get("high_threshold", 0.0) or 0.0),
                low_threshold=float(config.get("low_threshold", 0.0) or 0.0),
                margin_threshold=float(config.get("margin_threshold", 0.0) or 0.0),
                strong_threshold=float(config.get("strong_threshold", DEFAULT_STRONG_THRESHOLD) or DEFAULT_STRONG_THRESHOLD),
                support_threshold=float(config.get("support_threshold", DEFAULT_SUPPORT_THRESHOLD) or DEFAULT_SUPPORT_THRESHOLD),
                high_avg_threshold=float(config.get("high_avg_threshold", DEFAULT_HIGH_AVG_THRESHOLD) or DEFAULT_HIGH_AVG_THRESHOLD),
                success_k=success_k,
                label_rows=dev_labels,
                labels_path=dev_labels_path,
            )
            row = build_tuning_row(
                strategy=strategy,
                config=config,
                build_report=build_report,
                dev_report=dev_report,
            )
            rows.append(row)
            strategy_rows.append(row)

        best_build = select_best_split_row(strategy_rows, split_prefix="build")
        best_dev = select_best_split_row(strategy_rows, split_prefix="dev")
        best_global = select_best_global_row(strategy_rows)
        strategy_reports[strategy] = {
            "strategy": strategy,
            "total_configs": len(strategy_rows),
            "best_build": best_build,
            "best_dev": best_dev,
            "best_global": best_global,
        }

    recommended_row = select_best_global_row(rows)
    selected_build_report = evaluate_payload(
        payload=build_payload,
        results_path=build_results_path,
        strategy=str(recommended_row["strategy"]),
        score_direction=str(recommended_row["score_direction"]),
        high_threshold=float(recommended_row.get("high_threshold", 0.0) or 0.0),
        low_threshold=float(recommended_row.get("low_threshold", 0.0) or 0.0),
        margin_threshold=float(recommended_row.get("margin_threshold", 0.0) or 0.0),
        strong_threshold=float(recommended_row.get("strong_threshold", DEFAULT_STRONG_THRESHOLD) or DEFAULT_STRONG_THRESHOLD),
        support_threshold=float(recommended_row.get("support_threshold", DEFAULT_SUPPORT_THRESHOLD) or DEFAULT_SUPPORT_THRESHOLD),
        high_avg_threshold=float(recommended_row.get("high_avg_threshold", DEFAULT_HIGH_AVG_THRESHOLD) or DEFAULT_HIGH_AVG_THRESHOLD),
        success_k=success_k,
        label_rows=build_labels,
        labels_path=build_labels_path,
    )
    selected_dev_report = evaluate_payload(
        payload=dev_payload,
        results_path=dev_results_path,
        strategy=str(recommended_row["strategy"]),
        score_direction=str(recommended_row["score_direction"]),
        high_threshold=float(recommended_row.get("high_threshold", 0.0) or 0.0),
        low_threshold=float(recommended_row.get("low_threshold", 0.0) or 0.0),
        margin_threshold=float(recommended_row.get("margin_threshold", 0.0) or 0.0),
        strong_threshold=float(recommended_row.get("strong_threshold", DEFAULT_STRONG_THRESHOLD) or DEFAULT_STRONG_THRESHOLD),
        support_threshold=float(recommended_row.get("support_threshold", DEFAULT_SUPPORT_THRESHOLD) or DEFAULT_SUPPORT_THRESHOLD),
        high_avg_threshold=float(recommended_row.get("high_avg_threshold", DEFAULT_HIGH_AVG_THRESHOLD) or DEFAULT_HIGH_AVG_THRESHOLD),
        success_k=success_k,
        label_rows=dev_labels,
        labels_path=dev_labels_path,
    )

    report = {
        "inputs": {
            "build_results": to_repo_relative_path(build_results_path.resolve()),
            "dev_results": to_repo_relative_path(dev_results_path.resolve()),
            "build_labels": to_repo_relative_path(build_labels_path.resolve()),
            "dev_labels": to_repo_relative_path(dev_labels_path.resolve()),
        },
        "retrieval_configuration": {
            "retrieval_strategy": "dense",
            "rerank": "current",
            "query_mode": "original",
            "rerank_note": "current_rerank = online rerank + local fallback",
            "success_k": success_k,
        },
        "feature_fields": [
            "sample_id",
            "split",
            "should_abstain",
            "top1_score",
            "top1_rerank_score",
            "top2_score",
            "top1_top2_margin",
            "first_relevant_rank",
            "hit_at_10",
            "candidate_hit_at_50",
            "gold_in_candidate_not_final",
            "top3_support_features",
            "rerank_provider_top1",
            "rerank_provider_counts",
            "rerank_used_local_fallback",
        ],
        "search_space": {
            "strategies": strategies,
            "score_directions": score_directions,
            "high_threshold_grid": high_threshold_grid,
            "low_threshold_grid": low_threshold_grid,
            "margin_threshold_grid": margin_threshold_grid,
            "strong_threshold_grid": strong_threshold_grid,
            "support_threshold_grid": support_threshold_grid,
            "high_avg_threshold_grid": high_avg_threshold_grid,
        },
        "strategies": strategy_reports,
        "recommended_strategy": str(recommended_row["strategy"]),
        "recommended_config": recommended_row,
        "selected_build_report": selected_build_report,
        "selected_dev_report": selected_dev_report,
        "rows": rows,
    }
    write_json(output_path.resolve(), report)
    write_tuning_rows_csv(summary_path.resolve(), rows)
    return report


def predict_confidence(
    *,
    strategy: str,
    final_results: list[dict[str, object]] | None = None,
    top1_score: float,
    top2_score: float,
    top3_score: float = 0.0,
    score_margin: float,
    score_direction: str,
    high_threshold: float,
    low_threshold: float,
    margin_threshold: float,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    high_avg_threshold: float = DEFAULT_HIGH_AVG_THRESHOLD,
) -> tuple[str, dict[str, object]]:
    if strategy == "score_threshold":
        if meets_threshold(top1_score, high_threshold, score_direction):
            confidence = "high"
        elif is_worse_than(top1_score, low_threshold, score_direction):
            confidence = "low"
        else:
            confidence = "medium"
        return confidence, build_confidence_debug(
            top1_score=top1_score,
            top2_score=top2_score,
            top3_score=top3_score,
            score_direction=score_direction,
            thresholds={
                "high_threshold": high_threshold,
                "low_threshold": low_threshold,
                "margin_threshold": margin_threshold,
            },
        )

    if strategy == "score_margin":
        if score_margin >= margin_threshold:
            confidence = "high"
        elif score_margin < margin_threshold / 2:
            confidence = "low"
        else:
            confidence = "medium"
        return confidence, build_confidence_debug(
            top1_score=top1_score,
            top2_score=top2_score,
            top3_score=top3_score,
            score_direction=score_direction,
            thresholds={
                "high_threshold": high_threshold,
                "low_threshold": low_threshold,
                "margin_threshold": margin_threshold,
            },
        )

    if strategy == "rank_and_margin":
        if meets_threshold(top1_score, high_threshold, score_direction) and score_margin >= margin_threshold:
            confidence = "high"
        elif is_worse_than(top1_score, low_threshold, score_direction) or score_margin < margin_threshold / 2:
            confidence = "low"
        else:
            confidence = "medium"
        return confidence, build_confidence_debug(
            top1_score=top1_score,
            top2_score=top2_score,
            top3_score=top3_score,
            score_direction=score_direction,
            thresholds={
                "high_threshold": high_threshold,
                "low_threshold": low_threshold,
                "margin_threshold": margin_threshold,
            },
        )

    if strategy == "system_top3_support":
        final_results = list(final_results or [])
        top3 = final_results[:3]
        top1 = top3[0] if len(top3) >= 1 else {}
        top2 = top3[1] if len(top3) >= 2 else {}
        top3_item = top3[2] if len(top3) >= 3 else {}
        top1_source = str(top1.get("source_id") or "")

        top1_is_strong = meets_threshold(top1_score, strong_threshold, score_direction)
        top2_is_support = is_support_candidate(
            item=top2,
            top1_source_id=top1_source,
            support_threshold=support_threshold,
            score_direction=score_direction,
        )
        top3_is_support = is_support_candidate(
            item=top3_item,
            top1_source_id=top1_source,
            support_threshold=support_threshold,
            score_direction=score_direction,
        )
        strong_count = sum(
            1
            for item in top3
            if meets_threshold(extract_item_score(item), strong_threshold, score_direction)
        )
        support_count = int(top2_is_support) + int(top3_is_support)
        avg_top3_score = avg(extract_item_score(item) for item in top3)
        avg_meets = meets_threshold(avg_top3_score, high_avg_threshold, score_direction)

        if top1_is_strong and top2_is_support and top3_is_support and avg_meets:
            confidence = "high"
        elif top1_is_strong and support_count >= 1:
            confidence = "medium"
        else:
            confidence = "low"

        return confidence, build_confidence_debug(
            top1_score=top1_score,
            top2_score=top2_score,
            top3_score=top3_score,
            avg_top3_score=avg_top3_score,
            strong_count=strong_count,
            support_count=support_count,
            top1_is_strong=top1_is_strong,
            top2_is_support=top2_is_support,
            top3_is_support=top3_is_support,
            score_direction=score_direction,
            thresholds={
                "strong_threshold": strong_threshold,
                "support_threshold": support_threshold,
                "high_avg_threshold": high_avg_threshold,
            },
        )

    raise ValueError(f"未知 strategy: {strategy}")


def summarize_confidence(
    sample_rows: list[dict[str, object]],
    *,
    strategy: str,
    score_direction: str,
    high_threshold: float,
    low_threshold: float,
    margin_threshold: float,
    strong_threshold: float,
    support_threshold: float,
    high_avg_threshold: float,
    success_k: int,
) -> dict[str, object]:
    count_by_confidence = {"high": 0, "medium": 0, "low": 0}
    for row in sample_rows:
        count_by_confidence[str(row["predicted_confidence"])] += 1

    answerable_rows = [row for row in sample_rows if not bool(row.get("should_abstain", False))]
    should_abstain_rows = [row for row in sample_rows if bool(row.get("should_abstain", False))]
    high_rows = [row for row in answerable_rows if row["predicted_confidence"] == "high"]
    medium_rows = [row for row in answerable_rows if row["predicted_confidence"] == "medium"]
    low_rows = [row for row in answerable_rows if row["predicted_confidence"] == "low"]
    failures = [row for row in answerable_rows if not bool(row["retrieval_success"])]
    low_confidence_rows = [row for row in sample_rows if row["predicted_confidence"] == "low"]
    non_low_rows = [row for row in sample_rows if row["predicted_confidence"] != "low"]
    target_low_rows = [
        row
        for row in sample_rows
        if bool(row.get("should_abstain", False)) or not bool(row["retrieval_success"])
    ]
    abstention_true_positive_count = sum(
        1
        for row in sample_rows
        if bool(row.get("should_abstain", False)) and row["predicted_confidence"] == "low"
    )
    false_confident_count = sum(
        1
        for row in sample_rows
        if bool(row.get("should_abstain", False)) and row["predicted_confidence"] != "low"
    )
    answerable_over_abstention_count = sum(
        1
        for row in sample_rows
        if not bool(row.get("should_abstain", False)) and row["predicted_confidence"] == "low"
    )

    high_confidence_precision = safe_rate(
        numerator=sum(1 for row in high_rows if bool(row["retrieval_success"])),
        denominator=len(high_rows),
    )
    low_confidence_error_capture = safe_rate(
        numerator=sum(1 for row in low_rows if not bool(row["retrieval_success"])),
        denominator=len(failures),
    )
    medium_confidence_success_rate = safe_rate(
        numerator=sum(1 for row in medium_rows if bool(row["retrieval_success"])),
        denominator=len(medium_rows),
    )
    confidence_accuracy = safe_rate(
        numerator=sum(
            1
            for row in answerable_rows
            if (
                row["predicted_confidence"] == "low" and not bool(row["retrieval_success"])
            )
            or (
                row["predicted_confidence"] in {"high", "medium"} and bool(row["retrieval_success"])
            )
        ),
        denominator=len(answerable_rows),
    )
    abstention_precision = safe_rate(
        numerator=abstention_true_positive_count,
        denominator=len(low_confidence_rows),
    )
    abstention_recall = safe_rate(
        numerator=abstention_true_positive_count,
        denominator=len(should_abstain_rows),
    )
    low_confidence_capture_rate = safe_rate(
        numerator=sum(1 for row in target_low_rows if row["predicted_confidence"] == "low"),
        denominator=len(target_low_rows),
    )
    coverage = safe_rate(
        numerator=len(non_low_rows),
        denominator=len(sample_rows),
    )
    accuracy_if_answered = safe_rate(
        numerator=sum(
            1
            for row in non_low_rows
            if not bool(row.get("should_abstain", False)) and bool(row["retrieval_success"])
        ),
        denominator=len(non_low_rows),
    )
    rerank_provider_top1_counts = {
        "cohere": sum(1 for row in sample_rows if row.get("rerank_provider_top1") == "cohere"),
        "local": sum(1 for row in sample_rows if row.get("rerank_provider_top1") == "local"),
        "none": sum(1 for row in sample_rows if not row.get("rerank_provider_top1")),
    }
    rerank_fallback_sample_count = sum(1 for row in sample_rows if bool(row.get("rerank_used_local_fallback")))

    calibration_table = []
    for bucket in ("high", "medium", "low"):
        rows = [row for row in sample_rows if row["predicted_confidence"] == bucket]
        success_count = sum(1 for row in rows if bool(row["retrieval_success"]))
        fail_count = len(rows) - success_count
        calibration_table.append(
            {
                "confidence": bucket,
                "count": len(rows),
                "success_count": success_count,
                "fail_count": fail_count,
                "success_rate": safe_rate(success_count, len(rows)),
                "avg_top1_score": round(avg(row["top1_score"] for row in rows), 6),
                "avg_score_margin": round(avg(row["top1_top2_margin"] for row in rows), 6),
            }
        )

    return {
        "answerable_samples": len(answerable_rows),
        "should_abstain_samples": len(should_abstain_rows),
        "count_by_confidence": count_by_confidence,
        "high_confidence_precision": high_confidence_precision,
        "low_confidence_error_capture": low_confidence_error_capture,
        "medium_confidence_success_rate": medium_confidence_success_rate,
        "confidence_accuracy": confidence_accuracy,
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "false_confident_count": false_confident_count,
        "low_confidence_capture_rate": low_confidence_capture_rate,
        "answerable_over_abstention_count": answerable_over_abstention_count,
        "coverage": coverage,
        "accuracy_if_answered": accuracy_if_answered,
        "rerank_provider_top1_counts": rerank_provider_top1_counts,
        "rerank_fallback_sample_count": rerank_fallback_sample_count,
        "calibration_table": calibration_table,
        "metadata": {
            "strategy": strategy,
            "score_direction": score_direction,
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
            "margin_threshold": margin_threshold,
            "strong_threshold": strong_threshold,
            "support_threshold": support_threshold,
            "high_avg_threshold": high_avg_threshold,
            "success_k": success_k,
        },
    }


def write_summary_csv(path: Path, report: dict[str, object]) -> None:
    counts = dict(report.get("count_by_confidence") or {})
    fieldnames = [
        "experiment_name",
        "split",
        "strategy",
        "score_direction",
        "evaluated_samples",
        "success_k",
        "high_threshold",
        "low_threshold",
        "margin_threshold",
        "strong_threshold",
        "support_threshold",
        "high_avg_threshold",
        "answerable_samples",
        "should_abstain_samples",
        "high_confidence_precision",
        "low_confidence_error_capture",
        "medium_confidence_success_rate",
        "confidence_accuracy",
        "abstention_precision",
        "abstention_recall",
        "false_confident_count",
        "low_confidence_capture_rate",
        "answerable_over_abstention_count",
        "coverage",
        "accuracy_if_answered",
        "rerank_fallback_sample_count",
        "count_high",
        "count_medium",
        "count_low",
    ]
    row = {
        "experiment_name": report["experiment_name"],
        "split": report["split"],
        "strategy": report["strategy"],
        "score_direction": report["score_direction"],
        "evaluated_samples": report["evaluated_samples"],
        "success_k": report["success_k"],
        "high_threshold": report["high_threshold"],
        "low_threshold": report["low_threshold"],
        "margin_threshold": report["margin_threshold"],
        "strong_threshold": report["strong_threshold"],
        "support_threshold": report["support_threshold"],
        "high_avg_threshold": report["high_avg_threshold"],
        "answerable_samples": report.get("answerable_samples", 0),
        "should_abstain_samples": report.get("should_abstain_samples", 0),
        "high_confidence_precision": report["high_confidence_precision"],
        "low_confidence_error_capture": report["low_confidence_error_capture"],
        "medium_confidence_success_rate": report["medium_confidence_success_rate"],
        "confidence_accuracy": report["confidence_accuracy"],
        "abstention_precision": report.get("abstention_precision", 0.0),
        "abstention_recall": report.get("abstention_recall", 0.0),
        "false_confident_count": report.get("false_confident_count", 0),
        "low_confidence_capture_rate": report.get("low_confidence_capture_rate", 0.0),
        "answerable_over_abstention_count": report.get("answerable_over_abstention_count", 0),
        "coverage": report.get("coverage", 0.0),
        "accuracy_if_answered": report.get("accuracy_if_answered", 0.0),
        "rerank_fallback_sample_count": report.get("rerank_fallback_sample_count", 0),
        "count_high": counts.get("high", 0),
        "count_medium": counts.get("medium", 0),
        "count_low": counts.get("low", 0),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "experiment_name",
        "split",
        "strategy",
        "score_direction",
        "evaluated_samples",
        "strong_threshold",
        "support_threshold",
        "high_avg_threshold",
        "high_confidence_precision",
        "low_confidence_error_capture",
        "medium_confidence_success_rate",
        "confidence_accuracy",
        "abstention_precision",
        "abstention_recall",
        "false_confident_count",
        "low_confidence_capture_rate",
        "answerable_over_abstention_count",
        "coverage",
        "accuracy_if_answered",
        "count_high",
        "count_medium",
        "count_low",
        "low_ratio",
        "passes_constraints",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_tuning_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "strategy",
        "score_direction",
        "high_threshold",
        "low_threshold",
        "margin_threshold",
        "strong_threshold",
        "support_threshold",
        "high_avg_threshold",
        "build_abstention_precision",
        "build_abstention_recall",
        "build_false_confident_count",
        "build_low_confidence_capture_rate",
        "build_answerable_over_abstention_count",
        "build_over_abstention_rate",
        "build_coverage",
        "build_accuracy_if_answered",
        "dev_abstention_precision",
        "dev_abstention_recall",
        "dev_false_confident_count",
        "dev_low_confidence_capture_rate",
        "dev_answerable_over_abstention_count",
        "dev_over_abstention_rate",
        "dev_coverage",
        "dev_accuracy_if_answered",
        "avg_abstention_f1",
        "avg_accuracy_if_answered",
        "avg_low_confidence_capture_rate",
        "avg_coverage",
        "avg_over_abstention_rate",
        "stability_gap",
        "selection_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def infer_split(payload: dict[str, object]) -> str:
    experiment_name = str(payload.get("experiment_name") or "")
    dataset = str(payload.get("dataset") or "")
    for value in (experiment_name, dataset):
        if "_build" in value or "rag_build" in value:
            return "build"
        if "_dev" in value or "rag_dev" in value:
            return "dev"
        if "_test" in value or "rag_test" in value:
            return "test"
        if "_reserve" in value or "rag_reserve" in value:
            return "reserve"
    return "unknown"


def extract_score(results: list[dict[str, object]], *, index: int) -> float:
    if index >= len(results):
        return 0.0
    return extract_item_score(results[index])


def extract_item_score(item: dict[str, object]) -> float:
    if "rerank_score" in item:
        return to_float(item.get("rerank_score"))
    return to_float(item.get("score"))


def extract_rerank_score(results: list[dict[str, object]], *, index: int) -> float:
    if index >= len(results):
        return 0.0
    item = dict(results[index] or {})
    if "rerank_score" in item:
        return to_float(item.get("rerank_score"))
    return extract_item_score(item)


def load_results_payload(results_path: Path) -> dict[str, object]:
    return json.loads(results_path.resolve().read_text(encoding="utf-8"))


def load_label_rows(labels_path: Path | None) -> list[dict[str, object]]:
    if labels_path is None:
        return []
    rows = []
    for line in labels_path.resolve().read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def merge_samples_with_labels(
    *,
    payload: dict[str, object],
    label_rows: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    payload_samples = list(payload.get("per_sample") or [])
    if not label_rows:
        return [{"sample": sample, "label": {}} for sample in payload_samples]

    sample_by_id = {str(sample.get("id") or ""): sample for sample in payload_samples}
    merged = []
    seen = set()
    for label in label_rows:
        sample_id = str(label.get("id") or "")
        seen.add(sample_id)
        sample = sample_by_id.get(sample_id)
        if sample is None:
            sample = {
                "id": sample_id,
                "question_type": label.get("question_type", ""),
                "source_ids": list(label.get("source_ids") or []),
                "candidate_results": [],
                "final_results": [],
                "retrieved": [],
                "candidate_hit_at_k": {},
                "candidate_recall_at_k": {},
                "hit_at_k": {},
                "recall_at_k": {},
                "first_relevant_rank": 0,
                "mrr": 0.0,
                "gold_in_candidate_not_final": False,
                "errors": [],
            }
        merged.append({"sample": sample, "label": label})
    for sample in payload_samples:
        sample_id = str(sample.get("id") or "")
        if sample_id not in seen:
            merged.append({"sample": sample, "label": {}})
    return merged


def compact_confidence_row(report: dict[str, object]) -> dict[str, object]:
    counts = dict(report.get("count_by_confidence") or {})
    return {
        "experiment_name": report.get("experiment_name", ""),
        "split": report.get("split", ""),
        "strategy": report.get("strategy", ""),
        "score_direction": report.get("score_direction", ""),
        "evaluated_samples": int(report.get("evaluated_samples", 0) or 0),
        "strong_threshold": report.get("strong_threshold", ""),
        "support_threshold": report.get("support_threshold", ""),
        "high_avg_threshold": report.get("high_avg_threshold", ""),
        "high_confidence_precision": report.get("high_confidence_precision", 0.0),
        "low_confidence_error_capture": report.get("low_confidence_error_capture", 0.0),
        "medium_confidence_success_rate": report.get("medium_confidence_success_rate", 0.0),
        "confidence_accuracy": report.get("confidence_accuracy", 0.0),
        "abstention_precision": report.get("abstention_precision", 0.0),
        "abstention_recall": report.get("abstention_recall", 0.0),
        "false_confident_count": report.get("false_confident_count", 0),
        "low_confidence_capture_rate": report.get("low_confidence_capture_rate", 0.0),
        "answerable_over_abstention_count": report.get("answerable_over_abstention_count", 0),
        "coverage": report.get("coverage", 0.0),
        "accuracy_if_answered": report.get("accuracy_if_answered", 0.0),
        "count_high": int(counts.get("high", 0) or 0),
        "count_medium": int(counts.get("medium", 0) or 0),
        "count_low": int(counts.get("low", 0) or 0),
    }


def parse_threshold_grid(raw: str, default_value: float) -> list[float]:
    if raw.strip():
        values = [round(float(item.strip()), 6) for item in raw.split(",") if item.strip()]
    else:
        values = [
            round(max(0.0, default_value - 0.18), 6),
            round(max(0.0, default_value - 0.1), 6),
            round(max(0.0, default_value - 0.05), 6),
            round(default_value, 6),
            round(min(1.0, default_value + 0.05), 6),
            round(min(1.0, default_value + 0.12), 6),
        ]
    return sorted(set(values))


def parse_grid_with_default(raw: str, default_values: list[float]) -> list[float]:
    if raw.strip():
        return sorted(set(round(float(item.strip()), 6) for item in raw.split(",") if item.strip()))
    return sorted(set(round(value, 6) for value in default_values))


def parse_strategies(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return ["rank_and_margin", "system_top3_support"]
    return values


def compute_score_margin(top1_score: float, top2_score: float, score_direction: str) -> float:
    if score_direction == "higher_is_better":
        return top1_score - top2_score
    return top2_score - top1_score


def count_rerank_providers(results: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        provider = str(item.get("rerank_provider") or "")
        if not provider:
            continue
        counts[provider] = counts.get(provider, 0) + 1
    return counts


def is_support_candidate(
    *,
    item: dict[str, object],
    top1_source_id: str,
    support_threshold: float,
    score_direction: str,
) -> bool:
    if not item:
        return False
    if str(item.get("source_id") or "") != top1_source_id:
        return False
    return meets_threshold(extract_item_score(item), support_threshold, score_direction)


def meets_threshold(score: float, threshold: float, score_direction: str) -> bool:
    if score_direction == "higher_is_better":
        return score >= threshold
    return score <= threshold


def is_worse_than(score: float, threshold: float, score_direction: str) -> bool:
    if score_direction == "higher_is_better":
        return score < threshold
    return score > threshold


def build_confidence_debug(
    *,
    top1_score: float,
    top2_score: float,
    top3_score: float,
    avg_top3_score: float | None = None,
    strong_count: int = 0,
    support_count: int = 0,
    top1_is_strong: bool = False,
    top2_is_support: bool = False,
    top3_is_support: bool = False,
    score_direction: str,
    thresholds: dict[str, float],
) -> dict[str, object]:
    effective_avg = avg_top3_score
    if effective_avg is None:
        effective_avg = avg(score for score in (top1_score, top2_score, top3_score) if score != 0.0)
    return {
        "top1_score": round(top1_score, 6),
        "top2_score": round(top2_score, 6),
        "top3_score": round(top3_score, 6),
        "avg_top3_score": round(effective_avg, 6),
        "strong_count": strong_count,
        "support_count": support_count,
        "top1_is_strong": top1_is_strong,
        "top2_is_support": top2_is_support,
        "top3_is_support": top3_is_support,
        "score_direction": score_direction,
        "thresholds": thresholds,
    }


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def f1_score(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 6)


def avg(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(item) for item in values) / len(values)


def build_tuning_row(
    *,
    strategy: str,
    config: dict[str, object],
    build_report: dict[str, object],
    dev_report: dict[str, object],
) -> dict[str, object]:
    build_answerable_samples = int(build_report.get("answerable_samples", 0) or 0)
    dev_answerable_samples = int(dev_report.get("answerable_samples", 0) or 0)
    build_precision = to_float(build_report.get("abstention_precision"))
    build_recall = to_float(build_report.get("abstention_recall"))
    dev_precision = to_float(dev_report.get("abstention_precision"))
    dev_recall = to_float(dev_report.get("abstention_recall"))
    build_f1 = f1_score(build_precision, build_recall)
    dev_f1 = f1_score(dev_precision, dev_recall)
    avg_f1 = round((build_f1 + dev_f1) / 2, 6)
    avg_accuracy_if_answered = round(
        (
            to_float(build_report.get("accuracy_if_answered"))
            + to_float(dev_report.get("accuracy_if_answered"))
        )
        / 2,
        6,
    )
    avg_low_capture = round(
        (
            to_float(build_report.get("low_confidence_capture_rate"))
            + to_float(dev_report.get("low_confidence_capture_rate"))
        )
        / 2,
        6,
    )
    avg_coverage = round(
        (to_float(build_report.get("coverage")) + to_float(dev_report.get("coverage"))) / 2,
        6,
    )
    build_over_abstention_rate = safe_rate(
        int(build_report.get("answerable_over_abstention_count", 0) or 0),
        build_answerable_samples,
    )
    dev_over_abstention_rate = safe_rate(
        int(dev_report.get("answerable_over_abstention_count", 0) or 0),
        dev_answerable_samples,
    )
    avg_over_abstention_rate = round((build_over_abstention_rate + dev_over_abstention_rate) / 2, 6)
    stability_gap = round(
        abs(build_f1 - dev_f1)
        + abs(
            to_float(build_report.get("accuracy_if_answered"))
            - to_float(dev_report.get("accuracy_if_answered"))
        ),
        6,
    )
    selection_score = round(
        avg_f1
        + avg_accuracy_if_answered
        + 0.5 * avg_coverage
        + 0.25 * avg_low_capture
        - 0.5 * avg_over_abstention_rate
        - 0.25 * stability_gap,
        6,
    )
    return {
        "strategy": strategy,
        "score_direction": config.get("score_direction", ""),
        "high_threshold": config.get("high_threshold", ""),
        "low_threshold": config.get("low_threshold", ""),
        "margin_threshold": config.get("margin_threshold", ""),
        "strong_threshold": config.get("strong_threshold", ""),
        "support_threshold": config.get("support_threshold", ""),
        "high_avg_threshold": config.get("high_avg_threshold", ""),
        "build_abstention_precision": build_precision,
        "build_abstention_recall": build_recall,
        "build_false_confident_count": int(build_report.get("false_confident_count", 0) or 0),
        "build_low_confidence_capture_rate": to_float(build_report.get("low_confidence_capture_rate")),
        "build_answerable_over_abstention_count": int(build_report.get("answerable_over_abstention_count", 0) or 0),
        "build_coverage": to_float(build_report.get("coverage")),
        "build_accuracy_if_answered": to_float(build_report.get("accuracy_if_answered")),
        "dev_abstention_precision": dev_precision,
        "dev_abstention_recall": dev_recall,
        "dev_false_confident_count": int(dev_report.get("false_confident_count", 0) or 0),
        "dev_low_confidence_capture_rate": to_float(dev_report.get("low_confidence_capture_rate")),
        "dev_answerable_over_abstention_count": int(dev_report.get("answerable_over_abstention_count", 0) or 0),
        "dev_coverage": to_float(dev_report.get("coverage")),
        "dev_accuracy_if_answered": to_float(dev_report.get("accuracy_if_answered")),
        "build_abstention_f1": build_f1,
        "dev_abstention_f1": dev_f1,
        "avg_abstention_f1": avg_f1,
        "avg_accuracy_if_answered": avg_accuracy_if_answered,
        "avg_low_confidence_capture_rate": avg_low_capture,
        "avg_coverage": avg_coverage,
        "build_over_abstention_rate": build_over_abstention_rate,
        "dev_over_abstention_rate": dev_over_abstention_rate,
        "avg_over_abstention_rate": avg_over_abstention_rate,
        "stability_gap": stability_gap,
        "selection_score": selection_score,
    }


def iter_strategy_configs(
    *,
    strategy: str,
    score_directions: list[str],
    high_threshold_grid: list[float],
    low_threshold_grid: list[float],
    margin_threshold_grid: list[float],
    strong_threshold_grid: list[float],
    support_threshold_grid: list[float],
    high_avg_threshold_grid: list[float],
):
    if strategy == "rank_and_margin":
        for direction in score_directions:
            for high_threshold, low_threshold, margin_threshold in itertools.product(
                high_threshold_grid,
                low_threshold_grid,
                margin_threshold_grid,
            ):
                if low_threshold > high_threshold:
                    continue
                yield {
                    "score_direction": direction,
                    "high_threshold": high_threshold,
                    "low_threshold": low_threshold,
                    "margin_threshold": margin_threshold,
                }
        return

    if strategy == "score_threshold":
        for direction in score_directions:
            for high_threshold, low_threshold in itertools.product(
                high_threshold_grid,
                low_threshold_grid,
            ):
                if low_threshold > high_threshold:
                    continue
                yield {
                    "score_direction": direction,
                    "high_threshold": high_threshold,
                    "low_threshold": low_threshold,
                    "margin_threshold": 0.0,
                }
        return

    if strategy == "score_margin":
        for direction in score_directions:
            for margin_threshold in margin_threshold_grid:
                yield {
                    "score_direction": direction,
                    "high_threshold": 0.0,
                    "low_threshold": 0.0,
                    "margin_threshold": margin_threshold,
                }
        return

    if strategy == "system_top3_support":
        for direction in score_directions:
            for strong_threshold, support_threshold, high_avg_threshold in itertools.product(
                strong_threshold_grid,
                support_threshold_grid,
                high_avg_threshold_grid,
            ):
                yield {
                    "score_direction": direction,
                    "strong_threshold": strong_threshold,
                    "support_threshold": support_threshold,
                    "high_avg_threshold": high_avg_threshold,
                }
        return

    raise ValueError(f"未知 strategy: {strategy}")


def select_best_split_row(rows: list[dict[str, object]], *, split_prefix: str) -> dict[str, object]:
    return max(
        rows,
        key=lambda row: (
            float(row[f"{split_prefix}_abstention_precision"]),
            float(row[f"{split_prefix}_abstention_recall"]),
            float(row[f"{split_prefix}_accuracy_if_answered"]),
            float(row[f"{split_prefix}_low_confidence_capture_rate"]),
            -int(row[f"{split_prefix}_answerable_over_abstention_count"]),
        ),
    )


def select_best_global_row(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(
        rows,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["avg_abstention_f1"]),
            float(row["avg_accuracy_if_answered"]),
            float(row["avg_low_confidence_capture_rate"]),
            -float(row["stability_gap"]),
            strategy_rank(str(row["strategy"])),
        ),
    )


def strategy_rank(strategy: str) -> int:
    order = {
        "rank_and_margin": 3,
        "system_top3_support": 2,
        "score_threshold": 1,
        "score_margin": 0,
    }
    return order.get(strategy, -1)


def to_float(value: object) -> float:
    if value is None:
        return 0.0
    return float(value)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_repo_relative_path(path: Path) -> str:
    repo_root = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
