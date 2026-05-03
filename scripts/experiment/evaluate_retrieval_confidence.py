"""基于已有 retrieval 结果评估 retrieval-level confidence 策略。"""

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
    parser.add_argument("--results", required=True, help="已有 retrieval results JSON")
    parser.add_argument("--output", required=True, help="confidence eval JSON 输出路径")
    parser.add_argument("--summary", required=True, help="confidence eval CSV 输出路径")
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
    parser.add_argument("--diagnose-score-direction", action="store_true")
    parser.add_argument("--tune-thresholds", action="store_true")
    parser.add_argument("--success-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
) -> dict[str, object]:
    payload = load_results_payload(results_path)
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
) -> dict[str, object]:
    samples = list(payload.get("per_sample") or [])
    sample_rows = []
    for sample in samples:
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
        retrieval_success = bool(float(hit_at_k.get(str(success_k), 0.0)) >= 1.0)
        sample_rows.append(
            {
                "sample_id": str(sample.get("id") or ""),
                "question_type": str(sample.get("question_type") or ""),
                "source_ids": list(sample.get("source_ids") or []),
                "top1_score": round(top1_score, 6),
                "top2_score": round(top2_score, 6),
                "score_margin": score_margin,
                "predicted_confidence": predicted_confidence,
                "retrieval_success": retrieval_success,
                "hit_at_1": to_float(hit_at_k.get("1")),
                "hit_at_3": to_float(hit_at_k.get("3")),
                "hit_at_5": to_float(hit_at_k.get("5")),
                "hit_at_10": to_float(hit_at_k.get("10")),
                "confidence_debug": confidence_debug,
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
) -> dict[str, object]:
    payload = load_results_payload(results_path)
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
) -> dict[str, object]:
    if strategy != "system_top3_support":
        raise ValueError("--tune-thresholds 目前仅支持 system_top3_support")

    payload = load_results_payload(results_path)
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

    high_rows = [row for row in sample_rows if row["predicted_confidence"] == "high"]
    medium_rows = [row for row in sample_rows if row["predicted_confidence"] == "medium"]
    low_rows = [row for row in sample_rows if row["predicted_confidence"] == "low"]
    failures = [row for row in sample_rows if not bool(row["retrieval_success"])]

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
            for row in sample_rows
            if (
                row["predicted_confidence"] == "low" and not bool(row["retrieval_success"])
            )
            or (
                row["predicted_confidence"] in {"high", "medium"} and bool(row["retrieval_success"])
            )
        ),
        denominator=len(sample_rows),
    )

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
                "avg_score_margin": round(avg(row["score_margin"] for row in rows), 6),
            }
        )

    return {
        "count_by_confidence": count_by_confidence,
        "high_confidence_precision": high_confidence_precision,
        "low_confidence_error_capture": low_confidence_error_capture,
        "medium_confidence_success_rate": medium_confidence_success_rate,
        "confidence_accuracy": confidence_accuracy,
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
        "high_confidence_precision",
        "low_confidence_error_capture",
        "medium_confidence_success_rate",
        "confidence_accuracy",
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
        "high_confidence_precision": report["high_confidence_precision"],
        "low_confidence_error_capture": report["low_confidence_error_capture"],
        "medium_confidence_success_rate": report["medium_confidence_success_rate"],
        "confidence_accuracy": report["confidence_accuracy"],
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


def load_results_payload(results_path: Path) -> dict[str, object]:
    return json.loads(results_path.resolve().read_text(encoding="utf-8"))


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


def compute_score_margin(top1_score: float, top2_score: float, score_direction: str) -> float:
    if score_direction == "higher_is_better":
        return top1_score - top2_score
    return top2_score - top1_score


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


def avg(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(item) for item in values) / len(values)


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
