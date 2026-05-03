"""汇总毕业论文实验数据、结果表格与阶段性结论。"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("aiops-docs/experiment/reports")

DEFAULT_FILES = {
    "source_manifest": "aiops-docs/experiment/sources/source_manifest.json",
    "chunk_build_report": "aiops-docs/experiment/chunks/chunk_build_report.json",
    "annotation_pool_report": "aiops-docs/experiment/chunks/annotation_pool_report.json",
    "candidate_questions": "aiops-docs/experiment/rag/rag_candidate_questions.jsonl",
    "reviewed_candidates": "aiops-docs/experiment/rag/rag_candidate_questions.reviewed.jsonl",
    "candidate_review_import_report": "aiops-docs/experiment/rag/candidate_review_import_report.json",
    "rag_dataset": "aiops-docs/experiment/rag/experiment_rag_dataset.jsonl",
    "validated_dataset": "aiops-docs/experiment/rag/experiment_rag_dataset.validated.jsonl",
    "rag_validation_report": "aiops-docs/experiment/rag/rag_validation_report.json",
    "rag_split_report": "aiops-docs/experiment/rag/splits/rag_split_report.json",
    "rag_build": "aiops-docs/experiment/rag/splits/rag_build.jsonl",
    "rag_dev": "aiops-docs/experiment/rag/splits/rag_dev.jsonl",
    "rag_test": "aiops-docs/experiment/rag/splits/rag_test.jsonl",
    "rag_reserve": "aiops-docs/experiment/rag/splits/rag_reserve.jsonl",
    "retrieval_dense_no_rerank_build": "aiops-docs/experiment/results/dense_no_rerank_build.json",
    "retrieval_dense_no_rerank_dev": "aiops-docs/experiment/results/dense_no_rerank_dev.json",
    "retrieval_dense_current_rerank_build": "aiops-docs/experiment/results/dense_current_rerank_build.json",
    "retrieval_dense_current_rerank_dev": "aiops-docs/experiment/results/dense_current_rerank_dev.json",
    "retrieval_comparison": "aiops-docs/experiment/results/retrieval_experiment_comparison.json",
    "confidence_rank_and_margin_build": "aiops-docs/experiment/results/confidence_eval_dense_current_rerank_build.json",
    "confidence_rank_and_margin_dev": "aiops-docs/experiment/results/confidence_eval_dense_current_rerank_dev.json",
    "confidence_system_top3_support_build": "aiops-docs/experiment/results/confidence_eval_system_top3_support_dense_current_rerank_build.json",
    "confidence_system_top3_support_dev": "aiops-docs/experiment/results/confidence_eval_system_top3_support_dense_current_rerank_dev.json",
    "confidence_tuning_build": "aiops-docs/experiment/results/confidence_tuning_system_top3_support_build.json",
    "confidence_tuning_dev": "aiops-docs/experiment/results/confidence_tuning_system_top3_support_dev.json",
    "confidence_score_threshold_build": "aiops-docs/experiment/results/confidence_eval_score_threshold_dense_current_rerank_build.json",
    "confidence_score_threshold_dev": "aiops-docs/experiment/results/confidence_eval_score_threshold_dense_current_rerank_dev.json",
    "confidence_score_margin_build": "aiops-docs/experiment/results/confidence_eval_score_margin_dense_current_rerank_build.json",
    "confidence_score_margin_dev": "aiops-docs/experiment/results/confidence_eval_score_margin_dense_current_rerank_dev.json",
}

RETRIEVAL_KEYS = [
    "retrieval_dense_no_rerank_build",
    "retrieval_dense_no_rerank_dev",
    "retrieval_dense_current_rerank_build",
    "retrieval_dense_current_rerank_dev",
]

CONFIDENCE_EVAL_KEYS = [
    ("score_threshold", "build", "confidence_score_threshold_build"),
    ("score_threshold", "dev", "confidence_score_threshold_dev"),
    ("score_margin", "build", "confidence_score_margin_build"),
    ("score_margin", "dev", "confidence_score_margin_dev"),
    ("rank_and_margin", "build", "confidence_rank_and_margin_build"),
    ("rank_and_margin", "dev", "confidence_rank_and_margin_dev"),
    ("system_top3_support", "build", "confidence_system_top3_support_build"),
    ("system_top3_support", "dev", "confidence_system_top3_support_dev"),
]

CONFIDENCE_TUNING_KEYS = [
    ("system_top3_support_tuned", "build", "confidence_tuning_build"),
    ("system_top3_support_tuned", "dev", "confidence_tuning_dev"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理实验数据、结果和阶段性结论")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="汇总输出目录",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = summarize_experiment_results(output_dir=Path(args.output_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def summarize_experiment_results(
    *,
    output_dir: Path,
    files: dict[str, str] | None = None,
) -> dict[str, object]:
    file_map = {**DEFAULT_FILES, **(files or {})}
    warnings: list[str] = []
    missing_files: list[str] = []
    files_used: list[str] = []

    dataset = summarize_dataset(file_map, warnings, missing_files, files_used)
    retrieval = summarize_retrieval(file_map, warnings, missing_files, files_used)
    confidence = summarize_confidence(file_map, warnings, missing_files, files_used)

    findings = build_findings(dataset, retrieval, confidence, missing_files)
    generated_at = datetime.now(timezone.utc).isoformat()

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = output_dir / "experiment_progress_summary.md"
    dataset_csv_path = output_dir / "dataset_summary.csv"
    retrieval_csv_path = output_dir / "retrieval_results_summary.csv"
    confidence_csv_path = output_dir / "confidence_results_summary.csv"
    findings_json_path = output_dir / "experiment_findings.json"
    next_steps_path = output_dir / "next_steps.md"

    progress_markdown = build_progress_markdown(
        dataset=dataset,
        retrieval=retrieval,
        confidence=confidence,
        findings=findings,
        missing_files=missing_files,
        warnings=warnings,
    )
    next_steps_markdown = build_next_steps_markdown()

    progress_path.write_text(progress_markdown, encoding="utf-8")
    next_steps_path.write_text(next_steps_markdown, encoding="utf-8")
    write_csv(dataset_csv_path, dataset["rows"], DATASET_CSV_FIELDS)
    write_csv(retrieval_csv_path, retrieval["rows"], RETRIEVAL_CSV_FIELDS)
    write_csv(confidence_csv_path, confidence["rows"], CONFIDENCE_CSV_FIELDS)

    findings_payload = {
        "dataset_status": findings["dataset_status"],
        "retrieval_findings": findings["retrieval_findings"],
        "confidence_findings": findings["confidence_findings"],
        "limitations": findings["limitations"],
        "next_steps": findings["next_steps"],
        "files_used": sorted(set(files_used)),
        "missing_files": sorted(set(missing_files)),
        "warnings": warnings,
        "generated_at": generated_at,
    }
    findings_json_path.write_text(
        json.dumps(findings_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": to_repo_relative_path(output_dir),
        "generated_files": [
            to_repo_relative_path(progress_path),
            to_repo_relative_path(dataset_csv_path),
            to_repo_relative_path(retrieval_csv_path),
            to_repo_relative_path(confidence_csv_path),
            to_repo_relative_path(findings_json_path),
            to_repo_relative_path(next_steps_path),
        ],
        "missing_files": sorted(set(missing_files)),
        "warnings": warnings,
        "dataset_summary": dataset["summary"],
        "retrieval_summary": retrieval["summary"],
        "confidence_summary": confidence["summary"],
    }


def summarize_dataset(
    file_map: dict[str, str],
    warnings: list[str],
    missing_files: list[str],
    files_used: list[str],
) -> dict[str, object]:
    source_manifest = load_json_optional(file_map["source_manifest"], warnings, missing_files, files_used)
    chunk_report = load_json_optional(file_map["chunk_build_report"], warnings, missing_files, files_used)
    annotation_report = load_json_optional(file_map["annotation_pool_report"], warnings, missing_files, files_used)
    review_report = load_json_optional(
        file_map["candidate_review_import_report"],
        warnings,
        missing_files,
        files_used,
    )
    validation_report = load_json_optional(file_map["rag_validation_report"], warnings, missing_files, files_used)
    split_report = load_json_optional(file_map["rag_split_report"], warnings, missing_files, files_used)

    source_count = int(value_or_default(source_manifest, "total_sources", 0))
    total_chunks = int(value_or_default(chunk_report, "total_chunks", 0))
    chunks_by_source = dict(value_or_default(chunk_report, "count_by_source", {}))
    chunks_by_chunk_type = dict(value_or_default(chunk_report, "count_by_chunk_type", {}))
    annotation_pool_total = int(value_or_default(annotation_report, "candidate_chunks", 0))
    annotation_priority_distribution = dict(value_or_default(annotation_report, "count_by_annotation_priority", {}))
    candidate_questions_total = count_jsonl_optional(
        file_map["candidate_questions"],
        warnings,
        missing_files,
        files_used,
    )
    reviewed_total = int(
        value_or_default(review_report, "review_rows", value_or_default(review_report, "imported_reviews", 0))
    )
    revised_total = int(value_or_default(review_report, "revised", 0))
    rejected_total = int(value_or_default(review_report, "rejected", 0))
    final_rag_samples = count_jsonl_optional(
        file_map["validated_dataset"],
        warnings,
        missing_files,
        files_used,
    )
    valid_samples = int(value_or_default(validation_report, "valid_samples", 0))
    invalid_samples = int(value_or_default(validation_report, "invalid_samples", 0))
    split_counts = dict(value_or_default(split_report, "count_by_split", {}))
    split_build = int(split_counts.get("build", 0))
    split_dev = int(split_counts.get("dev", 0))
    split_test = int(split_counts.get("test", 0))
    split_reserve = int(split_counts.get("reserve", 0))

    rows: list[dict[str, object]] = []
    rows.append(make_dataset_row("dataset", "source_count", source_count, "source_manifest.total_sources"))
    rows.append(make_dataset_row("dataset", "total_chunks", total_chunks, "chunk_build_report.total_chunks"))
    for source_id, count in sorted(chunks_by_source.items()):
        rows.append(make_dataset_row("dataset", "chunks_by_source", count, source_id))
    for chunk_type, count in sorted(chunks_by_chunk_type.items()):
        rows.append(make_dataset_row("dataset", "chunks_by_chunk_type", count, chunk_type))
    rows.append(
        make_dataset_row(
            "annotation_pool",
            "annotation_pool_total",
            annotation_pool_total,
            "annotation_pool_report.candidate_chunks",
        )
    )
    for priority, count in sorted(annotation_priority_distribution.items()):
        rows.append(make_dataset_row("annotation_pool", "annotation_priority_distribution", count, priority))
    rows.append(
        make_dataset_row(
            "candidate_questions",
            "candidate_questions_total",
            candidate_questions_total,
            "rag_candidate_questions.jsonl",
        )
    )
    rows.append(make_dataset_row("review", "reviewed_total", reviewed_total, "candidate_review_import_report"))
    rows.append(make_dataset_row("review", "revised_total", revised_total, "candidate_review_import_report"))
    rows.append(make_dataset_row("review", "rejected_total", rejected_total, "candidate_review_import_report"))
    rows.append(make_dataset_row("rag_dataset", "final_rag_samples", final_rag_samples, "validated dataset lines"))
    rows.append(make_dataset_row("rag_dataset", "valid_samples", valid_samples, "rag_validation_report.valid_samples"))
    rows.append(make_dataset_row("rag_dataset", "invalid_samples", invalid_samples, "rag_validation_report.invalid_samples"))
    rows.append(make_dataset_row("split", "split_build", split_build, "rag_split_report.count_by_split.build"))
    rows.append(make_dataset_row("split", "split_dev", split_dev, "rag_split_report.count_by_split.dev"))
    rows.append(make_dataset_row("split", "split_test", split_test, "rag_split_report.count_by_split.test"))
    rows.append(make_dataset_row("split", "split_reserve", split_reserve, "rag_split_report.count_by_split.reserve"))

    summary = {
        "source_count": source_count,
        "total_chunks": total_chunks,
        "chunks_by_source": chunks_by_source,
        "chunks_by_chunk_type": chunks_by_chunk_type,
        "annotation_pool_total": annotation_pool_total,
        "annotation_priority_distribution": annotation_priority_distribution,
        "candidate_questions_total": candidate_questions_total,
        "reviewed_total": reviewed_total,
        "revised_total": revised_total,
        "rejected_total": rejected_total,
        "final_rag_samples": final_rag_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "split_counts": {
            "build": split_build,
            "dev": split_dev,
            "test": split_test,
            "reserve": split_reserve,
        },
    }
    return {"rows": rows, "summary": summary}


def summarize_retrieval(
    file_map: dict[str, str],
    warnings: list[str],
    missing_files: list[str],
    files_used: list[str],
) -> dict[str, object]:
    rows = []
    baselines_by_split: dict[str, dict[str, object]] = {}
    for key in RETRIEVAL_KEYS:
        payload = load_json_optional(file_map[key], warnings, missing_files, files_used)
        row = build_retrieval_row(payload, file_map[key], warnings)
        rows.append(row)
        if row["retrieval_strategy"] == "dense" and row["rerank"] == "none":
            baselines_by_split[str(row["split"])] = row

    for row in rows:
        baseline = baselines_by_split.get(str(row["split"]))
        if baseline:
            for metric_name in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr"):
                delta_key = f"delta_{metric_name}"
                row[delta_key] = round(to_float(row[metric_name]) - to_float(baseline[metric_name]), 6)

    rows.sort(key=lambda item: (str(item["split"]), str(item["rerank"]), str(item["experiment_name"])))
    summary = build_retrieval_summary(rows)
    return {"rows": rows, "summary": summary}


def summarize_confidence(
    file_map: dict[str, str],
    warnings: list[str],
    missing_files: list[str],
    files_used: list[str],
) -> dict[str, object]:
    rows = []
    for strategy, split, key in CONFIDENCE_EVAL_KEYS:
        payload = load_json_optional(file_map[key], warnings, missing_files, files_used)
        rows.append(build_confidence_eval_row(strategy, split, payload, file_map[key], warnings))
    for strategy, split, key in CONFIDENCE_TUNING_KEYS:
        payload = load_json_optional(file_map[key], warnings, missing_files, files_used)
        rows.append(build_confidence_tuning_row(strategy, split, payload, file_map[key], warnings))
    rows.sort(key=lambda item: (str(item["split"]), str(item["strategy"])))
    summary = build_confidence_summary(rows)
    return {"rows": rows, "summary": summary}


def build_retrieval_row(
    payload: dict[str, object] | None,
    path_text: str,
    warnings: list[str],
) -> dict[str, object]:
    if payload is None:
        return {
            "experiment_name": path_stem(path_text),
            "split": infer_split_from_name(path_text),
            "retrieval_strategy": "",
            "rerank": "",
            "evaluated_samples": "",
            "candidate_top_k": "",
            "final_top_k": "",
            "candidate_hit_at_10": "",
            "candidate_hit_at_20": "",
            "candidate_hit_at_50": "",
            "hit_at_1": "",
            "hit_at_3": "",
            "hit_at_5": "",
            "hit_at_10": "",
            "recall_at_1": "",
            "recall_at_3": "",
            "recall_at_5": "",
            "recall_at_10": "",
            "mrr": "",
            "delta_hit_at_1": "",
            "delta_hit_at_3": "",
            "delta_hit_at_5": "",
            "delta_hit_at_10": "",
            "delta_mrr": "",
            "gold_in_candidate_not_final_count": "",
            "gold_promoted_by_rerank_count": "",
            "gold_demoted_by_rerank_count": "",
            "notes": f"missing_file:{path_text}",
        }

    candidate_metrics = dict(payload.get("candidate_metrics") or {})
    final_metrics = dict(payload.get("final_metrics") or payload.get("metrics") or {})
    experiment_name = str(payload.get("experiment_name") or path_stem(path_text))
    split = infer_split_from_name(str(payload.get("dataset") or experiment_name))
    gold_in_candidate_not_final = payload.get("gold_in_candidate_not_final_count")
    if gold_in_candidate_not_final in (None, 0):
        computed_value = compute_gold_in_candidate_not_final(payload)
        if computed_value is not None:
            gold_in_candidate_not_final = computed_value

    notes = []
    rerank = str(payload.get("rerank") or "")
    if rerank == "current":
        notes.append("current system rerank; online rerank with local fallback")
    elif rerank == "none":
        notes.append("dense baseline without rerank")

    return {
        "experiment_name": experiment_name,
        "split": split,
        "retrieval_strategy": str(payload.get("retrieval_strategy") or ""),
        "rerank": rerank,
        "evaluated_samples": int(payload.get("evaluated_samples") or 0),
        "candidate_top_k": int(payload.get("candidate_top_k") or 0),
        "final_top_k": int(payload.get("final_top_k") or 0),
        "candidate_hit_at_10": to_float(candidate_metrics.get("candidate_hit_at_10")),
        "candidate_hit_at_20": to_float(candidate_metrics.get("candidate_hit_at_20")),
        "candidate_hit_at_50": to_float(candidate_metrics.get("candidate_hit_at_50")),
        "hit_at_1": to_float(final_metrics.get("hit_at_1")),
        "hit_at_3": to_float(final_metrics.get("hit_at_3")),
        "hit_at_5": to_float(final_metrics.get("hit_at_5")),
        "hit_at_10": to_float(final_metrics.get("hit_at_10")),
        "recall_at_1": to_float(final_metrics.get("recall_at_1")),
        "recall_at_3": to_float(final_metrics.get("recall_at_3")),
        "recall_at_5": to_float(final_metrics.get("recall_at_5")),
        "recall_at_10": to_float(final_metrics.get("recall_at_10")),
        "mrr": to_float(final_metrics.get("mrr")),
        "delta_hit_at_1": "",
        "delta_hit_at_3": "",
        "delta_hit_at_5": "",
        "delta_hit_at_10": "",
        "delta_mrr": "",
        "gold_in_candidate_not_final_count": int(gold_in_candidate_not_final or 0),
        "gold_promoted_by_rerank_count": int(payload.get("gold_promoted_by_rerank_count") or 0),
        "gold_demoted_by_rerank_count": int(payload.get("gold_demoted_by_rerank_count") or 0),
        "notes": "; ".join(notes),
    }


def build_confidence_eval_row(
    strategy: str,
    split: str,
    payload: dict[str, object] | None,
    path_text: str,
    warnings: list[str],
) -> dict[str, object]:
    if payload is None:
        return make_missing_confidence_row(strategy, split, path_text)
    counts = dict(payload.get("count_by_confidence") or {})
    evaluated_samples = int(payload.get("evaluated_samples") or 0)
    low_ratio = safe_rate(int(counts.get("low", 0) or 0), evaluated_samples)
    score_direction = payload.get("score_direction", "")
    if not score_direction:
        warnings.append(f"missing_metric:score_direction:{path_text}")
    return {
        "strategy": strategy,
        "split": split,
        "score_direction": score_direction,
        "high_confidence_precision": to_float(payload.get("high_confidence_precision")),
        "low_confidence_error_capture": to_float(payload.get("low_confidence_error_capture")),
        "confidence_accuracy": to_float(payload.get("confidence_accuracy")),
        "count_high": int(counts.get("high", 0) or 0),
        "count_medium": int(counts.get("medium", 0) or 0),
        "count_low": int(counts.get("low", 0) or 0),
        "low_ratio": low_ratio,
        "strong_threshold": payload.get("strong_threshold", ""),
        "support_threshold": payload.get("support_threshold", ""),
        "high_avg_threshold": payload.get("high_avg_threshold", ""),
        "notes": "" if score_direction else f"legacy output missing score_direction:{path_text}",
    }


def build_confidence_tuning_row(
    strategy: str,
    split: str,
    payload: dict[str, object] | None,
    path_text: str,
    warnings: list[str],
) -> dict[str, object]:
    if payload is None:
        return make_missing_confidence_row(strategy, split, path_text)
    best = dict(payload.get("best_config") or {})
    if not best:
        warnings.append(f"missing_metric:best_config:{path_text}")
    return {
        "strategy": strategy,
        "split": split,
        "score_direction": best.get("score_direction", ""),
        "high_confidence_precision": to_float(best.get("high_confidence_precision")),
        "low_confidence_error_capture": to_float(best.get("low_confidence_error_capture")),
        "confidence_accuracy": to_float(best.get("confidence_accuracy")),
        "count_high": int(best.get("count_high", 0) or 0),
        "count_medium": int(best.get("count_medium", 0) or 0),
        "count_low": int(best.get("count_low", 0) or 0),
        "low_ratio": to_float(best.get("low_ratio")),
        "strong_threshold": best.get("strong_threshold", ""),
        "support_threshold": best.get("support_threshold", ""),
        "high_avg_threshold": best.get("high_avg_threshold", ""),
        "notes": "best config from tuning grid search",
    }


def make_missing_confidence_row(strategy: str, split: str, path_text: str) -> dict[str, object]:
    return {
        "strategy": strategy,
        "split": split,
        "score_direction": "",
        "high_confidence_precision": "",
        "low_confidence_error_capture": "",
        "confidence_accuracy": "",
        "count_high": "",
        "count_medium": "",
        "count_low": "",
        "low_ratio": "",
        "strong_threshold": "",
        "support_threshold": "",
        "high_avg_threshold": "",
        "notes": f"missing_file:{path_text}",
    }


def build_retrieval_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    experiments = {str(row["experiment_name"]): row for row in rows}
    build_no = experiments.get("dense_no_rerank_build", {})
    build_cur = experiments.get("dense_current_rerank_build", {})
    dev_no = experiments.get("dense_no_rerank_dev", {})
    dev_cur = experiments.get("dense_current_rerank_dev", {})
    findings = [
        "dense baseline can recall many gold chunks into candidate top50" if has_value(build_no.get("candidate_hit_at_50")) else "",
        "final top10 ordering remains the bottleneck before rerank" if has_value(build_no.get("hit_at_10")) else "",
        "current rerank improves Hit@K and MRR on both build and dev" if positive_delta(build_cur, dev_cur) else "",
        "rerank mainly promotes gold chunks from candidate top50 into final top10/top5/top3/top1",
        "current rerank reflects the current system strategy: online rerank with local fallback",
    ]
    findings = [item for item in findings if item]
    return {"experiments": experiments, "findings": findings}


def build_confidence_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    strategies = {}
    for row in rows:
        strategies.setdefault(str(row["strategy"]), {})[str(row["split"])] = row
    findings = [
        "rank_and_margin is the most stable general baseline across current build/dev runs",
        "system_top3_support behaves more like a strong low-confidence interception strategy",
        "system_top3_support thresholds are not final and should be re-tuned after dataset expansion",
        "confidence results should not be over-interpreted yet because build/dev sample counts remain small",
    ]
    return {"strategies": strategies, "findings": findings}


def build_findings(
    dataset: dict[str, object],
    retrieval: dict[str, object],
    confidence: dict[str, object],
    missing_files: list[str],
) -> dict[str, object]:
    dataset_summary = dict(dataset["summary"])
    retrieval_summary = dict(retrieval["summary"])
    confidence_summary = dict(confidence["summary"])
    limitations = [
        "当前正式样本只有 64 条",
        "当前没有 test split",
        "Grundfos 样本偏少",
        "symptom_cause 样本偏少",
        "没有 cross_doc_multi",
        "没有 abstention_insufficient_evidence",
        "rerank 结果受 Cohere 429 fallback 影响",
        "confidence 阈值尚未最终确定",
    ]
    next_steps = [
        "扩展候选题到 200+，人工审核后得到 150–170 条 reviewed RAG samples",
        "增加 cross_doc_multi 和 abstention_insufficient_evidence",
        "重新 split: build/dev/test",
        "在 build/dev 上重新确定 confidence 参数",
        "冻结 test 后只跑最终策略",
        "补充 hybrid retrieval 对比",
        "将最终 confidence strategy 抽取到项目服务层",
    ]
    return {
        "dataset_status": {
            "final_rag_samples": dataset_summary.get("final_rag_samples", 0),
            "valid_samples": dataset_summary.get("valid_samples", 0),
            "split_counts": dataset_summary.get("split_counts", {}),
            "missing_files": sorted(set(missing_files)),
        },
        "retrieval_findings": retrieval_summary.get("findings", []),
        "confidence_findings": confidence_summary.get("findings", []),
        "limitations": limitations,
        "next_steps": next_steps,
    }


def build_progress_markdown(
    *,
    dataset: dict[str, object],
    retrieval: dict[str, object],
    confidence: dict[str, object],
    findings: dict[str, object],
    missing_files: list[str],
    warnings: list[str],
) -> str:
    dataset_summary = dict(dataset["summary"])
    retrieval_rows = list(retrieval["rows"])
    confidence_rows = list(confidence["rows"])
    split_counts = dict(dataset_summary.get("split_counts", {}))
    experiments = {str(row["experiment_name"]): row for row in retrieval_rows}

    build_no = experiments.get("dense_no_rerank_build", {})
    build_cur = experiments.get("dense_current_rerank_build", {})
    dev_no = experiments.get("dense_no_rerank_dev", {})
    dev_cur = experiments.get("dense_current_rerank_dev", {})

    lines = [
        "# Experiment Progress Summary",
        "",
        "## 1. Experiment Scope",
        "- 当前实验固定 `experiment_chunks.jsonl` 作为统一 evidence base。",
        "- 当前暂不验证切片策略效果。",
        "- 当前重点验证 retrieval、rerank、confidence strategy。",
        "- 当前实验属于 pilot/build-dev 阶段，不是最终 test 结果。",
        "",
        "## 2. Dataset Construction Summary",
        f"- source 文档数量：`{dataset_summary.get('source_count', 0)}`",
        f"- experiment_chunks 总数：`{dataset_summary.get('total_chunks', 0)}`",
        f"- 各 source chunk 数量：`{format_dict(dataset_summary.get('chunks_by_source', {}))}`",
        f"- 各 chunk_type 数量：`{format_dict(dataset_summary.get('chunks_by_chunk_type', {}))}`",
        f"- annotation_pool 数量：`{dataset_summary.get('annotation_pool_total', 0)}`",
        f"- annotation priority 分布：`{format_dict(dataset_summary.get('annotation_priority_distribution', {}))}`",
        f"- candidate questions 数量：`{dataset_summary.get('candidate_questions_total', 0)}`",
        f"- reviewed/revised/rejected：`{dataset_summary.get('reviewed_total', 0)}` / `{dataset_summary.get('revised_total', 0)}` / `{dataset_summary.get('rejected_total', 0)}`",
        f"- final RAG dataset 数量：`{dataset_summary.get('final_rag_samples', 0)}`",
        f"- validation valid/invalid：`{dataset_summary.get('valid_samples', 0)}` / `{dataset_summary.get('invalid_samples', 0)}`",
        f"- split 分布：`build={split_counts.get('build', 0)}, dev={split_counts.get('dev', 0)}, test={split_counts.get('test', 0)}, reserve={split_counts.get('reserve', 0)}`",
        "",
        "## 3. Current RAG Dataset Status",
        f"- 当前正式样本数：`{dataset_summary.get('final_rag_samples', 0)}`",
        f"- `build={split_counts.get('build', 0)}, dev={split_counts.get('dev', 0)}, test={split_counts.get('test', 0)}, reserve={split_counts.get('reserve', 0)}`",
        "- 当前 `test` 为空的原因：样本量还不足，后续扩展后再冻结 `test`。",
        "- 当前数据集主要用于流程验证、dev 调参与 pilot 实验。",
        "",
        "## 4. Retrieval Experiment Summary",
        f"- build evaluated_samples：`dense_no_rerank={build_no.get('evaluated_samples', '')}`, `dense_current_rerank={build_cur.get('evaluated_samples', '')}`",
        f"- dev evaluated_samples：`dense_no_rerank={dev_no.get('evaluated_samples', '')}`, `dense_current_rerank={dev_cur.get('evaluated_samples', '')}`",
        f"- build dense_no_rerank：`Hit@1/3/5/10={format_metric_tuple(build_no, ['hit_at_1','hit_at_3','hit_at_5','hit_at_10'])}`, `Recall@1/3/5/10={format_metric_tuple(build_no, ['recall_at_1','recall_at_3','recall_at_5','recall_at_10'])}`, `MRR={build_no.get('mrr', '')}`",
        f"- build dense_current_rerank：`Hit@1/3/5/10={format_metric_tuple(build_cur, ['hit_at_1','hit_at_3','hit_at_5','hit_at_10'])}`, `Recall@1/3/5/10={format_metric_tuple(build_cur, ['recall_at_1','recall_at_3','recall_at_5','recall_at_10'])}`, `MRR={build_cur.get('mrr', '')}`",
        f"- dev dense_no_rerank：`Hit@1/3/5/10={format_metric_tuple(dev_no, ['hit_at_1','hit_at_3','hit_at_5','hit_at_10'])}`, `Recall@1/3/5/10={format_metric_tuple(dev_no, ['recall_at_1','recall_at_3','recall_at_5','recall_at_10'])}`, `MRR={dev_no.get('mrr', '')}`",
        f"- dev dense_current_rerank：`Hit@1/3/5/10={format_metric_tuple(dev_cur, ['hit_at_1','hit_at_3','hit_at_5','hit_at_10'])}`, `Recall@1/3/5/10={format_metric_tuple(dev_cur, ['recall_at_1','recall_at_3','recall_at_5','recall_at_10'])}`, `MRR={dev_cur.get('mrr', '')}`",
        f"- candidate Hit@10/20/50：`build no_rerank={format_metric_tuple(build_no, ['candidate_hit_at_10','candidate_hit_at_20','candidate_hit_at_50'])}`, `build current={format_metric_tuple(build_cur, ['candidate_hit_at_10','candidate_hit_at_20','candidate_hit_at_50'])}`, `dev no_rerank={format_metric_tuple(dev_no, ['candidate_hit_at_10','candidate_hit_at_20','candidate_hit_at_50'])}`, `dev current={format_metric_tuple(dev_cur, ['candidate_hit_at_10','candidate_hit_at_20','candidate_hit_at_50'])}`",
        f"- gold_in_candidate_not_final_count：`build no_rerank={build_no.get('gold_in_candidate_not_final_count', '')}`, `build current={build_cur.get('gold_in_candidate_not_final_count', '')}`, `dev no_rerank={dev_no.get('gold_in_candidate_not_final_count', '')}`, `dev current={dev_cur.get('gold_in_candidate_not_final_count', '')}`",
        f"- gold_promoted_by_rerank_count：`build={build_cur.get('gold_promoted_by_rerank_count', '')}`, `dev={dev_cur.get('gold_promoted_by_rerank_count', '')}`",
        f"- gold_demoted_by_rerank_count：`build={build_cur.get('gold_demoted_by_rerank_count', '')}`, `dev={dev_cur.get('gold_demoted_by_rerank_count', '')}`",
        f"- rerank 相对 no_rerank 的 delta：`build Hit@10={build_cur.get('delta_hit_at_10', '')}, MRR={build_cur.get('delta_mrr', '')}; dev Hit@10={dev_cur.get('delta_hit_at_10', '')}, MRR={dev_cur.get('delta_mrr', '')}`",
        "",
        "## 5. Retrieval Findings",
    ]
    lines.extend(f"- {item}" for item in findings["retrieval_findings"])
    lines.extend(
        [
            "",
            "## 6. Confidence Experiment Summary",
        ]
    )
    for row in confidence_rows:
        lines.append(
            f"- `{row['strategy']} / {row['split']}`: "
            f"`high_precision={row['high_confidence_precision']}`, "
            f"`low_capture={row['low_confidence_error_capture']}`, "
            f"`confidence_accuracy={row['confidence_accuracy']}`, "
            f"`count_high={row['count_high']}`, `count_medium={row['count_medium']}`, `count_low={row['count_low']}`, "
            f"`low_ratio={row['low_ratio']}`, "
            f"`score_direction={row['score_direction'] or 'missing'}`, "
            f"`strong={row['strong_threshold']}`, `support={row['support_threshold']}`, `high_avg={row['high_avg_threshold']}`"
            + (f", notes=`{row['notes']}`" if row["notes"] else "")
        )
    lines.extend(
        [
            "",
            "## 7. Confidence Findings",
        ]
    )
    lines.extend(f"- {item}" for item in findings["confidence_findings"])
    lines.extend(
        [
            "",
            "## 8. Current Limitations",
        ]
    )
    lines.extend(f"- {item}" for item in findings["limitations"])
    lines.extend(
        [
            "",
            "## 9. Recommended Next Steps",
        ]
    )
    lines.extend(f"- {item}" for item in findings["next_steps"])
    if missing_files:
        lines.extend(["", "## Missing Files", *[f"- `{item}`" for item in sorted(set(missing_files))]])
    if warnings:
        lines.extend(["", "## Warnings", *[f"- `{item}`" for item in warnings]])
    lines.append("")
    return "\n".join(lines)


def build_next_steps_markdown() -> str:
    return "\n".join(
        [
            "# Next Steps",
            "",
            "## Priority 1：数据集扩展",
            "- 生成第二批候选题。",
            "- 完成人工审核。",
            "- 合并 reviewed candidates。",
            "- 构建扩展后的 experiment_rag_dataset。",
            "- 对扩展数据集执行 validate。",
            "- 重新执行 split。",
            "",
            "## Priority 2：最终检索实验",
            "- 运行 dense_no_rerank。",
            "- 运行 dense_current_rerank。",
            "- 运行 hybrid_no_rerank。",
            "- 运行 hybrid_current_rerank。",
            "",
            "## Priority 3：置信度策略复验",
            "- 在扩展后的 build/dev 上重新 tune system_top3_support。",
            "- 冻结 test 后只运行最终选定策略。",
            "",
            "## Priority 4：工程集成",
            "- 抽取 retrieval_confidence_service。",
            "- 接入 low_confidence_events。",
            "- 在前端治理模块展示 confidence_debug。",
            "",
        ]
    )


def compute_gold_in_candidate_not_final(payload: dict[str, object]) -> int | None:
    per_sample = list(payload.get("per_sample") or [])
    if not per_sample:
        return None
    total = 0
    for sample in per_sample:
        reference_ids = set(sample.get("reference_chunk_ids") or [])
        candidate_ids = [
            str(item.get("chunk_id") or "")
            for item in list(sample.get("candidate_results") or [])
            if item.get("chunk_id")
        ]
        final_ids = [
            str(item.get("chunk_id") or "")
            for item in list(sample.get("final_results") or [])
            if item.get("chunk_id")
        ]
        if reference_ids.intersection(candidate_ids) and not reference_ids.intersection(final_ids):
            total += 1
    return total


def load_json_optional(
    path_text: str,
    warnings: list[str],
    missing_files: list[str],
    files_used: list[str],
) -> dict[str, object] | None:
    path = Path(path_text)
    if not path.exists():
        missing_files.append(path_text)
        warnings.append(f"missing_file:{path_text}")
        return None
    files_used.append(to_repo_relative_path(path.resolve()))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        warnings.append(f"invalid_json:{path_text}")
        return None


def count_jsonl_optional(
    path_text: str,
    warnings: list[str],
    missing_files: list[str],
    files_used: list[str],
) -> int:
    path = Path(path_text)
    if not path.exists():
        missing_files.append(path_text)
        warnings.append(f"missing_file:{path_text}")
        return 0
    files_used.append(to_repo_relative_path(path.resolve()))
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def value_or_default(payload: dict[str, object] | None, key: str, default: object) -> object:
    if payload is None:
        return default
    return payload.get(key, default)


def make_dataset_row(section: str, metric: str, value: object, notes: str) -> dict[str, object]:
    return {
        "section": section,
        "metric": metric,
        "value": value,
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def infer_split_from_name(text: str) -> str:
    value = text.lower()
    if "_build" in value or "rag_build" in value:
        return "build"
    if "_dev" in value or "rag_dev" in value:
        return "dev"
    if "_test" in value or "rag_test" in value:
        return "test"
    if "_reserve" in value or "rag_reserve" in value:
        return "reserve"
    return "unknown"


def format_dict(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def format_metric_tuple(row: dict[str, object], keys: list[str]) -> str:
    return ", ".join(str(row.get(key, "")) for key in keys)


def positive_delta(build_cur: dict[str, object], dev_cur: dict[str, object]) -> bool:
    return (
        to_float(build_cur.get("delta_hit_at_10")) > 0
        and to_float(dev_cur.get("delta_hit_at_10")) > 0
        and to_float(build_cur.get("delta_mrr")) > 0
        and to_float(dev_cur.get("delta_mrr")) > 0
    )


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def to_float(value: object) -> float:
    if value in ("", None):
        return 0.0
    return round(float(value), 6)


def has_value(value: object) -> bool:
    return value not in ("", None)


def path_stem(path_text: str) -> str:
    return Path(path_text).stem


def to_repo_relative_path(path: Path) -> str:
    repo_root = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


DATASET_CSV_FIELDS = ["section", "metric", "value", "notes"]

RETRIEVAL_CSV_FIELDS = [
    "experiment_name",
    "split",
    "retrieval_strategy",
    "rerank",
    "evaluated_samples",
    "candidate_top_k",
    "final_top_k",
    "candidate_hit_at_10",
    "candidate_hit_at_20",
    "candidate_hit_at_50",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "delta_hit_at_1",
    "delta_hit_at_3",
    "delta_hit_at_5",
    "delta_hit_at_10",
    "delta_mrr",
    "gold_in_candidate_not_final_count",
    "gold_promoted_by_rerank_count",
    "gold_demoted_by_rerank_count",
    "notes",
]

CONFIDENCE_CSV_FIELDS = [
    "strategy",
    "split",
    "score_direction",
    "high_confidence_precision",
    "low_confidence_error_capture",
    "confidence_accuracy",
    "count_high",
    "count_medium",
    "count_low",
    "low_ratio",
    "strong_threshold",
    "support_threshold",
    "high_avg_threshold",
    "notes",
]


if __name__ == "__main__":
    raise SystemExit(main())
