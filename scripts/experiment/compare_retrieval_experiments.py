"""汇总并对比多个 retrieval 实验结果。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUTS = [
    "aiops-docs/experiment/results/dense_no_rerank_build.json",
    "aiops-docs/experiment/results/dense_no_rerank_dev.json",
    "aiops-docs/experiment/results/dense_current_rerank_build.json",
    "aiops-docs/experiment/results/dense_current_rerank_dev.json",
]

DEFAULT_OUTPUT = Path("aiops-docs/experiment/results/retrieval_experiment_comparison.json")
DEFAULT_CSV = Path("aiops-docs/experiment/results/thesis_tables/retrieval_experiment_comparison.csv")

CSV_FIELDS = [
    "experiment_name",
    "split",
    "retrieval_strategy",
    "rerank",
    "evaluated_samples",
    "skipped_abstain",
    "candidate_top_k",
    "final_top_k",
    "candidate_hit_at_10",
    "candidate_hit_at_20",
    "candidate_hit_at_50",
    "candidate_recall_at_50",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "gold_in_candidate_not_final_count",
    "gold_promoted_by_rerank_count",
    "gold_demoted_by_rerank_count",
    "rerank_provider_local_samples",
    "rerank_provider_cohere_samples",
    "rerank_provider_local_results",
    "rerank_provider_cohere_results",
    "rerank_fallback_sample_count",
    "rerank_fallback_result_count",
    "delta_hit_at_1",
    "delta_hit_at_3",
    "delta_hit_at_5",
    "delta_hit_at_10",
    "delta_mrr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总并对比 retrieval 实验结果")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=DEFAULT_INPUTS,
        help="实验结果 JSON 路径列表",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="汇总 JSON 输出路径",
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="论文表格 CSV 输出路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_retrieval_experiments(
        input_paths=[Path(item) for item in args.inputs],
        output_path=Path(args.output),
        csv_path=Path(args.csv),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def compare_retrieval_experiments(
    *,
    input_paths: list[Path],
    output_path: Path,
    csv_path: Path,
) -> dict[str, object]:
    resolved_inputs = [path.resolve() for path in input_paths]
    experiments = [load_json(path) for path in resolved_inputs]
    rows = [build_comparison_row(item) for item in experiments]

    baselines = {
        row["split"]: row
        for row in rows
        if row["retrieval_strategy"] == "dense" and row["rerank"] == "none"
    }

    for row in rows:
        baseline = baselines.get(str(row["split"]))
        apply_delta(row, baseline)

    rows.sort(key=sort_key)

    report = {
        "inputs": [to_repo_relative_path(path) for path in resolved_inputs],
        "baselines": {split: row["experiment_name"] for split, row in baselines.items()},
        "experiments": rows,
    }
    write_json(output_path.resolve(), report)
    write_csv(csv_path.resolve(), rows)
    return report


def build_comparison_row(payload: dict[str, object]) -> dict[str, object]:
    dataset = str(payload.get("dataset") or "")
    split = infer_split(payload, dataset)
    candidate_metrics = dict(payload.get("candidate_metrics") or {})
    final_metrics = dict(payload.get("final_metrics") or payload.get("metrics") or {})
    provider_stats = summarize_rerank_providers(payload)
    return {
        "experiment_name": str(payload.get("experiment_name") or ""),
        "split": split,
        "retrieval_strategy": str(payload.get("retrieval_strategy") or ""),
        "rerank": str(payload.get("rerank") or ""),
        "evaluated_samples": int(payload.get("evaluated_samples") or 0),
        "skipped_abstain": int(payload.get("skipped_abstain") or 0),
        "candidate_top_k": int(payload.get("candidate_top_k") or 0),
        "final_top_k": int(payload.get("final_top_k") or 0),
        "candidate_hit_at_10": to_float(candidate_metrics.get("candidate_hit_at_10")),
        "candidate_hit_at_20": to_float(candidate_metrics.get("candidate_hit_at_20")),
        "candidate_hit_at_50": to_float(candidate_metrics.get("candidate_hit_at_50")),
        "candidate_recall_at_50": to_float(candidate_metrics.get("candidate_recall_at_50")),
        "hit_at_1": to_float(final_metrics.get("hit_at_1")),
        "hit_at_3": to_float(final_metrics.get("hit_at_3")),
        "hit_at_5": to_float(final_metrics.get("hit_at_5")),
        "hit_at_10": to_float(final_metrics.get("hit_at_10")),
        "recall_at_1": to_float(final_metrics.get("recall_at_1")),
        "recall_at_3": to_float(final_metrics.get("recall_at_3")),
        "recall_at_5": to_float(final_metrics.get("recall_at_5")),
        "recall_at_10": to_float(final_metrics.get("recall_at_10")),
        "mrr": to_float(final_metrics.get("mrr")),
        "gold_in_candidate_not_final_count": int(payload.get("gold_in_candidate_not_final_count") or 0),
        "gold_promoted_by_rerank_count": int(payload.get("gold_promoted_by_rerank_count") or 0),
        "gold_demoted_by_rerank_count": int(payload.get("gold_demoted_by_rerank_count") or 0),
        **provider_stats,
        "delta_hit_at_1": 0.0,
        "delta_hit_at_3": 0.0,
        "delta_hit_at_5": 0.0,
        "delta_hit_at_10": 0.0,
        "delta_mrr": 0.0,
    }


def infer_split(payload: dict[str, object], dataset: str) -> str:
    experiment_name = str(payload.get("experiment_name") or "")
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


def apply_delta(row: dict[str, object], baseline: dict[str, object] | None) -> None:
    if baseline is None:
        return
    for metric_name in ("hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr"):
        delta_key = f"delta_{metric_name}"
        row[delta_key] = round(to_float(row[metric_name]) - to_float(baseline[metric_name]), 6)


def sort_key(row: dict[str, object]) -> tuple[str, int, str]:
    retrieval_order = {"dense": 0, "hybrid": 1}
    rerank_order = {"none": 0, "current": 1}
    return (
        str(row["split"]),
        retrieval_order.get(str(row["retrieval_strategy"]), 99),
        rerank_order.get(str(row["rerank"]), 99),
        str(row["experiment_name"]),
    )


def summarize_rerank_providers(payload: dict[str, object]) -> dict[str, int]:
    provider_local_samples = 0
    provider_cohere_samples = 0
    provider_local_results = 0
    provider_cohere_results = 0

    per_sample = payload.get("per_sample") or []
    for sample in per_sample:
        final_results = sample.get("final_results") or []
        sample_providers: set[str] = set()
        for item in final_results:
            provider = str(item.get("rerank_provider") or "").strip()
            if not provider:
                continue
            sample_providers.add(provider)
            if provider == "local":
                provider_local_results += 1
            elif provider == "cohere":
                provider_cohere_results += 1

        if "local" in sample_providers:
            provider_local_samples += 1
        if "cohere" in sample_providers:
            provider_cohere_samples += 1

    fallback_sample_count = provider_local_samples if str(payload.get("rerank") or "") == "current" else 0
    fallback_result_count = provider_local_results if str(payload.get("rerank") or "") == "current" else 0
    return {
        "rerank_provider_local_samples": provider_local_samples,
        "rerank_provider_cohere_samples": provider_cohere_samples,
        "rerank_provider_local_results": provider_local_results,
        "rerank_provider_cohere_results": provider_cohere_results,
        "rerank_fallback_sample_count": fallback_sample_count,
        "rerank_fallback_result_count": fallback_result_count,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_float(value: object) -> float:
    if value is None:
        return 0.0
    return round(float(value), 6)


def to_repo_relative_path(path: Path) -> str:
    repo_root = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
