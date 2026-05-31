"""Aggregate Stage A.2 answer-level judge results into thesis tables and report."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def expand_ci_fields(fields: list[str]) -> list[str]:
    output: list[str] = []
    for field in fields:
        output.extend([field, f"{field}_ci_low_95", f"{field}_ci_high_95"])
    return output


MEAN_FIELDS = [
    "faithfulness_supported_by_retrieved_mean",
    "faithfulness_supported_by_cited_mean",
    "correctness_strict_mean",
    "correctness_lenient_mean",
    "citation_precision_mean",
    "citation_recall_mean",
    "citation_f1_mean",
    "hallucination_rate_mean",
]
MAIN_FIELDS = [
    "experiment_name",
    "split",
    "evaluated_samples",
    "abstain_samples",
    *expand_ci_fields(MEAN_FIELDS),
    "abstention_precision",
    "abstention_recall",
    "generator_model",
    "judge_model",
    "prompt_version",
    "run_date",
]
QTYPE_MEAN_FIELDS = [
    "faithfulness_supported_by_retrieved_mean",
    "correctness_strict_mean",
    "citation_f1_mean",
    "hallucination_rate_mean",
]
QTYPE_FIELDS = [
    "experiment_name",
    "split",
    "question_type",
    "n_samples",
    *expand_ci_fields(QTYPE_MEAN_FIELDS),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Stage A answer-level judge JSONL into summary JSON, CSV tables, and Markdown report.",
        epilog=(
            "Experiment discipline: do not modify dataset, chunks, retrieval JSON, confidence JSON, "
            "or answers JSONL; should_abstain=true is reported separately and excluded from answerable means."
        ),
    )
    parser.add_argument("--judge-files", required=True, help="Comma-separated judge JSONL path list")
    parser.add_argument("--dataset", required=True, help="Expanded validated dataset JSONL")
    parser.add_argument("--split-files", required=True, help="Comma-separated split JSONL path list")
    parser.add_argument("--output-json", required=True, help="Summary JSON output")
    parser.add_argument("--output-csv", required=True, help="Main thesis CSV output")
    parser.add_argument("--output-report", required=True, help="Markdown report output")
    parser.add_argument("--by-question-type", action="store_true")
    parser.add_argument("--bootstrap-ci", type=int, default=1000)
    parser.add_argument("--strict-and-lenient", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    judge_paths = [path_from_repo(item.strip()).resolve() for item in args.judge_files.split(",") if item.strip()]
    split_paths = [path_from_repo(item.strip()).resolve() for item in args.split_files.split(",") if item.strip()]
    output_json = path_from_repo(args.output_json).resolve()
    output_csv = path_from_repo(args.output_csv).resolve()
    output_report = path_from_repo(args.output_report).resolve()
    by_qtype_json = infer_by_qtype_json_path(output_json)
    by_qtype_csv = infer_by_qtype_csv_path(output_csv)

    rows = []
    for path in judge_paths:
        rows.extend(read_jsonl(path))
    split_rows = []
    for path in split_paths:
        split_rows.extend(read_jsonl(path))
    split_ids = {str(row.get("id") or row.get("sample_id") or "") for row in split_rows}
    rows = [row for row in rows if str(row.get("sample_id") or "") in split_ids]
    if not rows:
        raise ValueError("No judge rows matched the provided split files")

    experiment_name = infer_experiment_name(judge_paths[0])
    split = str(rows[0].get("split") or infer_split_from_path(judge_paths[0]))
    answerable_rows = [row for row in rows if not row.get("should_abstain")]
    abstain_rows = [row for row in rows if row.get("should_abstain")]
    main_row, main_metrics = build_main_row(
        rows=rows,
        answerable_rows=answerable_rows,
        abstain_rows=abstain_rows,
        experiment_name=experiment_name,
        split=split,
        bootstrap_n=args.bootstrap_ci,
    )
    by_qtype_rows, by_qtype_json_payload = build_by_qtype_rows(
        answerable_rows=answerable_rows,
        experiment_name=experiment_name,
        split=split,
        bootstrap_n=args.bootstrap_ci,
    )
    summary = {
        "experiment_name": experiment_name,
        "split": split,
        "generated_at": current_timestamp(),
        "input_judge_files": [to_repo_relative(path) for path in judge_paths],
        "evaluated_samples": len(rows),
        "answerable_samples": len(answerable_rows),
        "should_abstain_samples": len(abstain_rows),
        "main_row": main_row,
        "metrics": main_metrics,
        "should_abstain_subset": build_abstain_subset(abstain_rows),
    }

    write_json(output_json, summary)
    write_csv(output_csv, [main_row], MAIN_FIELDS)
    if args.by_question_type:
        write_json(by_qtype_json, by_qtype_json_payload)
        write_csv(by_qtype_csv, by_qtype_rows, QTYPE_FIELDS)
    write_report(
        path=output_report,
        rows=rows,
        answerable_rows=answerable_rows,
        abstain_rows=abstain_rows,
        main_row=main_row,
        by_qtype_rows=by_qtype_rows,
        experiment_name=experiment_name,
        split=split,
        bootstrap_n=args.bootstrap_ci,
    )
    print(
        json.dumps(
            {
                "summary_json": to_repo_relative(output_json),
                "by_qtype_json": to_repo_relative(by_qtype_json) if args.by_question_type else None,
                "main_csv": to_repo_relative(output_csv),
                "by_qtype_csv": to_repo_relative(by_qtype_csv) if args.by_question_type else None,
                "report": to_repo_relative(output_report),
                "evaluated_samples": len(rows),
                "answerable_samples": len(answerable_rows),
                "should_abstain_samples": len(abstain_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_main_row(
    *,
    rows: list[dict[str, Any]],
    answerable_rows: list[dict[str, Any]],
    abstain_rows: list[dict[str, Any]],
    experiment_name: str,
    split: str,
    bootstrap_n: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_values = collect_metric_values(answerable_rows)
    row: dict[str, Any] = {
        "experiment_name": experiment_name,
        "split": split,
        "evaluated_samples": len(rows),
        "abstain_samples": len(abstain_rows),
    }
    metrics: dict[str, Any] = {}
    for field, values in metric_values.items():
        mean_value = mean(values) if values else 0.0
        low, high = bootstrap_ci(values, bootstrap_n)
        row[field] = round(mean_value, 6)
        row[f"{field}_ci_low_95"] = round(low, 6)
        row[f"{field}_ci_high_95"] = round(high, 6)
        metrics[field] = {"mean": mean_value, "ci_low_95": low, "ci_high_95": high}
    p, r = abstention_precision_recall(rows)
    row["abstention_precision"] = round(p, 6)
    row["abstention_recall"] = round(r, 6)
    row["generator_model"] = first_nonempty(
        ((row.get("answer_snapshot") or {}).get("generator_model") for row in rows)
    )
    row["judge_model"] = first_nonempty((row.get("judge_model") for row in rows))
    row["prompt_version"] = first_nonempty((row.get("judge_prompt_version") for row in rows))
    row["run_date"] = current_date()
    return row, metrics


def build_by_qtype_rows(
    *,
    answerable_rows: list[dict[str, Any]],
    experiment_name: str,
    split: str,
    bootstrap_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in answerable_rows:
        grouped[str(row.get("question_type") or "")].append(row)
    output = []
    payload = {"experiment_name": experiment_name, "split": split, "groups": {}}
    for qtype, rows in sorted(grouped.items()):
        metric_values = collect_metric_values(rows)
        out_row: dict[str, Any] = {
            "experiment_name": experiment_name,
            "split": split,
            "question_type": qtype,
            "n_samples": len(rows),
        }
        payload["groups"][qtype] = {"n_samples": len(rows)}
        for field in QTYPE_MEAN_FIELDS:
            values = metric_values[field]
            mean_value = mean(values) if values else 0.0
            low, high = bootstrap_ci(values, bootstrap_n)
            out_row[field] = round(mean_value, 6)
            out_row[f"{field}_ci_low_95"] = round(low, 6)
            out_row[f"{field}_ci_high_95"] = round(high, 6)
            payload["groups"][qtype][field] = {
                "mean": mean_value,
                "ci_low_95": low,
                "ci_high_95": high,
            }
        output.append(out_row)
    return output, payload


def collect_metric_values(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    values = {field: [] for field in MEAN_FIELDS}
    for row in rows:
        correctness = row.get("answer_correctness") or {}
        verdict = str(correctness.get("verdict") or "")
        faithfulness = row.get("faithfulness") or {}
        citation = row.get("citation_accuracy") or {}
        hallucination = row.get("hallucination") or {}
        values["faithfulness_supported_by_retrieved_mean"].append(
            float(faithfulness.get("score_supported_by_retrieved", 0.0) or 0.0)
        )
        values["faithfulness_supported_by_cited_mean"].append(
            float(faithfulness.get("score_supported_by_cited", 0.0) or 0.0)
        )
        values["correctness_strict_mean"].append(1.0 if verdict == "correct" else 0.0)
        values["correctness_lenient_mean"].append(float(correctness.get("score", 0.0) or 0.0))
        values["citation_precision_mean"].append(float(citation.get("precision", 0.0) or 0.0))
        values["citation_recall_mean"].append(float(citation.get("recall", 0.0) or 0.0))
        values["citation_f1_mean"].append(float(citation.get("f1", 0.0) or 0.0))
        values["hallucination_rate_mean"].append(float(hallucination.get("rate", 0.0) or 0.0))
    return values


def abstention_precision_recall(rows: list[dict[str, Any]]) -> tuple[float, float]:
    predicted = [row for row in rows if (row.get("abstention_check") or {}).get("model_abstained")]
    gold = [row for row in rows if row.get("should_abstain")]
    true_positive = [
        row
        for row in rows
        if row.get("should_abstain") and (row.get("abstention_check") or {}).get("model_abstained")
    ]
    return safe_div(len(true_positive), len(predicted)), safe_div(len(true_positive), len(gold))


def build_abstain_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "model_abstained": sum(1 for row in rows if (row.get("abstention_check") or {}).get("model_abstained")),
        "correct_abstention": sum(
            1 for row in rows if (row.get("abstention_check") or {}).get("abstention_correct")
        ),
        "by_question_type": dict(Counter(str(row.get("question_type") or "") for row in rows)),
    }


def bootstrap_ci(values: list[float], n: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1 or n <= 0:
        return values[0], values[0]
    rng = random.Random(20260508)
    samples = []
    for _ in range(n):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(mean(draw))
    samples.sort()
    low_idx = int(0.025 * (len(samples) - 1))
    high_idx = int(0.975 * (len(samples) - 1))
    return samples[low_idx], samples[high_idx]


def write_report(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    answerable_rows: list[dict[str, Any]],
    abstain_rows: list[dict[str, Any]],
    main_row: dict[str, Any],
    by_qtype_rows: list[dict[str, Any]],
    experiment_name: str,
    split: str,
    bootstrap_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    caveat = caveat_text(len(rows), bootstrap_n)
    content = [
        f"# Answer Evaluation Build Report",
        "",
        "## 1. 数据规模与拆分",
        f"- experiment: `{experiment_name}`",
        f"- split: `{split}`",
        f"- evaluated samples: `{len(rows)}`",
        f"- answerable samples used for answer-quality means: `{len(answerable_rows)}`",
        f"- should_abstain samples reported separately: `{len(abstain_rows)}`",
        f"- question_type distribution: `{dict(Counter(str(row.get('question_type') or '') for row in rows))}`",
        "",
        "## 2. 总体指标表",
        markdown_table([main_row], MAIN_FIELDS),
        "",
        "## 3. 按 question_type 分组",
        markdown_table(by_qtype_rows, QTYPE_FIELDS),
        summarize_qtype_gap(by_qtype_rows),
        "",
        "## 4. 置信区间说明",
        f"Mean metrics use bootstrap confidence intervals with N={bootstrap_n}. The bootstrap is computed over answerable samples only for answer-quality metrics so that should_abstain=true samples do not dilute answerable performance.",
        "",
        "## 5. 抽检结果",
        "Five build samples were exported to `aiops-docs/experiment/results/answer/expanded/human_review_samples_build.csv` for manual spot-checking. Cohen's kappa is intentionally left for the manual review step after human labels are filled.",
        "",
        "## 6. 诚实性 Caveat",
        caveat,
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def caveat_text(sample_count: int, bootstrap_n: int) -> str:
    return (
        "The LLM judge is an offline proxy evaluator rather than ground truth. Results depend on the "
        "generator model version, judge model version, prompt hashes, API behavior, and temperature=0 setting "
        "used in this run. The should_abstain=true subset is evaluated as a separate abstention subset and is "
        "not mixed into answerable faithfulness, correctness, citation, or hallucination means. Because this "
        f"build run contains only {sample_count} samples and uses bootstrap N={bootstrap_n}, small differences "
        "between metrics should not be described as statistically significant without additional validation."
    )


def summarize_qtype_gap(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No answerable question_type rows were available."
    best = max(rows, key=lambda row: float(row.get("correctness_strict_mean", 0.0) or 0.0))
    worst = min(rows, key=lambda row: float(row.get("correctness_strict_mean", 0.0) or 0.0))
    gap = float(best.get("correctness_strict_mean", 0.0) or 0.0) - float(
        worst.get("correctness_strict_mean", 0.0) or 0.0
    )
    return (
        f"The highest strict correctness group is `{best['question_type']}` and the lowest is "
        f"`{worst['question_type']}`, with a gap of `{gap:.6f}`."
    )


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def path_from_repo(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def to_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def current_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def current_date() -> str:
    return datetime.now(timezone.utc).astimezone().date().isoformat()


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def first_nonempty(values: Any) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def infer_experiment_name(path: Path) -> str:
    name = path.name
    return re.sub(r"_judge\.jsonl$", "", name)


def infer_split_from_path(path: Path) -> str:
    for split in ["build", "dev", "test", "reserve"]:
        if f"_{split}_" in path.name or path.name.endswith(f"_{split}.jsonl"):
            return split
    return ""


def infer_by_qtype_json_path(path: Path) -> Path:
    return path.with_name(path.name.replace("answer_eval_summary_", "answer_eval_by_qtype_"))


def infer_by_qtype_csv_path(path: Path) -> Path:
    return path.with_name(path.name.replace("answer_eval_main_", "answer_eval_by_qtype_"))


if __name__ == "__main__":
    sys.exit(main())
