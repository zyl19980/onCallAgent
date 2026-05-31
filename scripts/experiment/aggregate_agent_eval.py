"""Aggregate fixed replay agent evaluation judge results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODE_ORDER = {"A0": 0, "A1": 1, "A2": 2, "A3": 3}
METRIC_FIELDS = [
    "mode",
    "n_cases",
    "root_cause_accuracy_correct",
    "root_cause_accuracy_partial",
    "root_cause_accuracy_incorrect",
    "evidence_completeness_mean",
    "recommendation_correct",
    "recommendation_partial",
    "recommendation_incorrect",
    "tool_precision_mean",
    "tool_recall_mean",
    "tool_call_count_mean",
    "executed_steps_mean",
    "replan_count_mean",
    "latency_ms_mean",
]
FAULT_FIELDS = ["fault_type", *METRIC_FIELDS]
RAG_FIELDS = ["rag_relevant", *METRIC_FIELDS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate A0-A3 fixed replay agent judge results into thesis tables."
    )
    parser.add_argument("--judge-files", nargs="+", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--by-fault-type", action="store_true")
    parser.add_argument("--by-rag-relevant", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    judge_paths = [path_from_repo(item) for item in flatten_path_args(args.judge_files)]
    cases_path = path_from_repo(args.cases)
    output_csv = path_from_repo(args.output_csv)
    output_report = path_from_repo(args.output_report)

    cases = {str(row.get("case_id")): row for row in read_jsonl(cases_path)}
    if not cases:
        raise ValueError(f"no cases loaded from {cases_path}")

    rows = load_merged_rows(judge_paths=judge_paths, cases=cases)
    if not rows:
        raise ValueError("no judge rows loaded")

    main_rows = build_group_rows(rows, group_key="mode", fields=METRIC_FIELDS)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, main_rows, METRIC_FIELDS)

    by_fault_rows: list[dict[str, Any]] = []
    by_fault_csv: Path | None = None
    if args.by_fault_type:
        by_fault_csv = infer_sidecar_path(output_csv, "by_fault_type")
        by_fault_rows = build_group_rows(rows, group_key=("fault_type", "mode"), fields=FAULT_FIELDS)
        write_csv(by_fault_csv, by_fault_rows, FAULT_FIELDS)

    by_rag_rows: list[dict[str, Any]] = []
    by_rag_csv: Path | None = None
    if args.by_rag_relevant:
        by_rag_csv = infer_sidecar_path(output_csv, "by_rag_relevant")
        rag_rows = [row for row in rows if row.get("mode") in {"A2", "A3"}]
        by_rag_rows = build_group_rows(rag_rows, group_key=("rag_relevant", "mode"), fields=RAG_FIELDS)
        write_csv(by_rag_csv, by_rag_rows, RAG_FIELDS)

    write_report(
        path=output_report,
        judge_paths=judge_paths,
        main_rows=main_rows,
        by_fault_rows=by_fault_rows,
        by_fault_csv=by_fault_csv,
        by_rag_rows=by_rag_rows,
        by_rag_csv=by_rag_csv,
    )
    print(
        json.dumps(
            {
                "main_csv": to_repo_relative(output_csv),
                "report": to_repo_relative(output_report),
                "by_fault_type_csv": to_repo_relative(by_fault_csv) if by_fault_csv else None,
                "by_rag_relevant_csv": to_repo_relative(by_rag_csv) if by_rag_csv else None,
                "n_rows": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def load_merged_rows(
    *, judge_paths: list[Path], cases: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    judge_by_key: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    result_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for judge_path in judge_paths:
        for row in read_jsonl(judge_path):
            key = (str(row.get("mode") or ""), str(row.get("case_id") or ""))
            judge_by_key[key] = row

        result_path = infer_result_path(judge_path)
        if not result_path.exists():
            raise FileNotFoundError(
                f"cannot infer matching result file for {judge_path}: {result_path}"
            )
        for row in read_jsonl(result_path):
            key = (str(row.get("mode") or ""), str(row.get("case_id") or ""))
            result_by_key[key] = row

    merged = []
    for key, judge_row in judge_by_key.items():
        mode, case_id = key
        case = cases.get(case_id)
        if case is None:
            raise KeyError(f"judge row references unknown case_id: {case_id}")
        result = result_by_key.get(key)
        if result is None:
            raise KeyError(f"missing result row for {mode}::{case_id}")
        merged.append(
            {
                "case_id": case_id,
                "mode": mode,
                "fault_type": str(case.get("fault_type") or ""),
                "rag_relevant": bool(case.get("rag_relevant")),
                "root_cause_accuracy": str(judge_row.get("root_cause_accuracy") or ""),
                "evidence_completeness": as_float(judge_row.get("evidence_completeness")),
                "recommendation_actionability": str(
                    judge_row.get("recommendation_actionability") or ""
                ),
                "tool_precision": as_float(judge_row.get("tool_precision")),
                "tool_recall": as_float(judge_row.get("tool_recall")),
                "tool_call_count": as_float(result.get("tool_call_count")),
                "executed_steps": as_float(result.get("executed_steps")),
                "replan_count": as_float(result.get("replan_count")),
                "latency_ms": as_float(result.get("latency_ms")),
            }
        )
    return sorted(merged, key=lambda row: (MODE_ORDER.get(row["mode"], 99), row["case_id"]))


def build_group_rows(
    rows: list[dict[str, Any]], *, group_key: str | tuple[str, str], fields: list[str]
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(group_key, tuple):
            key = tuple(row[item] for item in group_key)
        else:
            key = row[group_key]
        grouped[key].append(row)

    output = []
    for key, group_rows in sorted(grouped.items(), key=group_sort_key):
        out = build_metric_row(group_rows)
        if isinstance(group_key, tuple):
            for field_name, value in zip(group_key, key, strict=True):
                out[field_name] = value
        else:
            out[group_key] = key
        output.append({field: out.get(field, "") for field in fields})
    return output


def build_metric_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": rows[0].get("mode") if rows else "",
        "n_cases": len(rows),
        "root_cause_accuracy_correct": count_verdict(rows, "root_cause_accuracy", "correct"),
        "root_cause_accuracy_partial": count_verdict(rows, "root_cause_accuracy", "partial"),
        "root_cause_accuracy_incorrect": count_verdict(rows, "root_cause_accuracy", "incorrect"),
        "evidence_completeness_mean": round_mean(rows, "evidence_completeness"),
        "recommendation_correct": count_verdict(rows, "recommendation_actionability", "correct"),
        "recommendation_partial": count_verdict(rows, "recommendation_actionability", "partial"),
        "recommendation_incorrect": count_verdict(rows, "recommendation_actionability", "incorrect"),
        "tool_precision_mean": round_mean(rows, "tool_precision"),
        "tool_recall_mean": round_mean(rows, "tool_recall"),
        "tool_call_count_mean": round_mean(rows, "tool_call_count"),
        "executed_steps_mean": round_mean(rows, "executed_steps"),
        "replan_count_mean": round_mean(rows, "replan_count"),
        "latency_ms_mean": round_mean(rows, "latency_ms"),
    }


def write_report(
    *,
    path: Path,
    judge_paths: list[Path],
    main_rows: list[dict[str, Any]],
    by_fault_rows: list[dict[str, Any]],
    by_fault_csv: Path | None,
    by_rag_rows: list[dict[str, Any]],
    by_rag_csv: Path | None,
) -> None:
    lines = [
        "# Agent Evaluation Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Inputs",
        "",
        *[f"- `{to_repo_relative(path)}`" for path in judge_paths],
        "",
        "## Main Results",
        "",
        markdown_table(main_rows, METRIC_FIELDS),
        "",
    ]
    if by_fault_rows and by_fault_csv:
        lines.extend(
            [
                "## By Fault Type",
                "",
                f"CSV: `{to_repo_relative(by_fault_csv)}`",
                "",
                markdown_table(by_fault_rows, FAULT_FIELDS),
                "",
            ]
        )
    if by_rag_rows and by_rag_csv:
        lines.extend(
            [
                "## By RAG Relevance",
                "",
                f"CSV: `{to_repo_relative(by_rag_csv)}`",
                "",
                markdown_table(by_rag_rows, RAG_FIELDS),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def count_verdict(rows: list[dict[str, Any]], field: str, verdict: str) -> int:
    return sum(1 for row in rows if str(row.get(field) or "").lower() == verdict)


def round_mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [as_float(row.get(field)) for row in rows]
    return round(mean(values), 6) if values else 0.0


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def group_sort_key(item: tuple[Any, list[dict[str, Any]]]) -> tuple[Any, ...]:
    key = item[0]
    if isinstance(key, tuple):
        normalized = []
        for part in key:
            if isinstance(part, bool):
                normalized.append(0 if part else 1)
            elif isinstance(part, str) and part in MODE_ORDER:
                normalized.append(MODE_ORDER[part])
            else:
                normalized.append(part)
        return tuple(normalized)
    if isinstance(key, str) and key in MODE_ORDER:
        return (MODE_ORDER[key],)
    return (key,)


def flatten_path_args(items: list[str]) -> list[str]:
    output = []
    for item in items:
        output.extend(part for part in item.split(",") if part.strip())
    return output


def infer_result_path(judge_path: Path) -> Path:
    name = judge_path.name
    if name.endswith("_judge.jsonl"):
        return judge_path.with_name(name.removesuffix("_judge.jsonl") + "_results.jsonl")
    return judge_path.with_name(name.replace("judge", "results"))


def infer_sidecar_path(output_csv: Path, suffix: str) -> Path:
    stem = output_csv.stem
    if stem.endswith("_main"):
        stem = stem.removesuffix("_main")
    return output_csv.with_name(f"{stem}_{suffix}.csv")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        rows.append(payload)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def path_from_repo(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def to_repo_relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
