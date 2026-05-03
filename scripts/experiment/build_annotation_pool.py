"""从实验 chunk 集合中筛选人工标注候选池。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

EXCLUDED_CHUNK_TYPES = {"front_matter", "other"}
PRIORITY_ORDER = {"high": 3, "medium": 2, "low": 1}

RECOMMENDED_QUESTION_TYPES = {
    "troubleshooting_procedure": ["troubleshooting_step", "symptom_cause"],
    "alarm_fault_code": ["parameter_or_fault_code", "symptom_cause"],
    "parameter_and_configuration": ["parameter_or_fault_code"],
    "safety_and_constraint": ["safety_or_constraint"],
    "maintenance_procedure": ["troubleshooting_step"],
    "installation_or_wiring": ["safety_or_constraint", "parameter_or_fault_code"],
    "concept_and_component": ["definition_or_component_lookup"],
}

HIGH_KEYWORDS = ("troubleshooting", "alarm", "fault", "warning", "parameter", "safety")
TABLE_WORDS = ("table", "figure", "menu", "overview", "list")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 experiment annotation pool")
    parser.add_argument(
        "--input",
        default="aiops-docs/experiment/chunks/experiment_chunks.jsonl",
        help="chunk JSONL 输入路径",
    )
    parser.add_argument(
        "--output",
        default="aiops-docs/experiment/chunks/experiment_annotation_pool.jsonl",
        help="annotation pool JSONL 输出路径",
    )
    parser.add_argument(
        "--report",
        default="aiops-docs/experiment/chunks/annotation_pool_report.json",
        help="annotation pool report 输出路径",
    )
    parser.add_argument(
        "--max-candidates-per-source",
        type=int,
        default=None,
        help="每个 source 的默认最大候选数",
    )
    parser.add_argument(
        "--source-quota",
        action="append",
        default=[],
        help="单个 source 的候选上限，格式 source_id=number，可重复",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_annotation_pool(
        input_path=Path(args.input),
        output_path=Path(args.output),
        report_path=Path(args.report),
        max_candidates_per_source=args.max_candidates_per_source,
        source_quotas=parse_source_quotas(args.source_quota),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_source_quotas(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        source_id, raw_limit = value.split("=", 1)
        source_id = source_id.strip()
        limit = int(raw_limit.strip())
        if not source_id or limit <= 0:
            raise ValueError(f"无效 source quota: {value}")
        result[source_id] = limit
    return result


def build_annotation_pool(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    max_candidates_per_source: int | None = None,
    source_quotas: dict[str, int] | None = None,
) -> dict[str, object]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    source_quotas = dict(source_quotas or {})

    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    candidates_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    exclusion_reasons = Counter()
    warnings: list[str] = []

    for row in rows:
        reasons = collect_exclusion_reasons(row)
        if reasons:
            exclusion_reasons.update(reasons)
            continue

        enriched = build_candidate_row(row)
        candidates_by_source[enriched["source_id"]].append(enriched)

    selected: list[dict[str, object]] = []
    source_quota_applied: dict[str, int] = {}

    for source_id in sorted(candidates_by_source):
        quota = source_quotas.get(source_id, max_candidates_per_source)
        ranked = rank_candidates_with_diversity(candidates_by_source[source_id])

        if quota is not None:
            source_quota_applied[source_id] = quota
            if len(ranked) > quota:
                warnings.append(f"source_quota_trimmed:{source_id}:{len(ranked)}->{quota}")
                ranked = ranked[:quota]

        selected.extend(ranked)

    selected.sort(
        key=lambda row: (
            row["source_id"],
            -PRIORITY_ORDER[row["annotation_priority"]],
            -int(row["_ranking_score"]),
            row["page_start"],
            row["chunk_id"],
        )
    )

    write_outputs(output_path, report_path, rows, selected, exclusion_reasons, source_quota_applied, warnings)
    return json.loads(report_path.read_text(encoding="utf-8"))


def collect_exclusion_reasons(row: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    chunk_type = str(row.get("chunk_type") or "")
    char_count = int(row.get("char_count") or 0)
    page_start = row.get("page_start")
    page_end = row.get("page_end")

    if chunk_type in EXCLUDED_CHUNK_TYPES:
        reasons.append(f"chunk_type:{chunk_type}")
    if char_count < 200:
        reasons.append("char_count_lt_200")
    if not bool(row.get("is_annotation_candidate", False)):
        reasons.append("is_annotation_candidate_false")
    if not is_valid_page_range(page_start, page_end):
        reasons.append("invalid_page_range")

    return reasons


def is_valid_page_range(page_start: object, page_end: object) -> bool:
    if not isinstance(page_start, int) or not isinstance(page_end, int):
        return False
    return page_start >= 1 and page_end >= page_start


def build_candidate_row(row: dict[str, object]) -> dict[str, object]:
    text = str(row["text"])
    chunk_type = str(row["chunk_type"])
    annotation_priority = determine_annotation_priority(row)
    recommended_question_types = RECOMMENDED_QUESTION_TYPES[chunk_type]
    quality_flags = derive_quality_flags(row)
    keyword_score = compute_keyword_score(text)

    enriched = dict(row)
    enriched["annotation_priority"] = annotation_priority
    enriched["recommended_question_types"] = recommended_question_types
    enriched["quality_flags"] = quality_flags
    enriched["_keyword_score"] = keyword_score
    enriched["_ranking_score"] = compute_ranking_score(enriched)
    enriched["_page_bucket"] = compute_page_bucket(enriched)
    return enriched


def determine_annotation_priority(row: dict[str, object]) -> str:
    chunk_type = str(row["chunk_type"])
    text = str(row["text"]).lower()
    keyword_score = compute_keyword_score(text)
    safety_level = str(row.get("safety_level") or "none")

    if chunk_type in {"troubleshooting_procedure", "alarm_fault_code"}:
        return "high"
    if chunk_type in {"parameter_and_configuration", "safety_and_constraint", "maintenance_procedure"}:
        if keyword_score >= 3 or safety_level in {"danger", "warning", "caution"}:
            return "high"
        return "medium"
    if chunk_type in {"installation_or_wiring", "concept_and_component"}:
        if keyword_score >= 2:
            return "medium"
        return "low"
    return "low"


def derive_quality_flags(row: dict[str, object]) -> list[str]:
    flags: list[str] = []
    char_count = int(row["char_count"])
    text = str(row["text"])
    page_start = row.get("page_start")
    page_end = row.get("page_end")

    if char_count < 260:
        flags.append("too_short")
    if char_count > 1200:
        flags.append("too_long")
    if not is_valid_page_range(page_start, page_end):
        flags.append("missing_page")
    if looks_like_table_broken(text):
        flags.append("likely_table_broken")
    if is_low_information(text, row):
        flags.append("low_information")

    return flags


def looks_like_table_broken(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False

    short_lines = sum(1 for line in lines if len(line) <= 28)
    header_like = sum(1 for line in lines[:5] if any(word in line.lower() for word in TABLE_WORDS))
    if short_lines / len(lines) >= 0.55:
        return True
    if header_like >= 2 and short_lines >= 3:
        return True
    return False


def is_low_information(text: str, row: dict[str, object]) -> bool:
    lowered = text.lower()
    keyword_score = compute_keyword_score(lowered)
    if int(row["char_count"]) < 320 and keyword_score == 0:
        return True
    if len(set(re.findall(r"[A-Za-z]{4,}", lowered))) < 18 and int(row["char_count"]) < 360:
        return True
    return False


def compute_keyword_score(text: str) -> int:
    lowered = text.lower()
    score = 0
    for keyword in HIGH_KEYWORDS:
        score += len(re.findall(r"\b" + re.escape(keyword) + r"\b", lowered))
    return score


def compute_ranking_score(row: dict[str, object]) -> int:
    score = PRIORITY_ORDER[row["annotation_priority"]] * 1000
    score += int(row["_keyword_score"]) * 25
    score += min(int(row["char_count"]), 950) // 10
    score += 30 if row.get("fault_code") else 0
    score += 20 if row.get("parameter_name") else 0
    score -= 25 * len(row["quality_flags"])
    return score


def compute_page_bucket(row: dict[str, object]) -> int:
    return (int(row["page_start"]) - 1) // 10


def rank_candidates_with_diversity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = sorted(
        rows,
        key=lambda row: (
            -PRIORITY_ORDER[row["annotation_priority"]],
            -int(row["_ranking_score"]),
            row["page_start"],
            row["chunk_id"],
        ),
    )
    buckets: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[int(row["_page_bucket"])].append(row)

    ordered_bucket_ids = sorted(
        buckets,
        key=lambda bucket_id: (
            -PRIORITY_ORDER[buckets[bucket_id][0]["annotation_priority"]],
            -int(buckets[bucket_id][0]["_ranking_score"]),
            bucket_id,
        ),
    )

    ranked: list[dict[str, object]] = []
    while ordered_bucket_ids:
        next_round: list[int] = []
        for bucket_id in ordered_bucket_ids:
            bucket = buckets[bucket_id]
            if not bucket:
                continue
            ranked.append(bucket.pop(0))
            if bucket:
                next_round.append(bucket_id)
        ordered_bucket_ids = next_round

    return ranked


def write_outputs(
    output_path: Path,
    report_path: Path,
    all_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    exclusion_reasons: Counter[str],
    source_quota_applied: dict[str, int],
    warnings: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_rows = [strip_internal_fields(row) for row in selected_rows]
    output_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows)
    output_path.write_text(output_text + ("\n" if output_rows else ""), encoding="utf-8")

    report = build_report(
        all_rows=all_rows,
        selected_rows=output_rows,
        exclusion_reasons=exclusion_reasons,
        source_quota_applied=source_quota_applied,
        warnings=warnings,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_internal_fields(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def build_report(
    all_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    exclusion_reasons: Counter[str],
    source_quota_applied: dict[str, int],
    warnings: list[str],
) -> dict[str, object]:
    count_by_source = Counter(row["source_id"] for row in selected_rows)
    count_by_chunk_type = Counter(row["chunk_type"] for row in selected_rows)
    count_by_priority = Counter(row["annotation_priority"] for row in selected_rows)
    question_type_distribution = Counter(
        question_type
        for row in selected_rows
        for question_type in row["recommended_question_types"]
    )

    report_warnings = list(warnings)
    if not selected_rows:
        report_warnings.append("no_candidates_selected")

    return {
        "total_chunks": len(all_rows),
        "candidate_chunks": len(selected_rows),
        "excluded_chunks": len(all_rows) - len(selected_rows),
        "count_by_source": dict(sorted(count_by_source.items())),
        "count_by_chunk_type": dict(sorted(count_by_chunk_type.items())),
        "count_by_annotation_priority": {
            key: count_by_priority.get(key, 0)
            for key in ("high", "medium", "low")
        },
        "recommended_question_type_distribution": dict(sorted(question_type_distribution.items())),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "source_quota_applied": dict(sorted(source_quota_applied.items())),
        "warnings": report_warnings,
    }


if __name__ == "__main__":
    raise SystemExit(main())
