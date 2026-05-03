"""导出候选题人工审核表。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

CSV_FIELDS = [
    "candidate_id",
    "source_id",
    "source_file",
    "page_start",
    "page_end",
    "chunk_type",
    "suggested_question_type",
    "generated_question",
    "generated_answer",
    "evidence_quote",
    "warnings",
    "review_status",
    "final_question",
    "final_answer",
    "final_question_type",
    "reviewer_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出候选题人工审核表")
    parser.add_argument(
        "--input",
        default="aiops-docs/experiment/rag/rag_candidate_questions.jsonl",
        help="候选题 JSONL 输入路径",
    )
    parser.add_argument(
        "--csv-output",
        default="aiops-docs/experiment/rag/candidate_review_sheet.csv",
        help="审核表 CSV 输出路径",
    )
    parser.add_argument(
        "--warnings-output",
        default="aiops-docs/experiment/rag/warning_candidates_for_review.jsonl",
        help="warning 候选 JSONL 输出路径",
    )
    parser.add_argument(
        "--warnings-only",
        action="store_true",
        help="仅导出带 warning 的候选到 CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = export_candidate_review_sheet(
        input_path=Path(args.input),
        csv_output_path=Path(args.csv_output),
        warnings_output_path=Path(args.warnings_output),
        warnings_only=args.warnings_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def export_candidate_review_sheet(
    input_path: Path,
    csv_output_path: Path,
    warnings_output_path: Path,
    warnings_only: bool = False,
) -> dict[str, object]:
    input_path = input_path.resolve()
    csv_output_path = csv_output_path.resolve()
    warnings_output_path = warnings_output_path.resolve()

    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    enriched_rows = [enrich_candidate_row(row) for row in rows]
    warning_rows = [row for row in enriched_rows if row["warnings"]]
    csv_rows = warning_rows if warnings_only else enriched_rows

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    warnings_output_path.parent.mkdir(parents=True, exist_ok=True)

    write_csv(csv_output_path, csv_rows)
    write_warning_jsonl(warnings_output_path, warning_rows)

    return {
        "input_candidates": len(enriched_rows),
        "csv_rows": len(csv_rows),
        "warning_rows": len(warning_rows),
        "warnings_only": warnings_only,
        "csv_output": to_repo_relative_path(csv_output_path),
        "warnings_output": to_repo_relative_path(warnings_output_path),
    }


def enrich_candidate_row(row: dict[str, object]) -> dict[str, object]:
    warnings = detect_candidate_warnings(row)
    enriched = dict(row)
    enriched["warnings"] = warnings
    return enriched


def detect_candidate_warnings(row: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    question = str(row.get("generated_question") or "").strip()
    answer = str(row.get("generated_answer") or "").strip()
    evidence = str(row.get("evidence_quote") or "").strip()
    chunk_type = str(row.get("chunk_type") or "")
    question_type = str(row.get("suggested_question_type") or "")

    if question_type == "definition_or_component_lookup" and chunk_type != "concept_and_component":
        warnings.append("definition_fallback_candidate")
    if "this issue" in question.lower():
        warnings.append("vague_subject_in_question")
    if len(answer) < 35:
        warnings.append("short_generated_answer")
    if len(evidence) < 40:
        warnings.append("short_evidence_quote")
    if looks_like_table_broken(evidence):
        warnings.append("likely_table_broken")
    if looks_heading_like(question):
        warnings.append("heading_like_question_subject")
    if answer and normalize_ws(answer) not in normalize_ws(evidence):
        warnings.append("answer_not_exact_substring_of_evidence")

    return unique_preserve_order(warnings)


def looks_like_table_broken(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False

    short_lines = sum(1 for line in lines if len(line) <= 32)
    alarm_lines = sum(1 for line in lines if re.search(r"\b(alarm|fault|parameter)\b", line, re.IGNORECASE))
    if short_lines / len(lines) >= 0.55:
        return True
    if alarm_lines >= 2 and short_lines >= 2:
        return True
    return False


def looks_heading_like(question: str) -> bool:
    normalized = question.lower()
    heading_fragments = (
        "corrective action",
        "description grundfos eye",
        "observed",
        "this issue",
    )
    return any(fragment in normalized for fragment in heading_fragments)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(build_csv_row(row))


def build_csv_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "candidate_id": row.get("candidate_id", ""),
        "source_id": row.get("source_id", ""),
        "source_file": row.get("source_file", ""),
        "page_start": row.get("page_start", ""),
        "page_end": row.get("page_end", ""),
        "chunk_type": row.get("chunk_type", ""),
        "suggested_question_type": row.get("suggested_question_type", ""),
        "generated_question": row.get("generated_question", ""),
        "generated_answer": row.get("generated_answer", ""),
        "evidence_quote": row.get("evidence_quote", ""),
        "warnings": "; ".join(row.get("warnings", [])),
        "review_status": "",
        "final_question": "",
        "final_answer": "",
        "final_question_type": "",
        "reviewer_notes": "",
    }


def write_warning_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def to_repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    repo_root = Path.cwd().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
