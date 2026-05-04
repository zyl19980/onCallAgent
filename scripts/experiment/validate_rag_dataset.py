"""校验正式 RAG 数据集。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

VALID_SPLITS = {"build", "dev", "test", "reserve"}
VALID_QUESTION_TYPES = {
    "troubleshooting_step",
    "symptom_cause",
    "parameter_or_fault_code",
    "safety_or_constraint",
    "definition_or_component_lookup",
    "cross_doc_multi",
    "abstention_insufficient_evidence",
}
VALID_REASONING_HOPS = {
    "single_chunk",
    "multi_chunk_same_doc",
    "multi_doc",
    "abstention",
}
VALID_CRITICALITY = {"low", "medium", "high", "safety_critical"}
VALID_EXPECTED_CONFIDENCE = {"low", "medium", "high"}
VALID_ANNOTATION_STATUS = {"reviewed", "pending_review", "rejected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验正式 RAG 数据集")
    parser.add_argument(
        "--dataset",
        default="aiops-docs/experiment/rag/experiment_rag_dataset.jsonl",
        help="正式 RAG dataset JSONL",
    )
    parser.add_argument(
        "--chunks",
        default="aiops-docs/experiment/chunks/experiment_chunks.jsonl",
        help="experiment chunks JSONL",
    )
    parser.add_argument(
        "--output",
        default="aiops-docs/experiment/rag/experiment_rag_dataset.validated.jsonl",
        help="通过校验后的 JSONL 输出路径",
    )
    parser.add_argument(
        "--report",
        default="aiops-docs/experiment/rag/rag_validation_report.json",
        help="校验报告 JSON 输出路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_rag_dataset(
        dataset_path=Path(args.dataset),
        chunks_path=Path(args.chunks),
        output_path=Path(args.output),
        report_path=Path(args.report),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def validate_rag_dataset(
    dataset_path: Path,
    chunks_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    dataset_path = dataset_path.resolve()
    chunks_path = chunks_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()

    dataset_rows = load_jsonl(dataset_path)
    chunk_rows = load_jsonl(chunks_path)
    chunks_by_id = {row["chunk_id"]: row for row in chunk_rows}

    id_counts = Counter(str(row.get("id") or "") for row in dataset_rows)
    user_input_counts = Counter(normalize_text(str(row.get("user_input") or "")) for row in dataset_rows)

    valid_rows: list[dict[str, object]] = []
    errors_by_type = Counter()
    warnings: list[str] = []
    normalized_reasoning_hops = 0
    short_answers = 0
    duplicate_user_input = 0

    for row in dataset_rows:
        normalized_row = dict(row)
        row_errors: list[str] = []
        sample_id = str(row.get("id") or "")
        should_abstain = bool(row.get("should_abstain", False))

        if not sample_id:
            row_errors.append("missing_id")
        elif id_counts[sample_id] > 1:
            row_errors.append("duplicate_id")

        split = str(row.get("split") or "")
        if split not in VALID_SPLITS:
            row_errors.append("invalid_split")

        user_input = str(row.get("user_input") or "").strip()
        if not user_input:
            row_errors.append("missing_user_input")

        reference_answer = str(row.get("reference_answer") or "").strip()
        if not reference_answer:
            row_errors.append("missing_reference_answer")

        reference_chunk_ids = row.get("reference_chunk_ids")
        if not isinstance(reference_chunk_ids, list):
            row_errors.append("reference_chunk_ids_not_list")
            reference_chunk_ids = []
        if not should_abstain and not reference_chunk_ids:
            row_errors.append("missing_reference_chunk_ids")

        resolved_chunks = []
        for chunk_id in reference_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                row_errors.append("missing_reference_chunk")
            else:
                resolved_chunks.append(chunk)

        reference_evidence = row.get("reference_evidence")
        if not isinstance(reference_evidence, list):
            row_errors.append("reference_evidence_not_list")
            reference_evidence = []
        if not should_abstain and not reference_evidence:
            row_errors.append("missing_reference_evidence")

        for evidence in reference_evidence:
            if not isinstance(evidence, dict):
                row_errors.append("invalid_reference_evidence_item")
                continue
            chunk_id = evidence.get("chunk_id")
            page_start = evidence.get("page_start")
            page_end = evidence.get("page_end")
            quote = str(evidence.get("quote") or "").strip()
            if not chunk_id:
                row_errors.append("missing_evidence_chunk_id")
            elif chunk_id not in reference_chunk_ids:
                row_errors.append("evidence_chunk_not_in_reference_chunk_ids")
            if page_start is None or page_end is None:
                row_errors.append("missing_evidence_pages")
            if not quote:
                row_errors.append("missing_evidence_quote")

        expected_source_files = row.get("expected_source_files")
        if not should_abstain and (not isinstance(expected_source_files, list) or not expected_source_files):
            row_errors.append("missing_expected_source_files")

        expected_page_numbers = row.get("expected_page_numbers")
        if not should_abstain and (not isinstance(expected_page_numbers, list) or not expected_page_numbers):
            row_errors.append("missing_expected_page_numbers")
        elif not should_abstain:
            expected_page_set = {int(page) for page in expected_page_numbers}
            chunk_page_set = {
                page
                for chunk in resolved_chunks
                for page in range(int(chunk["page_start"]), int(chunk["page_end"]) + 1)
            }
            if expected_page_set and chunk_page_set and expected_page_set.isdisjoint(chunk_page_set):
                row_errors.append("expected_pages_no_intersection")

        question_type = str(row.get("question_type") or "")
        if question_type not in VALID_QUESTION_TYPES:
            row_errors.append("invalid_question_type")

        raw_reasoning_hops = row.get("reasoning_hops")
        normalized_hops = normalize_reasoning_hops(raw_reasoning_hops, should_abstain)
        if not normalized_hops:
            row_errors.append("invalid_reasoning_hops")
        else:
            if normalized_hops != row.get("reasoning_hops"):
                normalized_reasoning_hops += 1
                normalized_row["reasoning_hops"] = normalized_hops

        criticality = str(row.get("criticality") or "")
        if criticality not in VALID_CRITICALITY:
            row_errors.append("invalid_criticality")

        expected_confidence = str(row.get("expected_confidence") or "")
        if expected_confidence not in VALID_EXPECTED_CONFIDENCE:
            row_errors.append("invalid_expected_confidence")

        annotation_status = str(row.get("annotation_status") or "")
        if annotation_status not in VALID_ANNOTATION_STATUS:
            row_errors.append("invalid_annotation_status")

        if should_abstain:
            if question_type != "abstention_insufficient_evidence":
                row_errors.append("abstain_invalid_question_type")
            raw_hops_text = str(raw_reasoning_hops).strip()
            if raw_hops_text and raw_hops_text != "abstention":
                row_errors.append("abstain_invalid_reasoning_hops")
            if expected_confidence != "low":
                row_errors.append("abstain_invalid_expected_confidence")

        if question_type == "cross_doc_multi":
            if len(reference_chunk_ids) < 2:
                row_errors.append("cross_doc_insufficient_reference_chunks")
            if normalized_hops and normalized_hops != "multi_doc":
                row_errors.append("cross_doc_invalid_reasoning_hops")

        if row_errors:
            errors_by_type.update(row_errors)
            continue

        if not should_abstain and len(reference_answer) < 30:
            short_answers += 1
        normalized_user_input = normalize_text(user_input)
        if normalized_user_input and user_input_counts[normalized_user_input] > 1:
            duplicate_user_input += 1

        valid_rows.append(normalized_row)

    if normalized_reasoning_hops:
        warnings.append(f"normalized_reasoning_hops:{normalized_reasoning_hops}")
    if duplicate_user_input:
        warnings.append(f"duplicate_user_input_candidates:{duplicate_user_input}")
    if short_answers:
        warnings.append(f"short_reference_answers:{short_answers}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, valid_rows)

    report = {
        "total_samples": len(dataset_rows),
        "valid_samples": len(valid_rows),
        "invalid_samples": len(dataset_rows) - len(valid_rows),
        "count_by_source": dict(sorted(Counter(flatten_source_ids(valid_rows)).items())),
        "count_by_question_type": dict(sorted(Counter(row["question_type"] for row in valid_rows).items())),
        "count_by_should_abstain": dict(
            sorted(Counter("true" if row.get("should_abstain") else "false" for row in valid_rows).items())
        ),
        "count_by_split": dict(sorted(Counter(row["split"] for row in valid_rows).items())),
        "errors_by_type": dict(sorted(errors_by_type.items())),
        "warnings": warnings,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def normalize_reasoning_hops(value: object, should_abstain: bool) -> str:
    if should_abstain:
        return "abstention"
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in VALID_REASONING_HOPS:
            return normalized
        if normalized.isdigit():
            value = int(normalized)
        else:
            return ""
    if isinstance(value, int):
        if value <= 1:
            return "single_chunk"
        if value == 2:
            return "multi_chunk_same_doc"
        if value >= 3:
            return "multi_doc"
    return ""


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def flatten_source_ids(rows: list[dict[str, object]]) -> list[str]:
    output: list[str] = []
    for row in rows:
        output.extend(row.get("source_ids", []))
    return output


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
