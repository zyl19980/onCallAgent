"""从 reviewed candidates 构建正式 RAG 数据集。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 reviewed candidates 构建正式 RAG 数据集")
    parser.add_argument(
        "--candidates",
        default="aiops-docs/experiment/rag/rag_candidate_questions.reviewed.jsonl",
        help="reviewed candidates JSONL",
    )
    parser.add_argument(
        "--chunks",
        default="aiops-docs/experiment/chunks/experiment_chunks.jsonl",
        help="experiment chunks JSONL",
    )
    parser.add_argument(
        "--output",
        default="aiops-docs/experiment/rag/experiment_rag_dataset.jsonl",
        help="正式 RAG dataset JSONL 输出路径",
    )
    parser.add_argument(
        "--report",
        default="aiops-docs/experiment/rag/rag_dataset_build_report.json",
        help="构建报告 JSON 输出路径",
    )
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="仅用于 pilot，允许纳入 pending_review 样本",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_rag_dataset_from_candidates(
        candidates_path=Path(args.candidates),
        chunks_path=Path(args.chunks),
        output_path=Path(args.output),
        report_path=Path(args.report),
        include_pending=args.include_pending,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_rag_dataset_from_candidates(
    candidates_path: Path,
    chunks_path: Path,
    output_path: Path,
    report_path: Path,
    include_pending: bool = False,
) -> dict[str, object]:
    candidates_path = candidates_path.resolve()
    chunks_path = chunks_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()

    candidate_rows = load_jsonl(candidates_path)
    chunk_rows = load_jsonl(chunks_path)
    chunks_by_id = {row["chunk_id"]: row for row in chunk_rows}

    converted_rows: list[dict[str, object]] = []
    skipped_rejected = 0
    skipped_pending = 0
    invalid_missing_chunk = 0
    invalid_missing_evidence = 0
    warnings: list[str] = []

    if include_pending:
        warnings.append("include_pending_enabled_for_pilot_only")

    rag_index = 1
    for candidate in candidate_rows:
        review_status = str(candidate.get("review_status") or "").strip()

        if review_status == "rejected":
            skipped_rejected += 1
            continue
        if review_status == "pending_review" and not include_pending:
            skipped_pending += 1
            continue
        if review_status not in {"approved", "revised", "pending_review"}:
            skipped_pending += 1
            continue

        question_type = resolve_question_type(candidate)
        should_abstain = bool(candidate.get("should_abstain"))
        reference_chunk_ids = normalize_chunk_ids(candidate.get("source_chunk_ids"))
        resolved_chunks, missing_chunk = resolve_chunks(reference_chunk_ids, chunks_by_id)

        if should_abstain and question_type == "abstention_insufficient_evidence":
            weak_chunk_ids = normalize_chunk_ids(candidate.get("weak_evidence_chunk_ids"))
            weak_chunks, weak_missing_chunk = resolve_chunks(weak_chunk_ids, chunks_by_id)
            if weak_missing_chunk:
                invalid_missing_chunk += 1
                continue

            dataset_row = build_abstention_dataset_row(
                candidate=candidate,
                weak_chunks=weak_chunks,
                rag_index=rag_index,
            )
            rag_index += 1
            converted_rows.append(dataset_row)
            continue

        if missing_chunk or not resolved_chunks:
            invalid_missing_chunk += 1
            continue

        evidence_quote = str(candidate.get("evidence_quote") or "").strip()
        if not evidence_quote:
            invalid_missing_evidence += 1
            continue

        dataset_row = build_dataset_row(
            candidate=candidate,
            resolved_chunks=resolved_chunks,
            evidence_quote=evidence_quote,
            rag_index=rag_index,
        )
        rag_index += 1
        converted_rows.append(dataset_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, converted_rows)

    report = {
        "total_candidates": len(candidate_rows),
        "converted": len(converted_rows),
        "skipped_rejected": skipped_rejected,
        "skipped_pending": skipped_pending,
        "invalid_missing_chunk": invalid_missing_chunk,
        "invalid_missing_evidence": invalid_missing_evidence,
        "count_by_source": dict(sorted(Counter(flatten_source_ids(converted_rows)).items())),
        "count_by_question_type": dict(
            sorted(Counter(row["question_type"] for row in converted_rows).items())
        ),
        "count_by_should_abstain": dict(
            sorted(Counter("true" if row["should_abstain"] else "false" for row in converted_rows).items())
        ),
        "cross_doc_multi_count": sum(
            1 for row in converted_rows if row["question_type"] == "cross_doc_multi"
        ),
        "abstention_count": sum(
            1
            for row in converted_rows
            if row["question_type"] == "abstention_insufficient_evidence"
        ),
        "warnings": warnings,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_dataset_row(
    candidate: dict[str, object],
    resolved_chunks: list[dict[str, object]],
    evidence_quote: str,
    rag_index: int,
) -> dict[str, object]:
    source_ids = unique_preserve_order(str(chunk["source_id"]) for chunk in resolved_chunks)
    collections = unique_preserve_order(str(chunk["collection"]) for chunk in resolved_chunks)
    expected_source_files = unique_preserve_order(str(chunk["source_file"]) for chunk in resolved_chunks)
    expected_page_numbers = flatten_page_numbers(resolved_chunks)
    reference_chunk_ids = [str(chunk["chunk_id"]) for chunk in resolved_chunks]

    user_input = (
        str(candidate.get("final_question") or "").strip()
        or str(candidate.get("generated_question") or "").strip()
    )
    reference_answer = (
        str(candidate.get("final_answer") or "").strip()
        or str(candidate.get("generated_answer") or "").strip()
    )
    question_type = (
        str(candidate.get("final_question_type") or "").strip()
        or str(candidate.get("suggested_question_type") or "").strip()
    )
    reasoning_hops = resolve_reasoning_hops(candidate, question_type, should_abstain=False)

    return {
        "id": f"rag_{rag_index:03d}",
        "split": "build",
        "source_ids": source_ids,
        "collections": collections,
        "user_input": user_input,
        "reference_answer": reference_answer,
        "reference_chunk_ids": reference_chunk_ids,
        "reference_evidence": [
            {
                "chunk_id": str(chunk["chunk_id"]),
                "page_start": int(chunk["page_start"]),
                "page_end": int(chunk["page_end"]),
                "quote": evidence_quote,
            }
            for chunk in resolved_chunks
        ],
        "expected_source_files": expected_source_files,
        "expected_page_numbers": expected_page_numbers,
        "question_type": question_type,
        "reasoning_hops": reasoning_hops,
        "criticality": str(candidate.get("suggested_criticality") or "medium"),
        "expected_confidence": "high",
        "should_abstain": False,
        "annotation_status": "reviewed",
        "annotator": str(candidate.get("generator") or "template_dry_run"),
        "reviewer": "human",
        "notes": build_notes(candidate),
    }


def build_abstention_dataset_row(
    candidate: dict[str, object],
    weak_chunks: list[dict[str, object]],
    rag_index: int,
) -> dict[str, object]:
    source_ids = unique_preserve_order(
        list(normalize_string_list(candidate.get("source_ids")))
        or [str(candidate.get("source_id") or "").strip()]
        or [str(chunk["source_id"]) for chunk in weak_chunks]
    )
    source_ids = [value for value in source_ids if value]
    collections = unique_preserve_order(str(chunk["collection"]) for chunk in weak_chunks)
    expected_source_files = unique_preserve_order(
        list(normalize_string_list(candidate.get("source_files")))
        or [str(candidate.get("source_file") or "").strip()]
        or [str(chunk["source_file"]) for chunk in weak_chunks]
    )
    expected_source_files = [value for value in expected_source_files if value]
    expected_page_numbers = flatten_page_numbers(weak_chunks)
    user_input = (
        str(candidate.get("final_question") or "").strip()
        or str(candidate.get("generated_question") or "").strip()
    )
    reference_answer = (
        str(candidate.get("final_answer") or "").strip()
        or str(candidate.get("generated_answer") or "").strip()
    )

    return {
        "id": f"rag_{rag_index:03d}",
        "split": "build",
        "source_ids": source_ids,
        "collections": collections,
        "user_input": user_input,
        "reference_answer": reference_answer,
        "reference_chunk_ids": [],
        "reference_evidence": [],
        "expected_source_files": expected_source_files,
        "expected_page_numbers": expected_page_numbers,
        "question_type": "abstention_insufficient_evidence",
        "reasoning_hops": "abstention",
        "criticality": str(candidate.get("suggested_criticality") or "medium"),
        "expected_confidence": "low",
        "should_abstain": True,
        "annotation_status": "reviewed",
        "annotator": str(candidate.get("generator") or "template_dry_run"),
        "reviewer": "human",
        "notes": build_notes(candidate),
    }


def build_notes(candidate: dict[str, object]) -> str:
    parts = [f"candidate_id={candidate['candidate_id']}"]
    abstention_reason = str(candidate.get("abstention_reason") or "").strip()
    if abstention_reason:
        parts.append(f"abstention_reason={abstention_reason}")
    reviewer_notes = str(candidate.get("reviewer_notes") or "").strip()
    if reviewer_notes:
        parts.append(f"reviewer_notes={reviewer_notes}")
    return " | ".join(parts)


def flatten_source_ids(rows: list[dict[str, object]]) -> list[str]:
    output: list[str] = []
    for row in rows:
        output.extend(row["source_ids"])
    return output


def flatten_page_numbers(chunks: list[dict[str, object]]) -> list[int]:
    page_numbers: list[int] = []
    for chunk in chunks:
        start = int(chunk["page_start"])
        end = int(chunk["page_end"])
        page_numbers.extend(range(start, end + 1))
    return unique_preserve_order(page_numbers)


def unique_preserve_order(values):
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def normalize_chunk_ids(value: object) -> list[str]:
    return [str(item).strip() for item in list(value or []) if str(item).strip()]


def normalize_string_list(value: object) -> list[str]:
    return [str(item).strip() for item in list(value or []) if str(item).strip()]


def resolve_chunks(
    chunk_ids: list[str],
    chunks_by_id: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], bool]:
    resolved_chunks: list[dict[str, object]] = []
    for chunk_id in chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            return [], True
        resolved_chunks.append(chunk)
    return resolved_chunks, False


def resolve_question_type(candidate: dict[str, object]) -> str:
    return (
        str(candidate.get("final_question_type") or "").strip()
        or str(candidate.get("suggested_question_type") or "").strip()
    )


def resolve_reasoning_hops(
    candidate: dict[str, object],
    question_type: str,
    should_abstain: bool,
) -> str:
    if should_abstain or question_type == "abstention_insufficient_evidence":
        return "abstention"

    raw_value = str(candidate.get("suggested_reasoning_hops") or "").strip()
    if question_type == "cross_doc_multi":
        return "multi_doc"
    if raw_value in {"single_chunk", "multi_chunk_same_doc", "multi_doc", "abstention"}:
        return raw_value
    if raw_value == "1":
        return "single_chunk"
    if raw_value == "2":
        return "multi_chunk_same_doc"
    if raw_value == "multi_doc":
        return "multi_doc"
    return "single_chunk"


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
