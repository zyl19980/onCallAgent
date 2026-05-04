"""合并多批 reviewed candidates，生成扩展版 reviewed candidates 文件。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


KEEP_REVIEW_STATUSES = {"approved", "revised"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 reviewed candidates")
    parser.add_argument(
        "--batch1",
        default="aiops-docs/experiment/rag/rag_candidate_questions.reviewed.jsonl",
        help="第一批 reviewed candidates JSONL",
    )
    parser.add_argument(
        "--batch2",
        default="aiops-docs/experiment/rag/rag_candidate_questions_batch2.reviewed.jsonl",
        help="第二批 reviewed candidates JSONL",
    )
    parser.add_argument(
        "--output",
        default="aiops-docs/experiment/rag/rag_candidate_questions.merged.reviewed.jsonl",
        help="合并后的 reviewed candidates JSONL",
    )
    parser.add_argument(
        "--report",
        default="aiops-docs/experiment/rag/merged_candidate_review_report.json",
        help="合并报告 JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = merge_reviewed_candidates(
        batch1_path=Path(args.batch1),
        batch2_path=Path(args.batch2),
        output_path=Path(args.output),
        report_path=Path(args.report),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def merge_reviewed_candidates(
    batch1_path: Path,
    batch2_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    batch1_rows = load_jsonl(batch1_path.resolve())
    batch2_rows = load_jsonl(batch2_path.resolve())

    merged_rows: list[dict[str, object]] = []
    duplicate_candidate_ids: list[str] = []
    seen_candidate_ids: set[str] = set()
    skipped_rejected = 0
    skipped_pending = 0

    for row in [*batch1_rows, *batch2_rows]:
        candidate_id = str(row.get("candidate_id") or "").strip()
        review_status = str(row.get("review_status") or "").strip()

        if not candidate_id:
            continue

        if candidate_id in seen_candidate_ids:
            duplicate_candidate_ids.append(candidate_id)
            continue
        seen_candidate_ids.add(candidate_id)

        if review_status == "rejected":
            skipped_rejected += 1
            continue
        if review_status == "pending_review":
            skipped_pending += 1
            continue
        if review_status not in KEEP_REVIEW_STATUSES:
            skipped_pending += 1
            continue

        merged_rows.append(dict(row))

    output_path = output_path.resolve()
    report_path = report_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_path, merged_rows)

    count_by_source = count_by_source_id(merged_rows)
    count_by_question_type = count_by_question_type_value(merged_rows)
    count_by_should_abstain = count_by_should_abstain_value(merged_rows)
    cross_doc_multi_count = count_by_question_type.get("cross_doc_multi", 0)
    abstention_count = count_by_question_type.get("abstention_insufficient_evidence", 0)

    report = {
        "batch1_total": len(batch1_rows),
        "batch2_total": len(batch2_rows),
        "total_input_candidates": len(batch1_rows) + len(batch2_rows),
        "merged_reviewed_candidates": len(merged_rows),
        "skipped_rejected": skipped_rejected,
        "skipped_pending": skipped_pending,
        "duplicate_candidate_ids": duplicate_candidate_ids,
        "count_by_source": count_by_source,
        "count_by_question_type": count_by_question_type,
        "count_by_should_abstain": count_by_should_abstain,
        "cross_doc_multi_count": cross_doc_multi_count,
        "abstention_count": abstention_count,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def count_by_source_id(rows: list[dict[str, object]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        source_id = str(row.get("source_id") or "").strip()
        if source_id:
            counter[source_id] += 1
    return dict(sorted(counter.items()))


def count_by_question_type_value(rows: list[dict[str, object]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        question_type = resolve_question_type(row)
        if question_type:
            counter[question_type] += 1
    return dict(sorted(counter.items()))


def count_by_should_abstain_value(rows: list[dict[str, object]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        key = "true" if bool(row.get("should_abstain")) else "false"
        counter[key] += 1
    return dict(counter)


def resolve_question_type(row: dict[str, object]) -> str:
    final_question_type = str(row.get("final_question_type") or "").strip()
    if final_question_type:
        return final_question_type
    return str(row.get("suggested_question_type") or "").strip()


if __name__ == "__main__":
    raise SystemExit(main())
