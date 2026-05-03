"""导入人工审核结果并合并回候选题集合。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


NORMALIZED_STATUS_MAP = {
    "approve": "approved",
    "approved": "approved",
    "keep": "approved",
    "revise": "revised",
    "revised": "revised",
    "reject": "rejected",
    "rejected": "rejected",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入候选题人工审核结果")
    parser.add_argument(
        "--review-sheet",
        default="aiops-docs/experiment/rag/candidate_review_sheet_reviewed.csv",
        help="人工审核表，支持 .csv 和 .xlsx",
    )
    parser.add_argument(
        "--candidates",
        default="aiops-docs/experiment/rag/rag_candidate_questions.jsonl",
        help="原始候选题 JSONL",
    )
    parser.add_argument(
        "--manual-review-output",
        default="aiops-docs/experiment/rag/manual_candidate_review.jsonl",
        help="导出的人工审核结果 JSONL",
    )
    parser.add_argument(
        "--reviewed-output",
        default="aiops-docs/experiment/rag/rag_candidate_questions.reviewed.jsonl",
        help="合并审核结果后的候选题 JSONL",
    )
    parser.add_argument(
        "--report",
        default="aiops-docs/experiment/rag/candidate_review_import_report.json",
        help="导入报告 JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = import_candidate_review(
        review_sheet_path=Path(args.review_sheet),
        candidates_path=Path(args.candidates),
        manual_review_output_path=Path(args.manual_review_output),
        reviewed_output_path=Path(args.reviewed_output),
        report_path=Path(args.report),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def import_candidate_review(
    review_sheet_path: Path,
    candidates_path: Path,
    manual_review_output_path: Path,
    reviewed_output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    review_sheet_path = review_sheet_path.resolve()
    candidates_path = candidates_path.resolve()
    manual_review_output_path = manual_review_output_path.resolve()
    reviewed_output_path = reviewed_output_path.resolve()
    report_path = report_path.resolve()

    candidate_rows = load_jsonl(candidates_path)
    candidate_by_id = {row["candidate_id"]: row for row in candidate_rows}
    review_rows = load_review_sheet(review_sheet_path)

    valid_reviews: list[dict[str, str]] = []
    invalid_review_rows: list[dict[str, object]] = []
    duplicate_candidate_ids: list[str] = []
    unknown_candidate_ids: list[str] = []
    seen_candidate_ids: set[str] = set()

    for row_number, row in enumerate(review_rows, start=2):
        candidate_id = normalize_cell(row.get("candidate_id"))
        review_status_raw = normalize_cell(row.get("review_status"))

        if not candidate_id or not review_status_raw:
            invalid_review_rows.append(
                {
                    "row_number": row_number,
                    "candidate_id": candidate_id,
                    "reason": "missing_candidate_id_or_review_status",
                }
            )
            continue

        normalized_status = normalize_review_status(review_status_raw)
        if not normalized_status:
            invalid_review_rows.append(
                {
                    "row_number": row_number,
                    "candidate_id": candidate_id,
                    "reason": f"unknown_review_status:{review_status_raw}",
                }
            )
            continue

        if candidate_id in seen_candidate_ids:
            duplicate_candidate_ids.append(candidate_id)
            continue
        seen_candidate_ids.add(candidate_id)

        if candidate_id not in candidate_by_id:
            unknown_candidate_ids.append(candidate_id)
            continue

        review_record = {
            "candidate_id": candidate_id,
            "review_status": normalized_status,
            "final_question": normalize_cell(row.get("final_question")),
            "final_answer": normalize_cell(row.get("final_answer")),
            "final_question_type": normalize_cell(row.get("final_question_type")),
            "reviewer_notes": normalize_cell(row.get("reviewer_notes")),
        }

        invalid_reason = validate_review_record(review_record)
        if invalid_reason:
            invalid_review_rows.append(
                {
                    "row_number": row_number,
                    "candidate_id": candidate_id,
                    "reason": invalid_reason,
                }
            )
            continue

        valid_reviews.append(review_record)

    manual_review_output_path.parent.mkdir(parents=True, exist_ok=True)
    reviewed_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(manual_review_output_path, valid_reviews)
    reviewed_rows = merge_reviews(candidate_rows, valid_reviews)
    write_jsonl(reviewed_output_path, reviewed_rows)

    status_counter = count_statuses(valid_reviews)
    pending_after_import = sum(
        1 for row in reviewed_rows if row.get("review_status") == "pending_review"
    )

    report = {
        "total_candidates": len(candidate_rows),
        "review_rows": len(review_rows),
        "imported_reviews": len(valid_reviews),
        "approved": status_counter["approved"],
        "revised": status_counter["revised"],
        "rejected": status_counter["rejected"],
        "pending_after_import": pending_after_import,
        "invalid_review_rows": invalid_review_rows,
        "duplicate_candidate_ids": duplicate_candidate_ids,
        "unknown_candidate_ids": unknown_candidate_ids,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_review_sheet(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    if suffix == ".xlsx":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("读取 .xlsx 需要 pandas") from exc
        frame = pd.read_excel(path)
        return frame.fillna("").to_dict(orient="records")
    raise ValueError(f"不支持的审核表格式: {path.suffix}")


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_review_status(value: str) -> str:
    return NORMALIZED_STATUS_MAP.get(value.strip().lower(), "")


def validate_review_record(record: dict[str, str]) -> str:
    if record["review_status"] == "revised":
        if not record["final_question"]:
            return "revised_missing_final_question"
        if not record["final_answer"]:
            return "revised_missing_final_answer"
        if not record["final_question_type"]:
            return "revised_missing_final_question_type"
    return ""


def merge_reviews(
    candidate_rows: list[dict[str, object]],
    valid_reviews: list[dict[str, str]],
) -> list[dict[str, object]]:
    review_by_id = {row["candidate_id"]: row for row in valid_reviews}
    merged: list[dict[str, object]] = []

    for row in candidate_rows:
        candidate_id = row["candidate_id"]
        review = review_by_id.get(candidate_id)
        updated = dict(row)
        if review:
            updated["review_status"] = review["review_status"]
            updated["final_question"] = review["final_question"]
            updated["final_answer"] = review["final_answer"]
            updated["final_question_type"] = review["final_question_type"]
            updated["reviewer_notes"] = review["reviewer_notes"]
        merged.append(updated)

    return merged


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


def count_statuses(rows: list[dict[str, str]]) -> dict[str, int]:
    counter = {"approved": 0, "revised": 0, "rejected": 0}
    for row in rows:
        status = row["review_status"]
        if status in counter:
            counter[status] += 1
    return counter


if __name__ == "__main__":
    raise SystemExit(main())
