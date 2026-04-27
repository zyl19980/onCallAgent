"""Evaluate RAG retrieval recall on a labeled JSONL set.

This script measures whether the current retrieval pipeline returns expected
evidence under different runtime recall parameters. It does not rebuild the
index, so chunk-size and overlap experiments must be run after re-indexing into
separate collections/corpora.

JSONL example:
{"id":"q1","query":"E01 告警怎么处理","expected_chunk_ids":["/tmp/manual.pdf::3"]}
{"id":"q2","query":"电源模块更换步骤","expected_file_names":["repair.pdf"],"expected_page_numbers":[5,6]}
{"id":"q3","query":"通信故障原因","expected_text":["CAN 总线", "终端电阻"]}

Supported expected fields:
- expected_chunk_ids: exact RetrievalCandidate.id matches.
- expected_file_names: matches metadata["_file_name"].
- expected_sources: matches metadata["_source"].
- expected_page_numbers: matches metadata["page_number"].
- expected_section_contains: substring match against metadata["section_path"].
- expected_text: substring match against candidate.content.

When multiple expected fields are provided, a candidate is counted as a hit if
it matches any exact/text-only criterion, or if it matches all provided metadata
criteria in the file/source/page/section group.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config  # noqa: E402
from app.services.hybrid_retrieval_service import (  # noqa: E402
    HybridRetrievalService,
    RetrievalCandidate,
)


@dataclass
class EvalCase:
    case_id: str
    query: str
    expected_chunk_ids: set[str]
    expected_file_names: set[str]
    expected_sources: set[str]
    expected_page_numbers: set[int]
    expected_section_contains: list[str]
    expected_text: list[str]


def _as_str_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value if item is not None}


def _as_int_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    result: set[int] = set()
    for item in value:
        try:
            result.add(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if str(item).strip()]


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = str(item.get("query") or item.get("question") or "").strip()
            if not query:
                raise ValueError(f"{path}:{line_no} missing query/question")

            cases.append(
                EvalCase(
                    case_id=str(item.get("id") or f"case-{line_no}"),
                    query=query,
                    expected_chunk_ids=_as_str_set(item.get("expected_chunk_ids")),
                    expected_file_names=_as_str_set(item.get("expected_file_names")),
                    expected_sources=_as_str_set(item.get("expected_sources")),
                    expected_page_numbers=_as_int_set(item.get("expected_page_numbers")),
                    expected_section_contains=_as_str_list(item.get("expected_section_contains")),
                    expected_text=_as_str_list(item.get("expected_text")),
                )
            )
    return cases


def candidate_matches(case: EvalCase, candidate: RetrievalCandidate) -> bool:
    metadata = candidate.metadata or {}

    if candidate.id in case.expected_chunk_ids:
        return True

    if any(text and text in candidate.content for text in case.expected_text):
        return True

    metadata_checks = []
    if case.expected_file_names:
        metadata_checks.append(str(metadata.get("_file_name")) in case.expected_file_names)
    if case.expected_sources:
        metadata_checks.append(str(metadata.get("_source")) in case.expected_sources)
    if case.expected_page_numbers:
        try:
            page_number = int(metadata.get("page_number"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            page_number = -1
        metadata_checks.append(page_number in case.expected_page_numbers)
    if case.expected_section_contains:
        section = str(metadata.get("section_path") or "")
        metadata_checks.append(any(fragment in section for fragment in case.expected_section_contains))

    return bool(metadata_checks) and all(metadata_checks)


def first_hit_rank(case: EvalCase, candidates: list[RetrievalCandidate]) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate_matches(case, candidate):
            return rank
    return None


def evaluate(
    cases: list[EvalCase],
    *,
    candidate_top_k: int,
    final_top_k: int,
    verbose: bool,
) -> dict[str, Any]:
    config.rag_candidate_top_k = candidate_top_k
    config.rag_final_top_k = final_top_k

    service = HybridRetrievalService()
    hit_ranks: list[int | None] = []
    rows: list[dict[str, Any]] = []

    for case in cases:
        retrieval = service.retrieve(case.query)
        candidates = retrieval.candidates
        rank = first_hit_rank(case, candidates)
        hit_ranks.append(rank)
        rows.append(
            {
                "id": case.case_id,
                "query": case.query,
                "hit": rank is not None,
                "rank": rank,
                "top_ids": [candidate.id for candidate in candidates[:final_top_k]],
                "top_sources": [
                    {
                        "file": candidate.metadata.get("_file_name"),
                        "page": candidate.metadata.get("page_number"),
                        "score": round(candidate.rerank_score, 4),
                    }
                    for candidate in candidates[:final_top_k]
                ],
            }
        )

    hits = [rank is not None for rank in hit_ranks]
    reciprocal_ranks = [0.0 if rank is None else 1.0 / rank for rank in hit_ranks]
    metrics = {
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "case_count": len(cases),
        "hit_count": sum(hits),
        "recall_at_final_k": sum(hits) / max(len(cases), 1),
        "mrr": mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
    }
    if verbose:
        metrics["cases"] = rows
    return metrics


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate current RAG retrieval recall.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL evaluation set.")
    parser.add_argument("--candidate-top-k", default="10,20,40", help="Comma-separated values.")
    parser.add_argument("--final-top-k", default="3,5,10", help="Comma-separated values.")
    parser.add_argument("--verbose", action="store_true", help="Include per-case hit details.")
    args = parser.parse_args()

    cases = load_cases(Path(args.dataset))
    if not cases:
        raise ValueError("Evaluation dataset is empty.")

    results: list[dict[str, Any]] = []
    for candidate_top_k in parse_int_list(args.candidate_top_k):
        for final_top_k in parse_int_list(args.final_top_k):
            if final_top_k > candidate_top_k:
                continue
            results.append(
                evaluate(
                    cases,
                    candidate_top_k=candidate_top_k,
                    final_top_k=final_top_k,
                    verbose=args.verbose,
                )
            )

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
