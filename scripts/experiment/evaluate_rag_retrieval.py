"""RAG 检索评测脚本。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


LIVE_DIAGNOSIS_OUTPUT = Path("aiops-docs/experiment/results/rag_live_diagnosis.json")
QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]+")
QUERY_STOPWORDS = {
    "请问", "一下", "这个", "那个", "什么", "是否", "可以", "需要",
    "怎么", "如何", "一下子", "我想", "帮忙", "请", "下",
}


@dataclass
class QueryPlan:
    query_mode: str
    main_query: str
    keyword_query: str
    expanded_query: str
    analysis: dict[str, object] = field(default_factory=dict)

    def active_queries(self) -> list[tuple[str, str]]:
        output = [("main_query", self.main_query)]
        if self.query_mode in {"original_keyword", "original_keyword_expanded"} and self.keyword_query:
            output.append(("keyword_query", self.keyword_query))
        if self.query_mode == "original_keyword_expanded" and self.expanded_query:
            output.append(("expanded_query", self.expanded_query))
        return output


class RetrievalAdapter(Protocol):
    def retrieve(
        self,
        sample: dict[str, object],
        *,
        top_k: int,
        collection_filter_from_sample: bool,
        source_filter_from_sample: bool,
    ) -> list[dict[str, object]]:
        ...


class RerankAdapter(Protocol):
    def rerank(
        self,
        *,
        sample: dict[str, object],
        candidate_results: list[dict[str, object]],
        final_top_k: int,
    ) -> dict[str, object]:
        ...


@dataclass
class RerankOutcome:
    final_results: list[dict[str, object]]
    reranked_results: list[dict[str, object]]
    rerank_available: bool
    rerank_errors: list[str]


class MockRetrievalAdapter:
    def __init__(
        self,
        all_chunks: list[dict[str, object]],
        chunks_by_id: dict[str, dict[str, object]],
        mock_miss_rate: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.all_chunks = all_chunks
        self.chunks_by_id = chunks_by_id
        self.mock_miss_rate = max(0.0, min(1.0, mock_miss_rate))
        self.seed = seed

    def retrieve(
        self,
        sample: dict[str, object],
        *,
        top_k: int,
        collection_filter_from_sample: bool,
        source_filter_from_sample: bool,
    ) -> list[dict[str, object]]:
        rng = random.Random(self.seed + stable_hash_int(str(sample["id"])))
        ref_ids = [str(chunk_id) for chunk_id in sample.get("reference_chunk_ids", [])]
        allowed_chunks = list(self.all_chunks)

        if collection_filter_from_sample:
            collections = set(str(item) for item in sample.get("collections", []))
            allowed_chunks = [chunk for chunk in allowed_chunks if str(chunk.get("collection")) in collections]
        if source_filter_from_sample:
            source_ids = set(str(item) for item in sample.get("source_ids", []))
            allowed_chunks = [chunk for chunk in allowed_chunks if str(chunk.get("source_id")) in source_ids]

        ref_chunks = []
        for chunk_id in ref_ids:
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk and rng.random() >= self.mock_miss_rate:
                ref_chunks.append(chunk)

        distractors = [chunk for chunk in allowed_chunks if str(chunk["chunk_id"]) not in set(ref_ids)]
        rng.shuffle(distractors)

        selected = ref_chunks + distractors
        selected = selected[:top_k]
        retrieved = []
        score = 1.0
        for rank, chunk in enumerate(selected, start=1):
            retrieved.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "source_id": chunk["source_id"],
                    "source_file": chunk["source_file"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "score": round(score, 6),
                }
            )
            score -= 0.01
        return retrieved


class LiveRetrievalAdapter:
    def __init__(
        self,
        chunks_by_id: dict[str, dict[str, object]],
        *,
        collection_name: str | None = None,
        dump_raw_retrieval: bool = False,
    ) -> None:
        self.chunks_by_id = chunks_by_id
        self.collection_name = collection_name or "experiment_manuals_all"
        self.dump_raw_retrieval = dump_raw_retrieval
        try:
            from app.services.vector_search_service import vector_search_service
        except Exception as exc:
            raise RuntimeError(
                "live 模式暂时无法接入，当前需要可用的 app.services.vector_search_service 及其依赖"
            ) from exc
        self.vector_search_service = vector_search_service

    def retrieve(
        self,
        sample: dict[str, object],
        *,
        top_k: int,
        collection_filter_from_sample: bool,
        source_filter_from_sample: bool,
    ) -> list[dict[str, object]]:
        diagnostic = self.retrieve_diagnostic(
            sample,
            top_k=top_k,
            collection_filter_from_sample=collection_filter_from_sample,
            source_filter_from_sample=source_filter_from_sample,
        )
        return diagnostic["normalized_results"]

    def retrieve_diagnostic(
        self,
        sample: dict[str, object],
        *,
        top_k: int,
        collection_filter_from_sample: bool,
        source_filter_from_sample: bool,
    ) -> dict[str, object]:
        query = resolve_sample_query(sample)
        results = self.vector_search_service.search_similar_documents(
            query,
            top_k=top_k,
            collection_name=self.collection_name,
        )

        collections = set(str(item) for item in sample.get("collections", []))
        source_ids = set(str(item) for item in sample.get("source_ids", []))

        normalized: list[dict[str, object]] = []
        missing_chunk_id_count = 0
        errors: list[str] = []
        for item in results:
            metadata = dict(getattr(item, "metadata", {}) or {})
            chunk_id = extract_live_chunk_id(item, metadata)
            chunk = self.chunks_by_id.get(str(chunk_id)) if chunk_id else None

            source_id = chunk.get("source_id") if chunk else metadata.get("source_id")
            source_file = chunk.get("source_file") if chunk else metadata.get("source_file") or metadata.get("_file_name")
            page_start = chunk.get("page_start") if chunk else metadata.get("page_start") or metadata.get("page_number")
            page_end = chunk.get("page_end") if chunk else metadata.get("page_end") or metadata.get("page_number")
            collection = chunk.get("collection") if chunk else metadata.get("collection")

            if collection_filter_from_sample and collections and str(collection) not in collections:
                continue
            if source_filter_from_sample and source_ids and str(source_id) not in source_ids:
                continue

            if not chunk_id:
                missing_chunk_id_count += 1
                errors.append("missing_chunk_id_in_retrieved")

            normalized.append(
                {
                    "chunk_id": str(chunk_id) if chunk_id else "",
                    "source_id": str(source_id) if source_id is not None else "",
                    "source_file": str(source_file) if source_file is not None else "",
                    "page_start": to_optional_int(page_start),
                    "page_end": to_optional_int(page_end),
                    "score": float(getattr(item, "score", 0.0)),
                    "raw": build_live_raw_payload(item, metadata),
                }
            )

        output = []
        for rank, item in enumerate(normalized[:top_k], start=1):
            enriched = dict(item)
            enriched["rank"] = rank
            output.append(enriched)
        return {
            "raw_result_count": len(results),
            "normalized_result_count": len(output),
            "normalized_results": output,
            "missing_chunk_id_count": missing_chunk_id_count,
            "errors": unique_preserve_order(errors),
        }


class HybridRetrievalAdapter:
    def __init__(
        self,
        chunks_by_id: dict[str, dict[str, object]],
        *,
        chunk_rows: list[dict[str, object]],
        collection_name: str | None = None,
    ) -> None:
        self.chunks_by_id = chunks_by_id
        self.collection_name = collection_name or "experiment_manuals_all"
        try:
            from app.services.bm25_search_service import BM25SearchService
            from app.services.hybrid_retrieval_service import RetrievalCandidate, hybrid_retrieval_service
            from app.services.vector_search_service import vector_search_service
        except Exception as exc:
            raise RuntimeError(
                "hybrid 检索 unavailable: 无法导入 app.services.hybrid_retrieval_service"
            ) from exc
        self.RetrievalCandidate = RetrievalCandidate
        self.bm25_service = BM25SearchService(collection_name=self.collection_name)
        self._load_bm25_corpus(chunk_rows)
        self.hybrid_retrieval_service = hybrid_retrieval_service
        self.vector_search_service = vector_search_service

    def retrieve(
        self,
        sample: dict[str, object],
        *,
        top_k: int,
        collection_filter_from_sample: bool,
        source_filter_from_sample: bool,
    ) -> list[dict[str, object]]:
        query = resolve_sample_query(sample)
        analysis = self.hybrid_retrieval_service._understand_query(query, "", [])
        vector_candidates = self._vector_recall(analysis)
        keyword_candidates = self._keyword_recall(analysis)
        fused_candidates = self.hybrid_retrieval_service._fuse_candidates(
            vector_candidates,
            keyword_candidates,
        )

        collections = set(str(item) for item in sample.get("collections", []))
        source_ids = set(str(item) for item in sample.get("source_ids", []))

        normalized: list[dict[str, object]] = []
        for candidate in fused_candidates:
            metadata = dict(candidate.metadata or {})
            chunk_id = str(candidate.id or "").strip()
            chunk = self.chunks_by_id.get(chunk_id, {})

            source_id = chunk.get("source_id") or metadata.get("source_id") or metadata.get("_source")
            source_file = chunk.get("source_file") or metadata.get("source_file") or metadata.get("_file_name")
            page_start = chunk.get("page_start") or metadata.get("page_start") or metadata.get("page_number")
            page_end = chunk.get("page_end") or metadata.get("page_end") or metadata.get("page_number")
            collection = chunk.get("collection") or metadata.get("collection")

            if collection_filter_from_sample and collections and str(collection) not in collections:
                continue
            if source_filter_from_sample and source_ids and str(source_id) not in source_ids:
                continue

            normalized.append(
                {
                    "chunk_id": chunk_id,
                    "source_id": str(source_id) if source_id is not None else "",
                    "source_file": str(source_file) if source_file is not None else "",
                    "page_start": to_optional_int(page_start),
                    "page_end": to_optional_int(page_end),
                    "score": float(candidate.fused_score),
                    "vector_score": float(candidate.vector_score),
                    "keyword_score": float(candidate.keyword_score),
                    "fused_score": float(candidate.fused_score),
                    "text": str(chunk.get("text") or candidate.content or ""),
                    "section_path": str(chunk.get("section_path") or metadata.get("section_path") or ""),
                    "matched_queries": list(candidate.matched_queries),
                }
            )

        output = []
        for rank, item in enumerate(normalized[:top_k], start=1):
            enriched = dict(item)
            enriched["rank"] = rank
            output.append(enriched)
        return output

    def _vector_recall(self, analysis) -> list[object]:
        collected: dict[str, object] = {}
        expanded_queries = list(getattr(analysis, "expanded_queries", []))
        per_query_limit = max(
            8,
            (50 + max(len(expanded_queries), 1) - 1) // max(len(expanded_queries), 1),
        )

        for query in expanded_queries:
            try:
                results = self.vector_search_service.search_similar_documents(
                    query,
                    top_k=per_query_limit,
                    collection_name=self.collection_name,
                )
            except Exception:
                continue

            for rank, item in enumerate(results, start=1):
                candidate = self.hybrid_retrieval_service._candidate_from_vector(item, rank, query)
                existing = collected.get(candidate.id)
                if existing is None or candidate.vector_score > existing.vector_score:
                    collected[candidate.id] = candidate
                elif query not in existing.matched_queries:
                    existing.matched_queries.append(query)

        return sorted(collected.values(), key=lambda item: item.vector_score, reverse=True)[:50]

    def _keyword_recall(self, analysis) -> list[object]:
        collected: dict[str, object] = {}
        for query in list(getattr(analysis, "expanded_queries", [])):
            results = self.bm25_service.search(query, top_k=50)
            for rank, item in enumerate(results, start=1):
                candidate = self.RetrievalCandidate(
                    id=item.id,
                    content=item.content,
                    metadata=item.metadata,
                    keyword_score=self.hybrid_retrieval_service._normalize_keyword_score(item.score, rank),
                    matched_queries=[query],
                )
                existing = collected.get(candidate.id)
                if existing is None or candidate.keyword_score > existing.keyword_score:
                    collected[candidate.id] = candidate
                elif query not in existing.matched_queries:
                    existing.matched_queries.append(query)

        return sorted(collected.values(), key=lambda item: item.keyword_score, reverse=True)[:50]

    def _load_bm25_corpus(self, chunk_rows: list[dict[str, object]]) -> None:
        corpus = []
        for row in chunk_rows:
            corpus.append(
                {
                    "content": str(row.get("text") or ""),
                    "chunk_id": str(row.get("chunk_id") or ""),
                    "source_id": str(row.get("source_id") or ""),
                    "source_file": str(row.get("source_file") or ""),
                    "collection": str(row.get("collection") or ""),
                    "page_start": to_optional_int(row.get("page_start")),
                    "page_end": to_optional_int(row.get("page_end")),
                    "section_path": str(row.get("section_path") or ""),
                }
            )

        tokenized_corpus = [self.bm25_service._tokenize(str(item.get("content") or "")) for item in corpus]
        doc_len = [len(tokens) for tokens in tokenized_corpus]
        avg_doc_len = sum(doc_len) / len(doc_len) if doc_len else 0.0
        doc_freq: dict[str, int] = defaultdict(int)
        for tokens in tokenized_corpus:
            for token in set(tokens):
                doc_freq[token] += 1

        self.bm25_service._corpus = corpus
        self.bm25_service._tokenized_corpus = tokenized_corpus
        self.bm25_service._doc_len = doc_len
        self.bm25_service._avg_doc_len = avg_doc_len
        self.bm25_service._doc_freq = dict(doc_freq)
        self.bm25_service._cache_mtime = float("inf")


class NoRerankAdapter:
    def rerank(
        self,
        *,
        sample: dict[str, object],
        candidate_results: list[dict[str, object]],
        final_top_k: int,
    ) -> dict[str, object]:
        reranked_results = []
        for rerank_rank, item in enumerate(candidate_results, start=1):
            reranked_results.append(
                {
                    **dict(item),
                    "rank": rerank_rank,
                    "original_rank": int(item["rank"]),
                    "rerank_rank": rerank_rank,
                    "original_score": float(item.get("score", 0.0)),
                    "rerank_score": float(item.get("score", 0.0)),
                    "score": float(item.get("score", 0.0)),
                }
            )
        return {
            "final_results": reranked_results[:final_top_k],
            "reranked_results": reranked_results,
            "rerank_available": True,
            "rerank_errors": [],
        }


class CurrentRerankAdapter:
    def __init__(self, chunks_by_id: dict[str, dict[str, object]]) -> None:
        self.chunks_by_id = chunks_by_id
        try:
            from app.services.hybrid_retrieval_service import hybrid_retrieval_service
        except Exception as exc:
            raise RuntimeError(
                "current rerank unavailable: 无法导入 app.services.hybrid_retrieval_service"
            ) from exc
        self.hybrid_retrieval_service = hybrid_retrieval_service

    def rerank(
        self,
        *,
        sample: dict[str, object],
        candidate_results: list[dict[str, object]],
        final_top_k: int,
    ) -> dict[str, object]:
        query = resolve_sample_query(sample)
        analysis = self.hybrid_retrieval_service._understand_query(query, "", [])
        retrieval_candidates = [
            self._build_retrieval_candidate(query=query, item=item)
            for item in candidate_results
        ]
        reranked_candidates, rerank_provider = self.hybrid_retrieval_service._rerank_candidates(
            analysis,
            retrieval_candidates,
        )
        original_by_id = {str(item["chunk_id"]): item for item in candidate_results}
        reranked_results = []
        for rerank_rank, candidate in enumerate(reranked_candidates, start=1):
            original = original_by_id.get(candidate.id)
            if original is None:
                continue
            reranked_results.append(
                {
                    **dict(original),
                    "rank": rerank_rank,
                    "original_rank": int(original["rank"]),
                    "rerank_rank": rerank_rank,
                    "original_score": float(original.get("score", 0.0)),
                    "rerank_score": float(candidate.rerank_score),
                    "score": float(candidate.rerank_score),
                    "rerank_provider": rerank_provider,
                }
            )
        return {
            "final_results": reranked_results[:final_top_k],
            "reranked_results": reranked_results,
            "rerank_available": True,
            "rerank_errors": [],
        }

    def _build_retrieval_candidate(self, *, query: str, item: dict[str, object]):
        from app.services.hybrid_retrieval_service import RetrievalCandidate

        chunk_id = str(item["chunk_id"])
        chunk = self.chunks_by_id.get(chunk_id, {})
        page_start = chunk.get("page_start") or item.get("page_start")
        metadata = {
            "chunk_id": chunk_id,
            "_file_name": str(chunk.get("source_file") or item.get("source_file") or ""),
            "section_path": str(chunk.get("section_path") or item.get("section_path") or ""),
            "page_number": to_optional_int(page_start),
        }
        content = str(chunk.get("text") or item.get("text") or "")
        original_rank = int(item["rank"])
        original_score = float(item.get("score", 0.0))
        similarity_score = float(item.get("vector_score", 1 / (1 + max(original_score, 0.0))))
        keyword_score = float(item.get("keyword_score", 0.0))
        fused_score = float(item.get("fused_score", 1 / (60 + original_rank)))
        candidate = RetrievalCandidate(
            id=chunk_id,
            content=content,
            metadata=metadata,
            vector_score=similarity_score,
            keyword_score=keyword_score,
            fused_score=fused_score,
            matched_queries=[query],
        )
        return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 检索评测脚本")
    parser.add_argument(
        "--dataset",
        default="aiops-docs/experiment/rag/splits/rag_build.jsonl",
        help="待评测数据集 JSONL",
    )
    parser.add_argument(
        "--chunks",
        default="aiops-docs/experiment/chunks/experiment_chunks.jsonl",
        help="experiment chunks JSONL",
    )
    parser.add_argument(
        "--output",
        default="aiops-docs/experiment/results/rag_retrieval_build_results.json",
        help="评测结果 JSON 输出路径",
    )
    parser.add_argument(
        "--summary",
        default="aiops-docs/experiment/results/thesis_tables/rag_retrieval_build_summary.csv",
        help="论文表格 CSV 输出路径",
    )
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--top-k", type=int, default=None, help="兼容旧参数；等价于 candidate-top-k/final-top-k 同值")
    parser.add_argument("--candidate-top-k", type=int, default=50)
    parser.add_argument("--final-top-k", type=int, default=10)
    parser.add_argument("--retrieval-strategy", default="dense", choices=["dense", "hybrid"])
    parser.add_argument("--rerank", default="none", choices=["none", "current"])
    parser.add_argument(
        "--query-mode",
        default="original",
        choices=["original", "original_keyword", "original_keyword_expanded"],
    )
    parser.add_argument("--experiment-name", default="rag_retrieval_experiment")
    parser.add_argument("--ks", default="1,3,5,10", help="以逗号分隔的 K 列表")
    parser.add_argument("--collection-filter-from-sample", action="store_true")
    parser.add_argument("--source-filter-from-sample", action="store_true")
    parser.add_argument("--mock-miss-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnose-live", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dump-raw-retrieval", action="store_true")
    parser.add_argument("--collection", default=None, help="live 模式显式指定 Milvus collection")
    parser.add_argument("--fail-on-missing-chunk-id", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.diagnose_live:
        if args.mode != "live":
            raise SystemExit("--diagnose-live 仅在 --mode live 下生效")
        report = diagnose_live_retrieval(
            dataset_path=Path(args.dataset),
            chunks_path=Path(args.chunks),
            output_path=LIVE_DIAGNOSIS_OUTPUT,
            top_k=resolve_top_k_values(
                legacy_top_k=args.top_k,
                candidate_top_k=args.candidate_top_k,
                final_top_k=args.final_top_k,
            )[0],
            limit=args.limit,
            collection_filter_from_sample=args.collection_filter_from_sample,
            source_filter_from_sample=args.source_filter_from_sample,
            collection_name=args.collection,
            dump_raw_retrieval=args.dump_raw_retrieval,
            fail_on_missing_chunk_id=args.fail_on_missing_chunk_id,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report["should_fail"] else 0
    report = evaluate_rag_retrieval(
        dataset_path=Path(args.dataset),
        chunks_path=Path(args.chunks),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        mode=args.mode,
        top_k=args.top_k,
        candidate_top_k=args.candidate_top_k,
        final_top_k=args.final_top_k,
        retrieval_strategy=args.retrieval_strategy,
        rerank=args.rerank,
        query_mode=args.query_mode,
        experiment_name=args.experiment_name,
        ks=parse_ks(args.ks),
        collection_filter_from_sample=args.collection_filter_from_sample,
        source_filter_from_sample=args.source_filter_from_sample,
        mock_miss_rate=args.mock_miss_rate,
        seed=args.seed,
        limit=args.limit,
        collection_name=args.collection,
        fail_on_missing_chunk_id=args.fail_on_missing_chunk_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report.get("should_fail") else 0


def evaluate_rag_retrieval(
    dataset_path: Path,
    chunks_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    mode: str = "mock",
    top_k: int | None = None,
    candidate_top_k: int | None = None,
    final_top_k: int | None = None,
    retrieval_strategy: str = "dense",
    rerank: str = "none",
    query_mode: str = "original",
    experiment_name: str = "rag_retrieval_experiment",
    ks: list[int] | None = None,
    collection_filter_from_sample: bool = False,
    source_filter_from_sample: bool = False,
    mock_miss_rate: float = 0.0,
    seed: int = 42,
    limit: int | None = None,
    collection_name: str | None = None,
    fail_on_missing_chunk_id: bool = False,
    adapter: RetrievalAdapter | None = None,
    rerank_adapter: RerankAdapter | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path.resolve()
    chunks_path = chunks_path.resolve()
    output_path = output_path.resolve()
    summary_path = summary_path.resolve()
    candidate_top_k, final_top_k = resolve_top_k_values(
        legacy_top_k=top_k,
        candidate_top_k=candidate_top_k,
        final_top_k=final_top_k,
    )
    validate_retrieval_config(
        mode=mode,
        retrieval_strategy=retrieval_strategy,
        rerank=rerank,
        query_mode=query_mode,
        candidate_top_k=candidate_top_k,
        final_top_k=final_top_k,
    )
    ks = sorted(set(ks or [1, 3, 5, 10]))
    candidate_ks = [10, 20, 50]

    dataset_rows = load_jsonl(dataset_path, limit=limit)
    chunk_rows = load_jsonl(chunks_path)
    chunks_by_id = {row["chunk_id"]: row for row in chunk_rows}

    if adapter is None:
        if mode == "mock":
            adapter = MockRetrievalAdapter(
                all_chunks=chunk_rows,
                chunks_by_id=chunks_by_id,
                mock_miss_rate=mock_miss_rate,
                seed=seed,
            )
        elif mode == "live" and retrieval_strategy == "dense":
            adapter = LiveRetrievalAdapter(
                chunks_by_id=chunks_by_id,
                collection_name=collection_name,
            )
        elif mode == "live" and retrieval_strategy == "hybrid":
            adapter = HybridRetrievalAdapter(
                chunks_by_id=chunks_by_id,
                chunk_rows=chunk_rows,
                collection_name=collection_name,
            )
        else:
            raise ValueError(f"未知模式或检索策略: mode={mode}, retrieval_strategy={retrieval_strategy}")
    if rerank_adapter is None:
        rerank_adapter = build_rerank_adapter(rerank=rerank, chunks_by_id=chunks_by_id)

    total_samples = len(dataset_rows)
    skipped_abstain = 0
    evaluated_rows: list[dict[str, object]] = []
    warnings: list[str] = []
    missing_chunk_id_count = 0
    rerank_errors: list[str] = []
    rerank_available = True
    gold_in_candidate_not_final_count = 0
    gold_promoted_by_rerank_count = 0
    gold_demoted_by_rerank_count = 0

    for sample in dataset_rows:
        if bool(sample.get("should_abstain", False)):
            skipped_abstain += 1
            continue

        query_plan = build_query_plan(sample, query_mode=query_mode)
        route_batches: list[dict[str, object]] = []
        errors: list[str] = []
        for query_type, query_text in query_plan.active_queries():
            route_sample = build_query_sample(sample, query_text)
            if mode == "live" and isinstance(adapter, LiveRetrievalAdapter):
                live_diagnostic = adapter.retrieve_diagnostic(
                    route_sample,
                    top_k=candidate_top_k,
                    collection_filter_from_sample=collection_filter_from_sample,
                    source_filter_from_sample=source_filter_from_sample,
                )
                retrieved_raw = live_diagnostic["normalized_results"]
                route_errors = list(live_diagnostic["errors"])
            else:
                retrieved_raw = adapter.retrieve(
                    route_sample,
                    top_k=candidate_top_k,
                    collection_filter_from_sample=collection_filter_from_sample,
                    source_filter_from_sample=source_filter_from_sample,
                )
                route_errors = []

            normalized_route = normalize_retrieved_batch(
                retrieved_raw,
                query_type=query_type,
                query_text=query_text,
            )
            route_errors.extend(normalized_route["errors"])
            missing_chunk_id_count += int(normalized_route["missing_chunk_id_count"])
            errors.extend(route_errors)
            route_batches.append(
                {
                    "query_type": query_type,
                    "query_text": query_text,
                    "candidate_results": normalized_route["candidate_results"],
                }
            )

        candidate_results, query_debug = merge_query_route_batches(
            sample=sample,
            route_batches=route_batches,
            candidate_top_k=candidate_top_k,
        )
        rerank_outcome = rerank_adapter.rerank(
            sample=sample,
            candidate_results=candidate_results,
            final_top_k=final_top_k,
        )
        final_results = list(rerank_outcome["final_results"])
        reranked_results = list(rerank_outcome["reranked_results"])
        rerank_available = rerank_available and bool(rerank_outcome["rerank_available"])
        rerank_errors.extend(str(item) for item in rerank_outcome["rerank_errors"])
        per_sample = evaluate_single_sample(
            sample,
            query_plan=query_plan,
            candidate_results=candidate_results,
            final_results=final_results,
            reranked_results=reranked_results,
            candidate_ks=candidate_ks,
            final_ks=ks,
            query_debug=query_debug,
        )
        per_sample["errors"] = unique_preserve_order(errors)
        gold_in_candidate_not_final_count += int(per_sample["gold_in_candidate_not_final"])
        gold_promoted_by_rerank_count += int(per_sample["gold_promoted_by_rerank"])
        gold_demoted_by_rerank_count += int(per_sample["gold_demoted_by_rerank"])
        evaluated_rows.append(per_sample)

    if missing_chunk_id_count:
        warnings.append(f"missing_chunk_id_count:{missing_chunk_id_count}")

    candidate_metrics = aggregate_named_metrics(
        evaluated_rows,
        metric_field_prefix="candidate",
        ks=candidate_ks,
        metric_names=["hit", "recall"],
        output_prefix="candidate_",
    )
    final_metrics = aggregate_named_metrics(
        evaluated_rows,
        metric_field_prefix="final",
        ks=ks,
        metric_names=["hit", "recall", "evidence_coverage", "source_accuracy", "page_accuracy"],
        output_prefix="",
        include_mrr=True,
    )
    metrics_by_question_type = aggregate_group_metrics(
        evaluated_rows,
        final_ks=ks,
        candidate_ks=candidate_ks,
        key_fn=lambda row: str(row["question_type"]),
    )
    metrics_by_source = aggregate_group_metrics(
        evaluated_rows,
        final_ks=ks,
        candidate_ks=candidate_ks,
        key_fn=lambda row: ",".join(sorted(row["source_ids"])),
    )

    result = {
        "dataset": to_repo_relative_path(dataset_path),
        "mode": mode,
        "experiment_name": experiment_name,
        "retrieval_strategy": retrieval_strategy,
        "rerank": rerank,
        "query_mode": query_mode,
        "rerank_available": rerank_available,
        "rerank_errors": unique_preserve_order(rerank_errors),
        "candidate_top_k": candidate_top_k,
        "final_top_k": final_top_k,
        "top_k": final_top_k,
        "ks": ks,
        "total_samples": total_samples,
        "evaluated_samples": len(evaluated_rows),
        "skipped_abstain": skipped_abstain,
        "missing_chunk_id_count": missing_chunk_id_count,
        "candidate_metrics": candidate_metrics,
        "final_metrics": final_metrics,
        "metrics": final_metrics,
        "gold_in_candidate_not_final_count": gold_in_candidate_not_final_count,
        "gold_promoted_by_rerank_count": gold_promoted_by_rerank_count,
        "gold_demoted_by_rerank_count": gold_demoted_by_rerank_count,
        "metrics_by_question_type": metrics_by_question_type,
        "metrics_by_source": metrics_by_source,
        "per_sample": evaluated_rows,
        "warnings": warnings,
        "should_fail": bool(fail_on_missing_chunk_id and missing_chunk_id_count > 0),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(summary_path, result)
    return result


def diagnose_live_retrieval(
    *,
    dataset_path: Path,
    chunks_path: Path,
    output_path: Path,
    top_k: int = 10,
    limit: int | None = None,
    collection_filter_from_sample: bool = False,
    source_filter_from_sample: bool = False,
    collection_name: str | None = None,
    dump_raw_retrieval: bool = False,
    fail_on_missing_chunk_id: bool = False,
    adapter: LiveRetrievalAdapter | None = None,
) -> dict[str, object]:
    dataset_path = dataset_path.resolve()
    chunks_path = chunks_path.resolve()
    output_path = output_path.resolve()

    dataset_rows = load_jsonl(dataset_path, limit=limit)
    chunk_rows = load_jsonl(chunks_path)
    chunks_by_id = {row["chunk_id"]: row for row in chunk_rows}

    if adapter is None:
        adapter = LiveRetrievalAdapter(
            chunks_by_id=chunks_by_id,
            collection_name=collection_name,
            dump_raw_retrieval=dump_raw_retrieval,
        )

    records: list[dict[str, object]] = []
    warnings: list[str] = []
    total_missing_chunk_id = 0

    for sample in dataset_rows:
        batch = adapter.retrieve_diagnostic(
            sample,
            top_k=top_k,
            collection_filter_from_sample=collection_filter_from_sample,
            source_filter_from_sample=source_filter_from_sample,
        )
        total_missing_chunk_id += int(batch["missing_chunk_id_count"])
        reference_chunk_ids = [str(item) for item in sample.get("reference_chunk_ids", [])]
        top_results = []
        matched_reference_chunk_ids: list[str] = []
        for item in batch["normalized_results"]:
            chunk_id = str(item.get("chunk_id") or "").strip()
            if chunk_id and chunk_id in reference_chunk_ids and chunk_id not in matched_reference_chunk_ids:
                matched_reference_chunk_ids.append(chunk_id)
            raw = dict(item.get("raw") or {})
            metadata = dict(raw.get("metadata") or {})
            top_results.append(
                {
                    "rank": int(item["rank"]),
                    "chunk_id": chunk_id,
                    "source_id": str(item.get("source_id") or ""),
                    "source_file": str(item.get("source_file") or ""),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "score": float(item.get("score", 0.0)),
                    "metadata_keys": sorted(str(key) for key in metadata.keys()),
                    "raw_preview": build_raw_preview(raw) if dump_raw_retrieval else "",
                }
            )

        records.append(
            {
                "sample_id": str(sample["id"]),
                "user_input": str(sample.get("user_input") or ""),
                "reference_chunk_ids": reference_chunk_ids,
                "sample_source_ids": [str(item) for item in sample.get("source_ids", [])],
                "sample_collections": [str(item) for item in sample.get("collections", [])],
                "raw_result_count": int(batch["raw_result_count"]),
                "normalized_result_count": int(batch["normalized_result_count"]),
                "top_results": top_results,
                "missing_chunk_id_count": int(batch["missing_chunk_id_count"]),
                "matched_reference_chunk_ids": matched_reference_chunk_ids,
                "hit_any_reference": bool(matched_reference_chunk_ids),
                "errors": list(batch["errors"]),
            }
        )

    if total_missing_chunk_id:
        warnings.append(f"missing_chunk_id_count:{total_missing_chunk_id}")

    result = {
        "dataset": to_repo_relative_path(dataset_path),
        "mode": "live",
        "collection": collection_name,
        "top_k": top_k,
        "limit": limit,
        "total_samples": len(dataset_rows),
        "records": records,
        "warnings": warnings,
        "missing_chunk_id_count": total_missing_chunk_id,
        "should_fail": bool(fail_on_missing_chunk_id and total_missing_chunk_id > 0),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def resolve_sample_query(sample: dict[str, object]) -> str:
    text = str(sample.get("user_input") or sample.get("question") or "").strip()
    if text:
        return text
    return str(sample.get("id") or "").strip()


def build_query_plan(sample: dict[str, object], *, query_mode: str) -> QueryPlan:
    main_query = resolve_sample_query(sample)
    summary = extract_conversation_summary(sample)
    recent_messages = extract_recent_messages(sample)
    analysis = build_query_analysis(main_query, summary=summary, recent_messages=recent_messages)

    keyword_query = str(analysis.get("keyword_query") or "").strip()
    if not keyword_query:
        keyword_query = build_keyword_query_fallback(main_query)

    expanded_query = build_expanded_query(
        main_query=main_query,
        keyword_query=keyword_query,
        analysis=analysis,
        summary=summary,
        recent_messages=recent_messages,
    )
    return QueryPlan(
        query_mode=query_mode,
        main_query=main_query,
        keyword_query=keyword_query,
        expanded_query=expanded_query,
        analysis=analysis,
    )


def extract_conversation_summary(sample: dict[str, object]) -> str:
    for key in ("conversation_summary", "summary"):
        text = str(sample.get(key) or "").strip()
        if text:
            return text
    return ""


def extract_recent_messages(sample: dict[str, object]) -> list[dict[str, str]]:
    for key in ("recent_messages", "recent_turns", "dialog_history", "history"):
        value = sample.get(key)
        if not isinstance(value, list):
            continue
        output: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                role = str(item.get("role") or "user").strip() or "user"
                content = str(item.get("content") or item.get("text") or "").strip()
                if content:
                    output.append({"role": role, "content": content})
            elif isinstance(item, str) and item.strip():
                output.append({"role": "user", "content": item.strip()})
        if output:
            return output
    return []


def build_query_analysis(
    main_query: str,
    *,
    summary: str,
    recent_messages: list[dict[str, str]],
) -> dict[str, object]:
    try:
        from app.services.hybrid_retrieval_service import hybrid_retrieval_service
    except Exception:
        keywords = extract_query_tokens(main_query)
        keyword_query = " ".join(keywords[:8]) or main_query
        expanded_queries = [main_query]
        if keyword_query and keyword_query != main_query:
            expanded_queries.append(keyword_query)
        return {
            "primary_query": main_query,
            "keyword_query": keyword_query,
            "expanded_queries": expanded_queries[:3],
            "keywords": keywords,
        }

    analysis = hybrid_retrieval_service._understand_query(main_query, summary, recent_messages)
    return {
        "primary_query": str(getattr(analysis, "primary_query", "") or main_query),
        "keyword_query": str(getattr(analysis, "keyword_query", "") or ""),
        "expanded_queries": list(getattr(analysis, "expanded_queries", []) or []),
        "keywords": list(getattr(analysis, "keywords", []) or []),
    }


def build_keyword_query_fallback(main_query: str) -> str:
    keywords = extract_query_tokens(main_query)
    return " ".join(keywords[:8]) or main_query


def build_expanded_query(
    *,
    main_query: str,
    keyword_query: str,
    analysis: dict[str, object],
    summary: str,
    recent_messages: list[dict[str, str]],
) -> str:
    expanded_queries = [str(item).strip() for item in analysis.get("expanded_queries", []) if str(item).strip()]
    primary_query = str(analysis.get("primary_query") or main_query).strip()
    for item in expanded_queries:
        if item not in {primary_query, keyword_query}:
            return item

    recent_user_text = " ".join(
        str(item.get("content") or "").strip()
        for item in recent_messages[-4:]
        if str(item.get("role") or "").strip() == "user"
    ).strip()
    if summary or recent_user_text:
        combined = " ".join(part for part in [summary[-120:], recent_user_text[-120:], main_query] if part).strip()
        if combined:
            return combined

    keywords = [str(item).strip() for item in analysis.get("keywords", []) if str(item).strip()]
    intent_terms = infer_expansion_intent_terms(main_query, keywords)
    parts = [main_query]
    if keyword_query and keyword_query != main_query:
        parts.append(keyword_query)
    if intent_terms:
        parts.append(" ".join(intent_terms))
    return " ".join(unique_preserve_order([part for part in parts if part])).strip()


def infer_expansion_intent_terms(main_query: str, keywords: list[str]) -> list[str]:
    text = " ".join([main_query.lower(), " ".join(keywords).lower()])
    if any(term in text for term in ("alarm", "fault", "error", "报码", "报错", "告警", "故障", "code")):
        return ["alarm", "fault", "troubleshooting", "procedure"]
    if any(term in text for term in ("parameter", "setting", "config", "参数", "设定", "配置")):
        return ["parameter", "setting", "configuration", "procedure"]
    if any(term in text for term in ("safety", "warning", "hazard", "安全", "警告")):
        return ["safety", "warning", "precaution", "procedure"]
    return ["troubleshooting", "maintenance", "procedure"]


def extract_query_tokens(text: str) -> list[str]:
    output: list[str] = []
    for token in QUERY_TOKEN_PATTERN.findall(str(text or "").lower()):
        cleaned = token.strip()
        if not cleaned or cleaned in QUERY_STOPWORDS or len(cleaned) <= 1:
            continue
        if cleaned not in output:
            output.append(cleaned)
    return output


def build_query_sample(sample: dict[str, object], query_text: str) -> dict[str, object]:
    route_sample = dict(sample)
    route_sample["user_input"] = query_text
    route_sample["question"] = query_text
    return route_sample


def normalize_retrieved_batch(
    retrieved_raw: list[dict[str, object]],
    *,
    query_type: str,
    query_text: str,
) -> dict[str, object]:
    candidate_results = []
    missing_chunk_id_count = 0
    errors: list[str] = []
    for item in retrieved_raw:
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id:
            missing_chunk_id_count += 1
            errors.append("missing_chunk_id_in_retrieved")
            continue
        normalized = {
            "rank": int(item["rank"]),
            "chunk_id": chunk_id,
            "source_id": str(item.get("source_id") or ""),
            "source_file": str(item.get("source_file") or ""),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "score": float(item.get("score", 0.0)),
            "matched_queries": list(item.get("matched_queries") or [query_text]),
            "matched_query_types": [query_type],
        }
        if "vector_score" in item:
            normalized["vector_score"] = float(item.get("vector_score", 0.0))
        if "keyword_score" in item:
            normalized["keyword_score"] = float(item.get("keyword_score", 0.0))
        if "fused_score" in item:
            normalized["fused_score"] = float(item.get("fused_score", 0.0))
        if "text" in item:
            normalized["text"] = str(item.get("text") or "")
        if "section_path" in item:
            normalized["section_path"] = str(item.get("section_path") or "")
        if "raw" in item:
            normalized["raw"] = item.get("raw")
        candidate_results.append(normalized)

    return {
        "candidate_results": candidate_results,
        "missing_chunk_id_count": missing_chunk_id_count,
        "errors": unique_preserve_order(errors),
    }


def merge_query_route_batches(
    *,
    sample: dict[str, object],
    route_batches: list[dict[str, object]],
    candidate_top_k: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(route_batches) == 1 and str(route_batches[0]["query_type"]) == "main_query":
        candidate_results = list(route_batches[0]["candidate_results"])[:candidate_top_k]
        route_chunk_ids = {str(item["chunk_id"]) for item in candidate_results}
        reference_chunk_ids = set(str(item) for item in sample.get("reference_chunk_ids", []))
        return candidate_results, {
            "raw_candidate_count": len(candidate_results),
            "union_candidate_count": len(candidate_results),
            "duplicate_candidate_ratio": 0.0,
            "matched_query_types": ["main_query"] if candidate_results else [],
            "source_query_hit": {
                "main_query": bool(reference_chunk_ids & route_chunk_ids),
                "keyword_query": False,
                "expanded_query": False,
            },
        }

    merged_by_chunk_id: dict[str, dict[str, object]] = {}
    raw_candidate_count = 0
    source_query_hit = {
        "main_query": False,
        "keyword_query": False,
        "expanded_query": False,
    }
    reference_chunk_ids = set(str(item) for item in sample.get("reference_chunk_ids", []))

    for batch in route_batches:
        query_type = str(batch["query_type"])
        query_text = str(batch["query_text"])
        candidate_results = list(batch["candidate_results"])
        raw_candidate_count += len(candidate_results)
        route_chunk_ids = {str(item["chunk_id"]) for item in candidate_results}
        if reference_chunk_ids & route_chunk_ids:
            source_query_hit[query_type] = True

        for item in candidate_results:
            chunk_id = str(item["chunk_id"])
            similarity_score = float(item.get("vector_score", 1 / (1 + max(float(item.get("score", 0.0)), 0.0))))
            current = merged_by_chunk_id.get(chunk_id)
            if current is None:
                merged_by_chunk_id[chunk_id] = {
                    **dict(item),
                    "best_rank": int(item["rank"]),
                    "union_score": 1 / (60 + int(item["rank"])),
                    "vector_score": similarity_score,
                    "matched_queries": unique_preserve_order(list(item.get("matched_queries") or [query_text])),
                    "matched_query_types": [query_type],
                }
                continue

            current["best_rank"] = min(int(current["best_rank"]), int(item["rank"]))
            current["union_score"] = float(current["union_score"]) + (1 / (60 + int(item["rank"])))
            current["vector_score"] = max(float(current.get("vector_score", 0.0)), similarity_score)
            for matched_query in list(item.get("matched_queries") or [query_text]):
                if matched_query not in current["matched_queries"]:
                    current["matched_queries"].append(matched_query)
            if query_type not in current["matched_query_types"]:
                current["matched_query_types"].append(query_type)
            current_similarity = float(current.get("vector_score", 0.0))
            if similarity_score >= current_similarity:
                current["score"] = float(item.get("score", 0.0))
            if "keyword_score" in item:
                current["keyword_score"] = max(float(current.get("keyword_score", 0.0)), float(item["keyword_score"]))
            if "fused_score" in item:
                current["fused_score"] = max(float(current.get("fused_score", 0.0)), float(item["fused_score"]))

    union_candidate_count = len(merged_by_chunk_id)
    duplicate_candidate_ratio = 0.0
    if raw_candidate_count > 0:
        duplicate_candidate_ratio = round((raw_candidate_count - union_candidate_count) / raw_candidate_count, 6)

    union_results = sorted(
        merged_by_chunk_id.values(),
        key=lambda item: (
            -float(item.get("union_score", 0.0)),
            -float(item.get("vector_score", 0.0)),
            int(item.get("best_rank", 999999)),
            str(item.get("chunk_id", "")),
        ),
    )
    candidate_results = []
    for rank, item in enumerate(union_results[:candidate_top_k], start=1):
        candidate_results.append(
            {
                **{key: value for key, value in item.items() if key not in {"best_rank", "union_score"}},
                "rank": rank,
                "score": float(item.get("score", 0.0)),
                "vector_score": float(item.get("vector_score", 0.0)),
                "matched_queries": list(item.get("matched_queries") or []),
                "matched_query_types": list(item.get("matched_query_types") or []),
            }
        )

    matched_query_types = []
    for query_type in ("main_query", "keyword_query", "expanded_query"):
        if any(query_type in item.get("matched_query_types", []) for item in candidate_results):
            matched_query_types.append(query_type)

    return candidate_results, {
        "raw_candidate_count": raw_candidate_count,
        "union_candidate_count": union_candidate_count,
        "duplicate_candidate_ratio": duplicate_candidate_ratio,
        "matched_query_types": matched_query_types,
        "source_query_hit": source_query_hit,
    }


def evaluate_single_sample(
    sample: dict[str, object],
    *,
    query_plan: QueryPlan,
    candidate_results: list[dict[str, object]],
    final_results: list[dict[str, object]],
    reranked_results: list[dict[str, object]],
    candidate_ks: list[int],
    final_ks: list[int],
    query_debug: dict[str, object],
) -> dict[str, object]:
    reference_chunk_ids = [str(item) for item in sample.get("reference_chunk_ids", [])]
    candidate_metrics = score_ranked_results(sample, candidate_results, candidate_ks)
    final_metrics = score_ranked_results(sample, final_results, final_ks)
    candidate_chunk_ids = {str(item["chunk_id"]) for item in candidate_results}
    final_chunk_ids = {str(item["chunk_id"]) for item in final_results}
    reference_chunk_set = set(reference_chunk_ids)
    gold_in_candidate_not_final = bool(reference_chunk_set & candidate_chunk_ids and not reference_chunk_set & final_chunk_ids)
    original_best_rank = best_gold_rank(reference_chunk_set, candidate_results, rank_field="rank")
    reranked_best_rank = best_gold_rank(reference_chunk_set, reranked_results, rank_field="rerank_rank")
    gold_promoted_by_rerank = bool(
        original_best_rank is not None and reranked_best_rank is not None and reranked_best_rank < original_best_rank
    )
    gold_demoted_by_rerank = bool(
        original_best_rank is not None and reranked_best_rank is not None and reranked_best_rank > original_best_rank
    )

    return {
        "id": sample["id"],
        "user_input": resolve_sample_query(sample),
        "question_type": sample["question_type"],
        "source_ids": sample.get("source_ids", []),
        "reference_chunk_ids": reference_chunk_ids,
        "query_mode": query_plan.query_mode,
        "main_query": query_plan.main_query,
        "keyword_query": query_plan.keyword_query,
        "expanded_query": query_plan.expanded_query,
        "raw_candidate_count": int(query_debug["raw_candidate_count"]),
        "union_candidate_count": int(query_debug["union_candidate_count"]),
        "duplicate_candidate_ratio": float(query_debug["duplicate_candidate_ratio"]),
        "matched_query_types": list(query_debug["matched_query_types"]),
        "source_query_hit": dict(query_debug["source_query_hit"]),
        "candidate_results": candidate_results,
        "final_results": final_results,
        "reranked_results": reranked_results,
        "retrieved": final_results,
        "candidate_hit_at_k": candidate_metrics["hit_at_k"],
        "candidate_recall_at_k": candidate_metrics["recall_at_k"],
        "hit_at_k": final_metrics["hit_at_k"],
        "recall_at_k": final_metrics["recall_at_k"],
        "first_relevant_rank": final_metrics["first_relevant_rank"],
        "mrr": final_metrics["mrr"],
        "evidence_coverage_at_k": final_metrics["evidence_coverage_at_k"],
        "source_accuracy_at_k": final_metrics["source_accuracy_at_k"],
        "page_accuracy_at_k": final_metrics["page_accuracy_at_k"],
        "gold_in_candidate_not_final": gold_in_candidate_not_final,
        "gold_promoted_by_rerank": gold_promoted_by_rerank,
        "gold_demoted_by_rerank": gold_demoted_by_rerank,
        "errors": [],
    }


def page_intersects(item: dict[str, object], expected_page_numbers: set[int]) -> bool:
    if not expected_page_numbers:
        return False
    page_start = item.get("page_start")
    page_end = item.get("page_end")
    if page_start is None or page_end is None:
        return False
    return not expected_page_numbers.isdisjoint(set(range(int(page_start), int(page_end) + 1)))


def score_ranked_results(
    sample: dict[str, object],
    results: list[dict[str, object]],
    ks: list[int],
) -> dict[str, object]:
    reference_chunk_set = set(str(item) for item in sample.get("reference_chunk_ids", []))
    expected_source_ids = set(str(item) for item in sample.get("source_ids", []))
    expected_page_numbers = set(int(item) for item in sample.get("expected_page_numbers", []))

    first_relevant_rank = None
    for item in results:
        if item["chunk_id"] in reference_chunk_set:
            first_relevant_rank = int(item["rank"])
            break
    mrr = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank

    hit_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    evidence_coverage_at_k: dict[str, float] = {}
    source_accuracy_at_k: dict[str, float] = {}
    page_accuracy_at_k: dict[str, float] = {}

    for k in ks:
        top_items = results[:k]
        top_chunk_ids = {item["chunk_id"] for item in top_items}
        matched_reference = reference_chunk_set & top_chunk_ids
        hit_at_k[str(k)] = 1.0 if matched_reference else 0.0
        recall_at_k[str(k)] = (
            len(matched_reference) / len(reference_chunk_set) if reference_chunk_set else 0.0
        )
        evidence_coverage_at_k[str(k)] = (
            1.0 if reference_chunk_set and reference_chunk_set.issubset(top_chunk_ids) else 0.0
        )
        source_accuracy_at_k[str(k)] = (
            1.0 if any(item["source_id"] in expected_source_ids for item in top_items) else 0.0
        )
        page_accuracy_at_k[str(k)] = 1.0 if any(page_intersects(item, expected_page_numbers) for item in top_items) else 0.0

    return {
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
        "first_relevant_rank": first_relevant_rank,
        "mrr": round(mrr, 6),
        "evidence_coverage_at_k": evidence_coverage_at_k,
        "source_accuracy_at_k": source_accuracy_at_k,
        "page_accuracy_at_k": page_accuracy_at_k,
    }


def aggregate_named_metrics(
    rows: list[dict[str, object]],
    *,
    metric_field_prefix: str,
    ks: list[int],
    metric_names: list[str],
    output_prefix: str = "",
    include_mrr: bool = False,
) -> dict[str, float]:
    if not rows:
        return {}

    prefix_to_row_field = {
        "hit": f"{metric_field_prefix}_hit_at_k" if metric_field_prefix != "final" else "hit_at_k",
        "recall": f"{metric_field_prefix}_recall_at_k" if metric_field_prefix != "final" else "recall_at_k",
        "evidence_coverage": "evidence_coverage_at_k",
        "source_accuracy": "source_accuracy_at_k",
        "page_accuracy": "page_accuracy_at_k",
    }
    output: dict[str, float] = {}
    for metric_name in metric_names:
        row_field = prefix_to_row_field[metric_name]
        for k in ks:
            key = str(k)
            metric_key = f"{output_prefix}{metric_name}_at_{k}"
            output[metric_key] = round(
                sum(row[row_field][key] for row in rows) / len(rows),
                6,
            )
    if include_mrr:
        output["mrr"] = round(sum(row["mrr"] for row in rows) / len(rows), 6)
    return output


def aggregate_group_metrics(
    rows: list[dict[str, object]],
    *,
    final_ks: list[int],
    candidate_ks: list[int],
    key_fn,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)

    output = {}
    for key, group_rows in sorted(grouped.items()):
        output[key] = {
            "samples": len(group_rows),
            "candidate_metrics": aggregate_named_metrics(
                group_rows,
                metric_field_prefix="candidate",
                ks=candidate_ks,
                metric_names=["hit", "recall"],
                output_prefix="candidate_",
            ),
            "final_metrics": aggregate_named_metrics(
                group_rows,
                metric_field_prefix="final",
                ks=final_ks,
                metric_names=["hit", "recall", "evidence_coverage", "source_accuracy", "page_accuracy"],
                output_prefix="",
                include_mrr=True,
            ),
        }
    return output


def write_summary_csv(path: Path, result: dict[str, object]) -> None:
    candidate_metric_names = sorted(result["candidate_metrics"].keys())
    final_metric_names = sorted(result["final_metrics"].keys())
    fieldnames = [
        "experiment_name",
        "dataset",
        "mode",
        "retrieval_strategy",
        "rerank",
        "query_mode",
        "rerank_available",
        "candidate_top_k",
        "final_top_k",
        "top_k",
        "ks",
        "total_samples",
        "evaluated_samples",
        "skipped_abstain",
        "missing_chunk_id_count",
        "gold_in_candidate_not_final_count",
        "gold_promoted_by_rerank_count",
        "gold_demoted_by_rerank_count",
        *candidate_metric_names,
        *final_metric_names,
    ]
    row = {
        "experiment_name": result["experiment_name"],
        "dataset": result["dataset"],
        "mode": result["mode"],
        "retrieval_strategy": result["retrieval_strategy"],
        "rerank": result["rerank"],
        "query_mode": result["query_mode"],
        "rerank_available": result["rerank_available"],
        "candidate_top_k": result["candidate_top_k"],
        "final_top_k": result["final_top_k"],
        "top_k": result["top_k"],
        "ks": ",".join(str(item) for item in result["ks"]),
        "total_samples": result["total_samples"],
        "evaluated_samples": result["evaluated_samples"],
        "skipped_abstain": result["skipped_abstain"],
        "missing_chunk_id_count": result["missing_chunk_id_count"],
        "gold_in_candidate_not_final_count": result["gold_in_candidate_not_final_count"],
        "gold_promoted_by_rerank_count": result["gold_promoted_by_rerank_count"],
        "gold_demoted_by_rerank_count": result["gold_demoted_by_rerank_count"],
    }
    row.update(result["candidate_metrics"])
    row.update(result["final_metrics"])

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def parse_ks(value: str) -> list[int]:
    ks = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ks.append(int(item))
    return ks


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is None:
        return rows
    return rows[:limit]


def extract_live_chunk_id(item: object, metadata: dict[str, object]) -> str:
    candidates = [
        getattr(item, "id", None),
        getattr(item, "pk", None),
        getattr(item, "primary_key", None),
        metadata.get("chunk_id"),
        metadata.get("id"),
        metadata.get("pk"),
        metadata.get("primary_key"),
        metadata.get("context_id"),
        metadata.get("document_chunk_id"),
    ]
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def build_live_raw_payload(item: object, metadata: dict[str, object]) -> dict[str, object]:
    return {
        "id": getattr(item, "id", None),
        "pk": getattr(item, "pk", None),
        "primary_key": getattr(item, "primary_key", None),
        "score": getattr(item, "score", None),
        "metadata": metadata,
    }


def build_raw_preview(raw: dict[str, object], *, limit: int = 600) -> str:
    text = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def to_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_top_k_values(
    *,
    legacy_top_k: int | None,
    candidate_top_k: int | None,
    final_top_k: int | None,
) -> tuple[int, int]:
    if legacy_top_k is not None:
        return legacy_top_k, legacy_top_k
    resolved_candidate = candidate_top_k or 50
    resolved_final = final_top_k or 10
    return resolved_candidate, resolved_final


def validate_retrieval_config(
    *,
    mode: str,
    retrieval_strategy: str,
    rerank: str,
    query_mode: str,
    candidate_top_k: int,
    final_top_k: int,
) -> None:
    if mode not in {"mock", "live"}:
        raise ValueError(f"未知 mode: {mode}")
    if retrieval_strategy not in {"dense", "hybrid"}:
        raise ValueError(f"当前仅支持 dense/hybrid 检索策略: {retrieval_strategy}")
    if rerank not in {"none", "current"}:
        raise ValueError(f"当前仅支持 rerank=none/current: {rerank}")
    if query_mode not in {"original", "original_keyword", "original_keyword_expanded"}:
        raise ValueError(f"未知 query_mode: {query_mode}")
    if candidate_top_k <= 0 or final_top_k <= 0:
        raise ValueError("candidate_top_k/final_top_k 必须大于 0")
    if final_top_k > candidate_top_k:
        raise ValueError("final_top_k 不能大于 candidate_top_k")


def build_rerank_adapter(
    *,
    rerank: str,
    chunks_by_id: dict[str, dict[str, object]],
) -> RerankAdapter:
    if rerank == "none":
        return NoRerankAdapter()
    if rerank == "current":
        return CurrentRerankAdapter(chunks_by_id=chunks_by_id)
    raise ValueError(f"当前仅支持 rerank=none/current: {rerank}")


def best_gold_rank(
    reference_chunk_set: set[str],
    rows: list[dict[str, object]],
    *,
    rank_field: str,
) -> int | None:
    best = None
    for row in rows:
        if str(row.get("chunk_id") or "") not in reference_chunk_set:
            continue
        rank = int(row[rank_field])
        best = rank if best is None else min(best, rank)
    return best


def stable_hash_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def to_repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    repo_root = Path.cwd().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
