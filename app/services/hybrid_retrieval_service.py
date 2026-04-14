"""混合检索、重排与置信度评估。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.bm25_search_service import bm25_search_service
from app.services.vector_search_service import SearchResult, vector_search_service


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]")
STOPWORDS = {
    "怎么", "如何", "请问", "一下", "这个", "那个", "什么", "是否", "可以", "需要",
    "怎么做", "处理", "问题", "告警", "故障", "机器", "设备", "agent",
}


@dataclass
class QueryUnderstandingResult:
    """查询理解结果。"""

    primary_query: str
    keyword_query: str
    expanded_queries: List[str]
    keywords: List[str]


@dataclass
class RetrievalCandidate:
    """候选检索结果。"""

    id: str
    content: str
    metadata: Dict[str, object]
    vector_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    matched_queries: List[str] = field(default_factory=list)

    def to_document(self) -> Document:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "vector_score": self.vector_score,
                "keyword_score": self.keyword_score,
                "fused_score": self.fused_score,
                "rerank_score": self.rerank_score,
            }
        )
        return Document(page_content=self.content, metadata=metadata)


@dataclass
class RetrievalResult:
    """混合检索结果。"""

    query_analysis: QueryUnderstandingResult
    candidates: List[RetrievalCandidate]
    references: List[Dict[str, object]]
    confidence: str
    queued_for_supplement: bool
    low_confidence_reason: str

    def context_text(self) -> str:
        parts = []
        for index, candidate in enumerate(self.candidates, start=1):
            parts.append(
                "\n".join(
                    [
                        f"【证据 {index}】",
                        f"来源: {self._format_reference(candidate.metadata)}",
                        f"重排分数: {candidate.rerank_score:.3f}",
                        candidate.content,
                    ]
                )
            )
        return "\n\n".join(parts)

    def documents(self) -> List[Document]:
        return [candidate.to_document() for candidate in self.candidates]

    def _format_reference(self, metadata: Dict[str, object]) -> str:
        file_name = str(metadata.get("_file_name", "未知来源"))
        page_number = metadata.get("page_number")
        section_path = str(metadata.get("section_path", "")).strip()
        parts = [file_name]
        if page_number:
            parts.append(f"第{page_number}页")
        if section_path:
            parts.append(section_path)
        return " / ".join(parts)


class HybridRetrievalService:
    """混合检索服务。"""

    def retrieve(
        self,
        query: str,
        summary: str = "",
        recent_messages: List[Dict[str, str]] | None = None,
    ) -> RetrievalResult:
        analysis = self._understand_query(query, summary, recent_messages or [])
        vector_candidates = self._vector_recall(analysis)
        keyword_candidates = self._keyword_recall(analysis)

        merged = self._fuse_candidates(vector_candidates, keyword_candidates)
        reranked = self._rerank_candidates(analysis, merged)
        top_candidates = reranked[: config.rag_final_top_k]

        confidence, reason = self._evaluate_confidence(top_candidates)
        references = [self._build_reference(candidate) for candidate in top_candidates]

        logger.info(
            f"混合检索完成: vector={len(vector_candidates)}, bm25={len(keyword_candidates)}, final={len(top_candidates)}, confidence={confidence}"
        )

        return RetrievalResult(
            query_analysis=analysis,
            candidates=top_candidates,
            references=references,
            confidence=confidence,
            queued_for_supplement=False,
            low_confidence_reason=reason,
        )

    def _understand_query(
        self,
        query: str,
        summary: str,
        recent_messages: List[Dict[str, str]],
    ) -> QueryUnderstandingResult:
        normalized = " ".join(query.strip().split())
        recent_context = " ".join(item["content"] for item in recent_messages[-4:] if item["role"] == "user")
        base_text = " ".join(part for part in [summary, recent_context, normalized] if part)

        keywords = self._extract_keywords(base_text or normalized)
        keyword_query = " ".join(keywords[:8]) or normalized

        expanded_queries = [normalized]
        if keyword_query and keyword_query != normalized:
            expanded_queries.append(keyword_query)
        if summary:
            expanded_queries.append(f"{summary[:120]} {normalized}")
        if recent_context and recent_context not in expanded_queries:
            expanded_queries.append(f"{recent_context[-120:]} {normalized}")

        unique_queries = []
        for item in expanded_queries:
            cleaned = item.strip()
            if cleaned and cleaned not in unique_queries:
                unique_queries.append(cleaned)

        return QueryUnderstandingResult(
            primary_query=normalized,
            keyword_query=keyword_query,
            expanded_queries=unique_queries[:3],
            keywords=keywords,
        )

    def _vector_recall(self, analysis: QueryUnderstandingResult) -> List[RetrievalCandidate]:
        collected: Dict[str, RetrievalCandidate] = {}
        per_query_limit = max(8, math.ceil(config.rag_candidate_top_k / max(len(analysis.expanded_queries), 1)))

        for query in analysis.expanded_queries:
            try:
                results = vector_search_service.search_similar_documents(query, top_k=per_query_limit)
            except Exception as exc:
                logger.warning(f"向量召回失败，query={query}: {exc}")
                continue

            for rank, item in enumerate(results, start=1):
                candidate = self._candidate_from_vector(item, rank, query)
                existing = collected.get(candidate.id)
                if existing is None or candidate.vector_score > existing.vector_score:
                    collected[candidate.id] = candidate
                elif query not in existing.matched_queries:
                    existing.matched_queries.append(query)

        return sorted(collected.values(), key=lambda item: item.vector_score, reverse=True)[
            : config.rag_candidate_top_k
        ]

    def _keyword_recall(self, analysis: QueryUnderstandingResult) -> List[RetrievalCandidate]:
        if not config.bm25_enabled:
            return []

        collected: Dict[str, RetrievalCandidate] = {}
        for query in analysis.expanded_queries:
            results = bm25_search_service.search(query, top_k=config.rag_candidate_top_k)
            for rank, item in enumerate(results, start=1):
                candidate = RetrievalCandidate(
                    id=item.id,
                    content=item.content,
                    metadata=item.metadata,
                    keyword_score=self._normalize_keyword_score(item.score, rank),
                    matched_queries=[query],
                )
                existing = collected.get(candidate.id)
                if existing is None or candidate.keyword_score > existing.keyword_score:
                    collected[candidate.id] = candidate
                elif query not in existing.matched_queries:
                    existing.matched_queries.append(query)

        return sorted(collected.values(), key=lambda item: item.keyword_score, reverse=True)[
            : config.rag_candidate_top_k
        ]

    def _fuse_candidates(
        self,
        vector_candidates: List[RetrievalCandidate],
        keyword_candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        fused: Dict[str, RetrievalCandidate] = {}

        for rank, candidate in enumerate(vector_candidates, start=1):
            item = fused.setdefault(candidate.id, candidate)
            item.fused_score += 1 / (60 + rank)
            item.vector_score = max(item.vector_score, candidate.vector_score)

        for rank, candidate in enumerate(keyword_candidates, start=1):
            item = fused.get(candidate.id)
            if item is None:
                item = candidate
                fused[candidate.id] = item
            item.fused_score += 1 / (60 + rank)
            item.keyword_score = max(item.keyword_score, candidate.keyword_score)
            for query in candidate.matched_queries:
                if query not in item.matched_queries:
                    item.matched_queries.append(query)

        return sorted(fused.values(), key=lambda item: item.fused_score, reverse=True)[
            : config.rag_candidate_top_k
        ]

    def _rerank_candidates(
        self,
        analysis: QueryUnderstandingResult,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        for candidate in candidates:
            overlap = self._token_overlap_ratio(analysis.keywords, candidate.content)
            metadata_text = " ".join(
                str(candidate.metadata.get(key, "")) for key in ("section_path", "_file_name", "page_number")
            )
            metadata_overlap = self._token_overlap_ratio(analysis.keywords, metadata_text)
            candidate.rerank_score = (
                0.45 * candidate.fused_score
                + 0.30 * candidate.vector_score
                + 0.15 * candidate.keyword_score
                + 0.07 * overlap
                + 0.03 * metadata_overlap
            )

        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)

    def _evaluate_confidence(self, candidates: List[RetrievalCandidate]) -> tuple[str, str]:
        if not candidates:
            return "low", "未检索到任何相关片段"

        top1 = candidates[0].rerank_score
        support_count = len([candidate for candidate in candidates[:3] if candidate.rerank_score >= 0.35])

        if top1 >= config.rag_confidence_threshold_high and support_count >= 2:
            return "high", ""
        if top1 < config.rag_confidence_threshold_low:
            return "low", f"最高证据分数偏低: {top1:.3f}"
        return "medium", f"证据支撑有限: top1={top1:.3f}, support={support_count}"

    def _build_reference(self, candidate: RetrievalCandidate) -> Dict[str, object]:
        metadata = candidate.metadata
        return {
            "file_name": metadata.get("_file_name", "未知来源"),
            "page_number": metadata.get("page_number"),
            "section_path": metadata.get("section_path"),
            "score": round(candidate.rerank_score, 4),
        }

    def _candidate_from_vector(
        self,
        result: SearchResult,
        rank: int,
        query: str,
    ) -> RetrievalCandidate:
        similarity_score = 1 / (1 + max(result.score, 0.0))
        metadata = result.metadata or {}
        candidate_id = str(metadata.get("chunk_id") or result.id or f"{metadata.get('_source', 'milvus')}::{rank}")
        return RetrievalCandidate(
            id=candidate_id,
            content=result.content,
            metadata=metadata,
            vector_score=similarity_score,
            matched_queries=[query],
        )

    def _normalize_keyword_score(self, score: float, rank: int) -> float:
        return min(1.0, (score / max(rank, 1)) + 0.2)

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = TOKEN_PATTERN.findall(text.lower())
        keywords: List[str] = []
        for token in tokens:
            if token in STOPWORDS or len(token.strip()) <= 1:
                continue
            if token not in keywords:
                keywords.append(token)
        return keywords[:12]

    def _token_overlap_ratio(self, keywords: Iterable[str], text: str) -> float:
        tokens = set(TOKEN_PATTERN.findall(text.lower()))
        keyword_list = [item for item in keywords if item]
        if not keyword_list:
            return 0.0
        hits = sum(1 for token in keyword_list if token in tokens)
        return hits / len(keyword_list)


hybrid_retrieval_service = HybridRetrievalService()
