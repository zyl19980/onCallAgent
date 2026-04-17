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
from app.services.reranker_service import reranker_service
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
    raw_rerank_score: float = 0.0
    rerank_score: float = 0.0
    rerank_source: str = "local"
    document_confidence: str = "low"
    matched_queries: List[str] = field(default_factory=list)

    def to_document(self) -> Document:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "vector_score": self.vector_score,
                "keyword_score": self.keyword_score,
                "fused_score": self.fused_score,
                "raw_rerank_score": self.raw_rerank_score,
                "rerank_score": self.rerank_score,
                "rerank_source": self.rerank_source,
                "document_confidence": self.document_confidence,
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
    rerank_provider: str = "local"
    confidence_debug: Dict[str, object] = field(default_factory=dict)

    def context_text(self) -> str:
        parts = []
        for index, candidate in enumerate(self.candidates, start=1):
            parts.append(
                "\n".join(
                    [
                        f"【证据 {index}】",
                        f"来源: {self._format_reference(candidate.metadata)}",
                        f"文档置信度: {candidate.document_confidence}",
                        f"重排来源: {candidate.rerank_source}",
                        f"重排分数: {candidate.rerank_score:.3f} (raw={candidate.raw_rerank_score:.3f})",
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
        logger.info(
            "查询理解完成: primary='{}', keyword='{}', expanded={}, keywords={}",
            analysis.primary_query,
            analysis.keyword_query,
            analysis.expanded_queries,
            analysis.keywords,
        )

        vector_candidates = self._vector_recall(analysis)
        keyword_candidates = self._keyword_recall(analysis)

        merged = self._fuse_candidates(vector_candidates, keyword_candidates)
        reranked, rerank_provider = self._rerank_candidates(analysis, merged)
        top_candidates = reranked[: config.rag_final_top_k]
        self._label_document_confidences(top_candidates)

        confidence, reason, confidence_debug = self._evaluate_confidence(top_candidates, rerank_provider)
        references = [self._build_reference(candidate) for candidate in top_candidates]

        logger.info(
            "混合检索完成: vector={}, bm25={}, final={}, rerank_provider={}, overall_confidence={}",
            len(vector_candidates),
            len(keyword_candidates),
            len(top_candidates),
            rerank_provider,
            confidence,
        )
        logger.info(
            "Top 文档摘要: {}",
            [
                {
                    "score": round(candidate.rerank_score, 4),
                    "label": candidate.document_confidence,
                    "source": candidate.metadata.get("_file_name", "未知来源"),
                }
                for candidate in top_candidates[:5]
            ],
        )

        return RetrievalResult(
            query_analysis=analysis,
            candidates=top_candidates,
            references=references,
            confidence=confidence,
            queued_for_supplement=False,
            low_confidence_reason=reason,
            rerank_provider=rerank_provider,
            confidence_debug=confidence_debug,
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
    ) -> tuple[List[RetrievalCandidate], str]:
        if not candidates:
            return [], "local"

        if config.rerank_enabled and reranker_service.is_online_enabled():
            try:
                return reranker_service.rerank_with_cohere(analysis.primary_query, candidates), "cohere"
            except Exception as exc:
                logger.warning(f"在线重排失败，回退到本地重排: {exc}")

        return self._local_rerank_candidates(analysis, candidates), "local"

    def _local_rerank_candidates(
        self,
        analysis: QueryUnderstandingResult,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        max_fused_score = max((candidate.fused_score for candidate in candidates), default=0.0) or 1.0

        for candidate in candidates:
            overlap = self._token_overlap_ratio(analysis.keywords, candidate.content)
            metadata_text = " ".join(
                str(candidate.metadata.get(key, "")) for key in ("section_path", "_file_name", "page_number")
            )
            metadata_overlap = self._token_overlap_ratio(analysis.keywords, metadata_text)
            fused_score = min(1.0, candidate.fused_score / max_fused_score)

            local_score = (
                0.35 * fused_score
                + 0.25 * self._clip_score(candidate.vector_score)
                + 0.20 * self._clip_score(candidate.keyword_score)
                + 0.15 * overlap
                + 0.05 * metadata_overlap
            )
            candidate.raw_rerank_score = local_score
            candidate.rerank_score = self._clip_score(local_score)
            candidate.rerank_source = "local"

        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)

    def _label_document_confidences(self, candidates: List[RetrievalCandidate]) -> None:
        for candidate in candidates:
            candidate.document_confidence = self._score_to_label(
                candidate.rerank_score,
                high_threshold=config.rag_document_confidence_threshold_high,
                low_threshold=config.rag_document_confidence_threshold_low,
            )

    def _evaluate_confidence(
        self,
        candidates: List[RetrievalCandidate],
        rerank_provider: str,
    ) -> tuple[str, str, Dict[str, object]]:
        if not candidates:
            debug = {
                "provider": rerank_provider,
                "reason": "未检索到任何相关片段",
                "top1Score": 0.0,
                "top2Score": 0.0,
                "avgTop3Score": 0.0,
                "supportCount": 0,
                "strongSupportCount": 0,
                "candidates": [],
            }
            return "low", "未检索到任何相关片段", debug

        top_candidates = candidates[:3]
        top1 = top_candidates[0].rerank_score
        top2 = top_candidates[1].rerank_score if len(top_candidates) > 1 else 0.0
        avg_top3 = sum(candidate.rerank_score for candidate in top_candidates) / len(top_candidates)
        support_count = len(
            [
                candidate
                for candidate in top_candidates
                if candidate.rerank_score >= config.rag_document_confidence_threshold_low
            ]
        )
        strong_support_count = len(
            [
                candidate
                for candidate in top_candidates
                if candidate.rerank_score >= config.rag_document_confidence_threshold_high
            ]
        )

        if top1 >= config.rag_confidence_threshold_high and support_count >= 2 and avg_top3 >= 0.55:
            confidence = "high"
            reason = ""
        elif top1 >= config.rag_confidence_threshold_low and support_count >= 1:
            confidence = "medium"
            reason = (
                "证据具备一定相关性，但强支撑不足"
                f": top1={top1:.3f}, support={support_count}, avg_top3={avg_top3:.3f}"
            )
        else:
            confidence = "low"
            reason = (
                "证据相关性偏弱"
                f": top1={top1:.3f}, top2={top2:.3f}, support={support_count}, avg_top3={avg_top3:.3f}"
            )

        debug = {
            "provider": rerank_provider,
            "reason": reason,
            "top1Score": round(top1, 4),
            "top2Score": round(top2, 4),
            "avgTop3Score": round(avg_top3, 4),
            "supportCount": support_count,
            "strongSupportCount": strong_support_count,
            "thresholds": {
                "overallHigh": config.rag_confidence_threshold_high,
                "overallLow": config.rag_confidence_threshold_low,
                "documentHigh": config.rag_document_confidence_threshold_high,
                "documentLow": config.rag_document_confidence_threshold_low,
            },
            "candidates": [self._build_candidate_debug(candidate) for candidate in candidates[:5]],
        }
        logger.info(
            "置信度评估完成: provider={}, overall={}, top1={}, top2={}, avg_top3={}, support={}",
            rerank_provider,
            confidence,
            round(top1, 4),
            round(top2, 4),
            round(avg_top3, 4),
            support_count,
        )
        return confidence, reason, debug

    def _build_reference(self, candidate: RetrievalCandidate) -> Dict[str, object]:
        metadata = candidate.metadata
        return {
            "file_name": metadata.get("_file_name", "未知来源"),
            "page_number": metadata.get("page_number"),
            "section_path": metadata.get("section_path"),
            "score": round(candidate.rerank_score, 4),
            "raw_score": round(candidate.raw_rerank_score, 4),
            "confidence": candidate.document_confidence,
            "rerank_source": candidate.rerank_source,
        }

    def _build_candidate_debug(self, candidate: RetrievalCandidate) -> Dict[str, object]:
        return {
            "id": candidate.id,
            "fileName": candidate.metadata.get("_file_name", "未知来源"),
            "pageNumber": candidate.metadata.get("page_number"),
            "sectionPath": candidate.metadata.get("section_path"),
            "vectorScore": round(candidate.vector_score, 4),
            "keywordScore": round(candidate.keyword_score, 4),
            "fusedScore": round(candidate.fused_score, 4),
            "rawRerankScore": round(candidate.raw_rerank_score, 4),
            "rerankScore": round(candidate.rerank_score, 4),
            "documentConfidence": candidate.document_confidence,
            "rerankSource": candidate.rerank_source,
            "matchedQueries": list(candidate.matched_queries),
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

    def _clip_score(self, score: float) -> float:
        return max(0.0, min(1.0, float(score)))

    def _score_to_label(self, score: float, high_threshold: float, low_threshold: float) -> str:
        if score >= high_threshold:
            return "high"
        if score >= low_threshold:
            return "medium"
        return "low"


hybrid_retrieval_service = HybridRetrievalService()
