"""基于本地语料的 BM25 检索服务。"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from loguru import logger

from app.services.knowledge_corpus_service import knowledge_corpus_service


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]")


@dataclass
class Bm25SearchResult:
    """BM25 检索结果。"""

    id: str
    content: str
    score: float
    metadata: Dict[str, object]


class BM25SearchService:
    """简单 BM25 实现。"""

    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name
        self._cache_mtime = 0.0
        self._corpus: List[Dict[str, object]] = []
        self._tokenized_corpus: List[List[str]] = []
        self._doc_len: List[int] = []
        self._doc_freq: Dict[str, int] = {}
        self._avg_doc_len = 0.0
        self.k1 = 1.5
        self.b = 0.75

    def search(self, query: str, top_k: int = 20) -> List[Bm25SearchResult]:
        self._ensure_index()
        if not self._corpus or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: List[tuple[int, float]] = []
        for index, doc_tokens in enumerate(self._tokenized_corpus):
            token_counts = Counter(doc_tokens)
            doc_len = self._doc_len[index]
            score = 0.0

            for token in query_tokens:
                if token not in token_counts:
                    continue
                df = self._doc_freq.get(token, 0)
                idf = math.log(1 + (len(self._tokenized_corpus) - df + 0.5) / (df + 0.5))
                tf = token_counts[token]
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1))
                score += idf * ((tf * (self.k1 + 1)) / max(denominator, 1e-9))

            if score > 0:
                scores.append((index, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        results: List[Bm25SearchResult] = []
        for index, score in scores[:top_k]:
            item = self._corpus[index]
            metadata = {key: value for key, value in item.items() if key != "content"}
            result_id = str(metadata.get("chunk_id") or f"{metadata.get('_source', 'local')}::{metadata.get('chunk_index', index)}")
            results.append(
                Bm25SearchResult(
                    id=result_id,
                    content=str(item.get("content", "")),
                    score=score,
                    metadata=metadata,
                )
            )
        return results

    def _ensure_index(self) -> None:
        corpus_path = Path(knowledge_corpus_service.get_corpus_path(self.collection_name))
        mtime = corpus_path.stat().st_mtime if corpus_path.exists() else 0.0
        if mtime <= self._cache_mtime:
            return

        self._corpus = knowledge_corpus_service.load_corpus(self.collection_name)
        self._tokenized_corpus = [self._tokenize(str(item.get("content", ""))) for item in self._corpus]
        self._doc_len = [len(tokens) for tokens in self._tokenized_corpus]
        self._avg_doc_len = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0

        doc_freq: Dict[str, int] = defaultdict(int)
        for tokens in self._tokenized_corpus:
            for token in set(tokens):
                doc_freq[token] += 1
        self._doc_freq = dict(doc_freq)
        self._cache_mtime = mtime

        logger.info(f"BM25 语料已刷新: 文档数={len(self._corpus)}")

    def _tokenize(self, text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text.lower())

bm25_search_service = BM25SearchService()
