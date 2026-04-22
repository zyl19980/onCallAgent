"""规则化问题指纹服务。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.hybrid_retrieval_service import QueryUnderstandingResult


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_TOKEN_PATTERN = re.compile(r"[^\w\u4e00-\u9fff]+")


@dataclass(frozen=True)
class QueryFingerprintResult:
    normalized_query: str
    query_fingerprint: str


class QueryFingerprintService:
    """基于规则生成稳定问题指纹。"""

    PHRASE_REPLACEMENTS = [
        ("怎么办", " handle "),
        ("怎么处理", " handle "),
        ("如何处理", " handle "),
        ("咋处理", " handle "),
        ("报码", " alarm "),
        ("报错", " alarm "),
        ("告警", " alarm "),
        ("排查", " troubleshoot "),
        ("诊断", " troubleshoot "),
    ]

    TOKEN_CANONICAL_MAP = {
        "handle": "handle",
        "alarm": "alarm",
        "troubleshoot": "troubleshoot",
        "报码": "alarm",
        "报错": "alarm",
        "告警": "alarm",
        "怎么处理": "handle",
        "如何处理": "handle",
        "怎么办": "handle",
        "排查": "troubleshoot",
        "诊断": "troubleshoot",
    }

    STOPWORDS = {
        "请问", "一下", "这个", "那个", "问题", "可以", "需要", "是否",
        "怎么", "如何", "一下子", "我想", "帮忙", "请", "下",
    }

    def build(
        self,
        raw_query: str,
        query_analysis: QueryUnderstandingResult,
    ) -> QueryFingerprintResult:
        base_query = query_analysis.primary_query or raw_query
        normalized_query = self._normalize_text(base_query)
        tokens = self._collect_tokens(normalized_query, query_analysis)
        fingerprint = "|".join(tokens) if tokens else normalized_query.replace(" ", "_")
        return QueryFingerprintResult(
            normalized_query=normalized_query,
            query_fingerprint=fingerprint,
        )

    def _collect_tokens(
        self,
        normalized_query: str,
        query_analysis: QueryUnderstandingResult,
    ) -> list[str]:
        candidates: list[str] = []
        candidates.extend(self._tokenize(normalized_query))
        candidates.extend(self._tokenize(query_analysis.keyword_query))
        candidates.extend(self._canonicalize_token(keyword) for keyword in query_analysis.keywords)

        stable_tokens: list[str] = []
        for token in candidates:
            canonical = self._canonicalize_token(token)
            if not canonical or canonical in self.STOPWORDS:
                continue
            if canonical not in stable_tokens:
                stable_tokens.append(canonical)

        return sorted(stable_tokens)

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower().strip()
        for original, replacement in self.PHRASE_REPLACEMENTS:
            normalized = normalized.replace(original, replacement)
        normalized = NON_TOKEN_PATTERN.sub(" ", normalized)
        normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
        return normalized

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return [self._canonicalize_token(token) for token in TOKEN_PATTERN.findall(text.lower())]

    def _canonicalize_token(self, token: str) -> str:
        clean = token.strip().lower()
        if not clean:
            return ""
        return self.TOKEN_CANONICAL_MAP.get(clean, clean)


query_fingerprint_service = QueryFingerprintService()
