"""在线/本地重排辅助服务。"""

from __future__ import annotations

from typing import Any, Iterable, List

import httpx
from loguru import logger

from app.config import config


class RerankerService:
    """封装在线 reranker 请求。"""

    def is_online_enabled(self) -> bool:
        provider = config.rerank_provider.strip().lower()
        return bool(config.rerank_enabled and provider == "cohere" and config.cohere_api_key.strip())

    def rerank_with_cohere(self, query: str, candidates: Iterable[Any]) -> List[Any]:
        candidate_list = list(candidates)
        if not candidate_list:
            return []
        if not config.cohere_api_key.strip():
            raise RuntimeError("Cohere API Key 未配置")

        payload = {
            "model": config.cohere_rerank_model,
            "query": query,
            "documents": [self._build_document_text(item) for item in candidate_list],
            "top_n": len(candidate_list),
        }
        headers = {
            "Authorization": f"Bearer {config.cohere_api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "开始调用 Cohere Rerank: model={}, candidates={}, query_len={}",
            config.cohere_rerank_model,
            len(candidate_list),
            len(query),
        )

        response = httpx.post(
            config.cohere_rerank_url,
            headers=headers,
            json=payload,
            timeout=config.rerank_request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not isinstance(results, list) or not results:
            raise RuntimeError("Cohere Rerank 返回为空")

        reranked: List[Any] = []
        seen_indexes: set[int] = set()
        for item in results:
            index = int(item.get("index", -1))
            if index < 0 or index >= len(candidate_list):
                continue

            candidate = candidate_list[index]
            raw_score = float(item.get("relevance_score", 0.0))
            candidate.raw_rerank_score = raw_score
            candidate.rerank_score = self.normalize_score(raw_score)
            candidate.rerank_source = "cohere"
            reranked.append(candidate)
            seen_indexes.add(index)

        for index, candidate in enumerate(candidate_list):
            if index in seen_indexes:
                continue
            candidate.raw_rerank_score = 0.0
            candidate.rerank_score = 0.0
            candidate.rerank_source = "cohere"
            reranked.append(candidate)

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        logger.info(
            "Cohere Rerank 调用完成: top_scores={}",
            [round(item.rerank_score, 4) for item in reranked[:5]],
        )
        return reranked

    def normalize_score(self, score: float) -> float:
        """将在线 reranker 分数约束到 [0, 1] 便于统一置信度判断。"""
        return max(0.0, min(1.0, float(score)))

    def _build_document_text(self, candidate: Any) -> str:
        metadata = getattr(candidate, "metadata", {}) or {}
        metadata_parts = [
            str(metadata.get("_file_name", "")).strip(),
            str(metadata.get("section_path", "")).strip(),
            f"第{metadata.get('page_number')}页" if metadata.get("page_number") else "",
        ]
        prefix = " | ".join([item for item in metadata_parts if item])
        content = str(getattr(candidate, "content", "")).strip()
        return f"{prefix}\n\n{content}" if prefix else content


reranker_service = RerankerService()
