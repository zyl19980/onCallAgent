"""本地知识语料服务，用于 BM25 与待补充审计。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from loguru import logger

from app.config import config


class KnowledgeCorpusService:
    """维护本地 chunk 语料文件。"""

    def __init__(self):
        self.default_corpus_path = Path(config.rag_corpus_path)
        self.default_corpus_path.parent.mkdir(parents=True, exist_ok=True)

    def replace_documents_for_source(
        self,
        source: str,
        documents: List[Document],
        collection_name: str | None = None,
    ) -> None:
        corpus = [
            item for item in self.load_corpus(collection_name)
            if item.get("_source") != source
        ]

        for doc in documents:
            metadata = dict(doc.metadata)
            corpus.append(
                {
                    "content": doc.page_content,
                    **metadata,
                }
            )

        self._write_corpus(corpus, collection_name)
        logger.info(
            f"本地语料已更新: source={source}, collection={collection_name or config.rag_collection_name}, chunk_count={len(documents)}"
        )

    def remove_source(self, source: str, collection_name: str | None = None) -> None:
        corpus = [
            item for item in self.load_corpus(collection_name)
            if item.get("_source") != source
        ]
        self._write_corpus(corpus, collection_name)

    def upsert_chunk(
        self,
        collection_name: str,
        chunk_key: str,
        text: str,
        metadata: Dict[str, object],
    ) -> None:
        normalized_metadata = dict(metadata)
        normalized_metadata["chunk_id"] = chunk_key

        corpus = self.load_corpus(collection_name)
        for index, item in enumerate(corpus):
            if str(item.get("chunk_id")) != chunk_key:
                continue
            merged_metadata = {key: value for key, value in item.items() if key != "content"}
            merged_metadata.update(normalized_metadata)
            corpus[index] = {
                "content": text,
                **merged_metadata,
            }
            self._write_corpus(corpus, collection_name)
            logger.info(
                f"本地语料单 chunk 已更新: collection={collection_name}, chunk_key={chunk_key}"
            )
            return

        corpus.append(
            {
                "content": text,
                **normalized_metadata,
            }
        )
        self._write_corpus(corpus, collection_name)
        logger.info(
            f"本地语料单 chunk 已新增: collection={collection_name}, chunk_key={chunk_key}"
        )

    def get_corpus_path(self, collection_name: str | None = None) -> Path:
        target = collection_name or config.rag_collection_name
        if target == config.rag_collection_name:
            return self.default_corpus_path

        sanitized = target.replace("/", "_").replace(" ", "_")
        return self.default_corpus_path.with_name(f"{self.default_corpus_path.stem}_{sanitized}.jsonl")

    def load_corpus(self, collection_name: str | None = None) -> List[Dict[str, object]]:
        corpus_path = self.get_corpus_path(collection_name)
        if not corpus_path.exists():
            return []

        corpus: List[Dict[str, object]] = []
        with corpus_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    corpus.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("跳过损坏的本地语料行")
        return corpus

    def _write_corpus(self, corpus: List[Dict[str, object]], collection_name: str | None = None) -> None:
        corpus_path = self.get_corpus_path(collection_name)
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with corpus_path.open("w", encoding="utf-8") as file:
            for item in corpus:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")


knowledge_corpus_service = KnowledgeCorpusService()
