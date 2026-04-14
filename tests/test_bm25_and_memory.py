import asyncio
import json

from app.services.bm25_search_service import BM25SearchService
from app.services.conversation_memory_service import ConversationMemoryService
from app.services.knowledge_corpus_service import knowledge_corpus_service


def test_bm25_search_hits_expected_document(tmp_path):
    corpus_path = tmp_path / "rag_corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "_source": "/tmp/a.md",
                        "_file_name": "a.md",
                        "chunk_index": 0,
                        "content": "主板报码 E01，建议检查电源模块和保险丝",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "_source": "/tmp/b.md",
                        "_file_name": "b.md",
                        "chunk_index": 1,
                        "content": "CPU 使用率过高时，先检查热点进程",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    original_path = knowledge_corpus_service.default_corpus_path
    knowledge_corpus_service.default_corpus_path = corpus_path
    try:
        service = BM25SearchService()
        results = service.search("E01 如何处理", top_k=5)
        assert results
        assert results[0].metadata["_file_name"] == "a.md"
    finally:
        knowledge_corpus_service.default_corpus_path = original_path


def test_collection_specific_corpus_path(tmp_path):
    original_path = knowledge_corpus_service.default_corpus_path
    knowledge_corpus_service.default_corpus_path = tmp_path / "rag_corpus.jsonl"
    try:
        path = knowledge_corpus_service.get_corpus_path("machine_repair_pdf")
        assert path.name == "rag_corpus_machine_repair_pdf.jsonl"
    finally:
        knowledge_corpus_service.default_corpus_path = original_path


def test_memory_service_summarizes_overflow():
    service = ConversationMemoryService()
    service.window_rounds = 1
    service.max_messages = 2

    async def fake_summarizer(prompt: str) -> str:
        return "摘要: 设备报码 E01，之前已经检查过电源。"

    asyncio.run(service.append_exchange("s1", "第一问", "第一答", fake_summarizer))
    asyncio.run(service.append_exchange("s1", "第二问", "第二答", fake_summarizer))

    assert service.get_summary("s1").startswith("摘要:")
    assert len(service.get_history("s1")) == 2
