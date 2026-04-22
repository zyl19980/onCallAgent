import pytest

from app.config import config
from app.services.hybrid_retrieval_service import (
    HybridRetrievalService,
    QueryUnderstandingResult,
    RetrievalCandidate,
    RetrievalResult,
)
from app.services.reranker_service import reranker_service
from app.tools.knowledge_tool import retrieve_knowledge


def test_local_rerank_assigns_document_confidence_and_high_overall_confidence(monkeypatch):
    monkeypatch.setattr(reranker_service, "is_online_enabled", lambda: False)

    service = HybridRetrievalService()
    analysis = QueryUnderstandingResult(
        primary_query="ABB 报码如何处理",
        keyword_query="abb 报码 处理",
        expanded_queries=["ABB 报码如何处理"],
        keywords=["abb", "报码", "处理"],
    )
    candidates = [
        RetrievalCandidate(
            id="doc-1",
            content="ABB 控制柜报码后请先处理电源并检查报警历史。",
            metadata={"_file_name": "a.md", "section_path": "报码处理", "page_number": 1},
            vector_score=0.95,
            keyword_score=0.90,
            fused_score=0.030,
        ),
        RetrievalCandidate(
            id="doc-2",
            content="ABB 维修流程：先确认报码，再执行处理步骤。",
            metadata={"_file_name": "b.md", "section_path": "处理步骤", "page_number": 2},
            vector_score=0.90,
            keyword_score=0.80,
            fused_score=0.029,
        ),
        RetrievalCandidate(
            id="doc-3",
            content="一般说明文档，与当前处理关系较弱。",
            metadata={"_file_name": "c.md", "section_path": "正文", "page_number": 3},
            vector_score=0.20,
            keyword_score=0.15,
            fused_score=0.010,
        ),
    ]

    reranked, provider = service._rerank_candidates(analysis, candidates)
    top_candidates = reranked[:3]
    service._label_document_confidences(top_candidates)
    confidence, reason, debug = service._evaluate_confidence(top_candidates, provider)

    assert provider == "local"
    assert top_candidates[0].document_confidence == "high"
    assert top_candidates[1].document_confidence == "high"
    assert top_candidates[2].document_confidence == "low"
    assert confidence == "high"
    assert reason == ""
    assert debug["supportCount"] >= 2


def test_cohere_rerank_normalizes_scores_and_orders_candidates(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.52},
                ]
            }

    def fake_post(url, headers, json, timeout):
        assert url == config.cohere_rerank_url
        assert json["model"] == config.cohere_rerank_model
        assert json["query"] == "如何处理报码"
        assert len(json["documents"]) == 2
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr(config, "cohere_api_key", "test-key")

    candidates = [
        RetrievalCandidate(id="doc-1", content="doc1", metadata={"_file_name": "a.md"}),
        RetrievalCandidate(id="doc-2", content="doc2", metadata={"_file_name": "b.md"}),
    ]

    reranked = reranker_service.rerank_with_cohere("如何处理报码", candidates)

    assert [item.id for item in reranked] == ["doc-2", "doc-1"]
    assert reranked[0].rerank_source == "cohere"
    assert reranked[0].rerank_score == pytest.approx(0.91)
    assert reranked[1].rerank_score == pytest.approx(0.52)


def test_retrieve_knowledge_tool_outputs_rag_debug(monkeypatch):
    retrieval = RetrievalResult(
        query_analysis=QueryUnderstandingResult(
            primary_query="E01 怎么处理",
            keyword_query="E01 处理",
            expanded_queries=["E01 怎么处理"],
            keywords=["e01", "处理"],
        ),
        candidates=[
            RetrievalCandidate(
                id="doc-1",
                content="主板报码 E01，建议检查电源模块。",
                metadata={"_file_name": "a.md", "section_path": "报码处理", "page_number": 1},
                raw_rerank_score=0.82,
                rerank_score=0.82,
                rerank_source="cohere",
                document_confidence="high",
            )
        ],
        references=[
            {
                "file_name": "a.md",
                "page_number": 1,
                "section_path": "报码处理",
                "score": 0.82,
                "raw_score": 0.82,
                "confidence": "high",
                "rerank_source": "cohere",
            }
        ],
        confidence="high",
        queued_for_supplement=False,
        low_confidence_reason="",
        rerank_provider="cohere",
        confidence_debug={
            "top1Score": 0.82,
            "top2Score": 0.0,
            "avgTop3Score": 0.82,
            "supportCount": 1,
        },
    )

    monkeypatch.setattr("app.tools.knowledge_tool.hybrid_retrieval_service.retrieve", lambda query: retrieval)

    text, docs = retrieve_knowledge.func("E01 怎么处理")

    assert "检索整体置信度: high" in text
    assert "重排来源: cohere" in text
    assert "文档置信度: high" in text
    assert docs[0].metadata["document_confidence"] == "high"
