import csv
import json
from pathlib import Path

from scripts.experiment.evaluate_rag_retrieval import (
    build_rerank_adapter,
    diagnose_live_retrieval,
    evaluate_rag_retrieval,
    extract_live_chunk_id,
    validate_retrieval_config,
)


class StaticAdapter:
    def __init__(self, mapping):
        self.mapping = mapping

    def retrieve(self, sample, *, top_k, collection_filter_from_sample, source_filter_from_sample):
        rows = self.mapping[sample["id"]]
        output = []
        for rank, item in enumerate(rows[:top_k], start=1):
            enriched = dict(item)
            enriched.setdefault("rank", rank)
            output.append(enriched)
        return output


class DiagnosticAdapter:
    def __init__(self, mapping):
        self.mapping = mapping

    def retrieve_diagnostic(
        self,
        sample,
        *,
        top_k,
        collection_filter_from_sample,
        source_filter_from_sample,
    ):
        return self.mapping[sample["id"]]


class FakeCurrentRerankAdapter:
    def rerank(self, *, sample, candidate_results, final_top_k):
        reordered = []
        ranking = ["chunk-2", "chunk-1", "chunk-3"]
        source_by_id = {item["chunk_id"]: item for item in candidate_results}
        for rerank_rank, chunk_id in enumerate(ranking, start=1):
            original = source_by_id[chunk_id]
            reordered.append(
                {
                    **dict(original),
                    "rank": rerank_rank,
                    "original_rank": original["rank"],
                    "rerank_rank": rerank_rank,
                    "original_score": float(original["score"]),
                    "rerank_score": 1.0 - (rerank_rank * 0.1),
                    "score": 1.0 - (rerank_rank * 0.1),
                    "rerank_provider": "fake",
                }
            )
        return {
            "final_results": reordered[:final_top_k],
            "reranked_results": reordered,
            "rerank_available": True,
            "rerank_errors": [],
        }


class FakeHybridLiveAdapter:
    def __init__(self, chunks_by_id):
        self.chunks_by_id = chunks_by_id

    def retrieve(self, sample, *, top_k, collection_filter_from_sample, source_filter_from_sample):
        del collection_filter_from_sample, source_filter_from_sample
        return [
            {
                "rank": 1,
                "chunk_id": "chunk-h1",
                "source_id": "source_h",
                "source_file": "source_h.pdf",
                "page_start": 5,
                "page_end": 5,
                "score": 0.91,
                "vector_score": 0.8,
                "keyword_score": 0.6,
                "fused_score": 0.91,
            },
            {
                "rank": 2,
                "chunk_id": "chunk-hx",
                "source_id": "source_x",
                "source_file": "source_x.pdf",
                "page_start": 8,
                "page_end": 8,
                "score": 0.5,
                "vector_score": 0.3,
                "keyword_score": 0.4,
                "fused_score": 0.5,
            },
        ][:top_k]


def test_evaluate_rag_retrieval_mock_mode_runs_and_writes_summary(tmp_path: Path):
    dataset_path = tmp_path / "rag_build.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "results.json"
    summary_path = tmp_path / "summary.csv"

    dataset_rows = [
        make_sample("s1", ["chunk-a"], [3], "source_a", "troubleshooting_step"),
        make_sample("s2", ["chunk-b"], [7], "source_b", "parameter_or_fault_code"),
    ]
    chunk_rows = [
        make_chunk("chunk-a", "source_a", "source_a.pdf", 3, 3),
        make_chunk("chunk-b", "source_b", "source_b.pdf", 7, 7),
        make_chunk("chunk-c", "source_c", "source_c.pdf", 9, 9),
    ]
    write_jsonl(dataset_path, dataset_rows)
    write_jsonl(chunks_path, chunk_rows)

    report = evaluate_rag_retrieval(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=output_path,
        summary_path=summary_path,
        mode="mock",
        candidate_top_k=3,
        final_top_k=3,
        retrieval_strategy="dense",
        rerank="none",
        experiment_name="mock_smoke",
        ks=[1, 3],
        mock_miss_rate=0.0,
        seed=42,
    )

    assert report["mode"] == "mock"
    assert report["experiment_name"] == "mock_smoke"
    assert report["total_samples"] == 2
    assert report["evaluated_samples"] == 2
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert "candidate_top_k" in rows[0]
    assert "candidate_hit_at_10" in rows[0]
    assert "hit_at_1" in rows[0]


def test_validate_retrieval_config_supports_dense_and_hybrid_with_none_and_current():
    for retrieval_strategy in ("dense", "hybrid"):
        for rerank in ("none", "current"):
            validate_retrieval_config(
                mode="live",
                retrieval_strategy=retrieval_strategy,
                rerank=rerank,
                candidate_top_k=50,
                final_top_k=10,
            )


def test_evaluate_rag_retrieval_metrics_and_warnings(tmp_path: Path):
    dataset_path = tmp_path / "rag_build.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "results.json"
    summary_path = tmp_path / "summary.csv"

    dataset_rows = [
        make_sample("q1", ["chunk-1", "chunk-2"], [3, 4], "source_a", "troubleshooting_step"),
        make_sample("q2", ["chunk-3"], [9], "source_b", "parameter_or_fault_code"),
        make_sample(
            "q3",
            [],
            [],
            "source_c",
            "abstention_insufficient_evidence",
            should_abstain=True,
        ),
    ]
    chunk_rows = [
        make_chunk("chunk-1", "source_a", "source_a.pdf", 3, 3),
        make_chunk("chunk-2", "source_a", "source_a.pdf", 4, 4),
        make_chunk("chunk-3", "source_b", "source_b.pdf", 9, 9),
        make_chunk("chunk-x", "source_x", "source_x.pdf", 11, 11),
    ]
    write_jsonl(dataset_path, dataset_rows)
    write_jsonl(chunks_path, chunk_rows)

    adapter = StaticAdapter(
        {
            "q1": [
                make_retrieved("chunk-1", "source_a", "source_a.pdf", 3, 3, 0.9),
                make_retrieved("chunk-x", "source_x", "source_x.pdf", 11, 11, 0.8),
                make_retrieved("chunk-2", "source_a", "source_a.pdf", 4, 4, 0.7),
            ],
            "q2": [
                {"rank": 1, "chunk_id": "", "source_id": "", "source_file": "", "page_start": None, "page_end": None, "score": 0.5},
                make_retrieved("chunk-x", "source_x", "source_x.pdf", 11, 11, 0.4),
                make_retrieved("chunk-3", "source_b", "source_b.pdf", 9, 9, 0.3),
            ],
        }
    )

    report = evaluate_rag_retrieval(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=output_path,
        summary_path=summary_path,
        mode="mock",
        candidate_top_k=3,
        final_top_k=3,
        ks=[1, 3],
        adapter=adapter,
    )

    assert report["skipped_abstain"] == 1
    assert report["evaluated_samples"] == 2
    assert report["candidate_metrics"]["candidate_hit_at_10"] == 1.0
    assert report["candidate_metrics"]["candidate_recall_at_10"] == 1.0
    assert report["final_metrics"]["hit_at_1"] == 0.5
    assert report["final_metrics"]["hit_at_3"] == 1.0
    assert report["final_metrics"]["recall_at_1"] == 0.25
    assert report["final_metrics"]["recall_at_3"] == 1.0
    assert report["final_metrics"]["mrr"] == 0.75
    assert report["final_metrics"]["evidence_coverage_at_1"] == 0.0
    assert report["final_metrics"]["evidence_coverage_at_3"] == 1.0
    assert any(item.startswith("missing_chunk_id_count:1") for item in report["warnings"])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    sample_by_id = {row["id"]: row for row in payload["per_sample"]}
    assert sample_by_id["q1"]["first_relevant_rank"] == 1
    assert sample_by_id["q1"]["evidence_coverage_at_k"]["3"] == 1.0
    assert sample_by_id["q2"]["mrr"] == 0.5
    assert "missing_chunk_id_in_retrieved" in sample_by_id["q2"]["errors"]
    assert len(sample_by_id["q1"]["candidate_results"]) == 3
    assert len(sample_by_id["q1"]["final_results"]) == 3


def test_candidate_top_k_final_top_k_and_top_k_compatibility(tmp_path: Path):
    dataset_path = tmp_path / "rag_build.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "results.json"
    summary_path = tmp_path / "summary.csv"

    dataset_rows = [make_sample("q1", ["chunk-2"], [4], "source_a", "troubleshooting_step")]
    chunk_rows = [
        make_chunk("chunk-1", "source_a", "source_a.pdf", 3, 3),
        make_chunk("chunk-2", "source_a", "source_a.pdf", 4, 4),
        make_chunk("chunk-3", "source_a", "source_a.pdf", 5, 5),
    ]
    write_jsonl(dataset_path, dataset_rows)
    write_jsonl(chunks_path, chunk_rows)

    adapter = StaticAdapter(
        {
            "q1": [
                make_retrieved("chunk-1", "source_a", "source_a.pdf", 3, 3, 0.9),
                make_retrieved("chunk-2", "source_a", "source_a.pdf", 4, 4, 0.8),
                make_retrieved("chunk-3", "source_a", "source_a.pdf", 5, 5, 0.7),
            ]
        }
    )

    report = evaluate_rag_retrieval(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=output_path,
        summary_path=summary_path,
        mode="mock",
        candidate_top_k=3,
        final_top_k=2,
        rerank="none",
        adapter=adapter,
    )

    row = report["per_sample"][0]
    assert report["candidate_top_k"] == 3
    assert report["final_top_k"] == 2
    assert len(row["candidate_results"]) == 3
    assert len(row["final_results"]) == 2
    assert [item["chunk_id"] for item in row["final_results"]] == [
        item["chunk_id"] for item in row["candidate_results"][:2]
    ]
    assert row["final_results"][0]["original_rank"] == 1
    assert row["final_results"][1]["rerank_rank"] == 2
    assert report["candidate_metrics"]["candidate_hit_at_10"] == 1.0
    assert report["final_metrics"]["hit_at_1"] == 0.0
    assert report["final_metrics"]["hit_at_3"] == 1.0

    legacy_report = evaluate_rag_retrieval(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=tmp_path / "legacy_results.json",
        summary_path=tmp_path / "legacy_summary.csv",
        mode="mock",
        top_k=2,
        rerank="none",
        adapter=adapter,
    )
    assert legacy_report["candidate_top_k"] == 2
    assert legacy_report["final_top_k"] == 2


def test_rerank_current_reorders_and_tracks_gold_promotion(tmp_path: Path):
    dataset_path = tmp_path / "rag_build.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "results.json"
    summary_path = tmp_path / "summary.csv"

    dataset_rows = [make_sample("q1", ["chunk-2"], [4], "source_a", "troubleshooting_step")]
    chunk_rows = [
        make_chunk("chunk-1", "source_a", "source_a.pdf", 3, 3),
        make_chunk("chunk-2", "source_a", "source_a.pdf", 4, 4),
        make_chunk("chunk-3", "source_a", "source_a.pdf", 5, 5),
    ]
    write_jsonl(dataset_path, dataset_rows)
    write_jsonl(chunks_path, chunk_rows)

    adapter = StaticAdapter(
        {
            "q1": [
                make_retrieved("chunk-1", "source_a", "source_a.pdf", 3, 3, 0.9),
                make_retrieved("chunk-2", "source_a", "source_a.pdf", 4, 4, 0.8),
                make_retrieved("chunk-3", "source_a", "source_a.pdf", 5, 5, 0.7),
            ]
        }
    )

    report = evaluate_rag_retrieval(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=output_path,
        summary_path=summary_path,
        mode="mock",
        candidate_top_k=3,
        final_top_k=2,
        rerank="current",
        adapter=adapter,
        rerank_adapter=FakeCurrentRerankAdapter(),
    )

    row = report["per_sample"][0]
    assert report["rerank_available"] is True
    assert report["rerank_errors"] == []
    assert report["gold_promoted_by_rerank_count"] == 1
    assert report["gold_demoted_by_rerank_count"] == 0
    assert report["gold_in_candidate_not_final_count"] == 0
    assert row["final_results"][0]["chunk_id"] == "chunk-2"
    assert row["final_results"][0]["original_rank"] == 2
    assert row["final_results"][0]["rerank_rank"] == 1
    assert row["gold_promoted_by_rerank"] is True
    assert report["final_metrics"]["hit_at_1"] == 1.0


def test_evaluate_rag_retrieval_live_hybrid_uses_hybrid_adapter(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "rag_build.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "results.json"
    summary_path = tmp_path / "summary.csv"

    dataset_rows = [make_sample("q1", ["chunk-h1"], [5], "source_h", "troubleshooting_step")]
    chunk_rows = [
        make_chunk("chunk-h1", "source_h", "source_h.pdf", 5, 5),
        make_chunk("chunk-hx", "source_x", "source_x.pdf", 8, 8),
    ]
    write_jsonl(dataset_path, dataset_rows)
    write_jsonl(chunks_path, chunk_rows)

    monkeypatch.setattr(
        "scripts.experiment.evaluate_rag_retrieval.HybridRetrievalAdapter",
        FakeHybridLiveAdapter,
    )

    report = evaluate_rag_retrieval(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=output_path,
        summary_path=summary_path,
        mode="live",
        candidate_top_k=2,
        final_top_k=1,
        retrieval_strategy="hybrid",
        rerank="none",
    )

    assert report["retrieval_strategy"] == "hybrid"
    assert report["candidate_metrics"]["candidate_hit_at_10"] == 1.0
    assert report["final_metrics"]["hit_at_1"] == 1.0
    assert report["per_sample"][0]["candidate_results"][0]["chunk_id"] == "chunk-h1"


def test_rerank_current_tracks_gold_demotion_and_candidate_not_final(tmp_path: Path):
    dataset_path = tmp_path / "rag_build.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "results.json"
    summary_path = tmp_path / "summary.csv"

    dataset_rows = [make_sample("q1", ["chunk-1"], [3], "source_a", "troubleshooting_step")]
    chunk_rows = [
        make_chunk("chunk-1", "source_a", "source_a.pdf", 3, 3),
        make_chunk("chunk-2", "source_a", "source_a.pdf", 4, 4),
        make_chunk("chunk-3", "source_a", "source_a.pdf", 5, 5),
    ]
    write_jsonl(dataset_path, dataset_rows)
    write_jsonl(chunks_path, chunk_rows)

    adapter = StaticAdapter(
        {
            "q1": [
                make_retrieved("chunk-1", "source_a", "source_a.pdf", 3, 3, 0.9),
                make_retrieved("chunk-2", "source_a", "source_a.pdf", 4, 4, 0.8),
                make_retrieved("chunk-3", "source_a", "source_a.pdf", 5, 5, 0.7),
            ]
        }
    )

    class DemoteRerankAdapter:
        def rerank(self, *, sample, candidate_results, final_top_k):
            reordered = []
            ranking = ["chunk-2", "chunk-3", "chunk-1"]
            source_by_id = {item["chunk_id"]: item for item in candidate_results}
            for rerank_rank, chunk_id in enumerate(ranking, start=1):
                original = source_by_id[chunk_id]
                reordered.append(
                    {
                        **dict(original),
                        "rank": rerank_rank,
                        "original_rank": original["rank"],
                        "rerank_rank": rerank_rank,
                        "original_score": float(original["score"]),
                        "rerank_score": 1.0 - (rerank_rank * 0.1),
                        "score": 1.0 - (rerank_rank * 0.1),
                        "rerank_provider": "fake",
                    }
                )
            return {
                "final_results": reordered[:final_top_k],
                "reranked_results": reordered,
                "rerank_available": True,
                "rerank_errors": [],
            }

    report = evaluate_rag_retrieval(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=output_path,
        summary_path=summary_path,
        mode="mock",
        candidate_top_k=3,
        final_top_k=2,
        rerank="current",
        adapter=adapter,
        rerank_adapter=DemoteRerankAdapter(),
    )

    row = report["per_sample"][0]
    assert report["gold_promoted_by_rerank_count"] == 0
    assert report["gold_demoted_by_rerank_count"] == 1
    assert report["gold_in_candidate_not_final_count"] == 1
    assert row["gold_demoted_by_rerank"] is True
    assert row["gold_in_candidate_not_final"] is True


def test_current_rerank_unavailable_has_clear_error(monkeypatch):
    class BrokenCurrentRerankAdapter:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("current rerank unavailable: test failure")

    monkeypatch.setattr(
        "scripts.experiment.evaluate_rag_retrieval.CurrentRerankAdapter",
        BrokenCurrentRerankAdapter,
    )

    try:
        build_rerank_adapter(rerank="current", chunks_by_id={})
    except RuntimeError as exc:
        assert "current rerank unavailable" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_diagnose_live_retrieval_writes_output_and_respects_limit(tmp_path: Path):
    dataset_path = tmp_path / "rag_build.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "rag_live_diagnosis.json"

    dataset_rows = [
        make_sample("d1", ["chunk-1"], [3], "source_a", "troubleshooting_step"),
        make_sample("d2", ["chunk-2"], [7], "source_b", "parameter_or_fault_code"),
    ]
    chunk_rows = [
        make_chunk("chunk-1", "source_a", "source_a.pdf", 3, 3),
        make_chunk("chunk-2", "source_b", "source_b.pdf", 7, 7),
    ]
    write_jsonl(dataset_path, dataset_rows)
    write_jsonl(chunks_path, chunk_rows)

    adapter = DiagnosticAdapter(
        {
            "d1": {
                "raw_result_count": 2,
                "normalized_result_count": 1,
                "missing_chunk_id_count": 0,
                "errors": [],
                "normalized_results": [
                    {
                        "rank": 1,
                        "chunk_id": "chunk-1",
                        "source_id": "source_a",
                        "source_file": "source_a.pdf",
                        "page_start": 3,
                        "page_end": 3,
                        "score": 0.2,
                        "raw": {"metadata": {"chunk_id": "chunk-1", "section_path": "A > B"}},
                    }
                ],
            }
        }
    )

    report = diagnose_live_retrieval(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=output_path,
        top_k=5,
        limit=1,
        dump_raw_retrieval=True,
        adapter=adapter,
    )

    assert report["total_samples"] == 1
    assert report["missing_chunk_id_count"] == 0
    assert report["should_fail"] is False
    assert output_path.exists()
    record = report["records"][0]
    assert record["sample_id"] == "d1"
    assert record["hit_any_reference"] is True
    assert record["matched_reference_chunk_ids"] == ["chunk-1"]
    assert record["top_results"][0]["metadata_keys"] == ["chunk_id", "section_path"]
    assert '"chunk_id": "chunk-1"' in record["top_results"][0]["raw_preview"]


def test_diagnose_live_retrieval_missing_chunk_id_warns_and_can_fail(tmp_path: Path):
    dataset_path = tmp_path / "rag_build.jsonl"
    chunks_path = tmp_path / "experiment_chunks.jsonl"
    output_path = tmp_path / "rag_live_diagnosis.json"

    dataset_rows = [make_sample("d1", ["chunk-1"], [3], "source_a", "troubleshooting_step")]
    chunk_rows = [make_chunk("chunk-1", "source_a", "source_a.pdf", 3, 3)]
    write_jsonl(dataset_path, dataset_rows)
    write_jsonl(chunks_path, chunk_rows)

    adapter = DiagnosticAdapter(
        {
            "d1": {
                "raw_result_count": 2,
                "normalized_result_count": 2,
                "missing_chunk_id_count": 1,
                "errors": ["missing_chunk_id_in_retrieved"],
                "normalized_results": [
                    {
                        "rank": 1,
                        "chunk_id": "",
                        "source_id": "",
                        "source_file": "",
                        "page_start": None,
                        "page_end": None,
                        "score": 0.4,
                        "raw": {"metadata": {"source_id": "source_a"}},
                    },
                    {
                        "rank": 2,
                        "chunk_id": "chunk-1",
                        "source_id": "source_a",
                        "source_file": "source_a.pdf",
                        "page_start": 3,
                        "page_end": 3,
                        "score": 0.5,
                        "raw": {"metadata": {"chunk_id": "chunk-1"}},
                    },
                ],
            }
        }
    )

    report = diagnose_live_retrieval(
        dataset_path=dataset_path,
        chunks_path=chunks_path,
        output_path=output_path,
        top_k=5,
        fail_on_missing_chunk_id=True,
        adapter=adapter,
    )

    assert report["missing_chunk_id_count"] == 1
    assert report["should_fail"] is True
    assert "missing_chunk_id_count:1" in report["warnings"]
    assert report["records"][0]["errors"] == ["missing_chunk_id_in_retrieved"]


def test_extract_live_chunk_id_tries_multiple_fields():
    class Hit:
        def __init__(self):
            self.id = ""
            self.pk = "chunk-from-pk"
            self.primary_key = ""

    assert extract_live_chunk_id(Hit(), {}) == "chunk-from-pk"
    assert extract_live_chunk_id(object(), {"document_chunk_id": "chunk-from-metadata"}) == "chunk-from-metadata"


def make_sample(sample_id, reference_chunk_ids, expected_pages, source_id, question_type, should_abstain=False):
    return {
        "id": sample_id,
        "split": "build",
        "source_ids": [source_id],
        "collections": [source_id],
        "user_input": f"Question {sample_id}",
        "reference_answer": "" if should_abstain else f"Answer {sample_id}",
        "reference_chunk_ids": reference_chunk_ids,
        "reference_evidence": [] if should_abstain else [{"chunk_id": reference_chunk_ids[0], "page_start": expected_pages[0], "page_end": expected_pages[0], "quote": "quote"}],
        "expected_source_files": [] if should_abstain else [f"{source_id}.pdf"],
        "expected_page_numbers": expected_pages,
        "question_type": question_type,
        "reasoning_hops": "abstention" if should_abstain else "single_chunk",
        "criticality": "medium",
        "expected_confidence": "high",
        "should_abstain": should_abstain,
        "annotation_status": "reviewed",
        "annotator": "template_dry_run",
        "reviewer": "human",
        "notes": "",
    }


def make_chunk(chunk_id, source_id, source_file, page_start, page_end):
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_file": source_file,
        "collection": source_id,
        "page_start": page_start,
        "page_end": page_end,
    }


def make_retrieved(chunk_id, source_id, source_file, page_start, page_end, score):
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_file": source_file,
        "page_start": page_start,
        "page_end": page_end,
        "score": score,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
