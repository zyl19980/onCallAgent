import csv
import json
from pathlib import Path

from scripts.experiment.compare_retrieval_experiments import compare_retrieval_experiments


def test_compare_retrieval_experiments_builds_rows_and_deltas(tmp_path: Path):
    no_build = tmp_path / "dense_no_rerank_build.json"
    cur_build = tmp_path / "dense_current_rerank_build.json"
    no_dev = tmp_path / "dense_no_rerank_dev.json"
    output_path = tmp_path / "comparison.json"
    csv_path = tmp_path / "comparison.csv"

    write_json(
        no_build,
        make_result(
            experiment_name="dense_no_rerank_build",
            dataset="aiops-docs/experiment/rag/splits/rag_build.jsonl",
            rerank="none",
            hit_at_1=0.3,
            hit_at_3=0.5,
            hit_at_5=0.6,
            hit_at_10=0.7,
            mrr=0.4,
            gold_in_candidate_not_final_count=8,
            gold_promoted_by_rerank_count=0,
            gold_demoted_by_rerank_count=0,
        ),
    )
    write_json(
        cur_build,
        make_result(
            experiment_name="dense_current_rerank_build",
            dataset="aiops-docs/experiment/rag/splits/rag_build.jsonl",
            rerank="current",
            hit_at_1=0.4,
            hit_at_3=0.6,
            hit_at_5=0.65,
            hit_at_10=0.75,
            mrr=0.5,
            gold_in_candidate_not_final_count=4,
            gold_promoted_by_rerank_count=5,
            gold_demoted_by_rerank_count=1,
        ),
    )
    write_json(
        no_dev,
        make_result(
            experiment_name="dense_no_rerank_dev",
            dataset="aiops-docs/experiment/rag/splits/rag_dev.jsonl",
            rerank="none",
            hit_at_1=0.1,
            hit_at_3=0.2,
            hit_at_5=0.3,
            hit_at_10=0.4,
            mrr=0.2,
            gold_in_candidate_not_final_count=9,
            gold_promoted_by_rerank_count=0,
            gold_demoted_by_rerank_count=0,
        ),
    )

    report = compare_retrieval_experiments(
        input_paths=[no_build, cur_build, no_dev],
        output_path=output_path,
        csv_path=csv_path,
    )

    assert output_path.exists()
    assert csv_path.exists()
    assert report["baselines"] == {"build": "dense_no_rerank_build", "dev": "dense_no_rerank_dev"}

    rows = report["experiments"]
    by_name = {row["experiment_name"]: row for row in rows}
    current = by_name["dense_current_rerank_build"]
    assert current["split"] == "build"
    assert current["candidate_hit_at_50"] == 0.9
    assert current["delta_hit_at_1"] == 0.1
    assert current["delta_hit_at_3"] == 0.1
    assert current["delta_hit_at_5"] == 0.05
    assert current["delta_hit_at_10"] == 0.05
    assert current["delta_mrr"] == 0.1
    assert current["gold_promoted_by_rerank_count"] == 5

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 3
    assert csv_rows[0]["experiment_name"]
    assert "delta_hit_at_10" in csv_rows[0]


def make_result(
    *,
    experiment_name: str,
    dataset: str,
    rerank: str,
    hit_at_1: float,
    hit_at_3: float,
    hit_at_5: float,
    hit_at_10: float,
    mrr: float,
    gold_in_candidate_not_final_count: int,
    gold_promoted_by_rerank_count: int,
    gold_demoted_by_rerank_count: int,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "experiment_name": experiment_name,
        "retrieval_strategy": "dense",
        "rerank": rerank,
        "evaluated_samples": 10,
        "candidate_top_k": 50,
        "final_top_k": 10,
        "candidate_metrics": {
            "candidate_hit_at_10": 0.5,
            "candidate_hit_at_20": 0.7,
            "candidate_hit_at_50": 0.9,
        },
        "final_metrics": {
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "hit_at_10": hit_at_10,
            "recall_at_1": hit_at_1,
            "recall_at_3": hit_at_3,
            "recall_at_5": hit_at_5,
            "recall_at_10": hit_at_10,
            "mrr": mrr,
        },
        "gold_in_candidate_not_final_count": gold_in_candidate_not_final_count,
        "gold_promoted_by_rerank_count": gold_promoted_by_rerank_count,
        "gold_demoted_by_rerank_count": gold_demoted_by_rerank_count,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
