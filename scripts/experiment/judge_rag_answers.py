"""Judge generated RAG answers for Stage A.2 answer-level evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = Path("scripts/experiment/prompts/answer_eval")
BASE_PROMPTS = {
    "claim_extract": PROMPT_DIR / "judge_claim_extract_v1.txt",
    "faithfulness": PROMPT_DIR / "judge_faithfulness_v1.txt",
    "correctness": PROMPT_DIR / "judge_correctness_v1.txt",
    "citation": PROMPT_DIR / "judge_citation_v1.txt",
}
FAITHFULNESS_PROMPTS = {
    "v1.0": PROMPT_DIR / "judge_faithfulness_v1.txt",
    "v1.1": PROMPT_DIR / "judge_faithfulness_v1.1.txt",
}
ABSTAIN_PREFIX = "INSUFFICIENT_EVIDENCE:"
VERSIONED_MODEL_RE = re.compile(r"(?:\d+(?:\.\d+)+|v\d+|\d{4}-\d{2}-\d{2})", re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge Stage A generated RAG answers for faithfulness, correctness, citation, and hallucination.",
        epilog=(
            "Experiment discipline: read generation.generated_answer with citation markers; "
            "temperature must be 0; do not modify answers, retrieval JSON, confidence JSON, dataset, or chunks; "
            "supporting_evidence_quote is post-validated as a continuous substring of retrieved chunks."
        ),
    )
    parser.add_argument("--answers", required=True, help="Stage A.1 answers JSONL")
    parser.add_argument("--dataset", required=True, help="Expanded validated dataset JSONL")
    parser.add_argument("--chunks", required=True, help="Experiment chunks JSONL")
    parser.add_argument("--judge-model", required=True, help="Exact judge model/version ID or env alias")
    parser.add_argument("--judge-prompt-version", default="v1.0")
    parser.add_argument("--output", required=True, help="Judge JSONL output path")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-output-tokens", type=int, default=1600)
    parser.add_argument("--enable-claim-extraction", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-for-human-review", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env", override=True)
    if args.temperature != 0.0:
        raise ValueError("Stage A.2 judge calls must use --temperature 0.0")
    judge_model = resolve_model_name(args.judge_model)
    validate_model_name(judge_model)

    answers_path = path_from_repo(args.answers).resolve()
    dataset_path = path_from_repo(args.dataset).resolve()
    chunks_path = path_from_repo(args.chunks).resolve()
    output_path = path_from_repo(args.output).resolve()
    error_path = build_error_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")
    if not error_path.exists():
        error_path.write_text("", encoding="utf-8")

    answers = read_jsonl(answers_path)
    dataset_by_id = {str(row.get("id")): row for row in read_jsonl(dataset_path)}
    chunks_by_id = {str(row.get("chunk_id")): row for row in read_jsonl(chunks_path)}
    checkpoint_ids = load_checkpoint_ids(output_path) if args.resume else set()
    selected_prompts = prompt_paths_for_version(args.judge_prompt_version)
    prompt_texts = {name: read_text(path) for name, path in selected_prompts.items()}
    prompt_hashes = {name: sha256_file(path_from_repo(path)) for name, path in selected_prompts.items()}
    run_id = build_run_id()
    git_sha = get_git_sha()
    client = build_openai_client()

    attempted = 0
    succeeded = 0
    failed = 0
    for batch in chunked(answers, args.batch_size):
        pending_rows = [
            answer_row
            for answer_row in batch
            if str(answer_row.get("sample_id") or "") and str(answer_row.get("sample_id") or "") not in checkpoint_ids
        ]
        attempted += len(pending_rows)
        if not pending_rows:
            continue

        def run_row(answer_row: dict[str, Any]) -> dict[str, Any]:
            generator_model = str((answer_row.get("generation") or {}).get("generator_model") or "")
            validate_different_family(generator_model=generator_model, judge_model=judge_model)
            return judge_one_answer(
                answer_row=answer_row,
                dataset_row=dataset_by_id.get(str(answer_row.get("sample_id") or ""), {}),
                chunks_by_id=chunks_by_id,
                client=client,
                judge_model=judge_model,
                judge_prompt_version=args.judge_prompt_version,
                prompt_texts=prompt_texts,
                prompt_hashes=prompt_hashes,
                temperature=args.temperature,
                request_timeout=args.request_timeout,
                max_output_tokens=args.max_output_tokens,
                run_id=run_id,
                git_sha=git_sha,
                enable_claim_extraction=args.enable_claim_extraction,
            )

        with ThreadPoolExecutor(max_workers=max(1, args.batch_size)) as executor:
            future_to_sample_id = {
                executor.submit(run_row, answer_row): str(answer_row.get("sample_id") or "")
                for answer_row in pending_rows
            }
            for future in as_completed(future_to_sample_id):
                sample_id = future_to_sample_id[future]
                try:
                    record = future.result()
                    append_jsonl(output_path, record)
                    checkpoint_ids.add(sample_id)
                    succeeded += 1
                except Exception as exc:
                    append_jsonl(
                        error_path,
                        {
                            "sample_id": sample_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "judge_model": judge_model,
                            "created_at": current_timestamp(),
                        },
                    )
                    failed += 1

    human_review_path = None
    if args.sample_for_human_review > 0:
        human_review_path = export_human_review_samples(
            judge_path=output_path,
            output_path=default_human_review_path(output_path),
            sample_size=args.sample_for_human_review,
        )

    summary = summarize(output_path=output_path, error_path=error_path, requested=answers)
    summary.update(
        {
            "attempted_this_run": attempted,
            "succeeded_this_run": succeeded,
            "failed_this_run": failed,
            "judge_model": judge_model,
            "human_review_csv": to_repo_relative(human_review_path) if human_review_path else None,
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


def judge_one_answer(
    *,
    answer_row: dict[str, Any],
    dataset_row: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    client: Any,
    judge_model: str,
    judge_prompt_version: str,
    prompt_texts: dict[str, str],
    prompt_hashes: dict[str, str],
    temperature: float,
    request_timeout: float,
    max_output_tokens: int,
    run_id: str,
    git_sha: str,
    enable_claim_extraction: bool,
) -> dict[str, Any]:
    sample_id = str(answer_row.get("sample_id") or "")
    generation = dict(answer_row.get("generation") or {})
    generated_answer = str(generation.get("generated_answer") or "")
    final_chunk_ids = list((answer_row.get("retrieval") or {}).get("final_chunk_ids") or [])
    context_chunks = list((answer_row.get("appendix") or {}).get("context_chunks") or [])
    retrieved_texts = [str(chunk.get("text") or "") for chunk in context_chunks]
    idx_to_id = {
        int(item["idx"]): str(item["chunk_id"])
        for item in (answer_row.get("appendix") or {}).get("chunk_index_map") or []
    }
    generated_cited_ids = list(generation.get("cited_chunk_ids") or [])
    invalid_citation_ids = sorted(set(generated_cited_ids) - set(final_chunk_ids))

    print(f"[judge] {sample_id} claim_extract", flush=True)
    claim_payload = call_json_step(
        client=client,
        model=judge_model,
        system_prompt=prompt_texts["claim_extract"],
        user_prompt=build_claim_extract_input(answer_row),
        temperature=temperature,
        request_timeout=request_timeout,
        max_output_tokens=max_output_tokens,
    )
    raw_claims = claim_payload.get("claims") or []
    claims = normalize_claims(raw_claims, sample_id=sample_id, idx_to_id=idx_to_id)

    print(f"[judge] {sample_id} faithfulness", flush=True)
    faithfulness_payload = call_json_step(
        client=client,
        model=judge_model,
        system_prompt=prompt_texts["faithfulness"],
        user_prompt=build_faithfulness_input(answer_row, claims),
        temperature=temperature,
        request_timeout=request_timeout,
        max_output_tokens=max_output_tokens,
    )
    print(f"[judge] {sample_id} correctness", flush=True)
    correctness_payload = call_json_step(
        client=client,
        model=judge_model,
        system_prompt=prompt_texts["correctness"],
        user_prompt=build_correctness_input(answer_row),
        temperature=temperature,
        request_timeout=request_timeout,
        max_output_tokens=max_output_tokens,
    )
    print(f"[judge] {sample_id} citation", flush=True)
    citation_payload = call_json_step(
        client=client,
        model=judge_model,
        system_prompt=prompt_texts["citation"],
        user_prompt=build_citation_input(answer_row),
        temperature=temperature,
        request_timeout=request_timeout,
        max_output_tokens=max_output_tokens,
    )

    claims = merge_faithfulness(
        claims=claims,
        faithfulness_payload=faithfulness_payload,
        retrieved_texts=retrieved_texts,
        final_chunk_ids=final_chunk_ids,
        idx_to_id=idx_to_id,
        context_chunks=context_chunks,
    )
    quote_invalid_count = sum(1 for claim in claims if claim.get("judge_quote_invalid"))
    n_claims = len(claims)
    n_supported_by_retrieved = sum(1 for claim in claims if claim.get("supported_by_retrieved"))
    n_supported_by_cited = sum(1 for claim in claims if claim.get("supported_by_cited"))
    citation_accuracy = build_citation_accuracy(
        citation_payload=citation_payload,
        generated_cited_ids=generated_cited_ids,
        final_chunk_ids=final_chunk_ids,
        retrieved_texts=retrieved_texts,
    )
    hallucination_count = n_claims - n_supported_by_retrieved
    model_abstained = bool(generation.get("abstained"))
    should_abstain = bool(answer_row.get("should_abstain"))

    return {
        "sample_id": sample_id,
        "split": answer_row.get("split"),
        "question_type": answer_row.get("question_type"),
        "should_abstain": should_abstain,
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
        "judge_run_id": run_id,
        "claims": claims,
        "faithfulness": {
            "n_claims": n_claims,
            "n_supported_by_retrieved": n_supported_by_retrieved,
            "n_supported_by_cited": n_supported_by_cited,
            "score_supported_by_retrieved": safe_div(n_supported_by_retrieved, n_claims),
            "score_supported_by_cited": safe_div(n_supported_by_cited, n_claims),
            "quote_invalid_count": quote_invalid_count,
        },
        "answer_correctness": normalize_correctness(correctness_payload),
        "citation_accuracy": citation_accuracy,
        "hallucination": {
            "n_unsupported_claims": hallucination_count,
            "rate": safe_div(hallucination_count, n_claims),
            "any_hallucination": hallucination_count > 0,
        },
        "abstention_check": {
            "should_abstain_label": should_abstain,
            "model_abstained": model_abstained,
            "abstention_correct": should_abstain == model_abstained,
        },
        "answer_snapshot": {
            "question": answer_row.get("question"),
            "gold_answer": answer_row.get("gold_answer"),
            "generated_answer": generated_answer,
            "generator_model": generation.get("generator_model"),
        },
        "postprocess": {
            "invalid_citation": bool(invalid_citation_ids),
            "invalid_citation_ids": invalid_citation_ids,
            "quote_invalid_count": quote_invalid_count,
            "enable_claim_extraction": enable_claim_extraction,
        },
        "meta": {
            "git_sha": git_sha,
            "timestamp": current_timestamp(),
            "generator_model": generation.get("generator_model"),
            "generated_answer_field_used": "generation.generated_answer",
            "prompt_sha256": {name: f"sha256:{value}" for name, value in prompt_hashes.items()},
        },
    }


def normalize_claims(
    raw_claims: list[dict[str, Any]],
    *,
    sample_id: str,
    idx_to_id: dict[int, str],
) -> list[dict[str, Any]]:
    claims = []
    for index, raw in enumerate(raw_claims, start=1):
        cited_indices = [int(item) for item in raw.get("cited_chunk_indices") or [] if str(item).isdigit()]
        claims.append(
            {
                "claim_id": f"{sample_id}::c{index}",
                "text": str(raw.get("claim_text") or raw.get("text") or "").strip(),
                "cited_chunk_ids": [idx_to_id[idx] for idx in cited_indices if idx in idx_to_id],
                "cited_chunk_indices": cited_indices,
                "supported_by_retrieved": False,
                "supported_by_cited": False,
                "supporting_evidence_quote": "",
                "judge_reasoning": "",
                "judge_quote_invalid": False,
            }
        )
    return [claim for claim in claims if claim["text"]]


def merge_faithfulness(
    *,
    claims: list[dict[str, Any]],
    faithfulness_payload: dict[str, Any],
    retrieved_texts: list[str],
    final_chunk_ids: list[str],
    idx_to_id: dict[int, str],
    context_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    faith_by_id = {str(item.get("claim_id")): item for item in faithfulness_payload.get("claims") or []}
    text_by_id = {str(chunk.get("chunk_id") or ""): str(chunk.get("text") or "") for chunk in context_chunks}
    for claim in claims:
        faith = faith_by_id.get(claim["claim_id"], {})
        verdict = str(faith.get("verdict") or "").lower()
        quote = str(faith.get("supporting_evidence_quote") or "").strip()
        quote_valid = quote_exists_in_texts(quote, retrieved_texts) if quote else True
        supported = verdict == "supported" and quote_valid
        claim["supported_by_retrieved"] = supported
        cited_ids = set(claim.get("cited_chunk_ids") or [])
        supporting_chunk_idx = parse_optional_int(faith.get("supporting_chunk_idx"))
        supporting_chunk_id = idx_to_id.get(supporting_chunk_idx) if supporting_chunk_idx is not None else None
        quote_supported_by_cited = bool(
            quote and cited_ids and any(quote_exists_in_texts(quote, [text_by_id.get(chunk_id, "")]) for chunk_id in cited_ids)
        )
        claim["supporting_chunk_idx"] = supporting_chunk_idx
        claim["supporting_chunk_id"] = supporting_chunk_id
        claim["supported_by_cited"] = supported and (
            bool(supporting_chunk_id and supporting_chunk_id in cited_ids) or quote_supported_by_cited
        )
        claim["supporting_evidence_quote"] = quote if quote_valid else ""
        claim["judge_reasoning"] = str(faith.get("explanation") or "").strip()
        claim["judge_quote_invalid"] = bool(quote and not quote_valid)
        if not quote_valid:
            claim["supported_by_retrieved"] = False
            claim["supported_by_cited"] = False
    return claims


def build_citation_accuracy(
    *,
    citation_payload: dict[str, Any],
    generated_cited_ids: list[str],
    final_chunk_ids: list[str],
    retrieved_texts: list[str],
) -> dict[str, Any]:
    checks = list(citation_payload.get("citation_checks") or [])
    valid_indices: set[int] = set()
    invalid_indices: set[int] = set()
    for check in checks:
        quote = str(check.get("supporting_evidence_quote") or "").strip()
        quote_valid = quote_exists_in_texts(quote, retrieved_texts) if quote else False
        is_valid = str(check.get("verdict") or "").lower() == "valid" and quote_valid
        check["judge_quote_invalid"] = bool(quote and not quote_valid)
        chunk_idx = parse_optional_int(check.get("chunk_idx"))
        if is_valid:
            if chunk_idx is not None:
                valid_indices.add(chunk_idx)
        elif chunk_idx is not None:
            invalid_indices.add(chunk_idx)
    generated_cited_ids = list(dict.fromkeys(generated_cited_ids))
    invalid_citation_ids = sorted(set(generated_cited_ids) - set(final_chunk_ids))
    n_citations = len(generated_cited_ids)
    valid_count = min(len(valid_indices), n_citations)
    n_missing = len(citation_payload.get("uncited_supported_claims") or [])
    precision = safe_div(valid_count, n_citations)
    recall = safe_div(valid_count, valid_count + n_missing)
    return {
        "n_citations": n_citations,
        "n_correct_citations": valid_count,
        "n_missing_citations": n_missing,
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0,
        "invalid_citation": bool(invalid_citation_ids),
        "invalid_citation_ids": invalid_citation_ids,
        "invalid_citation_indices": sorted(invalid_indices),
        "citation_checks": checks,
    }


def normalize_correctness(payload: dict[str, Any]) -> dict[str, Any]:
    verdict = str(payload.get("verdict") or "incorrect")
    if verdict in {"correct_abstention"}:
        score = 1.0
    elif verdict in {"should_have_abstained"}:
        score = 0.0
    else:
        score = float(payload.get("score", {"correct": 1.0, "partially_correct": 0.5}.get(verdict, 0.0)) or 0.0)
    return {
        "verdict": verdict,
        "score": max(0.0, min(1.0, score)),
        "missing_key_points": payload.get("missing_key_points") or [],
        "incorrect_points": payload.get("incorrect_points") or [],
        "supporting_evidence_quote": str(payload.get("supporting_evidence_quote") or ""),
        "judge_reasoning": str(payload.get("explanation") or payload.get("judge_reasoning") or ""),
    }


def parse_optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def build_claim_extract_input(answer_row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "Extract atomic factual claims from generated_answer. Preserve cited [chunk:N] indices for each claim.",
            "sample_id": answer_row.get("sample_id"),
            "question": answer_row.get("question"),
            "generated_answer": (answer_row.get("generation") or {}).get("generated_answer"),
            "retrieved_chunks": compact_chunks(answer_row),
        },
        ensure_ascii=False,
        indent=2,
    )


def build_faithfulness_input(answer_row: dict[str, Any], claims: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "task": "Judge each claim against retrieved_chunks only. Use claim_id exactly as given.",
            "sample_id": answer_row.get("sample_id"),
            "question": answer_row.get("question"),
            "generated_answer": (answer_row.get("generation") or {}).get("generated_answer"),
            "claims": [{"claim_id": c["claim_id"], "claim_text": c["text"]} for c in claims],
            "retrieved_chunks": compact_chunks(answer_row),
        },
        ensure_ascii=False,
        indent=2,
    )


def build_correctness_input(answer_row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "Judge generated answer correctness against gold_answer and gold evidence.",
            "sample_id": answer_row.get("sample_id"),
            "question": answer_row.get("question"),
            "gold_answer": answer_row.get("gold_answer"),
            "gold_evidence_chunk_ids": answer_row.get("gold_evidence_chunk_ids"),
            "gold_reference_evidence": (answer_row.get("appendix") or {}).get("gold_reference_evidence"),
            "generated_answer": (answer_row.get("generation") or {}).get("generated_answer"),
            "retrieved_chunks": compact_chunks(answer_row),
        },
        ensure_ascii=False,
        indent=2,
    )


def build_citation_input(answer_row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "Validate every [chunk:N] citation in generated_answer against retrieved_chunks.",
            "sample_id": answer_row.get("sample_id"),
            "question": answer_row.get("question"),
            "generated_answer": (answer_row.get("generation") or {}).get("generated_answer"),
            "cited_chunk_indices": (answer_row.get("generation") or {}).get("cited_chunk_indices"),
            "cited_chunk_ids": (answer_row.get("generation") or {}).get("cited_chunk_ids"),
            "retrieved_chunks": compact_chunks(answer_row),
        },
        ensure_ascii=False,
        indent=2,
    )


def compact_chunks(answer_row: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for chunk in (answer_row.get("appendix") or {}).get("context_chunks") or []:
        output.append(
            {
                "idx": chunk.get("idx"),
                "chunk_id": chunk.get("chunk_id"),
                "source_file": chunk.get("source_file"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "text": chunk.get("text"),
            }
        )
    return output


def call_json_step(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    request_timeout: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
                timeout=request_timeout,
            )
            return parse_json_content(response.choices[0].message.content or "")
        except Exception as exc:
            last_error = exc
            if attempt < 4 and is_retryable_error(exc):
                time.sleep(min(60.0, 2.0**attempt))
                continue
            raise
    raise RuntimeError(f"judge call failed: {last_error}")


def parse_json_content(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(content)
        if match:
            return json.loads(match.group(0))
        raise


def quote_exists_in_texts(quote: str, texts: list[str]) -> bool:
    normalized_quote = normalize_ws(quote)
    if not normalized_quote:
        return False
    return any(normalized_quote in normalize_ws(text) for text in texts)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def default_human_review_path(output_path: Path) -> Path:
    split = infer_split_from_name(output_path.name)
    return output_path.with_name(f"human_review_samples_{split}.csv")


def export_human_review_samples(*, judge_path: Path, output_path: Path, sample_size: int) -> Path:
    rows = read_jsonl(judge_path)
    rng = random.Random(20260508)
    selected = rows if len(rows) <= sample_size else rng.sample(rows, sample_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "question",
                "gold_answer",
                "generated_answer",
                "faithfulness_supported_by_cited",
                "correctness_verdict",
                "judge_reasoning_short",
            ],
        )
        writer.writeheader()
        for row in selected:
            answer = row.get("answer_snapshot") or {}
            writer.writerow(
                {
                    "sample_id": row.get("sample_id"),
                    "question": answer.get("question") or "",
                    "gold_answer": answer.get("gold_answer") or "",
                    "generated_answer": answer.get("generated_answer") or "",
                    "faithfulness_supported_by_cited": (row.get("faithfulness") or {}).get(
                        "score_supported_by_cited"
                    ),
                    "correctness_verdict": (row.get("answer_correctness") or {}).get("verdict"),
                    "judge_reasoning_short": first_sentence(
                        (row.get("answer_correctness") or {}).get("judge_reasoning") or ""
                    ),
                }
            )
    return output_path


def summarize(*, output_path: Path, error_path: Path, requested: list[dict[str, Any]]) -> dict[str, Any]:
    requested_ids = {str(row.get("sample_id") or "") for row in requested}
    rows = [row for row in read_jsonl(output_path) if str(row.get("sample_id") or "") in requested_ids]
    success_ids = {str(row.get("sample_id") or "") for row in rows}
    errors = [
        row
        for row in read_jsonl(error_path)
        if str(row.get("sample_id") or "") in requested_ids
        and str(row.get("sample_id") or "") not in success_ids
    ]
    return {
        "output": to_repo_relative(output_path),
        "errors": to_repo_relative(error_path),
        "requested_samples": len(requested_ids),
        "success_rows": len(rows),
        "error_rows": len(errors),
        "quote_invalid_count": sum((row.get("postprocess") or {}).get("quote_invalid_count", 0) for row in rows),
    }


def load_checkpoint_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    return {str(row.get("sample_id") or "") for row in read_jsonl(output_path) if row.get("sample_id")}


def build_openai_client() -> Any:
    base_url = os.environ.get("OPENAI_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url and os.environ.get("DASHSCOPE_API_BASE"):
        base_url = os.environ.get("DASHSCOPE_API_BASE")
        api_key = os.environ.get("DASHSCOPE_API_KEY") or api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set after loading .env")
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def prompt_paths_for_version(version: str) -> dict[str, Path]:
    if version not in FAITHFULNESS_PROMPTS:
        raise ValueError(f"unsupported --judge-prompt-version: {version}")
    prompts = dict(BASE_PROMPTS)
    prompts["faithfulness"] = FAITHFULNESS_PROMPTS[version]
    return prompts


def resolve_model_name(raw_model: str) -> str:
    return (os.environ.get(raw_model) or raw_model).strip()


def validate_model_name(model: str) -> None:
    if not model or not VERSIONED_MODEL_RE.search(model):
        raise ValueError("--judge-model must be an exact provider model/version ID")


def validate_different_family(*, generator_model: str, judge_model: str) -> None:
    if model_family(generator_model) == model_family(judge_model):
        raise ValueError(f"judge model must be a different family from generator: {generator_model}")


def model_family(model: str) -> str:
    value = model.lower()
    for family in ["gpt", "qwen", "deepseek", "claude"]:
        if value.startswith(family):
            return family
    return value.split("-", 1)[0]


def is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    if isinstance(status_code, int) and status_code >= 500:
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "timeout" in name or "apierror" in name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text(path: Path) -> str:
    return path_from_repo(path).read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_error_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_errors.jsonl")


def path_from_repo(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def to_repo_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def current_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def build_run_id() -> str:
    compact = re.sub(r"[^0-9]", "", current_timestamp())[:14]
    return f"stage_a2_{compact}"


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def infer_split_from_name(name: str) -> str:
    for split in ["build", "dev", "test", "reserve"]:
        if f"_{split}_" in name or name.endswith(f"_{split}.jsonl"):
            return split
    return "build"


def first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return parts[0] if parts and parts[0] else text.strip()


def chunked(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


if __name__ == "__main__":
    sys.exit(main())
