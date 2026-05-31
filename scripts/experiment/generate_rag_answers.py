"""Generate answer-level RAG outputs from frozen retrieval results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency in runtime environments
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = Path("scripts/experiment/prompts/answer_eval")
PROMPT_FILES = {
    "generator_v1": PROMPT_DIR / "generator_v1.txt",
    "generator_v1.1": PROMPT_DIR / "generator_v1.1.txt",
    "judge_claim_extract_v1": PROMPT_DIR / "judge_claim_extract_v1.txt",
    "judge_faithfulness_v1": PROMPT_DIR / "judge_faithfulness_v1.txt",
    "judge_faithfulness_v1.1": PROMPT_DIR / "judge_faithfulness_v1.1.txt",
    "judge_correctness_v1": PROMPT_DIR / "judge_correctness_v1.txt",
    "judge_citation_v1": PROMPT_DIR / "judge_citation_v1.txt",
}
GENERATOR_PROMPT_KEYS = {
    "v1.0": "generator_v1",
    "v1.1": "generator_v1.1",
}
DATED_MODEL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
VERSIONED_MODEL_RE = re.compile(r"(?:\d+(?:\.\d+)+|v\d+|\d{4}-\d{2}-\d{2})", re.IGNORECASE)
CITATION_RE = re.compile(r"\[chunk:(\d+)\]")
ABSTAIN_PREFIX = "INSUFFICIENT_EVIDENCE:"
RERANK_INTERPRETATION = "online rerank + local fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cited RAG answers from frozen expanded retrieval results.",
        epilog=(
            "Experiment discipline: do not regenerate the expanded dataset; do not re-chunk; "
            "do not re-index Milvus; do not overwrite existing retrieval JSON; current_rerank "
            "means online rerank + local fallback; generator model IDs must be exact provider "
            "model/version IDs; run build/dev before frozen test."
        ),
    )
    parser.add_argument("--dataset", required=True, help="Expanded validated dataset JSONL")
    parser.add_argument("--split-file", required=True, help="Frozen split JSONL to generate answers for")
    parser.add_argument("--retrieval-results", required=True, help="Frozen retrieval result JSON")
    parser.add_argument("--chunks", required=True, help="Experiment chunks JSONL")
    parser.add_argument(
        "--confidence-results",
        required=True,
        help="Confidence result JSON, or None when not available for this split",
    )
    parser.add_argument(
        "--generator-model",
        required=True,
        help="Exact model version, for example gpt-4o-mini-2024-07-18",
    )
    parser.add_argument("--generator-prompt-version", choices=sorted(GENERATOR_PROMPT_KEYS), default="v1.1")
    parser.add_argument(
        "--abstention-policy",
        required=True,
        choices=["confidence_bucket", "never_abstain", "gold_label"],
    )
    parser.add_argument("--output", required=True, help="Answer JSONL output path")
    parser.add_argument("--max-context-chunks", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env", override=True)

    args.generator_model = resolve_model_name(args.generator_model)
    validate_model_name(args.generator_model)
    if args.max_context_chunks <= 0:
        raise ValueError("--max-context-chunks must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    dataset_path = resolve_input_path(args.dataset, "dataset")
    split_path = resolve_input_path(args.split_file, "split_file")
    retrieval_path = resolve_input_path(args.retrieval_results, "retrieval_results")
    chunks_path = resolve_input_path(args.chunks, "chunks")
    confidence_path = resolve_optional_input_path(args.confidence_results, "confidence_results")
    output_path = path_from_repo(args.output).resolve()
    error_path = build_error_path(output_path)

    generator_prompt_key = GENERATOR_PROMPT_KEYS[args.generator_prompt_version]
    prompt_text = read_text_file(PROMPT_FILES[generator_prompt_key])
    prompt_hashes = hash_prompt_files()
    prompt_paths = {name: to_repo_relative(path_from_repo(path)) for name, path in PROMPT_FILES.items()}

    dataset_rows = read_jsonl(dataset_path)
    dataset_by_id = {str(row.get("id")): row for row in dataset_rows}
    split_rows = read_jsonl(split_path)
    retrieval_payload = read_json(retrieval_path)
    retrieval_by_id = {
        str(row.get("id") or row.get("sample_id")): row
        for row in retrieval_payload.get("per_sample", [])
    }
    chunks_by_id = {str(row.get("chunk_id")): row for row in read_jsonl(chunks_path)}
    confidence_by_id = load_confidence_by_id(confidence_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")
    if not error_path.exists():
        error_path.write_text("", encoding="utf-8")

    checkpoint_ids = load_checkpoint_ids(output_path) if args.resume else set()
    client = build_openai_client()

    attempted = 0
    succeeded = 0
    failed = 0
    deterministic_abstained = 0
    started_at = current_timestamp()
    run_id = build_run_id(started_at)
    git_sha = get_git_sha()

    for batch in chunked(split_rows, args.batch_size):
        for split_row in batch:
            sample_id = str(split_row.get("id") or "")
            if not sample_id:
                continue
            if sample_id in checkpoint_ids:
                continue
            attempted += 1
            sample = dict(dataset_by_id.get(sample_id) or {})
            sample.update(split_row)
            retrieval_row = retrieval_by_id.get(sample_id)
            confidence_row = confidence_by_id.get(sample_id)
            context_chunks = build_context_chunks(
                retrieval_row=retrieval_row,
                chunks_by_id=chunks_by_id,
                max_context_chunks=args.max_context_chunks,
            )
            pre_abstain_reason = determine_pre_abstain_reason(
                sample=sample,
                confidence_row=confidence_row,
                abstention_policy=args.abstention_policy,
            )

            try:
                if pre_abstain_reason:
                    generated_answer = f"{ABSTAIN_PREFIX} {pre_abstain_reason}"
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    deterministic_abstained += 1
                else:
                    user_prompt = build_user_prompt(
                        sample=sample,
                        context_chunks=context_chunks,
                        abstention_policy=args.abstention_policy,
                    )
                    generated_answer, usage, finish_reason, latency_ms = call_generator_with_retry(
                        client=client,
                        model=args.generator_model,
                        system_prompt=prompt_text,
                        user_prompt=user_prompt,
                        temperature=args.temperature,
                        seed=args.seed,
                    )
                    if not usage:
                        usage = estimate_usage(
                            model=args.generator_model,
                            system_prompt=prompt_text,
                            user_prompt=user_prompt,
                            generated_answer=generated_answer,
                        )

                record = build_output_record(
                    sample=sample,
                    retrieval_row=retrieval_row,
                    retrieval_payload=retrieval_payload,
                    context_chunks=context_chunks,
                    confidence_row=confidence_row,
                    generated_answer=generated_answer,
                    usage=usage,
                    args=args,
                    input_paths={
                        "dataset": dataset_path,
                        "split_file": split_path,
                        "retrieval_results": retrieval_path,
                        "chunks": chunks_path,
                        "confidence_results": confidence_path,
                    },
                    prompt_paths=prompt_paths,
                    prompt_hashes=prompt_hashes,
                    generator_prompt_key=generator_prompt_key,
                    started_at=started_at,
                    run_id=run_id,
                    git_sha=git_sha,
                    finish_reason=finish_reason if not pre_abstain_reason else "deterministic_abstain",
                    latency_ms=latency_ms if not pre_abstain_reason else 0,
                )
                append_jsonl(output_path, record)
                checkpoint_ids.add(sample_id)
                succeeded += 1
            except Exception as exc:  # single-sample failures must not stop a batch
                append_jsonl(
                    error_path,
                    {
                        "sample_id": sample_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "generator_model": args.generator_model,
                        "abstention_policy": args.abstention_policy,
                        "created_at": current_timestamp(),
                    },
                )
                checkpoint_ids.add(sample_id)
                failed += 1

    summary = summarize_run(
        output_path=output_path,
        error_path=error_path,
        requested_sample_ids=[str(row.get("id") or "") for row in split_rows],
        attempted=attempted,
        succeeded=succeeded,
        failed=failed,
        deterministic_abstained=deterministic_abstained,
        skipped_by_resume=len(checkpoint_ids) - attempted if args.resume else 0,
        started_at=started_at,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


def resolve_model_name(raw_model: str) -> str:
    env_value = os.environ.get(raw_model)
    if env_value:
        return env_value.strip()
    return raw_model.strip()


def validate_model_name(model: str) -> None:
    if not model:
        raise ValueError("--generator-model must not be empty")
    if model.startswith("gpt-") and not DATED_MODEL_RE.search(model):
        raise ValueError(
            "OpenAI --generator-model values must include an exact version date, "
            "for example gpt-4o-mini-2024-07-18"
        )
    if not VERSIONED_MODEL_RE.search(model):
        raise ValueError(
            "--generator-model must be an exact provider model/version ID, "
            "for example gpt-4o-mini-2024-07-18, qwen3.6-flash, or deepseek-v4-pro"
        )


def path_from_repo(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def resolve_optional_input_path(raw_path: str | None, label: str) -> Path | None:
    if raw_path is None:
        return None
    if raw_path.strip().lower() in {"", "none", "null"}:
        return None
    return resolve_input_path(raw_path, label)


def resolve_input_path(raw_path: str | Path, label: str) -> Path:
    path = path_from_repo(raw_path)
    if path.exists():
        return path.resolve()

    raw = str(raw_path)
    name = Path(raw).name
    legacy_prefix = "aiops-docs/experiment/results/"
    alternatives: list[Path] = []
    if raw.startswith(f"{legacy_prefix}retrieval/expanded/"):
        alternatives.append(
            REPO_ROOT / "aiops-docs/experiment/process_results/03_expanded_retrieval/raw_json" / name
        )
        alternatives.append(
            REPO_ROOT / "aiops-docs/experiment/process_results/04_query_decomposition/raw_json" / name
        )
        alternatives.append(REPO_ROOT / "aiops-docs/experiment/final_results/retrieval/raw_json" / name)
    if raw.startswith(f"{legacy_prefix}confidence/final/"):
        alternatives.append(REPO_ROOT / "aiops-docs/experiment/final_results/confidence/raw_json" / name)

    for alternative in alternatives:
        if alternative.exists():
            return alternative.resolve()
    raise FileNotFoundError(f"{label} not found: {raw_path}")


def build_error_path(output_path: Path) -> Path:
    if output_path.suffix == ".jsonl":
        return output_path.with_name(f"{output_path.stem}_errors.jsonl")
    return output_path.with_name(f"{output_path.name}_errors.jsonl")


def read_text_file(path: str | Path) -> str:
    return path_from_repo(path).read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint_ids(output_path: Path) -> set[str]:
    checkpoint_ids: set[str] = set()
    if not output_path.exists():
        return checkpoint_ids
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sample_id = str(row.get("sample_id") or row.get("id") or "")
        if sample_id:
            checkpoint_ids.add(sample_id)
    return checkpoint_ids


def load_confidence_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = read_json(path)
    return {
        str(row.get("sample_id") or row.get("id")): row
        for row in payload.get("per_sample", [])
    }


def build_context_chunks(
    *,
    retrieval_row: dict[str, Any] | None,
    chunks_by_id: dict[str, dict[str, Any]],
    max_context_chunks: int,
) -> list[dict[str, Any]]:
    if not retrieval_row:
        return []
    context_chunks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in retrieval_row.get("final_results", [])[:max_context_chunks]:
        chunk_id = str(result.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        chunk = chunks_by_id.get(chunk_id, {})
        seen.add(chunk_id)
        idx = len(context_chunks) + 1
        context_chunks.append(
            {
                "idx": idx,
                "chunk_id": chunk_id,
                "retrieval_rank": result.get("rank"),
                "original_rank": result.get("original_rank"),
                "rerank_rank": result.get("rerank_rank"),
                "score": result.get("score"),
                "original_score": result.get("original_score"),
                "rerank_score": result.get("rerank_score"),
                "rerank_provider": result.get("rerank_provider"),
                "source_id": chunk.get("source_id") or result.get("source_id"),
                "source_file": chunk.get("source_file") or result.get("source_file"),
                "page_start": chunk.get("page_start") or result.get("page_start"),
                "page_end": chunk.get("page_end") or result.get("page_end"),
                "section_path": chunk.get("section_path"),
                "title": chunk.get("title"),
                "chunk_type": chunk.get("chunk_type"),
                "text_hash": chunk.get("text_hash"),
                "text": chunk.get("text", ""),
            }
        )
    return context_chunks


def determine_pre_abstain_reason(
    *,
    sample: dict[str, Any],
    confidence_row: dict[str, Any] | None,
    abstention_policy: str,
) -> str:
    if abstention_policy == "gold_label" and bool(sample.get("should_abstain")):
        return "Gold label marks this sample as insufficient evidence."
    if abstention_policy == "confidence_bucket":
        if not confidence_row:
            return ""
        if str(confidence_row.get("predicted_confidence") or "").lower() == "low":
            return "Confidence bucket predicted low confidence."
    return ""


def build_user_prompt(
    *,
    sample: dict[str, Any],
    context_chunks: list[dict[str, Any]],
    abstention_policy: str,
) -> str:
    lines = [
        f"Sample ID: {sample.get('id', '')}",
        f"Question type: {sample.get('question_type', '')}",
        f"Abstention policy: {abstention_policy}",
        f"Question: {sample.get('user_input', '')}",
        "",
        "Retrieved chunks:",
    ]
    if not context_chunks:
        lines.append("(no retrieved chunks)")
    for chunk in context_chunks:
        lines.extend(
            [
                f"[chunk:{chunk['idx']}]",
                f"chunk_id: {chunk['chunk_id']}",
                (
                    "source: "
                    f"{chunk.get('source_file') or ''}, pages "
                    f"{chunk.get('page_start') or ''}-{chunk.get('page_end') or ''}"
                ),
                f"title: {chunk.get('title') or ''}",
                "text:",
                str(chunk.get("text") or ""),
                "",
            ]
        )
    lines.append("Answer:")
    return "\n".join(lines)


def build_openai_client() -> Any:
    base_url = os.environ.get("OPENAI_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url and os.environ.get("DASHSCOPE_API_BASE"):
        base_url = os.environ.get("DASHSCOPE_API_BASE")
        api_key = os.environ.get("DASHSCOPE_API_KEY") or api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set after loading .env")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"openai package is unavailable: {exc}") from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_generator_with_retry(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    seed: int,
) -> tuple[str, dict[str, int], str, int]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            started = time.monotonic()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                seed=seed,
                timeout=90,
            )
            latency_ms = int(round((time.monotonic() - started) * 1000))
            content = response.choices[0].message.content or ""
            finish_reason = str(response.choices[0].finish_reason or "")
            usage = response.usage
            usage_dict = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            return content.strip(), usage_dict, finish_reason, latency_ms
        except Exception as exc:
            last_error = exc
            if attempt < 4 and is_retryable_error(exc):
                time.sleep(min(60.0, 2.0**attempt))
                continue
            raise
    raise RuntimeError(f"generator call failed: {last_error}")


def is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    if isinstance(status_code, int) and status_code >= 500:
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "timeout" in name or "apierror" in name


def estimate_usage(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    generated_answer: str,
) -> dict[str, int]:
    prompt_tokens = estimate_tokens(system_prompt + "\n" + user_prompt, model)
    completion_tokens = estimate_tokens(generated_answer, model)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def estimate_tokens(text: str, model: str) -> int:
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def build_output_record(
    *,
    sample: dict[str, Any],
    retrieval_row: dict[str, Any] | None,
    retrieval_payload: dict[str, Any],
    context_chunks: list[dict[str, Any]],
    confidence_row: dict[str, Any] | None,
    generated_answer: str,
    usage: dict[str, int],
    args: argparse.Namespace,
    input_paths: dict[str, Path | None],
    prompt_paths: dict[str, str],
    prompt_hashes: dict[str, str],
    generator_prompt_key: str,
    started_at: str,
    run_id: str,
    git_sha: str,
    finish_reason: str,
    latency_ms: int,
) -> dict[str, Any]:
    cited_chunk_indices = parse_cited_chunk_indices(generated_answer)
    cited_chunk_ids = map_citations_to_chunk_ids(cited_chunk_indices, context_chunks)
    abstained = generated_answer.strip().upper().startswith(ABSTAIN_PREFIX)
    final_results = list((retrieval_row or {}).get("final_results") or [])
    final_chunk_ids = [str(item.get("chunk_id") or "") for item in final_results if item.get("chunk_id")]
    final_chunk_scores = [to_float_or_none(item.get("score")) for item in final_results if item.get("chunk_id")]
    generation_tokens = {
        "prompt": int(usage.get("prompt_tokens", 0) or 0),
        "completion": int(usage.get("completion_tokens", 0) or 0),
        "total": int(usage.get("total_tokens", 0) or 0),
    }
    return {
        "sample_id": str(sample.get("id") or ""),
        "split": sample.get("split"),
        "question_type": sample.get("question_type"),
        "should_abstain": bool(sample.get("should_abstain", False)),
        "question": sample.get("user_input", ""),
        "gold_answer": sample.get("reference_answer", ""),
        "gold_evidence_chunk_ids": sample.get("reference_chunk_ids") or [],
        "retrieval": {
            "experiment_name": retrieval_payload.get("experiment_name"),
            "split": sample.get("split"),
            "candidate_top_k": retrieval_payload.get("candidate_top_k"),
            "final_top_k": retrieval_payload.get("final_top_k"),
            "rerank_provider": summarize_rerank_provider(final_results),
            "rerank_interpretation": RERANK_INTERPRETATION,
            "final_chunk_ids": final_chunk_ids,
            "final_chunk_scores": final_chunk_scores,
        },
        "generation": {
            "generator_model": args.generator_model,
            "generator_prompt_version": args.generator_prompt_version,
            "system_prompt_hash": f"sha256:{prompt_hashes[generator_prompt_key]}",
            "generated_answer": generated_answer,
            "clean_answer": clean_generated_answer(generated_answer),
            "cited_chunk_ids": cited_chunk_ids,
            "cited_chunk_indices": cited_chunk_indices,
            "abstained": abstained,
            "abstention_reason": extract_abstention_reason(generated_answer) if abstained else None,
            "generation_latency_ms": latency_ms,
            "generation_tokens": generation_tokens,
            "generation_finish_reason": finish_reason,
        },
        "meta": {
            "run_id": run_id,
            "git_sha": git_sha,
            "timestamp": current_timestamp(),
            "run_started_at": started_at,
            "abstention_policy": args.abstention_policy,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_context_chunks": args.max_context_chunks,
            "confidence_available": confidence_row is not None,
            "predicted_confidence": (confidence_row or {}).get("predicted_confidence"),
            "prompt_paths": prompt_paths,
            "prompt_sha256": {name: f"sha256:{value}" for name, value in prompt_hashes.items()},
            "input_paths": {
                key: to_repo_relative(value) if value is not None else None
                for key, value in input_paths.items()
            },
        },
        "appendix": {
            "source_ids": sample.get("source_ids") or [],
            "gold_reference_evidence": sample.get("reference_evidence") or [],
            "retrieval_sample_found": retrieval_row is not None,
            "first_relevant_rank": (retrieval_row or {}).get("first_relevant_rank"),
            "hit_at_10": ((retrieval_row or {}).get("hit_at_k") or {}).get("10"),
            "chunk_index_map": [
                {"idx": chunk["idx"], "chunk_id": chunk["chunk_id"]} for chunk in context_chunks
            ],
            "context_chunks": context_chunks,
        },
    }


def parse_cited_chunk_indices(answer: str) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for match in CITATION_RE.finditer(answer):
        idx = int(match.group(1))
        if idx not in seen:
            output.append(idx)
            seen.add(idx)
    return output


def extract_abstention_reason(answer: str) -> str:
    stripped = answer.strip()
    if stripped.upper().startswith(ABSTAIN_PREFIX):
        return stripped[len(ABSTAIN_PREFIX) :].strip()
    return ""


def summarize_rerank_provider(final_results: list[dict[str, Any]]) -> str | None:
    providers = [str(item.get("rerank_provider") or "") for item in final_results]
    providers = [provider for provider in providers if provider]
    if not providers:
        return None
    unique = sorted(set(providers))
    if len(unique) == 1:
        return unique[0]
    return "mixed:" + ",".join(unique)


def to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def map_citations_to_chunk_ids(
    cited_chunk_indices: list[int],
    context_chunks: list[dict[str, Any]],
) -> list[str]:
    by_idx = {int(chunk["idx"]): str(chunk["chunk_id"]) for chunk in context_chunks}
    output = []
    for idx in cited_chunk_indices:
        chunk_id = by_idx.get(idx)
        if chunk_id:
            output.append(chunk_id)
    return output


def clean_generated_answer(answer: str) -> str:
    cleaned = CITATION_RE.sub("", answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def hash_prompt_files() -> dict[str, str]:
    hashes = {}
    for name, path in PROMPT_FILES.items():
        prompt_path = path_from_repo(path)
        data = prompt_path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def summarize_run(
    *,
    output_path: Path,
    error_path: Path,
    requested_sample_ids: list[str],
    attempted: int,
    succeeded: int,
    failed: int,
    deterministic_abstained: int,
    skipped_by_resume: int,
    started_at: str,
) -> dict[str, Any]:
    requested = {sample_id for sample_id in requested_sample_ids if sample_id}
    output_rows = [
        row for row in read_jsonl(output_path) if str(row.get("sample_id") or "") in requested
    ]
    success_ids = {str(row.get("sample_id") or "") for row in output_rows}
    error_rows = [
        row
        for row in read_jsonl(error_path)
        if str(row.get("sample_id") or "") in requested
        and str(row.get("sample_id") or "") not in success_ids
    ]
    prompt_tokens = sum(
        int(((row.get("generation") or {}).get("generation_tokens") or {}).get("prompt", 0) or 0)
        for row in output_rows
    )
    completion_tokens = sum(
        int(
            ((row.get("generation") or {}).get("generation_tokens") or {}).get(
                "completion", 0
            )
            or 0
        )
        for row in output_rows
    )
    total_tokens = sum(
        int(((row.get("generation") or {}).get("generation_tokens") or {}).get("total", 0) or 0)
        for row in output_rows
    )
    count = len(output_rows)
    return {
        "output": to_repo_relative(output_path),
        "errors": to_repo_relative(error_path),
        "started_at": started_at,
        "finished_at": current_timestamp(),
        "requested_samples": len(requested),
        "attempted_this_run": attempted,
        "succeeded_this_run": succeeded,
        "failed_this_run": failed,
        "skipped_by_resume": max(0, skipped_by_resume),
        "deterministic_abstained_this_run": deterministic_abstained,
        "output_success_rows_for_requested_samples": count,
        "error_rows_for_requested_samples": len(error_rows),
        "abstained_rows_for_requested_samples": sum(
            1 for row in output_rows if (row.get("generation") or {}).get("abstained")
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "avg_prompt_tokens": round(prompt_tokens / count, 6) if count else 0.0,
        "avg_completion_tokens": round(completion_tokens / count, 6) if count else 0.0,
        "avg_total_tokens": round(total_tokens / count, 6) if count else 0.0,
    }


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


def build_run_id(started_at: str) -> str:
    compact = re.sub(r"[^0-9]", "", started_at)[:14]
    return f"stage_a1_{compact}"


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def chunked(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


if __name__ == "__main__":
    sys.exit(main())
