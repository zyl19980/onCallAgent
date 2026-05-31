"""LLM-as-Judge for fixed replay agent evaluation results."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge A0-A3 agent evaluation outputs.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env", override=True)
    if args.temperature != 0.0:
        raise ValueError("judge temperature must be 0.0 for reproducibility")

    judge_model = resolve_model_name(args.judge_model)
    results_path = path_from_repo(args.results)
    cases_path = path_from_repo(args.cases)
    output_path = path_from_repo(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")

    results = read_jsonl(results_path)
    cases_by_id = {str(case.get("case_id")): case for case in read_jsonl(cases_path)}
    done = load_done_keys(output_path) if args.resume else set()
    client = build_openai_client()
    run_id = build_run_id()

    attempted = 0
    succeeded = 0
    failed = 0
    for batch in chunked(results, args.batch_size):
        for result in batch:
            key = make_result_key(result)
            if key in done:
                continue
            attempted += 1
            case = cases_by_id.get(str(result.get("case_id")))
            if not case:
                failed += 1
                append_jsonl(default_error_path(output_path), {"key": key, "error": "case not found"})
                continue
            try:
                record = judge_one(
                    result=result,
                    case=case,
                    client=client,
                    judge_model=judge_model,
                    temperature=args.temperature,
                    run_id=run_id,
                )
                append_jsonl(output_path, record)
                done.add(key)
                succeeded += 1
                print(f"[judge_agent_eval] {key} ok", flush=True)
            except Exception as exc:
                failed += 1
                append_jsonl(
                    default_error_path(output_path),
                    {
                        "key": key,
                        "case_id": result.get("case_id"),
                        "mode": result.get("mode"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "timestamp": current_timestamp(),
                    },
                )
                print(f"[judge_agent_eval] {key} failed: {exc}", flush=True)

    summary = {
        "judge_model": judge_model,
        "requested": len(results),
        "attempted_this_run": attempted,
        "succeeded_this_run": succeeded,
        "failed_this_run": failed,
        "output": str(output_path),
        "errors": str(default_error_path(output_path)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


def judge_one(
    *,
    result: dict[str, Any],
    case: dict[str, Any],
    client: Any,
    judge_model: str,
    temperature: float,
    run_id: str,
) -> dict[str, Any]:
    expected_tools = [str(tool) for tool in case.get("gold_expected_tools") or []]
    tools_called = [str(tool) for tool in result.get("tools_called") or []]
    tool_precision = precision(tools_called, expected_tools)
    tool_recall = recall(tools_called, expected_tools)

    judge_payload = call_judge_llm(
        result=result,
        case=case,
        client=client,
        judge_model=judge_model,
        temperature=temperature,
    )

    return {
        "case_id": result.get("case_id"),
        "mode": result.get("mode"),
        "model": result.get("model"),
        "judge_model": judge_model,
        "root_cause_accuracy": normalize_verdict(
            nested_get(judge_payload, "root_cause_accuracy", "verdict"),
            allowed={"correct", "partial", "incorrect"},
        ),
        "root_cause_accuracy_reasoning": truncate_reasoning(nested_get(judge_payload, "root_cause_accuracy", "reasoning")),
        "evidence_completeness": clamp_float(nested_get(judge_payload, "evidence_completeness", "score")),
        "evidence_completeness_reasoning": truncate_reasoning(nested_get(judge_payload, "evidence_completeness", "reasoning")),
        "recommendation_actionability": normalize_verdict(
            nested_get(judge_payload, "recommendation_actionability", "verdict"),
            allowed={"correct", "partial", "incorrect"},
        ),
        "recommendation_actionability_reasoning": truncate_reasoning(nested_get(judge_payload, "recommendation_actionability", "reasoning")),
        "tool_precision": tool_precision,
        "tool_precision_reasoning": f"called={sorted(set(tools_called))}; expected={sorted(set(expected_tools))}",
        "tool_recall": tool_recall,
        "tool_recall_reasoning": f"called={sorted(set(tools_called))}; expected={sorted(set(expected_tools))}",
        "raw_judge_output": judge_payload,
        "meta": {
            "run_id": run_id,
            "timestamp": current_timestamp(),
            "temperature": temperature,
        },
    }


def call_judge_llm(
    *,
    result: dict[str, Any],
    case: dict[str, Any],
    client: Any,
    judge_model: str,
    temperature: float,
) -> dict[str, Any]:
    system_prompt = (
        "你是严格的 LLM-as-Judge。请比较生成结果与 gold label。"
        "只输出 JSON，不要 Markdown。每项 reasoning 不超过 50 个词。"
        "JSON schema: {"
        "\"root_cause_accuracy\":{\"verdict\":\"correct|partial|incorrect\",\"reasoning\":\"...\"},"
        "\"evidence_completeness\":{\"score\":0.0,\"reasoning\":\"...\"},"
        "\"recommendation_actionability\":{\"verdict\":\"correct|partial|incorrect\",\"reasoning\":\"...\"}"
        "}。"
    )
    user_prompt = json.dumps(
        {
            "case_id": case.get("case_id"),
            "task_text": case.get("task_text"),
            "gold_root_cause": case.get("gold_root_cause"),
            "gold_evidence": case.get("gold_evidence"),
            "gold_recommendation": case.get("gold_recommendation"),
            "generated_root_cause": result.get("generated_root_cause"),
            "generated_recommendation": result.get("generated_recommendation"),
            "raw_output": str(result.get("raw_output") or "")[:8000],
            "rubric": {
                "root_cause_accuracy": "generated_root_cause 是否覆盖 gold_root_cause 的核心结论",
                "evidence_completeness": "generated_root_cause 引用的证据是否覆盖 gold_evidence 中的关键字段，0 到 1",
                "recommendation_actionability": "generated_recommendation 是否包含具体可执行操作和 payload 中的具体值",
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    content = call_chat(client=client, model=judge_model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
    return parse_json_object(content) or {
        "root_cause_accuracy": {"verdict": "incorrect", "reasoning": "judge output was not valid JSON"},
        "evidence_completeness": {"score": 0.0, "reasoning": "judge output was not valid JSON"},
        "recommendation_actionability": {"verdict": "incorrect", "reasoning": "judge output was not valid JSON"},
        "raw": content,
    }


def precision(called: list[str], expected: list[str]) -> float:
    called_set = set(called)
    if not called_set:
        return 0.0
    return round(len(called_set & set(expected)) / len(called_set), 4)


def recall(called: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    return round(len(set(called) & expected_set) / len(expected_set), 4)


def call_chat(*, client: Any, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
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
                timeout=120,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            retryable = status_code == 429 or (isinstance(status_code, int) and status_code >= 500)
            if retryable and attempt < 4:
                time.sleep(min(60.0, 2.0**attempt))
                continue
            raise
    raise RuntimeError(f"judge call failed: {last_error}")


def build_openai_client() -> Any:
    base_url = os.environ.get("OPENAI_API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url and os.environ.get("DASHSCOPE_API_BASE"):
        base_url = os.environ.get("DASHSCOPE_API_BASE")
        api_key = os.environ.get("DASHSCOPE_API_KEY") or api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or DASHSCOPE_API_KEY is not set after loading .env")
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    match = JSON_OBJECT_RE.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def normalize_verdict(value: Any, *, allowed: set[str]) -> str:
    verdict = str(value or "").strip().lower()
    return verdict if verdict in allowed else "incorrect"


def clamp_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)


def truncate_reasoning(value: Any, max_chars: int = 180) -> str:
    text = str(value or "").strip()
    return text[:max_chars]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("--batch-size must be positive")
    return [items[index:index + size] for index in range(0, len(items), size)]


def load_done_keys(path: Path) -> set[str]:
    return {make_result_key(row) for row in read_jsonl(path)}


def make_result_key(row: dict[str, Any]) -> str:
    return f"{row.get('mode')}::{row.get('case_id')}"


def path_from_repo(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_model_name(value: str) -> str:
    if value.startswith("$"):
        value = value[1:]
    return os.environ.get(value, value)


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_id() -> str:
    return f"agent-judge-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def default_error_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_errors.jsonl")


if __name__ == "__main__":
    raise SystemExit(main())
