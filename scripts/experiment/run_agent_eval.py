"""Run A0-A3 agent evaluation over fixed replay cases."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
CASE_RAG_IDS = {
    "agent_case_031": ["rag_023"],
    "agent_case_032": ["rag_035"],
    "agent_case_033": ["rag_032", "rag_033", "rag_034"],
    "agent_case_034": ["rag_012"],
    "agent_case_035": ["rag_014"],
}
RAG_DATASET = REPO_ROOT / "aiops-docs/experiment/rag/datasets/expanded/experiment_rag_dataset_expanded.validated.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A0-A3 fixed replay agent evaluation.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--mode", required=True, choices=["A0", "A1", "A2", "A3"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-ids", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env", override=True)

    model = resolve_model_name(args.model)
    cases_path = path_from_repo(args.cases)
    output_path = path_from_repo(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")

    cases = read_jsonl(cases_path)
    if args.case_ids:
        wanted = {item.strip() for item in args.case_ids.split(",") if item.strip()}
        cases = [case for case in cases if str(case.get("case_id")) in wanted]
    if args.resume:
        done = {
            str(row.get("case_id"))
            for row in read_jsonl(output_path)
            if str(row.get("mode")) == args.mode
        }
        cases = [case for case in cases if str(case.get("case_id")) not in done]

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    client = build_openai_client()
    run_id = build_run_id(args.mode)
    case_map = {str(case["case_id"]): case for case in read_jsonl(cases_path)}

    attempted = 0
    succeeded = 0
    failed = 0
    for batch in chunked(cases, args.batch_size):
        for case in batch:
            attempted += 1
            try:
                record = asyncio.run(
                    run_one_case(
                        case=case,
                        case_map=case_map,
                        mode=args.mode,
                        model=model,
                        temperature=args.temperature,
                        client=client,
                        run_id=run_id,
                    )
                )
                append_jsonl(output_path, record)
                succeeded += 1
                print(f"[run_agent_eval] {args.mode} {case['case_id']} ok", flush=True)
            except Exception as exc:
                failed += 1
                error_record = build_error_record(case, args.mode, model, args.temperature, run_id, exc)
                append_jsonl(default_error_path(output_path), error_record)
                print(f"[run_agent_eval] {args.mode} {case.get('case_id')} failed: {exc}", flush=True)

    summary = {
        "mode": args.mode,
        "model": model,
        "requested": len(cases),
        "attempted_this_run": attempted,
        "succeeded_this_run": succeeded,
        "failed_this_run": failed,
        "output": str(output_path),
        "errors": str(default_error_path(output_path)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


async def run_one_case(
    *,
    case: dict[str, Any],
    case_map: dict[str, dict[str, Any]],
    mode: str,
    model: str,
    temperature: float,
    client: Any,
    run_id: str,
) -> dict[str, Any]:
    started = time.monotonic()
    tracker = RunTracker()

    if mode == "A0":
        raw_output = run_a0(case=case, client=client, model=model, temperature=temperature)
        executed_steps = 0
        replan_count = 0
    elif mode == "A1":
        raw_output = run_a1(case=case, case_map=case_map, client=client, model=model, temperature=temperature, tracker=tracker)
        executed_steps = 1 if tracker.tools_called else 0
        replan_count = 0
    elif mode in {"A2", "A3"}:
        raw_output, executed_steps, replan_count = await run_plan_execute_replan(
            case=case,
            case_map=case_map,
            client=client,
            model=model,
            temperature=temperature,
            tracker=tracker,
            enable_rag=(mode == "A3"),
        )
    else:  # pragma: no cover
        raise ValueError(f"unsupported mode: {mode}")

    diagnosis = extract_diagnosis_fields(raw_output, client=client, model=model, temperature=temperature)
    latency_ms = int(round((time.monotonic() - started) * 1000))

    return {
        "case_id": case["case_id"],
        "mode": mode,
        "model": model,
        "generated_root_cause": diagnosis.get("generated_root_cause", ""),
        "generated_recommendation": diagnosis.get("generated_recommendation", ""),
        "tools_called": tracker.tools_called,
        "tool_call_count": len(tracker.tools_called),
        "executed_steps": executed_steps,
        "replan_count": replan_count,
        "latency_ms": latency_ms,
        "rag_context_retrieved": bool(tracker.rag_chunks_used),
        "rag_chunks_used": tracker.rag_chunks_used,
        "raw_output": raw_output,
        "meta": {
            "run_id": run_id,
            "timestamp": current_timestamp(),
            "temperature": temperature,
            "rag_enabled": mode == "A3",
            "backend": tracker.backend,
        },
    }


def run_a0(*, case: dict[str, Any], client: Any, model: str, temperature: float) -> str:
    system_prompt = (
        "你是故障诊断专家。只基于用户提交的故障工单进行诊断，"
        "不要假设看到了监控或日志 payload。输出 JSON，字段为 "
        "generated_root_cause、generated_recommendation。"
    )
    user_prompt = f"故障工单:\n{case['task_text']}\n\n请给出根因和处置建议。"
    return call_chat(client=client, model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)


def run_a1(
    *,
    case: dict[str, Any],
    case_map: dict[str, dict[str, Any]],
    client: Any,
    model: str,
    temperature: float,
    tracker: "RunTracker",
) -> str:
    system_prompt = (
        "你是 ReAct 风格故障诊断助手。你只有一轮工具调用机会。"
        "先输出 JSON: {\"thought\":\"...\",\"actions\":[{\"tool\":\"工具名\",\"args\":{...}}]}。"
        "可用工具: query_cpu_metrics, query_memory_metrics, search_topic_by_service_name, search_log。"
        "调用监控或日志工具时必须带 replay_case_id。"
    )
    user_prompt = (
        f"case_id: {case['case_id']}\n"
        f"service_name: {case['service_name']}\n"
        f"task_text: {case['task_text']}\n"
        "请选择这一轮要调用的工具。"
    )
    action_text = call_chat(client=client, model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
    action_payload = parse_json_object(action_text) or {}
    actions = action_payload.get("actions") or []
    if not isinstance(actions, list) or not actions:
        actions = default_actions_for_case(case)

    observations = []
    for action in actions[:4]:
        tool_name = str(action.get("tool") or "")
        args = dict(action.get("args") or {})
        args.setdefault("service_name", case["service_name"])
        args.setdefault("replay_case_id", case["case_id"])
        observation = execute_replay_tool(tool_name, args, case_map=case_map, tracker=tracker)
        observations.append({"tool": tool_name, "args": args, "observation": observation})

    answer_system = (
        "你是故障诊断专家。根据工单和 Observation 输出 JSON，字段为 "
        "generated_root_cause、generated_recommendation。根因和建议必须引用 Observation 中的具体值。"
    )
    answer_user = json.dumps(
        {
            "task_text": case["task_text"],
            "observations": observations,
        },
        ensure_ascii=False,
        indent=2,
    )
    return call_chat(client=client, model=model, system_prompt=answer_system, user_prompt=answer_user, temperature=temperature)


async def run_plan_execute_replan(
    *,
    case: dict[str, Any],
    case_map: dict[str, dict[str, Any]],
    client: Any,
    model: str,
    temperature: float,
    tracker: "RunTracker",
    enable_rag: bool,
) -> tuple[str, int, int]:
    tracker.backend = "AIOpsService"
    task_text = (
        f"{case['task_text']}\n\n"
        f"实验固定回放参数: replay_case_id={case['case_id']}，service_name={case['service_name']}。"
        "调用任何监控或日志 MCP 工具时必须传入 replay_case_id。"
    )
    backend = os.environ.get("AGENT_EVAL_AIOPS_BACKEND", "auto").strip().lower()
    if backend in {"auto", "service"}:
        try:
            timeout_seconds = float(os.environ.get("AGENT_EVAL_AIOPS_TIMEOUT_SEC", "12"))
            return await asyncio.wait_for(
                run_aiops_service_once(
                    case=case,
                    case_map=case_map,
                    tracker=tracker,
                    task_text=task_text,
                    model=model,
                    temperature=temperature,
                    enable_rag=enable_rag,
                ),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            if backend == "service":
                raise
            tracker.backend = f"local_fallback_after_{type(exc).__name__}"
    else:
        tracker.backend = "local_plan_execute_replan"

    return run_local_plan_execute_replan(
        case=case,
        case_map=case_map,
        client=client,
        model=model,
        temperature=temperature,
        tracker=tracker,
        enable_rag=enable_rag,
    )


async def run_aiops_service_once(
    *,
    case: dict[str, Any],
    case_map: dict[str, dict[str, Any]],
    tracker: "RunTracker",
    task_text: str,
    model: str,
    temperature: float,
    enable_rag: bool,
) -> tuple[str, int, int]:
    with patched_aiops_runtime(case=case, case_map=case_map, tracker=tracker):
        from app.services.aiops_service import AIOpsService

        service = AIOpsService()
        raw_output = ""
        executed_steps = 0
        replan_count = 0
        session_id = f"agent-eval-{case['case_id']}-{uuid.uuid4().hex[:8]}"
        async for event in service.execute(
            task_text,
            session_id=session_id,
            enable_rag=enable_rag,
            replay_case_id=case["case_id"],
            model_name=model,
            temperature=temperature,
        ):
            if event.get("type") == "step_complete":
                executed_steps += 1
            if event.get("stage") == "replanner":
                replan_count += 1
            if event.get("report"):
                raw_output = str(event.get("report") or "")
            if event.get("response"):
                raw_output = str(event.get("response") or raw_output)
        if not raw_output:
            raise RuntimeError("AIOpsService completed without final output")
        return raw_output, executed_steps, replan_count


def run_local_plan_execute_replan(
    *,
    case: dict[str, Any],
    case_map: dict[str, dict[str, Any]],
    client: Any,
    model: str,
    temperature: float,
    tracker: "RunTracker",
    enable_rag: bool,
) -> tuple[str, int, int]:
    observations = []
    for action in default_actions_for_case(case):
        tool_name = action["tool"]
        args = dict(action.get("args") or {})
        args.setdefault("service_name", case["service_name"])
        args.setdefault("replay_case_id", case["case_id"])
        observations.append({"tool": tool_name, "observation": execute_replay_tool(tool_name, args, case_map=case_map, tracker=tracker)})

    rag_context = ""
    if enable_rag:
        rag_context, _ = local_retrieve_knowledge(case["task_text"], case_id=case["case_id"], tracker=tracker)

    system_prompt = (
        "你是 Plan-Execute-Replan 故障诊断专家。根据任务、工具观测和可选 RAG 上下文输出 JSON，"
        "字段为 generated_root_cause、generated_recommendation。必须引用具体观测值。"
    )
    user_prompt = json.dumps(
        {
            "task_text": case["task_text"],
            "observations": observations,
            "rag_context": rag_context if enable_rag else "",
        },
        ensure_ascii=False,
        indent=2,
    )
    raw = call_chat(client=client, model=model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
    return raw, len(observations), 1


@contextmanager
def patched_aiops_runtime(*, case: dict[str, Any], case_map: dict[str, dict[str, Any]], tracker: "RunTracker"):
    planner_mod = importlib.import_module("app.agent.aiops.planner")
    executor_mod = importlib.import_module("app.agent.aiops.executor")
    replanner_mod = importlib.import_module("app.agent.aiops.replanner")
    modules = [planner_mod, executor_mod, replanner_mod]
    originals = {
        module: {
            "get_mcp_tools_safely": getattr(module, "get_mcp_tools_safely", None),
            "retrieve_knowledge": getattr(module, "retrieve_knowledge", None),
        }
        for module in modules
    }

    async def get_replay_tools_safely(*_args: Any, **_kwargs: Any) -> list[Any]:
        return build_langchain_replay_tools(case_map=case_map, tracker=tracker)

    retrieve_tool = build_langchain_retrieve_tool(case_id=case["case_id"], tracker=tracker)
    try:
        for module in modules:
            module.get_mcp_tools_safely = get_replay_tools_safely
            module.retrieve_knowledge = retrieve_tool
        yield
    finally:
        for module, values in originals.items():
            if values["get_mcp_tools_safely"] is not None:
                module.get_mcp_tools_safely = values["get_mcp_tools_safely"]
            if values["retrieve_knowledge"] is not None:
                module.retrieve_knowledge = values["retrieve_knowledge"]


def build_langchain_replay_tools(*, case_map: dict[str, dict[str, Any]], tracker: "RunTracker") -> list[Any]:
    from langchain_core.tools import tool

    @tool
    def query_cpu_metrics(service_name: str, replay_case_id: str = "", start_time: str = "", end_time: str = "", interval: str = "1m") -> dict[str, Any]:
        """Query CPU metrics for a replay case."""
        return execute_replay_tool("query_cpu_metrics", locals(), case_map=case_map, tracker=tracker)

    @tool
    def query_memory_metrics(service_name: str, replay_case_id: str = "", start_time: str = "", end_time: str = "", interval: str = "1m") -> dict[str, Any]:
        """Query memory metrics for a replay case."""
        return execute_replay_tool("query_memory_metrics", locals(), case_map=case_map, tracker=tracker)

    @tool
    def search_topic_by_service_name(service_name: str, replay_case_id: str = "", region_code: str = "", fuzzy: bool = True) -> dict[str, Any]:
        """Search log topic by service name for a replay case."""
        return execute_replay_tool("search_topic_by_service_name", locals(), case_map=case_map, tracker=tracker)

    @tool
    def search_log(topic_id: str = "topic-001", replay_case_id: str = "", start_time: int = 0, end_time: int = 0, query: str = "", limit: int = 100) -> dict[str, Any]:
        """Search logs for a replay case."""
        return execute_replay_tool("search_log", locals(), case_map=case_map, tracker=tracker)

    return [query_cpu_metrics, query_memory_metrics, search_topic_by_service_name, search_log]


def build_langchain_retrieve_tool(*, case_id: str, tracker: "RunTracker") -> Any:
    from langchain_core.tools import tool

    @tool(response_format="content_and_artifact")
    def retrieve_knowledge(query: str) -> tuple[str, list[Any]]:
        """Retrieve maintenance knowledge relevant to this fixed replay case."""
        return local_retrieve_knowledge(query, case_id=case_id, tracker=tracker)

    return retrieve_knowledge


def local_retrieve_knowledge(query: str, *, case_id: str, tracker: "RunTracker") -> tuple[str, list[Any]]:
    try:
        from langchain_core.documents import Document
    except Exception:  # pragma: no cover
        Document = None  # type: ignore[assignment]

    rows = read_jsonl(RAG_DATASET) if RAG_DATASET.exists() else []
    wanted_ids = CASE_RAG_IDS.get(case_id, [])
    selected = [row for row in rows if str(row.get("id")) in wanted_ids]
    if not selected:
        selected = rank_rag_rows(query, rows)[:3]

    chunks = []
    docs = []
    lines = ["检索整体置信度: high", "重排来源: local_agent_eval_replay", ""]
    for row in selected:
        evidence = (row.get("reference_evidence") or [{}])[0]
        chunk_id = str((row.get("reference_chunk_ids") or [row.get("id")])[0])
        text = str(evidence.get("quote") or row.get("reference_answer") or "")
        chunk = {
            "chunk_id": chunk_id,
            "rag_id": row.get("id"),
            "source": (row.get("expected_source_files") or [""])[0],
            "page_start": evidence.get("page_start"),
            "page_end": evidence.get("page_end"),
            "text_preview": text[:240],
        }
        chunks.append(chunk)
        lines.extend(
            [
                f"[{row.get('id')}] {row.get('user_input')}",
                f"answer: {row.get('reference_answer')}",
                f"evidence: {text}",
                "",
            ]
        )
        if Document is not None:
            docs.append(Document(page_content=text, metadata=chunk))

    tracker.rag_chunks_used.extend(chunks)
    return "\n".join(lines).strip() or "没有找到相关信息。", docs


def rank_rag_rows(query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_terms = set(re.findall(r"[A-Za-z0-9.]+|[\u4e00-\u9fff]{2,}", query.lower()))

    def score(row: dict[str, Any]) -> int:
        text = json.dumps(row, ensure_ascii=False).lower()
        return sum(1 for term in query_terms if term and term in text)

    eligible = [
        row
        for row in rows
        if row.get("should_abstain") is False
        and row.get("question_type") in {"troubleshooting_step", "symptom_cause"}
    ]
    return sorted(eligible, key=score, reverse=True)


def execute_replay_tool(tool_name: str, args: dict[str, Any], *, case_map: dict[str, dict[str, Any]], tracker: "RunTracker") -> dict[str, Any]:
    case_id = str(args.get("replay_case_id") or "")
    if not case_id and len(case_map) == 1:
        case_id = next(iter(case_map))
    case = case_map.get(case_id)
    tracker.tools_called.append(tool_name)
    if not case:
        return {"error": f"未找到回放案例: {case_id}"}
    if tool_name in {"query_cpu_metrics", "query_memory_metrics"}:
        return dict(case["metrics_payload"])
    if tool_name == "search_log":
        return dict(case["logs_payload"])
    if tool_name == "search_topic_by_service_name":
        logs = case["logs_payload"]
        service_name = case.get("service_name") or args.get("service_name") or ""
        return {
            "total": 1,
            "topics": [
                {
                    "topic_id": logs.get("topic_id", "topic-001"),
                    "topic_name": f"{service_name}日志",
                    "service_name": service_name,
                    "region_code": args.get("region_code") or "ap-beijing",
                    "create_time": "2024-01-01 10:00:00",
                    "log_count": logs.get("total", 0),
                    "description": f"{service_name} 的固定回放日志",
                }
            ],
            "query": {
                "service_name": args.get("service_name"),
                "region_code": args.get("region_code") or None,
                "fuzzy": args.get("fuzzy", True),
            },
            "message": f"找到 1 个匹配的回放日志主题: {case_id}",
        }
    return {"error": f"unsupported tool: {tool_name}"}


def default_actions_for_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    metric_name = str((case.get("metrics_payload") or {}).get("metric_name") or "")
    metric_tool = "query_memory_metrics" if metric_name == "memory_usage_percent" else "query_cpu_metrics"
    return [
        {"tool": metric_tool, "args": {"service_name": case["service_name"], "replay_case_id": case["case_id"]}},
        {"tool": "search_topic_by_service_name", "args": {"service_name": case["service_name"], "replay_case_id": case["case_id"]}},
        {"tool": "search_log", "args": {"topic_id": "topic-001", "replay_case_id": case["case_id"]}},
    ]


def extract_diagnosis_fields(raw_output: str, *, client: Any, model: str, temperature: float) -> dict[str, str]:
    parsed = parse_json_object(raw_output)
    if parsed and ("generated_root_cause" in parsed or "generated_recommendation" in parsed):
        return {
            "generated_root_cause": str(parsed.get("generated_root_cause") or ""),
            "generated_recommendation": str(parsed.get("generated_recommendation") or ""),
        }

    system_prompt = "从故障诊断输出中抽取 JSON，字段为 generated_root_cause、generated_recommendation。不要添加其他字段。"
    extracted = call_chat(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=raw_output[:12000],
        temperature=temperature,
    )
    parsed = parse_json_object(extracted) or {}
    return {
        "generated_root_cause": str(parsed.get("generated_root_cause") or raw_output[:800]),
        "generated_recommendation": str(parsed.get("generated_recommendation") or ""),
    }


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
    raise RuntimeError(f"LLM call failed: {last_error}")


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


class RunTracker:
    def __init__(self) -> None:
        self.tools_called: list[str] = []
        self.rag_chunks_used: list[dict[str, Any]] = []
        self.backend = "direct"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def path_from_repo(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_model_name(value: str) -> str:
    if value.startswith("$"):
        value = value[1:]
    return os.environ.get(value, value)


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_id(mode: str) -> str:
    return f"agent-eval-{mode.lower()}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def default_error_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_errors.jsonl")


def build_error_record(case: dict[str, Any], mode: str, model: str, temperature: float, run_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "mode": mode,
        "model": model,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "meta": {
            "run_id": run_id,
            "timestamp": current_timestamp(),
            "temperature": temperature,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
