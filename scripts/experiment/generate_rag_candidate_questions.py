"""从 annotation pool 生成待人工复核的 RAG 候选问题。"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCE_TARGETS = {
    "haascnc_com_rotary_troubleshooting_guide_ngc": 20,
    "rockwell_powerflex_520_series_user_manual_520_um001_en_e": 20,
    "abb_manual_for_induction_motors_and_generators_en": 15,
    "grundfos_nbe_nbse_nke_tpe_tped_installation_and_operating_instructions": 10,
    "s71200_system_manual": 15,
}

DEFAULT_QUESTION_TYPE_TARGETS = {
    "troubleshooting_step": 20,
    "symptom_cause": 15,
    "parameter_or_fault_code": 20,
    "safety_or_constraint": 15,
    "definition_or_component_lookup": 10,
}

PRIORITY_ORDER = {"high": 3, "medium": 2, "low": 1}
CHUNK_TYPE_BONUS = {
    "troubleshooting_procedure": 45,
    "alarm_fault_code": 40,
    "parameter_and_configuration": 28,
    "safety_and_constraint": 26,
    "maintenance_procedure": 18,
    "installation_or_wiring": 12,
    "concept_and_component": 10,
}
PRODUCT_LABELS = {
    "haascnc_com_rotary_troubleshooting_guide_ngc": "Haas rotary unit",
    "rockwell_powerflex_520_series_user_manual_520_um001_en_e": "PowerFlex 520-series drive",
    "abb_manual_for_induction_motors_and_generators_en": "ABB motor",
    "grundfos_nbe_nbse_nke_tpe_tped_installation_and_operating_instructions": "Grundfos pump",
    "s71200_system_manual": "Siemens S7-1200 PLC",
}
COMPONENT_TERMS = (
    "encoder",
    "motor",
    "pump",
    "relay",
    "module",
    "terminal",
    "connector",
    "bearing",
    "sensor",
    "switch",
    "controller",
    "control panel",
    "cpu",
    "drive",
    "brake",
    "shaft",
    "valve",
    "wiring",
    "cable",
    "fan",
)
ACTION_WORDS = (
    "check",
    "install",
    "set",
    "press",
    "remove",
    "verify",
    "replace",
    "adjust",
    "cycle",
    "disconnect",
    "connect",
    "open",
    "close",
    "clean",
    "inspect",
    "test",
    "tighten",
)
SAFETY_WORDS = ("warning", "danger", "caution", "notice", "must", "do not", "hazard")
KEYWORDS = ("troubleshooting", "alarm", "fault", "warning", "parameter", "safety")


@dataclass(slots=True)
class PoolCandidate:
    row: dict[str, object]
    possible_question_types: list[str]
    rank_cost: int
    fallback_definition: bool


@dataclass(slots=True)
class SelectionResult:
    row: dict[str, object]
    question_type: str
    fallback_definition: bool


@dataclass(slots=True)
class Edge:
    to: int
    rev: int
    cap: int
    cost: int


class MinCostMaxFlow:
    def __init__(self, node_count: int):
        self.graph: list[list[Edge]] = [[] for _ in range(node_count)]

    def add_edge(self, src: int, dst: int, cap: int, cost: int) -> None:
        forward = Edge(to=dst, rev=len(self.graph[dst]), cap=cap, cost=cost)
        backward = Edge(to=src, rev=len(self.graph[src]), cap=0, cost=-cost)
        self.graph[src].append(forward)
        self.graph[dst].append(backward)

    def min_cost_flow(self, src: int, sink: int, max_flow: int) -> tuple[int, int]:
        node_count = len(self.graph)
        total_flow = 0
        total_cost = 0
        potentials = [0] * node_count

        while total_flow < max_flow:
            distances = [10**18] * node_count
            prev_node = [-1] * node_count
            prev_edge = [-1] * node_count
            distances[src] = 0
            heap: list[tuple[int, int]] = [(0, src)]

            while heap:
                dist, node = heapq.heappop(heap)
                if dist != distances[node]:
                    continue
                for edge_index, edge in enumerate(self.graph[node]):
                    if edge.cap <= 0:
                        continue
                    new_cost = dist + edge.cost + potentials[node] - potentials[edge.to]
                    if new_cost < distances[edge.to]:
                        distances[edge.to] = new_cost
                        prev_node[edge.to] = node
                        prev_edge[edge.to] = edge_index
                        heapq.heappush(heap, (new_cost, edge.to))

            if distances[sink] == 10**18:
                break

            for node in range(node_count):
                if distances[node] < 10**18:
                    potentials[node] += distances[node]

            add_flow = max_flow - total_flow
            node = sink
            while node != src:
                edge = self.graph[prev_node[node]][prev_edge[node]]
                add_flow = min(add_flow, edge.cap)
                node = prev_node[node]

            node = sink
            while node != src:
                edge = self.graph[prev_node[node]][prev_edge[node]]
                edge.cap -= add_flow
                reverse_edge = self.graph[node][edge.rev]
                reverse_edge.cap += add_flow
                node = prev_node[node]

            total_flow += add_flow
            total_cost += add_flow * potentials[sink]

        return total_flow, total_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 RAG 候选问题")
    parser.add_argument(
        "--input",
        default="aiops-docs/experiment/chunks/experiment_annotation_pool.jsonl",
        help="annotation pool JSONL 输入路径",
    )
    parser.add_argument(
        "--output",
        default="aiops-docs/experiment/rag/rag_candidate_questions.jsonl",
        help="候选问题 JSONL 输出路径",
    )
    parser.add_argument(
        "--report",
        default="aiops-docs/experiment/rag/rag_candidate_generation_report.json",
        help="生成报告输出路径",
    )
    parser.add_argument(
        "--generator",
        default="template",
        choices=["template"],
        help="生成器类型，当前仅实现 template",
    )
    parser.add_argument(
        "--dry-run-template",
        action="store_true",
        help="显式启用无 LLM 的模板生成模式",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = generate_rag_candidate_questions(
        input_path=Path(args.input),
        output_path=Path(args.output),
        report_path=Path(args.report),
        generator=args.generator,
        dry_run_template=args.dry_run_template,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def generate_rag_candidate_questions(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    generator: str = "template",
    dry_run_template: bool = False,
    source_targets: dict[str, int] | None = None,
    question_type_targets: dict[str, int] | None = None,
) -> dict[str, object]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    source_targets = dict(source_targets or DEFAULT_SOURCE_TARGETS)
    question_type_targets = dict(question_type_targets or DEFAULT_QUESTION_TYPE_TARGETS)

    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = prepare_candidates(rows, source_targets)
    selections, selection_warnings = select_candidates(candidates, source_targets, question_type_targets)

    generator_name = "template_dry_run" if dry_run_template or generator == "template" else generator
    failed_generation_count = 0
    generated_rows: list[dict[str, object]] = []
    warnings = list(selection_warnings)
    flagged_review_rows = 0

    for selection in selections:
        try:
            row = build_generated_candidate(selection, generator_name)
        except Exception:
            failed_generation_count += 1
            continue
        if selection.row.get("quality_flags") or selection.fallback_definition:
            flagged_review_rows += 1
        generated_rows.append(row)

    generated_rows.sort(
        key=lambda row: (
            row["source_id"],
            row["page_start"],
            row["candidate_id"],
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in generated_rows)
    output_path.write_text(output_text + ("\n" if generated_rows else ""), encoding="utf-8")

    skipped_chunks = {
        "input_pool_rows": len(rows),
        "eligible_rows_considered": len(candidates),
        "unselected_rows": len(candidates) - len(selections),
    }
    if failed_generation_count:
        skipped_chunks["generation_failed"] = failed_generation_count
    if flagged_review_rows:
        warnings.append(f"selected_rows_need_manual_focus:{flagged_review_rows}")

    report = {
        "total_candidates": len(generated_rows),
        "count_by_source": dict(sorted(Counter(row["source_id"] for row in generated_rows).items())),
        "count_by_question_type": dict(
            sorted(Counter(row["suggested_question_type"] for row in generated_rows).items())
        ),
        "failed_generation_count": failed_generation_count,
        "skipped_chunks": skipped_chunks,
        "warnings": warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def prepare_candidates(rows: list[dict[str, object]], source_targets: dict[str, int]) -> list[PoolCandidate]:
    rows_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["source_id"] not in source_targets:
            continue
        rows_by_source[str(row["source_id"])].append(row)

    rank_costs: dict[str, int] = {}
    for source_id, source_rows in rows_by_source.items():
        ordered = rank_rows_for_source(source_rows)
        for index, row in enumerate(ordered):
            rank_costs[str(row["chunk_id"])] = index

    candidates: list[PoolCandidate] = []
    for source_id in sorted(rows_by_source):
        for row in rows_by_source[source_id]:
            possible_question_types, fallback_definition = determine_possible_question_types(row)
            if not possible_question_types:
                continue
            candidates.append(
                PoolCandidate(
                    row=row,
                    possible_question_types=possible_question_types,
                    rank_cost=rank_costs[str(row["chunk_id"])],
                    fallback_definition=fallback_definition,
                )
            )

    return candidates


def rank_rows_for_source(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    bucketed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        bucketed[(int(row["page_start"]) - 1) // 10].append(row)

    for bucket_rows in bucketed.values():
        bucket_rows.sort(key=lambda row: (-row_priority_score(row), int(row["page_start"]), row["chunk_id"]))

    bucket_order = sorted(
        bucketed,
        key=lambda bucket_id: (-row_priority_score(bucketed[bucket_id][0]), bucket_id),
    )

    ordered: list[dict[str, object]] = []
    active = list(bucket_order)
    while active:
        next_active: list[int] = []
        for bucket_id in active:
            bucket_rows = bucketed[bucket_id]
            if not bucket_rows:
                continue
            ordered.append(bucket_rows.pop(0))
            if bucket_rows:
                next_active.append(bucket_id)
        active = next_active

    return ordered


def row_priority_score(row: dict[str, object]) -> int:
    text = str(row["text"]).lower()
    char_count = int(row.get("char_count") or len(str(row["text"])))
    keyword_score = sum(len(re.findall(r"\b" + re.escape(keyword) + r"\b", text)) for keyword in KEYWORDS)
    score = PRIORITY_ORDER[str(row["annotation_priority"])] * 1000
    score += CHUNK_TYPE_BONUS.get(str(row["chunk_type"]), 0)
    score += min(char_count, 900) // 10
    score += keyword_score * 12
    score += 35 if row.get("fault_code") else 0
    score += 20 if row.get("parameter_name") else 0
    score -= len(row.get("quality_flags", [])) * 35
    return score


def determine_possible_question_types(row: dict[str, object]) -> tuple[list[str], bool]:
    question_types = list(row.get("recommended_question_types", []))
    fallback_definition = False

    if is_definition_fallback_candidate(row):
        if "definition_or_component_lookup" not in question_types:
            question_types.append("definition_or_component_lookup")
            fallback_definition = True

    return question_types, fallback_definition


def is_definition_fallback_candidate(row: dict[str, object]) -> bool:
    if str(row["chunk_type"]) in {"alarm_fault_code", "troubleshooting_procedure"}:
        return False
    haystack = f"{row['title']}\n{row['text']}".lower()
    return any(term in haystack for term in COMPONENT_TERMS)


def select_candidates(
    candidates: list[PoolCandidate],
    source_targets: dict[str, int],
    question_type_targets: dict[str, int],
) -> tuple[list[SelectionResult], list[str]]:
    total_target = sum(source_targets.values())
    if total_target != sum(question_type_targets.values()):
        raise ValueError("source targets 与 question type targets 总数不一致")

    sources = sorted(source_targets)
    qtypes = sorted(question_type_targets)
    source_node_base = 1
    row_node_base = source_node_base + len(sources)
    qtype_node_base = row_node_base + len(candidates)
    sink = qtype_node_base + len(qtypes)
    graph = MinCostMaxFlow(sink + 1)

    source_index = {source_id: idx for idx, source_id in enumerate(sources)}
    qtype_index = {qtype: idx for idx, qtype in enumerate(qtypes)}

    for source_id in sources:
        graph.add_edge(0, source_node_base + source_index[source_id], source_targets[source_id], 0)
    for qtype in qtypes:
        graph.add_edge(qtype_node_base + qtype_index[qtype], sink, question_type_targets[qtype], 0)

    row_metadata: dict[int, PoolCandidate] = {}
    row_qtype_edges: dict[int, list[tuple[str, int]]] = defaultdict(list)

    for row_idx, candidate in enumerate(candidates):
        node_id = row_node_base + row_idx
        row_metadata[node_id] = candidate
        source_id = str(candidate.row["source_id"])
        graph.add_edge(
            source_node_base + source_index[source_id],
            node_id,
            1,
            candidate.rank_cost * 3 + row_cost_penalty(candidate.row),
        )
        for qtype in candidate.possible_question_types:
            extra_cost = 0
            if qtype == "definition_or_component_lookup" and qtype not in candidate.row.get(
                "recommended_question_types", []
            ):
                extra_cost += 60
            edge_position = len(graph.graph[node_id])
            graph.add_edge(node_id, qtype_node_base + qtype_index[qtype], 1, extra_cost)
            row_qtype_edges[node_id].append((qtype, edge_position))

    flow, _cost = graph.min_cost_flow(0, sink, total_target)
    warnings: list[str] = []
    if flow != total_target:
        raise ValueError(f"无法满足目标配额，期望 {total_target}，实际 {flow}")

    selections: list[SelectionResult] = []
    for node_id, candidate in row_metadata.items():
        chosen_qtype = None
        for qtype, edge_position in row_qtype_edges[node_id]:
            edge = graph.graph[node_id][edge_position]
            if edge.cap == 0:
                chosen_qtype = qtype
                break
        if chosen_qtype is None:
            continue
        fallback_definition = bool(
            chosen_qtype == "definition_or_component_lookup"
            and chosen_qtype not in candidate.row.get("recommended_question_types", [])
        )
        if fallback_definition:
            warnings.append(f"definition_fallback_used:{candidate.row['chunk_id']}")
        selections.append(
            SelectionResult(
                row=candidate.row,
                question_type=chosen_qtype,
                fallback_definition=fallback_definition,
            )
        )

    return selections, warnings


def row_cost_penalty(row: dict[str, object]) -> int:
    penalty = 0
    penalty += len(row.get("quality_flags", [])) * 10
    penalty += 15 if row["annotation_priority"] == "low" else 0
    penalty += 5 if row["annotation_priority"] == "medium" else 0
    return penalty


def build_generated_candidate(selection: SelectionResult, generator_name: str) -> dict[str, object]:
    row = selection.row
    question_type = selection.question_type
    evidence_quote = select_evidence_quote(row, question_type)
    if not evidence_quote.strip():
        raise ValueError("empty evidence_quote")

    question = build_question(row, question_type)
    answer = build_answer(evidence_quote, question_type)

    candidate_id = build_candidate_id(row, question_type)
    return {
        "candidate_id": candidate_id,
        "source_chunk_ids": [row["chunk_id"]],
        "source_id": row["source_id"],
        "source_file": row["source_file"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "chunk_type": row["chunk_type"],
        "generated_question": question,
        "generated_answer": answer,
        "evidence_quote": evidence_quote,
        "suggested_question_type": question_type,
        "suggested_reasoning_hops": suggest_reasoning_hops(row, question_type),
        "suggested_criticality": suggest_criticality(row, question_type),
        "generator": generator_name,
        "review_status": "pending_review",
    }


def build_candidate_id(row: dict[str, object], question_type: str) -> str:
    stable_key = f"{row['chunk_id']}|{question_type}"
    digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:12]
    return f"{row['source_id']}::{question_type}::{digest}"


def select_evidence_quote(row: dict[str, object], question_type: str) -> str:
    text = str(row["text"]).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:320]

    if question_type == "safety_or_constraint":
        subset = find_line_window(lines, SAFETY_WORDS, max_lines=4)
    elif question_type == "troubleshooting_step":
        subset = find_line_window(lines, ACTION_WORDS, max_lines=5)
    elif question_type == "symptom_cause":
        subset = find_line_window(lines, ("cause", "symptom", "because", "fault", "alarm"), max_lines=5)
    elif question_type == "parameter_or_fault_code":
        anchors = []
        if row.get("parameter_name"):
            anchors.extend(str(row["parameter_name"]).lower().split("; "))
        if row.get("fault_code"):
            anchors.extend(str(row["fault_code"]).lower().split("; "))
        anchors.extend(["parameter", "fault", "alarm", "default", "value"])
        subset = find_line_window(lines, tuple(anchor for anchor in anchors if anchor), max_lines=5)
    else:
        subject = guess_component_name(row)
        subset = find_line_window(lines, (subject.lower(), "module", "motor", "pump", "relay", "controller"), max_lines=4)

    if not subset:
        subset = lines[:4]

    quote = "\n".join(subset).strip()
    if len(quote) > 420:
        quote = quote[:420].rstrip()
    return quote


def find_line_window(lines: list[str], keywords: tuple[str, ...], max_lines: int) -> list[str]:
    lowered = [line.lower() for line in lines]
    for index, line in enumerate(lowered):
        if any(keyword and keyword in line for keyword in keywords):
            end = min(len(lines), index + max_lines)
            return lines[index:end]
    return []


def build_question(row: dict[str, object], question_type: str) -> str:
    product = PRODUCT_LABELS.get(str(row["source_id"]), "device")
    title = simplify_title(str(row["title"]))
    fault_code = first_semicolon_value(str(row.get("fault_code") or ""))
    parameter_name = first_semicolon_value(str(row.get("parameter_name") or ""))
    component = guess_component_name(row)

    if question_type == "troubleshooting_step":
        topic = fault_code or title or component
        return f"How should I troubleshoot {topic} on the {product}?"
    if question_type == "symptom_cause":
        topic = fault_code or title or component
        return f"What could cause {topic} on the {product}?"
    if question_type == "parameter_or_fault_code":
        if fault_code:
            return f"What does {fault_code} indicate on the {product}?"
        if parameter_name:
            return f"What is {parameter_name} used for on the {product}?"
        return f"Which setting or parameter matters for {title} on the {product}?"
    if question_type == "safety_or_constraint":
        topic = title or component
        return f"What safety precaution or limit should I follow for {topic} on the {product}?"
    return f"What does {component} refer to on the {product}?"


def build_answer(evidence_quote: str, question_type: str) -> str:
    normalized = normalize_whitespace(evidence_quote)
    if question_type == "symptom_cause":
        return first_clause(normalized, 240)
    if question_type == "troubleshooting_step":
        return first_clause(normalized, 260)
    if question_type == "parameter_or_fault_code":
        return first_clause(normalized, 220)
    if question_type == "safety_or_constraint":
        return first_clause(normalized, 220)
    return first_clause(normalized, 220)


def first_clause(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    match = re.search(r"(?<=[.;])\s", text[:max_chars])
    if match:
        return text[: match.start() + 1].strip()
    return text[:max_chars].rstrip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def simplify_title(title: str) -> str:
    cleaned = normalize_whitespace(title).strip("-: ")
    if not cleaned or cleaned == "—":
        return "this issue"
    return cleaned


def first_semicolon_value(text: str) -> str:
    if not text:
        return ""
    return text.split(";")[0].strip()


def guess_component_name(row: dict[str, object]) -> str:
    title = simplify_title(str(row["title"]))
    haystack = f"{title} {row['text']}".lower()

    for term in COMPONENT_TERMS:
        if term in haystack:
            return term

    words = re.findall(r"[A-Za-z][A-Za-z0-9/\-]{2,}", title)
    if words:
        return words[0]
    return "this component"


def suggest_reasoning_hops(row: dict[str, object], question_type: str) -> int:
    if question_type == "symptom_cause":
        return 2
    if question_type == "troubleshooting_step":
        return 2 if "fault" in str(row["text"]).lower() or "alarm" in str(row["text"]).lower() else 1
    return 1


def suggest_criticality(row: dict[str, object], question_type: str) -> str:
    if question_type == "safety_or_constraint":
        return "high"
    if question_type == "symptom_cause":
        return "high" if row["chunk_type"] == "alarm_fault_code" else "medium"
    if question_type == "troubleshooting_step":
        return "high" if row["chunk_type"] == "alarm_fault_code" else "medium"
    if question_type == "parameter_or_fault_code":
        return "medium"
    return "low"


if __name__ == "__main__":
    raise SystemExit(main())
