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

DEFAULT_BATCH2_SOURCE_TARGETS = {
    "grundfos_nbe_nbse_nke_tpe_tped_installation_and_operating_instructions": 35,
    "abb_manual_for_induction_motors_and_generators_en": 30,
    "haascnc_com_rotary_troubleshooting_guide_ngc": 25,
    "rockwell_powerflex_520_series_user_manual_520_um001_en_e": 35,
    "s71200_system_manual": 35,
}

DEFAULT_BATCH2_QUESTION_TYPE_TARGETS = {
    "troubleshooting_step": 30,
    "symptom_cause": 35,
    "parameter_or_fault_code": 20,
    "safety_or_constraint": 25,
    "definition_or_component_lookup": 10,
    "cross_doc_multi": 15,
    "abstention_insufficient_evidence": 25,
}

DEFAULT_BATCH2_CROSS_DOC_SOURCE_TARGETS = {
    "grundfos_nbe_nbse_nke_tpe_tped_installation_and_operating_instructions": 3,
    "abb_manual_for_induction_motors_and_generators_en": 3,
    "haascnc_com_rotary_troubleshooting_guide_ngc": 3,
    "rockwell_powerflex_520_series_user_manual_520_um001_en_e": 3,
    "s71200_system_manual": 3,
}

DEFAULT_BATCH2_ABSTENTION_SOURCE_TARGETS = {
    "grundfos_nbe_nbse_nke_tpe_tped_installation_and_operating_instructions": 7,
    "abb_manual_for_induction_motors_and_generators_en": 5,
    "haascnc_com_rotary_troubleshooting_guide_ngc": 4,
    "rockwell_powerflex_520_series_user_manual_520_um001_en_e": 5,
    "s71200_system_manual": 4,
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
MULTI_DOC_THEMES = {
    "safety_or_constraint": ("warning", "danger", "caution", "must", "do not"),
    "symptom_cause": ("cause", "fault", "alarm", "symptom", "warning"),
    "parameter_or_fault_code": ("parameter", "setting", "default", "value", "limit"),
    "troubleshooting_step": ("check", "inspect", "verify", "replace", "adjust"),
}


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
class CrossDocCandidate:
    primary_source_id: str
    first: dict[str, object]
    second: dict[str, object]
    theme: str
    score: int


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
    parser.add_argument(
        "--existing-candidates",
        default="aiops-docs/experiment/rag/rag_candidate_questions.jsonl",
        help="已有候选题，用于去重 source_chunk_ids",
    )
    parser.add_argument(
        "--reviewed-candidates",
        default="aiops-docs/experiment/rag/rag_candidate_questions.reviewed.jsonl",
        help="已审核候选题，用于避免重复使用 chunk",
    )
    parser.add_argument("--exclude-existing-candidates", action="store_true")
    parser.add_argument("--exclude-reviewed-candidates", action="store_true")
    parser.add_argument("--append", action="store_true", help="追加写入输出文件")
    parser.add_argument("--candidate-prefix", default="", help="候选题 ID 前缀，例如 batch2")
    parser.add_argument(
        "--source-quota",
        action="append",
        default=[],
        help="source 配额，格式 source_id=35，可重复传入",
    )
    parser.add_argument(
        "--question-type-quota",
        action="append",
        default=[],
        help="question type 配额，格式 symptom_cause=30，可重复传入",
    )
    parser.add_argument("--normal-count", type=int, default=None)
    parser.add_argument("--cross-doc-count", type=int, default=0)
    parser.add_argument("--abstention-count", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_targets = parse_quota_args(args.source_quota) or DEFAULT_SOURCE_TARGETS
    question_type_targets = parse_quota_args(args.question_type_quota) or DEFAULT_QUESTION_TYPE_TARGETS
    report = generate_rag_candidate_questions(
        input_path=Path(args.input),
        output_path=Path(args.output),
        report_path=Path(args.report),
        generator=args.generator,
        dry_run_template=args.dry_run_template,
        source_targets=source_targets,
        question_type_targets=question_type_targets,
        existing_candidates_path=Path(args.existing_candidates),
        reviewed_candidates_path=Path(args.reviewed_candidates),
        exclude_existing_candidates=args.exclude_existing_candidates,
        exclude_reviewed_candidates=args.exclude_reviewed_candidates,
        append=args.append,
        candidate_prefix=args.candidate_prefix,
        normal_count=args.normal_count,
        cross_doc_count=args.cross_doc_count,
        abstention_count=args.abstention_count,
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
    existing_candidates_path: Path | None = None,
    reviewed_candidates_path: Path | None = None,
    exclude_existing_candidates: bool = False,
    exclude_reviewed_candidates: bool = False,
    append: bool = False,
    candidate_prefix: str = "",
    normal_count: int | None = None,
    cross_doc_count: int = 0,
    abstention_count: int = 0,
) -> dict[str, object]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    raw_source_targets = dict(source_targets or DEFAULT_SOURCE_TARGETS)
    raw_question_type_targets = dict(question_type_targets or DEFAULT_QUESTION_TYPE_TARGETS)
    total_target = sum(raw_question_type_targets.values())
    candidate_prefix = candidate_prefix.strip()
    if not candidate_prefix and "batch2" in output_path.stem:
        candidate_prefix = "batch2"

    normal_qtypes = [
        qtype
        for qtype in raw_question_type_targets
        if qtype not in {"cross_doc_multi", "abstention_insufficient_evidence"}
    ]
    if normal_count is None:
        normal_count = sum(raw_question_type_targets.get(qtype, 0) for qtype in normal_qtypes)
    cross_doc_count = int(cross_doc_count or raw_question_type_targets.get("cross_doc_multi", 0))
    abstention_count = int(
        abstention_count or raw_question_type_targets.get("abstention_insufficient_evidence", 0)
    )

    normal_question_type_targets = {
        qtype: int(raw_question_type_targets.get(qtype, 0))
        for qtype in normal_qtypes
        if int(raw_question_type_targets.get(qtype, 0)) > 0
    }
    if sum(normal_question_type_targets.values()) != normal_count:
        raise ValueError("normal single-doc question_type targets 总数必须等于 normal_count")

    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row_map = {str(row["chunk_id"]): row for row in rows}

    exclusion_chunk_ids = set()
    if exclude_existing_candidates and existing_candidates_path and existing_candidates_path.exists():
        exclusion_chunk_ids.update(load_candidate_chunk_ids(existing_candidates_path.resolve()))
    if exclude_reviewed_candidates and reviewed_candidates_path and reviewed_candidates_path.exists():
        exclusion_chunk_ids.update(load_candidate_chunk_ids(reviewed_candidates_path.resolve()))

    filtered_rows = [row for row in rows if str(row["chunk_id"]) not in exclusion_chunk_ids]
    excluded_existing_chunk_count = len(rows) - len(filtered_rows)

    generator_name = "template_dry_run" if dry_run_template or generator == "template" else generator
    warnings: list[str] = []
    if exclude_existing_candidates:
        warnings.append(f"exclude_existing_candidates:{len(exclusion_chunk_ids)}")
    if exclude_reviewed_candidates:
        warnings.append(f"exclude_reviewed_candidates:{len(exclusion_chunk_ids)}")

    cross_doc_targets = derive_cross_doc_source_targets(raw_source_targets, cross_doc_count)
    abstention_targets = derive_abstention_source_targets(raw_source_targets, abstention_count)
    nominal_normal_source_targets = derive_normal_source_targets(
        raw_source_targets,
        cross_doc_targets,
        abstention_targets,
    )
    if sum(nominal_normal_source_targets.values()) != normal_count:
        raise ValueError("normal source targets 总数必须等于 normal_count")

    used_chunk_ids: set[str] = set()
    cross_doc_rows, cross_doc_warnings = build_cross_doc_candidates(
        rows=filtered_rows,
        source_targets=cross_doc_targets,
        count=cross_doc_count,
        generator_name=generator_name,
        candidate_prefix=candidate_prefix,
        used_chunk_ids=used_chunk_ids,
    )
    warnings.extend(cross_doc_warnings)
    used_chunk_ids.update(iter_chunk_ids(cross_doc_rows))

    abstention_rows, abstention_warnings = build_abstention_candidates(
        rows=filtered_rows,
        source_targets=abstention_targets,
        count=abstention_count,
        generator_name=generator_name,
        candidate_prefix=candidate_prefix,
        used_chunk_ids=used_chunk_ids,
    )
    warnings.extend(abstention_warnings)
    used_chunk_ids.update(iter_chunk_ids(abstention_rows))

    normal_source_targets = rebalance_normal_source_targets(
        nominal_targets=nominal_normal_source_targets,
        filtered_rows=filtered_rows,
        used_chunk_ids=used_chunk_ids,
        normal_count=normal_count,
    )
    candidates = prepare_candidates(filtered_rows, normal_source_targets, used_chunk_ids)
    selections, selection_warnings = select_candidates(
        candidates,
        normal_source_targets,
        normal_question_type_targets,
    )
    warnings.extend(selection_warnings)

    failed_generation_count = 0
    normal_rows: list[dict[str, object]] = []
    flagged_review_rows = 0
    for selection in selections:
        try:
            row = build_generated_candidate(selection, generator_name, candidate_prefix)
        except Exception:
            failed_generation_count += 1
            continue
        if selection.row.get("quality_flags") or selection.fallback_definition:
            flagged_review_rows += 1
        normal_rows.append(row)
        used_chunk_ids.update(row["source_chunk_ids"])

    if flagged_review_rows:
        warnings.append(f"selected_rows_need_manual_focus:{flagged_review_rows}")

    generated_rows = normal_rows + cross_doc_rows + abstention_rows
    existing_rows = []
    if append and output_path.exists():
        existing_rows = read_jsonl(output_path)
    all_rows = existing_rows + generated_rows
    all_rows.sort(
        key=lambda row: (
            primary_sort_source(row),
            safe_int(row.get("page_start")),
            str(row["candidate_id"]),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in all_rows)
    output_path.write_text(output_text + ("\n" if all_rows else ""), encoding="utf-8")

    reused_chunk_count = count_reused_chunks(generated_rows)
    if reused_chunk_count:
        warnings.append(f"reused_chunks_in_batch:{reused_chunk_count}")

    skipped_chunks = {
        "input_pool_rows": len(rows),
        "eligible_rows_considered": len(candidates),
        "excluded_existing_rows": excluded_existing_chunk_count,
        "generated_rows": len(generated_rows),
    }
    if cross_doc_count:
        skipped_chunks["cross_doc_generated"] = len(cross_doc_rows)
    if abstention_count:
        skipped_chunks["abstention_generated"] = len(abstention_rows)

    report = {
        "total_candidates": len(generated_rows),
        "count_by_source": dict(sorted(Counter(primary_sort_source(row) for row in generated_rows).items())),
        "count_by_question_type": dict(
            sorted(Counter(str(row["suggested_question_type"]) for row in generated_rows).items())
        ),
        "reused_chunk_count": reused_chunk_count,
        "excluded_existing_chunk_count": excluded_existing_chunk_count,
        "failed_generation_count": failed_generation_count,
        "skipped_chunks": skipped_chunks,
        "warnings": warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def derive_cross_doc_source_targets(source_targets: dict[str, int], cross_doc_count: int) -> dict[str, int]:
    if cross_doc_count <= 0:
        return {source_id: 0 for source_id in source_targets}
    base = cross_doc_count // len(source_targets)
    remainder = cross_doc_count % len(source_targets)
    result = {}
    ordered = sorted(source_targets, key=lambda item: (-source_targets[item], item))
    for index, source_id in enumerate(ordered):
        result[source_id] = base + (1 if index < remainder else 0)
    return result


def derive_abstention_source_targets(source_targets: dict[str, int], abstention_count: int) -> dict[str, int]:
    default_weights = DEFAULT_BATCH2_ABSTENTION_SOURCE_TARGETS
    ordered = sorted(source_targets, key=lambda item: (-source_targets[item], item))
    if abstention_count <= 0:
        return {source_id: 0 for source_id in ordered}
    raw = {source_id: default_weights.get(source_id, 1) for source_id in ordered}
    total_raw = sum(raw.values())
    scaled = {source_id: int(abstention_count * raw[source_id] / total_raw) for source_id in ordered}
    assigned = sum(scaled.values())
    idx = 0
    while assigned < abstention_count:
        source_id = ordered[idx % len(ordered)]
        scaled[source_id] += 1
        assigned += 1
        idx += 1
    while assigned > abstention_count:
        source_id = ordered[idx % len(ordered)]
        if scaled[source_id] > 0:
            scaled[source_id] -= 1
            assigned -= 1
        idx += 1
    return scaled


def derive_normal_source_targets(
    source_targets: dict[str, int],
    cross_doc_targets: dict[str, int],
    abstention_targets: dict[str, int],
) -> dict[str, int]:
    result = {}
    for source_id, total in source_targets.items():
        result[source_id] = total - cross_doc_targets.get(source_id, 0) - abstention_targets.get(source_id, 0)
    return result


def rebalance_normal_source_targets(
    *,
    nominal_targets: dict[str, int],
    filtered_rows: list[dict[str, object]],
    used_chunk_ids: set[str],
    normal_count: int,
) -> dict[str, int]:
    remaining_available = Counter()
    for row in filtered_rows:
        if str(row["chunk_id"]) in used_chunk_ids:
            continue
        remaining_available[str(row["source_id"])] += 1

    adjusted = {
        source_id: min(int(target), int(remaining_available.get(source_id, 0)))
        for source_id, target in nominal_targets.items()
    }
    assigned = sum(adjusted.values())
    if assigned > normal_count:
        ordered = sorted(adjusted, key=lambda item: (adjusted[item] - nominal_targets[item], item))
        index = 0
        while assigned > normal_count and index < len(ordered):
            source_id = ordered[index]
            if adjusted[source_id] > 0:
                adjusted[source_id] -= 1
                assigned -= 1
            else:
                index += 1
        return adjusted

    if assigned < normal_count:
        ordered = sorted(
            nominal_targets,
            key=lambda item: (remaining_available.get(item, 0) - adjusted[item], nominal_targets[item], item),
            reverse=True,
        )
        index = 0
        while assigned < normal_count and ordered:
            source_id = ordered[index % len(ordered)]
            if adjusted[source_id] < remaining_available.get(source_id, 0):
                adjusted[source_id] += 1
                assigned += 1
            index += 1
            if index > len(ordered) * max(normal_count, 1):
                break
    return adjusted


def parse_quota_args(items: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"quota 格式错误: {item}")
        key, value = item.split("=", 1)
        result[key.strip()] = int(value.strip())
    return result


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_candidate_chunk_ids(path: Path) -> set[str]:
    chunk_ids: set[str] = set()
    for row in read_jsonl(path):
        for field in ("source_chunk_ids", "weak_evidence_chunk_ids"):
            for chunk_id in row.get(field, []) or []:
                chunk_ids.add(str(chunk_id))
    return chunk_ids


def prepare_candidates(
    rows: list[dict[str, object]],
    source_targets: dict[str, int],
    used_chunk_ids: set[str],
) -> list[PoolCandidate]:
    rows_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["source_id"] not in source_targets:
            continue
        if source_targets[str(row["source_id"])] <= 0:
            continue
        if str(row["chunk_id"]) in used_chunk_ids:
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

    bucket_order = sorted(bucketed, key=lambda bucket_id: (-row_priority_score(bucketed[bucket_id][0]), bucket_id))
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
    score = PRIORITY_ORDER[str(row.get("annotation_priority", "low"))] * 1000
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
            if qtype not in qtype_index:
                continue
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


def build_generated_candidate(
    selection: SelectionResult,
    generator_name: str,
    candidate_prefix: str,
) -> dict[str, object]:
    row = selection.row
    question_type = selection.question_type
    evidence_quote = select_evidence_quote(row, question_type)
    if not evidence_quote.strip():
        raise ValueError("empty evidence_quote")

    question = build_question(row, question_type)
    answer = build_answer(evidence_quote, question_type)
    candidate_id = build_candidate_id(row["chunk_id"], question_type, candidate_prefix)
    return {
        "candidate_id": candidate_id,
        "source_chunk_ids": [row["chunk_id"]],
        "source_id": row["source_id"],
        "source_ids": [row["source_id"]],
        "source_file": row["source_file"],
        "source_files": [row["source_file"]],
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
        "should_abstain": False,
    }


def build_cross_doc_candidates(
    *,
    rows: list[dict[str, object]],
    source_targets: dict[str, int],
    count: int,
    generator_name: str,
    candidate_prefix: str,
    used_chunk_ids: set[str],
) -> tuple[list[dict[str, object]], list[str]]:
    if count <= 0:
        return [], []
    warnings: list[str] = []
    rows_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["chunk_id"]) in used_chunk_ids:
            continue
        rows_by_source[str(row["source_id"])].append(row)
    for source_id, source_rows in rows_by_source.items():
        source_rows.sort(key=lambda row: (-row_priority_score(row), int(row["page_start"]), str(row["chunk_id"])))

    selected_rows: list[dict[str, object]] = []
    local_used = set(used_chunk_ids)
    selected_count_by_source = Counter()

    for primary_source_id in sorted(source_targets, key=lambda item: (-source_targets[item], item)):
        target = source_targets.get(primary_source_id, 0)
        if target <= 0:
            continue
        produced = 0
        for theme in ("symptom_cause", "safety_or_constraint", "parameter_or_fault_code", "troubleshooting_step"):
            if produced >= target:
                break
            for candidate in build_cross_doc_pairs_for_source(
                primary_source_id=primary_source_id,
                rows_by_source=rows_by_source,
                theme=theme,
                local_used=local_used,
            ):
                if str(candidate.first["chunk_id"]) in local_used or str(candidate.second["chunk_id"]) in local_used:
                    continue
                selected_rows.append(
                    build_cross_doc_row(candidate, generator_name=generator_name, candidate_prefix=candidate_prefix)
                )
                local_used.add(str(candidate.first["chunk_id"]))
                local_used.add(str(candidate.second["chunk_id"]))
                selected_count_by_source[primary_source_id] += 1
                produced += 1
                if produced >= target or len(selected_rows) >= count:
                    break
            if len(selected_rows) >= count:
                break
        if len(selected_rows) >= count:
            break

    if len(selected_rows) < count:
        warnings.append(f"cross_doc_multi_shortfall:{count - len(selected_rows)}")
    return selected_rows[:count], warnings


def build_cross_doc_pairs_for_source(
    *,
    primary_source_id: str,
    rows_by_source: dict[str, list[dict[str, object]]],
    theme: str,
    local_used: set[str],
) -> list[CrossDocCandidate]:
    primary_rows = [row for row in rows_by_source.get(primary_source_id, []) if str(row["chunk_id"]) not in local_used]
    other_source_ids = [source_id for source_id in sorted(rows_by_source) if source_id != primary_source_id]
    pairs: list[CrossDocCandidate] = []
    for first in primary_rows[:12]:
        for other_source_id in other_source_ids:
            secondary_rows = [
                row for row in rows_by_source.get(other_source_id, []) if str(row["chunk_id"]) not in local_used
            ]
            for second in secondary_rows[:8]:
                score = cross_doc_pair_score(first, second, theme)
                if score <= 0:
                    continue
                pairs.append(
                    CrossDocCandidate(
                        primary_source_id=primary_source_id,
                        first=first,
                        second=second,
                        theme=theme,
                        score=score,
                    )
                )
    pairs.sort(key=lambda item: (-item.score, int(item.first["page_start"]), int(item.second["page_start"])))
    return pairs


def cross_doc_pair_score(first: dict[str, object], second: dict[str, object], theme: str) -> int:
    first_text = f"{first['title']} {first['text']}".lower()
    second_text = f"{second['title']} {second['text']}".lower()
    keywords = MULTI_DOC_THEMES[theme]
    first_hits = sum(keyword in first_text for keyword in keywords)
    second_hits = sum(keyword in second_text for keyword in keywords)
    if first_hits == 0 or second_hits == 0:
        return 0
    score = row_priority_score(first) + row_priority_score(second)
    score += 120 if first["source_id"] != second["source_id"] else 0
    score += (first_hits + second_hits) * 15
    return score


def build_cross_doc_row(
    candidate: CrossDocCandidate,
    *,
    generator_name: str,
    candidate_prefix: str,
) -> dict[str, object]:
    first_quote = select_evidence_quote(candidate.first, candidate.theme)
    second_quote = select_evidence_quote(candidate.second, candidate.theme)
    question = build_cross_doc_question(candidate)
    answer = build_cross_doc_answer(first_quote, second_quote)
    primary = candidate.first
    secondary = candidate.second
    candidate_id = build_candidate_id(
        f"{primary['chunk_id']}|{secondary['chunk_id']}",
        "cross_doc_multi",
        candidate_prefix,
    )
    return {
        "candidate_id": candidate_id,
        "source_chunk_ids": [primary["chunk_id"], secondary["chunk_id"]],
        "source_id": primary["source_id"],
        "source_ids": [primary["source_id"], secondary["source_id"]],
        "source_file": primary["source_file"],
        "source_files": [primary["source_file"], secondary["source_file"]],
        "page_start": primary["page_start"],
        "page_end": secondary["page_end"],
        "page_spans": [
            {"source_id": primary["source_id"], "page_start": primary["page_start"], "page_end": primary["page_end"]},
            {"source_id": secondary["source_id"], "page_start": secondary["page_start"], "page_end": secondary["page_end"]},
        ],
        "chunk_type": "cross_doc_multi",
        "generated_question": question,
        "generated_answer": answer,
        "evidence_quote": first_quote + "\n\n" + second_quote,
        "suggested_question_type": "cross_doc_multi",
        "suggested_reasoning_hops": "multi_doc",
        "suggested_criticality": "medium",
        "generator": generator_name,
        "review_status": "pending_review",
        "should_abstain": False,
    }


def build_cross_doc_question(candidate: CrossDocCandidate) -> str:
    first_product = PRODUCT_LABELS.get(str(candidate.first["source_id"]), "device A")
    second_product = PRODUCT_LABELS.get(str(candidate.second["source_id"]), "device B")
    if candidate.theme == "safety_or_constraint":
        return f"What common safety precaution stands out when comparing the {first_product} and {second_product} manuals?"
    if candidate.theme == "parameter_or_fault_code":
        return f"What common setup or parameter-related constraint appears across the {first_product} and {second_product} manuals?"
    if candidate.theme == "troubleshooting_step":
        return f"If I compare the troubleshooting guidance in the {first_product} and {second_product} manuals, what first action is consistently emphasized?"
    return f"When comparing the {first_product} and {second_product} manuals, what common cause or symptom pattern appears across both documents?"


def build_cross_doc_answer(first_quote: str, second_quote: str) -> str:
    first_part = first_clause(normalize_whitespace(first_quote), 140)
    second_part = first_clause(normalize_whitespace(second_quote), 140)
    return f"Both documents point to related evidence: {first_part} / {second_part}"


def build_abstention_candidates(
    *,
    rows: list[dict[str, object]],
    source_targets: dict[str, int],
    count: int,
    generator_name: str,
    candidate_prefix: str,
    used_chunk_ids: set[str],
) -> tuple[list[dict[str, object]], list[str]]:
    if count <= 0:
        return [], []
    warnings: list[str] = []
    selected_rows: list[dict[str, object]] = []
    local_used = set(used_chunk_ids)
    rows_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["chunk_id"]) in local_used:
            continue
        rows_by_source[str(row["source_id"])].append(row)
    for source_id, source_rows in rows_by_source.items():
        source_rows.sort(key=lambda row: (-row_priority_score(row), int(row["page_start"]), str(row["chunk_id"])))

    for source_id in sorted(source_targets, key=lambda item: (-source_targets[item], item)):
        target = source_targets.get(source_id, 0)
        produced = 0
        for row in rows_by_source.get(source_id, []):
            chunk_id = str(row["chunk_id"])
            if chunk_id in local_used:
                continue
            selected_rows.append(
                build_abstention_row(row, generator_name=generator_name, candidate_prefix=candidate_prefix)
            )
            local_used.add(chunk_id)
            produced += 1
            if produced >= target or len(selected_rows) >= count:
                break
        if len(selected_rows) >= count:
            break
    if len(selected_rows) < count:
        warnings.append(f"abstention_shortfall:{count - len(selected_rows)}")
    return selected_rows[:count], warnings


def build_abstention_row(
    row: dict[str, object],
    *,
    generator_name: str,
    candidate_prefix: str,
) -> dict[str, object]:
    product = PRODUCT_LABELS.get(str(row["source_id"]), "device")
    component = guess_component_name(row)
    evidence_quote = ""
    abstention_reason = (
        "The available manual evidence is not sufficient to answer reliably; this question needs live operating data, site-specific conditions, or external diagnostics."
    )
    generated_question = build_abstention_question(row, product, component)
    candidate_id = build_candidate_id(row["chunk_id"], "abstention_insufficient_evidence", candidate_prefix)
    return {
        "candidate_id": candidate_id,
        "source_chunk_ids": [],
        "weak_evidence_chunk_ids": [row["chunk_id"]],
        "source_id": row["source_id"],
        "source_ids": [row["source_id"]],
        "source_file": row["source_file"],
        "source_files": [row["source_file"]],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "chunk_type": row["chunk_type"],
        "generated_question": generated_question,
        "generated_answer": abstention_reason,
        "reference_answer": abstention_reason,
        "evidence_quote": evidence_quote,
        "suggested_question_type": "abstention_insufficient_evidence",
        "suggested_reasoning_hops": "abstention",
        "suggested_criticality": "medium",
        "generator": generator_name,
        "review_status": "pending_review",
        "should_abstain": True,
        "abstention_reason": abstention_reason,
    }


def build_abstention_question(row: dict[str, object], product: str, component: str) -> str:
    if row.get("fault_code"):
        fault_code = first_semicolon_value(str(row["fault_code"]))
        return f"My {product} is showing {fault_code}; can you tell me the exact root cause right now without any live measurements?"
    if row.get("parameter_name"):
        parameter_name = first_semicolon_value(str(row["parameter_name"]))
        return f"What exact {parameter_name} value should I apply on the {product} for my current现场 condition if the manual does not include the needed operating data?"
    if str(row["chunk_type"]) == "safety_and_constraint":
        return f"Can I confirm that the {product} is safe to restart right now without现场 inspection or live electrical measurements?"
    return f"Can you determine whether the {component} on the {product} must be repaired immediately based only on the manual, without current现场 data?"


def build_candidate_id(stable_key: str, question_type: str, candidate_prefix: str) -> str:
    digest = hashlib.sha1(f"{stable_key}|{question_type}|{candidate_prefix}".encode("utf-8")).hexdigest()[:12]
    if candidate_prefix:
        return f"{candidate_prefix}::{question_type}::{digest}"
    if "::" in stable_key:
        source_id = stable_key.split("::", 1)[0]
    else:
        source_id = "candidate"
    return f"{source_id}::{question_type}::{digest}"


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


def suggest_reasoning_hops(row: dict[str, object], question_type: str) -> str:
    if question_type == "symptom_cause":
        return "multi_chunk_same_doc"
    if question_type == "troubleshooting_step":
        return "multi_chunk_same_doc" if "fault" in str(row["text"]).lower() or "alarm" in str(row["text"]).lower() else "single_chunk"
    return "single_chunk"


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


def count_reused_chunks(rows: list[dict[str, object]]) -> int:
    counts = Counter()
    for row in rows:
        for field in ("source_chunk_ids", "weak_evidence_chunk_ids"):
            for chunk_id in row.get(field, []) or []:
                counts[str(chunk_id)] += 1
    return sum(count - 1 for count in counts.values() if count > 1)


def iter_chunk_ids(rows: list[dict[str, object]]) -> set[str]:
    chunk_ids: set[str] = set()
    for row in rows:
        for field in ("source_chunk_ids", "weak_evidence_chunk_ids"):
            for chunk_id in row.get(field, []) or []:
                chunk_ids.add(str(chunk_id))
    return chunk_ids


def primary_sort_source(row: dict[str, object]) -> str:
    return str(row.get("source_id") or "")


def safe_int(value: object) -> int:
    if value in ("", None):
        return 0
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
