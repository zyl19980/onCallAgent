"""对正式 RAG 数据集做 split。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

VALID_SPLITS = {"build", "dev", "test", "reserve"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="切分正式 RAG 数据集")
    parser.add_argument(
        "--dataset",
        default="aiops-docs/experiment/rag/experiment_rag_dataset.validated.jsonl",
        help="validated RAG dataset JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="aiops-docs/experiment/rag/splits",
        help="split 输出目录",
    )
    parser.add_argument(
        "--mode",
        default="rebalance",
        choices=["respect_existing_split", "rebalance"],
        help="split 模式，默认 rebalance",
    )
    parser.add_argument("--build-size", type=int, default=30, help="build 目标数量")
    parser.add_argument("--dev-size", type=int, default=20, help="dev 目标数量")
    parser.add_argument("--test-size", type=int, default=0, help="test 目标数量")
    parser.add_argument(
        "--reserve-rest",
        action="store_true",
        help="将剩余样本全部放入 reserve",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = split_rag_dataset(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        mode=args.mode,
        build_size=args.build_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        reserve_rest=args.reserve_rest,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def split_rag_dataset(
    dataset_path: Path,
    output_dir: Path,
    mode: str = "rebalance",
    build_size: int = 30,
    dev_size: int = 20,
    test_size: int = 0,
    reserve_rest: bool = False,
    seed: int = 42,
) -> dict[str, object]:
    dataset_path = dataset_path.resolve()
    output_dir = output_dir.resolve()
    rows = load_jsonl(dataset_path)

    if mode == "respect_existing_split":
        assigned_rows = [dict(row) for row in rows]
    else:
        assigned_rows = rebalance_rows(
            rows=rows,
            build_size=build_size,
            dev_size=dev_size,
            test_size=test_size,
            reserve_rest=reserve_rest,
            seed=seed,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = {split: [] for split in ("build", "dev", "test", "reserve")}
    for row in assigned_rows:
        split = str(row.get("split") or "reserve")
        if split not in split_rows:
            split = "reserve"
            row["split"] = split
        split_rows[split].append(row)

    write_jsonl(output_dir / "rag_build.jsonl", split_rows["build"])
    write_jsonl(output_dir / "rag_dev.jsonl", split_rows["dev"])
    write_jsonl(output_dir / "rag_test.jsonl", split_rows["test"])
    write_jsonl(output_dir / "rag_reserve.jsonl", split_rows["reserve"])

    leakage_warnings = detect_leakage_warnings(assigned_rows)
    report = {
        "total_samples": len(assigned_rows),
        "count_by_split": {
            split: len(split_rows[split]) for split in ("build", "dev", "test", "reserve")
        },
        "count_by_source_per_split": {
            split: dict(
                sorted(
                    Counter(source for row in split_rows[split] for source in row.get("source_ids", [])).items()
                )
            )
            for split in ("build", "dev", "test", "reserve")
        },
        "count_by_question_type_per_split": {
            split: dict(sorted(Counter(row["question_type"] for row in split_rows[split]).items()))
            for split in ("build", "dev", "test", "reserve")
        },
        "count_by_should_abstain_per_split": {
            split: dict(
                sorted(
                    Counter(
                        "true" if row.get("should_abstain") else "false"
                        for row in split_rows[split]
                    ).items()
                )
            )
            for split in ("build", "dev", "test", "reserve")
        },
        "leakage_warnings": leakage_warnings,
        "seed": seed,
    }
    (output_dir / "rag_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def rebalance_rows(
    rows: list[dict[str, object]],
    build_size: int,
    dev_size: int,
    test_size: int,
    reserve_rest: bool,
    seed: int,
) -> list[dict[str, object]]:
    total = len(rows)
    assigned_targets = {"build": build_size, "dev": dev_size, "test": test_size}
    target_sum = sum(assigned_targets.values())
    if target_sum > total:
        raise ValueError(f"目标 split 数量超过样本总数: {target_sum} > {total}")

    reserve_size = total - target_sum if reserve_rest else 0
    if not reserve_rest and target_sum != total:
        raise ValueError("未启用 --reserve-rest 时，build/dev/test 总数必须等于样本总数")
    targets = {
        "build": build_size,
        "dev": dev_size,
        "test": test_size,
        "reserve": reserve_size,
    }

    groups = build_leakage_groups(rows)
    overall_source_counts = Counter(source for row in rows for source in row.get("source_ids", []))
    overall_qtype_counts = Counter(str(row["question_type"]) for row in rows)
    overall_abstain_counts = Counter(
        "true" if row.get("should_abstain") else "false" for row in rows
    )
    source_targets = build_label_targets(overall_source_counts, targets)
    qtype_targets = build_label_targets(overall_qtype_counts, targets)
    abstain_targets = build_label_targets(overall_abstain_counts, targets)
    rng = random.Random(seed)

    split_rows = {split: [] for split in ("build", "dev", "test", "reserve")}
    current_source = {split: Counter() for split in split_rows}
    current_qtype = {split: Counter() for split in split_rows}
    current_abstain = {split: Counter() for split in split_rows}
    source_quota_by_source = invert_split_label_targets(source_targets)
    groups_by_source: dict[str, list[list[dict[str, object]]]] = defaultdict(list)
    for group in groups:
        groups_by_source[group_primary_source(group)].append(group)

    source_order = sorted(
        groups_by_source,
        key=lambda source: (
            -sum(len(group) for group in groups_by_source[source]),
            source,
        ),
    )

    for source in source_order:
        source_groups = groups_by_source[source]
        source_groups.sort(
            key=lambda group: (
                -len(group),
                min(int(min_page(row)) for row in group),
                min(str(row["id"]) for row in group),
            )
        )
        rng.shuffle(source_groups)
        source_groups.sort(
            key=lambda group: (
                -len(group),
                min(int(min_page(row)) for row in group),
                min(str(row["id"]) for row in group),
            )
        )

        for group in source_groups:
            options = []
            for split in ("build", "dev", "test", "reserve"):
                remaining = targets[split] - len(split_rows[split])
                source_remaining = source_quota_by_source[source].get(split, 0) - current_source[split][source]
                if remaining < len(group):
                    continue
                if source_remaining < len(group):
                    continue
                score = assignment_score(
                    split=split,
                    group=group,
                    targets=targets,
                        split_rows=split_rows,
                        current_source=current_source,
                        current_qtype=current_qtype,
                        current_abstain=current_abstain,
                        source_targets=source_targets,
                        qtype_targets=qtype_targets,
                        abstain_targets=abstain_targets,
                    )
                options.append((score, split))

            if not options:
                for split in ("build", "dev", "test", "reserve"):
                    remaining = targets[split] - len(split_rows[split])
                    if remaining < len(group):
                        continue
                    score = assignment_score(
                        split=split,
                        group=group,
                        targets=targets,
                        split_rows=split_rows,
                        current_source=current_source,
                        current_qtype=current_qtype,
                        current_abstain=current_abstain,
                        source_targets=source_targets,
                        qtype_targets=qtype_targets,
                        abstain_targets=abstain_targets,
                    )
                    options.append((score + 500, split))

            if not options:
                raise ValueError("无法满足目标 split 数量，存在无法放置的 group")

            best_score = min(score for score, _split in options)
            best_splits = [split for score, split in options if math.isclose(score, best_score)]
            chosen_split = rng.choice(sorted(best_splits))

            for row in group:
                updated = dict(row)
                updated["split"] = chosen_split
                split_rows[chosen_split].append(updated)
                for row_source in updated.get("source_ids", []):
                    current_source[chosen_split][row_source] += 1
                current_qtype[chosen_split][str(updated["question_type"])] += 1
                current_abstain[chosen_split]["true" if updated.get("should_abstain") else "false"] += 1

    output: list[dict[str, object]] = []
    for split in ("build", "dev", "test", "reserve"):
        output.extend(sorted(split_rows[split], key=lambda row: str(row["id"])))
    return output


def build_leakage_groups(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    chunk_to_indices: dict[str, list[int]] = defaultdict(list)
    source_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        for chunk_id in row.get("reference_chunk_ids", []):
            chunk_to_indices[str(chunk_id)].append(idx)
        for source_id in row.get("source_ids", []):
            source_to_indices[str(source_id)].append(idx)

    for indices in chunk_to_indices.values():
        if len(indices) > 1:
            base = indices[0]
            for other in indices[1:]:
                union(base, other)

    for source_id, indices in source_to_indices.items():
        sorted_indices = sorted(indices, key=lambda i: min_page(rows[i]))
        for left, right in zip(sorted_indices, sorted_indices[1:]):
            if page_overlap_or_near(rows[left], rows[right]):
                union(left, right)

    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[find(idx)].append(dict(row))
    return list(groups.values())


def min_page(row: dict[str, object]) -> int:
    pages = row.get("expected_page_numbers", [])
    return min(pages) if pages else int(row.get("page_start", 0) or 0)


def page_overlap_or_near(left: dict[str, object], right: dict[str, object]) -> bool:
    left_pages = set(int(page) for page in left.get("expected_page_numbers", []))
    right_pages = set(int(page) for page in right.get("expected_page_numbers", []))
    return bool(left_pages & right_pages)


def assignment_score(
    split: str,
    group: list[dict[str, object]],
    targets: dict[str, int],
    split_rows: dict[str, list[dict[str, object]]],
    current_source: dict[str, Counter],
    current_qtype: dict[str, Counter],
    current_abstain: dict[str, Counter],
    source_targets: dict[str, dict[str, int]],
    qtype_targets: dict[str, dict[str, int]],
    abstain_targets: dict[str, dict[str, int]],
) -> float:
    target_size = targets[split]
    if target_size == 0:
        return 10_000.0 if group else 0.0

    new_size = len(split_rows[split]) + len(group)
    score = abs(target_size - new_size) * 10

    projected_source = current_source[split].copy()
    projected_qtype = current_qtype[split].copy()
    projected_abstain = current_abstain[split].copy()
    for row in group:
        for source in row.get("source_ids", []):
            projected_source[source] += 1
        projected_qtype[str(row["question_type"])] += 1
        projected_abstain["true" if row.get("should_abstain") else "false"] += 1

    group_sources = {source for row in group for source in row.get("source_ids", [])}
    group_qtypes = {str(row["question_type"]) for row in group}
    group_abstain_keys = {"true" if row.get("should_abstain") else "false" for row in group}

    for source in group_sources:
        target = source_targets[split].get(source, 0)
        current = current_source[split][source]
        projected = projected_source[source]
        score += abs(projected - target) * 6
        if projected > target:
            score += (projected - target) * 14
        if current >= target and target > 0:
            score += 25

    for qtype in group_qtypes:
        target = qtype_targets[split].get(qtype, 0)
        current = current_qtype[split][qtype]
        projected = projected_qtype[qtype]
        score += abs(projected - target) * 3
        if projected > target:
            score += (projected - target) * 8
        if current >= target and target > 0:
            score += 10

    for abstain_key in group_abstain_keys:
        target = abstain_targets[split].get(abstain_key, 0)
        current = current_abstain[split][abstain_key]
        projected = projected_abstain[abstain_key]
        score += abs(projected - target) * 4
        if projected > target:
            score += (projected - target) * 8

    if any(row.get("should_abstain") for row in group) or any(
        str(row.get("question_type")) == "cross_doc_multi" for row in group
    ):
        if split == "build":
            score += 18
        elif split in {"dev", "test"}:
            score -= 4

    return score


def build_label_targets(label_counts: Counter, split_targets: dict[str, int]) -> dict[str, dict[str, int]]:
    total = sum(label_counts.values())
    result = {split: {} for split in split_targets}

    for label, count in label_counts.items():
        allocations = {split: 0 for split in split_targets}
        remainders: list[tuple[float, str]] = []
        assigned = 0
        for split, split_target in split_targets.items():
            if total == 0:
                ideal = 0.0
            else:
                ideal = count * (split_target / total)
            base = int(math.floor(ideal))
            allocations[split] = base
            assigned += base
            remainders.append((ideal - base, split))

        remaining = count - assigned
        for _remainder, split in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if remaining <= 0:
                break
            allocations[split] += 1
            remaining -= 1

        for split, allocated in allocations.items():
            result[split][label] = allocated

    return result


def invert_split_label_targets(targets: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = defaultdict(dict)
    for split, label_map in targets.items():
        for label, count in label_map.items():
            output[label][split] = count
    return output


def group_primary_source(group: list[dict[str, object]]) -> str:
    for row in group:
        source_ids = row.get("source_ids", [])
        if source_ids:
            return str(source_ids[0])
    return "__unknown__"


def detect_leakage_warnings(rows: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    chunk_splits: dict[str, set[str]] = defaultdict(set)
    source_page_splits: dict[tuple[str, int], set[str]] = defaultdict(set)

    for row in rows:
        split = str(row.get("split") or "")
        for chunk_id in row.get("reference_chunk_ids", []):
            chunk_splits[str(chunk_id)].add(split)
        for source_id in row.get("source_ids", []):
            for page in row.get("expected_page_numbers", []):
                source_page_splits[(str(source_id), int(page))].add(split)

    for chunk_id, splits in sorted(chunk_splits.items()):
        if len(splits) > 1:
            warnings.append(f"reference_chunk_cross_split:{chunk_id}:{','.join(sorted(splits))}")

    for (source_id, page), splits in sorted(source_page_splits.items()):
        if len(splits) > 1:
            warnings.append(f"source_page_overlap_cross_split:{source_id}:page_{page}:{','.join(sorted(splits))}")

    return warnings


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if rows else ""), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
