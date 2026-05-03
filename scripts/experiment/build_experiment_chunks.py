"""构建毕业论文实验用的 PDF chunk 数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

VALID_CHUNK_TYPES = {
    "front_matter",
    "concept_and_component",
    "parameter_and_configuration",
    "alarm_fault_code",
    "troubleshooting_procedure",
    "safety_and_constraint",
    "maintenance_procedure",
    "installation_or_wiring",
    "other",
}

TARGET_CHUNK_SIZE = 850
MIN_CHUNK_SIZE = 700
MAX_CHUNK_SIZE = 920
CHUNK_OVERLAP = 140

FRONT_MATTER_PATTERNS = (
    re.compile(r"\btable of contents\b", re.IGNORECASE),
    re.compile(r"\bcontents\b", re.IGNORECASE),
    re.compile(r"\bpreface\b", re.IGNORECASE),
    re.compile(r"\blegal information\b", re.IGNORECASE),
    re.compile(r"\bsummary of changes\b", re.IGNORECASE),
    re.compile(r"\badditional resources\b", re.IGNORECASE),
    re.compile(r"\brecently updated\b", re.IGNORECASE),
    re.compile(r"\brevision\b", re.IGNORECASE),
    re.compile(r"\bcopyright\b", re.IGNORECASE),
    re.compile(r"\babout this publication\b", re.IGNORECASE),
    re.compile(r"\bpurpose of the manual\b", re.IGNORECASE),
    re.compile(r"\border numbers\b", re.IGNORECASE),
    re.compile(r"\bglossary\b", re.IGNORECASE),
    re.compile(r"\bindex\b", re.IGNORECASE),
)

SAFETY_KEYWORDS = ("danger", "warning", "caution", "notice", "hazard", "safety")
INSTALL_KEYWORDS = (
    "installation",
    "wiring",
    "mounting",
    "terminal block",
    "terminal",
    "cable",
    "connector",
    "connect",
    "power supply",
    "din rail",
)
MAINTENANCE_KEYWORDS = (
    "maintenance",
    "inspection",
    "repair",
    "replace",
    "replacement",
    "service",
    "lubricat",
    "oil leak",
    "grease",
)
PARAMETER_KEYWORDS = (
    "parameter",
    "configuration",
    "default",
    "setting",
    "set to",
    "range",
    "value",
    "address",
)
ALARM_KEYWORDS = ("alarm", "fault code", "error code", "trip", "servo overload")
TROUBLESHOOTING_KEYWORDS = (
    "troubleshooting",
    "symptom",
    "possible cause",
    "cause",
    "corrective action",
    "troubleshoot",
    "root cause",
)
CONCEPT_KEYWORDS = (
    "overview",
    "component",
    "product overview",
    "system overview",
    "introduction",
    "specifications",
    "features",
)

CASE_BOUNDARY_PATTERNS = (
    re.compile(r"(?im)(?=^alarm\s+[0-9]+(?:\.[0-9]+)*)"),
    re.compile(r"(?im)(?=^fault code\b)"),
    re.compile(r"(?im)(?=^symptom\b)"),
    re.compile(r"(?im)(?=^table\s+\d+[.\-]\d+)"),
)


@dataclass(slots=True)
class SourceEntry:
    source_id: str
    file_name: str
    file_path: str
    file_type: str
    page_count: int
    extractable_pages: int
    text_extractable_ratio: float
    recommended_usage: str
    needs_ocr: bool
    notes: list[str]


@dataclass(slots=True)
class RawChunkUnit:
    source_id: str
    source_file: str
    collection: str
    page_start: int
    page_end: int
    section_path: str
    chunk_type: str
    title: str
    text: str


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    source_file: str
    collection: str
    page_start: int
    page_end: int
    section_path: str
    chunk_index: int
    chunk_type: str
    title: str
    text: str
    fault_code: str
    parameter_name: str
    safety_level: str
    char_count: int
    token_estimate: int
    text_hash: str
    is_annotation_candidate: bool
    exclude_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 experiment_chunks.jsonl")
    parser.add_argument(
        "--source-root",
        default="docs/experiment_doc",
        help="实验 PDF 根目录，默认 docs/experiment_doc",
    )
    parser.add_argument(
        "--manifest",
        default="aiops-docs/experiment/sources/source_manifest.json",
        help="source manifest 路径",
    )
    parser.add_argument(
        "--output",
        default="aiops-docs/experiment/chunks/experiment_chunks.jsonl",
        help="chunk JSONL 输出路径",
    )
    parser.add_argument(
        "--report",
        default="aiops-docs/experiment/chunks/chunk_build_report.json",
        help="chunk 构建报告输出路径",
    )
    parser.add_argument(
        "--max-pages-per-source",
        type=int,
        default=None,
        help="限制每个 source 最多处理页数",
    )
    parser.add_argument(
        "--include-source",
        action="append",
        default=[],
        help="仅处理指定 source_id，可重复或用逗号分隔",
    )
    parser.add_argument(
        "--exclude-source",
        action="append",
        default=[],
        help="排除指定 source_id，可重复或用逗号分隔",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="全局最多输出多少个 chunk",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_experiment_chunks(
        source_root=Path(args.source_root),
        manifest_path=Path(args.manifest),
        output_path=Path(args.output),
        report_path=Path(args.report),
        max_pages_per_source=args.max_pages_per_source,
        include_sources=parse_source_filter_args(args.include_source),
        exclude_sources=parse_source_filter_args(args.exclude_source),
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_source_filter_args(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        for item in value.split(","):
            normalized = item.strip()
            if normalized:
                result.append(normalized)
    return result


def build_experiment_chunks(
    source_root: Path,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
    max_pages_per_source: int | None = None,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    source_root = source_root.resolve()
    manifest_path = manifest_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_entries = [
        SourceEntry(
            source_id=item["source_id"],
            file_name=item["file_name"],
            file_path=item["file_path"],
            file_type=item["file_type"],
            page_count=item["page_count"],
            extractable_pages=item["extractable_pages"],
            text_extractable_ratio=item["text_extractable_ratio"],
            recommended_usage=item.get("recommended_usage", ""),
            needs_ocr=bool(item.get("needs_ocr", False)),
            notes=list(item.get("notes", [])),
        )
        for item in manifest.get("sources", [])
    ]

    include_set = set(include_sources or [])
    exclude_set = set(exclude_sources or [])
    warnings: list[str] = []
    chunks: list[ChunkRecord] = []
    seen_chunk_ids: set[str] = set()

    filtered_sources = []
    for source in source_entries:
        if source.file_type != "pdf":
            warnings.append(f"skip_non_pdf:{source.source_id}")
            continue
        if include_set and source.source_id not in include_set:
            continue
        if source.source_id in exclude_set:
            continue
        if source.needs_ocr:
            warnings.append(f"source_needs_ocr:{source.source_id}")
        filtered_sources.append(source)

    for source in filtered_sources:
        file_path = resolve_source_file_path(source, source_root)
        page_texts = extract_pdf_page_texts(file_path, max_pages=max_pages_per_source)
        if max_pages_per_source is not None and len(page_texts) < source.page_count:
            warnings.append(
                f"truncated_source_pages:{source.source_id}:{len(page_texts)}/{source.page_count}"
            )

        units = build_source_units(source, page_texts)
        source_records = finalize_source_chunks(source, units)

        for record in source_records:
            if record.chunk_id in seen_chunk_ids:
                raise ValueError(f"重复 chunk_id: {record.chunk_id}")
            seen_chunk_ids.add(record.chunk_id)
            chunks.append(record)
            if limit is not None and len(chunks) >= limit:
                warnings.append(f"global_chunk_limit_reached:{limit}")
                break

        if limit is not None and len(chunks) >= limit:
            break

    write_chunk_outputs(output_path, report_path, chunks, warnings)
    return json.loads(report_path.read_text(encoding="utf-8"))


def resolve_source_file_path(source: SourceEntry, source_root: Path) -> Path:
    path = Path(source.file_path)
    if not path.is_absolute():
        repo_relative = Path.cwd().resolve() / path
        if repo_relative.exists():
            return repo_relative
        root_relative = source_root / source.file_name
        if root_relative.exists():
            return root_relative.resolve()
    if path.exists():
        return path.resolve()
    raise ValueError(f"源文件不存在: {source.file_path}")


def extract_pdf_page_texts(file_path: Path, max_pages: int | None = None) -> list[tuple[int, str]]:
    reader = PdfReader(str(file_path))
    page_texts: list[tuple[int, str]] = []

    for index, page in enumerate(reader.pages, start=1):
        if max_pages is not None and index > max_pages:
            break
        text = (page.extract_text() or "").replace("\x00", "").strip()
        page_texts.append((index, text))

    return page_texts


def build_source_units(source: SourceEntry, page_texts: list[tuple[int, str]]) -> list[RawChunkUnit]:
    raw_units: list[RawChunkUnit] = []
    current_section_path = ""

    for page_number, page_text in page_texts:
        cleaned_lines = normalize_page_lines(page_text)
        if not cleaned_lines:
            continue

        full_text = "\n".join(cleaned_lines).strip()
        if looks_like_front_matter(full_text, page_number):
            title = derive_block_title(cleaned_lines, current_section_path)
            current_section_path = choose_section_path(title, current_section_path)
            raw_units.append(
                RawChunkUnit(
                    source_id=source.source_id,
                    source_file=source.file_name,
                    collection=source.source_id,
                    page_start=page_number,
                    page_end=page_number,
                    section_path=current_section_path,
                    chunk_type="front_matter",
                    title=title,
                    text=full_text,
                )
            )
            continue

        blocks = split_page_into_blocks(cleaned_lines, current_section_path)
        for block in blocks:
            block_title = block["title"]
            current_section_path = choose_section_path(block_title, current_section_path)
            raw_units.append(
                RawChunkUnit(
                    source_id=source.source_id,
                    source_file=source.file_name,
                    collection=source.source_id,
                    page_start=page_number,
                    page_end=page_number,
                    section_path=current_section_path,
                    chunk_type=classify_chunk_type(block_title, block["text"], page_number),
                    title=block_title,
                    text=block["text"],
                )
            )

    merged_units = merge_adjacent_units(raw_units)
    return split_long_units(merged_units)


def normalize_page_lines(text: str) -> list[str]:
    normalized = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
        .replace("\t", " ")
        .replace("­", "")
    )
    lines: list[str] = []

    for raw_line in normalized.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if is_noise_line(line):
            continue
        lines.append(line)

    return lines


def is_noise_line(line: str) -> bool:
    if re.fullmatch(r"\d+\s*/\s*\d+", line):
        return True
    if re.fullmatch(r"[_\-.=]{3,}", line):
        return True
    if re.fullmatch(r"\d+", line):
        return True
    if "http://" in line or "https://" in line:
        return True
    if re.search(r"\bpublication\b", line, re.IGNORECASE) and re.search(r"\b\d{4}\b", line):
        return True
    if "System Manual" in line and re.search(r"\bA5E[0-9A-Z\-]+\b", line):
        return True
    return False


def looks_like_front_matter(text: str, page_number: int) -> bool:
    normalized = text.lower()
    if any(pattern.search(normalized) for pattern in FRONT_MATTER_PATTERNS):
        return True
    if page_number <= 2 and ("system manual" in normalized or "user manual" in normalized):
        return True
    if page_number <= 3 and "table of contents" in normalized:
        return True
    toc_lines = re.findall(r"\.{3,}\s*\d+\b", text)
    if len(toc_lines) >= 4:
        return True
    return False


def split_page_into_blocks(lines: list[str], fallback_section_path: str) -> list[dict[str, str]]:
    heading_positions = [
        index
        for index, line in enumerate(lines)
        if is_heading_candidate(line, index=index, total_lines=len(lines))
    ]

    if not heading_positions:
        text = "\n".join(lines).strip()
        return [{"title": derive_block_title(lines, fallback_section_path), "text": text}]

    blocks: list[dict[str, str]] = []
    starts = heading_positions + [len(lines)]

    for start_index, end_index in zip(starts, starts[1:]):
        block_lines = lines[start_index:end_index]
        if not block_lines:
            continue
        text = "\n".join(block_lines).strip()
        if not text:
            continue
        blocks.append(
            {
                "title": derive_block_title(block_lines, fallback_section_path),
                "text": text,
            }
        )

    return blocks or [{"title": derive_block_title(lines, fallback_section_path), "text": "\n".join(lines)}]


def is_heading_candidate(line: str, index: int, total_lines: int) -> bool:
    if len(line) < 3 or len(line) > 120:
        return False
    if re.search(r"[.!?;]$", line):
        return False
    if re.fullmatch(r"[A-Za-z0-9 \-_/().:&]+", line) is None:
        return False

    lowered = line.lower()
    near_top = index <= min(8, total_lines // 2 + 2)

    keyword_heading = (
        re.match(r"^(chapter|appendix|preface|installation|troubleshooting|symptom table|electrical diagrams|legal information|table of contents)\b", lowered)
        is not None
    )
    numbered_heading = (
        re.match(r"^(\d+(\.\d+){0,3}|[a-z])\s+[A-Z]", line) is not None
        or re.match(r"^(chapter|appendix)\s+[0-9A-Z.\-]+", lowered) is not None
    )
    short_title = len(line.split()) <= 12 and title_case_ratio(line) >= 0.7

    return bool(near_top and (keyword_heading or numbered_heading or short_title))


def title_case_ratio(line: str) -> float:
    words = [word for word in re.split(r"\s+", line) if word]
    if not words:
        return 0.0
    title_like = sum(1 for word in words if word[:1].isupper() or word.isupper())
    return title_like / len(words)


def derive_block_title(lines: list[str], fallback_section_path: str) -> str:
    for line in lines[:3]:
        if is_heading_candidate(line, 0, max(len(lines), 1)):
            return trim_title(line)
    if fallback_section_path:
        return trim_title(fallback_section_path)
    return trim_title(lines[0])


def choose_section_path(title: str, fallback_section_path: str) -> str:
    title = trim_title(title)
    if title and title.lower() != "untitled":
        if fallback_section_path and title.lower() in fallback_section_path.lower():
            return fallback_section_path
        return title
    return fallback_section_path or "Untitled"


def trim_title(text: str) -> str:
    title = re.sub(r"\s+", " ", text).strip(" -:\t")
    return title[:180] if title else "Untitled"


def classify_chunk_type(title: str, text: str, page_number: int) -> str:
    title_and_text = f"{title}\n{text}".lower()
    if looks_like_front_matter(title_and_text, page_number):
        return "front_matter"
    if has_any_keyword(title_and_text, SAFETY_KEYWORDS):
        return "safety_and_constraint"
    if has_any_keyword(title_and_text, ALARM_KEYWORDS) and has_any_keyword(
        title_and_text, TROUBLESHOOTING_KEYWORDS
    ):
        return "alarm_fault_code"
    if has_any_keyword(title_and_text, ALARM_KEYWORDS):
        return "alarm_fault_code"
    if has_any_keyword(title_and_text, TROUBLESHOOTING_KEYWORDS):
        return "troubleshooting_procedure"
    if has_any_keyword(title_and_text, PARAMETER_KEYWORDS) or re.search(
        r"\bP\d{3}\b|\[[^\]]{2,40}\]",
        text,
    ):
        return "parameter_and_configuration"
    if has_any_keyword(title_and_text, INSTALL_KEYWORDS):
        return "installation_or_wiring"
    if has_any_keyword(title_and_text, MAINTENANCE_KEYWORDS):
        return "maintenance_procedure"
    if has_any_keyword(title_and_text, CONCEPT_KEYWORDS):
        return "concept_and_component"
    return "other"


def has_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def merge_adjacent_units(units: list[RawChunkUnit]) -> list[RawChunkUnit]:
    if not units:
        return []

    merged: list[RawChunkUnit] = [units[0]]

    for unit in units[1:]:
        prev = merged[-1]
        same_source = prev.source_id == unit.source_id
        adjacent_pages = unit.page_start <= prev.page_end + 1
        same_type = prev.chunk_type == unit.chunk_type
        compatible_section = (
            prev.section_path == unit.section_path
            or prev.title == unit.title
            or prev.chunk_type == "front_matter"
        )
        short_block = min(len(prev.text), len(unit.text)) < 180
        same_page = prev.page_end == unit.page_start
        combined_size = len(prev.text) + 2 + len(unit.text)

        if (
            same_source
            and adjacent_pages
            and same_type
            and combined_size <= MAX_CHUNK_SIZE
            and (compatible_section or (same_page and short_block))
        ):
            merged[-1] = RawChunkUnit(
                source_id=prev.source_id,
                source_file=prev.source_file,
                collection=prev.collection,
                page_start=prev.page_start,
                page_end=unit.page_end,
                section_path=prev.section_path,
                chunk_type=prev.chunk_type,
                title=prev.title,
                text=f"{prev.text}\n\n{unit.text}".strip(),
            )
            continue

        merged.append(unit)

    return merged


def split_long_units(units: list[RawChunkUnit]) -> list[RawChunkUnit]:
    output: list[RawChunkUnit] = []

    for unit in units:
        parts = split_text_with_overlap(unit.text, unit.chunk_type)
        for part in parts:
            output.append(
                RawChunkUnit(
                    source_id=unit.source_id,
                    source_file=unit.source_file,
                    collection=unit.collection,
                    page_start=unit.page_start,
                    page_end=unit.page_end,
                    section_path=unit.section_path,
                    chunk_type=unit.chunk_type,
                    title=unit.title,
                    text=part,
                )
            )

    return output


def split_text_with_overlap(text: str, chunk_type: str) -> list[str]:
    clean_text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(clean_text) <= MAX_CHUNK_SIZE:
        return [clean_text]

    boundaries = find_case_boundaries(clean_text) if chunk_type in {
        "alarm_fault_code",
        "troubleshooting_procedure",
        "parameter_and_configuration",
    } else []

    parts: list[str] = []
    start = 0
    text_length = len(clean_text)

    while start < text_length:
        hard_end = min(start + MAX_CHUNK_SIZE, text_length)
        if hard_end >= text_length:
            parts.append(clean_text[start:].strip())
            break

        split_at = choose_split_point(clean_text, start, hard_end, boundaries)
        if split_at <= start:
            split_at = hard_end

        parts.append(clean_text[start:split_at].strip())
        next_start = max(split_at - CHUNK_OVERLAP, start + 1)
        start = next_start

    return [part for part in parts if part]


def find_case_boundaries(text: str) -> list[int]:
    positions: list[int] = []
    for pattern in CASE_BOUNDARY_PATTERNS:
        positions.extend(match.start() for match in pattern.finditer(text))
    return sorted(set(position for position in positions if position > 0))


def choose_split_point(text: str, start: int, hard_end: int, boundaries: list[int]) -> int:
    lower_bound = min(start + MIN_CHUNK_SIZE, hard_end)
    preferred_boundaries = [position for position in boundaries if lower_bound <= position <= hard_end]
    if preferred_boundaries:
        return preferred_boundaries[-1]

    window = text[start:hard_end]
    separator_patterns = (
        "\n\n",
        "\n",
        ". ",
        "; ",
        ": ",
        ", ",
        " ",
    )

    for separator in separator_patterns:
        relative = window.rfind(separator, MIN_CHUNK_SIZE, len(window))
        if relative != -1:
            return start + relative + len(separator.strip())

    return hard_end


def finalize_source_chunks(source: SourceEntry, units: list[RawChunkUnit]) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []

    for chunk_index, unit in enumerate(units):
        text = unit.text.strip()
        if not text:
            continue

        char_count = len(text)
        text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        chunk_id = build_chunk_id(
            source_id=unit.source_id,
            page_start=unit.page_start,
            page_end=unit.page_end,
            section_path=unit.section_path,
            chunk_type=unit.chunk_type,
            chunk_index=chunk_index,
            text_hash=text_hash,
        )
        annotation_candidate, exclude_reason = determine_annotation_candidate(
            unit.chunk_type,
            text,
        )

        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                source_id=unit.source_id,
                source_file=unit.source_file,
                collection=unit.collection,
                page_start=unit.page_start,
                page_end=unit.page_end,
                section_path=unit.section_path,
                chunk_index=chunk_index,
                chunk_type=unit.chunk_type,
                title=unit.title,
                text=text,
                fault_code=extract_fault_codes(text),
                parameter_name=extract_parameter_names(text),
                safety_level=detect_safety_level(text),
                char_count=char_count,
                token_estimate=max(1, round(char_count / 4)),
                text_hash=text_hash,
                is_annotation_candidate=annotation_candidate,
                exclude_reason=exclude_reason,
            )
        )

    return records


def build_chunk_id(
    source_id: str,
    page_start: int,
    page_end: int,
    section_path: str,
    chunk_type: str,
    chunk_index: int,
    text_hash: str,
) -> str:
    stable_bits = (
        f"{source_id}|{page_start}|{page_end}|{section_path}|"
        f"{chunk_type}|{chunk_index}|{text_hash}"
    )
    digest = hashlib.sha1(stable_bits.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}::{page_start}-{page_end}::{digest}"


def extract_fault_codes(text: str) -> str:
    candidates: list[str] = []

    patterns = (
        re.compile(r"(?i)\balarm\s+([0-9]+(?:\.[0-9]+)*)"),
        re.compile(r"(?i)\bfault code[s]?\s*[:#-]?\s*([A-Z0-9._-]+)"),
        re.compile(r"(?i)\berror code[s]?\s*[:#-]?\s*([A-Z0-9._-]+)"),
        re.compile(r"\b[EFA]\d{2,5}\b"),
    )

    for pattern in patterns:
        for match in pattern.finditer(text):
            value = match.group(1) if match.lastindex else match.group(0)
            value = re.sub(r"\s+", " ", value).strip()
            if pattern.pattern.startswith("(?i)\\balarm"):
                value = f"Alarm {value}"
            candidates.append(value)

    return "; ".join(unique_preserve_order(candidates)[:8])


def extract_parameter_names(text: str) -> str:
    candidates: list[str] = []

    for match in re.finditer(r"\bP\d{3}\b\s*(\[[^\]]{2,80}\])?", text):
        candidates.append(re.sub(r"\s+", " ", match.group(0)).strip())
    for match in re.finditer(r"(?i)\bparameter\s+([A-Za-z0-9_./\-\[\] ]{2,80})", text):
        candidates.append(re.sub(r"\s+", " ", match.group(1)).strip(" .,:;"))

    cleaned = []
    for candidate in candidates:
        candidate = candidate.strip()
        if 2 <= len(candidate) <= 80:
            cleaned.append(candidate)

    return "; ".join(unique_preserve_order(cleaned)[:6])


def detect_safety_level(text: str) -> str:
    lowered = text.lower()
    if "danger" in lowered:
        return "danger"
    if "warning" in lowered:
        return "warning"
    if "caution" in lowered:
        return "caution"
    if "notice" in lowered:
        return "notice"
    return "none"


def determine_annotation_candidate(chunk_type: str, text: str) -> tuple[bool, str]:
    lowered = text.lower()
    if chunk_type == "front_matter":
        return False, "front_matter"
    if "table of contents" in lowered or "contents" in lowered:
        return False, "table_of_contents"
    if "copyright" in lowered or "legal information" in lowered:
        return False, "legal_or_copyright"
    if len(text) < 120:
        return False, "too_short"
    return True, ""


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result


def write_chunk_outputs(
    output_path: Path,
    report_path: Path,
    chunks: list[ChunkRecord],
    warnings: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_lines = [json.dumps(asdict(chunk), ensure_ascii=False) for chunk in chunks]
    output_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

    report = build_chunk_report(chunks, warnings)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_chunk_report(
    chunks: list[ChunkRecord],
    warnings: list[str],
) -> dict[str, object]:
    count_by_source = Counter(chunk.source_id for chunk in chunks)
    count_by_chunk_type = Counter(chunk.chunk_type for chunk in chunks)
    chars_by_source: dict[str, list[int]] = defaultdict(list)

    for chunk in chunks:
        chars_by_source[chunk.source_id].append(chunk.char_count)

    report_warnings = list(warnings)
    other_ratio = count_by_chunk_type.get("other", 0) / len(chunks) if chunks else 0.0
    if other_ratio >= 0.3:
        report_warnings.append(f"high_other_ratio:{other_ratio:.2f}")

    for source_id, count in count_by_source.items():
        if count >= 800:
            report_warnings.append(f"source_chunk_volume_high:{source_id}:{count}")

    return {
        "total_chunks": len(chunks),
        "count_by_source": dict(sorted(count_by_source.items())),
        "count_by_chunk_type": dict(sorted(count_by_chunk_type.items())),
        "candidate_chunks": sum(1 for chunk in chunks if chunk.is_annotation_candidate),
        "excluded_chunks": sum(1 for chunk in chunks if not chunk.is_annotation_candidate),
        "avg_chars_by_source": {
            source_id: round(sum(values) / len(values), 2)
            for source_id, values in sorted(chars_by_source.items())
        },
        "warnings": report_warnings,
    }


if __name__ == "__main__":
    raise SystemExit(main())
