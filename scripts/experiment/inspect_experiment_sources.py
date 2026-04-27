"""扫描实验文档目录并生成 source manifest。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}
KEYWORDS = (
    "troubleshooting",
    "fault",
    "fault code",
    "alarm",
    "warning",
    "parameter",
    "procedure",
    "corrective action",
    "cause",
    "symptom",
)


@dataclass(slots=True)
class ExtractedTextStats:
    page_count: int
    extractable_pages: int
    empty_pages: int
    text_extractable_ratio: float
    avg_chars_per_page: float
    suspected_scanned: bool
    needs_ocr: bool
    text: str


@dataclass(slots=True)
class SourceManifestEntry:
    source_id: str
    file_name: str
    file_path: str
    file_type: str
    page_count: int
    extractable_pages: int
    empty_pages: int
    text_extractable_ratio: float
    avg_chars_per_page: float
    suspected_scanned: bool
    needs_ocr: bool
    keyword_hits: dict[str, int]
    recommended_usage: str
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="扫描 docs/experiment_doc 下的实验文档并生成 source manifest"
    )
    parser.add_argument(
        "--source-root",
        default="docs/experiment_doc",
        help="待扫描的实验文档目录，默认 docs/experiment_doc",
    )
    parser.add_argument(
        "--output",
        default="aiops-docs/experiment/sources/source_manifest.json",
        help="manifest 输出路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_source_manifest(Path(args.source_root), Path(args.output))
    print(
        json.dumps(
            {
                "source_root": manifest["source_root"],
                "output_path": manifest["output_path"],
                "total_sources": manifest["total_sources"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def generate_source_manifest(source_root: Path, output_path: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    output_path = output_path.resolve()

    if not source_root.exists() or not source_root.is_dir():
        raise ValueError(f"实验文档目录不存在: {source_root}")

    seen_source_ids: set[str] = set()
    entries = [
        inspect_source_file(path, source_root, seen_source_ids)
        for path in iter_supported_files(source_root)
    ]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": to_repo_relative_path(source_root),
        "output_path": to_repo_relative_path(output_path),
        "total_sources": len(entries),
        "sources": [asdict(entry) for entry in entries],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def iter_supported_files(source_root: Path) -> Iterable[Path]:
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def inspect_source_file(
    path: Path,
    source_root: Path,
    seen_source_ids: set[str],
) -> SourceManifestEntry:
    file_type = path.suffix.lower().lstrip(".")
    stats = inspect_pdf(path) if file_type == "pdf" else inspect_text_file(path)
    keyword_hits = collect_keyword_hits(stats.text)
    notes = build_notes(file_type, stats, keyword_hits)
    source_id = build_source_id(path, source_root, seen_source_ids)

    return SourceManifestEntry(
        source_id=source_id,
        file_name=path.name,
        file_path=to_repo_relative_path(path),
        file_type=file_type,
        page_count=stats.page_count,
        extractable_pages=stats.extractable_pages,
        empty_pages=stats.empty_pages,
        text_extractable_ratio=stats.text_extractable_ratio,
        avg_chars_per_page=stats.avg_chars_per_page,
        suspected_scanned=stats.suspected_scanned,
        needs_ocr=stats.needs_ocr,
        keyword_hits=keyword_hits,
        recommended_usage=recommend_usage(file_type, stats, keyword_hits),
        notes=notes,
    )


def inspect_pdf(path: Path) -> ExtractedTextStats:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("缺少 pypdf 依赖，无法扫描 PDF 实验文档") from exc

    reader = PdfReader(str(path))
    page_texts: list[str] = []

    for page in reader.pages:
        page_texts.append((page.extract_text() or "").strip())

    return summarize_page_texts(page_texts, is_pdf=True)


def inspect_text_file(path: Path) -> ExtractedTextStats:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return summarize_page_texts([text], is_pdf=False)


def summarize_page_texts(page_texts: list[str], is_pdf: bool) -> ExtractedTextStats:
    page_count = len(page_texts)
    cleaned_pages = [text.strip() for text in page_texts]
    extractable_pages = sum(1 for text in cleaned_pages if text)
    empty_pages = page_count - extractable_pages
    total_chars = sum(len(text) for text in cleaned_pages)
    ratio = round(extractable_pages / page_count, 4) if page_count else 0.0
    avg_chars_per_page = round(total_chars / page_count, 2) if page_count else 0.0

    suspected_scanned = bool(
        is_pdf
        and page_count
        and (
            extractable_pages == 0
            or ratio < 0.6
            or (ratio < 0.85 and avg_chars_per_page < 120)
            or (page_count >= 3 and avg_chars_per_page < 40)
        )
    )
    needs_ocr = bool(
        is_pdf
        and page_count
        and (
            extractable_pages == 0
            or ratio < 0.25
            or (suspected_scanned and avg_chars_per_page < 60)
        )
    )

    return ExtractedTextStats(
        page_count=page_count,
        extractable_pages=extractable_pages,
        empty_pages=empty_pages,
        text_extractable_ratio=ratio,
        avg_chars_per_page=avg_chars_per_page,
        suspected_scanned=suspected_scanned,
        needs_ocr=needs_ocr,
        text="\n".join(text for text in cleaned_pages if text),
    )


def collect_keyword_hits(text: str) -> dict[str, int]:
    normalized_text = re.sub(r"\s+", " ", text.lower()).strip()
    keyword_hits: dict[str, int] = {}

    for keyword in KEYWORDS:
        pattern = build_keyword_pattern(keyword)
        keyword_hits[keyword] = len(re.findall(pattern, normalized_text))

    return keyword_hits


def build_keyword_pattern(keyword: str) -> str:
    terms = [re.escape(part) for part in keyword.lower().split()]
    return r"\b" + r"\s+".join(terms) + r"\b"


def recommend_usage(
    file_type: str,
    stats: ExtractedTextStats,
    keyword_hits: dict[str, int],
) -> str:
    total_hits = sum(keyword_hits.values())
    troubleshooting_hits = (
        keyword_hits["troubleshooting"]
        + keyword_hits["procedure"]
        + keyword_hits["corrective action"]
    )

    if stats.needs_ocr:
        return "ocr_first"
    if troubleshooting_hits >= 2 and total_hits >= 6:
        return "rag_and_agent_candidate"
    if total_hits >= 3:
        return "rag_candidate"
    if file_type == "pdf" and stats.extractable_pages == 0:
        return "manual_review_required"
    return "reference_only"


def build_notes(
    file_type: str,
    stats: ExtractedTextStats,
    keyword_hits: dict[str, int],
) -> list[str]:
    notes: list[str] = []
    total_hits = sum(keyword_hits.values())

    if file_type == "pdf" and 0 < stats.extractable_pages < stats.page_count:
        notes.append("partial_text_extraction")
    if stats.suspected_scanned:
        notes.append("suspected_scanned_pdf")
    if stats.needs_ocr:
        notes.append("ocr_recommended")
    if total_hits == 0:
        notes.append("no_target_keywords_detected")
    if file_type != "pdf":
        notes.append("single_document_text_source")

    return notes


def build_source_id(path: Path, source_root: Path, seen_source_ids: set[str]) -> str:
    relative_stem = path.relative_to(source_root).with_suffix("").as_posix()
    normalized = re.sub(r"[^a-z0-9]+", "_", relative_stem.lower()).strip("_") or "source"
    source_id = normalized
    suffix = 2

    while source_id in seen_source_ids:
        source_id = f"{normalized}_{suffix}"
        suffix += 1

    seen_source_ids.add(source_id)
    return source_id


def to_repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    repo_root = Path.cwd().resolve()

    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
