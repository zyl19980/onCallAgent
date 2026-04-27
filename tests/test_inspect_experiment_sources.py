import json
from pathlib import Path

from scripts.experiment.inspect_experiment_sources import (
    collect_keyword_hits,
    generate_source_manifest,
    summarize_page_texts,
)


def test_collect_keyword_hits_counts_case_insensitively():
    text = (
        "Troubleshooting starts here. Fault code E01 is an alarm. "
        "The corrective action follows the procedure. "
        "Possible cause and symptom are documented."
    )

    hits = collect_keyword_hits(text)

    assert hits["troubleshooting"] == 1
    assert hits["fault"] == 1
    assert hits["fault code"] == 1
    assert hits["alarm"] == 1
    assert hits["corrective action"] == 1
    assert hits["procedure"] == 1
    assert hits["cause"] == 1
    assert hits["symptom"] == 1


def test_summarize_page_texts_flags_scanned_pdf():
    stats = summarize_page_texts(["", "", "short note"], is_pdf=True)

    assert stats.page_count == 3
    assert stats.extractable_pages == 1
    assert stats.empty_pages == 2
    assert stats.text_extractable_ratio == 0.3333
    assert stats.suspected_scanned is True
    assert stats.needs_ocr is True


def test_generate_source_manifest_writes_expected_entries(tmp_path: Path):
    docs_root = tmp_path / "docs" / "experiment_doc"
    docs_root.mkdir(parents=True)
    output_path = tmp_path / "aiops-docs" / "experiment" / "sources" / "source_manifest.json"

    (docs_root / "guide.md").write_text(
        "# Troubleshooting\nFault code E01 alarm corrective action procedure cause symptom\n",
        encoding="utf-8",
    )
    (docs_root / "notes.txt").write_text("warning parameter baseline", encoding="utf-8")
    write_text_pdf(
        docs_root / "manual.pdf",
        [
            "Troubleshooting fault code alarm procedure",
            "",
        ],
    )

    manifest = generate_source_manifest(docs_root, output_path)
    written_manifest = json.loads(output_path.read_text(encoding="utf-8"))

    assert manifest["total_sources"] == 3
    assert written_manifest["total_sources"] == 3

    sources = {entry["file_name"]: entry for entry in written_manifest["sources"]}

    assert sources["guide.md"]["file_type"] == "md"
    assert sources["guide.md"]["recommended_usage"] == "rag_and_agent_candidate"

    assert sources["notes.txt"]["file_type"] == "txt"
    assert sources["notes.txt"]["keyword_hits"]["warning"] == 1
    assert sources["notes.txt"]["keyword_hits"]["parameter"] == 1

    assert sources["manual.pdf"]["page_count"] == 2
    assert sources["manual.pdf"]["extractable_pages"] == 1
    assert sources["manual.pdf"]["empty_pages"] == 1
    assert sources["manual.pdf"]["text_extractable_ratio"] == 0.5
    assert sources["manual.pdf"]["keyword_hits"]["fault code"] == 1
    assert "partial_text_extraction" in sources["manual.pdf"]["notes"]


def write_text_pdf(path: Path, page_texts: list[str]) -> None:
    objects: list[bytes] = []
    page_ids: list[int] = []
    font_id = 3
    next_object_id = 4

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Count 0 /Kids [] >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_specs: list[tuple[int, int]] = []
    for text in page_texts:
        page_id = next_object_id
        content_id = next_object_id + 1
        next_object_id += 2
        page_ids.append(page_id)
        page_specs.append((page_id, content_id))

        escaped_text = escape_pdf_text(text)
        stream = b"BT /F1 12 Tf 72 720 Td (" + escaped_text + b") Tj ET"
        content = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
        page = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects.append(page)
        objects.append(content)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("ascii")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(pdf)


def escape_pdf_text(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return escaped.encode("latin-1", errors="ignore")
