import json
from pathlib import Path

from scripts.experiment.build_experiment_chunks import (
    VALID_CHUNK_TYPES,
    build_experiment_chunks,
)


def test_build_experiment_chunks_is_stable_and_valid(tmp_path: Path):
    docs_root = tmp_path / "docs" / "experiment_doc"
    docs_root.mkdir(parents=True)
    pdf_path = docs_root / "manual.pdf"
    write_text_pdf(
        pdf_path,
        [
            "Table of Contents\nInstallation/Wiring ........ 15\nTroubleshooting ........ 20\n",
            (
                "Chapter 1 Installation/Wiring\n"
                "P046 [Start Source x] default value is 2.\n"
                "Terminal block wiring procedure requires disconnecting power.\n"
                "WARNING disconnect electrical power before wiring.\n"
            ),
            (
                "Troubleshooting\n"
                "Alarm 5.108 SERVO OVERLOAD\n"
                "Symptom spindle stops unexpectedly.\n"
                "Possible Cause feeds and speeds are too high.\n"
                "Corrective Action decrease feeds and speeds and inspect cable connection.\n"
            ),
        ],
    )

    manifest_path = tmp_path / "aiops-docs" / "experiment" / "sources" / "source_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "sources": [
            {
                "source_id": "manual",
                "file_name": "manual.pdf",
                "file_path": str(pdf_path),
                "file_type": "pdf",
                "page_count": 3,
                "extractable_pages": 3,
                "text_extractable_ratio": 1.0,
                "recommended_usage": "rag_and_agent_candidate",
                "needs_ocr": False,
                "notes": [],
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path = tmp_path / "aiops-docs" / "experiment" / "chunks" / "experiment_chunks.jsonl"
    report_path = tmp_path / "aiops-docs" / "experiment" / "chunks" / "chunk_build_report.json"

    first_report = build_experiment_chunks(
        source_root=docs_root,
        manifest_path=manifest_path,
        output_path=output_path,
        report_path=report_path,
    )
    first_jsonl = output_path.read_text(encoding="utf-8")
    first_report_text = report_path.read_text(encoding="utf-8")

    second_report = build_experiment_chunks(
        source_root=docs_root,
        manifest_path=manifest_path,
        output_path=output_path,
        report_path=report_path,
    )
    second_jsonl = output_path.read_text(encoding="utf-8")
    second_report_text = report_path.read_text(encoding="utf-8")

    assert first_jsonl == second_jsonl
    assert first_report_text == second_report_text
    assert first_report == second_report

    chunks = [json.loads(line) for line in first_jsonl.splitlines() if line.strip()]
    assert chunks
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)

    for chunk in chunks:
        assert chunk["text"].strip()
        assert chunk["page_start"] >= 1
        assert chunk["page_end"] >= chunk["page_start"]
        assert chunk["chunk_type"] in VALID_CHUNK_TYPES

    assert first_report["total_chunks"] == len(chunks)
    assert first_report["excluded_chunks"] >= 1
    assert first_report["count_by_chunk_type"]["front_matter"] >= 1


def write_text_pdf(path: Path, page_texts: list[str]) -> None:
    objects: list[bytes] = []
    page_ids: list[int] = []
    font_id = 3
    next_object_id = 4

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Count 0 /Kids [] >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for text in page_texts:
        page_id = next_object_id
        content_id = next_object_id + 1
        next_object_id += 2
        page_ids.append(page_id)

        escaped_text = escape_pdf_text(text)
        stream = b"BT /F1 12 Tf 72 720 Td 14 TL (" + escaped_text.replace(b"\n", b") Tj T* (") + b") Tj ET"
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
