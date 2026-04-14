from app.services.document_splitter_service import document_splitter_service
from app.services.pdf_parser_service import PdfPage


def test_split_pdf_keeps_page_metadata():
    pages = [
        PdfPage(
            page_number=1,
            text=(
                "故障现象\n设备启动失败，控制面板报码 E01。\n"
                "处理步骤\n1. 检查电源。\n2. 检查保险丝。\n"
            ),
        )
    ]

    docs = document_splitter_service.split_pdf(pages, "/tmp/MachineRepaire.pdf")

    assert docs
    assert docs[0].metadata["_extension"] == ".pdf"
    assert docs[0].metadata["page_number"] == 1
    assert docs[0].metadata["_file_name"] == "MachineRepaire.pdf"
    assert "故障现象" in docs[0].page_content or "处理步骤" in docs[0].page_content
