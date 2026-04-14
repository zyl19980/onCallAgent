"""PDF 解析服务。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from loguru import logger


@dataclass
class PdfPage:
    """单页 PDF 文本。"""

    page_number: int
    text: str


class PdfParserService:
    """PDF 解析服务，当前仅支持可提取文本的 PDF。"""

    def parse(self, file_path: str) -> List[PdfPage]:
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"PDF 文件不存在: {file_path}")

        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "缺少 PDF 解析依赖 pypdf，请先安装项目依赖后再索引 PDF 文件"
            ) from exc

        logger.info(f"开始解析 PDF 文件: {path}")
        reader = PdfReader(str(path))

        pages: List[PdfPage] = []
        empty_pages: List[int] = []

        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                empty_pages.append(index)
                continue
            pages.append(PdfPage(page_number=index, text=text))

        if not pages:
            raise RuntimeError(
                "PDF 未提取到可用文本，当前版本仅支持文本型 PDF，不支持 OCR 扫描件"
            )

        if empty_pages:
            logger.warning(f"PDF 存在未提取到文本的页面: {empty_pages}")

        logger.info(f"PDF 解析完成: {path.name}, 可用页面数={len(pages)}")
        return pages


pdf_parser_service = PdfParserService()
