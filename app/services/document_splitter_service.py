"""文档分割服务模块。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config
from app.services.pdf_parser_service import PdfPage


class DocumentSplitterService:
    """文档分割服务。"""

    PDF_SECTION_PATTERNS = [
        (re.compile(r"^(故障现象|现象描述)[:：]?\s*$"), "fault_symptom"),
        (re.compile(r"^(故障原因|原因分析)[:：]?\s*$"), "root_cause"),
        (re.compile(r"^(处理步骤|维修步骤|操作步骤)[:：]?\s*$"), "repair_steps"),
        (re.compile(r"^(注意事项|风险提示)[:：]?\s*$"), "attention"),
    ]

    def __init__(self):
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2")],
            strip_headers=False,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.pdf_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=140,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            "文档分割服务初始化完成, markdown_chunk_size={}, pdf_chunk_size={}, overlap={}",
            self.chunk_size * 2,
            900,
            self.chunk_overlap,
        )

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        md_docs = self.markdown_splitter.split_text(content)
        docs_after_split = self.text_splitter.split_documents(md_docs)
        final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

        for index, doc in enumerate(final_docs):
            doc.metadata.update(
                {
                    "_source": file_path,
                    "_extension": ".md",
                    "_file_name": Path(file_path).name,
                    "chunk_type": "markdown",
                    "chunk_index": index,
                    "section_path": self._build_section_path(doc.metadata),
                }
            )

        logger.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
        return final_docs

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        docs = self.text_splitter.create_documents(
            texts=[content],
            metadatas=[
                {
                    "_source": file_path,
                    "_extension": Path(file_path).suffix,
                    "_file_name": Path(file_path).name,
                    "chunk_type": "text",
                    "section_path": "",
                }
            ],
        )

        for index, doc in enumerate(docs):
            doc.metadata["chunk_index"] = index

        logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
        return docs

    def split_pdf(self, pages: List[PdfPage], file_path: str = "") -> List[Document]:
        if not pages:
            logger.warning(f"PDF 页面为空: {file_path}")
            return []

        file_name = Path(file_path).name
        final_docs: List[Document] = []

        for page in pages:
            sections = self._split_pdf_page_into_sections(page.text)
            for section_index, section in enumerate(sections):
                docs = self.pdf_text_splitter.create_documents(
                    texts=[section["text"]],
                    metadatas=[
                        {
                            "_source": file_path,
                            "_extension": ".pdf",
                            "_file_name": file_name,
                            "page_number": page.page_number,
                            "section_path": section["title"],
                            "chunk_type": section["chunk_type"],
                            "section_index": section_index,
                        }
                    ],
                )
                final_docs.extend(docs)

        merged_docs = self._merge_pdf_small_chunks(final_docs)
        for index, doc in enumerate(merged_docs):
            doc.metadata["chunk_index"] = index

        logger.info(f"PDF 分割完成: {file_path} -> {len(merged_docs)} 个分片")
        return merged_docs

    def split_document(
        self,
        content: str,
        file_path: str = "",
        pages: List[PdfPage] | None = None,
    ) -> List[Document]:
        if file_path.endswith(".md"):
            return self.split_markdown(content, file_path)
        if file_path.endswith(".pdf"):
            return self.split_pdf(pages or [], file_path)
        return self.split_text(content, file_path)

    def _split_pdf_page_into_sections(self, text: str) -> List[Dict[str, str]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        sections: List[Dict[str, str]] = []
        current_title = "正文"
        current_type = "pdf_text"
        current_lines: List[str] = []

        for line in lines:
            title_info = self._detect_pdf_section_title(line)
            if title_info:
                if current_lines:
                    sections.append(
                        {
                            "title": current_title,
                            "chunk_type": current_type,
                            "text": "\n".join(current_lines).strip(),
                        }
                    )
                current_title = title_info["title"]
                current_type = title_info["chunk_type"]
                current_lines = [line]
                continue

            if self._looks_like_step_line(line) and current_lines:
                current_lines.append(line)
                continue

            current_lines.append(line)

        if current_lines:
            sections.append(
                {
                    "title": current_title,
                    "chunk_type": current_type,
                    "text": "\n".join(current_lines).strip(),
                }
            )

        return sections

    def _detect_pdf_section_title(self, line: str) -> Dict[str, str] | None:
        clean_line = line.strip()
        if len(clean_line) <= 32 and re.match(r"^[一二三四五六七八九十0-9.\-、（）()\sA-Za-z\u4e00-\u9fff]+$", clean_line):
            for pattern, chunk_type in self.PDF_SECTION_PATTERNS:
                if pattern.match(clean_line):
                    return {"title": clean_line, "chunk_type": chunk_type}

            if clean_line.endswith(("步骤", "说明", "流程", "处理", "分析")):
                return {"title": clean_line, "chunk_type": "pdf_section"}

        return None

    def _looks_like_step_line(self, line: str) -> bool:
        return bool(re.match(r"^(\d+[.)、]|步骤\s*\d+|[一二三四五六七八九十]+[、.])", line))

    def _merge_small_chunks(self, documents: List[Document], min_size: int = 300) -> List[Document]:
        if not documents:
            return []

        merged_docs: List[Document] = []
        current_doc: Document | None = None

        for doc in documents:
            doc_size = len(doc.page_content)
            if current_doc is None:
                current_doc = doc
                continue

            if doc_size < min_size and len(current_doc.page_content) < self.chunk_size * 2:
                current_doc.page_content += "\n\n" + doc.page_content
                continue

            merged_docs.append(current_doc)
            current_doc = doc

        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs

    def _merge_pdf_small_chunks(self, documents: List[Document], min_size: int = 250) -> List[Document]:
        if not documents:
            return []

        merged_docs: List[Document] = []
        current_doc: Document | None = None

        for doc in documents:
            if current_doc is None:
                current_doc = doc
                continue

            same_page = current_doc.metadata.get("page_number") == doc.metadata.get("page_number")
            same_section = current_doc.metadata.get("section_path") == doc.metadata.get("section_path")
            small_chunk = len(doc.page_content) < min_size

            if same_page and same_section and small_chunk:
                current_doc.page_content += "\n" + doc.page_content
                continue

            merged_docs.append(current_doc)
            current_doc = doc

        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs

    def _build_section_path(self, metadata: Dict[str, str]) -> str:
        headers = [metadata.get(key, "") for key in ("h1", "h2", "h3")]
        return " > ".join([header for header in headers if header])


document_splitter_service = DocumentSplitterService()
