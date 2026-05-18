from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.services.file_type_handlers import (
    FileHandlerRegistry,
    MarkdownFileHandler,
    PdfFileHandler,
    TextFileHandler,
    WordFileHandler,
)


def test_registry_lists_supported_extensions():
    registry = FileHandlerRegistry()

    assert registry.supported_extensions() == ["docx", "md", "pdf", "txt"]
    assert registry.supported_extensions(include_dot=True) == [".docx", ".md", ".pdf", ".txt"]


def test_registry_matches_handler_by_extension():
    registry = FileHandlerRegistry()

    assert isinstance(registry.get_handler("readme.md"), MarkdownFileHandler)
    assert isinstance(registry.get_handler("notes.txt"), TextFileHandler)
    assert isinstance(registry.get_handler("manual.pdf"), PdfFileHandler)
    assert isinstance(registry.get_handler("report.docx"), WordFileHandler)


def test_registry_rejects_unsupported_extension():
    registry = FileHandlerRegistry()

    with pytest.raises(ValueError, match="不支持的文件类型"):
        registry.get_handler("legacy.doc")


def test_markdown_handler_uses_markdown_splitter(monkeypatch, tmp_path):
    file_path = tmp_path / "knowledge.md"
    file_path.write_text("# 标题\n正文", encoding="utf-8")
    captured = {}

    def fake_split_markdown(content, source):
        captured["content"] = content
        captured["source"] = source
        return [Document(page_content="chunk")]

    monkeypatch.setattr(
        "app.services.file_type_handlers.document_splitter_service.split_markdown",
        fake_split_markdown,
    )

    docs = MarkdownFileHandler().load_and_split(file_path)

    assert docs[0].page_content == "chunk"
    assert captured["content"] == "# 标题\n正文"
    assert captured["source"] == file_path.as_posix()


def test_pdf_handler_uses_pypdf_loader(monkeypatch, tmp_path):
    file_path = tmp_path / "manual.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    class FakeLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content="第一页内容", metadata={"page": 0})]

    def fake_split_loaded_documents(documents, source):
        assert documents[0].page_content == "第一页内容"
        assert source == file_path.as_posix()
        return documents

    monkeypatch.setattr("app.services.file_type_handlers.PyPDFLoader", FakeLoader)
    monkeypatch.setattr(
        "app.services.file_type_handlers.document_splitter_service.split_loaded_documents",
        fake_split_loaded_documents,
    )

    docs = PdfFileHandler().load_and_split(file_path)

    assert docs[0].metadata["page"] == 0


def test_word_handler_uses_docx_loader(monkeypatch, tmp_path):
    file_path = tmp_path / "report.docx"
    file_path.write_bytes(b"fake docx")

    class FakeLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return [Document(page_content="Word 内容")]

    def fake_split_loaded_documents(documents, source):
        assert documents[0].page_content == "Word 内容"
        assert source == file_path.as_posix()
        return documents

    monkeypatch.setattr("app.services.file_type_handlers.Docx2txtLoader", FakeLoader)
    monkeypatch.setattr(
        "app.services.file_type_handlers.document_splitter_service.split_loaded_documents",
        fake_split_loaded_documents,
    )

    docs = WordFileHandler().load_and_split(file_path)

    assert docs[0].page_content == "Word 内容"
