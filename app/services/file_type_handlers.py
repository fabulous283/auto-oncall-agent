"""File type handlers for knowledge-base ingestion."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from loguru import logger

from app.services.document_splitter_service import document_splitter_service


class BaseFileHandler(ABC):
    """Base class for file-specific loading and splitting."""

    extensions: tuple[str, ...] = ()

    def supports(self, extension: str) -> bool:
        return self.normalize_extension(extension) in self.extensions

    @abstractmethod
    def load_and_split(self, file_path: Path) -> List[Document]:
        """Load a file and return split LangChain documents."""

    @staticmethod
    def normalize_extension(extension: str) -> str:
        extension = extension.lower().strip()
        return extension if extension.startswith(".") else f".{extension}"


class MarkdownFileHandler(BaseFileHandler):
    extensions = (".md",)

    def load_and_split(self, file_path: Path) -> List[Document]:
        content = file_path.read_text(encoding="utf-8")
        return document_splitter_service.split_markdown(content, file_path.as_posix())


class TextFileHandler(BaseFileHandler):
    extensions = (".txt",)

    def load_and_split(self, file_path: Path) -> List[Document]:
        content = file_path.read_text(encoding="utf-8")
        return document_splitter_service.split_text(content, file_path.as_posix())


class PdfFileHandler(BaseFileHandler):
    extensions = (".pdf",)

    def load_and_split(self, file_path: Path) -> List[Document]:
        loader = PyPDFLoader(str(file_path))
        loaded_docs = loader.load()
        return document_splitter_service.split_loaded_documents(loaded_docs, file_path.as_posix())


class WordFileHandler(BaseFileHandler):
    extensions = (".docx",)

    def load_and_split(self, file_path: Path) -> List[Document]:
        loader = Docx2txtLoader(str(file_path))
        loaded_docs = loader.load()
        return document_splitter_service.split_loaded_documents(loaded_docs, file_path.as_posix())


class FileHandlerRegistry:
    """Registry that dispatches files to the matching handler."""

    def __init__(self, handlers: Iterable[BaseFileHandler] | None = None):
        self.handlers = list(
            handlers
            or [
                MarkdownFileHandler(),
                TextFileHandler(),
                PdfFileHandler(),
                WordFileHandler(),
            ]
        )
        self._handlers_by_extension: Dict[str, BaseFileHandler] = {}
        for handler in self.handlers:
            for extension in handler.extensions:
                self._handlers_by_extension[extension] = handler

    def supported_extensions(self, include_dot: bool = False) -> list[str]:
        extensions = sorted(self._handlers_by_extension.keys())
        if include_dot:
            return extensions
        return [extension.lstrip(".") for extension in extensions]

    def get_handler(self, file_path: str | Path) -> BaseFileHandler:
        path = Path(file_path)
        extension = path.suffix.lower()
        handler = self._handlers_by_extension.get(extension)
        if handler is None:
            supported = ", ".join(self.supported_extensions(include_dot=True))
            raise ValueError(f"不支持的文件类型: {extension or '无扩展名'}，当前支持: {supported}")
        return handler

    def load_and_split(self, file_path: str | Path) -> List[Document]:
        path = Path(file_path).resolve()
        handler = self.get_handler(path)
        logger.info(f"文件类型处理器匹配成功: {path.name} -> {handler.__class__.__name__}")
        return handler.load_and_split(path)


file_handler_registry = FileHandlerRegistry()
