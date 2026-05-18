from langchain_core.documents import Document

from app.services.vector_index_service import VectorIndexService


def test_index_directory_scans_all_registered_file_types(monkeypatch, tmp_path):
    for filename in ["a.md", "b.txt", "c.pdf", "d.docx", "ignored.doc"]:
        (tmp_path / filename).write_text("content", encoding="utf-8")

    indexed_files = []

    class FakeRegistry:
        def supported_extensions(self, include_dot=False):
            assert include_dot is True
            return [".md", ".txt", ".pdf", ".docx"]

    service = VectorIndexService()
    monkeypatch.setattr("app.services.vector_index_service.file_handler_registry", FakeRegistry())
    monkeypatch.setattr(service, "index_single_file", lambda file_path: indexed_files.append(file_path))

    result = service.index_directory(str(tmp_path))

    indexed_names = sorted(path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for path in indexed_files)
    assert result.success is True
    assert result.total_files == 4
    assert indexed_names == ["a.md", "b.txt", "c.pdf", "d.docx"]


def test_index_single_file_uses_file_handler_registry(monkeypatch, tmp_path):
    file_path = tmp_path / "manual.pdf"
    file_path.write_bytes(b"%PDF-1.4")
    normalized_path = file_path.resolve().as_posix()
    added_documents = []
    deleted_sources = []

    class FakeRegistry:
        def load_and_split(self, path):
            assert path == file_path.resolve()
            return [Document(page_content="PDF chunk", metadata={"_source": normalized_path})]

    class FakeVectorStoreManager:
        def delete_by_source(self, source):
            deleted_sources.append(source)

        def add_documents(self, documents):
            added_documents.extend(documents)

    monkeypatch.setattr("app.services.vector_index_service.file_handler_registry", FakeRegistry())
    monkeypatch.setattr("app.services.vector_index_service.vector_store_manager", FakeVectorStoreManager())

    VectorIndexService().index_single_file(str(file_path))

    assert deleted_sources == [normalized_path]
    assert [doc.page_content for doc in added_documents] == ["PDF chunk"]


def test_index_single_file_continues_when_knowledge_graph_fails(monkeypatch, tmp_path):
    file_path = tmp_path / "manual.md"
    file_path.write_text("# title", encoding="utf-8")
    added_documents = []

    class FakeRegistry:
        def load_and_split(self, path):
            return [Document(page_content="chunk")]

    class FakeVectorStoreManager:
        def delete_by_source(self, source):
            return None

        def add_documents(self, documents):
            added_documents.extend(documents)

    monkeypatch.setattr("app.services.vector_index_service.config.knowledge_graph_enabled", True)
    monkeypatch.setattr("app.services.vector_index_service.file_handler_registry", FakeRegistry())
    monkeypatch.setattr("app.services.vector_index_service.vector_store_manager", FakeVectorStoreManager())
    monkeypatch.setattr(
        VectorIndexService,
        "_index_knowledge_graph",
        lambda self, source_file, documents: (_ for _ in ()).throw(RuntimeError("kg failed")),
    )

    VectorIndexService().index_single_file(str(file_path))

    assert [doc.page_content for doc in added_documents] == ["chunk"]
