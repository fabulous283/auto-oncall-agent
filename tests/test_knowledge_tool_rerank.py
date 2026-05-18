from langchain_core.documents import Document

from app.tools import knowledge_tool


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


class FakeVectorStore:
    def __init__(self, docs):
        self.docs = docs
        self.search_kwargs = None

    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return FakeRetriever(self.docs)


def test_retrieve_knowledge_uses_hybrid_retrieval(monkeypatch):
    docs = [
        Document(page_content="候选文档 4", metadata={"_file_name": "4.md"}),
        Document(page_content="候选文档 2", metadata={"_file_name": "2.md"}),
        Document(page_content="候选文档 0", metadata={"_file_name": "0.md"}),
    ]

    class FakeHybridRetrievalService:
        def retrieve(self, query):
            assert query == "怎么处理内存告警"
            return docs

    monkeypatch.setattr(knowledge_tool, "hybrid_retrieval_service", FakeHybridRetrievalService())

    context, docs = knowledge_tool.retrieve_knowledge.invoke({"query": "怎么处理内存告警"})

    assert [doc.page_content for doc in docs] == ["候选文档 4", "候选文档 2", "候选文档 0"]
    assert "候选文档 4" in context
    assert "候选文档 2" in context
    assert "候选文档 0" in context
