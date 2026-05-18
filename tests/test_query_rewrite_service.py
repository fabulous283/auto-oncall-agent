from app.services.query_rewrite_service import QueryRewriteService


async def test_query_rewrite_parse_json_from_model_response(monkeypatch):
    service = QueryRewriteService()

    class FakeModel:
        async def ainvoke(self, prompt):
            return type(
                "Result",
                (),
                {
                    "content": """
                    {
                      "original_query": "CPU 告警咋办",
                      "rewritten_query": "CPU 使用率高告警排查步骤",
                      "keywords": ["CPU 告警", "排查"],
                      "sub_questions": ["CPU 高有哪些原因？"],
                      "intent": "aiops_knowledge",
                      "filters": {"doc_type": "aiops"}
                    }
                    """
                },
            )()

    monkeypatch.setattr(service, "_get_model", lambda: FakeModel())

    result = await service.rewrite("CPU 告警咋办")

    assert result.original_query == "CPU 告警咋办"
    assert result.rewritten_query == "CPU 使用率高告警排查步骤"
    assert result.keywords == ["CPU 告警", "排查"]
    assert result.sub_questions == ["CPU 高有哪些原因？"]
    assert result.intent == "aiops_knowledge"
    assert result.filters == {"doc_type": "aiops"}


async def test_query_rewrite_falls_back_on_invalid_json(monkeypatch):
    service = QueryRewriteService()

    class FakeModel:
        async def ainvoke(self, prompt):
            return type("Result", (), {"content": "not json"})()

    monkeypatch.setattr(service, "_get_model", lambda: FakeModel())

    result = await service.rewrite("CPU 告警咋办")

    assert result.original_query == "CPU 告警咋办"
    assert result.rewritten_query == "CPU 告警咋办"
    assert result.keywords == ["CPU 告警咋办"]
