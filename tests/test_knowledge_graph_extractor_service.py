from app.services.knowledge_graph_extractor_service import KnowledgeGraphExtractorService


def test_parse_extraction_filters_relations_with_missing_entities():
    raw = """
    ```json
    {
      "entities": [
        {
          "id": "cpu_high_usage",
          "name": "CPU 使用率高",
          "type": "故障",
          "description": "CPU 高负载"
        },
        {
          "id": "check_top_process",
          "name": "查看高占用进程",
          "type": "排查动作",
          "description": "使用 top 查看进程"
        }
      ],
      "relations": [
        {
          "source": "cpu_high_usage",
          "target": "check_top_process",
          "relation": "排查",
          "description": "CPU 高时查看高占用进程",
          "evidence": "查看高占用进程"
        },
        {
          "source": "cpu_high_usage",
          "target": "missing_entity",
          "relation": "导致",
          "description": "非法关系",
          "evidence": "无"
        }
      ]
    }
    ```
    """

    result = KnowledgeGraphExtractorService().parse_extraction(
        raw,
        source_file="cpu.md",
        chunk_index=1,
        chunk_id="chunk-1",
    )

    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert result.relations[0].source_file == "cpu.md"
    assert result.relations[0].chunk_index == 1
    assert result.relations[0].chunk_id == "chunk-1"
