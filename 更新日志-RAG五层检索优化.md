# 更新日志 - RAG 五层检索优化

## 1. 这次新增了什么

本次把 RAG 检索从“单路向量召回 + rerank”升级成五层检索管线：

```text
原始问题
-> 查询改写
-> RAG 内部问题路由
-> 向量 / 多查询向量 / 关键词 / 图谱 多路召回
-> 融合去重
-> qwen3-rerank 统一精排
-> 拼接上下文给大模型
```

对外工具仍然是 `retrieve_knowledge`，Agent 调用方式不变。

## 2. 为什么要做五层优化

单纯向量检索适合语义相近的问题，但有几个弱点：

1. 用户问题太口语化时，向量检索可能召回不准。
2. 配置项、接口名、文件名这类精确术语，关键词检索往往更稳。
3. 故障、原因、排查动作之间有结构化关系，知识图谱更适合表达。
4. 复杂问题可能要拆成多个子问题分别召回。

所以新增五层：

- 查询改写：把口语问题变成更适合检索的问题。
- RAG 内部路由：判断该走哪些召回通道。
- 多路召回：同时查向量库、关键词索引、知识图谱。
- 融合去重：把重复 chunk 合并，给多路命中的内容加分。
- 统一 rerank：最终仍由 `qwen3-rerank` 精排。

## 3. 入库流程怎么变

原来：

```text
文档 -> 分片 -> 写 Milvus
```

现在：

```text
文档 -> 分片
-> 补充 _chunk_id / _chunk_index / _source
-> 写 SQLite FTS chunk 索引
-> 自动生成知识图谱
-> 写 Milvus
```

新增的 chunk 索引用于关键词召回。

数据库路径：

```text
data/rag_chunks.db
```

表结构：

```text
rag_chunks      保存 chunk 原文、metadata、source_file、chunk_index
rag_chunks_fts  FTS5 全文索引
```

## 4. 检索流程怎么变

`retrieve_knowledge` 原来直接查 Milvus，现在改为调用：

```text
hybrid_retrieval_service.retrieve(query)
```

内部流程：

1. `query_rewrite_service` 调用 Qwen 输出：

```json
{
  "original_query": "CPU 告警咋排查",
  "rewritten_query": "CPU 使用率高告警的排查步骤、原因和处理方案",
  "keywords": ["CPU 告警", "CPU 使用率高", "排查"],
  "sub_questions": ["CPU 使用率高有哪些原因？"],
  "intent": "knowledge_qa",
  "filters": {}
}
```

2. `rag_query_router_service` 决定召回通道：

```text
vector
multi_query_vector
keyword
graph
```

3. `hybrid_retrieval_service` 执行多路召回。

4. 多路结果统一转换成 `RetrievedDocument`。

5. 去重融合后转成 `Document`。

6. 调用现有 `rerank_service.rerank()`。

## 5. 每一路召回做什么

### vector

用 `rewritten_query` 查 Milvus。

### multi_query_vector

用 `rewritten_query + sub_questions` 多次查 Milvus，然后合并。

### keyword

用 `keywords` 查 SQLite FTS5 chunk 索引。

适合：

- 文件名
- 配置项
- 接口名
- 函数名
- 精确术语

### graph

用 `keywords / rewritten_query` 查 SQLite 轻量知识图谱。

返回内容会转换成文本：

```text
【知识图谱关系】
CPU 使用率高 --排查--> 查看高占用进程
说明: CPU 使用率高时需要查看高占用进程
依据: 原文依据
```

这样图谱关系也能参与 rerank 和上下文拼接。

## 6. 多路召回如何合并

每个召回结果都会变成：

```python
RetrievedDocument(
    content="chunk 内容",
    metadata={...},
    source="vector",
    score=1.0,
    chunk_id="...",
    source_file="...",
    chunk_index=1,
    sources=["vector"]
)
```

去重优先级：

```text
1. chunk_id
2. source_file + chunk_index
3. content hash
```

如果同一个 chunk 被多路召回命中，会合并 sources，并加权加分。

第一版权重：

```text
vector = 1.0
multi_query_vector = 0.8
keyword = 0.7
graph = 0.6
```

## 7. rerank 放在哪里

rerank 放在最后。

原因是多路召回结果分数来源不同，不能直接比较。先做粗融合，只保留一批候选，再统一交给 `qwen3-rerank` 让模型从语义相关性角度重新排序。

如果 rerank 失败，仍然保持之前的策略：不静默降级，不返回未精排结果。

## 8. 改了哪些文件

新增：

```text
app/models/retrieval.py
app/services/query_rewrite_service.py
app/services/rag_query_router_service.py
app/services/chunk_index_store_service.py
app/services/hybrid_retrieval_service.py
更新日志-RAG五层检索优化.md
```

修改：

```text
app/config.py
app/services/vector_index_service.py
app/tools/knowledge_tool.py
docker-compose.app.yml
README.md
```

## 9. 如何验证效果

入库时看日志：

```text
chunk 关键词索引写入完成
知识图谱入库完成
文件索引完成
```

检索时看日志：

```text
查询改写完成
RAG 内部路由: routes=...
混合检索完成: candidates=..., fused=..., reranked=...
```

适合测试的问题：

```text
CPU 告警怎么排查？
cpu_high_usage.md 里面说了什么？
RAG_TOP_K 这个配置有什么用？
服务不可用有哪些可能原因？
```

这些问题应该分别触发向量、多查询、关键词和图谱召回。
