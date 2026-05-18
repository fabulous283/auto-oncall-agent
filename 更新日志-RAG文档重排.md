# 更新日志 - RAG 文档重排

## 1. 本次新增了什么

本次给 RAG 知识库问答增加了“召回后重排”能力。

原来的 RAG 链路是：

```text
用户问题 -> 向量检索 Milvus -> 取 topK 文档 -> 拼接上下文 -> 大模型回答
```

现在的 RAG 链路是：

```text
用户问题 -> 向量检索 Milvus 先召回 10 条候选文档 -> 阿里云百炼 qwen3-rerank 重排 -> 保留前 3 条 -> 拼接上下文 -> 大模型回答
```

这样做的核心原因是：向量检索负责“先把可能相关的文档找出来”，重排模型负责“再判断这些文档和问题到底谁更相关”。在 RAG 里，重排通常放在初召回之后、上下文拼接之前。

## 2. 为什么不是直接把 topK 从 3 改大

如果只是把向量检索的 topK 从 3 改成 10，然后全部塞给大模型，会有两个问题：

1. 噪声更多：召回数量越多，里面越可能混入不太相关的 chunk。
2. token 成本更高：更多文档进入上下文，会增加大模型输入长度。

所以本次采用“两段式检索”：

1. `RAG_RECALL_TOP_K=10`：Milvus 先召回 10 条候选文档。
2. `RAG_RERANK_TOP_K=3`：百炼重排后只保留最相关的 3 条。

这样既给重排模型足够的候选空间，又控制了最终进入大模型的上下文长度。

## 3. 使用的模型和接口

本次默认使用阿里云百炼平台的文本重排模型：

```text
qwen3-rerank
```

调用地址配置为：

```text
https://dashscope.aliyuncs.com/compatible-api/v1/reranks
```

代码里把基础地址做成了配置项：

```text
DASHSCOPE_RERANK_BASE_URL=https://dashscope.aliyuncs.com/compatible-api/v1
```

最终服务会在后面拼上 `/reranks`。

请求体核心结构是：

```json
{
  "model": "qwen3-rerank",
  "query": "用户问题",
  "documents": ["候选文档1", "候选文档2"],
  "top_n": 3
}
```

响应里会读取：

```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.93
    }
  ]
}
```

其中：

- `index` 表示这个结果对应原始候选文档列表中的第几个文档。
- `relevance_score` 表示百炼重排模型给出的相关性分数。

## 4. 改了哪些文件

### `app/config.py`

新增了 4 个配置项：

```python
rag_recall_top_k: int = 10
rag_rerank_top_k: int = 3
rag_rerank_model: str = "qwen3-rerank"
dashscope_rerank_base_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1"
```

含义：

- `rag_recall_top_k`：向量数据库先召回多少条候选文档。
- `rag_rerank_top_k`：重排后最终保留多少条文档。
- `rag_rerank_model`：使用哪个百炼重排模型。
- `dashscope_rerank_base_url`：百炼重排兼容接口的基础地址。

### `app/services/rerank_service.py`

新增了百炼重排服务。

它负责：

1. 检查 `DASHSCOPE_API_KEY` 是否存在。
2. 把候选 `Document` 的 `page_content` 组装成 `documents`。
3. 调用百炼 `/reranks` 接口。
4. 根据返回的 `index` 重新排列文档。
5. 把重排信息写回文档 metadata：

```python
metadata["rerank_score"] = ...
metadata["rerank_model"] = "qwen3-rerank"
metadata["rerank_original_index"] = ...
```

这样后续如果要排查“为什么这条文档排在前面”，可以从 metadata 里看到重排分数和原始位置。

### `app/tools/knowledge_tool.py`

这里是 RAG 知识检索工具的主入口。

原来是：

```python
retriever = vector_store.as_retriever(search_kwargs={"k": config.rag_top_k})
docs = retriever.invoke(query)
context = format_docs(docs)
```

现在是：

```python
retriever = vector_store.as_retriever(search_kwargs={"k": config.rag_recall_top_k})
recalled_docs = retriever.invoke(query)
docs = rerank_service.rerank(
    query=query,
    documents=recalled_docs,
    top_k=config.rag_rerank_top_k,
)
context = format_docs(docs)
```

也就是说，大模型看到的上下文不再是 Milvus 原始排序，而是百炼重排后的排序。

### `docker-compose.app.yml`

补充了容器环境变量：

```yaml
RAG_RECALL_TOP_K: "10"
RAG_RERANK_TOP_K: "3"
RAG_RERANK_MODEL: qwen3-rerank
DASHSCOPE_RERANK_BASE_URL: https://dashscope.aliyuncs.com/compatible-api/v1
```

### `README.md`

在配置说明里补充了新的 RAG 重排配置项，方便后续部署时对照。

### `tests/test_rerank_service.py`

新增百炼重排服务测试，覆盖：

1. 能按百炼返回的 `index` 重新排列文档。
2. 能写入 `rerank_score`、`rerank_model`、`rerank_original_index`。
3. 缺少 `DASHSCOPE_API_KEY` 时直接报错。
4. HTTP 调用失败时直接报错。

### `tests/test_knowledge_tool_rerank.py`

新增知识库工具链路测试，验证：

1. Milvus retriever 使用 `RAG_RECALL_TOP_K=10` 召回候选文档。
2. 召回后的文档会进入 `rerank_service.rerank`。
3. 最终返回给 Agent 的是重排后的 3 条文档。

## 5. 关于“不做降级”

本次明确没有实现“重排失败后退回原始向量检索结果”的逻辑。

原因是你要求“不做降级”。所以当前行为是：

- 百炼接口失败：返回检索错误，不使用原始召回结果。
- API Key 缺失：返回检索错误，不使用原始召回结果。
- 百炼响应格式不对：返回检索错误，不使用原始召回结果。

这样可以避免系统表面看起来还能回答，但实际已经绕过重排模型，导致效果和预期不一致。

## 6. 如何验证功能生效

可以从日志里观察这两条信息：

```text
开始调用百炼重排: model=qwen3-rerank, candidates=10, top_n=3
向量召回 10 个候选文档，重排后保留 3 个文档
```

也可以从返回的 `Document.metadata` 里看是否包含：

```python
rerank_score
rerank_model
rerank_original_index
```

如果这些字段存在，说明文档已经经过百炼重排。

## 7. 本次验证结果

已经执行过 Python 静态编译检查：

```bash
python -m py_compile app/services/rerank_service.py app/tools/knowledge_tool.py app/config.py tests/test_rerank_service.py tests/test_knowledge_tool_rerank.py
```

编译检查通过。

尝试执行 pytest 时，当前环境缺少 pytest：

```text
No module named pytest
```

所以本次没有完成 pytest 运行。测试代码已经补充好，安装测试依赖后可以执行：

```bash
python -m pytest tests/test_rerank_service.py tests/test_knowledge_tool_rerank.py -q
```
