# Agentic RAG Evaluation Report

## Scope

本报告对应 Step 15 的 Agent 全面测试与评估。评估只新增测试与文档，未修改 `src` 下的 Agent、RAG、Tool Registry、认证或 FastAPI 核心逻辑。

真实模型评估使用独立 Agent 实例，保留生产代码产生的 `AgentRequestState` 和 Observability 日志。回答正文只保存有限长度摘要，不把完整 Prompt、thinking 或大段 Evidence 写入报告。

## Environment

| 项目 | 配置 |
|---|---|
| Python | 3.14 Windows `.venv` |
| LLM | Ollama `qwen3:4b` |
| Ollama URL | `http://127.0.0.1:11434` |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Vector DB | Chroma persisted database |
| Collection | `rag_documents` |
| Knowledge base | `rag_data/` |
| Chroma records | 10 |

依赖检查和 Chroma 初始化均成功。Ollama `/api/tags` 返回 `qwen3:4b`。

## Router Evaluation

Case 目录包含：

- 普通问题 5 个：问候、自我介绍、Python、笑话等
- 知识库问题 4 个：RAG、向量数据库、知识库检索、Tool 调用

本次真实端到端执行了 2 个代表性 Case：

| Case | routing_decision | tool_call_count | 结果 |
|---|---|---:|---|
| `你好，请介绍一下你自己。` | `llm_false` | 0 | 通过 |
| `什么是 RAG？` | `llm_true` | 1 | 通过 |

Router Smoke Accuracy：

- 普通问题：`1/1 = 100%`
- 知识库问题：`1/1 = 100%`
- 总体：`2/2 = 100%`

## RAG Evaluation

实际执行了高相关、模糊和无关三类问题。所有 Case 都生成了完整 State，并保留 `retrieval_result_count`、`best_distance`、`retrieval_reliable`、`answer_strategy`。

| Case | result_count | best_distance | reliable | strategy | Tool calls |
|---|---:|---:|---|---|---:|
| 高相关：RAG 定义与减少幻觉 | 4 | 0.8865 | false | cautious | 2 |
| 模糊：介绍这个系统 | 4 | 0.9292 | false | cautious | 2 |
| 无关：量子计算/黑洞 | 4 | 1.0524 | false | cautious | 2 |

说明：本项目的可靠性阈值和 Qwen 相关性判断共同决定 `retrieval_reliable`。无关问题没有被当作可靠证据，最终回答保持谨慎并明确知识库边界。

验收观察：高相关问题本次实际值为 `retrieval_reliable=false`（不是预期的 `true`）。这属于当前检索可靠性策略的评估发现，已如实保留；没有为了通过评估修改生产逻辑。

## Query Rewrite

Rewrite Case：`知识库中是否有关于量子计算和黑洞的内容？`

- 第一次检索：完成
- 证据可靠性不足：完成
- `query_rewrite`：非空
- 第二次 Tool 调用：完成
- `tool_call_count = 2`
- 没有第三次检索

实际 State 示例中的 Rewrite 值为模型生成的检索问题，例如：`知识库中是否有关于计算机相关的内容？`。

## Memory

真实多轮 Case：

1. `什么是 RAG？`
2. `它有什么优点？`

验证结果：

- Memory 包含 4 条消息
- 顺序为 `user / assistant / user / assistant`
- 第二轮 Router 能看到第一轮上下文
- 第二轮能够将“它”关联到 RAG
- 第二轮仍遵守最多两次 Tool 调用限制

## Stability

稳定性测试使用测试层异常模拟，不修改生产代码：

| 场景 | 验证结果 |
|---|---|
| Ollama/Agent Streaming 异常 | SSE 返回 `type=error`，进程不崩溃 |
| SSE 客户端提前关闭 | Generator 被取消，残缺 assistant 不写入 Memory |
| 错误验证码 | 抛出 `InvalidVerificationCodeError` |
| 过期验证码 | 抛出 `ExpiredVerificationCodeError` |
| Session 失效 | 抛出 `InvalidSessionError` |
| Chroma/RAG Tool 异常 | Agent 返回包含 `search_knowledge_base` 的明确错误 |

最近一次稳定性命令共运行 7 个测试：6 个通过，1 个真实 Memory 代词测试因未设置 `RUN_REAL_EVALUATION=1` 按设计跳过。跳过不计为通过。

## Multi-user Isolation

使用两个不同 `user_id` 获取独立 Agent：

- 用户 A：`我喜欢Python`
- 用户 B：`我喜欢Java`

结果：两份 Memory 对象不同，历史内容不交叉，用户 A 只能看到 Python 上下文，用户 B 只能看到 Java 上下文。

## Example AgentRequestState

```json
{
  "request_id": "020c5483-db78-4b8e-89ea-6fbe3bc0ba1b",
  "question": "知识库中是否有关于量子计算和黑洞的内容？",
  "routing_decision": "llm_true",
  "tool_call_count": 2,
  "retrieval_result_count": 4,
  "best_distance": 1.0640977621078491,
  "retrieval_reliable": false,
  "relevance_decision": "llm_false",
  "query_rewrite": "知识库中是否有关于计算机相关的内容？",
  "answer_strategy": "cautious",
  "sources": [
    "rag_data\\ragziliao.txt",
    "rag_data\\ziliao2.txt",
    "rag_data\\rag.txt"
  ],
  "total_time_ms": 122256.519
}
```

## Execution

静态检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile tests\evaluation\*.py
```

真实评估：

```powershell
$env:RUN_REAL_EVALUATION="1"
.\.venv\Scripts\python.exe -m unittest tests.evaluation.test_agent_evaluation -v
```

结果：4 个评估测试通过，内部覆盖 7 个真实 Agent Case（Router 2、Retrieval 3、Rewrite 1、Memory 1）。

当前默认分类命令（未设置 `RUN_REAL_EVALUATION`）结果为：多用户隔离 1 个通过；真实 Router/RAG/Rewrite 及 Memory 代词用例按设计跳过。设置 `RUN_REAL_EVALUATION=1` 才会再次访问真实 Ollama/Chroma。

本次补充验证：Ollama `/api/tags` 返回 HTTP 200，Chroma collection `rag_documents` 当前记录数为 10；测试目录全部 `py_compile` 通过。

分类测试文件：

- `test_router_evaluation.py`
- `test_rag_evaluation.py`
- `test_rewrite_evaluation.py`
- `test_memory_evaluation.py`
- `test_stability_evaluation.py`

真实分类测试默认跳过，需要显式设置 `RUN_REAL_EVALUATION=1`。稳定性和多用户隔离测试可以直接运行。
