# 五分钟面试 Demo 流程

## 第一分钟：介绍项目架构

打开 README 的系统架构 Mermaid 图，或直接打开 [docs/architecture.md](architecture.md)。

讲解主链路：用户进入 Web Chat UI，经 FastAPI 和 Authentication Layer 后进入 Agent Controller；Router 判断是否调用唯一的 RAG Tool，检索结果经过 Reliability / Query Rewrite 后交给 Qwen3:4b 生成，最后通过 SSE 返回并记录 Memory、Sources 和 Agent State。

## 第二分钟：RAG 知识库问答

在项目根目录运行：

```powershell
python demo.py
```

选择 `2`，使用问题“什么是 RAG？”。

展示：

- Router 是否判断为知识库问题。
- `rag_tool` 调用次数。
- Retriever 返回的文档数量。
- 最终回答中的真实 Sources。

讲解 Document Loader、Embedding、Chroma 和 Retriever 如何把项目资料转换为可引用的 Evidence。

## 第三分钟：Memory 多轮对话

返回菜单后选择 `4`，演示：

```text
我的名字叫张三。
我叫什么？
```

展示 Memory 中的 user / assistant 消息顺序，并说明当前是 Python 进程内的短期记忆，服务层按登录用户隔离 Agent，不是持久化聊天数据库。

## 第四分钟：Agent State

返回菜单后选择 `5`，展示一次 RAG 请求的完整公开 State：

- `request_id`
- `routing_decision`
- `tool_call_count`
- `retrieval_result_count`
- `best_distance`
- `retrieval_reliable`
- `answer_strategy`
- `sources`

讲解 State 如何把模型决策、检索质量、回答策略和来源串联起来，并与结构化 Observability 日志配合定位问题。

## 第五分钟：回答为什么设计 Agent

面试官如果问“为什么不是普通聊天机器人”，可以用三句话收尾：

1. 普通问题不必检索，Router 让工具调用成为有条件的决策。
2. 知识库回答不能只看召回结果，还要评估证据可靠性，不足时受控 Rewrite，避免无限循环和幻觉。
3. State、Sources、Memory 隔离、SSE 和 Evaluation 让系统从一次模型调用变成可调试、可验证、可展示的工程应用。
