# Agentic RAG System 面试指南

## 1. 项目一句话介绍

基于 Qwen3 和 LangChain 生态构建 Agentic RAG 智能问答系统，实现 LLM 动态路由、知识增强检索、证据可靠性判断、受控 Query Rewrite、多轮短期 Memory，以及带 Agent State 和结构化日志的可观测执行流程。

这句话适合放在简历项目描述中。它只概括当前仓库已经实现的能力，不把实验性设想描述成线上功能。

## 2. 项目核心亮点

### Agent 决策能力

普通 RAG 往往是固定流程：

```text
用户问题
  ↓
固定检索
  ↓
回答
```

本项目先由 Router 判断问题是否需要知识库：

```text
用户问题
  ↓
Router
  ↓
判断是否调用 Tool
  ↓
RAG 检索（必要时）
  ↓
可靠性判断
  ↓
回答
```

Agent 的价值在于让检索成为有条件的动作。普通问候或一般知识问题可以直接回答；与项目知识库相关的问题才调用 RAG Tool，从而减少无意义的检索，并能在证据不足时进入受控的 Query Rewrite 流程。

### RAG 增强

- **Document Loader**：读取 `rag_data/` 中的项目资料，为后续处理提供统一文档输入。
- **Embedding**：将文档和查询转换为向量，使系统能够按语义相似度检索，而不只依赖关键词。
- **Chroma Vector Database**：持久化向量和文档片段，提供本地可复现的向量存储。
- **Retriever**：根据问题召回候选片段，并保留内容、来源、页码和 distance。
- **Query Rewrite**：第一次证据不足时，让模型生成更适合知识库检索的新 Query；最多进行一次二次检索。
- **Evidence Reliability**：结合结果数量、distance 和相关性判断，决定回答策略是 reliable、cautious 还是 insufficient。

### Agent State 可观测

每次请求生成独立的 `AgentRequestState`，主要字段包括：

- `request_id`
- `routing_decision`
- `tool_call_count`
- `retrieval_result_count`
- `best_distance`
- `retrieval_reliable`
- `answer_strategy`
- `sources`

这些字段将 Router、Tool、Retrieval、回答策略和来源串成一条可检查的执行记录。企业 Agent 需要可观测性，是因为模型输出具有不确定性，只有记录每个阶段的决策和耗时，才能定位误检索、重复调用、证据不足和性能回退等问题，而不是只看到一段最终文本。

### Memory 设计

- 支持当前 Agent 实例内的多轮 user / assistant 对话。
- 通过最大消息数量限制控制短期上下文规模。
- 服务层按登录用户创建隔离的 Agent，因此不同用户的 Session 不共享 Memory。
- Memory 当前是 Python 进程内短期记忆，不是数据库持久化的聊天历史。

### 工程化能力

- **FastAPI**：提供认证、Chat 页面和 SSE 接口。
- **SSE Streaming**：将 status、token、sources、done、error 等事件实时传递给浏览器。
- **Authentication**：QQ 邮箱验证码注册/登录，使用 SQLite 和 Session Cookie。
- **Evaluation**：覆盖 Router、RAG、Rewrite、Memory、稳定性和多用户隔离。
- **Performance Monitoring**：记录 Router、Retrieval、Reliability、Generation 和 Total 阶段耗时，并保留 Fast Path、Prompt 优化、Embedding 预热等优化记录。
