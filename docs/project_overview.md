# Agentic RAG System 项目概述

## 1. 项目背景

传统问答系统通常只能依赖固定提示词或一次性检索，面对复杂问题时容易出现检索无关、证据不足和回答不可追溯等问题。本项目围绕 Agentic RAG（具备决策能力的检索增强生成）构建一个可运行、可观测、可评估的智能问答系统，让模型先判断是否需要知识库，再根据检索证据决定回答策略。

## 2. 项目目标

构建一个基于大语言模型的智能问答 Agent：对普通问题直接回答，对知识库问题执行受控检索，在证据不足时进行 Query Rewrite，并在最终回答中保留来源信息和请求运行状态。

## 3. 系统架构

```text
用户
  ↓
FastAPI / SSE
  ↓
Agent Router
  ↓
Tool Registry
  ↓
RAG Retrieval（Embedding + Chroma）
  ↓
Evidence Reliability / Query Rewrite
  ↓
LLM Generation（Ollama / Qwen3:4b）
  ↓
Memory、Sources、AgentRequestState
```

系统通过 Session Cookie 识别用户，并为不同用户维护相互隔离的短期对话 Memory。Streaming 请求使用 SSE 将状态、回答 token、来源和结束事件发送给浏览器。

## 4. 核心模块

### Agent 模块

- Router：判断当前问题是否需要查询知识库。
- State 管理：通过 `AgentRequestState` 记录 request、检索、策略、来源和耗时信息。
- Tool 调用：通过 Tool Registry 统一查找和执行 RAG Tool。
- Loop 控制：最多执行两次知识库检索，证据不足时触发 Query Rewrite。

### RAG 模块

- 文档加载：读取 `rag_data/` 中的知识库文件。
- Chunk 切分：将文档拆分为适合向量检索的片段。
- Embedding：使用 Sentence Transformer 生成向量表示。
- Chroma 向量检索：根据问题召回候选证据，并保留 source、page、distance 等信息。

### Memory 模块

- 保存多轮 user / assistant 消息。
- 通过固定长度限制控制短期上下文规模。
- 按用户隔离 Agent 实例和对话上下文，不共享不同用户的 Memory。

### Authentication 模块

- QQ 邮箱验证码注册与登录。
- SQLite 保存用户、验证码和 Session 的服务端数据。
- 使用 HttpOnly Session Cookie 保护 Chat 和 SSE 接口。

### Evaluation 模块

- Router 测试。
- RAG 检索测试。
- Query Rewrite 测试。
- Memory 多轮测试。
- Stability 异常测试和多用户隔离测试。

## 5. 技术亮点

- Agentic RAG：由 LLM Router 决定是否使用知识库。
- Query Rewrite：第一次证据不足时生成更适合检索的新 Query。
- Retrieval Reliability：结合距离、结果数量和模型判断评估证据可靠性。
- SSE Streaming：通过 FastAPI 将生成内容实时传递给浏览器。
- Observability：使用 request id、结构化日志和 AgentRequestState 跟踪请求阶段。
- Performance Optimization：包含 Chroma 预热、高置信 Fast Path、Prompt 压缩和阶段状态反馈。
- Evidence Traceability：最终回答只引用实际检索到的 source/page。

## 6. 后续优化方向

- 多 Agent 协作与任务分解。
- 使用更强的本地或云端模型。
- 云端部署、弹性扩容和集中式日志。
- 扩大知识库规模并增加更多文档格式。
- 增加离线评测集、质量回归和线上反馈闭环。
