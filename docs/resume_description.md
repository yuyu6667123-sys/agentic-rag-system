# Agentic RAG 智能问答系统

## 版本 1：简历版

**项目名称：** Agentic RAG 智能问答系统

**项目描述：** 面向项目知识库的 LLM Agent 应用，支持动态路由、受控检索、证据可靠性判断、多轮短期记忆和可追溯来源。

**技术栈：** Python、FastAPI、Qwen3、Ollama、Chroma、Sentence Transformer、Embedding、RAG、Agent、SQLite、SSE

**项目职责：**

- 设计 Agent Router，根据问题和对话上下文判断直接回答或调用 RAG Tool，并保留关键词 fallback 提升异常场景可用性。
- 实现受控 Agent Loop：对检索结果进行 distance、relevance 和 evidence sufficiency 评估，证据不足时生成 Query Rewrite，最多执行两次检索。
- 建立统一 AgentRequestState 和结构化运行日志，记录 request id、工具调用、检索质量、回答策略、来源和阶段耗时。
- 实现有界短期 Memory 与按用户隔离的 Agent 实例，支持多轮指代并避免不同 Session 共享上下文。
- 通过 FastAPI、SSE Streaming、QQ 邮箱验证码和 Session Cookie 串联模型能力与可访问的 Web 应用，并使用 Evaluation 测试 Router、RAG、Rewrite、Memory 和稳定性。

## 版本 2：Boss 直聘 / 招聘网站版

我负责开发一个基于 Qwen3 和 Chroma 的 Agentic RAG 智能问答系统，解决普通问题不需要检索、知识库问题需要可靠证据以及多轮对话上下文管理等问题。系统先由 LLM Router 判断是否调用知识库，检索后结合 distance、相关性和证据充分性选择回答策略；证据不足时自动生成 Query Rewrite，并将检索次数限制为最多两次。工程上使用 FastAPI 提供服务和 SSE 流式输出，使用 SQLite + Session Cookie 完成 QQ 邮箱验证码登录，利用 AgentRequestState、结构化日志和 Evaluation 测试追踪每次请求，Memory 按用户隔离并设置长度上限。

## 版本 3：面试自我介绍版

我主要负责开发这个 Agentic RAG 智能问答项目，核心目标是让模型能够根据问题决定是否需要访问项目知识库，而不是每次都固定检索。项目中我设计了 Router、RAG Tool 和受控 Agent Loop，对检索结果同时检查距离、相关性和证据充分性，必要时进行一次 Query Rewrite，再生成带真实来源的回答。其中比较有挑战的是如何在模型不稳定、证据不足和多轮上下文之间保持可控，所以我增加了关键词 fallback、最多两次 Tool 调用、Answer Strategy、AgentRequestState 和结构化日志。最终通过 FastAPI、SSE、认证和按用户隔离的短期 Memory，把这套流程接成了可测试的应用。
