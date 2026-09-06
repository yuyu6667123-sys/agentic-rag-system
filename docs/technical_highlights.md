# Agentic RAG 技术亮点

## 1. Agent 架构设计

普通 RAG 通常是固定的“问题 → 检索 → 生成”流程，所有问题都会触发检索，难以区分闲聊、一般问题和知识库问题。

本项目采用 Agentic RAG：

```text
用户问题
  ↓
Agent Router
  ↓
是否需要工具
  ├─ 否 → LLM 直接回答
  └─ 是 → RAG 检索 → 证据评估 → 回答
```

Agent 的价值是把检索变成有条件的决策动作，并在证据不足时进入有限、可追踪的后续步骤，而不是无条件执行固定链路。

## 2. 动态路由

Router 使用 Qwen3 判断当前问题是否需要查询项目知识库：

- `false`：不调用 RAG Tool，直接进入最终回答。
- `true`：通过 Tool Registry 调用唯一的 `search_knowledge_base()`。

Router 只要求模型返回结构化 JSON；当模型返回非法 JSON、字段缺失、类型错误或调用异常时，系统使用已有关键词判断作为 fallback。这样既利用语义判断，也保留了可预测的异常降级路径。

## 3. Retrieval Reliability

检索结果本身不等于可靠证据。系统综合以下信息判断是否可以正常回答：

- **distance**：衡量查询和召回片段的向量距离，辅助识别低相似度结果。
- **relevance**：由模型判断召回内容是否与用户问题相关。
- **evidence**：由模型判断现有片段是否足以支撑回答。

如果第一次证据不足，系统生成 Query Rewrite 后再检索一次；最多两次 Tool 调用，避免无限循环。最终根据评估结果选择 reliable、cautious 或 insufficient 策略，不能用没有证据支持的内容冒充知识库事实。

## 4. Memory 系统

- 保存 user / assistant 消息，支持多轮上下文和代词指代。
- Memory 在 Python 进程内维护，并限制最大消息数量，避免历史无限增长。
- 服务层按 `user_id` 隔离 Agent 实例，不同用户的 Session 不共享对话历史。
- 当前 Memory 是短期能力，不宣称提供聊天历史数据库或长期记忆。

## 5. 可观测性

每次请求创建独立的 `AgentRequestState`，关键字段包括：

- `request_id`
- `routing_decision`
- `tool_call_count`
- `retrieval_result_count`
- `answer_strategy`

同时记录 `best_distance`、`retrieval_reliable`、`relevance_decision`、`query_rewrite`、`sources` 和 `total_time_ms`。这让一次请求的路由、检索、回答策略和来源可以被复盘，也方便异常定位、性能分析和前端 Agent Trace 展示。

## 6. 工程优化

Step16 阶段在不改变接口和核心流程的前提下完成了几项低风险优化：

- **Fast Path**：高置信检索结果满足配置条件时，跳过重复的相关性/充分性判断。
- **Prompt 压缩**：删除最终回答 Prompt 中的重复约束，减少无效输入字符。
- **History 限制**：生成阶段只注入有限的最近对话，Agent Memory 仍保持用户隔离和消息上限。
- **性能日志**：记录 Router、Retrieval、Generation 和 Total 等阶段耗时，为后续 P50/P95 分析提供依据。
- **SSE 状态事件**：在 token 之前反馈分析、检索和生成阶段，改善用户对长响应的感知。
