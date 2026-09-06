# Agentic RAG Performance Report

## Scope

Step 16 的首轮优化只涉及低风险性能可观测性和用户反馈，不改变 Router、RAG、Reliability、Query Rewrite、Memory、认证或既有 SSE `token/done/error` 事件语义。

## Baseline

对真实 Ollama `qwen3:4b` 和 Chroma 的问题“什么是 RAG？”进行了一次基线测量：

| 阶段 | 耗时 |
|---|---:|
| Router | 12,326.8 ms |
| Chroma Retrieval（含本进程 Embedding 冷启动） | 11,756.3 ms |
| Relevance | 16,999.1 ms |
| Evidence Sufficiency | 11,789.2 ms |
| Final Generation | 58,179.6 ms |
| Total | 111,051.9 ms |

本次请求共调用 4 次 LLM、1 次 Chroma Tool。最大瓶颈是最终 Qwen 生成，其次是 Relevance 判断。

## Implemented Optimizations

1. `Agent` 增加结构化阶段耗时日志：`router`、`retrieval`、`reliability`、`sufficiency`、`query_rewrite`、`generation` 和 `total`。日志只记录阶段、耗时、计数和结果摘要，不记录完整 Prompt、thinking 或 Evidence 正文。
2. FastAPI 启动时预热已缓存的 Chroma/Embedding。预热失败只记录 warning，不阻塞服务启动；后续 RAG 请求仍使用原有明确异常。
3. SSE 增加可选的 `status` 事件：分析、检索、生成。原有 `token`、`done`、`error` 事件不变，前端将状态显示为连接区文案。

## Before / After

本轮没有合并 LLM 判断、改变可靠性阈值或调整 `top_k`，因此不会声称已经降低模型推理耗时。请求内的算法和 LLM 次数保持不变；预热将 Embedding 冷启动从首个用户请求前移到服务启动阶段。阶段日志可用于后续收集多请求 P50/P95 和首 token 时间，再决定是否进行更高风险优化。

| 指标 | 优化前 | 本轮优化后 |
|---|---|---|
| LLM 调用次数 | 可靠 RAG 通常 4 次，Rewrite 最多 5 次 | 不变 |
| Chroma 配置 | `top_k=4`，原阈值 | 不变 |
| 首次请求 Embedding 冷启动 | 计入首个请求 | 服务启动预热时承担 |
| 用户前置反馈 | 仅生成中 | SSE status：分析/检索/生成 |
| 最终 token / done / error 契约 | 保持 | 保持 |

## Verification

- Agent、认证、SSE、State、Observability 和评估测试应继续使用现有命令运行。
- 当前新增分类测试默认不访问真实模型；真实评估通过 `RUN_REAL_EVALUATION=1` 显式开启。
- 只读健康检查：Ollama `/api/tags` HTTP 200；Chroma collection `rag_documents` 有 10 条记录。

## Deferred High-risk Changes

以下优化暂不采用：合并 Relevance/Sufficiency、跳过 Router、改变距离阈值、修改 `top_k`、更换模型或限制生成长度。它们可能改变 `AgentRequestState`、Answer Strategy、Query Rewrite 或回答质量，需要独立 A/B 评估。

## Step 16-B: High-confidence Fast Path

`Agent` 现在支持两个配置项：

- `HIGH_CONFIDENCE_DISTANCE`，默认 `0.65`
- `HIGH_CONFIDENCE_MIN_RESULTS`，默认 `4`

只有当一次检索返回至少 4 条结果且 `best_distance <= 0.65` 时，才跳过 Relevance 和 Sufficiency 两个重复的 LLM 判断。Router、RAG Tool、Reliability 状态、最终生成、Sources 和 Memory 均保留。

真实问题“什么是 RAG？”的对比：

| 指标 | 优化前基线 | Fast Path 后 |
|---|---:|---:|
| LLM 调用次数 | 4 | 2 |
| Tool 调用次数 | 1 | 1 |
| 最佳距离 | 0.6431 | 0.6431 |
| Retrieval reliability | true | true |
| Relevance state | `llm_true` | `skipped_high_confidence` |
| 总耗时 | 111,051.9 ms | 52,732.0 ms |

此次真实测量减少约 52.5% 总耗时；主要来自跳过约 12-17 秒的判断请求，最终生成仍由 Qwen3:4b 主导。低置信、模糊、无关和 Query Rewrite 场景不会命中该路径。

## Step 16-C: Generation Prompt Optimization

最终回答 Prompt 现在只保留一份核心约束，并记录以下长度指标：`system_chars`、`history_chars`、`evidence_chars`、`prompt_chars` 和 `estimated_input_tokens`。历史注入最多保留最近 2 轮（4 条消息），而 Agent 内存仍保持原有 12 条消息上限和用户隔离。

Ollama 最终生成请求通过 `options.num_predict` 支持可配置上限，环境变量为 `OLLAMA_MAX_NEW_TOKENS`，默认 `128`；该参数同时用于普通非流式回答和最终 Streaming，Router、Relevance、Sufficiency、Rewrite 不受影响。

本轮尝试对 `128/96/64` 做真实端到端对比时，Ollama 在 Router 请求阶段出现 120 秒及 180 秒读取超时，无法形成三组完整、可比的生成时间和质量数据。该环境问题已如实记录，未将未完成实验宣称为性能收益，也没有因此改变默认上限或生产模型。

已完成的确定性验证：生成请求包含 `options: {"num_predict": 128}`；6 轮历史注入时 Prompt 只包含最近 2 轮；核心 Agent、SSE、Memory、认证和 Observability 回归保持通过。

因此当前可确认的优化收益是：Prompt 重复内容减少、历史输入有界、生成长度可配置；真实生成平均耗时、首 token 时间和回答质量的量化变化需要在 Ollama 稳定后重新运行三组基准。
