# Agentic RAG System 面试问题库

## Agent 问题

### Q1：为什么不用普通 RAG？

普通 RAG 固定执行“检索再回答”，即使是问候或不需要项目资料的问题也会增加检索和上下文成本。本项目先用 Router 判断是否需要知识库，再按证据可靠性决定回答策略；当证据不足时还可以受控地进行一次 Query Rewrite。这样检索是 Agent 的决策动作，而不是每次请求的固定步骤。

### Q2：Router 为什么需要 LLM 判断？

知识库问题的表达方式很多，单纯关键词规则容易漏检，也容易把普通问题误判为知识库问题。LLM Router 可以结合问题语义和当前对话上下文做分类；代码中仍保留关键词 fallback，避免模型返回非法 JSON 或服务异常时整条链路失去判断能力。

### Q3：为什么需要 Retrieval Reliability？

检索到片段不等于检索到了可用证据。系统需要同时观察结果数量、最佳 distance、相关性判断和证据充分性，再选择 reliable、cautious 或 insufficient 策略。这样可以在证据不充分时明确边界，降低把相似但不相关内容当成事实的风险。

### Q4：如果检索结果错误怎么办？

先通过 distance、结果数量和相关性判断识别低质量证据；如果第一次证据不足，生成 Query Rewrite 后最多再检索一次。两次检索后仍不可靠，就使用谨慎或不足策略回答，明确说明当前知识库无法确定，而不是继续无限调用或编造内容。最终来源只来自实际检索结果。

## RAG 问题

### Q：为什么选择 Chroma？

Chroma 适合本地原型和可复现实验，部署成本低，能够持久化向量、文档片段和 metadata。对于当前项目规模，它比引入远程向量服务更容易启动和调试，同时保留后续替换存储层的边界。

### Q：为什么使用 Embedding 模型？

Embedding 将文档和查询映射到同一向量空间，Retriever 可以按语义相似度召回内容。相比只做关键词匹配，它对同义表达和自然语言问题更鲁棒；distance 也能作为证据质量评估的输入。

### Q：top-k 如何确定？

top-k 需要在召回覆盖率、上下文长度和检索耗时之间折中。当前实现使用已有配置和评估结果，不在 Demo 层重新改变检索参数；实际项目中应通过高相关、模糊、无关问题的离线评估，以及延迟和回答质量对比来确定合理范围。

### Q：Query Rewrite 解决什么问题？

用户问题可能过于模糊、带代词，或第一次检索没有命中合适片段。Query Rewrite 根据原问题、历史上下文和第一次 Evidence 生成更适合向量检索的新 Query。当前流程最多二次检索，避免无限循环。

## 性能问题

### Q：为什么系统响应慢？

一次 RAG 请求的主要耗时通常来自多个 LLM 阶段和最终 Generation：Router、Retrieval、Relevance / Reliability 判断、必要时的 Rewrite，以及最终回答生成。Embedding 首次加载也可能造成冷启动延迟。项目通过阶段性能日志记录 `router`、`retrieval`、`generation` 和 `total`，用实际数据定位瓶颈。

Step16 已采用的低风险优化包括：

- **Fast Path**：高置信检索时跳过重复的相关性/充分性判断。
- **Prompt 优化**：删除重复约束，减少无效输入。
- **History 限制**：只向生成 Prompt 注入有限的最近对话，同时保留 Memory 的用户隔离和消息上限。
- **性能日志**：记录各阶段耗时，支持继续做 P50/P95 和首 token 分析。

## 工程问题

### Q：如何保证用户数据隔离？

认证成功后服务端从 Session 得到 `user_id`，并按用户创建或复用独立 Agent 实例。短期 Memory 属于该 Agent，不通过全局共享，因此用户 A 的上下文不会被用户 B 读取。Chat 和 SSE 接口也都要求有效 Session。

### Q：如何处理模型异常？

Agent 将 Ollama 调用错误转换为明确的 Agent 错误；SSE 层发送 `error` 事件并结束当前流。Streaming 只有在收到 `done=true` 后才保存 assistant Memory，中途异常或取消不会保存残缺回答。Demo 层进一步把常见服务不可用情况转换成可操作提示。

### Q：如何测试 Agent？

测试分为单元、接口和真实能力评估：Router 验证普通/知识库分类，RAG 记录结果数量、distance 和可靠性，Rewrite 验证二次 Tool 调用，Memory 验证多轮和多用户隔离，Stability 模拟 Ollama、SSE、认证和知识库异常。真实 Ollama/Chroma 评估通过显式环境变量开启，避免普通回归测试依赖外部服务。
