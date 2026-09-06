# Demo 演示指南

`demo.py` 是面向 GitHub 访客和秋招面试的命令行演示入口。它只调用现有 `Agent` 接口和性能报告，不复制 Router、RAG、Memory 或 State 的业务逻辑。

## 1. 如何启动

在项目根目录运行：

```powershell
python demo.py
```

Demo 会显示菜单：

1. 普通 Agent 问答
2. RAG 知识库问答
3. Query Rewrite 演示
4. Memory 多轮对话
5. Agent State 追踪
6. 性能分析展示
7. 退出

普通问答、RAG、Query Rewrite 和 Memory 演示需要 Ollama 服务及对应模型；RAG 演示还需要已初始化的 Chroma 知识库。若依赖未启动，Demo 会输出可操作的提示，不会显示 Python traceback。

## 2. 面试推荐展示顺序

推荐按以下顺序讲解：

### 第一：RAG 知识库问答

选择 `2`，演示“什么是 RAG？”。重点展示 Router 决策、Tool 调用次数、检索结果数量和实际来源。

### 第二：Memory 连续对话

选择 `4`，演示“我的名字叫张三”以及“我叫什么？”。重点说明上下文保存在当前 Agent 的短期 Memory 中，并按用户隔离。

### 第三：Agent State

选择 `5`，展示一次请求的 `request_id`、routing、检索结果、距离、可靠性、Answer Strategy、Sources 和耗时。

### 第四：Query Rewrite

选择 `3`，演示带上下文的第二轮问题。Demo 会展示真实的 `query_rewrite`；如果第一次证据已足够，会明确显示“未触发”，不会伪造改写结果。

### 第五：性能分析

选择 `6`，读取 `docs/performance_report.md`，讲解 Fast Path、Prompt 优化、Embedding 预热和 SSE 状态反馈。

## 3. 面试讲解重点

### 为什么需要 Router

不是所有问题都需要检索。普通寒暄可以直接回答，只有与项目知识库相关的问题才调用 RAG Tool，从而减少无意义的检索和额外上下文。

### 为什么不是简单 RAG

系统不是每次都固定“先检索再回答”，而是由 Router、Evidence Reliability 和 Answer Strategy 共同决定执行路径；证据不足时才触发受控的 Query Rewrite，并且最多检索两次。

### 为什么需要 Reliability

检索结果数量、距离和相关性判断共同决定证据是否可靠。证据不足时，回答会明确说明边界，避免把相似但无关的片段当成事实。

### 为什么需要 State

`AgentRequestState` 将一次请求的关键阶段统一记录下来，便于调试、性能分析、Agent Trace 展示和回归评估。

### 为什么需要 Memory 隔离

多轮问答需要上下文，但不同用户之间不能共享历史。当前服务按用户身份创建独立 Agent，短期 Memory 只在对应用户范围内生效。

## 4. 故障提示

- 看到“Ollama 服务未启动”：运行 `ollama serve`，并确认已准备 `qwen3:4b`。
- 看到“知识库未初始化”：检查 Chroma 数据和项目的知识库 ingest 流程。
- 性能模式不依赖 Ollama，只读取已有报告；如果报告缺失，检查 `docs/performance_report.md`。
