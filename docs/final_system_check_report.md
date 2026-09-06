# Agentic RAG System Final Check

检查日期：2026-09-05

本次为 Step 16-Final 全链路健康检查，只进行只读检查和隔离测试，未修改业务代码、配置或认证逻辑。

## Environment

| 检查项 | 结果 | 证据 |
|---|---|---|
| Python | PASS | Python 3.14.3 |
| 虚拟环境 | PASS | `.venv\\Scripts\\python.exe` 可启动 |
| 依赖导入 | PASS | torch、transformers、LangChain、Chroma、FastAPI、uvicorn、sentence-transformers、dotenv、requests、pypdf 均可导入 |
| 依赖兼容性 | PASS | `uv pip check --python .venv\\Scripts\\python.exe`：All installed packages are compatible |
| `pip list` | FAIL | `.venv` 没有 pip 模块；`python -m pip list` 返回 `No module named pip`。可用 `uv pip list` 替代 |
| `.env` | PASS | 文件存在；未在报告中输出内容 |
| `.env.example` | FAIL | SMTP_USERNAME、SMTP_FROM、SMTP_PASSWORD 不是占位符，包含看似真实的凭据格式 |
| 项目依赖声明 | FAIL | `pyproject.toml` 的 `dependencies = []`，新环境无法从项目元数据恢复运行依赖 |

已安装关键版本：`langchain 0.3.27`、`langchain-community 0.3.27`、`langchain-chroma 0.2.5`、`chromadb 1.5.9`、`fastapi 0.141.1`、`uvicorn 0.52.4`、`sentence-transformers 5.7.0`、`torch 2.14.0`。

## Startup

| 检查项 | 结果 | 证据 |
|---|---|---|
| Ollama 服务 | PASS | `GET http://127.0.0.1:11434/api/tags` 返回 HTTP 200 |
| Ollama 模型 | PASS | 返回 `qwen3:4b` |
| FastAPI 启动 | PASS | `python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000` 启动成功，Chroma warmup 完成 |
| `/` 未登录 | PASS | HTTP 303，Location `/login` |
| `/login` | PASS | HTTP 200，页面可返回 |
| `/docs` | PASS | HTTP 200 |
| `/auth/me` 未登录 | PASS | HTTP 401 |
| `/chat/stream` 未登录 | PASS | HTTP 401 |

## Authentication

认证单元及接口测试共 13 个，全部通过：

- 新用户注册/登录
- 已存在用户登录
- 正确、错误、过期、重复验证码
- 验证码冷却和邮箱格式校验
- Session Cookie、HttpOnly、SameSite
- 退出登录和受保护路由
- 多用户 Agent/Memory 隔离

真实 QQ SMTP 未发送邮件，测试使用隔离的邮件发送替身；`.env.example` 凭据暴露问题必须在部署前处理。

## Chat UI

| 检查项 | 结果 |
|---|---|
| 登录页可返回 | PASS |
| 已登录页面包含用户邮箱、输入框、发送/停止/退出控件 | PASS（接口页面断言） |
| 未登录跳转登录 | PASS |
| SSE 状态事件 | PASS |

隔离 SSE 验证事件顺序为：

```text
status(analysis)
status(retrieval)
status(generation)
token*
done
```

原有 `token/done/error` 事件契约仍通过回归测试。

## Agent

核心 State、Router、Fast Path、Answer Strategy、Sources、Memory 和 Observability 测试均通过。

真实 Ollama 最终生成在本次健康检查中未能完成：`/api/generate` 使用 `num_predict=16` 的短诊断请求在 30 秒内 ReadTimeout。Ollama 标签接口正常，因此更可能是当前模型生成进程/运行负载问题，而不是 HTTP 服务未启动。

## RAG

| 检查项 | 结果 | 证据 |
|---|---|---|
| Chroma collection | PASS | `rag_documents` |
| Chroma records | PASS | 10 条记录 |
| Embedding 加载 | PASS | `BAAI/bge-small-zh-v1.5` 可加载，服务启动 warmup 成功 |
| 真实检索 | PASS | “什么是RAG？”返回 4 条结果，best distance 约 0.6431，来源正常 |
| 空知识库模拟 | PASS | 进入二次检索/`insufficient` 策略，不崩溃，不进行第三次检索 |
| Fast Path | PASS（此前真实验证） | `skipped_high_confidence`，Tool 1 次 |

实际持久化库位于 `src/chroma_db`，而源文档位于 `rag_data/`。

## Memory

| 检查项 | 结果 |
|---|---|
| 两轮消息顺序 | PASS：`user/assistant/user/assistant` |
| 最近历史限制 | PASS：Agent 内存最多 12 条消息，Prompt 最多注入最近 2 轮 |
| 清空历史 | PASS |
| 多用户隔离 | PASS |
| 代词多轮真实模型测试 | 未执行 | 需要显式 `RUN_REAL_EVALUATION=1` 且当前 Ollama 生成超时 |

## Streaming

| 场景 | 结果 |
|---|---|
| 正常 status/token/done | PASS（隔离 Agent + FastAPI 测试） |
| Agent/Ollama 异常转 SSE error | PASS |
| 客户端提前关闭 | PASS，残缺 assistant 不写入 Memory |
| 真实 Ollama Streaming | 当前环境未完成 | `/api/generate` 非流式短诊断已超时 |

## Evaluation

稳定性和隔离评估：

```text
Ran 7 tests in 0.121s
OK (skipped=1)
```

核心/认证/SSE/State/评估分类命令：

```text
Ran 26 tests
OK (skipped=4)
```

Observability：

```text
Ran 3 tests
OK
```

真实评估案例此前已完成并通过：Router、Retrieval、Query Rewrite、Memory，共覆盖 7 个真实 Agent Case。当前分类测试默认跳过真实 Ollama，需要 `RUN_REAL_EVALUATION=1` 才会重跑。

## Performance

`docs/performance_report.md` 已包含：

- Step 16-A：阶段计时、Chroma warmup、SSE status
- Step 16-B：高置信 Fast Path
- Step 16-C：Prompt 压缩、历史限制、`OLLAMA_MAX_NEW_TOKENS`

性能日志字段已验证：

```text
stage=router
stage=retrieval
stage=generation
stage=total
```

Fast Path 此前真实验证：

- LLM 调用：4 次降至 2 次
- Tool 调用：保持 1 次
- `relevance_decision=skipped_high_confidence`
- `retrieval_reliable=true`

`128/96/64` 真实生成对比本次无法完成，因为 Ollama 生成阶段超时；报告没有虚构平均生成时间、首 token 时间或质量提升。

## Problems Found

### FAIL-1：Ollama 生成请求超时

- 位置：运行时 `http://127.0.0.1:11434/api/generate`
- 原因：`/api/tags` 正常，但短生成请求在 30 秒内 ReadTimeout；Step16-C 的 120/180 秒 Router 诊断也曾超时。
- 影响：真实最终回答、真实 Streaming 和 128/96/64 质量基准无法在本次验收中完成。
- 建议：单独检查 Ollama 进程、模型加载状态、CPU/GPU 资源和模型日志；确认服务稳定后再重跑真实 Evaluation。未在本次验收中重启或修改 Ollama。

### FAIL-2：`.env.example` 包含疑似真实 SMTP 凭据

- 位置：[`.env.example`](../.env.example)
- 原因：模板中的 SMTP_USERNAME、SMTP_FROM、SMTP_PASSWORD 不是占位符。
- 影响：存在凭据泄露和误提交风险。
- 建议：立即撤销/轮换该 SMTP 授权码，并将模板改为明显的占位符；不要把真实 `.env` 提交到 Git。

### FAIL-3：项目元数据未声明依赖

- 位置：[pyproject.toml](../pyproject.toml)
- 原因：`dependencies = []`。
- 影响：新用户创建虚拟环境后无法仅凭项目配置恢复运行环境。
- 建议：下一开发步骤补齐依赖声明或明确以锁文件/安装脚本为唯一恢复入口。

### FAIL-4：Git 忽略规则不完整

- 位置：[`.gitignore`](../.gitignore)
- 当前已忽略：`.env`、`data/*.db`、测试临时目录、`__pycache__`、`*.py[cod]`。
- 缺少：`src/chroma_db/`、本地模型/Embedding 缓存目录及其他缓存文件规则。
- 建议：部署前确认向量库和模型文件的提交策略；若不入库，增加精确的忽略规则。

## Overall Assessment

**整体状态：有条件通过。**

应用结构、FastAPI 启动、认证、Chroma 初始化、RAG Tool、Agent State、Memory 隔离、SSE 合同和本地异常处理均通过检查。当前不能称为“从零可复现且生产就绪”，主要阻塞项是 Ollama 生成不稳定、敏感 SMTP 配置泄露、项目依赖元数据为空和 Git 忽略规则不完整。

本次没有修改代码，未进入后续开发步骤。
