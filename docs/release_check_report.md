# Release Check Report

检查日期：2026-09-05

本次 Step 16-G 只修复公开发布所需的配置、依赖元数据和忽略规则，没有修改 Agent、RAG、Memory、SSE、认证或 FastAPI 核心逻辑。

## 修改内容

- `.env.example`：移除真实邮箱和 SMTP 授权码，改为 `your_qq_email@qq.com`、`your_qq_smtp_authorization_code` 等占位符；保留完整认证配置模板。
- `pyproject.toml`：声明 FastAPI、uvicorn、requests、dotenv、pypdf、LangChain、Chroma、sentence-transformers、transformers 和 Torch 依赖范围。
- `uv.lock`：根据新的项目依赖声明刷新并通过锁文件一致性检查。
- `.gitignore`：增加运行数据库、Chroma、模型缓存、权重文件、测试/类型检查缓存、日志和 `.idea` 忽略规则，同时保留 `.env.example` 可提交。
- `README.md`：补充新环境安装、Ollama、CPU/CUDA 和测试说明。
- `docs/release_check_report.md`：本报告。

## 风险检查

| 检查项 | 结果 | 说明 |
|---|---|---|
| `.env.example` 无真实凭据 | PASS | 邮箱、SMTP 密码均为占位符 |
| `.env` 不纳入发布 | PASS | `.gitignore` 忽略 `.env`；本机 `.env` 未在报告中读取输出 |
| 个人绝对路径 | PASS（发布文件） | 源代码无个人用户名绝对路径；`.idea` 已整体忽略 |
| 硬编码敏感信息 | PASS（发布文件） | 未发现 SMTP token、API key 或密码常量；运行时从环境读取 |
| 数据库/向量库 | PASS | `data/`、`src/chroma_db/`、`chroma_db/` 已忽略 |
| 模型缓存/权重 | PASS | `.cache/`、Hugging Face/模型目录及常见权重扩展名已忽略 |
| 临时测试文件 | PASS | 测试临时目录规则已存在；没有新增临时业务文件 |
| 业务逻辑改动 | PASS | 本阶段未修改 `src` |

注意：本机 `.env` 当前仍包含本地 SMTP 配置，因其已被 Git 忽略没有删除；若凭据曾经被提交到远程仓库，必须在发布前撤销并轮换授权码，且清理 Git 历史。

## Dependency Verification

```text
uv lock --check
Resolved 146 packages

uv pip check --python .venv\\Scripts\\python.exe
All installed packages are compatible
```

关键导入检查通过：

```text
torch
transformers
langchain
langchain_community
langchain_chroma
langchain_huggingface
chromadb
fastapi
uvicorn
sentence_transformers
dotenv
requests
pypdf
```

当前 `.venv` 没有 `pip` 模块，因此 `python -m pip list` 不可用；使用 `uv pip list` 和 `uv pip check` 完成等价环境核验。

## Tests

认证、Agent、SSE、State、稳定性、Memory 和分类评估测试：

```text
Ran 26 tests
OK (skipped=4)
```

跳过项是默认不访问真实 Ollama 的评估用例，需要显式设置 `RUN_REAL_EVALUATION=1`。

Observability：

```text
Ran 3 tests
OK
```

此前真实 Ollama + Chroma 评估：

- Router：代表性普通/知识库 Case 通过
- Retrieval：高相关、模糊、无关 Case 完成
- Query Rewrite：二次检索和 `tool_call_count=2` 通过
- Memory：两轮代词上下文和消息顺序通过

## Syntax

```text
15 个 src Python 文件 py_compile 通过
```

## Runtime Readiness

- Ollama `/api/tags`：HTTP 200，模型 `qwen3:4b` 存在。
- Chroma collection：`rag_documents`，10 条记录。
- FastAPI：可启动，Chroma warmup 完成。
- 认证、登录拦截、SSE error/done、Memory 隔离回归通过。

## Remaining Release Risks

1. `pyproject.toml` 已声明依赖，但 GPU/CUDA wheel 仍需按平台和驱动单独安装，README 已说明。
2. 当前真实 Ollama `/api/generate` 在健康检查期间出现读取超时；发布前应在目标机器单独确认模型资源和响应稳定性。
3. `rag_data/` 是示例知识库；生产部署应确认其授权、内容和更新流程。
4. `uv.lock` 当前按 Python 3.14 解析；若支持其他 Python 版本，应在对应平台重新验证锁文件和 Torch wheel。

## Conclusion

**发布配置检查：PASS（有条件）。**

公开仓库所需的凭据模板、依赖声明和忽略规则已补齐，代码回归和语法检查通过。发布前仍必须轮换任何曾暴露的 SMTP 授权码，并在目标环境确认 Ollama 生成稳定；本报告没有为了通过检查修改业务逻辑。
