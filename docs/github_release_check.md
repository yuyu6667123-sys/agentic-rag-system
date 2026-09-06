# GitHub Release Checklist

## Code

- 核心 Agent、RAG、认证、Memory、SSE 和 Tool Registry 代码已完成 Step16-Final 健康检查。
- 本次发布整理不修改 `src/`、`tests/` 或业务执行流程。
- `python -m compileall src` 用于发布前语法检查。

## Security

- `.env` 未纳入版本控制，`.env.example` 只保留配置占位符。
- SQLite 用户数据库、验证码和 Session 数据位于忽略的 `data/` 目录。
- Chroma 数据、模型权重、Embedding 缓存、Python 缓存和运行日志均已加入忽略规则。
- SMTP 密码、授权码、API Key、Token 和 Secret 不应写入代码、README 或文档。
- 发布前仍应人工复核 Git diff 和仓库历史，确认没有误提交敏感信息。

## Documentation

- [README](../README.md) 包含项目定位、架构、功能、技术栈、Demo、Evaluation、Performance、Quick Start 和项目结构。
- [项目概述](project_overview.md) 说明背景、模块和技术亮点。
- [架构图](architecture.md) 提供 GitHub 原生 Mermaid 图。
- [Demo 指南](demo_guide.md) 和 [面试材料](interview_guide.md) 说明演示和讲解路径。
- [Evaluation 报告](evaluation_report.md)、[性能报告](performance_report.md) 和 [发布检查记录](release_check_report.md) 保留验证依据。

## Reproducibility

- 使用 `pyproject.toml` 和 `uv.lock` 固定依赖解析结果。
- 安装环境：

  ```powershell
  uv venv --python 3.14
  uv sync
  Copy-Item .env.example .env
  ```

- 启动 Ollama 并准备 `qwen3:4b`：

  ```powershell
  ollama pull qwen3:4b
  python run.py
  ```

- 真实 Evaluation 需要本地 Ollama 和 Chroma 资源；普通单元/API 测试不应把真实凭据写入仓库。

## Demo

- Web 入口：`python run.py`，打开 `http://127.0.0.1:8000/`。
- CLI 入口：`python demo.py`。
- 推荐展示顺序：RAG 问答 → Memory 多轮 → Agent State → Query Rewrite → 性能报告。
- `screenshots/` 只存放实际运行截图，不提交虚构图片。

## Known Limitations

- 依赖本地 Ollama 服务和 `qwen3:4b` 模型，模型未启动时无法完成真实问答。
- GPU / CPU、显卡驱动和模型量化差异会影响 Embedding 加载、检索和生成延迟。
- 示例知识库规模有限，评估结果不能代表所有领域和大规模生产数据。
- Memory 是当前进程内的短期记忆，不提供聊天历史持久化。
- QQ SMTP 需要用户自行配置授权码，仓库不提供真实邮箱凭据。
