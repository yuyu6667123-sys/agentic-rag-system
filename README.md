# Agentic RAG

基于 FastAPI、Ollama、LangChain 和 Chroma 的 Agentic RAG 示例项目，包含 QQ 邮箱验证码认证、短期 Memory、SSE Streaming 和可追溯来源。

## Requirements

- Python 3.14+
- Ollama，模型 `qwen3:4b`
- Windows/Linux/macOS 均可运行；向量库和 Embedding 首次启动可能需要下载模型文件

## Setup

```powershell
uv venv --python 3.14
uv sync
Copy-Item .env.example .env
```

编辑 `.env`，填写 QQ SMTP 授权码和其他本地配置。不要把 `.env` 提交到 Git。

启动 Ollama 后运行 API：

```powershell
ollama pull qwen3:4b
uv run uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

## GPU / CUDA

`pyproject.toml` 声明的是可安装的 Torch 基础范围。Torch 的 CPU/CUDA wheel 取决于操作系统、Python 和驱动组合；不要在项目依赖中硬编码本机 CUDA 路径。

如果需要 NVIDIA GPU，请先按 PyTorch 官方选择器安装与驱动匹配的 CUDA wheel，再执行其余依赖安装。例如使用官方索引提供的对应 `torch`/`torchvision` 版本。确认：

```python
import torch
print(torch.cuda.is_available())
```

若返回 `False`，Embedding 会按代码自动使用 CPU；这不影响功能，但首次加载和检索会更慢。CUDA Toolkit、显卡驱动和 Ollama GPU 支持属于运行环境配置，不由本项目自动安装或管理。

## Tests

```powershell
uv pip check --python .venv\\Scripts\\python.exe
uv run python -m unittest tests.test_auth_service tests.test_auth_api tests.test_api_server tests.test_agent_request_state
```

真实 Ollama/Chroma 评估需要服务运行，并显式设置：

```powershell
$env:RUN_REAL_EVALUATION="1"
uv run python -m unittest tests.evaluation.test_agent_evaluation -v
```
