"""从 Chroma 检索上下文，并调用 Ollama 生成回答。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CHROMA_DIR = SCRIPT_DIR / "chroma_db"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION_NAME = "rag_documents"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def _build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Embedding 设备: %s", device)
    LOGGER.info("Embedding 模型: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_vector_store(
    chroma_dir: Path,
    model_name: str,
    collection_name: str,
) -> Chroma | None:
    if not chroma_dir.exists() or not chroma_dir.is_dir():
        LOGGER.error("Chroma 数据库目录不存在: %s", chroma_dir)
        LOGGER.error("请先运行 chroma_ingest.py 建立向量库")
        return None
    try:
        embeddings = _build_embeddings(model_name)
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
        )
    except Exception:
        LOGGER.exception("加载 Chroma 数据库失败: %s", chroma_dir)
        return None


def retrieve_context(vector_store: Chroma, question: str, top_k: int) -> list[tuple[Any, float]]:
    """返回最相关的文本块及相似度距离。"""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    return vector_store.similarity_search_with_score(question, k=top_k)


def _source_label(document: Any) -> str:
    source = document.metadata.get("source", "未知来源")
    page = document.metadata.get("page")
    if page is not None:
        return f"{source}（PDF 第 {int(page) + 1} 页）"
    return str(source)


def _build_prompt(question: str, results: list[tuple[Any, float]]) -> str:
    context_parts: list[str] = []
    for index, (document, _score) in enumerate(results, start=1):
        context_parts.append(
            f"[资料 {index}] 来源：{_source_label(document)}\n{document.page_content.strip()}"
        )
    context = "\n\n".join(context_parts)
    return (
        "你是一个严谨的中文文档问答助手。请只依据下方检索资料回答问题，"
        "不要使用资料之外的猜测或常识。如果资料中没有答案，请明确回答：根据当前文档无法确定。\n\n"
        f"检索资料：\n{context}\n\n用户问题：{question}\n\n请直接给出简洁、准确的中文答案。"
    )


def _call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response: requests.Response | None = None
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response")
        if not answer and isinstance(data.get("message"), dict):
            answer = data["message"].get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Ollama 响应中没有有效的 response 字段")
        return answer.strip()
    except requests.exceptions.RequestException as exc:
        LOGGER.error("Ollama 请求失败: %s", exc)
        LOGGER.error("请求 URL: %s", url)
        if response is not None:
            LOGGER.error("HTTP 状态码: %s", response.status_code)
            LOGGER.error("Ollama 原始响应: %s", response.text)
        raise
    except (ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("Ollama 响应解析失败: %s", exc)
        LOGGER.error("请求 URL: %s", url)
        if response is not None:
            LOGGER.error("HTTP 状态码: %s", response.status_code)
            LOGGER.error("Ollama 原始响应: %s", response.text)
        raise
    except Exception as exc:
        LOGGER.error("Ollama 请求处理出现未预期异常: %s", exc)
        LOGGER.error("请求 URL: %s", url)
        if response is not None:
            LOGGER.error("HTTP 状态码: %s", response.status_code)
            LOGGER.error("Ollama 原始响应: %s", response.text)
        raise


def answer_question(
    vector_store: Chroma,
    question: str,
    top_k: int,
    ollama_url: str,
    ollama_model: str,
    ollama_timeout: float,
) -> str:
    results = retrieve_context(vector_store, question, top_k)
    if not results:
        return "根据当前文档无法确定。"

    print("\n================ 检索正文 ================")
    for index, (document, _score) in enumerate(results, start=1):
        print(f"\n【正文 {index}】")
        print(document.page_content.strip())

    print("\n================ 参考出处 ================")
    for index, (document, score) in enumerate(results, start=1):
        print(f"[{index}] {_source_label(document)} | 距离：{score:.4f}")

    prompt = _build_prompt(question, results)
    return _call_ollama(ollama_url, ollama_model, prompt, ollama_timeout)


def _ask_questions(
    vector_store: Chroma,
    question: str | None,
    top_k: int,
    ollama_url: str,
    ollama_model: str,
    ollama_timeout: float,
) -> None:
    questions = [question] if question else None
    while True:
        if questions is None:
            try:
                current = input("\n请输入问题（输入 quit 退出）：").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not current or current.lower() in {"quit", "exit", "q", "退出"}:
                return
        else:
            current = questions.pop(0)

        try:
            answer = answer_question(
                vector_store,
                current,
                top_k,
                ollama_url,
                ollama_model,
                ollama_timeout,
            )
            print(f"\n回答：\n{answer}")
        except Exception:
            LOGGER.exception("问题处理失败")

        if questions is not None:
            return


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Chroma 检索与 Ollama 问答")
    parser.add_argument("question", nargs="?", help="问题；省略时进入交互模式")
    parser.add_argument("--chroma-dir", type=Path, default=None, help="Chroma 持久化目录")
    parser.add_argument("--model-name", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "4")))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL))
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("OLLAMA_TIMEOUT", "120")))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    chroma_dir = args.chroma_dir or _env_path("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    vector_store = _load_vector_store(chroma_dir, args.model_name, args.collection)
    if vector_store is None:
        sys.exit(1)
    LOGGER.info("Ollama 地址: %s", args.ollama_url)
    LOGGER.info("Ollama 模型: %s", args.ollama_model)
    _ask_questions(
        vector_store,
        args.question,
        args.top_k,
        args.ollama_url,
        args.ollama_model,
        args.timeout,
    )


if __name__ == "__main__":
    main()
