"""Structured knowledge-base retrieval using the existing RAG configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma

try:
    from ..rag_qa import (
        DEFAULT_CHROMA_DIR,
        DEFAULT_COLLECTION_NAME,
        DEFAULT_MODEL_NAME,
        _env_path,
        _load_vector_store,
        retrieve_context,
    )
except ImportError:
    from rag_qa import (
        DEFAULT_CHROMA_DIR,
        DEFAULT_COLLECTION_NAME,
        DEFAULT_MODEL_NAME,
        _env_path,
        _load_vector_store,
        retrieve_context,
    )


SRC_DIR = Path(__file__).resolve().parent.parent


class RagToolError(RuntimeError):
    """Raised when the knowledge-base tool cannot complete a search."""


class KnowledgeBaseNotFoundError(RagToolError):
    """Raised when the configured persisted Chroma database is missing."""


class KnowledgeBaseInitializationError(RagToolError):
    """Raised when the existing RAG code cannot initialize Chroma."""


@lru_cache(maxsize=1)
def _get_vector_store() -> Chroma:
    """Initialize Chroma and its embedding model once, then reuse the instance."""
    load_dotenv(SRC_DIR / ".env")
    chroma_dir = _env_path("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    collection_name = os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME)

    if not chroma_dir.is_dir():
        raise KnowledgeBaseNotFoundError(
            f"Chroma 数据库目录不存在: {chroma_dir}。"
            "请先运行 chroma_ingest.py 建立知识库，或检查 CHROMA_DIR 配置。"
        )

    vector_store = _load_vector_store(chroma_dir, model_name, collection_name)
    if vector_store is None:
        raise KnowledgeBaseInitializationError(
            f"Chroma 初始化失败: path={chroma_dir}, collection={collection_name!r}。"
            "请检查数据库完整性，并确保查询端配置与入库配置一致。"
        )
    return vector_store


def search_knowledge_base(
    query: str,
    top_k: int = 4,
    filters: dict | None = None,
) -> dict:
    """Search Chroma and return evidence without calling an LLM."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k 必须是大于 0 的整数")
    if filters is not None and not isinstance(filters, dict):
        raise ValueError("filters 必须是 dict 或 None")

    normalized_query = query.strip()
    vector_store = _get_vector_store()

    try:
        if filters:
            matches = vector_store.similarity_search_with_score(
                normalized_query,
                k=top_k,
                filter=filters,
            )
        else:
            matches = retrieve_context(vector_store, normalized_query, top_k)
    except Exception as exc:
        raise RagToolError(
            "Chroma 检索失败: "
            f"query={normalized_query!r}, top_k={top_k}, filters={filters!r}。"
        ) from exc

    results: list[dict[str, Any]] = []
    for document, distance in matches:
        metadata = document.metadata if isinstance(document.metadata, dict) else {}
        results.append(
            {
                "content": document.page_content,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "distance": float(distance),
            }
        )

    return {"query": normalized_query, "results": results}


__all__ = [
    "KnowledgeBaseInitializationError",
    "KnowledgeBaseNotFoundError",
    "RagToolError",
    "search_knowledge_base",
]
