"""切分文档、生成向量并持久化到 Chroma。"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from document_loader import load_documents
except ImportError:
    from outputs.document_loader import load_documents


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DOCUMENT_DIR = SCRIPT_DIR / "rag_data"
DEFAULT_CHROMA_DIR = SCRIPT_DIR / "chroma_db"
DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION_NAME = "rag_documents"


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


def _split_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_documents(documents)


def _document_ids(chunks: list) -> list[str]:
    ids: list[str] = []
    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", ""))
        page = str(chunk.metadata.get("page", ""))
        digest = hashlib.sha1(
            f"{source}|{page}|{index}|{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        ids.append(digest)
    return ids


def _sync_vector_store(vector_store: Chroma, chunks: list, ids: list[str]) -> tuple[int, int]:
    """比较当前切片与 Chroma，删除过期数据并添加缺失数据。"""
    existing_ids = set(vector_store.get().get("ids", []))
    desired_ids = set(ids)
    stale_ids = existing_ids - desired_ids
    missing_ids = desired_ids - existing_ids

    LOGGER.info("Chroma 现有切片数量: %d", len(existing_ids))
    LOGGER.info("当前文档切片数量: %d", len(desired_ids))
    if not stale_ids and not missing_ids:
        LOGGER.info("Chroma 与当前文档一致，无需更新")
        return 0, 0

    if stale_ids:
        vector_store.delete(ids=list(stale_ids))
        LOGGER.info("已删除过期向量切片: %d", len(stale_ids))

    if missing_ids:
        chunks_by_id = {item_id: chunk for item_id, chunk in zip(ids, chunks)}
        missing_chunks = [chunks_by_id[item_id] for item_id in missing_ids]
        vector_store.add_documents(missing_chunks, ids=list(missing_ids))
        LOGGER.info("已新增向量切片: %d", len(missing_ids))

    LOGGER.info("Chroma 已同步到当前文档目录")
    return len(missing_ids), len(stale_ids)


def ingest_documents(
    document_dir: str | Path,
    chroma_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    reset: bool = False,
) -> int:
    """加载、切分并同步到 Chroma，返回本次新增的切片数量。"""
    source_dir = Path(document_dir).expanduser()
    persist_dir = Path(chroma_dir).expanduser()
    if not source_dir.exists() or not source_dir.is_dir():
        LOGGER.error("文档目录不存在或不是文件夹: %s", source_dir)
        return 0
    if chunk_overlap >= chunk_size:
        LOGGER.error("chunk_overlap 必须小于 chunk_size")
        return 0

    documents = load_documents(source_dir)
    LOGGER.info("原始 Document 数量: %d", len(documents))
    chunks = _split_documents(documents, chunk_size, chunk_overlap)
    LOGGER.info("切分后 Document 数量: %d", len(chunks))

    if not documents:
        LOGGER.warning("当前文档目录没有可入库的文件")

    # --reset 保留为强制全量重建选项。
    if reset and persist_dir.exists():
        LOGGER.info("清理 Chroma 持久化目录: %s", persist_dir)
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = _build_embeddings(model_name)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    ids = _document_ids(chunks)
    added_count, deleted_count = _sync_vector_store(vector_store, chunks, ids)
    LOGGER.info("Chroma 持久化目录: %s", persist_dir)
    if deleted_count:
        LOGGER.info("本次同步删除过期切片: %d", deleted_count)
    return added_count


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")
    parser = argparse.ArgumentParser(description="将文档向量化并写入 Chroma")
    parser.add_argument("document_dir", nargs="?", type=Path, default=None, help="文档目录")
    parser.add_argument("--chroma-dir", type=Path, default=None, help="Chroma 持久化目录")
    parser.add_argument("--model-name", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "800")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "120")))
    parser.add_argument("--reset", action="store_true", help="清空当前 Chroma 目录后重新入库")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    document_dir = args.document_dir or _env_path("DOCUMENT_DIR", DEFAULT_DOCUMENT_DIR)
    chroma_dir = args.chroma_dir or _env_path("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    try:
        added = ingest_documents(
            document_dir=document_dir,
            chroma_dir=chroma_dir,
            model_name=args.model_name,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            reset=args.reset,
        )
        print(f"处理完成，本次新增 {added} 个向量切片")
    except Exception:
        LOGGER.exception("RAG 向量入库失败")
        print("处理失败，请查看上方错误日志")


if __name__ == "__main__":
    main()
