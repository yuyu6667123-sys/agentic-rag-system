"""批量加载 PDF、TXT 和 Markdown 文档。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
LOGGER = logging.getLogger(__name__)


def _iter_supported_files(folder: Path) -> Iterable[Path]:
    """递归返回支持的文件，并按路径排序保证处理顺序稳定。"""
    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: str(path).casefold(),
    )


def _load_file(file_path: Path) -> list[Document]:
    """使用对应的 LangChain 加载器读取单个文件。"""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        # Markdown 使用文本加载器，保留原文并避免引入额外解析依赖。
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    documents = loader.load()
    for document in documents:
        document.metadata.setdefault("source", str(file_path))
    return documents


def load_documents(folder_path: str | Path) -> list[Document]:
    """递归加载文件夹中的 PDF、TXT、MD，单个文件失败时继续处理。"""
    folder = Path(folder_path).expanduser()
    if not folder.exists():
        LOGGER.error("输入文件夹不存在: %s", folder)
        return []
    if not folder.is_dir():
        LOGGER.error("输入路径不是文件夹: %s", folder)
        return []

    documents: list[Document] = []
    files = list(_iter_supported_files(folder))
    if not files:
        LOGGER.warning("文件夹中没有支持的文件（.pdf、.txt、.md）: %s", folder)
        return documents

    for file_path in files:
        try:
            loaded = _load_file(file_path)
            documents.extend(loaded)
            LOGGER.info("已加载 %d 个文档片段: %s", len(loaded), file_path)
        except Exception:
            LOGGER.exception("读取文件失败，已跳过: %s", file_path)

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="批量加载 PDF、TXT 和 Markdown 文件")
    parser.add_argument("folder", type=Path, help="待加载的文件夹路径")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    documents = load_documents(args.folder)
    print(f"加载完成，共 {len(documents)} 个 Document 对象")


if __name__ == "__main__":
    main()
